"""Build a masked copy of the served DeepSeek V4 Flash checkpoint (PLAN_MODEL_SPECIALIZE.md §5.3).

The mask keeps K of the 216 routed experts per layer. The 216 - K experts with the lowest REAP rank
in ``REAP_K216_PLAN.json`` drop out; the mandatory experts of the two tool categories always stay.
With ``--rank expert_rank_bot.json`` (``capture.py``, plan §6.5), the least used experts of the
bot's traffic drop out instead, under the same mandatory rule; the folder is then ``tp1-botmask-k{K}``.

- Layers 3-42 (learned router): ``gate.bias[e] = -1e4`` for each dropped expert ``e``. The router
  adds the bias only to choose the top-k, so it never chooses ``e``, and the kept experts keep their
  choice and their weights.
- Layers 0-2 (hash routing): each ``tid2eid`` entry of a dropped expert points to the kept expert
  with the most similar router row (cosine of ``gate.weight``) that is not in that token's row yet
  and holds less than twice the mean load. The cap keeps the even load of hash routing; without it,
  a few central experts take most of the entries. The first prune remapped to "high-saliency kept
  experts" and did not publish the exact rule.

The output folder holds symlinks to the unchanged files of the base folder, edited copies of the
carried files that hold the router tensors, ``rank-sliced-tp1-manifest.json`` with the new SHA-256 of
those copies, and ``MASK.json``. The symlinks are absolute: mount the base folder read-only at the
same path in the container. One folder takes about 12.3 GB (three carried files).
"""

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np

BASE = Path.home() / "Projects/DeepSeek-v4-Flash-One-DGX-Spark/data/tp1"
OUT_ROOT = Path(__file__).resolve().parent
HASH_LAYERS = 3
MASK_BIAS = -1e4
MANDATORY_PER_CATEGORY = 32
_DTYPES = {"F32": np.float32, "I32": np.int32, "I64": np.int64, "BF16": np.uint16}


def header(path):
    """``(data start, tensor table)`` of a safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def read_tensor(path, name):
    start, table = header(path)
    t = table[name]
    a, b = t["data_offsets"]
    with open(path, "rb") as f:
        f.seek(start + a)
        return np.frombuffer(f.read(b - a), _DTYPES[t["dtype"]]).reshape(t["shape"]).copy()


def write_tensor(path, name, array):
    """Overwrite the bytes of one tensor in place; dtype and shape must stay the same."""
    start, table = header(path)
    t = table[name]
    a, b = t["data_offsets"]
    data = np.ascontiguousarray(array, _DTYPES[t["dtype"]])
    if list(data.shape) != t["shape"] or data.nbytes != b - a:
        raise ValueError(f"{name}: {data.dtype}{list(data.shape)} does not fit {t['dtype']}{t['shape']}")
    with open(path, "r+b") as f:
        f.seek(start + a)
        f.write(data.tobytes())


def bf16_to_f32(raw):
    return (raw.astype(np.uint32) << 16).view(np.float32)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bot_order(rank):
    """``{layer: current ids, most used first}`` from an ``expert_rank_bot.json`` record."""
    return {int(layer): [e["id"] for e in items] for layer, items in rank["layers"].items()}


def drop_lists(plan, keep, bot=None):
    """``{layer: (current ids, original ids)}`` of the experts to drop, lowest rank first.

    The rank is the REAP rank of the plan, or ``bot`` (``bot_order``) if given. The checkpoint numbers
    its experts 0-215 in the sorted order of ``keep_by_layer``."""
    maps = plan["keep_maps"]
    out = {}
    for layer in range(len(maps["keep_by_layer"])):
        kept, ranked = maps["keep_by_layer"][str(layer)], maps["ranked_by_layer"][str(layer)]
        mandatory = {e for cat in maps["structured_ranked_by_category"].values()
                     for e in cat[str(layer)][:MANDATORY_PER_CATEGORY]}
        current = {e: i for i, e in enumerate(kept)}
        if bot is None:
            order = [current[e] for e in reversed(ranked) if e in current]
        else:
            if sorted(bot[layer]) != list(range(len(kept))):
                raise ValueError(f"layer {layer}: the bot rank does not list the {len(kept)} current ids")
            order = list(reversed(bot[layer]))
        candidates = [c for c in order if kept[c] not in mandatory]
        n = len(kept) - keep
        if n > len(candidates):
            raise ValueError(f"layer {layer}: {len(mandatory)} mandatory experts leave no room for K={keep}")
        dropped = candidates[:n]
        out[layer] = (dropped, [kept[c] for c in dropped])
    return out


def similarity_order(gate_weight, drop):
    """``{dropped id: kept ids, most similar router row first}``."""
    rows = gate_weight / np.linalg.norm(gate_weight, axis=1, keepdims=True)
    kept = np.array(sorted(set(range(len(rows))) - set(drop)))
    return {d: kept[np.argsort(-(rows[kept] @ rows[d]), kind="stable")].tolist() for d in drop}


def remap_hash(tid2eid, drop, order):
    """A copy of ``tid2eid`` with each dropped id replaced by the first id of ``order[id]`` that is
    not in the row yet and holds less than twice the mean load of the kept experts (if none does,
    the first id not in the row). Returns the copy and the number of entries replaced."""
    out = tid2eid.copy()
    dropped = np.isin(out, list(drop))
    kept = len(next(iter(order.values())))
    load = np.bincount(out[~dropped], minlength=kept + len(drop)).tolist()
    cap = 2 * out.size / kept
    for r in np.flatnonzero(dropped.any(axis=1)):
        row = out[r]
        for j in np.flatnonzero(dropped[r]):
            taken, ids = set(row.tolist()), order[int(row[j])]
            c = next((c for c in ids if c not in taken and load[c] < cap), None)
            row[j] = c if c is not None else next(c for c in ids if c not in taken)
            load[row[j]] += 1
    return out, int(dropped.sum())


def edits(base, drops):
    """``{file: {tensor: new array}}`` and the hash entries replaced per layer."""
    weight_map = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    changes, remapped = {}, {}
    for layer, (drop, _) in drops.items():
        if not drop:
            continue
        if layer < HASH_LAYERS:
            name = f"layers.{layer}.ffn.gate.tid2eid"
            path = base / weight_map[name]
            gate = bf16_to_f32(read_tensor(base / weight_map[f"layers.{layer}.ffn.gate.weight"],
                                           f"layers.{layer}.ffn.gate.weight"))
            new, remapped[layer] = remap_hash(read_tensor(path, name), drop,
                                              similarity_order(gate, drop))
        else:
            name = f"layers.{layer}.ffn.gate.bias"
            path = base / weight_map[name]
            new = read_tensor(path, name)
            new[drop] = MASK_BIAS
        changes.setdefault(weight_map[name], {})[name] = new
    return changes, remapped


def build(base, keep, out, dry_run=False, rank=None):
    """Write the masked folder; return the ``MASK.json`` record. ``rank``: path of an
    ``expert_rank_bot.json``, or None for the REAP rank."""
    plan = json.loads((base / "REAP_K216_PLAN.json").read_text())
    bot = bot_order(json.loads(Path(rank).read_text())) if rank else None
    drops = drop_lists(plan, keep, bot)
    changes, remapped = edits(base, drops)
    record = {"keep": keep, "base": str(base), "mask_bias": MASK_BIAS,
              "rank": ({"file": str(rank), "sha256": sha256_file(rank)} if rank
                       else "REAP_K216_PLAN.json ranked_by_layer"),
              "hash_rule": "most similar gate.weight row among the kept experts, not in the row yet, "
                           "below twice the mean load",
              "edited_files": sorted(changes), "hash_entries_remapped": remapped,
              "layers": {layer: {"drop": cur, "drop_original": orig}
                         for layer, (cur, orig) in drops.items()}}
    if dry_run:
        return record
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((base / "rank-sliced-tp1-manifest.json").read_text())
    for src in sorted(base.iterdir()):
        if src.name in changes:
            shutil.copyfile(src, out / src.name)
            for name, array in changes[src.name].items():
                write_tensor(out / src.name, name, array)
                if not np.array_equal(read_tensor(out / src.name, name), array):
                    raise RuntimeError(f"{src.name}: {name} did not write")
        elif src.name != "rank-sliced-tp1-manifest.json" and src.is_file():
            (out / src.name).symlink_to(src.resolve())
    for item in manifest["files"]:
        if item["name"] in changes:
            item["sha256"] = sha256_file(out / item["name"])
    (out / "rank-sliced-tp1-manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "MASK.json").write_text(json.dumps(record, indent=1))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--keep", type=int, required=True, help="experts kept per layer (K)")
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--rank", type=Path, help="expert_rank_bot.json; default: the REAP rank")
    parser.add_argument("--out", type=Path,
                        help="default: tp1-mask-k{K} (REAP) or tp1-botmask-k{K} (--rank) next to this script")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = parser.parse_args(argv)
    out = args.out or OUT_ROOT / f"tp1-{'botmask' if args.rank else 'mask'}-k{args.keep}"
    record = build(args.base, args.keep, out, args.dry_run, args.rank)
    drops = [len(v["drop"]) for v in record["layers"].values()]
    print(f"K={args.keep}: drop {min(drops)}-{max(drops)} experts per layer; "
          f"hash entries remapped {record['hash_entries_remapped']}; "
          f"edited {', '.join(record['edited_files'])}")
    print("dry run, nothing written" if args.dry_run else f"wrote {out}")


if __name__ == "__main__":
    main()
