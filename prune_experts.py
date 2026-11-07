"""Build a physically pruned copy of a mask folder (PLAN_PHYSICAL_PRUNE.md §3, PLAN_MODEL_SPECIALIZE.md §7).

A mask folder (``mask_checkpoint.py``, maybe with ``topk_checkpoint.py`` on top) hides the dropped experts from
the router but still loads all of them. This tool removes them from the files:

- ``MASK.json`` gives the dropped ids per layer. The kept ids, in ascending order, get the new ids 0 … K-1.
- Expert files (``exl3-layer-{L}-tp1-rank0.safetensors``): each expert is 12 separate tensors
  ``layers.{L}.ffn.experts.{E}.{proj}.rank0.{part}``. The new file holds the tensors of the kept experts, renamed
  to the new ids, with the bytes streamed unchanged from the source. No new quantization.
- Router tensors (carried files): the rows of ``gate.weight`` and the entries of ``gate.bias`` follow the keep
  order. The masked bias (-1e4) leaves with the dropped rows. Each ``tid2eid`` value (layers 0-2) gets its new
  id; a dropped id in ``tid2eid`` stops the tool, because the mask must have moved all of them.
- ``mtp.*`` tensors do not change: vLLM skips them in the target model, and the DSpark draft is its own folder.
- ``config.json`` (``n_routed_experts``, ``hybrid_tr3_tail.experts_per_layer``,
  ``reap_pruning.kept_experts_per_layer``), ``quantization_config.json`` (``rank_sliced.experts_per_layer``),
  ``model.safetensors.index.json``, ``tier-bitmap-3.0bpw.json`` and ``rank-sliced-tp1-manifest.json`` get the
  new values. vLLM stops if ``experts_per_layer`` is not ``n_routed_experts``.
- ``REAP_K{K}_PLAN.json``: ``keep_by_layer`` in the ids of the 256-expert model; ``mtp_keep`` does not change.
- ``EXL3_MANIFEST.json`` describes the TP4 source repo and stays a link, as in the mask folders.

The 43 expert files of ``data/tp1`` are ``root:600``, so run the build in a root container (README). Other files
link as in ``topk_checkpoint.py``. ``PRUNE.json`` records the keep map, the byte sums and the self-check. Run the
self-check alone with ``--check``.
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from mask_checkpoint import _DTYPES, BASE, HASH_LAYERS, MASK_BIAS, header, read_tensor, sha256_file
from topk_checkpoint import COPY_BYTES, MANIFEST, VERIFY, data_order, link_files, tensor_digest

EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(.+)$")
INDEX = "model.safetensors.index.json"
TIERS = "tier-bitmap-3.0bpw.json"
REAP = "REAP_K216_PLAN.json"
EDITED = {"config.json", "quantization_config.json", INDEX, TIERS, MANIFEST, "MASK.json", "TOPK.json", "PRUNE.json"}


def keep_lists(mask, experts):
    """``{layer: kept current ids, ascending}`` from a ``MASK.json`` record; the new id is the position."""
    out = {}
    for layer, item in mask["layers"].items():
        drop = set(item["drop"])
        if not drop <= set(range(experts)):
            raise ValueError(f"layer {layer}: a dropped id is outside 0-{experts - 1}")
        out[int(layer)] = [e for e in range(experts) if e not in drop]
    if len({len(k) for k in out.values()}) != 1:
        raise ValueError("the layers keep different numbers of experts")
    return out


def lookup(keep, experts):
    """Array ``old id -> new id``, -1 for a dropped id."""
    lut = np.full(experts, -1, np.int64)
    lut[keep] = np.arange(len(keep))
    return lut


def expert_rename(table, keep):
    """``{source name: new name or None}`` for an expert file; raise on a tensor that is not an expert."""
    new = {L: {e: i for i, e in enumerate(k)} for L, k in keep.items()}
    out = {}
    for name in table:
        if name == "__metadata__":
            continue
        m = EXPERT_RE.match(name)
        if not m:
            raise ValueError(f"{name}: not an expert tensor")
        layer, expert = int(m.group(1)), int(m.group(2))
        i = new[layer].get(expert)
        out[name] = None if i is None else f"layers.{layer}.ffn.experts.{i}.{m.group(3)}"
    return out


def write_file(src, dst, rename, new):
    """Write ``dst`` from ``src``: each tensor renamed by ``rename`` (None drops it, a missing name keeps it), with
    the arrays of ``new`` (source name -> array, same dtype) in place of the source bytes.

    The header keeps the tensor order and ``__metadata__`` and is padded with spaces to a multiple of 8 bytes.
    Returns ``(new tensor table, data bytes)``."""
    start, table = header(src)
    names = [n for n in data_order(table) if rename.get(n, n) is not None]
    blobs = {}
    for name, array in new.items():
        if np.dtype(_DTYPES[table[name]["dtype"]]) != array.dtype:
            raise ValueError(f"{name}: {array.dtype} does not fit {table[name]['dtype']}")
        blobs[name] = np.ascontiguousarray(array).tobytes()
    offsets, offset = {}, 0
    for name in names:
        a, b = table[name]["data_offsets"]
        size = len(blobs[name]) if name in blobs else b - a
        offsets[name] = [offset, offset + size]
        offset += size
    out = {}
    for name, info in table.items():
        if name == "__metadata__":
            out[name] = info
        elif name in offsets:
            out[rename.get(name, name)] = {**info, "data_offsets": offsets[name],
                                           **({"shape": list(new[name].shape)} if name in new else {})}
    head = json.dumps(out, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    with open(src, "rb") as fi, open(dst, "xb") as fo:
        fo.write(len(head).to_bytes(8, "little") + head)
        for name in names:
            if name in blobs:
                fo.write(blobs[name])
                continue
            a, b = table[name]["data_offsets"]
            fi.seek(start + a)
            left = b - a
            while left:
                chunk = fi.read(min(left, COPY_BYTES))
                if not chunk:
                    raise RuntimeError(f"{src.name}: {name} ends early")
                fo.write(chunk)
                left -= len(chunk)
    return out, offset


def router_edits(src, weight_map, keep, experts):
    """``{file: {tensor: new array}}`` for the router tensors of every layer in ``keep``."""
    changes = {}
    for layer, kept in keep.items():
        lut = lookup(kept, experts)
        names = [f"layers.{layer}.ffn.gate.weight"]
        names.append(f"layers.{layer}.ffn.gate.tid2eid" if layer < HASH_LAYERS else f"layers.{layer}.ffn.gate.bias")
        for name in names:
            array = read_tensor(src / weight_map[name], name)
            if name.endswith("tid2eid"):
                new = lut[array].astype(array.dtype)
                if (lut[array] < 0).any():
                    raise ValueError(f"{name}: holds a dropped id; the mask did not move it")
            else:
                if len(array) != experts:
                    raise ValueError(f"{name}: {len(array)} rows, expected {experts}")
                new = array[kept]
                if name.endswith("bias"):
                    dropped = np.setdiff1d(np.arange(experts), kept)
                    if not (array[dropped] == MASK_BIAS).all() or (new == MASK_BIAS).any():
                        raise ValueError(f"{name}: the masked bias is not on exactly the dropped ids")
            changes.setdefault(weight_map[name], {})[name] = new
    return changes


def edit_json(src, out, keep, experts, plan_name, data_change, tensors_removed, renames):
    k = len(next(iter(keep.values())))
    cfg = json.loads((src / "config.json").read_text())
    if cfg["n_routed_experts"] != experts or cfg["hybrid_tr3_tail"]["experts_per_layer"] != experts:
        raise ValueError("config.json: the expert count is not the source count")
    cfg["n_routed_experts"] = k
    cfg["hybrid_tr3_tail"]["experts_per_layer"] = k
    cfg["reap_pruning"]["kept_experts_per_layer"] = k
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    quant = json.loads((src / "quantization_config.json").read_text())
    quant["rank_sliced"]["experts_per_layer"] = k
    (out / "quantization_config.json").write_text(json.dumps(quant, indent=2) + "\n")

    index = json.loads((src / INDEX).read_text())
    weight_map = {}
    for name, file in index["weight_map"].items():
        new = renames.get(name, name)
        if new is not None:
            weight_map[new] = file
    index["weight_map"] = dict(sorted(weight_map.items()))
    index.setdefault("metadata", {})["total_size"] = index.get("metadata", {}).get("total_size", 0) + data_change
    (out / INDEX).write_text(json.dumps(index, indent=2) + "\n")

    tiers = json.loads((src / TIERS).read_text())
    for layer, kept in keep.items():
        bits = tiers[str(layer)]["k"]
        if len(bits) != experts:
            raise ValueError(f"{TIERS}: layer {layer} has {len(bits)} entries")
        tiers[str(layer)]["k"] = [bits[e] for e in kept]
    counts = {}
    for layer in keep:
        for b in tiers[str(layer)]["k"]:
            counts[str(b)] = counts.get(str(b), 0) + 1
    tiers.update(experts_per_layer=k, counts=counts, total_expert_triplets=sum(counts.values()))

    plan = json.loads((src / REAP).read_text())
    maps = plan["keep_maps"]
    for layer, kept in keep.items():
        original = maps["keep_by_layer"][str(layer)]
        if len(original) != experts:
            raise ValueError(f"{REAP}: layer {layer} keeps {len(original)} experts")
        maps["keep_by_layer"][str(layer)] = [original[e] for e in kept]
        held = set(maps["keep_by_layer"][str(layer)])
        maps["ranked_by_layer"][str(layer)] = [e for e in maps["ranked_by_layer"][str(layer)] if e in held]
    maps["keep_count"] = k
    plan["kept_experts_per_layer"] = k
    plan["target_label"] = plan_name.removesuffix("_PLAN.json")
    plan["physical_prune_note"] = ("keep_by_layer: the REAP K216 keep map, then the keep map of MASK.json "
                                   "(prune_experts.py); mtp_keep and mtp_ranked do not change")
    (out / plan_name).write_text(json.dumps(plan, indent=1))
    tiers["reap_plan_sha256"] = sha256_file(out / plan_name)
    (out / TIERS).write_text(json.dumps(tiers, indent=1))

    manifest = json.loads((src / MANIFEST).read_text())
    for item in manifest["files"]:
        if not (out / item["name"]).is_symlink():
            item["bytes"] = (out / item["name"]).stat().st_size
            item["sha256"] = sha256_file(out / item["name"])
    manifest["tensor_bytes"] += data_change
    manifest["tensor_count"] -= tensors_removed
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2))


def self_check(src, out, verify=VERIFY):
    """Check a pruned folder against its source; raise on a failure, return a summary."""
    record = json.loads((out / "PRUNE.json").read_text())
    keep = {int(L): v for L, v in record["keep"].items()}
    experts, k = record["source_experts"], len(next(iter(keep.values())))
    s_map = json.loads((src / INDEX).read_text())["weight_map"]
    o_map = json.loads((out / INDEX).read_text())["weight_map"]
    cfg = json.loads((out / "config.json").read_text())
    quant = json.loads((out / "quantization_config.json").read_text())
    if {cfg["n_routed_experts"], cfg["hybrid_tr3_tail"]["experts_per_layer"],
            quant["rank_sliced"]["experts_per_layer"]} != {k}:
        raise ValueError("config: the expert counts are not all K")
    summary = {"experts_same_bytes": 0, "router_tensors": 0, "other_tensors_same_bytes": 0}
    renames = {}
    for name in s_map:
        m = EXPERT_RE.match(name)
        if m and int(m.group(1)) in keep:
            e = int(m.group(2))
            kept = keep[int(m.group(1))]
            renames[name] = f"layers.{m.group(1)}.ffn.experts.{kept.index(e)}.{m.group(3)}" if e in kept else None
    expected = {renames.get(n, n): s_map[n] for n in s_map if renames.get(n, n) is not None}
    if o_map != expected:
        raise ValueError(f"{INDEX}: the weight map is not the renamed source map")
    source_name = {new: old for old, new in renames.items() if new is not None}
    tables, held = {}, {}
    for new, file in sorted(expected.items(), key=lambda kv: kv[1]):
        old = source_name.get(new, new)
        held.setdefault(file, set()).add(new)
        if file not in tables:
            tables[file] = (header(src / file), header(out / file))
        (s_start, s_table), (o_start, o_table) = tables[file]
        a, b = s_table[old], o_table[new]
        if EXPERT_RE.match(new) or not re.search(r"\.ffn\.gate\.(weight|bias|tid2eid)$", new) or new.startswith("mtp."):
            if a["dtype"] != b["dtype"] or a["shape"] != b["shape"]:
                raise ValueError(f"{new}: the shape or dtype changed")
            if tensor_digest(src / file, s_start, a) != tensor_digest(out / file, o_start, b):
                raise ValueError(f"{new}: the bytes changed")
            summary["experts_same_bytes" if EXPERT_RE.match(new) else "other_tensors_same_bytes"] += 1
            continue
        layer = int(new.split(".")[1])
        s_arr, o_arr = read_tensor(src / file, old), read_tensor(out / file, new)
        if new.endswith("tid2eid"):
            if not np.array_equal(np.asarray(keep[layer])[o_arr], s_arr):
                raise ValueError(f"{new}: does not map back to the source ids")
        elif not np.array_equal(o_arr, s_arr[keep[layer]]):
            raise ValueError(f"{new}: the rows are not the source rows in the keep order")
        summary["router_tensors"] += 1
    for file, ((_, s_table), (o_start, o_table)) in tables.items():
        if set(o_table) - {"__metadata__"} != held[file]:
            raise ValueError(f"{file}: holds tensors that the weight map does not list")
        if s_table.get("__metadata__") != o_table.get("__metadata__") or o_start % 8:
            raise ValueError(f"{file}: __metadata__ changed or the header is not padded to 8 bytes")
        last = data_order(o_table)[-1]
        if (out / file).stat().st_size != o_start + o_table[last]["data_offsets"][1]:
            raise ValueError(f"{file}: the file size does not match the header")
    result = subprocess.run(["python3", str(verify), str(out)], capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"verify_tp1_manifest.py failed: {result.stderr.strip()[-500:]}")
    summary["verify_tp1_manifest"] = json.loads(result.stdout)
    return summary


def build(src, out, verify=VERIFY, base=BASE):
    """Write the pruned folder, run the self-check and return the ``PRUNE.json`` record."""
    src, out = Path(src), Path(out)
    mask = json.loads((src / "MASK.json").read_text())
    experts = json.loads((src / "config.json").read_text())["n_routed_experts"]
    keep = keep_lists(mask, experts)
    k = len(next(iter(keep.values())))
    weight_map = json.loads((src / INDEX).read_text())["weight_map"]
    expert_files = sorted({f for n, f in weight_map.items() if EXPERT_RE.match(n)})
    router = router_edits(src, weight_map, keep, experts)
    out.mkdir(exist_ok=False)
    renames, data_change, removed = {}, 0, 0
    for file in expert_files:
        s_start, table = header(src / file)
        rename = expert_rename(table, keep)
        _, size = write_file(src / file, out / file, rename, {})
        data_change += size - ((src / file).stat().st_size - s_start)
        renames.update(rename)
        removed += sum(v is None for v in rename.values())
    for file, new in router.items():
        _, size = write_file(src / file, out / file, {}, new)
        data_change += size - ((src / file).stat().st_size - header(src / file)[0])
    shutil.copyfile(src / "MASK.json", out / "MASK.json")
    if (src / "TOPK.json").exists():
        shutil.copyfile(src / "TOPK.json", out / "TOPK.json")
    plan_name = f"REAP_K{k}_PLAN.json"
    links = link_files(src, out, EDITED | set(expert_files) | set(router), base)
    edit_json(src, out, keep, experts, plan_name, data_change, removed, renames)
    record = {"source": str(src.resolve()), "source_experts": experts, "kept_experts": k,
              "keep": {str(L): v for L, v in sorted(keep.items())},
              "rewritten": expert_files + sorted(router), "links": links,
              "tensor_bytes_change": data_change, "tensors_removed": removed, "plan": plan_name}
    (out / "PRUNE.json").write_text(json.dumps(record, indent=1))
    try:
        record["self_check"] = self_check(src, out, verify)
    except Exception as error:
        record["self_check"] = f"FAIL: {error}"
    (out / "PRUNE.json").write_text(json.dumps(record, indent=1))
    if isinstance(record["self_check"], str):
        raise RuntimeError(f"{out}: self-check {record['self_check']}")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="a mask folder with MASK.json")
    parser.add_argument("--out", type=Path, required=True, help="the new folder; must not exist")
    parser.add_argument("--verify", type=Path, default=VERIFY, help="verify_tp1_manifest.py")
    parser.add_argument("--check", action="store_true", help="run only the self-check on --out")
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(self_check(args.source, args.out, args.verify), indent=1))
        print(f"self-check PASS: {args.out}")
        return
    record = build(args.source, args.out, args.verify)
    gib = -record["tensor_bytes_change"] / 2**30
    print(f"experts {record['source_experts']} -> {record['kept_experts']} per layer: rewrote "
          f"{len(record['rewritten'])} files, removed {record['tensors_removed']} tensors ({gib:.2f} GiB); "
          f"{len(record['links']['symlink'])} symlinks, {len(record['links']['hardlink'])} hardlinks")
    print(f"self-check PASS; wrote {args.out}")


if __name__ == "__main__":
    main()
