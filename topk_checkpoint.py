"""Build a top-k copy of a served DeepSeek V4 Flash checkpoint folder (PLAN_MODEL_SPECIALIZE.md §8,
PLAN_OVERNIGHT_TOPK.md §3). Each token then uses ``--top-k`` routed experts in place of 6.

- ``config.json``: an edited copy with ``num_experts_per_tok`` = top-k. Nothing else changes.
- Layers 3-42 (learned router): vLLM reads top-k from the config. No tensor changes.
- Layers 0-2 (hash routing): vLLM sizes ``tid2eid`` as (vocab, ``num_experts_per_tok``), so each row
  drops 6 - top-k ids. The dropped ids have the lowest estimated router score for that token:
  score[t, e] = gate.weight[e] · (embed.weight[t] × ffn_norm.weight) of the same layer, BF16 -> F32.
  The router score is sqrtsoftplus of the logit of the RMS-normed FFN input. Both steps keep the
  order, so the order of the logits is the order of the scores. The tool uses the token embedding in
  place of the real FFN input (embedding plus attention, mixed by mHC): the rule is an estimate. The
  kept ids keep their old column order.

The file that holds ``tid2eid`` (``carried-001.safetensors``) gets new tensor shapes, so the tool
writes a new file: a new header with the new shapes and offsets, in the same tensor order, with
``__metadata__`` kept and spaces as padding to a multiple of 8 bytes. The bytes of every other tensor
stream unchanged from the source. ``embed.weight`` is read in chunks of 4,096 rows, never whole.

Other files of the source folder:
- a symlink becomes a symlink to its resolved target;
- a real file becomes a hardlink, because the container does not mount the source folder;
- a real file inside the base folder ``data/tp1`` becomes a symlink. The container mounts the base
  folder at its host path, and the kernel (``protected_hardlinks``) does not allow a hardlink to a
  file of another owner.

``rank-sliced-tp1-manifest.json`` gets the new ``bytes`` and ``sha256`` of the rewritten file, and
``tensor_bytes`` changes by the change in tensor data. ``MASK.json`` is copied. ``TOPK.json``
records the rule, the diagnostics and the self-check. The self-check runs after each write; run it
alone with ``--check``.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

from mask_checkpoint import _DTYPES, BASE, HASH_LAYERS, bf16_to_f32, header, read_tensor, sha256_file

MANIFEST = "rank-sliced-tp1-manifest.json"
RANK = Path.home() / "Projects/stock-research-model-eval/model_eval/results/expert_rank_bot.json"
VERIFY = Path.home() / "Projects/DeepSeek-v4-Flash-One-DGX-Spark/image-patch/verify_tp1_manifest.py"
CHUNK_ROWS = 4096
COPY_BYTES = 16 << 20
RMS_EPS = 1e-6
RULE = ("drop the ids with the lowest estimated router logit gate.weight[e] · (rmsnorm(embed.weight[t]) "
        "× ffn_norm.weight) of the same layer; estimate from the token embedding in place of the real "
        "FFN input (embedding plus attention, mixed by mHC); the kept ids keep their column order")


def tid_name(layer):
    return f"layers.{layer}.ffn.gate.tid2eid"


def edit_config(text, top_k):
    """``(new text, old top-k)``: ``text`` with only the value of ``num_experts_per_tok`` changed."""
    cfg = json.loads(text)
    old = cfg["num_experts_per_tok"]
    if not 1 <= top_k < old:
        raise ValueError(f"top-k {top_k} must be between 1 and {old - 1}")
    new_text, n = re.subn(r'("num_experts_per_tok"\s*:\s*)' + str(old) + r"\b",
                          lambda m: m.group(1) + str(top_k), text)
    if n != 1 or json.loads(new_text) != {**cfg, "num_experts_per_tok": top_k}:
        raise ValueError("config.json: num_experts_per_tok did not change exactly once")
    return new_text, old


def read_rows(path, start, info, lo, hi):
    """Rows ``lo:hi`` of a 2-D tensor, read by file offset (``start``, ``info`` from ``header``)."""
    dtype = np.dtype(_DTYPES[info["dtype"]])
    cols = info["shape"][1]
    with open(path, "rb") as f:
        f.seek(start + info["data_offsets"][0] + lo * cols * dtype.itemsize)
        return np.frombuffer(f.read((hi - lo) * cols * dtype.itemsize), dtype).reshape(hi - lo, cols)


def cut_rows(tid, scores, top_k):
    """Each row of ``tid`` without its ``width - top_k`` ids of lowest score; the kept ids keep their
    column order. ``scores[r, j]`` is the score of ``tid[r, j]``. Ties drop the left column first."""
    drop = tid.shape[1] - top_k
    keep = np.ones(tid.shape, bool)
    np.put_along_axis(keep, np.argsort(scores, axis=1, kind="stable")[:, :drop], False, axis=1)
    return tid[keep].reshape(len(tid), top_k)


def sqrtsoftplus(x):
    return np.sqrt(np.logaddexp(0, x))


def new_tables(src, top_k, layers=range(HASH_LAYERS)):
    """``({layer: new tid2eid}, {layer: estimated weight share dropped})`` for a source folder.

    The weight share is the mean over tokens of sqrtsoftplus(dropped logits) / sqrtsoftplus(all six
    logits), from the same estimate. It is a diagnostic only."""
    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name):
        return read_tensor(src / weight_map[name], name)

    tids = {i: tensor(tid_name(i)) for i in layers}
    gates = {i: bf16_to_f32(tensor(f"layers.{i}.ffn.gate.weight")) for i in layers}
    norms = {i: bf16_to_f32(tensor(f"layers.{i}.ffn_norm.weight")) for i in layers}
    path = src / weight_map["embed.weight"]
    start, table = header(path)
    vocab = table["embed.weight"]["shape"][0]
    for i in layers:
        if len(tids[i]) != vocab:
            raise ValueError(f"{tid_name(i)}: {len(tids[i])} rows, embed.weight has {vocab}")
    out = {i: np.empty((vocab, top_k), tids[i].dtype) for i in layers}
    dropped_share = {i: 0.0 for i in layers}
    for lo in range(0, vocab, CHUNK_ROWS):
        hi = min(lo + CHUNK_ROWS, vocab)
        x = bf16_to_f32(read_rows(path, start, table["embed.weight"], lo, hi))
        x = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + RMS_EPS)
        for i in layers:
            logits = np.take_along_axis((x * norms[i]) @ gates[i].T, tids[i][lo:hi], axis=1)
            out[i][lo:hi] = cut_rows(tids[i][lo:hi], logits, top_k)
            w = sqrtsoftplus(logits)
            kept = np.sort(w, axis=1)[:, tids[i].shape[1] - top_k:].sum(axis=1)
            dropped_share[i] += float((1 - kept / w.sum(axis=1)).sum())
    return out, {i: dropped_share[i] / vocab for i in layers}


def data_order(table):
    return sorted((k for k in table if k != "__metadata__"), key=lambda k: table[k]["data_offsets"][0])


def rewrite(src, dst, new):
    """Write ``dst`` as ``src`` with the tensors of ``new`` (name -> array, same dtype) replaced.

    The header keeps the tensor order and ``__metadata__``, gets new shapes and offsets, and is
    padded with spaces to a multiple of 8 bytes. Every other tensor's bytes stream from ``src``.
    Returns the new tensor table."""
    start, table = header(src)
    names = data_order(table)
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
        else:
            out[name] = {**info, "data_offsets": offsets[name]}
            if name in new:
                out[name]["shape"] = list(new[name].shape)
    head = json.dumps(out, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    with open(src, "rb") as fi, open(dst, "xb") as fo:
        fo.write(struct.pack("<Q", len(head)) + head)
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
    return out


def inside(path, folder):
    return Path(path).resolve().is_relative_to(Path(folder).resolve())


def link_files(src, out, skip, base=BASE):
    """Link every file of ``src`` not in ``skip`` into ``out``; returns ``{kind: [names]}``."""
    done = {"symlink": [], "hardlink": []}
    for path in sorted(src.iterdir()):
        if path.name in skip:
            continue
        if path.is_symlink() or (path.is_file() and inside(path, base)):
            (out / path.name).symlink_to(path.resolve())
            done["symlink"].append(path.name)
        elif path.is_file():
            os.link(path, out / path.name)
            done["hardlink"].append(path.name)
        else:
            raise ValueError(f"{path}: not a file or a symlink")
    return done


def least_used_share(src_tid, out_tid, order):
    """Share of rows whose dropped ids are the least used ids of the row in the bot rank.

    ``order``: the ids of one layer, most used first (``expert_rank_bot.json``)."""
    position = np.empty(len(order), np.int64)
    position[np.asarray(order)] = np.arange(len(order))
    drop = src_tid.shape[1] - out_tid.shape[1]
    least = np.zeros(src_tid.shape, bool)
    np.put_along_axis(least, np.argsort(-position[src_tid], axis=1, kind="stable")[:, :drop], True, axis=1)
    dropped = ~(src_tid[:, :, None] == out_tid[:, None, :]).any(axis=2)
    return float((dropped == least).all(axis=1).mean())


def tensor_digest(path, start, info):
    digest = hashlib.sha256()
    a, b = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(start + a)
        left = b - a
        while left:
            chunk = f.read(min(left, COPY_BYTES))
            if not chunk:
                raise RuntimeError(f"{path.name}: tensor ends early")
            digest.update(chunk)
            left -= len(chunk)
    return digest.hexdigest()


def check_rows(src_tid, out_tid, top_k):
    """Raise unless each row of ``out_tid`` is a subset of the source row, in its column order,
    with no duplicate added."""
    if out_tid.shape != (len(src_tid), top_k):
        raise ValueError(f"shape {out_tid.shape}, expected {(len(src_tid), top_k)}")
    match = out_tid[:, :, None] == src_tid[:, None, :]
    if not match.any(axis=2).all():
        raise ValueError("an id is not in the source row")
    if ((out_tid[:, :, None] == out_tid[:, None, :]).sum(axis=2) > match.sum(axis=2)).any():
        raise ValueError("a duplicate id was added")
    unique = (np.sort(src_tid, axis=1)[:, 1:] != np.sort(src_tid, axis=1)[:, :-1]).all(axis=1)
    if (np.diff(match.argmax(axis=2), axis=1)[unique] <= 0).any():
        raise ValueError("the kept ids changed their column order")


def self_check(src, out, top_k, verify=VERIFY):
    """Check a written folder against its source (PLAN_OVERNIGHT_TOPK.md §3 step 7); raise on a
    failure, return a summary."""
    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted({weight_map[tid_name(i)] for i in range(HASH_LAYERS)})
    summary = {"rewritten": files, "tensors_same_bytes": 0}
    if json.loads((out / "config.json").read_text())["num_experts_per_tok"] != top_k:
        raise ValueError("config.json: num_experts_per_tok is not the top-k")
    for name in files:
        (s_start, s_table), (o_start, o_table) = header(src / name), header(out / name)
        if list(s_table) != list(o_table) or s_table.get("__metadata__") != o_table.get("__metadata__"):
            raise ValueError(f"{name}: the tensor order or __metadata__ changed")
        if o_start % 8:
            raise ValueError(f"{name}: the header is not padded to 8 bytes")
        if (out / name).stat().st_size != o_start + o_table[data_order(o_table)[-1]]["data_offsets"][1]:
            raise ValueError(f"{name}: the file size does not match the header")
        for tensor, info in s_table.items():
            if tensor == "__metadata__":
                continue
            if tensor.endswith(".ffn.gate.tid2eid"):
                check_rows(read_tensor(src / name, tensor), read_tensor(out / name, tensor), top_k)
                continue
            if o_table[tensor]["shape"] != info["shape"] or o_table[tensor]["dtype"] != info["dtype"]:
                raise ValueError(f"{name}: {tensor} changed its shape or dtype")
            if tensor_digest(src / name, s_start, info) != tensor_digest(out / name, o_start, o_table[tensor]):
                raise ValueError(f"{name}: {tensor} changed its bytes")
            summary["tensors_same_bytes"] += 1
    result = subprocess.run(["python3", str(verify), str(out), "--skip-checksums"],
                            capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"verify_tp1_manifest.py failed: {result.stderr.strip()[-500:]}")
    summary["verify_tp1_manifest"] = json.loads(result.stdout)
    manifest = {item["name"]: item for item in json.loads((out / MANIFEST).read_text())["files"]}
    for name in files:
        if sha256_file(out / name) != manifest[name]["sha256"]:
            raise ValueError(f"{name}: the SHA-256 does not match the manifest")
    summary["sha256_matches_manifest"] = True
    return summary


def build(src, top_k, out, rank=RANK, base=BASE, verify=VERIFY):
    """Write the top-k folder, run the self-check and return the ``TOPK.json`` record."""
    src, out = Path(src), Path(out)
    config, old = edit_config((src / "config.json").read_text(), top_k)
    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    tables, dropped_share = new_tables(src, top_k)
    for i, table in tables.items():
        if table.shape[1] != top_k or read_tensor(src / weight_map[tid_name(i)], tid_name(i)).shape[1] != old:
            raise ValueError(f"{tid_name(i)}: the source table does not have {old} columns")
    changes = {}
    for i, table in tables.items():
        changes.setdefault(weight_map[tid_name(i)], {})[tid_name(i)] = table
    out.mkdir(exist_ok=False)
    (out / "config.json").write_text(config)
    data_change = 0
    for name, new in changes.items():
        rewrite(src / name, out / name, new)
        data_change += ((out / name).stat().st_size - header(out / name)[0]
                        - (src / name).stat().st_size + header(src / name)[0])
    if (src / "MASK.json").exists():
        shutil.copyfile(src / "MASK.json", out / "MASK.json")
    links = link_files(src, out, {"config.json", MANIFEST, "MASK.json", "TOPK.json", *changes}, base)
    manifest = json.loads((src / MANIFEST).read_text())
    for item in manifest["files"]:
        if item["name"] in changes:
            item["bytes"] = (out / item["name"]).stat().st_size
            item["sha256"] = sha256_file(out / item["name"])
    if "tensor_bytes" in manifest:
        manifest["tensor_bytes"] += data_change
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2))
    record = {"top_k": top_k, "source_top_k": old, "source": str(src.resolve()), "rule": RULE,
              "estimated_weight_share_dropped": dropped_share, "rewritten": sorted(changes),
              "links": links, "tensor_bytes_change": data_change}
    if rank and Path(rank).exists():
        order = {int(layer): [e["id"] for e in items]
                 for layer, items in json.loads(Path(rank).read_text())["layers"].items()}
        record["dropped_is_least_used"] = {
            "rank": {"file": str(rank), "sha256": sha256_file(rank)},
            "random": 1 / math.comb(old, old - top_k),
            "share": {i: least_used_share(read_tensor(src / weight_map[tid_name(i)], tid_name(i)),
                                          tables[i], order[i]) for i in tables}}
    try:
        record["self_check"] = self_check(src, out, top_k, verify)
    except Exception as error:
        record["self_check"] = f"FAIL: {error}"
    (out / "TOPK.json").write_text(json.dumps(record, indent=1))
    if isinstance(record["self_check"], str):
        raise RuntimeError(f"{out}: self-check {record['self_check']}")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="a tp1 folder (mask folder or data/tp1)")
    parser.add_argument("--top-k", type=int, required=True, help="experts per token")
    parser.add_argument("--out", type=Path, required=True, help="the new folder; must not exist")
    parser.add_argument("--rank", type=Path, default=RANK, help="expert_rank_bot.json, for a diagnostic")
    parser.add_argument("--check", action="store_true", help="run only the self-check on --out")
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(self_check(args.source, args.out, args.top_k), indent=1))
        print(f"self-check PASS: {args.out}")
        return
    record = build(args.source, args.top_k, args.out, args.rank)
    print(f"top-k {record['source_top_k']} -> {record['top_k']}: rewrote {', '.join(record['rewritten'])}; "
          f"{len(record['links']['symlink'])} symlinks, {len(record['links']['hardlink'])} hardlinks; "
          f"weight share dropped (estimate) {record['estimated_weight_share_dropped']}")
    if "dropped_is_least_used" in record:
        d = record["dropped_is_least_used"]
        print(f"dropped = least used in the bot rank: {d['share']} (random {d['random']:.3f})")
    print(f"self-check PASS; wrote {args.out}")


if __name__ == "__main__":
    main()
