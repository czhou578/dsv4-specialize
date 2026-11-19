"""Prune the routed experts of a Qwen3.5/3.6 MoE checkpoint (plans/PLAN_QWEN_PRUNE.md §5).

The checkpoint stores each routed expert of a decoder layer as its own tensors
(``...layers.<L>.mlp.experts.<E>.<proj>.<part>``), the router as ``...layers.<L>.mlp.gate.weight`` (one row per expert),
and the experts of the MTP draft as stacked tensors (``mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}``, first
dimension = experts) with their own ``mtp.layers.0.mlp.gate.weight``.

The tool copies the checkpoint with K experts per layer: it keeps the K most important experts of each layer by the
rank file of ``capture.py --rank``, gives them the new ids 0..K-1 in their old order, keeps the same rows of the router,
slices the stacked tensors of the draft, rewrites ``model.safetensors.index.json``, and sets ``num_experts`` in the
config. It copies bytes and does no new quantization. vLLM reads one ``num_experts`` for the model and the draft, so
both get K experts. The rank of the draft is the layer after the last decoder layer in the rank file.

``--check`` compares every tensor of a pruned folder with its source, byte for byte.
"""

import argparse
import json
import re
import shutil
import struct
from pathlib import Path

EXPERT = re.compile(r"^(?!mtp\.)(.*layers\.(\d+)\.mlp\.experts\.)(\d+)(\..+)$")
GATE = re.compile(r"^(?!mtp\.).*layers\.(\d+)\.mlp\.gate\.weight$")
DRAFT = re.compile(r"^mtp\.layers\.0\.mlp\.(experts\.(gate_up_proj|down_proj)|gate\.weight)$")
CHUNK = 64 << 20   # the largest piece of a tensor in memory while a file is written or checked


def read_header(path):
    """``(header, data start)`` of a safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def chunks(source):
    """The bytes of a tensor source, in pieces of at most ``CHUNK`` bytes from a file. ``source`` is bytes, a function
    that returns bytes, or a list of ``(file, offset, size)`` ranges to read in order."""
    if callable(source):
        source = source()
    if isinstance(source, (bytes, bytearray)):
        yield source
        return
    for path, offset, size in source:
        with open(path, "rb") as f:
            f.seek(offset)
            while size:
                piece = f.read(min(size, CHUNK))
                if not piece:
                    raise EOFError(f"{path}: the file ends before offset {offset} + {size}")
                yield piece
                size -= len(piece)


def write_streaming(path, items, metadata=None):
    """Write ``(name, dtype, shape, size, source)`` items in this order, with no gap between them. See ``chunks``
    for ``source``: with file ranges, no tensor sits in memory as a whole."""
    header, offset = ({"__metadata__": metadata} if metadata else {}), 0
    for name, dtype, shape, size, _ in items:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        for name, _, _, size, source in items:
            written = 0
            for piece in chunks(source):
                f.write(piece)
                written += len(piece)
            if written != size:
                raise ValueError(f"{name}: {written} bytes, the header says {size}")


def write_safetensors(path, tensors, metadata=None):
    """Write ``(name, dtype, shape, bytes)`` items in this order, with no gap between them."""
    write_streaming(path, [(name, dtype, shape, len(data), data) for name, dtype, shape, data in tensors], metadata)


def keep_sets(rank, k, num_layers, draft="rank"):
    """``{layer: sorted kept ids}`` for the decoder layers, and under ``"mtp"`` for the draft: the first ``k`` experts of
    each layer of the rank file. ``draft="first"`` keeps the experts 0..k-1 of the draft (a rank file with no draft);
    ``draft="keep"`` keeps the draft whole (its tensors are copied unchanged; serve its draft from the source folder)."""
    layers = rank["layers"]
    keep = {l: sorted(e["id"] for e in layers[str(l)][:k]) for l in range(num_layers)}
    if draft == "keep":
        keep["mtp"] = None
    elif draft == "first":
        keep["mtp"] = list(range(k))
    elif str(num_layers) not in layers:
        raise ValueError(f"the rank file has no layer {num_layers} for the draft: capture it, or pass --draft first")
    else:
        keep["mtp"] = sorted(e["id"] for e in layers[str(num_layers)][:k])
    if any(v is not None and len(set(v)) != k for v in keep.values()):
        raise ValueError(f"a layer of the rank file has fewer than {k} distinct experts")
    return keep


def plan(name, keep):
    """What happens to a tensor: ``(new name, rows)``. The new name is None for a dropped expert; ``rows`` are the kept
    indexes of the first dimension, or None for a plain copy."""
    if m := EXPERT.match(name):
        ids = keep[int(m.group(2))]
        e = int(m.group(3))
        return (f"{m.group(1)}{ids.index(e)}{m.group(4)}" if e in ids else None), None
    if m := GATE.match(name):
        return name, keep[int(m.group(1))]
    if DRAFT.match(name):
        return name, keep["mtp"]   # None: the draft is kept whole
    return name, None


def row_ranges(path, offset, shape, size, rows):
    """The ``(file, offset, size)`` ranges of the kept rows of the first dimension; adjacent rows join one range."""
    row, out = size // shape[0], []
    for r in rows:
        if out and out[-1][1] + out[-1][2] == offset + r * row:
            out[-1] = (path, out[-1][1], out[-1][2] + row)
        else:
            out.append((path, offset + r * row, row))
    return out


def tensors_of(path, keep):
    """Yield ``(new name, dtype, new shape, size, ranges, source name)`` for the kept tensors of a file, in file order.
    ``ranges`` are ``(file, offset, size)`` ranges of the source file (see ``chunks``); no tensor bytes are read."""
    header, start = read_header(path)
    names = sorted((k for k in header if k != "__metadata__"), key=lambda k: header[k]["data_offsets"][0])
    for name in names:
        new, rows = plan(name, keep)
        if new is None:
            continue
        h = header[name]
        offset, size, shape = start + h["data_offsets"][0], h["data_offsets"][1] - h["data_offsets"][0], h["shape"]
        ranges = [(path, offset, size)]
        if rows is not None:
            ranges, shape = row_ranges(path, offset, shape, size, rows), [len(rows), *shape[1:]]
        yield new, h["dtype"], shape, sum(r[2] for r in ranges), ranges, name


def config_layers(source):
    config = json.loads((Path(source) / "config.json").read_text())
    text = config.get("text_config", config)
    return config, text, int(text["num_hidden_layers"]), int(text["num_experts"])


def prune(source, out, rank, k, draft="rank"):
    """Write the pruned folder; return ``(tensors kept, tensors dropped, bytes written)``."""
    source, out = Path(source), Path(out)
    config, text, num_layers, experts = config_layers(source)
    if not 0 < k < experts:
        raise ValueError(f"keep must be between 1 and {experts - 1}")
    keep = keep_sets(rank, k, num_layers, draft)
    out.mkdir(parents=True, exist_ok=False)
    weight_map, kept, dropped, size = {}, 0, 0, 0
    for f in sorted(source.glob("*.safetensors")):
        items = [t[:5] for t in tensors_of(f, keep)]
        write_streaming(out / f.name, items, read_header(f)[0].get("__metadata__"))
        weight_map.update({t[0]: f.name for t in items})
        kept, size = kept + len(items), size + sum(t[3] for t in items)
        dropped += len(read_header(f)[0]) - 1 - len(items)
    for f in source.iterdir():
        if f.suffix != ".safetensors" and f.is_file() and f.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(f, out / f.name)
    text["num_experts"] = k
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    index = source / "model.safetensors.index.json"
    meta = json.loads(index.read_text()).get("metadata", {}) if index.exists() else {}
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": meta, "weight_map": weight_map}, indent=1))
    (out / "PRUNE.json").write_text(json.dumps({"source": str(source), "keep": k, "draft": draft, "experts": experts,
                                                "kept_ids": {str(l): v for l, v in keep.items()}}))
    return kept, dropped, size


def check(source, out):
    """Compare every tensor of the pruned folder with its source; return the number of tensors compared."""
    source, out = Path(source), Path(out)
    info = json.loads((out / "PRUNE.json").read_text())
    keep = {(int(l) if l != "mtp" else l): v for l, v in info["kept_ids"].items()}
    _, text, num_layers, _ = config_layers(out)
    assert text["num_experts"] == info["keep"] and all(v is None or len(v) == info["keep"] for v in keep.values())
    n = 0
    for f in sorted(source.glob("*.safetensors")):
        header, start = read_header(out / f.name)
        before = n
        with open(out / f.name, "rb") as g:
            for new, dtype, shape, size, ranges, _ in tensors_of(f, keep):
                h = header.get(new)
                if h is None or (h["dtype"], h["shape"], h["data_offsets"][1] - h["data_offsets"][0]) != (dtype, shape, size):
                    raise AssertionError(f"{f.name}: {new} is missing or has another shape")
                g.seek(start + h["data_offsets"][0])
                if any(g.read(len(piece)) != piece for piece in chunks(ranges)):
                    raise AssertionError(f"{f.name}: {new} differs from its source")
                n += 1
        if len(header) - ("__metadata__" in header) != n - before:
            raise AssertionError(f"{f.name}: the tensor count differs")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, required=True, help="the checkpoint folder (a snapshot of the hub cache)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=Path, help="the rank file of capture.py --rank")
    ap.add_argument("--keep", type=int, help="the experts to keep in each layer")
    ap.add_argument("--draft", choices=("rank", "first", "keep"), default="rank",
                    help="the experts of the MTP draft: by the rank, the first K, or all (serve the draft from the source)")
    ap.add_argument("--check", action="store_true", help="compare an existing pruned folder with its source and exit")
    a = ap.parse_args(argv)
    if not a.check:
        kept, dropped, size = prune(a.source, a.out, json.loads(a.rank.read_text()), a.keep, a.draft)
        print(f"kept {kept} tensors, dropped {dropped}, wrote {size / 2**30:.2f} GiB -> {a.out}")
    print(f"check: {check(a.source, a.out)} tensors equal their source")


if __name__ == "__main__":
    main()
