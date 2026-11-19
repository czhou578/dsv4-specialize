"""NVFP4 experts for the MTP draft of Qwen3.6-35B-A3B (plans/PLAN_QWEN_PRUNE.md §7.7).

The checkpoint stores the 256 experts of the draft in BF16, stacked (``mtp.layers.0.mlp.experts.gate_up_proj``
[256, 1024, 2048] and ``down_proj`` [256, 2048, 512], 1.5 GiB), and its quant config ignores the draft
(``re:^mtp.*``). This tool:

- ``--build OUT``: writes a draft folder. The shards with no draft tensor are symbolic links to the source. The
  shard of the draft is rewritten: the two stacked tensors become, for each expert, ``gate_proj``, ``up_proj`` and
  ``down_proj`` with ``weight_packed`` (u8), ``weight_scale`` (e4m3, one per 16 weights), ``weight_global_scale``
  (f32) and ``input_global_scale`` (f32, 1.0: weight-only mode does not use it). Gate and up share one global
  scale, as vLLM needs. The packing and the scales are those of compressed-tensors (``generate_gparam``,
  ``calculate_qparams``, ``NVFP4PackedCompressor``).
- ``--patch-config DIR``: changes the quant config of a folder (the target folder, and the draft folder): the draft
  experts get a weight-only NVFP4 group (no input quantization, NVFP4A16), and the rest of the draft stays BF16.
  vLLM builds the draft with the quant config of the target, so the target folder needs the patch.
- ``--check OUT``: the format check (one expert of the checkpoint, unpacked and packed again with its own global
  scale, gives the same scales and the same values; only the sign of some zeros differs) and the error of every new
  expert against its BF16 source.

Serve the draft with ``"moe_backend": "marlin"`` in the speculative config: the MoE backend of the target
(flashinfer_cutlass) has no weight-only mode.
"""

import argparse
import copy
import json
import shutil
import struct
import sys
from pathlib import Path

import torch
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.utils.helpers import calculate_qparams, generate_gparam

from qwen_prune import read_header, write_streaming

GATE_UP, DOWN = "mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj"
PREFIX = "mtp.layers.0.mlp.experts"
IGNORE_OLD, IGNORE_NEW = "re:^mtp.*", r"re:^mtp\.(?!layers\.0\.mlp\.experts\.).*"
EXPERTS_OLD = r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$"
EXPERTS_NEW = r"re:^(?!mtp\.).*mlp\.experts\.\d+\.(gate|up|down)_proj$"
DRAFT_TARGET = r"re:^mtp\.layers\.0\.mlp\.experts\.\d+\.(gate|up|down)_proj$"
DTYPES = {torch.uint8: "U8", torch.float8_e4m3fn: "F8_E4M3", torch.float32: "F32", torch.bfloat16: "BF16"}


def args_of(config):
    """The weight args of the NVFP4 group of the checkpoint."""
    groups = config["quantization_config"]["config_groups"]
    return QuantizationArgs(**next(g["weights"] for g in groups.values() if g["format"] == "nvfp4-pack-quantized"))


def quantize(w, args, global_scale=None):
    """``{weight_packed, weight_scale, weight_global_scale}`` of one matrix [N, K] (float32)."""
    g = generate_gparam(w.min(), w.max()) if global_scale is None else global_scale
    groups = w.reshape(w.shape[0], -1, args.group_size)
    scale, _ = calculate_qparams(groups.amin(-1), groups.amax(-1), args, global_scale=g)
    out = NVFP4PackedCompressor.compress({"weight": w, "weight_scale": scale, "weight_global_scale": g},
                                         QuantizationScheme(targets=["Linear"], weights=args))
    return {k: out[k] for k in ("weight_packed", "weight_scale", "weight_global_scale")}


def dequantize(t):
    packed, scale, g = t["weight_packed"], t["weight_scale"], t["weight_global_scale"]
    q = unpack_fp4_from_uint8(packed, packed.shape[0], packed.shape[1] * 2).float()
    return (q.reshape(q.shape[0], -1, 16) * (scale.float() / g).unsqueeze(-1)).reshape(q.shape)


def expert_tensors(gate_up, down, args):
    """The tensors of one expert: gate_up [2I, K] and down [K, I] in float32. Gate and up share a global scale."""
    half = gate_up.shape[0] // 2
    g13 = generate_gparam(gate_up.min(), gate_up.max())
    one = torch.ones(1, dtype=torch.float32)
    out = {}
    for proj, w, g in (("gate_proj", gate_up[:half], g13), ("up_proj", gate_up[half:], g13), ("down_proj", down, None)):
        out.update({f"{proj}.{k}": v for k, v in quantize(w.contiguous(), args, g).items()})
        out[f"{proj}.input_global_scale"] = one
    return out


def patch_quant_config(config):
    """The config with a weight-only NVFP4 group for the draft experts (a copy)."""
    config = copy.deepcopy(config)
    q = config["quantization_config"]
    if IGNORE_OLD not in q["ignore"]:
        raise ValueError(f"the quant config has no {IGNORE_OLD!r}: already patched, or another checkpoint")
    q["ignore"] = [IGNORE_NEW if x == IGNORE_OLD else x for x in q["ignore"]]
    groups = q["config_groups"]
    nvfp4 = next(k for k, g in groups.items() if g["format"] == "nvfp4-pack-quantized")
    groups[nvfp4]["targets"] = [EXPERTS_NEW if t == EXPERTS_OLD else t for t in groups[nvfp4]["targets"]]
    groups["group_draft_experts"] = {"format": "nvfp4-pack-quantized", "weights": groups[nvfp4]["weights"],
                                     "input_activations": None, "output_activations": None, "targets": [DRAFT_TARGET]}
    return config


def read_tensor(path, name, row=None):
    """A tensor of a safetensors file, or only its ``row`` of the first dimension (one expert of a stacked tensor)."""
    header, start = read_header(path)
    h = header[name]
    size, shape = h["data_offsets"][1] - h["data_offsets"][0], h["shape"]
    offset = start + h["data_offsets"][0]
    if row is not None:
        size //= shape[0]
        offset, shape = offset + row * size, shape[1:]
    with open(path, "rb") as f:
        f.seek(offset)
        data = bytearray(f.read(size))
    dtype = {v: k for k, v in DTYPES.items()}[h["dtype"]]
    return torch.frombuffer(data, dtype=dtype).reshape(shape)


def to_bytes(t):
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def build(source, out):
    source, out = Path(source), Path(out)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][GATE_UP]
    config = json.loads((source / "config.json").read_text())
    args = args_of(config)
    out.mkdir(parents=True, exist_ok=False)
    for f in source.iterdir():
        if f.name in ("config.json", "model.safetensors.index.json", shard):
            continue
        (out / f.name).symlink_to(f.resolve())
    (out / "config.json").write_text(json.dumps(patch_quant_config(config), indent=2) + "\n")
    header, start = read_header(source / shard)
    names = sorted((k for k in header if k != "__metadata__"), key=lambda k: header[k]["data_offsets"][0])
    items = [(k, header[k]["dtype"], header[k]["shape"], header[k]["data_offsets"][1] - header[k]["data_offsets"][0],
              [(source / shard, start + header[k]["data_offsets"][0], header[k]["data_offsets"][1] - header[k]["data_offsets"][0])])
             for k in names if k not in (GATE_UP, DOWN)]

    def expert(e):
        return expert_tensors(read_tensor(source / shard, GATE_UP, e).float(), read_tensor(source / shard, DOWN, e).float(), args)

    current = {}   # the tensors of the expert that the writer is at; one expert sits in memory at a time

    def part(e, k):
        def get():
            if current.get("e") != e:
                current.clear()
                current.update(e=e, t=expert(e))
            return to_bytes(current["t"][k])
        return get

    shapes = expert(0)   # all experts have the same shapes, so the header is known before the writer quantizes
    for e in range(header[GATE_UP]["shape"][0]):
        for k, v in shapes.items():
            items.append((f"{PREFIX}.{e}.{k}", DTYPES[v.dtype], v.shape, v.numel() * v.element_size(), part(e, k)))
    write_streaming(out / shard, items, header.get("__metadata__"))
    wm = {k: v for k, v in index["weight_map"].items() if k not in (GATE_UP, DOWN)}
    wm.update({name: shard for name, *_ in items if name.startswith(PREFIX + ".")})
    (out / "model.safetensors.index.json").write_text(json.dumps({**index, "weight_map": wm}, indent=1))
    new = sum(size for name, _, _, size, _ in items if name.startswith(PREFIX + "."))
    old = sum(header[k]["data_offsets"][1] - header[k]["data_offsets"][0] for k in (GATE_UP, DOWN))
    return old, new


def check(source, out, sample_layer=39):
    """``(the values equal, share of equal scales, worst and mean relative error of the new experts)``."""
    source, out = Path(source), Path(out)
    config = json.loads((source / "config.json").read_text())
    args = args_of(config)
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    base = f"model.language_model.layers.{sample_layer}.mlp.experts.0.gate_proj"
    t = {k: read_tensor(source / index[f"{base}.{k}"], f"{base}.{k}") for k in ("weight_packed", "weight_scale", "weight_global_scale")}
    again = quantize(dequantize(t), args, t["weight_global_scale"])
    same_values = torch.equal(dequantize(again), dequantize(t))   # the bytes differ only in the sign of zeros
    same_scale = (again["weight_scale"].view(torch.uint8) == t["weight_scale"].view(torch.uint8)).float().mean().item()
    shard = index[GATE_UP]
    errs = []
    for e in range(read_header(source / shard)[0][GATE_UP]["shape"][0]):
        gate_up, down = read_tensor(source / shard, GATE_UP, e), read_tensor(source / shard, DOWN, e)
        for proj, ref in (("gate_proj", gate_up[:gate_up.shape[0] // 2]), ("up_proj", gate_up[gate_up.shape[0] // 2:]),
                          ("down_proj", down)):
            q = {k: read_tensor(out / shard, f"{PREFIX}.{e}.{proj}.{k}") for k in ("weight_packed", "weight_scale", "weight_global_scale")}
            ref = ref.float()
            errs.append(((dequantize(q) - ref).norm() / ref.norm()).item())
    return same_values, same_scale, max(errs), sum(errs) / len(errs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, required=True, help="the checkpoint folder (a snapshot of the hub cache)")
    cmd = ap.add_mutually_exclusive_group(required=True)
    cmd.add_argument("--build", type=Path, metavar="OUT")
    cmd.add_argument("--patch-config", type=Path, metavar="DIR")
    cmd.add_argument("--check", type=Path, metavar="OUT")
    a = ap.parse_args(argv)
    if a.build:
        old, new = build(a.source, a.build)
        print(f"draft experts: {old / 2**30:.2f} GiB in BF16 -> {new / 2**30:.2f} GiB in NVFP4 -> {a.build}")
    elif a.patch_config:
        f = a.patch_config / "config.json"
        if f.is_symlink():
            sys.exit(f"{f} is a link: patch a copy")
        f.write_text(json.dumps(patch_quant_config(json.loads(f.read_text())), indent=2) + "\n")
        print(f"patched {f}")
    else:
        same, ss, worst, mean = check(a.source, a.check)
        print(f"format check: values equal {same}; {ss:.4%} of the scales equal the checkpoint's")
        print(f"draft experts against BF16: relative error mean {mean:.4f}, worst {worst:.4f}")


if __name__ == "__main__":
    main()
