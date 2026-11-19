"""Tests of qwen_draft_quant.py on small synthetic tensors, with the target matching of vLLM 0.33 (no GPU)."""

import json

import torch
from torch import nn
from vllm.model_executor.layers.quantization.compressed_tensors.utils import find_matched_target, should_ignore_layer

import qwen_draft_quant as q
from qwen_prune import write_safetensors

WEIGHTS = {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16, "symmetric": True,
           "dynamic": False, "scale_dtype": "torch.float8_e4m3fn"}
CONFIG = {"num_experts": 4, "quantization_config": {
    "quant_method": "compressed-tensors", "ignore": ["model.visual.x", q.IGNORE_OLD],
    "config_groups": {
        "group_0": {"format": "float-quantized", "weights": {"num_bits": 8, "type": "float"},
                    "targets": [r"re:.*self_attn\.(q|k|v|o)_proj$", "re:.*lm_head"]},
        "group_1": {"format": "nvfp4-pack-quantized", "weights": WEIGHTS, "input_activations": {"num_bits": 4},
                    "targets": [q.EXPERTS_OLD, r"re:.*shared_expert\.(gate|up|down)_proj$"]}}}}


def scheme_of(config, name):
    """The group of a layer, as vLLM 0.33 finds it: the ignore list first, then the first matching target."""
    qc = config["quantization_config"]
    if should_ignore_layer(name, ignore=qc["ignore"]):
        return "ignored"
    targets = [t for g in qc["config_groups"].values() for t in g["targets"]]
    hit = find_matched_target(name, nn.Linear(1, 1), targets)
    return next(k for k, g in qc["config_groups"].items() if hit in g["targets"])


def test_the_patch_routes_the_draft_experts_to_a_weight_only_group():
    new = q.patch_quant_config(CONFIG)
    assert scheme_of(new, "mtp.layers.0.mlp.experts.7.gate_proj") == "group_draft_experts"
    assert scheme_of(new, "mtp.layers.0.mlp.experts.255.down_proj") == "group_draft_experts"
    assert new["quantization_config"]["config_groups"]["group_draft_experts"]["input_activations"] is None
    for name in ("mtp.fc", "mtp.layers.0.self_attn.q_proj", "mtp.layers.0.mlp.shared_expert.gate_proj", "mtp.layers.0.mlp.gate"):
        assert scheme_of(new, name) == "ignored", name   # the rest of the draft stays BF16
    assert scheme_of(new, "model.language_model.layers.3.mlp.experts.5.up_proj") == "group_1"   # the target is unchanged
    assert scheme_of(new, "model.language_model.layers.3.self_attn.q_proj") == "group_0"
    assert scheme_of(CONFIG, "mtp.layers.0.mlp.experts.7.gate_proj") == "ignored"   # the source is not changed
    try:
        q.patch_quant_config(new)
        raise AssertionError("a second patch must fail")
    except ValueError:
        pass


def test_an_expert_quantizes_with_a_shared_gate_up_scale():
    torch.manual_seed(0)
    args = q.args_of(CONFIG)
    gate_up, down = torch.randn(64, 32) * 0.02, torch.randn(32, 32) * 0.05
    t = q.expert_tensors(gate_up, down, args)
    assert t["gate_proj.weight_global_scale"].item() == t["up_proj.weight_global_scale"].item()
    assert t["gate_proj.weight_packed"].shape == (32, 16) and t["gate_proj.weight_scale"].shape == (32, 2)
    assert t["gate_proj.weight_scale"].dtype == torch.float8_e4m3fn and t["down_proj.input_global_scale"].item() == 1.0
    for proj, ref in (("gate_proj", gate_up[:32]), ("up_proj", gate_up[32:]), ("down_proj", down)):
        deq = q.dequantize({k: t[f"{proj}.{k}"] for k in ("weight_packed", "weight_scale", "weight_global_scale")})
        assert ((deq - ref).norm() / ref.norm()).item() < 0.15   # 4 bits with a scale per 16 weights


def test_build_writes_links_and_one_new_shard(tmp_path):
    torch.manual_seed(1)
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(CONFIG))
    (src / "tokenizer.json").write_text("{}")
    gate_up = (torch.randn(4, 64, 32) * 0.02).to(torch.bfloat16)
    down = (torch.randn(4, 32, 32) * 0.02).to(torch.bfloat16)
    other = torch.ones(8, dtype=torch.bfloat16)
    b = lambda t: t.contiguous().view(torch.uint8).numpy().tobytes()
    write_safetensors(src / "a.safetensors", [("model.x", "BF16", [8], b(other))])
    write_safetensors(src / "b.safetensors", [("mtp.norm.weight", "BF16", [8], b(other)),
                                              (q.GATE_UP, "BF16", [4, 64, 32], b(gate_up)), (q.DOWN, "BF16", [4, 32, 32], b(down))])
    (src / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {
        "model.x": "a.safetensors", "mtp.norm.weight": "b.safetensors", q.GATE_UP: "b.safetensors", q.DOWN: "b.safetensors"}}))
    old, new = q.build(src, out)
    assert (out / "a.safetensors").is_symlink() and (out / "tokenizer.json").is_symlink()
    assert not (out / "b.safetensors").is_symlink() and new < old / 3
    wm = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert q.GATE_UP not in wm and wm[f"{q.PREFIX}.3.down_proj.weight_packed"] == "b.safetensors" and wm["model.x"] == "a.safetensors"
    assert torch.equal(q.read_tensor(out / "b.safetensors", "mtp.norm.weight"), other)
    assert scheme_of(json.loads((out / "config.json").read_text()), f"{q.PREFIX}.0.gate_proj") == "group_draft_experts"
    t = {k: q.read_tensor(out / "b.safetensors", f"{q.PREFIX}.2.up_proj.{k}") for k in ("weight_packed", "weight_scale", "weight_global_scale")}
    ref = gate_up[2, 32:].float()
    assert ((q.dequantize(t) - ref).norm() / ref.norm()).item() < 0.15
