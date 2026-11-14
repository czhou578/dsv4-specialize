"""Tests of qwen_mtp_patch/sitecustomize.py with stand-in configs (no vLLM)."""

import importlib.util
import sys
import types
from pathlib import Path

spec = importlib.util.spec_from_file_location("qwen_mtp_patch", Path(__file__).parent / "qwen_mtp_patch/sitecustomize.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)
sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, mp._Finder)]   # no hook on the real vLLM in tests


def configs(target_experts, draft_experts, same=False):
    target = types.SimpleNamespace(hf_text_config=types.SimpleNamespace(num_experts=target_experts, hidden_size=2048))
    draft = target if same else types.SimpleNamespace(hf_text_config=types.SimpleNamespace(num_experts=draft_experts))
    return types.SimpleNamespace(model_config=target, quant_config="q",
                                 speculative_config=types.SimpleNamespace(draft_model_config=draft))


def test_the_draft_gets_its_own_expert_count():
    vc = configs(208, 256)
    out = mp.draft_config(vc)
    assert out is not vc and out.model_config.hf_text_config.num_experts == 256
    assert out.model_config.hf_text_config.hidden_size == 2048 and out.quant_config == "q"
    assert vc.model_config.hf_text_config.num_experts == 208   # the target config is unchanged


def test_no_change_for_equal_counts_or_no_draft():
    for vc in (configs(256, 256), configs(208, 0, same=True)):
        assert mp.draft_config(vc) is vc
    vc = configs(208, 256)
    vc.speculative_config = None
    assert mp.draft_config(vc) is vc


def test_patch_wraps_init_once():
    seen = []

    class MTP:
        def __init__(self, *, vllm_config, prefix=""):
            seen.append(vllm_config.model_config.hf_text_config.num_experts)

    module = types.SimpleNamespace(Qwen3_5MTP=MTP)
    mp.patch(module)
    mp.patch(module)
    MTP(vllm_config=configs(176, 256))
    assert seen == [256]
