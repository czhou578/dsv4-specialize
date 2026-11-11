"""Tests of the FP8 head of the target (plans/PLAN_SERVE_TUNE.md §7.2). They need torch, vLLM and a GPU: run
./test_fp8_head.sh, which runs them in the server image (about 0.3 GB of GPU memory, safe beside the live server).
"""

import sys
from pathlib import Path

try:
    import pytest
except ImportError:   # python -m unittest in the venv: the image of ./test_fp8_head.sh has pytest and torch
    import unittest
    raise unittest.SkipTest("needs pytest, torch and a GPU: run ./test_fp8_head.sh")

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent / "image-patch"))
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402
from vllm.model_executor.layers.logits_processor import LogitsProcessor  # noqa: E402
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead  # noqa: E402
import fp8_target_head as fh  # noqa: E402

V, H, ROWS = 16384, 4096, 256


@pytest.fixture(scope="module")
def parts():
    """A real ParallelLMHead and LogitsProcessor (TP 1); the hidden states point at known rows, so each has a top-1."""
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="nccl",
                                     distributed_init_method="tcp://127.0.0.1:29517")
        initialize_model_parallel(1, 1)
    def make():   # the same weights on each call
        g = torch.Generator(device="cuda").manual_seed(0)
        head = ParallelLMHead(V, H, params_dtype=torch.bfloat16)
        head.weight.data.copy_(torch.randn(V, H, generator=g, device="cuda").mul_(0.02).bfloat16())
        return head, LogitsProcessor(V)
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        head, lp = make()
    g = torch.Generator(device="cuda").manual_seed(1)
    rows = torch.randint(0, V, (ROWS,), generator=g, device="cuda")
    hidden = head.weight[rows].float() * 40 + torch.randn(ROWS, H, generator=g, device="cuda")
    return make, head, lp, hidden.bfloat16(), rows


def test_the_fp8_logits_match_bf16(parts, monkeypatch):
    make, head, lp, hidden, rows = parts
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        head, lp = make()
    ref = lp(head, hidden).float()
    monkeypatch.setenv("VLLM_DSPARK_FP8_DRAFT_HEAD", "0")
    fh.install(head)
    out = lp(head, hidden).float()
    cos = torch.nn.functional.cosine_similarity(out, ref, dim=-1)
    assert out.shape == ref.shape == (ROWS, V)
    assert cos.min().item() > 0.999
    assert (out.argmax(-1) == ref.argmax(-1)).float().mean().item() == 1.0
    assert (ref.argmax(-1) == rows).float().mean().item() == 1.0   # the hidden states have a clear top-1


def test_install_frees_the_bf16_weight_and_the_draft_shares_the_fp8_copy(parts, monkeypatch):
    make, _, _, hidden, _ = parts
    monkeypatch.setenv("VLLM_DSPARK_FP8_DRAFT_HEAD", "0")
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        target, lp = make()
    draft = type("Draft", (), {})()
    draft.lm_head = target                       # load_dspark_model: has_own_lm_head = False
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    freed = fh.install(target)
    torch.cuda.synchronize()
    assert freed == V * H * 2 and target.weight.numel() == 0
    assert before - torch.cuda.memory_allocated() >= V * H * 2 - V * H - V * 2 - 4096   # BF16 out, FP8 in
    assert draft.lm_head.quant_method.head.weight_fp8 is target.quant_method.head.weight_fp8
    assert lp(draft.lm_head, hidden).shape == (ROWS, V)


def test_the_fp8_draft_head_must_stay_off(parts, monkeypatch):
    make = parts[0]
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        head, _ = make()
    monkeypatch.setenv("VLLM_DSPARK_FP8_DRAFT_HEAD", "1")
    with pytest.raises(RuntimeError, match="DSPARK_FP8_DRAFT_HEAD=0"):
        fh.install(head)
    assert head.weight.numel() == V * H   # nothing changed


def test_a_cuda_graph_replay_equals_an_eager_call(parts, monkeypatch):
    make, _, _, hidden, _ = parts
    monkeypatch.setenv("VLLM_DSPARK_FP8_DRAFT_HEAD", "0")
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        head, lp = make()
    fh.install(head)
    x = hidden[:6].clone()   # one verify step at depth 5: 6 rows
    eager = lp(head, x)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        lp(head, x)          # warm-up outside the capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = lp(head, x)
    x.copy_(hidden[6:12])
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, lp(head, hidden[6:12]))
    assert not torch.equal(out, eager)
