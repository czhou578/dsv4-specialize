"""Tests of the loader patch (plans/PLAN_LOADER_PATCH.md §5.2). They need vLLM: run ./test_loader_patch.sh, which
runs them in the server image with the patched files of image-patch/perlayer/ mounted. No GPU memory is used.
"""

from types import SimpleNamespace

try:
    import pytest
except ImportError:   # python -m unittest in the venv: the image of ./test_loader_patch.sh has pytest and vLLM
    import unittest
    raise unittest.SkipTest("needs pytest and vLLM: run ./test_loader_patch.sh")

torch = pytest.importorskip("torch")
exl3 = pytest.importorskip("vllm.model_executor.layers.quantization.exl3")
model = pytest.importorskip("vllm.models.deepseek_v4.nvidia.model")

KS = [160, 160, 160] + [150 - (i % 7) for i in range(40)]   # 43 MoE layers, 8 counts


def metadata(counts):
    return {"bits": 3.0, "codebook": "mcg", "experts_per_layer": counts, "moe_layers": [0, 42], "tp": 1,
            "tensor_schema": "layers.{L}.ffn.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"}


def config(counts):
    cfg = exl3.Exl3Config.__new__(exl3.Exl3Config)
    cfg._configure_rank_sliced(metadata(counts))
    return cfg


def test_a_uniform_count_is_the_count_of_every_layer():
    cfg = config(160)
    assert cfg.rank_sliced_layer_experts("model.layers.26.ffn.experts") == 160
    assert cfg.rank_sliced_layer_bitrates("model.layers.26.ffn.experts") == (3,) * 160


def test_a_per_layer_list_gives_each_layer_its_count():
    cfg = config(KS)
    for layer in (0, 3, 26, 42):
        assert cfg.rank_sliced_layer_experts(f"model.layers.{layer}.ffn.experts") == KS[layer]
    assert len(cfg.rank_sliced_layer_bitrates("model.layers.26.ffn.experts")) == KS[26]
    with pytest.raises(ValueError, match="outside the rank-sliced MoE layers"):
        cfg.rank_sliced_layer_experts("model.layers.43.ffn.experts")


def test_a_bad_list_stops_the_load():
    for bad in (KS[:-1], KS[:-1] + [0]):   # a missing layer; a zero count
        with pytest.raises(ValueError, match="one positive count"):
            config(bad)
    mixed = {**metadata(KS), "bits": "mixed", "k_values": [3, 4], "bits_per_expert": "t.json:k"}
    with pytest.raises(NotImplementedError, match="mixed-bitrate"):
        exl3.Exl3Config.__new__(exl3.Exl3Config)._configure_rank_sliced(mixed)


def test_the_model_reads_the_count_of_its_layer_and_the_draft_keeps_its_own():
    cfg = SimpleNamespace(n_routed_experts=160, n_routed_experts_per_layer=KS)
    assert model._layer_routed_experts(cfg, "model.layers.26.ffn") == KS[26]
    assert model._layer_routed_experts(SimpleNamespace(n_routed_experts=160), "model.layers.26.ffn") == 160
    draft = SimpleNamespace(n_routed_experts=64, n_routed_experts_per_layer=KS)   # dspark.py sets 64 for its build
    assert model._layer_routed_experts(draft, "model.layers.43.ffn") == 64
    with pytest.raises(ValueError, match="outside"):
        model._layer_routed_experts(SimpleNamespace(n_routed_experts=150, n_routed_experts_per_layer=KS),
                                    "model.layers.0.ffn")


class FakeSpec:
    def __init__(self, nbytes):
        self.shape, self.dtype, self.device = (nbytes,), torch.uint8, torch.device("cpu")


class FakeApi:
    """plan_weights and plan of SparkInfer, with a scratch size that grows with the count and the capacity."""
    def plan_weights(self, **kw):
        return SimpleNamespace(num_experts=kw["num_experts"])

    @staticmethod
    def plan_fn(max_tokens, block_m, weight_plan):
        return None, FakeSpec(1000 * max_tokens + weight_plan.num_experts)


def test_the_pool_takes_the_largest_scratch_and_each_count_gets_a_view_of_it():
    layer = SimpleNamespace(exl3_layer_bitrates=(3,), activation=SimpleNamespace(value="silu"),
                            exl3_params_dtype=torch.bfloat16, exl3_hidden_size=4096,
                            exl3_intermediate_size_per_partition=2048, exl3_trellis_tile_config=(128, 128, 32, 512))
    counts = sorted(set(KS))
    pool = exl3._plan_scratch_pool(FakeApi(), layer, counts, FakeApi.plan_fn,
                                   {"decode": (32, 8), "prefill": (8224, 64)}, torch.device("cpu"))
    assert {r: t.numel() for r, t in pool.items()} == {"decode": 32000 + 160, "prefill": 8224000 + 160}
    small = exl3._pool_view(pool["prefill"], FakeSpec(8224000 + 144))
    assert small.data_ptr() == pool["prefill"].data_ptr() and small.numel() == 8224000 + 144
    with pytest.raises(RuntimeError, match="exceeds its pool"):
        exl3._pool_view(pool["decode"], FakeSpec(32000 + 161))
