"""Tests of the router patch (plans/PLAN_FEWER_EXPERTS.md §6). They need torch and vLLM: run ./test_moe_drop.sh,
which runs them in the server image (the CUDA graph test needs a few MB of GPU memory, safe beside the live server).
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "image-patch"))
import moe_drop as md  # noqa: E402

E, K, M, SCALE = 216, 6, 256, 1.5


def inputs(seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)   # the DSv4 bias is near-uniform (about 8.08)
    return (torch.randn(M, E, generator=g) * 2).to(device), (8.08 + 0.01 * torch.randn(E, generator=g)).to(device)


def stock(logits, bias, k, table=None, tokens=None):
    """The stock fallback router: top-k by score + bias, weights from the unbiased scores, renormalized, scaled."""
    scores = torch.sqrt(F.softplus(logits.float()))
    ids = table[tokens] if table is not None else torch.topk(scores + bias, k=k, dim=-1).indices
    w = scores.gather(1, ids)
    return w / w.sum(-1, keepdim=True) * SCALE, ids.to(torch.int32)


def hash_case(seed=1):
    g = torch.Generator().manual_seed(seed)
    table = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(1000)])
    logits = inputs(seed)[0]
    return (logits, *stock(logits, None, K, table, torch.randint(0, 1000, (M,), generator=g)))


def rule(drop="none", norm="kept", layers="all"):
    return md.parse({"MOE_DROP": drop, "MOE_NORM": norm, "MOE_LAYERS": layers})


def test_parse_and_layers(monkeypatch):
    r = md.parse({"MOE_DROP": "threshold:0.1:4", "MOE_NORM": "anchor:8", "MOE_LAYERS": "3-5,40"})
    assert (r.drop, r.t, r.min_keep, r.norm, r.k2) == ("threshold", 0.1, 4, "anchor", 8)
    assert r.layers == frozenset({3, 4, 5, 40}) and r.applies(4) and not r.applies(6)
    assert not md.parse({}).active and not md.parse({}).applies(3)
    with pytest.raises(ValueError):
        md.parse({"MOE_DROP": "top:4"})
    monkeypatch.setattr(md, "_next_layer", 0)   # the draft (64 experts) gets no layer index
    assert md.next_layer(64, False) == -1 and [md.next_layer(216, i < 3) for i in range(4)] == [0, 1, 2, 3]


def test_lowest1_kept_is_top5():
    logits, bias = inputs()
    w6, ids6 = stock(logits, bias, 6)
    w5, ids5 = stock(logits, bias, 5)
    w, ids = md.apply(rule("lowest:1"), w6, ids6, logits, bias, False, SCALE)
    assert ids.shape == (M, K) and ((ids == -1).sum(-1) == 1).all()
    for r in range(M):   # the same experts with the same weights, in any slot order
        got, want = dict(zip(ids[r].tolist(), w[r].tolist())), dict(zip(ids5[r].tolist(), w5[r].tolist()))
        got.pop(-1)
        assert got.keys() == want.keys() and all(abs(got[i] - want[i]) < 1e-5 for i in got)


def test_lowest_hash_drops_weakest_weight():
    logits, w6, ids6 = hash_case()
    w, ids = md.apply(rule("lowest:2"), w6, ids6, logits, None, True, SCALE)
    weakest = torch.argsort(w6, dim=-1)[:, :2]
    assert (ids.gather(1, weakest) == -1).all() and ((ids == -1).sum(-1) == 2).all()
    assert torch.allclose(w.sum(-1), torch.full((M,), SCALE), atol=1e-5)


def test_anchor6_keeps_trained_weights():
    logits, bias = inputs()
    w6, ids6 = stock(logits, bias, 6)
    w, ids = md.apply(rule("lowest:1", "anchor:6"), w6, ids6, logits, bias, False, SCALE)
    kept = ids >= 0
    assert torch.allclose(w[kept], w6[kept], atol=1e-6) and (w[~kept] == 0).all() and (w.sum(-1) < SCALE).all()


def test_anchor8_uses_top8_mass():
    logits, bias = inputs()
    w6, ids6 = stock(logits, bias, 6)
    w, ids = md.apply(rule("lowest:2", "anchor:8"), w6, ids6, logits, bias, False, SCALE)
    scores = torch.sqrt(F.softplus(logits))
    mass = scores.gather(1, torch.topk(scores + bias, 8).indices).sum(-1, keepdim=True)
    want = torch.where(ids >= 0, scores.gather(1, ids6.long()) / mass * SCALE, torch.zeros(()))
    assert torch.allclose(w, want, atol=1e-6)
    # the hash layers have six candidates only: anchor:8 acts as anchor:6 there
    hl, hw, hids = hash_case()
    a8 = md.apply(rule("lowest:2", "anchor:8"), hw, hids, hl, None, True, SCALE)
    a6 = md.apply(rule("lowest:2", "anchor:6"), hw, hids, hl, None, True, SCALE)
    assert torch.equal(a8[0], a6[0]) and torch.equal(a8[1], a6[1])


def test_threshold_keeps_min():
    logits, bias = inputs()
    w6, ids6 = stock(logits, bias, 6)
    w, ids = md.apply(rule("threshold:0.16:4"), w6, ids6, logits, bias, False, SCALE)
    shares = w6 / w6.sum(-1, keepdim=True)
    rank = torch.argsort(torch.argsort(shares, dim=-1, descending=True), dim=-1)
    dropped = ids == -1   # below the share, and not one of the 4 strongest
    assert ids.shape == (M, K) and ((~dropped).sum(-1) >= 4).all() and dropped.any()
    assert torch.equal(dropped, (shares < 0.16) & (rank >= 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_cuda_graph_replay():
    r = rule("threshold:0.12:3", "anchor:8")
    logits, bias = inputs(device="cuda")
    w6, ids6 = stock(logits, bias, 6)
    static = [logits.clone(), w6.clone(), ids6.clone()]
    md.apply(r, static[1], static[2], static[0], bias, False, SCALE)  # warm up
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = md.apply(r, static[1], static[2], static[0], bias, False, SCALE)
    logits2, _ = inputs(seed=7, device="cuda")
    w2, ids2 = stock(logits2, bias, 6)
    [dst.copy_(src) for dst, src in zip(static, (logits2, w2, ids2))]
    graph.replay()
    want = md.apply(r, w2, ids2, logits2, bias, False, SCALE)
    assert torch.equal(out[1], want[1]) and torch.allclose(out[0], want[0], atol=1e-6)


def test_patched_router():
    router_mod = pytest.importorskip("vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router")
    if not hasattr(router_mod, "moe_drop"):
        pytest.skip("the image's router file is not the patched one: run ./test_moe_drop.sh")
    logits, bias = inputs()
    router = router_mod.FusedTopKBiasRouter(6, E, torch.nn.Parameter(bias, requires_grad=False), True, SCALE,
                                            scoring_func="sqrtsoftplus")
    hidden = torch.zeros(M, 8)
    base = router_mod.fused_topk_bias(hidden, logits, "sqrtsoftplus", bias, 6, True, torch.int32,
                                      routed_scaling_factor=SCALE)
    router_mod.moe_drop.RULE = rule()
    w, ids = router._compute_routing(hidden, logits, torch.int32)
    assert torch.equal(w, base[0]) and torch.equal(ids, base[1])   # MOE_DROP=none: the stock output, bit for bit
    router_mod.moe_drop.RULE = rule("lowest:1", "anchor:6")
    w, ids = router._compute_routing(hidden, logits, torch.int32)
    want = md.apply(rule("lowest:1", "anchor:6"), base[0], base[1], logits, bias, False, SCALE)
    assert torch.equal(ids, want[1]) and torch.allclose(w, want[0])

def test_capture_profile():
    """The capture hook of Phase A (PLAN_FEWER_EXPERTS.md §5) against a direct count."""
    cap = pytest.importorskip("vllm.model_executor.layers.fused_moe.routed_experts_capturer")
    if not hasattr(cap, "PROFILE_RULES"):
        pytest.skip("the image's capturer is not the patched one: run ./test_moe_drop.sh")
    from types import SimpleNamespace
    rules = [md.parse({"MOE_DROP": r}) for r in cap.PROFILE_RULES]
    stub = SimpleNamespace(num_experts=E, num_hash_layers=3, moe_drop=md, prof_rules=rules, prof={
        k: torch.zeros(s, dtype=torch.float64) for k, s in {"prof_pos": (4, K), "prof_hist": (4, 2, 10),
        "prof_valid": (4,), "prof_kept": (4, len(rules)), "prof_union": (4, len(rules)), "prof_windows": (4,)}.items()})
    logits, bias = inputs()
    w6, ids6 = stock(logits, bias, 6)
    ids, w = ids6.long().clone(), w6.double()
    ids[0, 0] = E                                    # one padded token: not counted, its window neither
    cap.RoutedExpertsCapturer.capture_profile(stub, 3, ids, w)
    p = stub.prof
    assert p["prof_valid"][3] == M - 1 and p["prof_windows"][3] == M // 6 - 1
    assert p["prof_kept"][3, 0] == 6 * (M - 1) and p["prof_kept"][3, 1] == 5 * (M - 1)
    union = sum(len(set(ids[i:i + 6].flatten().tolist())) for i in range(6, (M // 6) * 6, 6))
    assert p["prof_union"][3, 0] == union
    top5 = sum(len(set(ids[i:i + 6, :5].flatten().tolist())) for i in range(6, (M // 6) * 6, 6))
    assert p["prof_union"][3, 1] == top5             # layers 3-42: lowest:1 drops the last slot (selection order)
    assert torch.allclose(p["prof_pos"][3].sum(), torch.tensor(M - 1.0, dtype=torch.float64))
    assert p["prof_hist"][3].sum() == 2 * (M - 1)


def test_anchor_on_the_live_top5_folder():
    """C6 (PLAN_FEWER_EXPERTS.md §9): the live folder (top-k 5, 5-column hash tables) with MOE_NORM=anchor:6 only."""
    r = rule("none", "anchor:6")
    logits, bias = inputs()
    w5, ids5 = stock(logits, bias, 5)
    w, ids = md.apply(r, w5, ids5, logits, bias, False, SCALE)
    scores = torch.sqrt(F.softplus(logits))
    mass6 = scores.gather(1, torch.topk(scores + bias, 6).indices).sum(-1, keepdim=True)
    assert torch.equal(ids, ids5) and torch.allclose(w, scores.gather(1, ids5.long()) / mass6 * SCALE, atol=1e-6)
    g = torch.Generator().manual_seed(1)
    table = torch.stack([torch.randperm(E, generator=g)[:5] for _ in range(1000)])
    hl = inputs(1)[0]
    hw, hids = stock(hl, None, 5, table, torch.randint(0, 1000, (M,), generator=g))
    w2, ids2 = md.apply(r, hw, hids, hl, None, True, SCALE)   # the hash layers keep the stock rule
    assert torch.equal(ids2, hids) and torch.allclose(w2, hw, atol=1e-6)
