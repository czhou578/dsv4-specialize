"""CPU tests of image-patch/layer_skip.py (plans/PLAN_FEWER_LAYERS.md §6.5): the parser, the probe and Gram sums against
float64, and the skip file. They need torch and numpy, not vLLM: run ./test_layer_skip.sh (the server image)."""

import json
import os
import sys
from pathlib import Path

try:
    import pytest
except ImportError:   # python -m unittest in the venv: the server image has pytest and torch
    import unittest
    raise unittest.SkipTest("needs pytest and torch: run ./test_layer_skip.sh")

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parent / "image-patch"))
import layer_skip as ls  # noqa: E402

H, HC = 64, 4


def test_parse():
    assert ls.parse("") == {} == ls.parse(None)
    assert ls.parse("L17,F21") == {17: "L", 21: "F"} and ls.parse("A9, F9") == {9: "L"}
    assert ls.canon(ls.parse("F21,A4,L17")) == "A4,L17,F21" and ls.canon({}) == "none"
    assert ls.parse("L1", num_layers=8, protect_first=1, protect_last=2) == {1: "L"}
    for spec in ["L2", "A0", "F40", "L42", "L17,L17", "L17,A17", "F17,L17", "X17", "L", "l17", "L17x"]:
        with pytest.raises(ValueError):
            ls.parse(spec)


def states(n, tokens=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(tokens, HC, H, generator=g).to(torch.bfloat16) for _ in range(n)]


def run(probe, events, s, positions, count_from=0):
    probe.begin(positions, count_from)
    for (layer, half), st in zip(events, s):
        probe.event(layer, half, st)


def test_probe_sums_match_a_float64_reference():
    # layer 0 runs whole, layer 1 skips its FFN, layer 2 skips its attention, layer 3 runs whole
    probe, s, pos = ls.Probe(4, "cpu"), states(7), torch.arange(5)
    run(probe, [(0, "A"), (0, "F"), (1, "A"), (2, "F"), (3, "A"), (3, "F"), (None, None)], s, pos, count_from=2)
    want = {"A0": (0, 1), "F0": (1, 2), "L0": (0, 2), "A1": (2, 3), "F2": (3, 4), "A3": (4, 5), "F3": (5, 6),
            "L3": (4, 6)}
    w = (pos >= 2).double()
    for name, (i, j) in want.items():
        a, b = s[i].double().reshape(5, -1), s[j].double().reshape(5, -1)
        layer, c = int(name[1:]), ls.KINDS.index(name[0])
        cos = ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)) * w).sum().item()
        rel = ((b - a).square().sum(1) / a.square().sum(1) * w).sum().item()
        assert probe.cos[layer, c].item() == pytest.approx(cos, rel=1e-5)
        assert probe.rel[layer, c].item() == pytest.approx(rel, rel=1e-5)
        assert probe.tokens[layer, c].item() == 3
    seen = {(int(n[1:]), ls.KINDS.index(n[0])) for n in want}
    assert all(probe.tokens[i, c] == 0 for i in range(4) for c in range(3) if (i, c) not in seen)
    probe, s = ls.Probe(4, "cpu"), states(5, seed=1)
    run(probe, [(0, "A"), (0, "F"), (2, "A"), (2, "F"), (None, None)], s, pos)   # layer 1: L skip
    a, b = s[1].double().reshape(5, -1), s[2].double().reshape(5, -1)            # F0 ends at the state of layer 2
    assert probe.cos[0, 1].item() == pytest.approx((torch.nn.functional.cosine_similarity(a, b)).sum().item(), rel=1e-5)
    assert probe.tokens[1].sum() == 0


def test_gram_sums_match_a_float64_reference():
    probe, s, pos = ls.Probe(4, "cpu"), states(3, seed=2), torch.arange(5)
    probe.set_gram(["F0", "L0"])
    run(probe, [(0, "A"), (0, "F"), (None, None)], s, pos, count_from=1)
    w = (pos >= 1).double().repeat_interleave(HC)[:, None]
    for name, (i, j) in {"F0": (1, 2), "L0": (0, 2)}.items():
        x = s[i].double().reshape(-1, H)
        d = s[j].double().reshape(-1, H) - x
        xtx, xtd, rows = probe.gram[name]
        assert torch.allclose(xtx.double(), (x * w).T @ x, rtol=1e-4, atol=1e-3)
        assert torch.allclose(xtd.double(), (x * w).T @ d, rtol=1e-4, atol=1e-3) and rows == 4 * HC
    assert set(probe.gram_arrays()) == {f"{k}_{n}" for k in ("xtx", "xtd", "rows") for n in ("F0", "L0")}
    with pytest.raises(ValueError):
        probe.set_gram(["F0", "L0", "A1", "A2"])


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn, self.ffn = torch.nn.Linear(16, 16, bias=False), torch.nn.Linear(16, 64, bias=False)


def controller(env, **kw):
    mods = {i: Layer() for i in range(8)}
    return ls.Controller(mods, 8, env=env, **{"protect_first": 1, "protect_last": 2, **kw}), mods


def test_controller_refuses_unsafe_boots(tmp_path):
    f = str(tmp_path / "skip.json")
    for env, kw in [({"LAYER_SKIP": "L3"}, {}),                         # the compile cache is on
                    ({"LAYER_SKIP_FILE": f}, {"compile_cache": False}),  # a run-time file needs eager
                    ({"LAYER_PROBE": "1", "LAYER_SKIP_FILE": f}, {"compile_cache": False}),
                    ({"LAYER_HEAL": "/x.safetensors"}, {"eager": True}),
                    ({"LAYER_PROBE": "1"}, {"eager": True}),              # no folder for the sums
                    ({"LAYER_SKIP": "L0"}, {"eager": True})]:
        with pytest.raises((RuntimeError, ValueError)):
            controller(env, **kw)


def test_controller_fixed_set_and_boot_line():
    c, mods = controller({"LAYER_SKIP": "F4,L2"}, compile_cache=False)
    assert [(m.layer_idx, m.layer_skip, m.layer_probe) for m in mods.values()] == \
        [(i, {2: "L", 4: "F"}.get(i), None) for i in range(8)] and not c.runtime
    gib = (2 * 16 * 16 + 16 * 64 * 2) * 4 / 2**30
    assert c.boot_line() == f"layer skip: set L2,F4 (fixed); skipped weights {gib:.2f} GiB; heal none; probe off"


def write(path, cfg):
    path.write_text(json.dumps(cfg))
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))   # a new mtime, also on a coarse clock


def test_controller_reads_the_skip_file_and_writes_the_sums(tmp_path):
    f, pos = tmp_path / "skip.json", torch.arange(6)
    c, mods = controller({"LAYER_SKIP_FILE": str(f), "LAYER_PROBE": "1"}, eager=True)
    c.step_begin(pos)                                     # no file yet: no skip
    assert c.loads == 0 and all(m.layer_skip is None and m.layer_probe is c.probe for m in mods.values())
    write(f, {"skip": "L3,A5", "count_from": 2, "gram": ["F1"]})
    c.step_begin(pos)
    assert (c.loads, c.count_from, mods[3].layer_skip, mods[5].layer_skip) == (1, 2, "L", "A")
    s = states(3, tokens=6, seed=3)
    for (layer, half), st in zip([(1, "A"), (1, "F"), (None, None)], s):
        c.probe.event(layer, half, st)
    c.step_end(s[2])
    with np.load(tmp_path / "layer_probe.npz") as z:
        assert (str(z["skip"]), int(z["loads"]), str(z["error"]), int(z["steps"])) == ("L3,A5", 1, "", 1)
        assert z["tokens"][1].tolist() == [4, 4, 4] and str(z["gram"]) == "F1"
    write(f, {"skip": "L0"})                              # refused: the old set stays, the error is in the sums
    c.step_begin(pos)
    c.step_end(s[2])
    with np.load(tmp_path / "layer_probe.npz") as z:
        assert mods[3].layer_skip == "L" and c.loads == 1
        assert str(z["skip"]) == "L3,A5" and "cannot skip" in str(z["error"])
    write(f, {"skip": "", "gram": ["F1"], "gram_flush": 1})   # the flush writes the Gram sums of F1
    c.step_begin(pos)
    assert c.error == "" and all(m.layer_skip is None for m in mods.values())
    with np.load(tmp_path / "layer_gram.npz") as z:
        assert int(z["flush"]) == 1 and z["xtx_F1"].shape == (H, H) and float(z["rows_F1"]) == 4 * HC
    write(f, {"skip": "", "sums": False, "gram": []})
    c.step_begin(pos)
    assert c.probe.gram == {} and c.probe.sums is False
