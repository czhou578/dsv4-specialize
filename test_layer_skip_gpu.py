"""GPU tests of the patched decoder layer (plans/PLAN_FEWER_LAYERS.md §6.5): stub layers with the real mHC kernels
and random linear maps in place of the attention and the FFN. They need the server image and the patched model file:
run ./test_layer_skip.sh after image-patch/build_skip.sh (about 1 GB of GPU memory).
"""

try:
    import pytest
except ImportError:   # python -m unittest in the venv: the server image has pytest, torch and vLLM
    import unittest
    raise unittest.SkipTest("needs the server image: run ./test_layer_skip.sh")

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
from vllm.models.deepseek_v4.nvidia import layer_skip as ls  # noqa: E402
from vllm.models.deepseek_v4.nvidia import model as m  # noqa: E402

H, HC, T = 4096, 4, 24
EPS, ITERS, ALPHA = 1e-6, 20, 2.0


class Norm(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.weight = torch.nn.Parameter((1 + 0.1 * torch.randn(H, generator=g, device="cuda")).bfloat16())
        self.variance_epsilon = EPS


class Map(torch.nn.Module):
    """A random linear map with the call signature of the attention (positions, x, None) or the FFN (x, ids)."""
    def __init__(self, g, arg):
        super().__init__()
        self.w, self.arg = (torch.randn(H, H, generator=g, device="cuda") / H ** 0.5).bfloat16(), arg

    def forward(self, *args):
        return args[self.arg] @ self.w


class Probe:   # records the events of the patched layer
    def __init__(self):
        self.events = []

    def event(self, layer, half, state):
        self.events.append((layer, half, state))


def make_layer(idx, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    layer = m.DeepseekV4DecoderLayer.__new__(m.DeepseekV4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer._use_b12x_mhc, layer.hidden_size, layer.rms_norm_eps = False, H, EPS
    layer.hc_mult, layer.hc_sinkhorn_iters, layer.hc_eps, layer.hc_post_alpha = HC, ITERS, EPS, ALPHA
    mix = (2 + HC) * HC
    for half in ("attn", "ffn"):
        setattr(layer, f"hc_{half}_fn", 0.01 * torch.randn(mix, HC * H, generator=g, device="cuda"))
        setattr(layer, f"hc_{half}_scale", 0.5 + torch.rand(3, generator=g, device="cuda"))
        setattr(layer, f"hc_{half}_base", 0.1 * torch.randn(mix, generator=g, device="cuda"))
    layer.attn_norm, layer.ffn_norm, layer.attn, layer.ffn = Norm(g), Norm(g), Map(g, 1), Map(g, 0)
    layer.layer_skip, layer.layer_probe, layer.layer_idx = None, None, idx
    return layer


def pre(layer, half, x, residual, post, comb):
    """The fused post-pre kernel, by hand: merge x into the residual, then the pre of ``half`` of ``layer``."""
    norm = getattr(layer, f"{half}_norm")
    return m.mhc_fused_post_pre_tilelang(
        x, residual, post, comb, getattr(layer, f"hc_{half}_fn"), getattr(layer, f"hc_{half}_scale"),
        getattr(layer, f"hc_{half}_base"), EPS, EPS, EPS, ALPHA, ITERS, n_splits=1, tile_n=1,
        norm_weight=norm.weight.data, norm_eps=EPS)


@pytest.fixture(scope="module")
def inputs():
    """The outputs of a first layer (no skip): the inputs of the layer under test."""
    g = torch.Generator(device="cuda").manual_seed(7)
    first = make_layer(2, 1)
    stream = torch.randn(T, HC, H, generator=g, device="cuda").bfloat16()
    return first(stream, torch.arange(T, device="cuda"), None)


def run(layer, inputs, skip, probe=None):
    layer.layer_skip, layer.layer_probe = skip, probe
    x, residual, post, comb = inputs
    return layer(x, torch.arange(T, device="cuda"), None, post, comb, residual)


def same(a, b):
    return all(torch.equal(p, q) for p, q in zip(a, b, strict=True))


def test_no_skip_equals_the_kernel_sequence(inputs):
    layer, probe = make_layer(3, 2), Probe()
    out = run(layer, inputs, None, probe)
    r1, p1, c1, x1 = pre(layer, "attn", *inputs)
    r2, p2, c2, x2 = pre(layer, "ffn", layer.attn(None, x1, None), r1, p1, c1)
    assert same(out, (layer.ffn(x2, None), r2, p2, c2))
    assert [(e[0], e[1]) for e in probe.events] == [(3, "A"), (3, "F")]
    assert torch.equal(probe.events[0][2], r1) and torch.equal(probe.events[1][2], r2)


def test_l_skip_returns_the_inputs(inputs):
    out = run(make_layer(3, 2), inputs, "L", Probe())
    assert all(o is i for o, i in zip(out, inputs))


def test_a_skip_runs_the_ffn_pre_on_the_inputs(inputs):
    layer, probe = make_layer(3, 2), Probe()
    out = run(layer, inputs, "A", probe)
    r, p, c, x = pre(layer, "ffn", *inputs)
    assert same(out, (layer.ffn(x, None), r, p, c))
    assert [(e[0], e[1]) for e in probe.events] == [(3, "F")] and torch.equal(probe.events[0][2], r)


def test_f_skip_returns_the_attention_output_unmerged(inputs):
    layer, probe = make_layer(3, 2), Probe()
    out = run(layer, inputs, "F", probe)
    r, p, c, x = pre(layer, "attn", *inputs)
    assert same(out, (layer.attn(None, x, None), r, p, c))
    assert [(e[0], e[1]) for e in probe.events] == [(3, "A")]
    nxt = make_layer(4, 3)                     # the next layer merges the attention output with the F mix values
    r2, *_ = pre(nxt, "attn", *out)
    assert torch.allclose(r2.float(), m.mhc_post_tilelang(*out).float())


def test_probe_sums_of_a_stack_with_a_skip(inputs):
    layers = {i: make_layer(i, 10 + i) for i in range(3, 7)}
    c = ls.Controller(layers, 10, protect_first=3, protect_last=3, eager=True, compile_cache=False,
                      env={"LAYER_PROBE": "1", "LAYER_PROBE_DIR": "/tmp", "LAYER_SKIP": "L4,F5"})
    pos = torch.arange(T, device="cuda")
    c.step_begin(pos)
    x, residual, post, comb = inputs
    for i in range(3, 7):
        x, residual, post, comb = layers[i](x, pos, None, post, comb, residual)
    c.step_end(m.mhc_post_tilelang(x, residual, post, comb))
    t = c.probe.tokens.cpu()
    done = {(3, "A"), (3, "F"), (3, "L"), (5, "A"), (6, "A"), (6, "F"), (6, "L")}
    for i in range(10):
        for k, kind in enumerate(ls.KINDS):
            assert t[i, k].item() == (T if (i, kind) in done else 0), (i, kind)
    cos = (c.probe.cos.cpu() / t.clamp_min(1))
    assert ((cos[t > 0] > -1) & (cos[t > 0] <= 1 + 1e-6)).all()
