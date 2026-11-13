"""GPU tests of the patched decoder layer (plans/PLAN_FEWER_LAYERS.md §6.5): stub layers with the real mHC kernels of
both paths (b12x, as on the live server, and tilelang) and random linear maps in place of the attention and the FFN.
They need the server image and the patched model file: run ./test_layer_skip.sh after image-patch/build_skip.sh
(about 1 GB of GPU memory).
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


@pytest.fixture(autouse=True, scope="module")
def scratch():
    """The b12x mHC scratch without vLLM's workspace manager: fresh buffers for each plan."""
    old = m._get_b12x_plan_scratch

    def plain(plan):
        bufs = [torch.empty(tuple(s[0]), dtype=s[1], device="cuda") for s in plan.shapes_and_dtypes()]
        return bufs[0] if len(bufs) == 1 else tuple(bufs)
    m._get_b12x_plan_scratch = plain
    yield
    m._get_b12x_plan_scratch = old


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


def make_layer(idx, seed, b12x):
    g = torch.Generator(device="cuda").manual_seed(seed)
    layer = m.DeepseekV4DecoderLayer.__new__(m.DeepseekV4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer._use_b12x_mhc, layer.hidden_size, layer.rms_norm_eps = b12x, H, EPS
    layer.hc_mult, layer.hc_sinkhorn_iters, layer.hc_eps, layer.hc_post_alpha = HC, ITERS, EPS, ALPHA
    mix = (2 + HC) * HC
    for half in ("attn", "ffn"):
        fn = 0.01 * torch.randn(mix, HC * H, generator=g, device="cuda")
        setattr(layer, f"hc_{half}_fn", fn)
        setattr(layer, f"hc_{half}_fn_bf16", fn.bfloat16())
        setattr(layer, f"hc_{half}_scale", 0.5 + torch.rand(3, generator=g, device="cuda"))
        setattr(layer, f"hc_{half}_base", 0.1 * torch.randn(mix, generator=g, device="cuda"))
    layer.hc_attn_fn_broadcast = layer.hc_attn_fn.view(-1, HC, H).sum(dim=1)
    from sparkinfer.norm.mhc import DEFAULT_BLOCK_K
    layer._b12x_mhc_block_k = int(DEFAULT_BLOCK_K) if b12x else 0
    layer._b12x_mhc_split_k = HC * H // int(DEFAULT_BLOCK_K) if b12x else 0
    layer.attn_norm, layer.ffn_norm, layer.attn, layer.ffn = Norm(g), Norm(g), Map(g, 1), Map(g, 0)
    layer.layer_skip, layer.layer_probe, layer.layer_idx = None, None, idx
    return layer


def pre(layer, half, x, residual, post, comb):
    """The post-pre of the layer's own path, by hand: merge x into the residual, then the pre of ``half``."""
    return layer.hc_post_pre(
        x, residual, post, comb, getattr(layer, f"hc_{half}_fn"), getattr(layer, f"hc_{half}_scale"),
        getattr(layer, f"hc_{half}_base"), norm_weight=getattr(layer, f"{half}_norm").weight.data, norm_eps=EPS,
        hc_fn_bf16=layer.hc_ffn_fn_bf16 if half == "ffn" else None)


def post(layer, x, residual, p, c):
    if layer._use_b12x_mhc:
        from sparkinfer.norm.mhc import run_post
        return run_post(x, residual, p, c)
    return m.mhc_post_tilelang(x, residual, p, c)


@pytest.fixture(scope="module", params=["b12x", "tilelang"])
def path(request):
    """The path, and the outputs of a first layer of that path (no skip): the inputs of the layer under test."""
    b12x = request.param == "b12x"
    g = torch.Generator(device="cuda").manual_seed(7)
    first = make_layer(2, 1, b12x)
    shape = (T, H) if b12x else (T, HC, H)   # b12x: the embedding and the broadcast pre; tilelang: the stream pre
    return b12x, first(torch.randn(shape, generator=g, device="cuda").bfloat16(), torch.arange(T, device="cuda"), None)


def run(layer, inputs, skip, probe=None):
    layer.layer_skip, layer.layer_probe = skip, probe
    x, residual, p, c = inputs
    return layer(x, torch.arange(T, device="cuda"), None, p, c, residual)


def same(a, b):
    return all(torch.equal(p, q) for p, q in zip(a, b, strict=True))


def test_no_skip_equals_the_kernel_sequence(path):
    b12x, inputs = path
    layer, probe = make_layer(3, 2, b12x), Probe()
    out = run(layer, inputs, None, probe)
    r1, p1, c1, x1 = pre(layer, "attn", *inputs)
    r2, p2, c2, x2 = pre(layer, "ffn", layer.attn(None, x1, None), r1, p1, c1)
    assert same(out, (layer.ffn(x2, None), r2, p2, c2))
    assert [(e[0], e[1]) for e in probe.events] == [(3, "A"), (3, "F")]
    assert torch.equal(probe.events[0][2], r1) and torch.equal(probe.events[1][2], r2)


def test_l_skip_returns_the_inputs(path):
    b12x, inputs = path
    out = run(make_layer(3, 2, b12x), inputs, "L", Probe())
    assert all(o is i for o, i in zip(out, inputs))


def test_a_skip_runs_the_ffn_pre_on_the_inputs(path):
    b12x, inputs = path
    layer, probe = make_layer(3, 2, b12x), Probe()
    out = run(layer, inputs, "A", probe)
    r, p, c, x = pre(layer, "ffn", *inputs)
    assert same(out, (layer.ffn(x, None), r, p, c))
    assert [(e[0], e[1]) for e in probe.events] == [(3, "F")] and torch.equal(probe.events[0][2], r)


def test_f_skip_returns_the_attention_output_unmerged(path):
    b12x, inputs = path
    layer, probe = make_layer(3, 2, b12x), Probe()
    out = run(layer, inputs, "F", probe)
    r, p, c, x = pre(layer, "attn", *inputs)
    assert same(out, (layer.attn(None, x, None), r, p, c))
    assert [(e[0], e[1]) for e in probe.events] == [(3, "A")]
    nxt = make_layer(4, 3, b12x)               # the next layer merges the attention output with the F mix values
    r2, *_ = pre(nxt, "attn", *out)
    assert torch.allclose(r2.float(), post(nxt, *out).float())


def test_probe_sums_of_a_stack_with_a_skip(path):
    b12x, inputs = path
    layers = {i: make_layer(i, 10 + i, b12x) for i in range(3, 7)}
    c = ls.Controller(layers, 10, protect_first=3, protect_last=3, eager=True, compile_cache=False,
                      env={"LAYER_PROBE": "1", "LAYER_PROBE_DIR": "/tmp", "LAYER_SKIP": "L4,F5"})
    pos = torch.arange(T, device="cuda")
    c.step_begin(pos)
    x, residual, p, cm = inputs
    for i in range(3, 7):
        x, residual, p, cm = layers[i](x, pos, None, p, cm, residual)
    c.step_end(post(layers[6], x, residual, p, cm))
    t = c.probe.tokens.cpu()
    done = {(3, "A"), (3, "F"), (3, "L"), (5, "A"), (6, "A"), (6, "F"), (6, "L")}
    for i in range(10):
        for k, kind in enumerate(ls.KINDS):
            assert t[i, k].item() == (T if (i, kind) in done else 0), (i, kind)
    cos = c.probe.cos.cpu() / t.clamp_min(1)
    assert ((cos[t > 0] > -1) & (cos[t > 0] <= 1 + 1e-6)).all()


ORDER = """
import sys, torch
from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
torch.zeros(1, device="cuda")
if sys.argv[1] == "early":
    import flashinfer.comm
mhc_post_tilelang(*(torch.rand(s, device="cuda").to(d) for s, d in
                    [((8, 4096), torch.bfloat16), ((8, 4, 4096), torch.bfloat16), ((8, 4, 1), torch.float32),
                     ((8, 4, 4), torch.float32)]))
torch.cuda.synchronize()
import vllm.model_executor.warmup.kernel_warmup  # the import of the kernel warmup in an eager boot
"""


@pytest.mark.parametrize("order", ["late", "early"])
def test_flashinfer_needs_its_cuda_runtime_before_a_tilelang_kernel(order):
    """PLAN_FEWER_LAYERS.md §7.7: after a tilelang kernel, flashinfer.comm finds tilelang's libcudart stub; the patch
    imports it in DeepseekV4Model.__init__ of an eager boot, before the first kernel."""
    import subprocess
    import sys
    proc = subprocess.run([sys.executable, "-c", ORDER, order], capture_output=True, text=True, timeout=600)
    if order == "late":
        assert "libcudart_stub.so: undefined symbol: cudaDeviceReset" in proc.stderr
    else:
        assert proc.returncode == 0, proc.stderr[-2000:]
    assert "import flashinfer.comm  # noqa: F401" in open(m.__file__).read()
