"""Layer skip and probe of DeepSeek V4 (dsv4-specialize plans/PLAN_FEWER_LAYERS.md §6.1 and §6.2).

``compose-skip.yml`` mounts this file next to the patched model file (``layer-skip.diff``), which makes one
``Controller`` for the decoder layers of the target. A block is a whole layer (``L17``), or its attention (``A17``) or
FFN half (``F17``). The controller sets ``layer_skip`` (None, "A", "F" or "L") on each layer, and the patched forward
reads it. ``LAYER_SKIP`` is a fixed set (``L17,F21``). Eager boots only: ``LAYER_SKIP_FILE`` is a JSON file, read at
the start of a step when its mtime changes (``{"skip": "L17", "count_from": 0, "sums": true, "gram": ["L17"],
"gram_flush": 0}``), and ``LAYER_PROBE=1`` writes the probe sums to ``layer_probe.npz`` after each step. ``LAYER_HEAL``
(§9) is not in this patch yet. Only torch and numpy are needed, so ``test_layer_skip.py`` runs without vLLM.
"""

import json
import logging
import os
import re

import numpy as np
import torch

logger = logging.getLogger("vllm.models.deepseek_v4.layer_skip")

KINDS = ("A", "F", "L")  # the column order of the probe sums
_BLOCK = re.compile(r"^([AFL])(\d+)$")
MAX_GRAM = 3


def parse(spec, num_layers=43, protect_first=3, protect_last=3):
    """The skip set of ``spec`` (``"L17,F21"``) as ``{layer: kind}``. Refused: a block in the first ``protect_first``
    layers (hash routing) or the last ``protect_last`` layers (the draft reads them), a block twice, and an L block with
    an A or F block of its layer. An A and an F block of one layer become one L block."""
    skips = {}
    items = [s.strip() for s in str(spec or "").replace(" ", ",").split(",") if s.strip()]
    if len(set(items)) != len(items):
        twice = sorted({s for s in items if items.count(s) > 1})
        raise ValueError(f"layer skip: {','.join(twice)} occurs twice")
    for item in items:
        m = _BLOCK.match(item)
        if not m:
            raise ValueError(f"layer skip: {item!r} is not a block (A17, F17 or L17)")
        kind, layer = m.group(1), int(m.group(2))
        if layer < protect_first or layer >= num_layers - protect_last:
            raise ValueError(f"layer skip: {item}: the layers 0-{protect_first - 1} and "
                             f"{num_layers - protect_last}-{num_layers - 1} cannot skip")
        old = skips.get(layer)
        if old is not None and "L" in (old, kind):
            raise ValueError(f"layer skip: L{layer} together with another block of layer {layer}")
        skips[layer] = kind if old is None else "L"
    return skips


def canon(skips):
    """The canonical text of a skip set, by layer: ``"F12,L17"``; ``"none"`` for no skip."""
    return ",".join(f"{kind}{layer}" for layer, kind in sorted(skips.items())) or "none"


def _check_gram(names):
    names = list(names or [])
    if len(names) > MAX_GRAM or len(set(names)) != len(names) or not all(_BLOCK.match(n) for n in names):
        raise ValueError(f"layer skip: gram needs at most {MAX_GRAM} distinct blocks, got {names}")
    return names


def _savez(path, **arrays):
    """Write the arrays atomically: a temporary ``.npz`` next to the file, then a rename."""
    tmp = path + ".tmp.npz"  # np.savez keeps a name that ends with .npz
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


class Probe:
    """The probe sums of §6.2 over the counted tokens of a step. A state is the merged residual of the mHC streams,
    ``(tokens, hc_mult, hidden)``: ``event(layer, "A" or "F", state)`` comes before the attention or the FFN of a layer,
    ``event(None, None, state)`` after the last layer. Each event ends the blocks that the event before it started."""

    def __init__(self, num_layers, device):
        self.cos, self.rel, self.tokens = (torch.zeros((num_layers, len(KINDS)), dtype=torch.float64, device=device)
                                           for _ in range(3))
        self.steps, self.sums, self.weight = 0, True, None
        self.gram = {}  # block -> [XᵀX, XᵀΔ, rows]
        self.open, self.l_start = [], {}  # the open blocks (kind, layer, start state); layer -> state before its attention

    def set_gram(self, names):
        """Start new Gram sums at these blocks (the old sums go)."""
        self.gram = {n: None for n in _check_gram(names)}

    def begin(self, positions, count_from=0):
        """The start of a step: a token counts when its position is at least ``count_from``."""
        self.open, self.l_start = [], {}
        self.weight = (positions.reshape(-1) >= count_from).to(torch.float32)

    def event(self, layer, half, state):
        if not self.sums and not self.gram:
            return
        for kind, start_layer, start in self.open:
            self._add(kind, start_layer, start, state)
            if kind == "F" and start_layer in self.l_start:
                self._add("L", start_layer, self.l_start.pop(start_layer), state)
        self.open = []
        if half == "A":
            self.open.append(("A", layer, state))
            self.l_start[layer] = state
        elif half == "F":
            self.open.append(("F", layer, state))

    def _add(self, kind, layer, before, after):
        w = self.weight
        if w is None or w.shape[0] != before.shape[0]:
            w = torch.ones(before.shape[0], dtype=torch.float32, device=before.device)
        name = f"{kind}{layer}"
        if self.sums:
            a = before.reshape(before.shape[0], -1).float()
            b = after.reshape(after.shape[0], -1).float()
            na, nb = a.square().sum(1), b.square().sum(1)
            cos = (a * b).sum(1) / (na * nb).sqrt().clamp_min(1e-30)
            rel = (b - a).square().sum(1) / na.clamp_min(1e-30)
            c = KINDS.index(kind)
            self.cos[layer, c] += (cos * w).sum().double()
            self.rel[layer, c] += (rel * w).sum().double()
            self.tokens[layer, c] += w.sum().double()
        if name in self.gram:
            self._add_gram(name, before, after, w)

    def _add_gram(self, name, before, after, w):
        """XᵀX and XᵀΔ, with one row for each stream of each counted token, in FP32 with full-precision matmuls."""
        hidden = before.shape[-1]
        x = before.reshape(-1, hidden).float()
        d = after.reshape(-1, hidden).float() - x
        xw = x * w.repeat_interleave(x.shape[0] // w.shape[0])[:, None]
        g = self.gram[name]
        if g is None:
            z = torch.zeros((hidden, hidden), dtype=torch.float32, device=x.device)
            g = self.gram[name] = [z, z.clone(), 0.0]
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            g[0].addmm_(xw.T, x)
            g[1].addmm_(xw.T, d)
        finally:
            torch.set_float32_matmul_precision(precision)
        g[2] += float(w.sum()) * (x.shape[0] // w.shape[0])

    def arrays(self):
        return {"cos": self.cos.cpu().numpy(), "rel": self.rel.cpu().numpy(), "tokens": self.tokens.cpu().numpy(),
                "steps": np.array(self.steps)}

    def gram_arrays(self):
        return {f"{k}_{name}": v for name, g in self.gram.items() if g is not None
                for k, v in (("xtx", g[0].cpu().numpy()), ("xtd", g[1].cpu().numpy()), ("rows", np.array(g[2])))}


def _bytes(module):
    return sum(t.numel() * t.element_size() for t in [*module.parameters(), *module.buffers()])


class Controller:
    """The skip and the probe of one target model: ``layers`` maps the layer number to the decoder layer."""

    def __init__(self, layers, num_layers, protect_first=3, protect_last=3, eager=False, compile_cache=True,
                 env=None):
        env = os.environ if env is None else env
        self.layers, self.num_layers = dict(layers), num_layers
        self.protect = (protect_first, protect_last)
        self.spec = env.get("LAYER_SKIP", "")
        self.file = env.get("LAYER_SKIP_FILE", "")
        self.probe_on = env.get("LAYER_PROBE", "0") == "1"
        self.out_dir = env.get("LAYER_PROBE_DIR") or (os.path.dirname(self.file) if self.file else "")
        if env.get("LAYER_HEAL"):
            raise RuntimeError("layer skip: LAYER_HEAL needs the heal patch of PLAN_FEWER_LAYERS.md §9")
        if (self.file or self.probe_on) and not eager:
            raise RuntimeError("layer skip: LAYER_SKIP_FILE and LAYER_PROBE need --enforce-eager")
        if self.probe_on and not self.out_dir:
            raise RuntimeError("layer skip: LAYER_PROBE needs LAYER_SKIP_FILE or LAYER_PROBE_DIR")
        if not eager and compile_cache:
            # The compile cache key has no environment variable in it: a boot could load the graph of another skip set.
            raise RuntimeError("layer skip: set VLLM_DISABLE_COMPILE_CACHE=1 (PLAN_FEWER_LAYERS.md §6.1)")
        self.runtime = bool(self.file or self.probe_on)
        self.skips = parse(self.spec, num_layers, *self.protect)
        self.probe, self.mtime, self.loads, self.error, self.count_from, self.gram_flush = None, None, 0, "", 0, None
        for i, layer in self.layers.items():
            layer.layer_idx = i
        self.apply(self.skips)

    def apply(self, skips):
        for i, layer in self.layers.items():
            layer.layer_skip = skips.get(i)
            layer.layer_probe = self.probe

    def boot_line(self):
        """The one boot line of §6.1: the skip set, the GiB of the skipped weights, the heal file and the probe."""
        parts = {"A": "attn", "F": "ffn"}
        size = sum(_bytes(self.layers[i]) if kind == "L" else _bytes(getattr(self.layers[i], parts[kind]))
                   for i, kind in self.skips.items() if i in self.layers)
        return (f"layer skip: set {canon(self.skips)} ({'run-time file ' + self.file if self.file else 'fixed'}); "
                f"skipped weights {size / 2**30:.2f} GiB; heal none; probe {'on' if self.probe_on else 'off'}")

    def step_begin(self, positions):
        """The start of a model step (eager only): read the skip file if it changed, and start the probe sums."""
        if self.probe_on and self.probe is None:
            self.probe = Probe(self.num_layers, positions.device)
            self.apply(self.skips)
        if self.file:
            self._reload()
        if self.probe is not None:
            self.probe.begin(positions, self.count_from)

    def step_end(self, state):
        """The end of a model step: the state after the last layer ends the last blocks; write the sums."""
        if self.probe is None:
            return
        self.probe.event(None, None, state)
        self.probe.steps += 1
        _savez(os.path.join(self.out_dir, "layer_probe.npz"), **self.probe.arrays(),
               skip=np.array(canon(self.skips)), loads=np.array(self.loads), error=np.array(self.error),
               gram=np.array(",".join(self.probe.gram)), count_from=np.array(self.count_from))

    def _reload(self):
        try:
            mtime = os.stat(self.file).st_mtime_ns
        except FileNotFoundError:
            return
        if mtime == self.mtime:
            return
        self.mtime = mtime
        try:
            with open(self.file) as f:
                cfg = json.load(f)
            skips = parse(cfg.get("skip", ""), self.num_layers, *self.protect)
            gram = _check_gram(cfg.get("gram"))
            count_from = int(cfg.get("count_from", 0))
        except (OSError, ValueError, TypeError, AttributeError) as exc:  # keep the old set; the client sees the error
            self.error = f"{type(exc).__name__}: {exc}"[:300]
            logger.error("layer skip: the skip file is refused, the set stays %s: %s", canon(self.skips), self.error)
            return
        self.error = ""
        self.loads += 1
        self.skips, self.count_from = skips, count_from
        if self.probe is not None:
            flush = cfg.get("gram_flush")
            if flush is not None and flush != self.gram_flush:
                _savez(os.path.join(self.out_dir, "layer_gram.npz"), flush=np.array(flush),
                       **self.probe.gram_arrays())
            self.gram_flush = flush
            self.probe.sums = bool(cfg.get("sums", True))
            if gram != list(self.probe.gram):
                self.probe.set_gram(gram)
        self.apply(skips)
        logger.warning("layer skip: run-time set %s (load %d); count from %d; gram %s", canon(skips), self.loads,
                       count_from, ",".join(gram) or "none")
