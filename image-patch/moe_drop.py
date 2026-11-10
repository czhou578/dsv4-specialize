"""Fewer experts per token: the drop rule and the gain anchor of the router patch (plans/PLAN_FEWER_EXPERTS.md §6).

The patched router (``fused_topk_bias_router.py`` in this folder) boots at the trained top-k (6). After the stock
top-k, ``apply`` marks some slots as dropped: their expert id becomes -1 and their weight 0. The EXL3 route pack
skips an id outside 0 … E-1, so a dropped slot reads no expert weights. Then ``apply`` divides the kept weights by a
new denominator and multiplies them by the routed scaling factor (1.5).

The rule comes from three environment variables, read once at import. The router runs inside the CUDA graphs, so a
rule cannot change per request.

- ``MOE_DROP``: ``none``, ``lowest:N`` (drop the N weakest slots), or ``threshold:T:MIN`` (drop every slot whose
  share of the six weights is below T, but keep at least MIN slots, the strongest first).
  "Weakest" means the lowest selection score (score + bias) in layers 3-42, where the stock router picks by that
  score, and the lowest weight in the hash layers 0-2. So ``lowest:1`` in layers 3-42 is exactly a top-5 router.
- ``MOE_NORM``: ``kept`` (divide by the sum of the kept weights: the stock rule at a smaller top-k) or
  ``anchor:K2`` (divide by the mass of the top K2 experts, Chen and Yao 2026). With K2 = 6, the denominator is the
  sum of all six selected weights. With K2 > 6, layers 3-42 take the top K2 of the full score vector by selection
  score; the hash layers have only six candidates, so they keep K2 = 6.
- ``MOE_LAYERS``: ``all`` or a list of layer ranges such as ``3-15,30-42``. Other layers keep all six slots.

``MOE_TARGET_EXPERTS`` lists the routed-expert counts of the target model (default ``160,216``: the pruned folder
and the mask folders). The DSpark draft has 64 experts, so the rule never reaches it.

Every shape stays static and nothing reads a value back to the host, so ``apply`` is safe under CUDA graph capture.
Only torch is imported here, so the tests run without vLLM.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Rule:
    drop: str = "none"          # none | lowest | threshold
    n: int = 0                  # lowest:N
    t: float = 0.0              # threshold:T:MIN
    min_keep: int = 0
    norm: str = "kept"          # kept | anchor
    k2: int = 0                 # anchor:K2
    layers: frozenset | None = None   # None: all layers
    targets: frozenset = frozenset({160, 216})

    @property
    def active(self) -> bool:
        return self.drop != "none" or self.norm != "kept"

    def applies(self, layer: int) -> bool:
        return self.active and layer >= 0 and (self.layers is None or layer in self.layers)

    def label(self) -> str:
        drop = {"none": "none", "lowest": f"lowest:{self.n}", "threshold": f"threshold:{self.t:g}:{self.min_keep}"}
        norm = "kept" if self.norm == "kept" else f"anchor:{self.k2}"
        layers = "all" if self.layers is None else ",".join(str(x) for x in sorted(self.layers))
        return f"drop={drop[self.drop]} norm={norm} layers={layers}"


def parse_layers(text: str) -> frozenset | None:
    """``all`` -> None; ``3-15,30`` -> {3, …, 15, 30}."""
    text = text.strip()
    if text in ("", "all"):
        return None
    out = set()
    for part in text.split(","):
        lo, _, hi = part.strip().partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return frozenset(out)


def parse(env=None) -> Rule:
    env = os.environ if env is None else env
    drop, n, t, min_keep = "none", 0, 0.0, 0
    d = env.get("MOE_DROP", "none").strip().split(":")
    if d[0] == "lowest":
        drop, n = "lowest", int(d[1])
    elif d[0] == "threshold":
        drop, t, min_keep = "threshold", float(d[1]), int(d[2])
    elif d[0] != "none":
        raise ValueError(f"MOE_DROP={env.get('MOE_DROP')!r}: use none, lowest:N or threshold:T:MIN")
    nm = env.get("MOE_NORM", "kept").strip().split(":")
    if nm[0] == "kept":
        norm, k2 = "kept", 0
    elif nm[0] == "anchor":
        norm, k2 = "anchor", int(nm[1])
    else:
        raise ValueError(f"MOE_NORM={env.get('MOE_NORM')!r}: use kept or anchor:K2")
    targets = frozenset(int(x) for x in env.get("MOE_TARGET_EXPERTS", "160,216").split(",") if x.strip())
    return Rule(drop, n, t, min_keep, norm, k2, parse_layers(env.get("MOE_LAYERS", "all")), targets)


RULE = parse()
_next_layer = 0


def next_layer(num_experts: int, is_hash: bool) -> int:
    """The layer index of a new target router (in build order), or -1 for another router (the draft).

    The model builds its 43 decoder layers in order, so the n-th target router is layer n. The boot log gets one
    line per router: check that layers 0-2, and only those, have hash tables.
    """
    global _next_layer
    if num_experts not in RULE.targets:
        return -1
    layer = _next_layer
    _next_layer += 1
    if RULE.active:
        print(f"[moe_drop] layer {layer} experts={num_experts} hash={is_hash} applies={RULE.applies(layer)} "
              f"{RULE.label()}", flush=True)
    return layer


def drop_mask(rule: Rule, shares: torch.Tensor, select_key: torch.Tensor) -> torch.Tensor:
    """[M, K] bool: True for a dropped slot. ``select_key`` orders the slots for ``lowest`` (high = strong)."""
    m, k = shares.shape
    if rule.drop == "lowest":
        n = max(0, min(rule.n, k - 1))
        weakest = torch.argsort(select_key, dim=-1)[:, :n]
        return torch.zeros_like(shares, dtype=torch.bool).scatter(1, weakest, True)
    if rule.drop == "threshold":
        order = torch.argsort(shares, dim=-1, descending=True)
        rank = torch.empty_like(order).scatter_(1, order, torch.arange(k, device=shares.device).expand(m, k))
        return (shares < rule.t) & (rank >= rule.min_keep)
    return torch.zeros_like(shares, dtype=torch.bool)


def apply(rule: Rule, weights: torch.Tensor, ids: torch.Tensor, logits: torch.Tensor,
          bias: torch.Tensor | None, is_hash: bool, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The new ``(weights, ids)`` of one layer.

    ``weights`` [M, K] is the stock output: the selected unbiased scores divided by their sum, times ``scale``.
    ``ids`` [M, K] holds the selected experts; ``logits`` [M, E] the router logits; ``bias`` [E] the selection
    bias (None in the hash layers).
    """
    w = weights.float()
    shares = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    scores = None
    if is_hash or bias is None:
        select_key = shares
    else:
        scores = torch.sqrt(F.softplus(logits.float()))
        select_key = (scores + bias.float()).gather(1, ids.long())
    drop = drop_mask(rule, shares, select_key)
    keep = (~drop).to(w.dtype)
    if rule.norm == "kept":
        kept = shares * keep
        new = kept / kept.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    elif rule.k2 > w.shape[-1] and scores is not None:
        top = torch.topk(scores + bias.float(), k=rule.k2, dim=-1).indices
        mass = scores.gather(1, top).sum(dim=-1, keepdim=True).clamp(min=1e-20)
        new = scores.gather(1, ids.long()) * keep / mass
    else:  # anchor:K2 with K2 <= K, or a hash layer: the six selected weights are the anchor
        new = shares * keep
    out_w = torch.where(drop, torch.zeros_like(new), new * scale).to(weights.dtype)
    out_ids = torch.where(drop, torch.full_like(ids, -1), ids)
    return out_w, out_ids
