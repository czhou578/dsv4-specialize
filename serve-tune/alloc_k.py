"""Compute the K of each layer for a mask with a target mean K (PLAN_SERVE_TUNE.md §7.3).

The input is a rank file with the routing weight share of each expert per layer (``weight_share``,
``expert_rank_weight.json``, PLAN_BETTER_RANK.md §5). Each layer drops its experts in the order of
``mask_checkpoint.py --rank`` with the same file: the lowest of the rank order first, and never a
mandatory expert of the two tool categories. ``mask_checkpoint.py --keep-per-layer`` with the output
list thus drops exactly the experts that this tool counts.

- Layers 0-2 (hash routing) keep 160 experts, because the hash remap of a smaller K has no test.
- Layers 3-42 keep between 128 and 216 experts.
- The 43 K values sum to exactly 43 × the target.

Two rules set the number of experts that each layer drops:

- ``global``: drop the next expert with the smallest weight share, in any layer, until the sum is
  correct. A layer at 128 stops.
- ``even``: drop about the same weight share in each layer. Each layer drops its next experts while
  their cumulative share stays at or under a share s. The tool finds the largest s that does not
  drop too many experts. Then it drops the next experts with the smallest share until the sum is
  correct.

The tool writes the list of ``--rule`` to ``--out`` and prints one summary line per rule.
"""

import argparse
import heapq
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # mask_checkpoint.py is in the repo root
import mask_checkpoint as mc  # noqa: E402

RANK = Path.home() / "Projects/dsv4-specialize/results/expert_rank_weight.json"
HASH_K = 160
K_MIN = 128
RULES = ("global", "even")


def layer_costs(plan, rank):
    """``{layer: weight shares of the experts that can drop, in drop order}`` for the layers with a
    learned router. Each list stops at K_MIN."""
    sizes = plan["keep_maps"]["keep_by_layer"]
    out = {}
    for layer, ids in mc.drop_order(plan, mc.bot_order(rank)).items():
        if layer >= mc.HASH_LAYERS:
            share = {e["id"]: e["weight_share"] for e in rank["layers"][str(layer)]}
            out[layer] = [share[c] for c in ids[:len(sizes[str(layer)]) - K_MIN]]
    return out


def fill(costs, drops, total):
    """Add one drop at a time to the layer whose next expert has the smallest share, until the drops
    sum to ``total``. Change and return ``drops``."""
    heap = [(c[drops[layer]], layer) for layer, c in costs.items() if drops[layer] < len(c)]
    heapq.heapify(heap)
    for _ in range(total - sum(drops.values())):
        _, layer = heapq.heappop(heap)
        drops[layer] += 1
        if drops[layer] < len(costs[layer]):
            heapq.heappush(heap, (costs[layer][drops[layer]], layer))
    return drops


def even(costs, total):
    """``{layer: drops}`` of the ``even`` rule."""
    sums = {layer: np.cumsum(c) for layer, c in costs.items()}

    def drops(s):
        return {layer: int(np.searchsorted(c, s, side="right")) for layer, c in sums.items()}

    shares = np.unique(np.concatenate([[-1.0], *sums.values()]))  # s = -1 drops nothing
    lo, hi = 0, len(shares)  # shares[lo] does not drop too many experts; shares[hi] does
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if sum(drops(shares[mid]).values()) <= total else (lo, mid)
    return fill(costs, drops(shares[lo]), total)


def allocate(plan, rank, target, rule):
    """``(K list, costs, {layer: drops})`` of one rule; costs and drops cover the learned layers."""
    sizes = plan["keep_maps"]["keep_by_layer"]
    costs = layer_costs(plan, rank)
    learned_k = len(sizes) * target - mc.HASH_LAYERS * HASH_K
    total = sum(len(sizes[str(layer)]) for layer in costs) - learned_k
    room = sum(map(len, costs.values()))
    if not 0 <= total <= room:
        raise ValueError(f"mean K {target} is out of reach: {total} drops, room for 0-{room}")
    drops = fill(costs, dict.fromkeys(costs, 0), total) if rule == "global" else even(costs, total)
    klist = [HASH_K if layer < mc.HASH_LAYERS else len(sizes[str(layer)]) - drops[layer]
             for layer in range(len(sizes))]
    return klist, costs, drops


def summary(rule, klist, costs, drops):
    """One line: the K of the learned layers and the weight share that they drop."""
    k = klist[mc.HASH_LAYERS:]
    lost = [sum(costs[layer][:n]) for layer, n in drops.items()]
    return (f"{rule}: mean K {sum(klist) / len(klist):g}; layers {mc.HASH_LAYERS}-{len(klist) - 1}: "
            f"K {min(k)}-{max(k)}, mean {np.mean(k):.2f}; weight share dropped: sum {sum(lost):.4f}, "
            f"mean per layer {np.mean(lost):.4f} ({min(lost):.4f}-{max(lost):.4f})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", type=int, required=True, help="the mean K over all layers")
    parser.add_argument("--rule", choices=RULES, default="global", help="the rule of the list in --out")
    parser.add_argument("--rank", type=Path, default=RANK, help="a rank file with weight_share")
    parser.add_argument("--base", type=Path, default=mc.BASE, help="the folder with REAP_K216_PLAN.json")
    parser.add_argument("--out", type=Path, help="the JSON list of K values; default: print only")
    args = parser.parse_args(argv)
    plan = json.loads((args.base / "REAP_K216_PLAN.json").read_text())
    rank = json.loads(args.rank.read_text())
    lists = {}
    for rule in RULES:
        klist, costs, drops = allocate(plan, rank, args.target, rule)
        print(summary(rule, klist, costs, drops))
        lists[rule] = klist
    if args.out:
        args.out.write_text(json.dumps(lists[args.rule]))
        print(f"wrote {args.out} ({args.rule})")


if __name__ == "__main__":
    main()
