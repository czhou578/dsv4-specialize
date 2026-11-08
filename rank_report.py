"""The report of PLAN_BETTER_RANK.md §5: what each prune step removes under each rank, how the ranks differ, and
which experts the floor per skill protects. Reads the rank files that ``capture.py --rank`` wrote."""

import argparse
import json
from pathlib import Path

import numpy as np

STEPS = ((216, 176), (176, 160), (160, 144), (144, 128))


def arrays(rank):
    """``(layers, experts)`` arrays of count and weight, indexed by expert id, from a rank file."""
    layers = rank["layers"]
    n_layers, n = len(layers), len(layers["0"])
    count, weight = np.zeros((n_layers, n)), np.zeros((n_layers, n))
    for l in range(n_layers):
        for e in layers[str(l)]:
            count[l, e["id"]], weight[l, e["id"]] = e["count"], e.get("weight", 0.0)
    return count, weight


def orders(rank):
    return {l: [e["id"] for e in items] for l, items in rank["layers"].items()}


def removed_share(order, score, lo, hi):
    """The share of ``score`` (summed over layers) that the experts at rank positions ``lo:hi`` hold."""
    part = sum(score[int(l)][ids[lo:hi]].sum() for l, ids in order.items())
    return 100 * part / score.sum()


def overlap(order_a, order_b, k):
    """Mean over layers of the overlap between the lowest ``216 - k`` experts of two orders."""
    n = len(next(iter(order_a.values())))
    return float(np.mean([len(set(a[k:]) & set(b[k:])) / (n - k) for a, b in
                          ((order_a[l], order_b[l]) for l in order_a)]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    ranks = {o: json.loads((args.results / f"expert_rank_{o}.json").read_text())
             for o in ("count", "weight", "skill_max", "skill_floor") if (args.results / f"expert_rank_{o}.json").exists()}
    count, weight = arrays(ranks["weight"])
    print(f"{ranks['weight']['segments']} segments, {ranks['weight']['prompt_tokens']} prompt tokens, "
          f"{ranks['weight']['weighted_segments']} with weights")
    print("\nRouting share removed by each step (count share | weight share), under each order:")
    print(f"{'step':<14}" + "".join(f"{o:>24}" for o in ranks))
    for hi, lo in STEPS:
        row = f"K {hi} -> {lo:<6}"
        for o, r in ranks.items():
            od = orders(r)
            row += f"{removed_share(od, count, lo, hi):>11.2f}% {removed_share(od, weight, lo, hi):>10.2f}%"
        print(row)
    print("\nOverlap of the dropped experts between orders (mean over layers):")
    names = list(ranks)
    for k in (160, 144):
        print(f"  K {k}: " + ", ".join(f"{a}/{b} {overlap(orders(ranks[a]), orders(ranks[b]), k):.2f}"
                                    for i, a in enumerate(names) for b in names[i + 1:]))
    for o in ("skill_floor", "skill_max"):
        if o in ranks:
            print(f"\nExperts that {o} keeps above the cut and weight does not (per layer):")
            for k in (160, 144):
                moved = [len(set(orders(ranks[o])[l][:k]) - set(orders(ranks['weight'])[l][:k])) for l in orders(ranks['weight'])]
                print(f"  K {k}: mean {np.mean(moved):.1f}, max {max(moved)} per layer")
    skills = ranks["weight"]["per_skill"]
    print("\nComputed tokens per skill: " + ", ".join(f"{s} {v['tokens']}" for s, v in sorted(skills.items())))


if __name__ == "__main__":
    main()
