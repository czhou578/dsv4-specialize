"""The drop files of the cold-expert points (plans/PLAN_COLD_EXPERTS.md §4 and §6).

Each point drops the live drops (the drop lists of the live ``MASK.json``) plus a set of cold experts in layers 3-42:

- R0: nothing more. It is the live set as a mask.
- M1: the mandatory experts (``mask_checkpoint.mandatory``) with a weight share below 0.1%.
- D1: the coldest ordinary experts, while their summed weight share stays at or below 1% in each layer.
- MD: M1 and D1.
- S15, S20, ... (``--share 0.15 0.2``): MD plus every kept expert with a weight share below 0.15%, 0.2%, ... The
  second window (``plans/PLAN_COLD_CUT2.md``) uses them.

The weight share comes from the rank file of the live set (``capture.py --rank ... --order weight``). Layers 0-2 (hash
routing) keep the live drops only. The tool writes ``cold-{r0,m1,d1,md}.json`` (``{layer: current ids}``, the input of
``mask_checkpoint.py --drop-file``), plus ``cold-s15.json``, ... for ``--share``, and prints the table of §4 as a check.

``--report R0.jsonl POINT.jsonl`` prints the comparison of §5.2 for two replay files: the summed NLL, the decisions
and the tool names against R0, the NLL change per skill, and each target whose tool names change.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import mask_checkpoint as mc

S = Path(__file__).resolve().parent
RANK = S / "results/expert_rank_k160.json"
LIVE = Path.home() / "Projects/DeepSeek-v4-Flash-One-DGX-Spark/data/tp1-k160t5/MASK.json"
EXPERT_GIB = 60.71 / 6880          # one routed expert of the live set (PRUNE.md)
M_SHARE, D_BUDGET = 0.001, 0.01
POINTS = ("r0", "m1", "d1", "md")


def cold_sets(plan, rank, live, m_share=M_SHARE, d_budget=D_BUDGET):
    """``{point: {layer: sorted current ids to drop}}`` for the four points."""
    kept_by_layer = plan["keep_maps"]["keep_by_layer"]
    out = {p: {} for p in POINTS}
    for layer in range(len(kept_by_layer)):
        ids = kept_by_layer[str(layer)]
        drop = set(live["layers"][str(layer)]["drop"])
        m1 = d1 = set()
        if layer >= mc.HASH_LAYERS:
            share = {e["id"]: e["weight_share"] for e in rank["layers"][str(layer)]}
            must = mc.mandatory(plan, layer)
            kept = [c for c in range(len(ids)) if c not in drop]
            m1 = {c for c in kept if ids[c] in must and share[c] < m_share}
            lost, d1 = 0.0, set()
            for c in sorted((c for c in kept if ids[c] not in must), key=lambda c: (share[c], c)):
                if lost + share[c] > d_budget:
                    break
                lost += share[c]
                d1.add(c)
        for point, extra in zip(POINTS, (set(), m1, d1, m1 | d1)):
            out[point][layer] = sorted(drop | extra)
    return out


def share_sets(plan, rank, live, md, shares):
    """``{s<share>: {layer: drops}}``: MD plus every kept expert of layers 3-42 below the share, for each share in %."""
    kept_by_layer = plan["keep_maps"]["keep_by_layer"]
    out = {}
    for pct in shares:
        name = f"s{round(pct * 100):02d}"   # 0.15 -> s15, 0.2 -> s20
        out[name] = {}
        for layer in range(len(kept_by_layer)):
            drop = set(live["layers"][str(layer)]["drop"]) | set(md[layer])
            if layer >= mc.HASH_LAYERS:
                share = {e["id"]: e["weight_share"] for e in rank["layers"][str(layer)]}
                drop |= {c for c in range(len(kept_by_layer[str(layer)])) if share[c] < pct / 100}
            out[name][layer] = sorted(drop)
    return out


def table(sets, plan, rank):
    """The rows of the table of §4: the extra drops, their mandatory count, GiB, the lost share and the K."""
    kept_by_layer, live = plan["keep_maps"]["keep_by_layer"], sets["r0"]
    rows = []
    for point in sets:
        extra, must, lost, ks = 0, 0, [], []
        for layer, drop in sets[point].items():
            add = set(drop) - set(live[layer])
            ids = kept_by_layer[str(layer)]
            extra += len(add)
            must += len({ids[c] for c in add} & mc.mandatory(plan, layer))
            if layer >= mc.HASH_LAYERS:
                share = {e["id"]: e["weight_share"] for e in rank["layers"][str(layer)]}
                lost.append(sum(share[c] for c in add))
                ks.append(len(ids) - len(drop))
        k_all = [len(kept_by_layer[str(layer)]) - len(d) for layer, d in sets[point].items()]
        rows.append({"point": point, "experts": extra, "mandatory": must, "gib": round(extra * EXPERT_GIB, 2),
                     "lost_mean_pct": round(100 * sum(lost) / len(lost), 2), "lost_max_pct": round(100 * max(lost), 2),
                     "k_min": min(ks), "k_max": max(ks), "mean_k": round(sum(k_all) / len(k_all))})
    return rows


def report(r0_path, point_path):
    """The comparison of §5.2 against R0, as lines of text."""
    def load(path):
        rows = map(json.loads, Path(path).read_text().splitlines())
        return {(r["session"], r["segment"], r["index"]): r for r in rows}
    r0, pt = load(r0_path), load(point_path)
    keys = sorted(r0.keys() & pt.keys())
    nll = lambda rows: sum(rows[k]["reasoning_nll"] + rows[k]["answer_nll"] for k in keys)  # noqa: E731
    count = lambda rows, f: sum(int(rows[k][f]) for k in keys)  # noqa: E731
    change = 100 * (nll(pt) / nll(r0) - 1)
    dec = count(pt, "decision_ok") - count(r0, "decision_ok")
    tools = count(pt, "tool_names_ok") - count(r0, "tool_names_ok")
    ok = change <= 0.5 and tools >= -1 and dec >= -1
    lines = [f"targets {len(keys)}; summed NLL {change:+.2f}% against R0; decisions {dec:+d}; tool names {tools:+d}; "
             f"no measurable cost (without the smoke test): {'yes' if ok else 'no'}"]
    skills = defaultdict(list)
    for k in keys:
        skills[r0[k]["skill"]].append(k)
    for skill, ks in sorted(skills.items()):
        a = sum(r0[k]["reasoning_nll"] + r0[k]["answer_nll"] for k in ks)
        b = sum(pt[k]["reasoning_nll"] + pt[k]["answer_nll"] for k in ks)
        lines.append(f"  {skill:<14} {len(ks):>3} targets  NLL {100 * (b / a - 1):+6.2f}%")
    for k in keys:
        if r0[k]["tool_names_ok"] != pt[k]["tool_names_ok"]:
            lines.append(f"  tool names {r0[k]['tool_names_ok']} -> {pt[k]['tool_names_ok']} of {r0[k]['tool_calls']}: "
                         f"{k[0]} segment {k[1]} index {k[2]} ({r0[k]['skill']})")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rank", type=Path, default=RANK, help="the rank file of the live set (weight order)")
    parser.add_argument("--live", type=Path, default=LIVE, help="the MASK.json of the live set")
    parser.add_argument("--base", type=Path, default=mc.BASE, help="the folder with REAP_K216_PLAN.json")
    parser.add_argument("--out-dir", type=Path, default=S / "results")
    parser.add_argument("--share", type=float, nargs="*", default=(), metavar="PCT",
                        help="more points: MD plus every expert below this weight share in %%, for example 0.15 0.2")
    parser.add_argument("--report", nargs=2, type=Path, metavar=("R0_JSONL", "POINT_JSONL"))
    args = parser.parse_args(argv)
    if args.report:
        print("\n".join(report(*args.report)))
        return
    plan = json.loads((args.base / "REAP_K216_PLAN.json").read_text())
    rank, live = json.loads(args.rank.read_text()), json.loads(args.live.read_text())
    sets = cold_sets(plan, rank, live)
    sets.update(share_sets(plan, rank, live, sets["md"], args.share))
    for point, drops in sets.items():
        (args.out_dir / f"cold-{point}.json").write_text(json.dumps({str(k): v for k, v in drops.items()}))
    print("point  experts  mandatory   GiB  lost/layer mean (max)  K in layers 3-42  mean K")
    for r in table(sets, plan, rank):
        print(f"{r['point']:<6}{r['experts']:>8}{r['mandatory']:>11}{r['gib']:>6.2f}  {r['lost_mean_pct']:>5.2f}% "
              f"({r['lost_max_pct']:.2f}%)        {r['k_min']}-{r['k_max']}          {r['mean_k']}")
    print(f"wrote {', '.join(f'cold-{p}.json' for p in sets)} to {args.out_dir}")


if __name__ == "__main__":
    main()
