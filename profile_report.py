"""The routing profile report of plans/PLAN_FEWER_EXPERTS.md §5 (Phase A).

Input: the JSONL of a ``capture.py --sums`` run on a ``ROUTING_PROFILE=1`` capture boot. Each row holds the
``prof_*`` sums of one request (``image-patch/routed_experts_capturer.py``). Output: a JSON file and a Markdown
report with, per layer group and per skill, the share of the 5th and 6th experts, and for each rule the mean
experts per token and the mean window union (distinct experts per 6 tokens) against ``lowest:1`` (top-k 5).

The choice for point C5: the threshold rule whose mean experts per token is in [4.3, 4.5] and closest to 4.4
(to 0.01; a tie goes to the smaller window union), with a window union below ``lowest:1``. No such rule: the gate of §5 fails and C5 is skipped.

    python profile_report.py results/capture-profile-k160.jsonl results/profile-k160
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RULES = ("none", "lowest:1", "lowest:2") + tuple(
    f"threshold:{t}:{n}" for t in ("0.05", "0.08", "0.10", "0.12", "0.15") for n in (3, 4))
GROUPS = {"0-2 (hash)": range(0, 3), "3-15": range(3, 16), "16-29": range(16, 30), "30-42": range(30, 43)}
KEYS = ("prof_pos", "prof_hist", "prof_valid", "prof_kept", "prof_union", "prof_windows")


def total(rows):
    return {k: sum(np.array(r[k], dtype=float) for r in rows) for k in KEYS}


def summary(t, layers=None):
    """The profile of a set of layers (default: all), as plain numbers."""
    sel = list(range(len(t["prof_valid"]))) if layers is None else list(layers)
    valid, windows = t["prof_valid"][sel].sum(), t["prof_windows"][sel].sum()
    hist = t["prof_hist"][sel].sum(axis=0) / max(valid, 1)          # [2, 10]: 5th and 6th share
    cum = np.cumsum(hist, axis=1)                                     # share of tokens below 2.5 %, 5 %, …
    out = {"tokens": int(valid / max(len(sel), 1)),
           "position_share": np.round(t["prof_pos"][sel].sum(axis=0) / max(valid, 1), 4).tolist(),
           "below": {f"{'5th' if j == 0 else '6th'}<{p}%": round(float(cum[j, p * 2 // 5 - 1]), 4)
                     for j in (0, 1) for p in (5, 10, 15)},
           "rules": {}}
    union5 = t["prof_union"][sel, 1].sum() / max(windows, 1)
    for r, name in enumerate(RULES):
        union = t["prof_union"][sel, r].sum() / max(windows, 1)
        out["rules"][name] = {"experts_per_token": round(float(t["prof_kept"][sel, r].sum() / max(valid, 1)), 3),
                              "window_union": round(float(union), 2),
                              "union_vs_top5": round(float(union / max(union5, 1e-9) - 1), 4)}
    return out


def choose_c5(rules):
    ok = [(round(abs(v["experts_per_token"] - 4.4), 2), v["window_union"], name) for name, v in rules.items()
          if name.startswith("threshold") and 4.3 <= v["experts_per_token"] <= 4.5 and v["union_vs_top5"] < 0]
    return min(ok)[2] if ok else None


def report(rows):
    rows = [r for r in rows if "error" not in r and "prof_kept" in r]
    if not rows:
        raise ValueError("no rows with a routing profile: boot the capture with ROUTING_PROFILE=1")
    t = total(rows)
    result = {"segments": len(rows), "all": summary(t),
              "groups": {g: summary(t, ls) for g, ls in GROUPS.items()},
              "per_layer": {l: summary(t, [l]) for l in range(len(t["prof_valid"]))}}
    by_skill = defaultdict(list)
    for r in rows:
        by_skill[r["skill"]].append(r)
    result["skills"] = {s: summary(total(rs)) for s, rs in sorted(by_skill.items())}
    result["c5"] = choose_c5(result["all"]["rules"])
    return result


def markdown(res):
    a = res["all"]
    lines = [f"# Routing profile (Phase A): {res['segments']} segments, {a['tokens']} computed tokens per layer", "",
             "Share of each rank position (weights inside the top 6, all layers): "
             + ", ".join(f"{i + 1}: {x:.1%}" for i, x in enumerate(a["position_share"])), "",
             "| Layers | 5th share | 6th share | 5th < 10 % | 6th < 5 % | 6th < 10 % |", "|---|---|---|---|---|---|"]
    for g, s in [("all", a)] + list(res["groups"].items()):
        lines.append(f"| {g} | {s['position_share'][4]:.1%} | {s['position_share'][5]:.1%} | {s['below']['5th<10%']:.1%} "
                     f"| {s['below']['6th<5%']:.1%} | {s['below']['6th<10%']:.1%} |")
    lines += ["", "| Rule | Experts per token | Window union (of 6 tokens) | Union against top-k 5 |", "|---|---|---|---|"]
    for name, v in a["rules"].items():
        lines.append(f"| `{name}` | {v['experts_per_token']:.2f} | {v['window_union']:.1f} | {v['union_vs_top5']:+.1%} |")
    lines += ["", "| Skill | Tokens per layer | 5th share | 6th share | 6th < 10 % |", "|---|---|---|---|---|"]
    for s, v in res["skills"].items():
        lines.append(f"| {s} | {v['tokens']} | {v['position_share'][4]:.1%} | {v['position_share'][5]:.1%} "
                     f"| {v['below']['6th<10%']:.1%} |")
    lines += ["", f"C5 threshold (mean 4.3 to 4.5 experts, union below top-k 5): `{res['c5'] or 'none: gate fails'}`"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("rows", type=Path, help="capture JSONL with prof_* fields")
    parser.add_argument("out", type=Path, help="output stem: writes OUT.json and OUT.md")
    args = parser.parse_args(argv)
    res = report([json.loads(line) for line in args.rows.open()])
    args.out.with_suffix(".json").write_text(json.dumps(res))
    args.out.with_suffix(".md").write_text(markdown(res))
    print(f"{res['segments']} segments; C5: {res['c5']}")


if __name__ == "__main__":
    main()
