"""The tables and the decisions of plans/PLAN_FEWER_EXPERTS.md Phase C (§7) and Phase D (§8).

It reads the files that ``fewer_night.sh`` writes to ``results/``: ``fewer-cN.jsonl``, ``fewer-cN-compare.txt``
(against the stock baseline, ``baseline-k216.jsonl``) and ``fewer-c5.rule``.

    python fewer_report.py               # the Phase C table (Markdown)
    python fewer_report.py --decide      # the point for Phase D, or "none" (the table of §7)
    python fewer_report.py --c5-norm     # the better anchor of C3 and C4, for C5
"""

import argparse
import re
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
RULES = {"fewer-c0": ("none", "kept"), "fewer-c1": ("lowest:1", "kept"), "fewer-c2": ("lowest:1", "anchor:6"),
         "fewer-c3": ("lowest:2", "anchor:6"), "fewer-c4": ("lowest:2", "anchor:8")}
REFS = {"botmask-k160": "K 160, top-k 6 (the mask)", "botmask5-k160": "K 160, top-k 5 (the live rule)",
        "botmask4-k160": "K 160, top-k 4 (plain cut)"}
NOISE_NLL = 0.23      # points: the NLL change between two runs of one model
NOISE_DECISIONS = 3   # of 60


def rules(r=R):
    out = dict(RULES)
    c5 = r / "fewer-c5.rule"
    if c5.exists():
        out["fewer-c5"] = tuple(c5.read_text().split())
    return out


def point(label, r=R):
    """The replay result of one point against the stock baseline, or None if it did not run."""
    cmp, rows = r / f"{label}-compare.txt", r / f"{label}.jsonl"
    if not cmp.exists() or not rows.exists():
        return None
    vals, gates = {}, {}
    for line in cmp.read_text().splitlines():
        m = re.match(r"^(\w+)\s+([-\d.]+)\s+([-\d.]+)\s*$", line)
        if m:
            vals[m[1]] = (float(m[2]), float(m[3]))
        m = re.match(r"^(PASS|FAIL)\s+(\w+)", line)
        if m:
            gates[m[2]] = m[1] == "PASS"
    if "nll" not in vals:
        return None
    pct = {k: 100 * (vals[k][1] / vals[k][0] - 1) for k in ("nll", "reasoning_nll", "answer_nll") if k in vals}
    n = int(vals.get("targets", (60, 60))[1])
    dec = (round(vals["decision"][0] * n), round(vals["decision"][1] * n))
    tools = vals.get("tool_names", (0, 0))
    return {"label": label, "rows": sum(1 for _ in rows.open()), "nll_pct": pct["nll"],
            "reasoning_pct": pct.get("reasoning_nll"), "answer_pct": pct.get("answer_nll"), "decisions": dec,
            "tool_names": tools, "nll_gate": gates.get("nll", False), "tool_gate": gates.get("tool", False)}


def passes(p):
    return bool(p and p["nll_gate"] and p["tool_gate"] and p["rows"] >= 60
                and p["decisions"][1] >= p["decisions"][0] - NOISE_DECISIONS)


def decide(r=R):
    """§7: a passing top-4 or threshold point (lowest NLL), else C2 if it passes and beats C1 by more than the
    noise, else none."""
    pts = {lab: point(lab, r) for lab in ("fewer-c1", "fewer-c2", "fewer-c3", "fewer-c4", "fewer-c5")}
    fast = [p for lab, p in pts.items() if lab in ("fewer-c3", "fewer-c4", "fewer-c5") and passes(p)]
    if fast:
        return min(fast, key=lambda p: p["nll_pct"])["label"]
    c1, c2 = pts["fewer-c1"], pts["fewer-c2"]
    if passes(c2) and c1 and c2["nll_pct"] < c1["nll_pct"] - NOISE_NLL:
        return "fewer-c2"
    return "none"


def c5_norm(r=R):
    c3, c4 = point("fewer-c3", r), point("fewer-c4", r)
    if c3 and c4:
        return "anchor:8" if c4["nll_pct"] < c3["nll_pct"] else "anchor:6"
    return "anchor:6" if c3 else "anchor:8" if c4 else "anchor:6"


def table(r=R):
    lines = ["| Point | Rule | Summed NLL | Reasoning | Answer | Decisions (of 60) | Tool names | Gates |",
             "|---|---|---|---|---|---|---|---|"]
    rows = [(lab, f"`{d}` / `{n}`") for lab, (d, n) in rules(r).items()] + list(REFS.items())
    for lab, rule in rows:
        p = point(lab, r)
        if p is None:
            lines.append(f"| {lab} | {rule} | not run | | | | | |")
            continue
        lines.append(f"| {lab} | {rule} | {p['nll_pct']:+.2f}% | {p['reasoning_pct']:+.2f}% | {p['answer_pct']:+.2f}% "
                     f"| {p['decisions'][0]} → {p['decisions'][1]} | {p['tool_names'][1]:.1%} "
                     f"| {'pass' if passes(p) else 'fail'} |")
    return "\n".join(lines) + f"\n\nDecision for Phase D: `{decide(r)}`\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=R)
    parser.add_argument("--decide", action="store_true")
    parser.add_argument("--c5-norm", action="store_true")
    args = parser.parse_args(argv)
    print(decide(args.results) if args.decide else c5_norm(args.results) if args.c5_norm else table(args.results))


if __name__ == "__main__":
    main()
