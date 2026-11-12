"""The report of plans/PLAN_MD_REVIEW.md §6: the judge rules, the weak prompts, the skills, the leaks and the run times.

It reads the judge rows of the eval repo (``model_eval/results/e2e/judge-*.jsonl``) and the run folders of
``~/.hermes-e2e``. Run it with the bot's venv after review_window.sh, or again at any time:

  python review_report.py --live live-1 live-2 --new md-1 md-2

Rule 1 (the judge test): the new model fails if its losses against the live runs reach the one-sided sign-test line
(p < 0.05, ties out). Rule 2 (the controls): a control of one model against itself must not reach the line in either
direction. The report gates nothing else: the leaks and the run times are readings for the review.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

EVAL = Path("/home/colin-spark/Projects/stock-research-model-eval")
JUDGE = EVAL / "model_eval/results/e2e"
RUNS = Path.home() / ".hermes-e2e"


def p_loss(losses, decided):
    """One-sided sign test: the chance of this many losses or more of ``decided`` pairs (as ``judge.p_loss``)."""
    return sum(math.comb(decided, k) for k in range(losses, decided + 1)) / 2 ** decided


def fail_line(decided, alpha=0.05):
    """The fewest losses of ``decided`` pairs that fail the one-sided sign test at ``alpha``."""
    return next((k for k in range(decided + 1) if p_loss(k, decided) < alpha), decided + 1)


def verdict(rows, two_sided=False):
    """``(counts, fail line, failed)`` of judge rows; ``two_sided`` also fails on the wins (a control)."""
    n = {v: sum(r["verdict"] == v for r in rows) for v in ("win", "loss", "tie", "error")}
    decided = n["win"] + n["loss"]
    line = fail_line(decided)
    return n, line, n["loss"] >= line or (two_sided and n["win"] >= line)


def weak(test, bases):
    """``[(new label, prompt id)]``: the prompts of a new run that lose against every base run."""
    out = []
    for new, rows in test.items():
        by_id = defaultdict(list)
        for r in rows:
            by_id[r["id"]].append(r["verdict"])
        out += [(new, i) for i, v in sorted(by_id.items()) if len(v) == len(bases) and set(v) == {"loss"}]
    return out


def by_skill(rows, skill_of):
    """``{skill: [wins, losses, ties]}`` of the judge rows; an error counts as a tie."""
    out = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        out[skill_of.get(r["id"], "?")][{"win": 0, "loss": 1}.get(r["verdict"], 2)] += 1
    return dict(sorted(out.items()))


def times(done, live, new):
    """``{skill: (mean live seconds, mean new seconds, change in percent)}`` over the prompts that exit 0 in every
    run; ``done`` is ``{label: {prompt id: done record}}``."""
    labels = [*live, *new]
    ids = set.intersection(*({i for i, r in done[lb].items() if r.get("exit") == 0} for lb in labels))
    out, sums = {}, defaultdict(lambda: [0.0, 0.0])
    for i in ids:
        skill = done[labels[0]][i]["skill"]
        for side, group in ((0, live), (1, new)):
            sums[skill][side] += sum(done[lb][i]["seconds"] for lb in group) / len(group)
    sums["all"] = [sum(v[0] for v in sums.values()), sum(v[1] for v in sums.values())]
    for skill, (a, b) in sorted(sums.items()):
        out[skill] = (round(a), round(b), round(100 * (b / a - 1), 1) if a else None)
    return out


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def runs(label):
    """``{prompt id: done record}`` of a run folder."""
    return {r["id"]: r for r in (json.loads(p.read_text()) for p in sorted((RUNS / label).glob("runs/*/done.json")))}


def line_of(name, rows, two_sided=False):
    """The report line and the result: True (fail), False (pass) or None (no judge rows: a missing file)."""
    if not rows:
        return f"{name}: NO JUDGE ROWS (the judge file is missing or empty)", None
    n, line, failed = verdict(rows, two_sided)
    return (f"{name}: {n['win']} W / {n['loss']} L / {n['tie']} T, {n['error']} errors; "
            f"{n['win'] + n['loss']} decided, the fail line {line}: {'FAIL' if failed else 'pass'}"), failed


def result(failed):
    """``FAIL``, ``PASS`` or ``NO DATA`` from a list of the results of ``line_of``."""
    return "FAIL" if any(failed) else "NO DATA" if None in failed else "PASS"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--live", nargs="+", required=True, help="the base runs on the live model")
    parser.add_argument("--new", nargs="+", required=True, help="the runs on the new model")
    args = parser.parse_args(argv)
    base = "+".join(args.live)
    test = {n: load(JUDGE / f"judge-{base}-vs-{n}.jsonl") for n in args.new}
    every = [r for rows in test.values() for r in rows]
    runs_failed = []
    for name, rows in test.items():
        text, failed = line_of(f"{name} against {base}", rows)
        runs_failed.append(None if failed is None else False)   # rule 1 is on all pairs; a missing file is no data
        print(text)
    text, failed = line_of("Rule 1, all test pairs", every)
    print(text)
    controls = [(g[i], g[j]) for g in (args.live, args.new) for i in range(len(g)) for j in range(i + 1, len(g))]
    c_failed = []
    for a, b in controls:
        text, f = line_of(f"Rule 2, control {a} against {b} (win = {b})", load(JUDGE / f"judge-{a}-vs-{b}.jsonl"),
                          two_sided=True)
        c_failed.append(f)
        print(text)
    print(f"Rule 1: {result([failed, *runs_failed])}; rule 2: {result(c_failed)} ({len(controls)} controls)")
    done = {lb: runs(lb) for lb in (*args.live, *args.new)}
    skill_of = {i: r["skill"] for d in done.values() for i, r in d.items()}
    print("Prompts that lose against every live run:",
          ", ".join(f"{n} {i} ({skill_of.get(i, '?')})" for n, i in weak(test, args.live)) or "none")
    print("Per skill, W / L / T:", "; ".join(f"{s} {w}/{lo}/{t}" for s, (w, lo, t) in by_skill(every, skill_of).items()))
    sys.path.insert(0, str(EVAL))
    from model_eval.e2e.judge import reports   # the bot's venv only
    for lb in done:
        reps = reports(lb)
        print(f"{lb}: {len(done[lb])} prompts, {sum(r.get('exit') == 0 for r in done[lb].values())} exit 0, "
              f"{len(reps)} reports, {sum('</s>' in rep for _, rep in reps.values())} with a literal </s>")
    print("Run time per skill (mean live s, mean new s, change %):",
          "; ".join(f"{s} {a}/{b}/{p:+}" for s, (a, b, p) in times(done, args.live, args.new).items() if p is not None))


if __name__ == "__main__":
    main()
