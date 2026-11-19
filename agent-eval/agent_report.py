"""The report of the capacity measurement (plans/PLAN_AGENT_CAPACITY.md §7 and §11).

  python3 agent_report.py Q4                 # the table of a point, per model and per family
  python3 agent_report.py Q4 --ref Q1        # and the difference to a reference point, with the gates G1 to G4
  python3 agent_report.py --show results/agent-eval/Q4/qwen-L1-s0   # one transcript, for the manual read

The difference of two points uses only the tasks that both have. It averages the seeds of each task first, then it
takes a 95% interval from a bootstrap over the tasks (2000 samples, a fixed seed).
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import agent_eval as ae

RESULTS = ("pass", "hallucination", "wrong", "stopped", "server_error")
G1_POINTS, G2_POINTS, G4_MB = 3.0, 5.0, 3072
BURST, LOOP = 16, 5   # a run with 16 calls or more in one turn, or 5 repeated calls or more: a sign of a decode loop


def rows_of(name: str) -> list[dict]:
    f = ae.OUT / name / "rows.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else float("nan")


def summary(rows: list[dict]) -> dict:
    n = len(rows)
    count = {r: sum(x["result"] == r for x in rows) for r in RESULTS}
    secs = sorted(x["seconds"] for x in rows)
    return {"runs": n, **{r: pct(c, n) for r, c in count.items()},
            "events": {k: sum(x["events"][k] > 0 for x in rows) for k in ("H1", "H2", "H3", "H4")},
            "median_s": statistics.median(secs) if secs else None,
            "p90_s": secs[int(0.9 * (len(secs) - 1))] if secs else None,
            "prompt_first": statistics.median([x["prompt_tokens_first"] or 0 for x in rows]) if rows else None,
            "bursts": sum(x.get("max_calls_turn", 0) >= BURST for x in rows),
            "loops": sum(x.get("repeat_calls", 0) >= LOOP for x in rows)}


def per_task(rows: list[dict], result: str) -> dict[str, float]:
    by = defaultdict(list)
    for x in rows:
        by[x["task"]].append(x["result"] == result)
    return {t: sum(v) / len(v) for t, v in by.items()}


def diff(new: list[dict], ref: list[dict], result: str, samples=2000) -> tuple[float, float, float]:
    """The difference in points of the rate of `result`, new minus ref, on the common tasks, and its 95% interval."""
    a, b = per_task(new, result), per_task(ref, result)
    tasks = sorted(a.keys() & b.keys())
    if not tasks:
        return float("nan"), float("nan"), float("nan")
    d = [100 * (a[t] - b[t]) for t in tasks]
    rng = random.Random(0)
    boot = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(samples))
    return statistics.fmean(d), boot[int(0.025 * samples)], boot[int(0.975 * samples)]


def watch_stops(t0: float, t1: float) -> int:
    log = ae.S / "logs/pair-watch.log"
    if not log.exists():
        return 0
    import time
    n = 0
    for line in log.read_text().splitlines():
        if "WATCH:" in line:
            try:
                ts = time.mktime(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue
            n += t0 <= ts <= t1
    return n


def table(name: str, ref: str | None = None) -> str:
    rows, out = rows_of(name), []
    meta_f = ae.OUT / name / "point.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    out.append(f"# Point {name}: mix {meta.get('mix')}, ctx {meta.get('ctx')}, seeds {meta.get('seeds')}; lowest "
               f"available {meta.get('lowest_available_mb')} MB, lowest swap-free {meta.get('lowest_swap_free_mb')} MB")
    out.append("\n| Model | Family | Runs | Pass | Halluc. | Wrong | Stopped | Server err. | H1 | H2 | H3 | H4 | "
               "Median s | p90 s | Prompt tokens (first) | Bursts | Loops |\n|" + "---|" * 17)
    for model in sorted({x["model"] for x in rows}):
        mrows = [x for x in rows if x["model"] == model]
        for fam in ["all"] + sorted({x["family"] for x in mrows}):
            s = summary(mrows if fam == "all" else [x for x in mrows if x["family"] == fam])
            e = s["events"]
            out.append(f"| {model} | {fam} | {s['runs']} | {s['pass']:.1f}% | {s['hallucination']:.1f}% | "
                       f"{s['wrong']:.1f}% | {s['stopped']:.1f}% | {s['server_error']:.1f}% | {e['H1']} | {e['H2']} | "
                       f"{e['H3']} | {e['H4']} | {s['median_s']:.0f} | {s['p90_s']:.0f} | {s['prompt_first']:.0f} | {s['bursts']} | {s['loops']} |")
    if ref:
        rrows = rows_of(ref)
        out.append(f"\n## Against {ref}\n\n| Model | Halluc. diff (95% CI) | Pass diff (95% CI) | G1 | G2 | G3 | G4 |"
                   "\n|---|---|---|---|---|---|---|")
        for model in sorted({x["model"] for x in rows} & {x["model"] for x in rrows}):
            new, old = [x for x in rows if x["model"] == model], [x for x in rrows if x["model"] == model]
            h, p = diff(new, old, "hallucination"), diff(new, old, "pass")
            g3 = sum(x["result"] == "server_error" for x in new) == 0
            low = meta.get("lowest_available_mb")
            g4 = low is not None and low >= G4_MB and watch_stops(meta.get("start", 0), meta.get("end", 0)) == 0
            ok = lambda b: "pass" if b else "FAIL"  # noqa: E731
            out.append(f"| {model} | {h[0]:+.1f} ({h[1]:+.1f} to {h[2]:+.1f}) | {p[0]:+.1f} ({p[1]:+.1f} to "
                       f"{p[2]:+.1f}) | {ok(h[0] <= G1_POINTS)} | {ok(p[0] >= -G2_POINTS)} | {ok(g3)} | {ok(g4)} |")
    return "\n".join(out)


def show(run_dir: Path) -> str:
    tr = json.loads((run_dir / "transcript.json").read_text())
    out = [f"{tr['task']} {tr['model']} seed {tr['seed']} ctx {tr['ctx']}: {tr['result']} (stop {tr['stop']}, "
           f"{tr['seconds']} s)", f"events: {json.dumps(tr['events'])}"]
    for i, t in enumerate(tr["turns"]):
        out.append(f"\n--- turn {i}: {t['finish']}, {t['usage'].get('completion_tokens')} tokens, {t['seconds']} s")
        if t.get("reasoning"):
            out.append(f"[reasoning] {t['reasoning'][:600]}")
        if t.get("content"):
            out.append(f"[text] {t['content'][:600]}")
        for c in t["calls"]:
            out.append(f"[call] {c['name']} {c['arguments'][:300]}" + (f"  INVALID: {c['error']}" if c.get("error") else ""))
            out.append(f"[result] {c.get('result', '')[:400]}")
    out.append(f"\nreport: {json.dumps(tr['report'], indent=1)}")
    if tr.get("check"):
        out.append(f"check: exit {tr['check']['exit']}\n{tr['check']['output'][-800:]}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("point", nargs="?")
    ap.add_argument("--ref")
    ap.add_argument("--show", type=Path)
    a = ap.parse_args()
    print(show(a.show) if a.show else table(a.point, a.ref))


if __name__ == "__main__":
    main()
