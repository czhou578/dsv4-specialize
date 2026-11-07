"""The speed result of one test point (PLAN_MODEL_SPECIALIZE.md §4.4), from the files that
``stepf-speed.sh`` and ``topk_run.sh`` write to ``model_eval/results``:

- bot decode: the last line of ``speed-bot-{label}.log`` (rate per run, median, spread, acceptance);
- code decode: ``speed-code-{label}-{1,2,3}.json`` (rate per run = decoded tokens / decode time, then the
  median of the runs) and the acceptance from ``speed-code-{label}-counters-{before,after}.json``.

Each change against the baseline counts only if it is larger than the noise floor.
"""

import argparse
import json
import statistics
from pathlib import Path

RESULTS = Path.home() / "Projects/stock-research-model-eval/model_eval/results"
BASELINE = {"bot": {"rate": 33.64, "noise": 1.42, "accepted_per_draft": 2.25},
            "code": {"rate": 38.86, "noise": 1.65, "accepted_per_draft": 2.67}}


def code_rate(path):
    trials = [t for t in json.loads(path.read_text())["code_c1_decode"]["trials"] if t.get("decode_tok_s")]
    tokens = sum(t["completion_tokens"] - 1 for t in trials)
    return tokens / sum((t["completion_tokens"] - 1) / t["decode_tok_s"] for t in trials)


def acceptance(before, after):
    drafts = after["drafts"] - before["drafts"]
    return {"drafts": drafts, "accepted_per_draft": round((after["accepted_tokens"] - before["accepted_tokens"]) / drafts, 3),
            "per_pos": [round((after["per_pos"][p] - before["per_pos"].get(p, 0)) / drafts, 3)
                        for p in sorted(after["per_pos"], key=int)]}


def compare(test, rate):
    base = BASELINE[test]
    change = rate - base["rate"]
    return {"change": round(change, 2), "change_pct": round(100 * change / base["rate"], 1),
            "above_noise": abs(change) > base["noise"]}


def result(label, results=RESULTS):
    out = {"label": label}
    bot_log = results / f"speed-bot-{label}.log"
    if bot_log.exists():
        try:
            bot = json.loads(bot_log.read_text().strip().splitlines()[-1])
            out["bot"] = {"rate": bot["rate"], "run_rates": bot["run_rates"], "spread": bot["rate_spread"],
                          "acceptance": bot.get("acceptance", {}), **compare("bot", bot["rate"])}
        except (ValueError, KeyError, IndexError) as error:
            out["bot"] = {"error": f"no summary line: {error}"}
    runs = []
    for n in (1, 2, 3):
        path = results / f"speed-code-{label}-{n}.json"
        try:
            runs.append(code_rate(path))
        except (OSError, ValueError, KeyError, ZeroDivisionError):
            pass
    if runs:
        rate = statistics.median(runs)
        out["code"] = {"rate": round(rate, 2), "run_rates": [round(r, 2) for r in runs],
                       "spread": round(max(runs) - min(runs), 2), **compare("code", rate)}
        try:
            before, after = (json.loads((results / f"speed-code-{label}-counters-{w}.json").read_text())
                             for w in ("before", "after"))
            out["code"]["acceptance"] = acceptance(before, after)
        except (OSError, ValueError, KeyError, ZeroDivisionError) as error:
            out["code"]["acceptance"] = {"error": str(error)}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("labels", nargs="+", help="for example botmask-k160 botmask5-k160")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    for label in args.labels:
        print(json.dumps(result(label, args.results)))


if __name__ == "__main__":
    main()
