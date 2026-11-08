"""The capture data of the better rank (plans/PLAN_BETTER_RANK.md §3): one ``sessions.jsonl`` under
``model_eval/data/rank/`` of the bot repo's eval worktree, from several exports, with the leak rule applied.

Sources, each exported with ``export_sessions.py --all-calibration`` into ``model_eval/data/rank/<name>/``:

- ``main``: the calibration split of ``model_eval/data`` (the first capture's 34 segments), copied as it is.
- a sandbox label (``~/.hermes-e2e/<label>/home``): only the ``daily-report`` and ``quant`` sessions. The other
  sessions ran the 17 held-out prompts, which must stay out of the rank.
- ``rank-*`` labels (the new prompts of §3.2): every session.
- ``live`` (``~/.hermes``): the ``cron`` sessions since 2026-11-02 with at least one target.

``tools.json`` comes from ``model_eval/data`` (a sandbox has no request dumps).
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

# The replay stays in the bot repo, and the rank data sits next to the replay data: import it from the eval worktree.
EVAL = Path.home() / "Projects/stock-research-model-eval"
sys.path.insert(0, str(EVAL))
from model_eval.replay import DATA  # noqa: E402

RANK = DATA / "rank"
SANDBOX_SKILLS = ("daily-report", "quant")
LIVE_SINCE = "2026-11-02"


def first_user(session):
    return next((m.get("content") for seg in session["segments"] for m in seg if m.get("role") == "user"), "")


def keep(session, name, held_out):
    """The rule of the module docstring; ``held_out``: ids and first messages of the held-out sessions."""
    if session["id"] in held_out["ids"] or first_user(session) in held_out["texts"]:
        return False
    if name == "main":
        return session["split"] == "calibration"
    if name == "live":
        return session["source"] == "cron" and session["id"].split("_")[-2] >= LIVE_SINCE.replace("-", "") \
            and session["targets"] > 0
    if name.startswith("rank-"):
        return session["targets"] > 0
    return session["skill"] in SANDBOX_SKILLS and session["targets"] > 0


def merge(names, data=DATA, out=RANK):
    main = [json.loads(l) for l in (data / "sessions.jsonl").open()]
    held_out = {"ids": {s["id"] for s in main if s["split"] == "held_out"},
                "texts": {first_user(s) for s in main if s["split"] == "held_out"}}
    rows, counts = [], Counter()
    for name in names:
        path = data / "sessions.jsonl" if name == "main" else out / name / "sessions.jsonl"
        for line in path.open():
            s = json.loads(line)
            if keep(s, name, held_out):
                s["split"], s["rank_source"] = "calibration", name
                rows.append(s)
                counts[(name, s["skill"])] += 1
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sessions.jsonl").open("w") as f:
        for s in rows:
            f.write(json.dumps(s) + "\n")
    shutil.copyfile(data / "tools.json", out / "tools.json")
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="+", help="main, live, or an exported label under model_eval/data/rank/")
    args = parser.parse_args(argv)
    counts = merge(args.names)
    for (name, skill), n in sorted(counts.items()):
        print(f"{name:<18}{skill:<15}{n:>4}")
    print(f"{sum(counts.values())} sessions -> {RANK / 'sessions.jsonl'}")


if __name__ == "__main__":
    main()
