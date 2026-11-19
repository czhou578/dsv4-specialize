"""The self-check of the task set (plans/PLAN_AGENT_CAPACITY.md §4). A task enters the set only when it passes.

For all tasks: the fields of task.json, and the data paths exist.
C, D (code) normal: check_cmd fails on the files as given, and passes with solution/ on top; test_cmd (if any) passes
    with the solution; the protected files exist.
C, D trap: each path of trap_missing does not exist in the workspace, and test_cmd (if any) fails.
L, R, W normal: each key regex matches reference.answer, and each item of reference.evidence passes the H2 check.
L, R, W trap: no regex of trap_probe matches a file of the workspace (L, R) or a page of the page set (W).

  python3 selfcheck.py            # all tasks
  python3 selfcheck.py C1 C2      # some tasks
"""

import re
import shutil
import sys
import tempfile
from pathlib import Path

import agent_eval as ae
import checks
import sandbox
import tools

FIELDS = {"id": str, "family": str, "trap": bool, "prompt": str}


def check(tid: str, pages: tools.Pages) -> list[str]:
    task, d = ae.load_task(tid), ae.TASKS / tid
    errs = [f"field {k}: not a {t.__name__}" for k, t in FIELDS.items() if not isinstance(task.get(k), t)]
    if errs:
        return errs
    if task["id"] != tid or task["family"] not in "LCDRW":
        return [f"bad id or family: {task['id']} {task['family']}"]
    errs += [f"no data {p}" for p in task.get("data", []) if not (ae.DATA / p).exists()]
    if errs:
        return errs
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        ae.workspace(task, ws)
        if task["family"] in "CD":
            errs += code(task, d, ws)
        elif task["trap"]:
            texts = ([p["text"] for p in pages.pages.values()] if task["family"] == "W" else
                     [f.read_text(errors="replace") for f in ws.rglob("*") if f.is_file()])
            errs += [f"trap_probe {rx!r} matches a source" for rx in task.get("trap_probe", [])
                     if any(re.search(rx, t, re.I) for t in texts)]
            if not task.get("trap_probe"):
                errs.append("a trap needs trap_probe")
        else:
            ref = task.get("reference", {})
            errs += [f"key {k!r} does not match the reference answer" for k in task.get("key", [])
                     if not re.search(k, ref.get("answer", ""), re.I | re.S)]
            errs += [f"reference evidence: {e}" for item in ref.get("evidence", [])
                     if (e := checks.evidence_error(item, [ws], pages))]
            if not task.get("key") or not ref.get("evidence"):
                errs.append("a normal task needs key and reference.evidence")
    return errs


def code(task: dict, d: Path, ws: Path) -> list[str]:
    errs = [f"protected file missing: {p}" for p in task.get("protected", []) if not (ws / p).is_file()]
    if task["trap"]:
        errs += [f"trap_missing exists: {p}" for p in task.get("trap_missing", []) if (ws / p).exists()]
        if not task.get("trap_missing"):
            errs.append("a trap needs trap_missing")
        if task.get("test_cmd") and sandbox.run(str(ws), task["test_cmd"], 300)["exit"] == 0:
            errs.append("test_cmd passes on a trap")
        return errs
    if not (task.get("check_cmd") and (d / "hidden").is_dir() and (d / "solution").is_dir()):
        return errs + ["a code task needs check_cmd, hidden/ and solution/"]
    shutil.copytree(d / "hidden", ws / ".hidden")
    if sandbox.run(str(ws), task["check_cmd"], 300)["exit"] == 0:
        errs.append("check_cmd passes with no solution")
    shutil.copytree(d / "solution", ws, dirs_exist_ok=True)
    r = sandbox.run(str(ws), task["check_cmd"], 300)
    if r["exit"] != 0:
        errs.append(f"check_cmd fails with the solution: {r['output'][-600:]}")
    if task.get("test_cmd"):
        r = sandbox.run(str(ws), task["test_cmd"], 300)
        if r["exit"] != 0:
            errs.append(f"test_cmd fails with the solution: {r['output'][-600:]}")
    return errs


def main(ids: list[str]) -> int:
    pages, bad = tools.Pages(ae.DATA / "web/pages.jsonl"), 0
    for tid in ids or ae.all_tasks():
        errs = check(tid, pages)
        bad += bool(errs)
        print(f"{tid}: {'ok' if not errs else 'FAIL'}" + "".join(f"\n  {e}" for e in errs))
    print(f"{bad} of {len(ids or ae.all_tasks())} tasks fail")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
