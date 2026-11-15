"""The agent loop of the capacity measurement (plans/PLAN_AGENT_CAPACITY.md §5).

One run: one task, one model, one seed, one context size. The loop calls /v1/chat/completions of the model server
directly, with the sampling defaults of the server, and stops at the report call, 25 turns, the time limit, three
turns with no tool call, or a full context. Then it grades the run (the hidden tests, the key) and computes the
events and the result (checks.py).

  python3 agent_eval.py run --task L1 --model qwen --keep
"""

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import checks
import sandbox
import tools

AE = Path(__file__).resolve().parent
S = Path(os.environ.get("AGENT_EVAL_S", "/home/colin-spark/Projects/dsv4-specialize"))
TASKS = Path(os.environ.get("AGENT_EVAL_TASKS", AE / "tasks"))
DATA = Path(os.environ.get("AGENT_EVAL_DATA", S / "results/agent-eval/data"))
OUT = Path(os.environ.get("AGENT_EVAL_OUT", S / "results/agent-eval"))
MEM = Path(os.environ.get("AGENT_EVAL_MEM", S / "results/pair-mem.log"))
MODELS = {   # url, served name, max_tokens per turn (Qwen: the cap of the gateway), time limit per run in s
    "qwen": ("http://127.0.0.1:8889", "second-model-fit", 8192, 1200),
    "deepseek": ("http://127.0.0.1:8888", "deepseek-v4-flash-0731-k160t5", 32000, 3600),
}
MAX_TURNS, MAX_NUDGES = 25, 3
SYSTEM = """You are a careful software and research agent. You work in the folder /work, which holds the files of \
the task. Use the tools to read files, search, run commands and write files. The run tool executes a bash command \
in /work with no network access.

Rules:
- Use only facts that you read with a tool in this task. Do not answer from memory.
- When the files or pages do not hold the answer, or a file that the task needs does not exist, do not guess: call \
report with status "not_found" and say what is missing.
- Do not change test files unless the task tells you to.
- Finish with exactly one call of the report tool. In evidence, give each source (a path relative to /work, or a \
URL) and an exact quote copied from it.
- Set tests_passed only from the output of a test run that you did in this task."""
NUDGE = "Continue the task with a tool call. When you are finished, call the report tool."


def load_task(task_id: str) -> dict:
    return json.loads((TASKS / task_id / "task.json").read_text())


def all_tasks(families: str = "") -> list[str]:
    ids = sorted(p.parent.name for p in TASKS.glob("*/task.json"))
    return [i for i in ids if not families or load_task(i)["family"] in families.split(",")]


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def workspace(task: dict, ws: Path) -> None:
    """A new workspace: the files of the task, and its data (paths under DATA) at the same relative paths."""
    shutil.rmtree(ws, ignore_errors=True)
    shutil.copytree(TASKS / task["id"] / "files", ws) if (TASKS / task["id"] / "files").is_dir() else ws.mkdir(parents=True)
    for rel in task.get("data", []):
        src, dst = DATA / rel, ws / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy(src, dst)


def padding(tokens: int) -> list[dict]:
    """Earlier turns of a long job: read_file calls on the frozen repo (DATA/repo) up to about `tokens` prompt tokens."""
    if tokens <= 0:
        return []
    msgs = [{"role": "user", "content": "First read the files of the repository in /work/repo, one by one."}]
    chars, n = 0, 0
    files = sorted(p for p in (DATA / "repo").rglob("*") if p.suffix in {".py", ".sh", ".md"} and p.is_file())
    for f in files * 4:
        rel = str(f.relative_to(DATA))
        text = tools.read_file(DATA, rel)
        n += 1
        msgs += [{"role": "assistant", "content": None, "tool_calls": [{"id": f"pad{n}", "type": "function",
                  "function": {"name": "read_file", "arguments": json.dumps({"path": rel})}}]},
                 {"role": "tool", "tool_call_id": f"pad{n}", "content": text}]
        chars += len(text) + 120
        if chars / 3.0 >= tokens:   # about 3 characters per token for code with line numbers
            break
    return msgs + [{"role": "assistant", "content": "I have read the files. What is the next task?"}]


def chat(model: str, messages: list, seed: int, timeout: int = 1800) -> dict:
    url, name, max_tokens, _ = MODELS[model]
    body = {"model": name, "messages": messages, "tools": tools.SCHEMAS, "max_tokens": max_tokens, "seed": seed}
    req = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_task(task: dict, model: str, seed: int, ctx: int, run_dir: Path, pages: tools.Pages, keep=False) -> dict:
    ws = run_dir / "ws"
    workspace(task, ws)
    prot = {p: sha(ws / p) for p in task.get("protected", [])}
    msgs = [{"role": "system", "content": SYSTEM}] + padding(ctx) + [{"role": "user", "content": task["prompt"]}]
    tr = {"task": task["id"], "model": model, "seed": seed, "ctx": ctx, "turns": [], "report": None, "error": None,
          "stop": "turns", "retries": 0}
    t0, nudges = time.monotonic(), 0
    for turn in range(MAX_TURNS):
        if time.monotonic() - t0 > MODELS[model][3]:
            tr["stop"] = "time"
            break
        t1 = time.monotonic()
        try:
            resp = chat(model, msgs, zlib.crc32(f"{task['id']}-{seed}-{turn}".encode()))
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            if e.code == 400 and "context" in body.lower():
                tr["stop"] = "context"
            else:
                tr["error"], tr["stop"] = f"HTTP {e.code}: {body}", "server_error"
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if tr["retries"] == 0:   # one retry for a lost connection
                tr["retries"] = 1
                time.sleep(15)
                continue
            tr["error"], tr["stop"] = f"{type(e).__name__}: {e}", "server_error"
            break
        choice = resp["choices"][0]
        m = choice["message"]
        reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
        rec = {"content": m.get("content"), "reasoning": reasoning, "finish": choice.get("finish_reason"),
               "usage": resp.get("usage", {}), "seconds": round(time.monotonic() - t1, 2), "calls": []}
        tr["turns"].append(rec)
        msgs.append({"role": "assistant", "content": m.get("content"), "reasoning": reasoning or None,
                     **({"tool_calls": m["tool_calls"]} if m.get("tool_calls") else {})})
        if not m.get("tool_calls"):
            nudges += 1
            if choice.get("finish_reason") == "length" or nudges >= MAX_NUDGES:
                tr["stop"] = "length" if choice.get("finish_reason") == "length" else "no_tool"
                break
            msgs.append({"role": "user", "content": NUDGE})
            continue
        for tc in m["tool_calls"]:
            fn = tc["function"]
            args, err = checks.validate_call(fn["name"], fn.get("arguments"))
            call = {"name": fn["name"], "arguments": fn.get("arguments"), "args": args, "error": err, "record": {}}
            if err:
                text = f"error: invalid call: {err}"
            elif fn["name"] == "report":
                text, tr["report"] = "Report received.", args
            else:
                text, call["record"] = tools.call(ws, pages, fn["name"], args)
            call["result"] = sandbox.clip(text, 2000)
            rec["calls"].append(call)
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
            if tr["report"] is not None:
                break
        if tr["report"] is not None:
            tr["stop"] = "report"
            break
    tr["seconds"] = round(time.monotonic() - t0, 1)
    tr["protected"] = {"before": prot, "after": {p: sha(ws / p) for p in prot}}
    if task.get("check_cmd") and tr["report"] and tr["report"]["status"] == "done":
        shutil.rmtree(ws / ".hidden", ignore_errors=True)
        shutil.copytree(TASKS / task["id"] / "hidden", ws / ".hidden")
        tr["check"] = sandbox.run(str(ws), task["check_cmd"], timeout=300)
    ev = checks.events(tr, task, [ws, TASKS / task["id"] / "files"], pages)
    tr["events"], tr["result"] = ev, checks.result(tr, task, ev)
    (run_dir / "transcript.json").write_text(json.dumps(tr, indent=1))
    if not keep:
        shutil.rmtree(ws, ignore_errors=True)
    usage = [t["usage"] for t in tr["turns"]]
    return {"task": task["id"], "family": task["family"], "trap": task["trap"], "model": model, "seed": seed,
            "ctx": ctx, "result": tr["result"], "events": {k: len(v) for k, v in ev.items()},
            "status": (tr["report"] or {}).get("status"), "stop": tr["stop"], "turns": len(tr["turns"]),
            "check_exit": (tr.get("check") or {}).get("exit"), "seconds": tr["seconds"],
            "prompt_tokens_first": usage[0].get("prompt_tokens") if usage else None,
            "prompt_tokens_last": usage[-1].get("prompt_tokens") if usage else None,
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage), "error": tr["error"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one run, for a manual check")
    r.add_argument("--task", required=True)
    r.add_argument("--model", choices=MODELS, default="qwen")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--ctx", type=int, default=0, help="padding in prompt tokens")
    r.add_argument("--keep", action="store_true", help="keep the workspace")
    a = ap.parse_args()
    row = run_task(load_task(a.task), a.model, a.seed, a.ctx, OUT / "manual" / f"{a.model}-{a.task}-s{a.seed}",
                   tools.Pages(DATA / "web/pages.jsonl"), a.keep)
    print(json.dumps(row, indent=1))

if __name__ == "__main__":
    main()
