"""Build the calibration and held-out data of the Qwen prune (plans/PLAN_QWEN_PRUNE.md §3).

Three sources become one ``sessions.jsonl`` in the format of the bot's ``model_eval`` data, so that ``capture.py`` and the
replay read it unchanged:

- ``claude``: the Claude Code transcripts of ``~/.claude/projects``. Each session becomes chat messages in the OpenAI
  format with the tools of Hermes: a shell call becomes ``terminal``, a file read ``read_file``, an edit ``patch``, a
  web search ``web_search``. Calls of tools with no Hermes match, thinking blocks and images are left out. Long tool
  results are cut. A session is split into segments of about 40,000 tokens.
- ``repo``: files of the user's repositories, each as a short session: a request, a ``read_file`` call, the file.
- The bot's own sessions (``discord``, ``subagent``) pass through unchanged, with their split.

The system prompt and the tool definitions are those of the bot's Hermes data. The split of the new sessions is by
session: ``held_out`` for a share of the sessions (by a hash) and for every session of a held-out project.

The output can hold secrets from the transcripts: it stays local, in a folder that .gitignore covers. A pass replaces
strings that look like keys with ``[REDACTED]``.
"""

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

HOME = Path.home()
BOT_DATA = HOME / "Projects/stock-research-model-eval/model_eval/data"
SECRET = re.compile(r"sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{30,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}"
                    r"|xox[bap]-[A-Za-z0-9-]{20,}|Bearer [A-Za-z0-9._-]{20,}"
                    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)
NOISE = re.compile(r"<system-reminder>.*?</system-reminder>|<local-command-\w+>.*?</local-command-\w+>"
                   r"|<command-(name|message|args)>.*?</command-\1>", re.S)
CODE = {".py", ".ts", ".tsx", ".js", ".go", ".sql", ".sh", ".cu", ".c", ".cpp", ".h", ".rs", ".md", ".yml", ".yaml", ".toml"}


def clean(text):
    return SECRET.sub("[REDACTED]", NOISE.sub("", text or "")).strip()


def cut(text, limit=12_000):
    """A long tool result keeps its first 8,000 and last 3,000 characters."""
    return text if len(text) <= limit else text[:8000] + "\n[... cut ...]\n" + text[-3000:]


def hermes_call(name, args):
    """The Hermes tool call for a Claude Code tool call, or None when Hermes has no such tool."""
    a = args or {}
    if name == "Bash":
        return "terminal", {"command": a.get("command", "")}
    if name == "Read":
        return "read_file", {k2: a[k1] for k1, k2 in (("file_path", "path"), ("offset", "offset"), ("limit", "limit")) if k1 in a}
    if name == "Write":
        return "write_file", {"path": a.get("file_path", ""), "content": a.get("content", "")}
    if name == "Edit":
        return "patch", {"mode": "replace", "path": a.get("file_path", ""), "old_string": a.get("old_string", ""),
                         "new_string": a.get("new_string", ""), **({"replace_all": True} if a.get("replace_all") else {})}
    if name in ("Grep", "Glob"):
        out = {"pattern": a.get("pattern", ""), **({"path": a["path"]} if a.get("path") else {})}
        return "search_files", {**out, **({"target": "files"} if name == "Glob" else {}),
                                **({"file_glob": a["glob"]} if a.get("glob") else {})}
    if name == "WebSearch":
        return "web_search", {"query": a.get("query", "")}
    if name == "WebFetch":
        return "web_extract", {"urls": [a.get("url", "")]}
    return None


def result_text(content):
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text") or "" for b in content or [] if isinstance(b, dict) and b.get("type") == "text")


def transcript_messages(lines):
    """The chat messages of one transcript: user texts, assistant turns (blocks merged by message id) with their mapped
    tool calls, and the results of the kept calls."""
    out, kept, last_id = [], set(), None
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") not in ("user", "assistant") or any(r.get(k) for k in (
                "isSidechain", "isMeta", "isCompactSummary", "isVisibleInTranscriptOnly")):
            continue
        m = r.get("message") or {}
        blocks = m.get("content")
        blocks = [{"type": "text", "text": blocks}] if isinstance(blocks, str) else [b for b in blocks or [] if isinstance(b, dict)]
        if r["type"] == "assistant":
            if m.get("id") != last_id or not out or out[-1]["role"] != "assistant":
                out.append({"role": "assistant", "content": "", "target": True})
                last_id = m.get("id")
            turn = out[-1]
            for b in blocks:
                if b.get("type") == "text":
                    turn["content"] = (turn["content"] + "\n" + clean(b.get("text"))).strip()
                elif b.get("type") == "tool_use" and (call := hermes_call(b.get("name"), b.get("input"))):
                    kept.add(b["id"])
                    turn.setdefault("tool_calls", []).append({"id": b["id"], "type": "function", "function": {
                        "name": call[0], "arguments": clean(json.dumps(call[1]))}})
            continue
        for b in blocks:
            if b.get("type") == "tool_result" and b.get("tool_use_id") in kept:
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": cut(clean(result_text(b.get("content"))))})
            elif b.get("type") == "text" and (text := clean(b.get("text"))):
                out.append({"role": "user", "content": cut(text, 40_000)})
    return [m for m in out if m["role"] != "assistant" or m["content"] or m.get("tool_calls")]


def size(m):
    return len(m.get("content") or "") + sum(len(c["function"]["arguments"]) for c in m.get("tool_calls", []))


def split_segments(messages, seg_chars=160_000, hard=360_000, least=2_000):
    """Segments of about ``seg_chars`` characters. A cut comes before a user text; a segment past ``hard`` is cut before
    the next assistant turn, and the new segment starts with a short user message."""
    segs, cur, n = [], [], 0
    for m in messages:
        soft = m["role"] == "user" and n >= seg_chars
        if cur and (soft or (m["role"] == "assistant" and n >= hard)):
            segs.append(cur)
            cur, n = ([] if soft else [{"role": "user", "content": "Continue the task."}]), 0
        cur.append(m)
        n += size(m)
    segs = [s if s[0]["role"] == "user" else [{"role": "user", "content": "Continue the task."}, *s] for s in [*segs, cur] if s]
    return [s for s in segs if sum(map(size, s)) >= least and any(m.get("target") for m in s)]


def project_name(folder, home=HOME):
    """``llm-reader`` for the transcript folder ``-home-<user>-Projects-llm-reader`` (the folder is the path with "-")."""
    prefix = str(home).replace("/", "-")
    rest = folder[len(prefix):] if folder.startswith(prefix) else re.sub(r"^-home-[^-]+", "", folder)
    return rest.lstrip("-").removeprefix("Projects-")


def limit_calibration(sessions, budgets):
    """Keep at most ``budgets[source]`` estimated tokens of calibration segments per source: one segment at a time from
    each project in turn, in a fixed hash order, so that no project fills the budget alone. Other segments are dropped."""
    out = [s for s in sessions if s["split"] != "calibration" or s["source"] not in budgets]
    for source, budget in budgets.items():
        queues = defaultdict(list)
        for s in sessions:
            if s["split"] == "calibration" and s["source"] == source:
                queues[s["skill"]] += [(hashlib.sha256(f"{s['id']}:{g}".encode()).hexdigest(), s["id"], g) for g in range(len(s["segments"]))]
        queues, keep, total = {k: sorted(v) for k, v in sorted(queues.items())}, defaultdict(list), 0
        by_id = {s["id"]: s for s in sessions}
        while total < budget and any(queues.values()):
            for q in queues.values():
                if q and total < budget:
                    _, sid, g = q.pop(0)
                    keep[sid].append(g)
                    total += sum(map(size, by_id[sid]["segments"][g])) // 4
        out += [{**by_id[sid], "segments": [by_id[sid]["segments"][g] for g in sorted(gs)]} for sid, gs in keep.items()]
    return out


def is_held_out(key, share):
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100 < share


def claude_sessions(root, system, held_projects=(), share=15):
    for f in sorted(Path(root).glob("*/*.jsonl")):
        project = project_name(f.parent.name)
        if "-tmp-" in f.parent.name or project.startswith("tmp-"):
            continue
        with f.open(errors="replace") as fh:
            segs = split_segments(transcript_messages(fh))
        if segs:
            held = project in held_projects or is_held_out(f.stem, share)
            yield {"id": f"cc_{project}_{f.stem[:8]}", "skill": project, "source": "claude", "system_prompt": system,
                   "split": "held_out" if held else "calibration", "segments": segs}


def repo_sessions(repo, system, files=6, held=False):
    """``files`` code files of a repository, in a fixed hash order: a request, a read_file call, and the file."""
    repo = Path(repo)
    names = subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True, text=True).stdout.split("\n")
    names = [n for n in names if Path(n).suffix in CODE and (repo / n).is_file() and 500 <= (repo / n).stat().st_size <= 40_000]
    for n in sorted(names, key=lambda n: hashlib.sha256(n.encode()).hexdigest())[:files]:
        call = {"id": "call_" + hashlib.sha256(n.encode()).hexdigest()[:12], "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": n})}}
        seg = [{"role": "user", "content": f"In the repository {repo.name}, read `{n}`. Explain what it does, then propose "
                                           "one concrete improvement as a patch."},
               {"role": "assistant", "content": "", "tool_calls": [call], "target": True},
               {"role": "tool", "tool_call_id": call["id"], "content": cut(clean((repo / n).read_text(errors="replace")), 40_000)}]
        yield {"id": f"repo_{repo.name}_{call['id'][5:]}", "skill": repo.name, "source": "repo", "system_prompt": system,
               "split": "held_out" if held else "calibration", "segments": [seg]}


def load(path):
    with Path(path).open() as f:   # not splitlines(): a JSON string can hold a Unicode line separator
        return [json.loads(line) for line in f]


def build(out, transcripts, repos, held_repos=(), bot=BOT_DATA, share=15, budgets=None):
    bot_sessions = load(Path(bot) / "sessions.jsonl")
    system = bot_sessions[0]["system_prompt"]
    sessions = [s for s in bot_sessions if s["source"] in ("discord", "subagent")]
    sessions += claude_sessions(transcripts, system, held_repos, share)
    for repo in repos:
        sessions += repo_sessions(repo, system, held=Path(repo).name in held_repos)
    sessions = limit_calibration(sessions, budgets or {})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sessions.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sessions))
    (out / "tools.json").write_text((Path(bot) / "tools.json").read_text())
    return sessions


def stats(sessions):
    """Lines of text: the sessions, the segments and the estimated tokens (characters / 4) per source, split and project."""
    rows = defaultdict(Counter)
    for s in sessions:
        chars = sum(size(m) for seg in s["segments"] for m in seg)
        for key in ((s["source"], s["split"], "*"), (s["source"], s["split"], s["skill"])):
            rows[key].update({"sessions": 1, "segments": len(s["segments"]), "tokens": chars // 4})
    return [f"{src:<9}{split:<12}{skill:<28}{c['sessions']:>5} sessions{c['segments']:>6} segments{c['tokens']:>11,} tokens"
            for (src, split, skill), c in sorted(rows.items())]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "qwen-data")
    ap.add_argument("--transcripts", type=Path, default=HOME / ".claude/projects")
    ap.add_argument("--repo", action="append", type=Path, default=[], help="a repository on disk; give it once per repository")
    ap.add_argument("--held-out", action="append", default=[], help="a project or repository name that is held out whole")
    ap.add_argument("--budget", default="claude=1100000,discord=600000",
                    help="the calibration tokens to keep per source, as source=tokens pairs")
    ap.add_argument("--stats", action="store_true", help="print the table of an existing output folder and exit")
    a = ap.parse_args(argv)
    if a.stats:
        sessions = load(a.out / "sessions.jsonl")
    else:
        budgets = {k: int(v) for k, v in (pair.split("=") for pair in a.budget.split(",") if pair)}
        sessions = build(a.out, a.transcripts, a.repo, tuple(a.held_out), budgets=budgets)
    print("\n".join(stats(sessions)))


if __name__ == "__main__":
    main()
