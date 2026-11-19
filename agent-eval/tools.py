"""The tools of the agent (plans/PLAN_AGENT_CAPACITY.md §5): their schemas and their functions.

Each run has its own workspace. A path argument is relative to the workspace; "/work/x" and "./x" mean the same file.
A path that leaves the workspace, also through a symbolic link, is an error. ``search`` and ``fetch`` read a frozen
page set (a JSON-lines file with url, title, text and an optional status), not the network.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

import sandbox

STR, INT = {"type": "string"}, {"type": "integer"}


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required, "additionalProperties": False}}}


SCHEMAS = [
    _fn("read_file", "Read a text file. Each line starts with its number and '| '. Gives at most 400 lines a call.",
        {"path": STR, "start": INT, "end": INT}, ["path"]),
    _fn("grep", "Search the files under a path for a Python regular expression. Gives 'path:line: text', at most 100.",
        {"pattern": STR, "path": STR}, ["pattern"]),
    _fn("list_dir", "List a folder. A folder name ends with '/'.", {"path": STR}, []),
    _fn("write_file", "Write a text file (create or replace it).", {"path": STR, "content": STR}, ["path", "content"]),
    _fn("run", "Run a bash command in /work with no network and a limit of 120 s. Gives the exit code and the output.",
        {"command": STR}, ["command"]),
    _fn("search", "Search the web. Gives at most 5 results: the URL, the title and a snippet.", {"query": STR}, ["query"]),
    _fn("fetch", "Fetch a web page as text, 12000 characters a call from 'start'.", {"url": STR, "start": INT}, ["url"]),
    _fn("report", "Finish the task. Call it once, as the last call.", {
        "status": {"type": "string", "enum": ["done", "not_found", "failed"]},
        "answer": STR,
        "evidence": {"type": "array", "items": {"type": "object", "properties": {
            "source": STR, "quote": STR, "line": INT}, "required": ["source", "quote"], "additionalProperties": False}},
        "tests_passed": {"type": "boolean"}}, ["status", "answer", "evidence"]),
]
NAMES = {s["function"]["name"] for s in SCHEMAS}
SCHEMA = {s["function"]["name"]: s["function"]["parameters"] for s in SCHEMAS}


class ToolError(Exception):
    pass


def resolve(ws: Path, path: str) -> Path:
    """The file of a path argument inside the workspace, or ToolError."""
    p = re.sub(r"^(/work(/|$)|\./)", "", (path or ".").strip()) or "."
    full = (ws / p).resolve()
    if not full.is_relative_to(ws.resolve()):
        raise ToolError(f"the path {path!r} is outside /work")
    return full


class Pages:
    """The frozen page set: a term-overlap search and a fetch by URL."""

    def __init__(self, path: Path | None):
        self.pages = {}
        if path and path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    page = json.loads(line)
                    self.pages[page["url"]] = page
        self.terms = {u: Counter(self.words(p["title"] + " " + p["text"])) for u, p in self.pages.items()}
        df = Counter(w for c in self.terms.values() for w in c)
        self.idf = {w: math.log(1 + len(self.pages) / n) for w, n in df.items()}

    @staticmethod
    def words(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query: str) -> str:
        q = set(self.words(query))
        score = {u: sum(self.idf[w] * (1 + math.log(c[w])) for w in q if w in c) for u, c in self.terms.items()}
        best = [u for u in sorted(score, key=lambda u: (-score[u], u)) if score[u] > 0][:5]
        if not best:
            return "No results."
        out = []
        for u in best:
            text = self.pages[u]["text"]
            m = next((m for w in q if (m := re.search(rf"\b{re.escape(w)}\b", text, re.I))), None)
            i = max(0, m.start() - 80) if m else 0
            out.append(f"{u}\n{self.pages[u]['title']}\n...{' '.join(text[i:i + 240].split())}...")
        return "\n\n".join(out)

    def fetch(self, url: str, start: int = 0) -> str:
        page = self.pages.get(url.strip())
        if page is None or page.get("status", 200) != 200:
            raise ToolError(f"HTTP 404 Not Found: {url}")
        text = page["text"]
        chunk = text[start:start + 12000]
        more = f"\n[more: call fetch with start={start + 12000}]" if start + 12000 < len(text) else ""
        return f"{page['title']}\n\n{chunk}{more}"


def read_file(ws, path, start=1, end=None):
    f = resolve(ws, path)
    if not f.is_file():
        raise ToolError(f"no such file: {path}")
    lines = f.read_text(errors="replace").splitlines()
    start = max(1, start or 1)
    end = min(len(lines), end or start + 399, start + 399)
    body = "\n".join(f"{i}| {lines[i - 1]}" for i in range(start, end + 1))
    rest = f"\n[the file has {len(lines)} lines]" if end < len(lines) or start > 1 else ""
    return sandbox.clip(body, 30_000) + rest


def grep(ws, pattern, path="."):
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"bad regular expression: {e}")
    root, hits = resolve(ws, path), []
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    for f in files:
        if f.stat().st_size > 2_000_000 or not f.resolve().is_relative_to(ws.resolve()):
            continue
        try:
            text = f.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{f.relative_to(ws)}:{n}: {line[:300]}")
                if len(hits) == 100:
                    return "\n".join(hits) + "\n[stopped at 100 matches]"
    return "\n".join(hits) or "No matches."


def list_dir(ws, path="."):
    d = resolve(ws, path)
    if not d.is_dir():
        raise ToolError(f"no such folder: {path}")
    names = sorted(p.name + ("/" if p.is_dir() else "") for p in d.iterdir())
    return "\n".join(names[:500]) or "(empty)"


def write_file(ws, path, content):
    f = resolve(ws, path)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content)
    return f"wrote {len(content)} characters to {f.relative_to(ws.resolve())}"


def call(ws: Path, pages: Pages, name: str, args: dict) -> tuple[str, dict]:
    """Run one valid tool call. Returns the text for the model and a record for the transcript."""
    try:
        if name == "run":
            r = sandbox.run(str(ws), args["command"])
            return f"exit code: {r['exit']}\n{r['output']}", r
        if name == "search":
            return pages.search(args["query"]), {}
        if name == "fetch":
            return pages.fetch(args["url"], args.get("start", 0)), {}
        fn = {"read_file": read_file, "grep": grep, "list_dir": list_dir, "write_file": write_file}[name]
        return fn(ws, **args), {}
    except (ToolError, OSError) as e:
        return f"error: {e}", {"error": str(e)}
