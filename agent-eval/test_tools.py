"""Tests of tools.py and sandbox.py. The sandbox tests need Docker and the image agent-eval-sandbox:1."""

import json
import subprocess

import pytest

import sandbox
import tools

HAVE_IMAGE = subprocess.run(["docker", "image", "inspect", sandbox.IMAGE], capture_output=True).returncode == 0
needs_image = pytest.mark.skipif(not HAVE_IMAGE, reason="no sandbox image")


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    (w / "src").mkdir(parents=True)
    (w / "src" / "a.py").write_text("def alpha():\n    return 1\n\ndef beta():\n    return alpha() + 1\n")
    return w


def test_resolve_keeps_paths_inside(ws, tmp_path):
    assert tools.resolve(ws, "/work/src/a.py") == (ws / "src/a.py").resolve()
    assert tools.resolve(ws, "./src") == (ws / "src").resolve()
    (tmp_path / "secret").write_text("x")
    (ws / "link").symlink_to(tmp_path / "secret")
    for bad in ["../secret", "/etc/passwd", "link"]:
        with pytest.raises(tools.ToolError):
            tools.resolve(ws, bad)


def test_read_grep_list_write(ws):
    assert tools.read_file(ws, "src/a.py", 4, 5) == "4| def beta():\n5|     return alpha() + 1\n[the file has 5 lines]"
    assert tools.grep(ws, r"alpha\(") == "src/a.py:1: def alpha():\nsrc/a.py:5:     return alpha() + 1"
    assert tools.grep(ws, "zeta") == "No matches."
    assert tools.write_file(ws, "out/b.txt", "hi") == "wrote 2 characters to out/b.txt"
    assert tools.list_dir(ws) == "out/\nsrc/"
    text, rec = tools.call(ws, tools.Pages(None), "read_file", {"path": "nope.py"})
    assert text == "error: no such file: nope.py" and rec == {"error": "no such file: nope.py"}
    text, _ = tools.call(ws, tools.Pages(None), "grep", {"pattern": "("})
    assert text.startswith("error: bad regular expression")


def test_pages(tmp_path):
    f = tmp_path / "pages.jsonl"
    rows = [{"url": "https://a.test/x", "title": "Rust borrow checker", "text": "The borrow checker enforces rules."},
            {"url": "https://a.test/y", "title": "Python GIL", "text": "The GIL is a lock. " * 1000},
            {"url": "https://a.test/dead", "title": "Old borrow page", "text": "borrow", "status": 404}]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    pages = tools.Pages(f)
    hits = pages.search("borrow checker")
    assert hits.startswith("https://a.test/x\nRust borrow checker\n") and "https://a.test/dead" in hits
    assert pages.search("quantum") == "No results."
    with pytest.raises(tools.ToolError, match="404"):
        pages.fetch("https://a.test/dead")
    with pytest.raises(tools.ToolError, match="404"):
        pages.fetch("https://a.test/none")
    first = pages.fetch("https://a.test/y")
    assert first.startswith("Python GIL\n\nThe GIL") and first.endswith("[more: call fetch with start=12000]")
    assert not pages.fetch("https://a.test/y", 12000).endswith("]")


def test_schemas_are_unique():
    assert len(tools.NAMES) == len(tools.SCHEMAS) == 8 and "report" in tools.SCHEMA


def test_clip():
    assert sandbox.clip("abc", 10) == "abc"
    assert sandbox.clip("a" * 30, 10) == "aaaaa\n[... 20 characters left out ...]\naaaaa"


@needs_image
def test_run_isolates_the_command(ws):
    r = sandbox.run(str(ws), "python3 -c 'print(6*7)'; touch made.txt; echo err >&2; exit 3")
    assert r["exit"] == 3 and "42" in r["output"] and "err" in r["output"] and (ws / "made.txt").exists()
    r = sandbox.run(str(ws), "python3 -c 'import socket; socket.create_connection((\"1.1.1.1\", 80), 2)'")
    assert r["exit"] != 0 and "unreachable" in r["output"].lower()
    r = sandbox.run(str(ws), "touch /etc/x")
    assert r["exit"] != 0 and "read-only" in r["output"].lower()


@needs_image
def test_run_limits(ws):
    assert sandbox.run(str(ws), "sleep 30", timeout=2)["exit"] == 124
    assert sandbox.run(str(ws), "python3 -c 'x = bytearray(1500 * 2**20)'")["exit"] == 137


@needs_image
def test_toolchains(ws):
    r = sandbox.run(str(ws), "g++ --version | head -1; rustc --version; cargo --version; python3 -m pytest --version")
    assert r["exit"] == 0 and "rustc 1.90.0" in r["output"] and "pytest" in r["output"], r["output"]
    text, rec = tools.call(ws, tools.Pages(None), "run", {"command": "echo hi"})
    assert text == "exit code: 0\nhi\n" and rec["exit"] == 0
