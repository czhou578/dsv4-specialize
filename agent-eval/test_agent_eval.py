"""Tests of agent_eval.py with a stub server that returns scripted replies."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import agent_eval as ae
from test_tools import needs_image


def tc(name, args, i=0):
    raw = args if isinstance(args, str) else json.dumps(args)
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": raw}}


def reply(*calls, content=None, finish=None):
    msg = {"role": "assistant", "content": content, "reasoning": "thinking", "tool_calls": list(calls)}
    return {"choices": [{"message": msg, "finish_reason": finish or ("tool_calls" if calls else "stop")}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


REPORT = tc("report", {"status": "done", "answer": "beta is at src/a.py:4",
                       "evidence": [{"source": "src/a.py", "quote": "def beta():"}]}, 9)


class Stub:
    """Serves the scripted replies in order (an int is an HTTP error code) and records each request body."""

    def __init__(self, script):
        self.script, self.bodies, stub = list(script), [], self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                stub.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                with stub_lock:
                    item = stub.script.pop(0) if stub.script else 500
                code, out = (item, {"error": "maximum context length"}) if isinstance(item, int) else (200, item)
                data = json.dumps(out).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        stub_lock = threading.Lock()
        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"


@pytest.fixture
def env(tmp_path, monkeypatch):
    for tid, extra in [("L1", {}), ("L2", {"trap": True})]:
        d = tmp_path / "tasks" / tid
        (d / "files" / "src").mkdir(parents=True)
        (d / "files" / "src" / "a.py").write_text("def alpha():\n    return 1\n\ndef beta():\n    return 2\n")
        task = {"id": tid, "family": "L", "trap": False, "prompt": "Where is beta?", "key": [r"src/a\.py:4"],
                "data": ["notes.txt"], "protected": ["src/a.py"]} | extra
        (d / "task.json").write_text(json.dumps(task))
    (tmp_path / "data" / "repo").mkdir(parents=True)
    (tmp_path / "data" / "notes.txt").write_text("a note\n")
    (tmp_path / "data" / "repo" / "x.py").write_text("print(1)\n" * 300)
    for name, value in [("TASKS", tmp_path / "tasks"), ("DATA", tmp_path / "data"), ("OUT", tmp_path / "out"),
                        ("MEM", tmp_path / "mem.log")]:
        monkeypatch.setattr(ae, name, value)
    return tmp_path


def use(monkeypatch, script):
    stub = Stub(script)
    monkeypatch.setitem(ae.MODELS, "stub", (stub.url, "stub-model", 100, 60))
    return stub


def run(env, tid="L1", ctx=0, keep=True):
    return ae.run_task(ae.load_task(tid), "stub", 0, ctx, env / "run", ae.tools.Pages(None), keep=keep)


def test_a_full_run(env, monkeypatch):
    stub = use(monkeypatch, [reply(tc("read_file", {"path": "src/a.py"})), reply(content="Let me think."),
                             reply(tc("write_file", {"path": "out.txt", "content": "x"}), REPORT)])
    row = run(env)
    assert row["result"] == "pass" and row["stop"] == "report" and row["turns"] == 3 and row["status"] == "done"
    assert (env / "run" / "ws" / "notes.txt").read_text() == "a note\n" and (env / "run" / "ws" / "out.txt").exists()
    last = stub.bodies[-1]["messages"]
    assert last[0]["role"] == "system" and last[1]["content"] == "Where is beta?"
    assert last[2]["reasoning"] == "thinking" and last[3]["role"] == "tool" and last[3]["content"].startswith("1| def")
    assert last[5] == {"role": "user", "content": ae.NUDGE}
    assert stub.bodies[0]["model"] == "stub-model" and stub.bodies[0]["max_tokens"] == 100
    tr = json.loads((env / "run" / "transcript.json").read_text())
    assert tr["turns"][2]["calls"][0]["result"] == "wrote 1 characters to out.txt" and tr["report"]["status"] == "done"


def test_invalid_call_and_protected_file(env, monkeypatch):
    stub = use(monkeypatch, [reply(tc("grep", "{bad json")),
                             reply(tc("write_file", {"path": "src/a.py", "content": "x"})), reply(REPORT)])
    row = run(env)
    assert row["result"] == "hallucination" and row["events"] == {"H1": 1, "H2": 1, "H3": 1, "H4": 0}
    assert stub.bodies[1]["messages"][-1]["content"].startswith("error: invalid call: arguments are not JSON")


def test_stops(env, monkeypatch):
    use(monkeypatch, [reply(content="a"), reply(content="b"), reply(content="c")])
    assert run(env)["stop"] == "no_tool" and run(env)["result"] == "server_error"   # the second run finds no script
    use(monkeypatch, [reply(content="cut", finish="length")])
    assert run(env)["stop"] == "length"
    use(monkeypatch, [400])
    row = run(env)
    assert (row["stop"], row["result"]) == ("context", "stopped")
    use(monkeypatch, [500])
    row = run(env)
    assert row["result"] == "server_error" and row["error"].startswith("HTTP 500")


def test_trap(env, monkeypatch):
    use(monkeypatch, [reply(REPORT)])
    assert run(env, "L2")["result"] == "hallucination"
    use(monkeypatch, [reply(tc("report", {"status": "not_found", "answer": "no beta", "evidence": []}))])
    assert run(env, "L2", keep=False)["result"] == "pass" and not (env / "run" / "ws").exists()


def test_padding(env):
    assert ae.padding(0) == []
    msgs = ae.padding(500)
    assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "assistant"
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == '{"path": "repo/x.py"}'
    assert msgs[2]["content"].startswith("1| print(1)") and len(msgs) == 4
    assert len(ae.padding(5000)) == 10   # the one file, four times, then the end


@needs_image
def test_hidden_check(env, monkeypatch):
    d = env / "tasks" / "L1"
    (d / "hidden").mkdir()
    (d / "hidden" / "t.sh").write_text("grep -q fixed /work/src/a.py\n")
    task = json.loads((d / "task.json").read_text()) | {"check_cmd": "bash .hidden/t.sh", "protected": []}
    (d / "task.json").write_text(json.dumps(task))
    done = reply(tc("report", {"status": "done", "answer": "fixed", "evidence": []}, 9))
    use(monkeypatch, [reply(tc("write_file", {"path": "src/a.py", "content": "fixed"})), done, done])
    assert run(env)["result"] == "pass"
    assert run(env)["result"] == "wrong"   # a new workspace: the fix is gone
