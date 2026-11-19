"""qwen_data.py: the conversion of a Claude Code transcript, the segments, the split and the repository sessions
(plans/PLAN_QWEN_PRUNE.md §5)."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import qwen_data as qd


def line(kind, content, mid=None, **flags):
    return json.dumps({"type": kind, "message": {"role": kind, "content": content, **({"id": mid} if mid else {})},
                       "sessionId": "s", "cwd": "/secret/path", "gitBranch": "main", **flags})


def use(tid, name, **args):
    return {"type": "tool_use", "id": tid, "name": name, "input": args}


def result(tid, content):
    return {"type": "tool_result", "tool_use_id": tid, "content": content}


TRANSCRIPT = [
    line("user", "Fix the bug <system-reminder>hidden rule</system-reminder>"),
    line("assistant", [{"type": "thinking", "thinking": "private thought", "signature": "x"}], "m1"),
    line("assistant", [{"type": "text", "text": "I read the file."}], "m1"),
    line("assistant", [use("t1", "Read", file_path="a.py", limit=20)], "m1"),
    line("assistant", [use("t2", "Monitor", command="tail")], "m1"),
    line("user", [result("t1", [{"type": "text", "text": "x = 1"}, {"type": "image", "source": {}}])]),
    line("user", [result("t2", "monitor started")]),
    line("assistant", [use("t3", "Edit", file_path="a.py", old_string="1", new_string="2", replace_all=True)], "m2"),
    line("user", [result("t3", "ok " + "y" * 20_000)]),
    line("assistant", [use("t4", "Bash", command="curl -H 'Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123'")], "m3"),
    line("user", [result("t4", "token ghp_" + "a" * 36)]),
    line("user", "meta text", isMeta=True),
    line("assistant", [{"type": "text", "text": "side"}], "m9", isSidechain=True),
    line("assistant", [{"type": "text", "text": "Done."}], "m4"),
    "not json",
]


class TranscriptTest(unittest.TestCase):
    def setUp(self):
        self.m = qd.transcript_messages(TRANSCRIPT)

    def test_roles_and_merged_assistant_blocks(self):
        self.assertEqual([m["role"] for m in self.m],
                         ["user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"])
        self.assertEqual(self.m[0]["content"], "Fix the bug")                       # the reminder is gone
        first = self.m[1]
        self.assertEqual((first["content"], first["target"]), ("I read the file.", True))   # no thinking text
        self.assertEqual(len(first["tool_calls"]), 1)                               # Monitor has no Hermes tool

    def test_tool_calls_map_to_hermes_tools(self):
        calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for m in self.m for c in m.get("tool_calls", [])]
        self.assertEqual(calls[0], ("read_file", {"path": "a.py", "limit": 20}))
        self.assertEqual(calls[1], ("patch", {"mode": "replace", "path": "a.py", "old_string": "1", "new_string": "2",
                                              "replace_all": True}))
        self.assertEqual(calls[2][0], "terminal")
        self.assertEqual(qd.hermes_call("Glob", {"pattern": "*.py"}), ("search_files", {"pattern": "*.py", "target": "files"}))
        self.assertEqual(qd.hermes_call("WebSearch", {"query": "q"}), ("web_search", {"query": "q"}))
        self.assertIsNone(qd.hermes_call("Agent", {}))

    def test_results_pair_with_kept_calls_and_long_results_are_cut(self):
        tools = [m for m in self.m if m["role"] == "tool"]
        self.assertEqual([t["tool_call_id"] for t in tools], ["t1", "t3", "t4"])   # the Monitor result is dropped
        self.assertEqual(tools[0]["content"], "x = 1")                             # the image block is dropped
        self.assertLess(len(tools[1]["content"]), 12_000)
        self.assertIn("[... cut ...]", tools[1]["content"])

    def test_secrets_and_metadata_do_not_pass(self):
        text = json.dumps(self.m)
        for secret in ("abcdefghijklmnopqrstuvwxyz0123", "ghp_" + "a" * 36, "/secret/path", "private thought", "hidden rule"):
            self.assertNotIn(secret, text)
        self.assertEqual(text.count("[REDACTED]"), 2)


class SegmentTest(unittest.TestCase):
    def test_cuts_before_a_user_text_and_hard_cuts_before_an_assistant(self):
        turn = lambda k: [{"role": "user", "content": f"q{k}"}, {"role": "assistant", "content": "a" * 900, "target": True}]
        segs = qd.split_segments([m for k in range(6) for m in turn(k)], seg_chars=1500, hard=10_000, least=100)
        self.assertEqual([len(s) for s in segs], [4, 4, 4])
        self.assertTrue(all(s[0]["role"] == "user" for s in segs))
        long = [{"role": "user", "content": "q"}] + [{"role": "assistant", "content": "a" * 900, "target": True}] * 5
        segs = qd.split_segments(long, seg_chars=1500, hard=2000, least=100)
        self.assertEqual(segs[1][0], {"role": "user", "content": "Continue the task."})
        self.assertEqual(qd.split_segments([{"role": "user", "content": "x" * 5000}], least=100), [])   # no target
        self.assertEqual(qd.split_segments(long[1:], least=100)[0][0]["content"], "Continue the task.")   # no user first


class BudgetTest(unittest.TestCase):
    def test_the_budget_takes_segments_from_each_project_in_turn(self):
        seg = lambda n: [{"role": "user", "content": "x" * n}, {"role": "assistant", "content": "a", "target": True}]
        s = lambda sid, skill, split, n: {"id": sid, "skill": skill, "split": split, "source": "claude", "segments": [seg(4000)] * n}
        sessions = [s("big", "alpha", "calibration", 10), s("small", "beta", "calibration", 2), s("held", "alpha", "held_out", 5),
                    {**s("bot", "research", "calibration", 3), "source": "discord"}]
        out = {x["id"]: x for x in qd.limit_calibration(sessions, {"claude": 4000})}      # 4 segments of 1,000 tokens
        self.assertEqual((len(out["big"]["segments"]), len(out["small"]["segments"])), (2, 2))
        self.assertEqual((len(out["held"]["segments"]), len(out["bot"]["segments"])), (5, 3))   # not calibration, no budget
        self.assertEqual(qd.project_name("-home-colin-spark-Projects-llm-reader", Path("/home/colin-spark")), "llm-reader")
        self.assertEqual(qd.project_name("-home-colin-spark--hermes-stock-research", Path("/home/colin-spark")), "hermes-stock-research")
        self.assertEqual(qd.project_name("-home-u-Projects-alpha", Path("/home/other")), "alpha")


class BuildTest(unittest.TestCase):
    def test_the_output_holds_the_three_sources_with_a_split_by_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "bot").mkdir()
            bot = [{"id": "b1", "skill": "research", "split": "held_out", "source": "discord", "system_prompt": "be hermes",
                    "segments": [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "target": True}]]},
                   {"id": "b2", "skill": "x", "split": "calibration", "source": "cli", "system_prompt": "p", "segments": []}]
            (d / "bot/sessions.jsonl").write_text("".join(json.dumps(s) + "\n" for s in bot))
            (d / "bot/tools.json").write_text('{"tools": []}')
            body = "\n".join([line("user", "Explain " + "z" * 3000), line("assistant", [{"type": "text", "text": "ok"}], "m1")])
            for project, n in (("-home-u-Projects-alpha", 30), ("-home-u-Projects-heldrepo", 2), ("-tmp-claude-1000-x", 2)):
                (d / "tr" / project).mkdir(parents=True)
                for k in range(n):
                    (d / "tr" / project / f"{k:08d}-aaaa.jsonl").write_text(body)
            repo = d / "myrepo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            for k in range(8):
                (repo / f"f{k}.py").write_text("print(1)\n" * 100)
            (repo / "tiny.py").write_text("x")
            (repo / "data.bin").write_text("b" * 2000)
            subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
            sessions = qd.build(d / "out", d / "tr", [repo], held_repos=("heldrepo",), bot=d / "bot")
            saved = [json.loads(x) for x in (d / "out/sessions.jsonl").read_text().splitlines()]
        self.assertEqual(len(saved), len(sessions))
        by = lambda src: [s for s in saved if s["source"] == src]
        self.assertEqual([s["id"] for s in by("discord")], ["b1"])                  # the cli session is not bot traffic
        claude = by("claude")
        self.assertEqual(len(claude), 32)                                           # no tmp project
        self.assertTrue(all(s["split"] == "held_out" for s in claude if s["skill"] == "heldrepo"))
        held = sum(s["split"] == "held_out" for s in claude if s["skill"] == "alpha")
        self.assertTrue(0 < held < 15)                                              # about 15% by the hash
        self.assertTrue(all(s["system_prompt"] == "be hermes" for s in claude))
        repo_s = by("repo")
        self.assertEqual(len(repo_s), 6)
        seg = repo_s[0]["segments"][0]
        self.assertEqual([m["role"] for m in seg], ["user", "assistant", "tool"])
        self.assertEqual(seg[1]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(seg[2]["tool_call_id"], seg[1]["tool_calls"][0]["id"])
        self.assertTrue(any("claude" in row and "alpha" in row for row in qd.stats(saved)))


if __name__ == "__main__":
    unittest.main()
