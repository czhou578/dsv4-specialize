"""junk_routes.py against a fake server (plans/PLAN_JUNK_CAPTURE.md §4): no model server, no GPU."""

import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import junk_routes as jr


def b64(array):
    buf = io.BytesIO()
    np.save(buf, array)
    return base64.b64encode(buf.getvalue()).decode()


class Fake:
    """A capture server: n prompt tokens (raw: one per character; chat: 10 per message), random top-6 ids per row
    plus the row of the generated token, and a sums file that moves by the counts of the prompt rows."""

    def __init__(self, sums=None, layers=43, move=True, routed=True):
        self.calls, self.sums, self.layers, self.move, self.routed = [], sums, layers, move, routed
        if sums:
            self.write(np.zeros((43, 216)), np.zeros((43, 216)), np.zeros(43), 0)

    def write(self, count, weight, tokens, steps):
        with open(f"{self.sums}.tmp", "wb") as f:
            np.savez(f, count=count, weight=weight, tokens=tokens, steps=np.array(steps))
        os.replace(f"{self.sums}.tmp", self.sums)

    def __call__(self, base, path, body, timeout=None):
        self.calls.append((path, body))
        n = len(body["prompt"]) if "prompt" in body else 10 * len(body["messages"])
        rows = np.random.default_rng(n).random((n + 1, self.layers, 216)).argsort(axis=2)[..., :6]
        if self.sums and self.move:
            with np.load(self.sums) as z:
                count = z["count"] + np.stack([np.bincount(rows[:n, l].ravel(), minlength=216) for l in range(43)])
                self.write(count, z["weight"] + count * 0.25, z["tokens"] + n, int(z["steps"]) + 1)
        choice = {"text": "x", **({"routed_experts": b64(rows)} if self.routed else {})}
        return {"choices": [choice], "usage": {"prompt_tokens": n}}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.prompts = self.d / "prompts.json"
        self.prompts.write_text(json.dumps({"probes": [{"name": "c2", "rule": "digits", "prompt": "x" * 300},
                                                       {"name": "d3", "rule": "none", "prompt": "y" * 40}],
                                            "greedy": {"prompt": "z" * 20}}))

    def test_probes_count_every_round_cold_and_a_rerun_skips_them(self):
        fake, out, sums = Fake(self.d / "sums.npz"), self.d / "cap.jsonl", self.d / "sums.npz"
        with mock.patch.object(jr, "post", fake):
            self.assertEqual(jr.capture("u", out, "m", self.prompts, rounds=2, sums=sums), 0)
            got = rows(out)
            keys = [f"probe-{n}-r{r}" for r in (1, 2) for n in ("c2", "d3", "greedy")]
            self.assertEqual([r["key"] for r in got], keys)
            c2 = got[0]
            ids = np.load(self.d / "cap.jsonl.ids/probe-c2-r1.npy")
            self.assertEqual((ids.shape, ids.dtype, c2["prompt_tokens"], c2["unfilled_rows"]),
                             ((300, 43, 6), np.uint8, 300, 0))
            self.assertTrue(all(sum(layer) == 300 * 6 for layer in c2["counts"]))
            self.assertTrue(all(sum(layer) == jr.TAIL * 6 for layer in c2["tail_counts"]))
            self.assertEqual(c2["last"], ids[-1].tolist())          # the probe position, not the generated token
            self.assertEqual((c2["wcount"], c2["steps"]), (c2["counts"], 1))
            self.assertEqual({p for p, _ in fake.calls}, {"/v1/completions"})
            self.assertTrue(all(b["max_tokens"] == 1 and b["add_special_tokens"] is False for _, b in fake.calls))
            self.assertEqual(len({b["cache_salt"] for _, b in fake.calls}), 6)
            self.assertEqual(jr.capture("u", out, "m", self.prompts, rounds=2, sums=sums), 0)
        self.assertEqual(len(fake.calls), 6)

    def test_a_failed_request_is_recorded_and_retried_on_a_rerun(self):
        out = self.d / "cap.jsonl"
        with mock.patch.object(jr, "post", Fake(layers=42)):
            self.assertEqual(jr.capture("u", out, "m", self.prompts, rounds=1), 3)
        self.assertTrue(all(r["error"].startswith("ValueError") for r in rows(out)))
        with mock.patch.object(jr, "post", Fake()):
            self.assertEqual(jr.capture("u", out, "m", self.prompts, rounds=1), 0)
        self.assertEqual([("error" in r) for r in rows(out)], [True] * 3 + [False] * 3)

    def test_smoke(self):
        sums = self.d / "sums.npz"
        with mock.patch.object(jr, "post", Fake(sums)):
            self.assertEqual(jr.smoke("u", "m", sums), 0)
        with mock.patch.object(jr, "post", Fake(routed=False)):
            self.assertEqual(jr.smoke("u", "m"), 1)
        with mock.patch.object(jr, "post", Fake(sums, move=False)), mock.patch.object(jr.time, "sleep"):
            self.assertEqual(jr.smoke("u", "m", sums), 1)

    def test_pilot_rebuilds_the_messages_of_the_last_turn(self):
        ae = SimpleNamespace(SYSTEM="sys", NUDGE="nudge", padding=lambda n: [], tools=SimpleNamespace(SCHEMAS=["t"]),
                             load_task=lambda task: {"prompt": f"do {task}"})
        call = lambda name, result: {"name": name, "arguments": "{}", "result": result}
        turns = [{"content": None, "reasoning": "r0", "calls": [call("list_dir", "a"), call("read_file", "b")]},
                 {"content": "hm", "reasoning": "", "calls": []},
                 {"content": "done", "reasoning": "r2", "calls": [call("report", "Report received.")]}]
        for name, t in (("deepseek-C2-s0", turns), ("deepseek-C2-s1", turns), ("deepseek-D1-s0", [])):
            (self.d / name).mkdir()
            (self.d / name / "transcript.json").write_text(json.dumps({"task": name.split("-")[1], "ctx": 0,
                                                                       "turns": t}))
        msgs = jr.pilot_messages({"task": "C2", "ctx": 0, "turns": turns}, ae)
        roles = ["system", "user", "assistant", "tool", "tool", "assistant", "user"]
        self.assertEqual([m["role"] for m in msgs], roles)
        self.assertEqual([m.get("tool_call_id") for m in msgs[3:5]], ["c0_0", "c0_1"])
        self.assertEqual((msgs[1]["content"], msgs[5]["reasoning"], msgs[6]["content"]), ("do C2", None, "nudge"))
        self.assertNotIn("tool_calls", msgs[5])
        fake, out = Fake(), self.d / "cap.jsonl"
        with mock.patch.object(jr, "agent_eval", lambda: ae), mock.patch.object(jr, "post", fake):
            self.assertEqual(jr.capture("u", out, "m", pilot=self.d), 0)
        got = rows(out)
        self.assertEqual([(r["key"], r["kind"], r["family"], r["prompt_tokens"]) for r in got],
                         [("pilot-C2-s0", "pilot", "C", 70)])
        self.assertEqual((fake.calls[0][0], fake.calls[0][1]["tools"]), ("/v1/chat/completions", ["t"]))

    def test_capture_needs_a_source(self):
        with self.assertRaises(SystemExit):
            jr.main(["capture", "--out", str(self.d / "x.jsonl")])


if __name__ == "__main__":
    unittest.main()
