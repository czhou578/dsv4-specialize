"""qwen_gate.py: the pick of the held-out targets, the record, the score of a recorded turn and the comparison
(plans/PLAN_QWEN_PRUNE.md §4)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import qwen_gate as qg

END = 77


def session(sid, skill, split="held_out", source="claude", turns=3, system="be hermes"):
    seg = []
    for k in range(turns):
        seg += [{"role": "user", "content": f"q{k}"}, {"role": "assistant", "content": f"a{k}", "target": True}]
    return {"id": sid, "skill": skill, "split": split, "source": source, "system_prompt": system, "segments": [seg, seg]}


class FakeServer:
    """A context of n messages has 10 * n + 1 tokens with the generation prompt. The scored turn is: 3 reasoning
    tokens, ``</think>``, then ``{"name": "terminal"}`` in 4 tokens. ``slip`` gives rank 2 to the tokens at these turn
    positions; every token has the logprob ``-lp``."""

    def __init__(self, lp=0.5, slip=()):
        self.lp, self.slip, self.bodies = lp, set(slip), []

    def __call__(self, base, path, body, timeout=None):
        self.bodies.append((path, body))
        if path == "/tokenize":
            return {"tokens": [END] if "prompt" in body else [0] * (10 * len(body["messages"]) + 1)}
        if "prompt_logprobs" not in body:
            return {"choices": [{"message": {"content": "done", "reasoning": "why", "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
                    "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 9}}
        turn = [(5, "x"), (5, "y"), (5, "z"), (END, "</think>"), (6, '{"name": "'), (7, "term"), (8, "inal"), (9, '"}')]
        entry = lambda k, t, text: {str(t): {"logprob": -self.lp, "rank": 2 if k in self.slip else 1, "decoded_token": text}}
        context = 10 * (len(body["messages"]) - 1) + 1
        return {"prompt_logprobs": [None] + [{"0": {"logprob": -9.0, "rank": 5, "decoded_token": "c"}}] * (context - 1)
                + [entry(k, t, text) for k, (t, text) in enumerate(turn)]}


class GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.sessions = [session("h1", "alpha"), session("h2", "beta"), session("h3", "alpha", source="discord"),
                         session("c1", "alpha", split="calibration")]

    def tearDown(self):
        self.tmp.cleanup()

    def run_record(self, fake, n=4):
        with mock.patch.object(qg, "post", fake):
            qg.record("http://x", "m", self.sessions, [], n, self.d / "rec.jsonl")
        return qg.load(self.d / "rec.jsonl")

    def run_score(self, fake, name):
        with mock.patch.object(qg, "post", fake):
            qg.score("http://x", "m", self.sessions, [], qg.load(self.d / "rec.jsonl"), self.d / name)
        return qg.load(self.d / name)

    def test_pick_takes_held_out_targets_one_per_segment_group_after_group(self):
        picked = qg.pick(self.sessions, 5)
        self.assertEqual(len(picked), 5)
        self.assertEqual({p[0] for p in picked}, {"h1", "h2", "h3"})              # no calibration session
        self.assertEqual(len({p[:2] for p in picked}), 5)                          # one target per segment
        self.assertEqual({p[0] for p in picked[:3]}, {"h1", "h2", "h3"})          # each group once before a second turn
        self.assertTrue(all(p[2] > 0 and self.sessions[0]["segments"][0][p[2]].get("target") for p in picked))
        self.assertEqual(qg.pick(self.sessions, 5), picked)                        # a fixed order
        more = qg.pick(self.sessions, 9)
        self.assertEqual((len({p[:2] for p in more[:6]}), len({p[:2] for p in more}), len(set(more))), (6, 6, 9))

    def test_record_sends_the_context_and_keeps_the_turn(self):
        fake = FakeServer()
        rows = self.run_record(fake)
        self.assertEqual(len(rows), 4)
        body = fake.bodies[0][1]
        self.assertEqual((body["temperature"], body["top_p"], body["messages"][0]["role"]), (0.6, 0.95, "system"))
        self.assertNotIn("target", json.dumps(body["messages"]))
        self.assertEqual(len(body["messages"]), 1 + rows[0]["index"])               # the segment up to the target
        self.assertEqual(rows[0]["turn"]["reasoning_content"], "why")
        self.assertEqual(rows[0]["turn"]["tool_calls"][0]["function"]["name"], "terminal")
        self.assertEqual(len(self.run_record(FakeServer())), 4)                     # a rerun adds nothing

    def test_score_reads_the_turn_tokens_after_the_context(self):
        self.run_record(FakeServer())
        row = self.run_score(FakeServer(lp=0.5), "ref.jsonl")[0]
        self.assertEqual((row["tokens"], row["reasoning_tokens"], row["top1"]), (8, 4, 8))
        self.assertAlmostEqual(row["nll"], 4.0)
        self.assertEqual((row["decision_ok"], row["tool_calls"], row["tool_names_ok"]), (True, 1, 1))
        slipped = self.run_score(FakeServer(lp=0.5, slip={4, 6}), "slip.jsonl")[0]   # the first answer token, and "inal"
        self.assertEqual((slipped["decision_ok"], slipped["tool_names_ok"], slipped["top1"]), (False, 0, 6))

    def test_compare_applies_the_three_gates(self):
        self.run_record(FakeServer())
        self.run_score(FakeServer(lp=0.50), "ref.jsonl")
        self.run_score(FakeServer(lp=0.51), "ok.jsonl")
        self.run_score(FakeServer(lp=0.52, slip={6}), "bad.jsonl")
        ref, new, gates = qg.compare(self.d / "ref.jsonl", self.d / "ok.jsonl")
        self.assertAlmostEqual(new["nll"] / ref["nll"], 1.02)
        self.assertTrue(all(gates.values()))
        _, new, gates = qg.compare(self.d / "ref.jsonl", self.d / "bad.jsonl")
        self.assertEqual(list(gates.values()), [False, False, True])                 # +4% NLL, no tool name; decisions hold
        self.assertEqual(new["tool_names"], 0.0)

    def test_a_failed_turn_is_recorded_and_left_out(self):
        self.run_record(FakeServer())
        def broken(base, path, body, timeout=None):
            if path == "/tokenize" and "prompt" in body:
                return {"tokens": [END]}
            raise RuntimeError("boom")
        rows = self.run_score(broken, "err.jsonl")
        self.assertTrue(all("boom" in r["error"] for r in rows))
        self.run_score(FakeServer(), "ref.jsonl")
        self.assertEqual(qg.compare(self.d / "ref.jsonl", self.d / "err.jsonl")[0]["turns"], 0)


if __name__ == "__main__":
    unittest.main()
