"""capture.py: count the routed experts of the calibration segments."""

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import capture


def session(sid, skill, split="calibration", source="discord", system="be the bot"):
    seg = [{"role": "user", "content": "q"},
           {"role": "assistant", "content": "", "reasoning_content": "why", "target": True},
           {"role": "tool", "content": "r", "tool_call_id": "c"},
           {"role": "assistant", "content": "done", "target": True},
           {"role": "user", "content": "thanks"}]
    return {"id": sid, "skill": skill, "split": split, "source": source,
            "system_prompt": system, "segments": [seg]}


def b64(array):
    buf = io.BytesIO()
    np.save(buf, np.asarray(array, dtype=np.uint8))
    return base64.b64encode(buf.getvalue()).decode()


class FakeServer:
    """``/tokenize`` gives the shared prefix [1, 2]; each chat reply routes the prompt tokens
    ``[1, 2, x, ...]`` with two layers and top-2: prefix rows pick experts 0 and 1, the others 4 and 5."""

    def __init__(self, prompts):
        self.prompts = list(prompts)
        self.chats = 0

    def __call__(self, base, path, body, timeout=None):
        if path == "/tokenize":
            # the server renders the default effort line unless the request passes the bot's effort
            effort = (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
            return {"tokens": [1, 2] if effort == "medium" else [7, 1, 2]}
        prompt = self.prompts[self.chats]
        self.chats += 1
        rows = [[[0, 1], [0, 1]] if t in (1, 2) else [[4, 5], [4, 5]] for t in prompt]
        return {"prompt_token_ids": prompt, "choices": [{"routed_experts": b64(rows)}]}


class CaptureTest(unittest.TestCase):
    def test_segments_take_the_last_target_of_bot_calibration_traffic(self):
        sessions = [session("a", "research"), session("held", "research", split="held_out"),
                    session("cli", "research", source="cli"), session("sub", "quant", source="subagent")]
        picked = [(s["id"], g, i) for s, g, i in capture.segments(sessions)]
        self.assertEqual(picked, [("a", 0, 3), ("sub", 0, 3)])

    def test_the_body_is_the_replay_request_without_logprobs(self):
        s = session("a", "research", system=None)
        body = capture.capture_body("m", s, s["segments"][0], 3, [], "medium")
        self.assertNotIn("prompt_logprobs", body)
        self.assertEqual((body["max_tokens"], body["return_token_ids"]), (1, True))
        self.assertEqual(body["messages"][0]["role"], "user")  # no empty system message

    def test_count_skips_unfilled_rows_and_the_start(self):
        routed = np.array([[[3, 4], [0, 0]],      # layer 1 unfilled
                           [[3, 5], [1, 2]],
                           [[7, 7], [1, 2]]],     # layer 0 unfilled
                          dtype=np.uint8)
        counts, unfilled = capture.count(routed)
        self.assertEqual(unfilled, 2)
        self.assertEqual(counts.shape, (2, capture.EXPERTS))
        self.assertEqual((counts[0, 3], counts[0, 4], counts[0, 5], counts[0, 7]), (2, 1, 1, 0))
        self.assertEqual((counts[1, 0], counts[1, 1], counts[1, 2]), (0, 2, 2))
        counts, _ = capture.count(routed, start=2)
        self.assertEqual(counts.sum(), 2)

    def test_run_counts_the_shared_prefix_once_and_resumes(self):
        sessions = [session("a", "research"), session("b", "quant")]
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp), Path(tmp) / "counts.jsonl"
            (data / "sessions.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sessions))
            (data / "tools.json").write_text(json.dumps({"tools": []}))
            server = FakeServer([[1, 2, 9, 9], [1, 2, 8]])
            with mock.patch.object(capture, "post", server):
                capture.run("http://x", "m", out, data=data)
            rows = capture.load(out)
            self.assertEqual([r["prefix_counted"] for r in rows], [True, False])
            self.assertEqual([r["prefix_tokens"] for r in rows], [2, 2])
            self.assertEqual([np.array(r["counts"])[0, [0, 4]].tolist() for r in rows],
                             [[2, 2], [0, 1]])
            with mock.patch.object(capture, "post", FakeServer([])):
                capture.run("http://x", "m", out, data=data)  # both done: no request
            self.assertEqual(len(capture.load(out)), 2)

            result = capture.rank(capture.load(out))
            self.assertEqual((result["segments"], result["prompt_tokens"]), (2, 7))
            top = result["layers"]["0"][:2]
            self.assertEqual([(e["id"], e["count"]) for e in top], [(4, 3), (5, 3)])
            self.assertAlmostEqual(sum(e["share"] for e in result["layers"]["0"]), 1.0, places=4)
            self.assertEqual(sorted(result["per_skill"]), ["quant", "research"])

    def test_a_failed_segment_is_recorded_and_not_ranked(self):
        rows = [{"skill": "research", "error": "boom"},
                {"skill": "research", "prompt_tokens": 3, "unfilled_rows": 0,
                 "counts": [[1] + [0] * (capture.EXPERTS - 1)]}]
        self.assertEqual(capture.rank(rows)["segments"], 1)
        with self.assertRaises(ValueError):
            capture.rank(rows[:1])


if __name__ == "__main__":
    unittest.main()
