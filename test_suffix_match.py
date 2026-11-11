"""Tests of suffix_match.py (plans/PLAN_SERVE_TUNE.md §8.1), on synthetic token lists."""

import unittest

import numpy as np

import suffix_match as sm


def session(started, reasoning):
    call = {"function": {"name": "terminal", "arguments": '{"command": "ls", "timeout": 5}'}}
    seg = [{"role": "user", "content": "q"},
           {"role": "assistant", "content": "", "reasoning_content": reasoning, "tool_calls": [call], "target": True},
           {"role": "tool", "content": "r", "tool_call_id": "c"},
           {"role": "assistant", "content": "done", "reasoning_content": "copy"}]
    return {"started_at": started, "system_prompt": "sys", "segments": [seg]}


class SuffixMatchTest(unittest.TestCase):
    def test_longest_match_and_most_recent_occurrence(self):
        length, end = sm.scan(np.array([1, 2, 3, 4, 7, 9, 1, 2, 3, 4, 8, 9, 5, 2, 3, 4, 0, 1, 2, 3, 4]),
                              np.array([21, 16, 3]))
        # 1 2 3 4 ends at 4 and at 10, and the later one wins; 0 1 2 3 4 and 5 2 3 4 do not occur before
        assert list(length) == [4, 0, 0] and end[0] == 10
        length, end = sm.scan(np.array([5, 1, 2, 3, 4, 6, 9, 1, 2, 3, 4, 7, 5, 1, 2, 3, 4]), np.array([17]))
        assert length[0] == 5 and end[0] == 5      # the longer match wins over the more recent 1 2 3 4

    def test_correct_run_and_period(self):
        tok = np.array([1, 2, 3, 4, 7, 8, 9, 0, 1, 2, 3, 4, 7, 8, 5, 6])
        pos, end = np.array([12]), np.array([4])
        assert sm.correct(tok, pos, np.array([16]), end, 5)[0] == 2   # 7 8 match, then 9 differs from 5
        assert sm.correct(tok, pos, np.array([13]), end, 5)[0] == 1   # the output ends after one token
        assert sm.correct(tok, pos, np.array([16]), np.array([-1]), 5)[0] == 0
        tok, pos = np.array([3] + [7] * 10 + [4]), np.array([9])   # an overlap repeats with its period
        length, end = sm.scan(tok, pos)
        assert length[0] == 7 and end[0] == 8      # the draft never reads the token at 9 or later
        assert sm.correct(tok, pos, np.array([12]), end, 5)[0] == 2   # 7 7, then 4

    def test_stream_pieces(self):
        chars = lambda texts: [[ord(c) for c in t] for t in texts]  # noqa: E731
        tok, out, stop, first = sm.stream([session(2.0, "why"), session(1.0, "old")], chars, "|tools")
        text = "".join(map(chr, tok))
        assert text.startswith("sys|tools<｜User｜>q<｜Assistant｜><think>old</think>")
        call = ('<｜DSML｜invoke name="terminal">\n<｜DSML｜parameter name="command" string="true">ls</｜DSML｜parameter>'
                '\n<｜DSML｜parameter name="timeout" string="false">5</｜DSML｜parameter>\n</｜DSML｜invoke>')
        target = "</think>\n\n<｜DSML｜tool_calls>\n" + call + "\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>"
        assert "".join(chr(t) for t, o in zip(tok, out) if o) == "old" + target + "why" + target
        assert "copy" not in text and "done<｜end▁of▁sentence｜>" in text   # a copy is prompt, without reasoning
        assert first[0] == 0 and first[-1] == len(tok) // 2
        assert stop[np.flatnonzero(out)[0]] == text.index("<tool_result>")

    def test_report_and_gate(self):
        tok, out = np.array(list(range(10, 40)) * 2), np.arange(60) >= 50   # session 2 repeats session 1
        res = sm.report(tok, out, np.full(60, 60), np.repeat([0, 30], 30))
        assert res["output_positions"] == 10 and res["search_tokens"] == 60
        row = {(r["m"], r["d"]): r for r in res["rows"]}
        # each output position matches 20 to 29 tokens of session 1; the run stops at the end (60)
        assert row[4, 5]["share"] == 1.0 and row[16, 3]["share"] == 1.0 and row[16, 3]["older"] == 1.0
        assert row[16, 5]["mean"] == (6 * 5 + 4 + 3 + 2 + 1) / 10 and row[16, 3]["mean"] == 2.7
        assert row[16, 3]["gate"] == round((2.7 - 1.77) / 2.77, 4) and res["gate_pass"]


if __name__ == "__main__":
    unittest.main()
