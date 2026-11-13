"""layer_screen.py: the data pick, the skip file, the BI pass and the screen against a fake probe boot
(plans/PLAN_FEWER_LAYERS.md §6.5)."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import layer_screen as ls

THINK, END = 90, 91


def session(sid, skill, split="calibration", source="discord", turns=4, system="be the bot"):
    seg = []
    for k in range(turns):
        seg += [{"role": "user", "content": f"q{k}" * 20},
                {"role": "assistant", "content": "a" * (50 * (k + 1)), "reasoning_content": "r", "target": True}]
    return {"id": sid, "skill": skill, "split": split, "source": source, "system_prompt": system, "segments": [seg]}


def data(folder, sessions):
    folder = Path(folder)
    (folder / "sessions.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sessions))
    (folder / "tools.json").write_text(json.dumps({"tools": []}))
    return folder


class PickTest(unittest.TestCase):
    def test_sets_share_no_segment_and_skip_held_out_and_cli(self):
        sessions = [session(f"s{k}", ["research", "quant", "earnings"][k % 3], turns=6) for k in range(12)]
        sessions += [session("held", "research", split="held_out"), session("cli", "quant", source="cli")]
        sets = ls.pick(sessions, [], screen_tokens=300, probe_tokens=600, screen_max=150, probe_max=10_000)
        screen, probe = sets["screen"], sets["probe"]
        segs = [tuple(r[:2]) for r in screen + probe]
        self.assertEqual(len(segs), len(set(segs)))                       # one target per segment, no overlap
        self.assertFalse({r[0] for r in screen + probe} & {"held", "cli"})
        self.assertTrue(all(r[3] <= 150 for r in screen) and sum(r[3] for r in screen) >= 300)
        self.assertGreaterEqual(sum(r[3] for r in probe), 600)
        self.assertEqual({sessions[int(r[0][1:])]["skill"] for r in screen[:3]}, {"research", "quant", "earnings"})
        self.assertEqual({r[2] for r in screen}, {3})                     # the longest turn under the cap (index 5 is over)


class SkipFileTest(unittest.TestCase):
    def test_each_write_gets_a_new_mtime_and_the_flush_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ls.SkipFile(tmp)
            start, seen = f.flush, []
            for k in range(5):
                f.write("L5", count_from=k, flush=k == 3)
                seen.append(Path(tmp, "skip.json").stat().st_mtime_ns)
            self.assertEqual(len(set(seen)), 5)
            cfg = json.loads(Path(tmp, "skip.json").read_text())
            self.assertEqual((cfg["skip"], cfg["count_from"], cfg["gram_flush"]), ("L5", 4, start + 1))

    def test_check_refuses_a_wrong_set_an_error_and_cached_tokens(self):
        ok = {"error": np.array(""), "skip": np.array("L5")}
        ls.check(ok, "L5", {"usage": {}})
        for after, skip, resp in [(None, "L5", {}), ({**ok, "error": np.array("bad")}, "L5", {}), (ok, "F5", {}),
                                  (ok, "L5", {"usage": {"prompt_tokens_details": {"cached_tokens": 3}}})]:
            with self.assertRaises(RuntimeError):
                ls.check(after, skip, resp)


if __name__ == "__main__":
    unittest.main()
