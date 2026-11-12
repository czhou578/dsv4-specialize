"""Tests of review_report.py on synthetic judge rows and run records (no model, no run folder)."""

import unittest

import review_report as rr


def rows(*verdicts, pid="p"):
    return [{"id": f"{pid}{i}", "verdict": v} for i, v in enumerate(verdicts)]


class ReviewReportTest(unittest.TestCase):
    def test_the_fail_line_is_the_sign_test_table_of_the_final_review(self):
        # PLAN_FINAL_REVIEW.md §2.3 and §7.1: 15 decided -> 12, 40 -> 26, 50 -> 32, 60 -> 37, 70 -> 43, 80 -> 48
        for decided, line in ((0, 1), (15, 12), (20, 15), (40, 26), (50, 32), (60, 37), (70, 43), (76, 46), (80, 48)):
            self.assertEqual(rr.fail_line(decided), line, decided)

    def test_the_test_fails_on_losses_and_a_control_fails_in_both_directions(self):
        n, line, failed = rr.verdict(rows(*["loss"] * 12, *["win"] * 3, "tie", "error"))
        self.assertEqual((n["loss"], n["win"], n["tie"], n["error"], line, failed), (12, 3, 1, 1, 12, True))
        self.assertFalse(rr.verdict(rows(*["win"] * 12, *["loss"] * 3))[2])
        self.assertTrue(rr.verdict(rows(*["win"] * 12, *["loss"] * 3), two_sided=True)[2])
        self.assertFalse(rr.verdict(rows(*["loss"] * 11, *["win"] * 4), two_sided=True)[2])

    def test_a_weak_prompt_loses_against_every_base(self):
        test = {"md-1": [{"id": "a", "verdict": "loss"}, {"id": "a", "verdict": "loss"},
                         {"id": "b", "verdict": "loss"}, {"id": "b", "verdict": "tie"},
                         {"id": "c", "verdict": "loss"}],   # c: one base only, not judged against the other
                "md-2": [{"id": "b", "verdict": "loss"}, {"id": "b", "verdict": "loss"}]}
        self.assertEqual(rr.weak(test, ["live-1", "live-2"]), [("md-1", "a"), ("md-2", "b")])

    def test_the_skills_count_wins_losses_and_ties(self):
        got = rr.by_skill(rows("win", "loss", "tie", "error", "win"), {"p0": "quant", "p1": "quant", "p2": "quant"})
        self.assertEqual(got, {"?": [1, 0, 1], "quant": [1, 1, 1]})

    def test_the_times_use_the_prompts_that_exit_0_in_every_run(self):
        def rec(skill, seconds, exit=0):
            return {"skill": skill, "seconds": seconds, "exit": exit}
        done = {"live-1": {"a": rec("quant", 100), "b": rec("research", 200), "c": rec("quant", 50)},
                "live-2": {"a": rec("quant", 120), "b": rec("research", 220), "c": rec("quant", 50)},
                "md-1": {"a": rec("quant", 130), "b": rec("research", 210), "c": rec("quant", 50, exit=1)},
                "md-2": {"a": rec("quant", 110), "b": rec("research", 210)}}
        got = rr.times(done, ["live-1", "live-2"], ["md-1", "md-2"])
        self.assertEqual(got["quant"], (110, 120, 9.1))
        self.assertEqual(got["research"], (210, 210, 0.0))
        self.assertEqual(got["all"], (320, 330, 3.1))


if __name__ == "__main__":
    unittest.main()
