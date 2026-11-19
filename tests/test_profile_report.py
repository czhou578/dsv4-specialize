"""Tests of profile_report.py (plans/PLAN_FEWER_EXPERTS.md §5)."""

import unittest

import numpy as np

import profile_report as pr

L, R = 43, len(pr.RULES)


def row(skill, kept_per_token, union, tokens=600):
    windows = tokens // 6
    hist = np.zeros((L, 2, 10))
    hist[:, 0, 5] = tokens          # every 5th share in [12.5 %, 15 %)
    hist[:, 1, 1] = tokens          # every 6th share in [2.5 %, 5 %)
    return {"skill": skill, "prof_valid": [tokens] * L, "prof_windows": [windows] * L,
            "prof_pos": np.tile([0.3, 0.2, 0.18, 0.14, 0.13, 0.05], (L, 1)) * tokens,
            "prof_hist": hist.tolist(),
            "prof_kept": np.tile(np.array(kept_per_token) * tokens, (L, 1)).tolist(),
            "prof_union": np.tile(np.array(union) * windows, (L, 1)).tolist()}


KEPT = [6, 5, 4] + [5.5, 5.4, 4.9, 4.8, 4.6, 4.5, 4.45, 4.35, 4.2, 4.1]
UNION = [30, 27, 23] + [29, 28, 27, 26, 26.5, 26, 25, 24, 23, 22]


class ProfileReportTest(unittest.TestCase):
    def test_summary_and_choice(self):
        res = pr.report([row("research", KEPT, UNION), row("quant", KEPT, UNION)])
        a = res["all"]
        assert res["segments"] == 2 and a["tokens"] == 1200
        assert a["position_share"][5] == 0.05
        assert a["below"]["6th<5%"] == 1.0 and a["below"]["5th<10%"] == 0.0 and a["below"]["5th<15%"] == 1.0
        assert a["rules"]["none"]["experts_per_token"] == 6 and a["rules"]["lowest:1"]["window_union"] == 27
        assert a["rules"]["lowest:2"]["union_vs_top5"] == round(23 / 27 - 1, 4)
        # in [4.3, 4.5]: threshold:0.10:4 (4.5), 0.12:3 (4.45) and 0.12:4 (4.35); the last two tie at 0.05 from 4.4,
        # and the smaller window union wins
        assert res["c5"] == "threshold:0.12:4"
        assert set(res["skills"]) == {"quant", "research"} and "3-15" in res["groups"]
        assert "C5 threshold" in pr.markdown(res)

    def test_no_choice_when_union_grows(self):
        union = [30, 20, 18] + [29] * 10          # every threshold rule has a larger union than top-k 5
        assert pr.report([row("research", KEPT, union)])["c5"] is None


if __name__ == "__main__":
    unittest.main()
