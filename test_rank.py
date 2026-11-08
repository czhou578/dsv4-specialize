"""capture.py: the sums protocol of the patched capture container and the rank orders (PLAN_BETTER_RANK.md §4 and
§5)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import capture

L, E = 3, 6  # layers, experts


def sums(count, weight, tokens, steps):
    return {"count": np.array(count, float), "weight": np.array(weight, float),
            "tokens": np.array(tokens, float), "steps": np.array(steps)}


def row(skill, counts, weight=None, tokens=1000):
    r = {"skill": skill, "prompt_tokens": tokens, "unfilled_rows": 0, "counts": counts}
    if weight is not None:
        r.update(weight=weight, wcount=counts, computed_tokens=[tokens] * L, steps=1)
    return r


class SumsTest(unittest.TestCase):
    def test_the_delta_is_the_request_alone_and_the_first_request_counts_from_zero(self):
        before = sums(np.ones((L, E)), np.full((L, E), 0.5), [10] * L, 2)
        after = sums(np.full((L, E), 3), np.full((L, E), 0.75), [14] * L, 3)
        d = capture.sums_delta(before, after)
        self.assertEqual(d["wcount"][0], [2] * E)
        self.assertEqual(d["weight"][1], [0.25] * E)
        self.assertEqual((d["computed_tokens"], d["steps"]), ([4] * L, 1))
        self.assertEqual(capture.sums_delta(None, after)["wcount"][2], [3] * E)

    def test_read_sums_reads_the_npz_and_returns_none_without_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "routing_sums.npz"
            self.assertIsNone(capture.read_sums(path))
            np.savez(path, count=np.ones((L, E)), weight=np.ones((L, E)), tokens=np.ones(L), steps=np.array(7))
            self.assertEqual(int(capture.read_sums(path)["steps"]), 7)


class RankTest(unittest.TestCase):
    def rows(self):
        c1 = np.zeros((L, E), int); c1[:, 0] = 90; c1[:, 1] = 10; c1[:, 2] = 5
        w1 = np.zeros((L, E)); w1[:, 0] = 9.0; w1[:, 1] = 0.5; w1[:, 2] = 4.0  # expert 2: rare but heavy
        c2 = np.zeros((L, E), int); c2[:, 3] = 20; c2[:, 4] = 1
        w2 = np.zeros((L, E)); w2[:, 3] = 2.0; w2[:, 4] = 1.9                   # a small skill that needs expert 4
        return [row("research", c1.tolist(), w1.tolist(), 100000), row("quant", c2.tolist(), w2.tolist(), 60000)]

    def test_the_three_orders(self):
        rows = self.rows()
        by_count = [e["id"] for e in capture.rank(rows, "count")["layers"]["0"]]
        by_weight = [e["id"] for e in capture.rank(rows, "weight")["layers"]["0"]]
        by_skill = [e["id"] for e in capture.rank(rows, "skill_max")["layers"]["0"]]
        self.assertEqual(by_count[:3], [0, 3, 1])      # 90, 20, 10
        self.assertEqual(by_weight[:3], [0, 2, 3])     # 9.0, 4.0, 2.0: the heavy rare expert moves up
        self.assertEqual(by_skill[:3], [0, 3, 4])      # quant gives 4 a share of 1.9/3.9 = 0.49
        r = capture.rank(rows, "weight")
        self.assertEqual((r["order"], r["weighted_segments"]), ("weight", 2))
        self.assertAlmostEqual(r["layers"]["0"][0]["weight_share"], 9 / 17.4, places=5)
        self.assertEqual(r["per_skill"]["quant"]["tokens"], 60000)

    def test_a_small_skill_does_not_set_the_floor(self):
        rows = self.rows()
        by_skill = [e["id"] for e in capture.rank(rows, "skill_max", min_skill_tokens=70000)["layers"]["0"]]
        self.assertEqual(by_skill[:3], [0, 2, 1])      # only research counts: 9, 4, 0.5

    def test_the_floor_order_and_a_nan_slot(self):
        rows = self.rows()
        r = capture.rank(rows, "skill_floor")
        by_floor = [e["id"] for e in r["layers"]["0"]]
        self.assertEqual(by_floor[:3], [0, 3, 2])      # best rank 0: expert 0 (research) and 3 (quant); then 2 by weight
        self.assertEqual(r["layers"]["0"][1]["best_rank"], 0)
        self.assertEqual(r["nan_slots"], 0)
        rows[1]["weight"][0][3] = float("nan")          # the quant row's slot 3 is NaN: estimate from the other row
        r = capture.rank(rows, "weight")
        self.assertEqual(r["nan_slots"], 1)
        self.assertTrue(np.isfinite(r["layers"]["0"][0]["weight"]))

    def test_rows_without_weights_fall_back_to_the_count_order(self):
        rows = [row("research", np.tile([5, 4, 3, 2, 1, 0], (L, 1)).tolist())]
        r = capture.rank(rows, "weight")
        self.assertEqual((r["order"], r["weighted_segments"]), ("count", 0))
        self.assertEqual(r["layers"]["1"][0]["weight"], 0.0)
        with self.assertRaises(ValueError):
            capture.rank(rows, "random")


if __name__ == "__main__":
    unittest.main()
