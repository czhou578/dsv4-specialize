"""alloc_k.py on synthetic data: 5 layers (3 hash layers) of 10 experts, K between 5 and 10."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import alloc_k as ak
import mask_checkpoint as mc

# weight share per id. Ids 0 and 9 are mandatory, so the experts drop in the order 8, 7, 6, 5, 4.
# Layer 3 is flat: 0.04, 0.04, 0.06, 0.06, 0.08. Layer 4 is peaked: 0.03 (4 times), 0.05, 0.05.
SHARES = {3: [0.3, 0.2, 0.1, 0.08, 0.08, 0.06, 0.06, 0.04, 0.04, 0.04],
          4: [0.47, 0.2, 0.1, 0.05, 0.05, 0.03, 0.03, 0.03, 0.03, 0.01]}
IDS = list(range(10))
PLAN = {"keep_maps": {"keep_by_layer": {str(i): IDS for i in range(5)},
                      "ranked_by_layer": {str(i): IDS for i in range(5)},
                      "structured_ranked_by_category": {"tools": {str(i): [9] for i in range(5)},
                                                        "agentic": {str(i): [0] for i in range(5)}}}}
RANK = {"layers": {str(i): [{"id": e, "weight_share": SHARES.get(i, [0.1] * 10)[e]}
                            for e in sorted(IDS, key=lambda e: -SHARES.get(i, [0.1] * 10)[e])]
                   for i in range(5)}}


@mock.patch.object(mc, "MANDATORY_PER_CATEGORY", 1)
@mock.patch.multiple(ak, HASH_K=8, K_MIN=5)
class AllocTest(unittest.TestCase):
    def test_the_rules_differ_on_a_flat_and_a_peaked_layer(self):
        self.assertEqual(ak.allocate(PLAN, RANK, 8, "global")[0], [8, 8, 8, 10, 6])  # 4 × 0.03
        self.assertEqual(ak.allocate(PLAN, RANK, 8, "even")[0], [8, 8, 8, 8, 8])  # 0.08 and 0.06

    def test_the_sum_is_exact_and_the_bounds_hold(self):
        for target in (7, 8):
            for rule in ak.RULES:
                klist = ak.allocate(PLAN, RANK, target, rule)[0]
                self.assertEqual(sum(klist), 5 * target)
                self.assertEqual(klist[:3], [8, 8, 8])
                self.assertTrue(all(5 <= k <= 10 for k in klist[3:]), klist)
        # without the floor, layer 4 drops a sixth expert (0.05) before layer 3 drops its fourth (0.06)
        self.assertEqual(ak.allocate(PLAN, RANK, 7, "global")[0][3:], [6, 5])
        for target in (6, 9):  # 14 drops, room for 10; -1 drops
            with self.assertRaises(ValueError):
                ak.allocate(PLAN, RANK, target, "global")

    def test_even_fills_the_sum_with_the_smallest_next_share(self):
        self.assertEqual(ak.even({3: [0.05, 0.05], 4: [0.05, 0.05]}, 3), {3: 2, 4: 1})
        self.assertEqual(ak.even({3: [0.0, 0.1]}, 0), {3: 0})  # a share of 0 does not drop by itself

    def test_mask_checkpoint_drops_the_counted_experts_and_keeps_the_mandatory_experts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "REAP_K216_PLAN.json").write_text(json.dumps(PLAN))
            rank = tmp / "rank.json"
            rank.write_text(json.dumps(RANK))
            for rule in ak.RULES:
                out = tmp / f"klist-{rule}.json"
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed):
                    ak.main(["--target", "8", "--rule", rule, "--rank", str(rank), "--base", str(tmp),
                             "--out", str(out)])
                self.assertIn("global: mean K 8; layers 3-4: K 6-10", printed.getvalue())
                self.assertIn("even: mean K 8; layers 3-4: K 8-8", printed.getvalue())
                klist = json.loads(out.read_text())
                self.assertEqual(klist, ak.allocate(PLAN, RANK, 8, rule)[0])
                with mock.patch.object(mc, "edits", return_value=({}, {})):
                    record = mc.build(tmp, klist, tmp / "mask", dry_run=True, rank=rank)
                for layer, item in record["layers"].items():
                    drop = set(item["drop"])
                    self.assertEqual(len(drop), 10 - klist[layer])
                    self.assertFalse(drop & {0, 9})
                    share = SHARES.get(layer, [0.1] * 10)
                    kept = set(IDS) - drop - {0, 9}
                    self.assertLessEqual(max((share[e] for e in drop), default=0), min(share[e] for e in kept))


if __name__ == "__main__":
    unittest.main()
