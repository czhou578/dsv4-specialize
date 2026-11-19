"""cold_sets.py on synthetic data: 4 layers (3 hash layers) of 10 experts; 8 and 9 are the live drops."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cold_sets as cs
import mask_checkpoint as mc

IDS = list(range(10))
PLAN = {"keep_maps": {"keep_by_layer": {str(i): IDS for i in range(4)},
                      "structured_ranked_by_category": {"tools": {str(i): [0, 1] for i in range(4)},
                                                        "agentic": {str(i): [2, 3] for i in range(4)}}}}
LIVE = {"layers": {str(i): {"drop": [9, 8]} for i in range(4)}}
# layer 3: mandatory 0-3 (0 and 2 are below 0.1%; 3 is at 0.1%), ordinary 4-7, live drops 8 and 9
SHARE3 = [0.0005, 0.2, 0.0009, 0.001, 0.004, 0.005, 0.003, 0.7856, 0.0, 0.0]
RANK = {"layers": {str(i): [{"id": e, "weight_share": (SHARE3 if i == 3 else [0.1] * 10)[e]} for e in IDS]
                   for i in range(4)}}


def row(index, skill, nll, decision, tools):
    return {"session": "s", "segment": 0, "index": index, "skill": skill, "reasoning_nll": nll, "answer_nll": 0.0,
            "decision_ok": decision, "tool_names_ok": tools, "tool_calls": 1}


@mock.patch.object(mc, "MANDATORY_PER_CATEGORY", 2)  # mandatory: 0, 1, 2, 3
class ColdSetsTest(unittest.TestCase):
    def test_the_rules_of_m1_d1_and_md(self):
        sets = cs.cold_sets(PLAN, RANK, LIVE)
        self.assertEqual(sets["r0"][3], [8, 9])
        self.assertEqual(sets["m1"][3], [0, 2, 8, 9])         # mandatory below 0.1%; 3 is not below
        self.assertEqual(sets["d1"][3], [4, 6, 8, 9])         # 0.003 + 0.004 = 0.7%; 0.005 more would pass 1%
        self.assertEqual(sets["md"][3], [0, 2, 4, 6, 8, 9])
        for point in cs.POINTS:                              # the hash layers keep the live drops only
            self.assertEqual([sets[point][i] for i in range(3)], [[8, 9]] * 3)

    def test_a_share_point_is_md_plus_every_expert_below_the_share(self):
        sets = cs.cold_sets(PLAN, RANK, LIVE)
        s = cs.share_sets(PLAN, RANK, LIVE, sets["md"], [0.35])
        self.assertEqual(list(s), ["s35"])
        self.assertEqual(s["s35"][3], [0, 2, 3, 4, 6, 8, 9])   # MD, plus 3 (0.1%) and 6 (0.3%); 5 (0.5%) stays
        self.assertEqual([s["s35"][i] for i in range(3)], [[8, 9]] * 3)
        rows = {r["point"]: r for r in cs.table(sets | s, PLAN, RANK)}
        self.assertEqual((rows["s35"]["experts"], rows["s35"]["mandatory"], rows["s35"]["k_min"]), (5, 3, 3))

    def test_the_table_of_the_points(self):
        rows = {r["point"]: r for r in cs.table(cs.cold_sets(PLAN, RANK, LIVE), PLAN, RANK)}
        self.assertEqual((rows["m1"]["experts"], rows["m1"]["mandatory"], rows["m1"]["lost_mean_pct"]), (2, 2, 0.14))
        self.assertEqual((rows["d1"]["experts"], rows["d1"]["mandatory"], rows["d1"]["lost_max_pct"]), (2, 0, 0.7))
        self.assertEqual([rows[p]["k_min"] for p in cs.POINTS], [8, 6, 6, 4])
        self.assertEqual(rows["md"]["mean_k"], round((3 * 8 + 4) / 4))

    def test_main_writes_four_drop_files_for_mask_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "REAP_K216_PLAN.json").write_text(json.dumps(PLAN))
            (tmp / "rank.json").write_text(json.dumps(RANK))
            (tmp / "live.json").write_text(json.dumps(LIVE))
            with mock.patch("builtins.print"):
                cs.main(["--rank", str(tmp / "rank.json"), "--live", str(tmp / "live.json"), "--base", str(tmp),
                         "--out-dir", str(tmp)])
            md = json.loads((tmp / "cold-md.json").read_text())
            self.assertEqual(sorted(md, key=int), ["0", "1", "2", "3"])
            self.assertEqual(mc.file_drops(PLAN, tmp / "cold-md.json")[3][0], [0, 2, 4, 6, 8, 9])

    def test_the_report_against_r0(self):
        with tempfile.TemporaryDirectory() as tmp:
            r0, pt = Path(tmp) / "r0.jsonl", Path(tmp) / "pt.jsonl"
            r0.write_text("".join(json.dumps(r) + "\n" for r in
                                  [row(1, "quant", 10.0, True, 1), row(2, "earnings", 10.0, True, 1)]))
            pt.write_text("".join(json.dumps(r) + "\n" for r in
                                  [row(1, "quant", 10.2, True, 0), row(2, "earnings", 10.0, False, 1)]))
            lines = cs.report(r0, pt)
            self.assertEqual(lines[0], "targets 2; summed NLL +1.00% against R0; decisions -1; tool names -1; "
                                       "no measurable cost (without the smoke test): no")
            self.assertIn("quant", lines[2])
            self.assertIn("+2.00%", lines[2])
            self.assertEqual(lines[-1], "  tool names 1 -> 0 of 1: s segment 0 index 1 (quant)")


if __name__ == "__main__":
    unittest.main()
