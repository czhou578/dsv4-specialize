"""code_rank.py on synthetic captures (plans/PLAN_CODE_RANK.md §5): no server, no GPU."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import code_rank as cr


def arr(extra=(), add=0.0, layers=range(3, 43), kept_only=False):
    """Ones (on the experts that MD keeps only, with kept_only), plus `add` on `extra` in `layers`."""
    a = np.ones((43, 216))
    if kept_only:
        a[:, :56] = 0
        a[3:, 56:66] = 0
    for l in layers:
        a[l, list(extra)] += add
    return a.tolist()


class RankTest(unittest.TestCase):
    """K 160 removes 0-55 in every layer; MD also removes 56-65 in layers 3-42. Pool: 66 x 40 = 2640 experts."""

    def setUp(self):
        self.d = d = Path(tempfile.mkdtemp())
        keep = lambda first: {str(l): list(range(first if l >= 3 else 56, 216)) for l in range(43)}
        (d / "k160.json").write_text(json.dumps({"keep": keep(56)}))
        (d / "md.json").write_text(json.dumps({"keep": keep(66)}))
        (d / "cold.json").write_text(json.dumps({str(l): [e for e in range(216) if e not in keep(66)[str(l)]]
                                                 for l in range(43)}))
        nan = arr()
        nan[10][5] = float("nan")
        (d / "bot.jsonl").write_text("\n".join(json.dumps({"counts": arr(), "weight": w}) for w in (arr(), nan)))
        # coding: experts 60 and 61 hot in every layer 3-42; the probes: c3 on 60, c2 on 61 and 62
        code = [{"counts": arr(), "weight": arr((60, 61), 20)}, {"error": "URLError"}]
        (d / "code.jsonl").write_text("\n".join(map(json.dumps, code)))
        pilot = [{"kind": "pilot", "name": "R1", "weight": arr((60, 61), 5)},
                 {"kind": "probe", "name": "c3", "weight": arr((60,), 40, kept_only=True)},
                 {"kind": "probe", "name": "c2", "weight": arr((61, 62), 40, kept_only=True)}]
        (d / "pilot.jsonl").write_text("\n".join(map(json.dumps, pilot)))

    def run_rank(self, **kw):
        d = self.d
        return cr.rank(d / "code.jsonl", d / "pilot.jsonl", d / "bot.jsonl", d / "k160.json", d / "md.json",
                       d / "cold.json", d / "out", **kw)

    def test_the_order_the_curve_and_the_candidates(self):
        res = self.run_rank()
        self.assertEqual((res["coding_rows"], res["basis"], res["pool"]), (1, "weight", 2640))
        self.assertAlmostEqual(res["curve"][0]["bot"], (56 * 43 + 400) / (43 * 216))   # the NaN bot row is left out
        cands = {x["name"]: x for x in res["candidates"]}
        # 60 and 61 lead the order, layer by layer: (3, 60), (3, 61), (4, 60), ... c3 (1600 of 8080 dropped at MD)
        # needs 22 experts 60 back: n = 43. c2 needs 62s too, which come after the cap: not reached, n = 486.
        self.assertEqual((cands["code1"]["added"], cands["code1"]["reached"]), (43, True))
        self.assertEqual((cands["code2"]["added"], cands["code2"]["reached"]), (486, False))
        drops = json.loads(Path(cands["code1"]["drop"]).read_text())
        self.assertEqual(sum(map(len, drops.values())), 56 * 43 + 400 - 43)
        self.assertNotIn(60, drops["3"])
        self.assertIn(60, drops["42"])
        self.assertEqual(cands["code1"]["k"], round((43 * 216 - (56 * 43 + 400 - 43)) / 43))
        self.assertIn("NOT REACHED", (self.d / "out/code-rank.md").read_text())

    def test_the_cap_marks_a_candidate_not_reached(self):
        res = self.run_rank(n_max=3)
        self.assertTrue(all(x["added"] <= 3 for x in res["candidates"]))
        self.assertEqual(len(res["candidates"]), 1)                   # both hit the cap: one folder only
        self.assertFalse(res["candidates"][0]["reached"])


if __name__ == "__main__":
    unittest.main()
