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


def removed(block, skip):
    """The probe gets no sums for a block that the skip set removes: L removes A, F and L; A or F removes itself and L."""
    for b in [x for x in skip.split(",") if x]:
        if b[1:] == block[1:] and (b[0] == "L" or block[0] in (b[0], "L")):
            return True
    return False


class FakeProbe:
    """The probe boot: each chat request reads skip.json, adds its sums to layer_probe.npz (as the patch does), and
    returns prompt logprobs whose NLL is ``base`` plus the rises of the skip set."""

    def __init__(self, folder, bi=None, rise=None, base=0.5, cached=0, runs=None):
        self.dir, self.bi, self.rise, self.base = Path(folder), bi or {}, rise or {}, base
        self.cached, self.runs = cached, runs
        self.sums = {k: np.zeros((43, 3)) for k in ("cos", "rel", "tokens")}
        self.loads, self.flush, self.chats, self.gram = 0, None, 0, None

    @staticmethod
    def render(messages):
        toks = [1, 2, 3] if messages and messages[0]["role"] == "system" else []
        return toks + [100 + k for m in messages if m["role"] != "system" for k in range(len(json.dumps(m)) // 10)]

    def __call__(self, base, path, body, timeout=None):
        if path == "/tokenize":
            if "prompt" in body:
                return {"tokens": [{"<think>": THINK, "</think>": END}[body["prompt"]]]}
            return {"tokens": self.render(body["messages"])}
        cfg = json.loads((self.dir / "skip.json").read_text())
        self.loads += 1
        if cfg["gram_flush"] != self.flush:
            np.savez(self.dir / "layer_gram.npz", flush=np.array(cfg["gram_flush"]),
                     **{f"xtx_{n}": np.eye(2) for n in self.gram or []})
            self.flush = cfg["gram_flush"]
        self.gram = cfg["gram"]
        skip = self.runs if self.runs is not None else cfg["skip"]
        n = len(self.render(body["messages"])) - cfg["count_from"]
        for layer in range(43):
            for c, kind in enumerate(ls.KINDS):
                b = f"{kind}{layer}"
                if cfg["sums"] and not removed(b, skip):
                    bi = self.bi.get(b, 0.5)
                    self.sums["cos"][layer, c] += (1 - bi) * n
                    self.sums["rel"][layer, c] += 2 * bi * n
                    self.sums["tokens"][layer, c] += n
        self.chats += 1
        np.savez(self.dir / "layer_probe.npz", **self.sums, steps=np.array(self.chats), skip=np.array(skip or "none"),
                 loads=np.array(self.loads), error=np.array(""))
        resp = {"usage": {"prompt_tokens_details": {"cached_tokens": self.cached}}}
        if "prompt_logprobs" in body:
            lp = -(self.base + sum(self.rise.get(b, 0.0) for b in skip.split(",") if b))
            toks = [THINK] + [7] * 4 + [END] + [8] * 5
            resp["prompt_logprobs"] = [None] + [{str(t): {"logprob": 0.0 if t in (THINK,) else lp, "rank": 1,
                                                          "decoded_token": "x"}} for t in toks]
        return resp


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


class ClientTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = data(self.tmp.name, [session("a", "research"), session("b", "quant")])
        self.items = [["a", 0, 3, 0], ["b", 0, 5, 0], ["a", 0, 7, 0]]

    def tearDown(self):
        self.tmp.cleanup()

    def client(self, fake):
        patch = mock.patch.object(ls, "post", fake)
        patch.start()
        self.addCleanup(patch.stop)
        return ls.Client("http://x", "m", self.d, data=self.d)

    def test_bi_reads_the_blocks_and_counts_the_shared_prefix_once(self):
        fake = FakeProbe(self.d, bi={"L17": 0.01, "F17": 0.004, "A4": 0.002})
        c = self.client(fake)
        table = ls.bi_table(c.bi(self.items, "F21"))
        self.assertAlmostEqual(table["L17"][0], 0.01)
        self.assertAlmostEqual(table["F17"][0], 0.004)
        self.assertNotIn("F21", table)
        self.assertNotIn("L21", table)
        self.assertIn("A21", table)
        lengths = [len(fake.render(c.body(s, g, i, False)["messages"])) for s, g, i, _ in self.items]
        self.assertEqual(c.bi(self.items)["tokens"][17][0], sum(lengths) - 2 * 3)   # the prefix [1, 2, 3] once

    def test_screen_gives_the_nll_of_the_set(self):
        c = self.client(FakeProbe(self.d, rise={"L17": 0.01, "F9": 0.002}))
        r = c.screen(self.items)
        s = c.screen(self.items, "F9,L17")
        self.assertAlmostEqual(r["nll"], 0.5)
        self.assertAlmostEqual(100 * (s["nll"] / r["nll"] - 1), 2.4)
        self.assertEqual(r["scored_tokens"], 3 * 10)

    def test_a_wrong_set_or_cached_tokens_stop_the_client(self):
        with self.assertRaisesRegex(RuntimeError, "runs L5"):
            self.client(FakeProbe(self.d, runs="L5")).screen(self.items, "L17")
        with self.assertRaisesRegex(RuntimeError, "cached"):
            self.client(FakeProbe(self.d, cached=5)).bi(self.items)

    def test_gram_flushes_the_sums_to_the_out_file(self):
        out = Path(self.tmp.name) / "gram.npz"
        c = self.client(FakeProbe(self.d))
        c.gram(self.items, ["L17", "F9"], out)
        with np.load(out) as z:
            self.assertEqual((int(z["flush"]), sorted(z.files)), (c.file.flush, ["flush", "xtx_F9", "xtx_L17"]))


class FakeClient:
    """The client with a model of the screen: each block adds its rise to the NLL; a fake clock moves 180 s per
    screen and 480 s per BI pass."""

    def __init__(self, bi, rise, base=0.5):
        self.bi_of, self.rise, self.base, self.now, self.calls, self.grams = bi, rise, base, 0.0, [], []

    def clock(self):
        return self.now

    def screen(self, items, skip=""):
        self.now += 180
        self.calls.append(("screen", skip))
        return {"nll": self.base * (1 + sum(self.rise.get(b, 5.0) for b in ls.blocks_of(skip)) / 100),
                "scored_tokens": 10}

    def bi(self, items, skip="", gram=()):
        self.now += 480
        self.calls.append(("bi", skip))
        cos, tok = np.zeros((43, 3)), np.zeros((43, 3))
        for layer in range(43):
            for c, kind in enumerate(ls.KINDS):
                if not removed(f"{kind}{layer}", skip):
                    cos[layer, c] = 100 * (1 - self.bi_of.get(f"{kind}{layer}", 0.2 + layer / 100))
                    tok[layer, c] = 100
        return {"cos": cos.tolist(), "rel": np.zeros((43, 3)).tolist(), "tokens": tok.tolist()}

    def gram(self, items, names, out):
        self.now += 600
        self.grams.append(names)


def greedy(tmp, fake, deadline=1e9, stop=1.0):
    return ls.Greedy(fake, {"screen": [], "probe": []}, Path(tmp) / "log.jsonl", deadline, stop, clock=fake.clock)


class GreedyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        bi = {f"{k}{l}": v for k, l, v in [("L", 30, 0.01), ("F", 30, 0.005), ("A", 30, 0.001), ("F", 12, 0.006),
                                           ("L", 12, 0.02), ("A", 2, 0.0001), ("F", 41, 0.0001)]}
        rise = {"F30": 0.2, "L30": 0.9, "F12": 0.3, "L12": 1.5, **{f"A{l}": 0.05 * (l % 7) for l in range(3, 40)}}
        self.fake = FakeClient(bi, rise)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_rounds_of_section_7(self):
        greedy(self.tmp.name, self.fake).run()
        keys = [k for k, s in self.fake.calls]
        screens = [s for k, s in self.fake.calls if k == "screen"]
        r1 = [s for s in screens[1:29]]
        self.assertEqual(sorted(b[0] for b in r1), ["A"] * 8 + ["F"] * 10 + ["L"] * 10)
        self.assertEqual(r1[:3], ["L30", "F30", "A30"])                   # the lowest BI of each kind first
        self.assertFalse({"A2", "F41"} & set(r1))                          # protected layers are not candidates
        a_rises = sorted((self.fake.rise[b], b) for b in r1 if b[0] == "A")
        self.assertEqual(screens[29], ls.canon(b for _, b in a_rises[:6]))   # the 6 A blocks with the lowest rises
        self.assertEqual(self.fake.grams, [["F30", "F12", "L30"]])        # the 3 lowest L or F rises
        self.assertEqual(keys.count("bi"), 3)                              # no skip, then the sets of 1 and 2
        self.assertEqual([s for k, s in self.fake.calls if k == "bi"][1], "F30")
        r2 = screens[30:36]
        self.assertTrue(all("F30" in s and len(ls.blocks_of(s)) == 2 for s in r2))
        self.assertTrue(all("F12,F30" in s and len(ls.blocks_of(s)) == 3 for s in screens[36:42]))
        self.assertEqual(screens[-1], "")                                   # R2
        report = ls.report(json.loads(x) for x in (Path(self.tmp.name) / "log.jsonl").read_text().splitlines())
        self.assertIn("rises by +0.20% (at most 0.25%)", report)
        self.assertIn("the screen is stable", report)
        self.assertIn("Rank correlation of BI and the rise", report)

    def test_the_stop_rule_and_the_deadline(self):
        self.fake.rise.update({"F30": 1.2, "F12": 1.3, "L30": 2.0})
        greedy(self.tmp.name, self.fake, stop=1.0).run()
        self.assertEqual(self.fake.grams, [])
        self.assertEqual([k for k, s in self.fake.calls].count("bi"), 1)
        self.assertIn("(more than 1%): stop the memory track",
                      ls.report(json.loads(x) for x in (Path(self.tmp.name) / "log.jsonl").read_text().splitlines()))
        fake = FakeClient({}, {})
        with tempfile.TemporaryDirectory() as tmp:   # R and BI at 660 s, 12 screens to 2820 s, then R2 at 3000 s
            greedy(tmp, fake, deadline=3000).run()
        screens = [s for k, s in fake.calls if k == "screen"]
        self.assertEqual((screens[0], screens[-1], len(screens), fake.now), ("", "", 1 + 12 + 1, 3000))

    def test_a_rerun_skips_the_logged_steps(self):
        greedy(self.tmp.name, self.fake).run()
        again = FakeClient({}, {})
        greedy(self.tmp.name, again).run()
        self.assertEqual(again.calls, [])


class FitTest(unittest.TestCase):
    def test_fit_finds_a_known_map(self):
        g = np.random.default_rng(0)
        x = g.normal(size=(4000, 16))
        m = 0.1 * g.normal(size=(16, 16))
        d = x @ m + 0.001 * g.normal(size=(4000, 16))
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(Path(tmp, "gram.npz"), xtx_F24=x.T @ x, xtd_F24=x.T @ d, rows_F24=np.array(4000.0))
            notes = ls.fit(Path(tmp, "gram.npz"), ["F24"], Path(tmp, "heal.npz"))
            with np.load(Path(tmp, "heal.npz")) as z:
                self.assertEqual((z.files, z["heal_F24"].dtype), (["heal_F24"], np.float32))
                np.testing.assert_allclose(z["heal_F24"], m, atol=1e-3)
        self.assertAlmostEqual(notes["F24"]["removed_per_row"], float((x @ m).var(axis=0).sum() + 0), delta=0.3)


if __name__ == "__main__":
    unittest.main()
