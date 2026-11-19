"""junk_routes.py against a fake server (plans/PLAN_JUNK_CAPTURE.md §4): no model server, no GPU."""

import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import junk_routes as jr


def b64(array):
    buf = io.BytesIO()
    np.save(buf, array)
    return base64.b64encode(buf.getvalue()).decode()


class Fake:
    """A capture server: n prompt tokens (raw: one per character; chat: 10 per message), random top-6 ids per row
    plus the row of the generated token, and a sums file that moves by the counts of the prompt rows."""

    def __init__(self, sums=None, layers=43, move=True, routed=True, graph=0):
        self.calls, self.sums, self.layers, self.move, self.routed = [], sums, layers, move, routed
        self.graph = graph   # a step of this many tokens or fewer replays a CUDA graph: its sums wait for the next step
        if sums:
            self.write(np.zeros((43, 216)), np.zeros((43, 216)), np.zeros(43), 0)

    def write(self, count, weight, tokens, steps):
        with open(f"{self.sums}.tmp", "wb") as f:
            np.savez(f, count=count, weight=weight, tokens=tokens, steps=np.array(steps))
        os.replace(f"{self.sums}.tmp", self.sums)

    def __call__(self, base, path, body, timeout=None):
        self.calls.append((path, body))
        n = len(body["prompt"]) if "prompt" in body else 10 * len(body["messages"])
        rows = np.random.default_rng(n).random((n + 1, self.layers, 216)).argsort(axis=2)[..., :6]
        if self.sums and self.move and n > self.graph:
            with np.load(self.sums) as z:
                count = z["count"] + np.stack([np.bincount(rows[:n, l].ravel(), minlength=216) for l in range(43)])
                self.write(count, z["weight"] + count * 0.25, z["tokens"] + n, int(z["steps"]) + 1)
        choice = {"text": "x", **({"routed_experts": b64(rows)} if self.routed else {})}
        return {"choices": [choice], "usage": {"prompt_tokens": n}}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.prompts = self.d / "prompts.json"
        self.prompts.write_text(json.dumps({"probes": [{"name": "c2", "rule": "digits", "prompt": "x" * 300},
                                                       {"name": "d3", "rule": "none", "prompt": "y" * 40}],
                                            "greedy": {"prompt": "z" * 20}}))

    def test_probes_count_every_round_cold_and_a_rerun_skips_them(self):
        fake, out, sums = Fake(self.d / "sums.npz"), self.d / "cap.jsonl", self.d / "sums.npz"
        with mock.patch.object(jr, "post", fake):
            self.assertEqual(jr.capture("u", out, "m", self.prompts, rounds=2, sums=sums), 0)
            got = rows(out)
            keys = [f"probe-{n}-r{r}" for r in (1, 2) for n in ("c2", "d3", "greedy")]
            self.assertEqual([r["key"] for r in got], keys)
            c2 = got[0]
            ids = np.load(self.d / "cap.jsonl.ids/probe-c2-r1.npy")
            self.assertEqual((ids.shape, ids.dtype, c2["prompt_tokens"], c2["unfilled_rows"]),
                             ((300, 43, 6), np.uint8, 300, 0))
            self.assertTrue(all(sum(layer) == 300 * 6 for layer in c2["counts"]))
            self.assertTrue(all(sum(layer) == jr.TAIL * 6 for layer in c2["tail_counts"]))
            self.assertEqual(c2["last"], ids[-1].tolist())          # the probe position, not the generated token
            self.assertEqual((c2["wcount"], c2["steps"]), (c2["counts"], 1))
            self.assertEqual({p for p, _ in fake.calls}, {"/v1/completions"})
            self.assertTrue(all(b["max_tokens"] == 1 and b["add_special_tokens"] is False for _, b in fake.calls))
            self.assertEqual(len({b["cache_salt"] for _, b in fake.calls}), 6)
            self.assertEqual(jr.capture("u", out, "m", self.prompts, rounds=2, sums=sums), 0)
        self.assertEqual(len(fake.calls), 6)

    def test_a_failed_request_is_recorded_and_retried_on_a_rerun(self):
        out = self.d / "cap.jsonl"
        with mock.patch.object(jr, "post", Fake(layers=42)):
            self.assertEqual(jr.capture("u", out, "m", self.prompts, rounds=1), 3)
        self.assertTrue(all(r["error"].startswith("ValueError") for r in rows(out)))
        with mock.patch.object(jr, "post", Fake()):
            self.assertEqual(jr.capture("u", out, "m", self.prompts, rounds=1), 0)
        self.assertEqual([("error" in r) for r in rows(out)], [True] * 3 + [False] * 3)

    def test_smoke(self):
        sums = self.d / "sums.npz"
        with mock.patch.object(jr, "post", Fake(sums, graph=24)):
            self.assertEqual(jr.smoke("u", "m", sums), 0)     # the smoke prompt runs eagerly
        with mock.patch.object(jr, "post", Fake(routed=False)):
            self.assertEqual(jr.smoke("u", "m"), 1)
        with mock.patch.object(jr, "post", Fake(sums, move=False)), mock.patch.object(jr.time, "sleep"):
            self.assertEqual(jr.smoke("u", "m", sums), 1)

    def test_pilot_rebuilds_the_messages_of_the_last_turn(self):
        ae = SimpleNamespace(SYSTEM="sys", NUDGE="nudge", padding=lambda n: [], tools=SimpleNamespace(SCHEMAS=["t"]),
                             load_task=lambda task: {"prompt": f"do {task}"})
        call = lambda name, result: {"name": name, "arguments": "{}", "result": result}
        turns = [{"content": None, "reasoning": "r0", "calls": [call("list_dir", "a"), call("read_file", "b")]},
                 {"content": "hm", "reasoning": "", "calls": []},
                 {"content": "done", "reasoning": "r2", "calls": [call("report", "Report received.")]}]
        for name, t in (("deepseek-C2-s0", turns), ("deepseek-C2-s1", turns), ("deepseek-D1-s0", [])):
            (self.d / name).mkdir()
            (self.d / name / "transcript.json").write_text(json.dumps({"task": name.split("-")[1], "ctx": 0,
                                                                       "turns": t}))
        msgs = jr.pilot_messages({"task": "C2", "ctx": 0, "turns": turns}, ae)
        roles = ["system", "user", "assistant", "tool", "tool", "assistant", "user"]
        self.assertEqual([m["role"] for m in msgs], roles)
        self.assertEqual([m.get("tool_call_id") for m in msgs[3:5]], ["c0_0", "c0_1"])
        self.assertEqual((msgs[1]["content"], msgs[5]["reasoning"], msgs[6]["content"]), ("do C2", None, "nudge"))
        self.assertNotIn("tool_calls", msgs[5])
        fake, out = Fake(), self.d / "cap.jsonl"
        with mock.patch.object(jr, "agent_eval", lambda: ae), mock.patch.object(jr, "post", fake):
            self.assertEqual(jr.capture("u", out, "m", pilot=self.d), 0)
        got = rows(out)
        self.assertEqual([(r["key"], r["kind"], r["family"], r["prompt_tokens"]) for r in got],
                         [("pilot-C2-s0", "pilot", "C", 70)])
        self.assertEqual((fake.calls[0][0], fake.calls[0][1]["tools"]), ("/v1/chat/completions", ["t"]))

    def test_capture_needs_a_source(self):
        with self.assertRaises(SystemExit):
            jr.main(["capture", "--out", str(self.d / "x.jsonl")])


def arr(extra=(), add=0.0):
    """A (43, 216) list of ones, plus `add` on the experts `extra` in layers 3-42."""
    a = np.ones((43, 216))
    a[3:, list(extra)] += add
    return a.tolist()


def row(key, kind, name, extra=(), add=0.0, last=None):
    a = arr(extra, add)
    fields = {"key": key, "kind": kind, "name": name, "prompt_tokens": 100, "counts": a, "tail_counts": a, "weight": a,
              "last": last or [[200, 201, 202, 203, 204, 205]] * 43}
    return {**fields, **({"family": name.rstrip("0123456789")} if kind == "pilot" else {"round": 1})}


class ReportTest(unittest.TestCase):
    """K 160 removes experts 0-55 in every layer; MD also removes 56-65 in layers 3-42 (400 MD-extra experts)."""

    def setUp(self):
        self.d = d = Path(tempfile.mkdtemp())
        keep = lambda first: {str(l): list(range(first if l >= 3 else 56, 216)) for l in range(43)}
        (d / "k160.json").write_text(json.dumps({"keep": keep(56)}))
        (d / "md.json").write_text(json.dumps({"keep": keep(66)}))
        (d / "cold.json").write_text(json.dumps({str(l): [e for e in range(216) if e not in keep(66)[str(l)]]
                                                 for l in range(43)}))
        nan = arr()
        nan[10][5] = float("nan")
        (d / "bot.jsonl").write_text("\n".join(json.dumps({"counts": arr(), "weight": w, "prompt_tokens": 50})
                                               for w in (arr(), nan)))
        c3_last = [[56, 57, 58, 59, 60, 0]] * 43
        caps = [row("probe-c3-r1", "probe", "c3", (56, 57, 58), 30, c3_last),
                row("probe-c3-r2", "probe", "c3", (56, 57, 58), 30, c3_last),
                row("probe-c2-r1", "probe", "c2"),
                row("pilot-C1-s0", "pilot", "C1", (56, 57, 58), 10),
                row("pilot-C2-s0", "pilot", "C2", (61, 62, 63, 64, 65), 50),   # held out: must not choose X
                row("pilot-D1-s0", "pilot", "D1"), {"key": "pilot-C4-s0", "kind": "pilot", "error": "URLError"}]
        (d / "cap.jsonl").write_text("\n".join(map(json.dumps, caps)))

    def test_report_applies_the_rules_and_select_writes_the_drop_file(self):
        d = self.d
        res = jr.report(d / "cap.jsonl", d / "out", d / "k160.json", d / "md.json", d / "bot.jsonl")
        g = res["groups"]
        self.assertEqual(list(g), ["probe c3", "probe c2", "pilot C", "pilot D", "pilot all", "pilot held-out", "bot"])
        self.assertAlmostEqual(g["bot"]["picks"]["md"], 400 / (43 * 216))
        self.assertAlmostEqual(g["bot"]["picks"]["k160_hash"], 56 / 216)
        self.assertEqual(g["bot"]["weight_rows"], 1)                       # the row with a NaN slot is left out
        self.assertEqual((g["probe c3"]["rows"], g["probe c3"]["last_k160"], g["probe c3"]["last_md"]), (2, 43, 200))
        self.assertEqual(res["missing"], ["pilot-C4-s0"])
        self.assertEqual((res["rules"]["G1"], res["rules"]["G2"], res["rules"]["G3"]), (True, False, True))
        x = res["x"]["experts"]
        # C1 + D1 per layer: 12 of 462 on each of 56-58; 84 x 12 >= 2000 / 2
        self.assertEqual((len(x), res["x"]["basis"], res["x"]["pool"]), (84, "weight", 400))
        self.assertTrue(all(l >= 3 and e in (56, 57, 58) for l, e in x))
        self.assertIn("build MD + X", res["decision"])
        self.assertIn("G1 (c3", (d / "out/report.md").read_text())
        self.assertEqual(jr.select(d / "out/report.json", d / "cold.json", d / "mdx.json"), 0)
        drops = json.loads((d / "mdx.json").read_text())
        self.assertEqual(len(drops), 43)
        self.assertNotIn(58, drops["42"])
        self.assertIn(65, drops["42"])
        self.assertEqual(sum(map(len, drops.values())), 56 * 43 + 400 - 84)

    def test_x_stops_at_the_cap_and_select_refuses_a_kept_expert(self):
        rows = [row("pilot-D1-s0", "pilot", "D1", (56, 57, 58), 10)]
        r160, rmd = jr.removed_sets(self.d / "k160.json", self.d / "md.json")
        self.assertEqual(len(jr.choose_x(rows, rmd, cap=5)["experts"]), 5)
        (self.d / "bad.json").write_text(json.dumps({"x": {"experts": [[20, 100]]}}))
        with self.assertRaises(ValueError):
            jr.select(self.d / "bad.json", self.d / "cold.json", self.d / "x.json")


if __name__ == "__main__":
    unittest.main()
