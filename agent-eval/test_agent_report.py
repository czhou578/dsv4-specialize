"""Tests of agent_report.py on synthetic rows."""

import json

import pytest

import agent_eval as ae
import agent_report as rep


def row(task, seed, result, model="qwen", family="L", secs=10.0):
    return {"task": task, "seed": seed, "result": result, "model": model, "family": family, "seconds": secs,
            "events": {"H1": 0, "H2": int(result == "hallucination"), "H3": 0, "H4": 0}, "prompt_tokens_first": 1000}


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(ae, "OUT", tmp_path)
    monkeypatch.setattr(ae, "S", tmp_path)

    def write(name, rows, low=5000):
        (tmp_path / name).mkdir()
        (tmp_path / name / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        meta = {"mix": {"qwen": 1}, "ctx": 0, "seeds": 2, "start": 0, "end": 1, "lowest_available_mb": low}
        (tmp_path / name / "point.json").write_text(json.dumps(meta))
    return write


def test_summary():
    s = rep.summary([row("L1", 0, "pass"), row("L1", 1, "hallucination", secs=30), row("L2", 0, "wrong")])
    assert s["runs"] == 3 and round(s["pass"], 1) == 33.3 and s["events"]["H2"] == 1 and s["median_s"] == 10
    assert s["bursts"] == 0 and s["loops"] == 0   # rows before 2026-11-15 have no call stats
    s = rep.summary([row("L1", 0, "pass") | {"max_calls_turn": 57, "repeat_calls": 9}])
    assert s["bursts"] == 1 and s["loops"] == 1


def test_diff_uses_common_tasks():
    ref = [row("L1", 0, "pass"), row("L1", 1, "pass"), row("L2", 0, "pass"), row("L9", 0, "wrong")]
    new = [row("L1", 0, "pass"), row("L1", 1, "hallucination"), row("L2", 0, "pass")]
    d, lo, hi = rep.diff(new, ref, "pass")
    assert d == -25.0 and lo <= d <= hi   # L1: 50 points down, L2: no change; L9 is not common
    assert rep.diff(new, ref, "hallucination")[0] == 25.0


def test_table_and_gates(out):
    out("Q1", [row(f"L{i}", s, "pass") for i in range(1, 9) for s in range(2)])
    out("Q4", [row(f"L{i}", s, "hallucination" if i == 1 else "pass") for i in range(1, 9) for s in range(2)],
        low=2500)
    text = rep.table("Q4", "Q1")
    assert "| qwen | all | 16 | 87.5% | 12.5% |" in text
    # G1 and G2 fail (12.5 points), G3 passes, G4 fails at 2500 MB
    assert "| qwen | +12.5 (+0.0 to +37.5) | -12.5 (-37.5 to +0.0) | FAIL | FAIL | pass | FAIL |" in text


def test_show(tmp_path):
    tr = {"task": "L1", "model": "qwen", "seed": 0, "ctx": 0, "result": "pass", "stop": "report", "seconds": 3,
          "events": {}, "report": {"status": "done"}, "turns": [{"finish": "tool_calls", "usage": {}, "seconds": 1,
          "reasoning": "hmm", "content": None, "calls": [{"name": "grep", "arguments": "{}", "error": "bad",
                                                           "result": "error"}]}]}
    (tmp_path / "transcript.json").write_text(json.dumps(tr))
    text = rep.show(tmp_path)
    assert "[reasoning] hmm" in text and "[call] grep {}  INVALID: bad" in text
