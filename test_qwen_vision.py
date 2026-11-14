"""Tests of qwen_vision.py with a fake server (no model, no Chromium)."""

import json
from unittest import mock

import qwen_facts
import qwen_vision as qv


def test_every_image_has_three_questions_with_answers():
    assert len(qv.IMAGES) == 12 and {k for _, k, *_ in qv.IMAGES} == {"code", "chart", "ui", "diagram"}
    for name, _, w, h, html, qa in qv.IMAGES:
        assert len(qa) == 3 and all(q.endswith("?") and a for q, a in qa), name
        assert w >= 700 and h >= 300 and "<" in html


def test_answer_sends_the_image_and_resumes(tmp_path):
    (tmp_path / "a.png").write_bytes(b"PNG")
    (tmp_path / "questions.json").write_text(json.dumps([
        {"image": "a.png", "kind": "code", "question": "Which port?", "answer": "8889"},
        {"image": "a.png", "kind": "ui", "question": "Which label?", "answer": "Save"}]))
    seen = []

    def fake(base, path, body, timeout=None):
        seen.append(body["messages"][0]["content"])
        return {"choices": [{"message": {"content": "It is 8889."}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 900, "completion_tokens": 9}}
    with mock.patch.object(qwen_facts, "post", fake):
        qv.answer("u", "m", tmp_path, tmp_path / "out.jsonl", limit=1)
        qv.answer("u", "m", tmp_path, tmp_path / "out.jsonl")   # resumes: only the second question is sent
    assert len(seen) == 2 and seen[0][0]["image_url"]["url"] == "data:image/png;base64,UE5H"
    assert seen[0][1]["text"] == "Which port? Answer in one short sentence."
    rows = qwen_facts.load(tmp_path / "out.jsonl")
    assert [r["question"] for r in rows] == ["[a.png] Which port?", "[a.png] Which label?"] and rows[0]["topic"] == "code"


def test_report_counts_per_kind_and_lists_differences(tmp_path, capsys):
    g = lambda q, kind, grade: json.dumps({"question": q, "topic": kind, "grade": grade}) + "\n"
    (tmp_path / "a.jsonl").write_text(g("q1", "code", "CORRECT") + g("q2", "ui", "CORRECT"))
    (tmp_path / "b.jsonl").write_text(g("q1", "code", "CORRECT") + g("q2", "ui", "INCORRECT"))
    qv.report([tmp_path / "a.jsonl", tmp_path / "b.jsonl"])
    out = capsys.readouterr().out
    assert "correct 2 of 2; code 1 of 1, ui 1 of 1" in out and "correct 1 of 2; code 1 of 1, ui 0 of 1" in out
    assert "differs: q2: CORRECT / INCORRECT" in out
