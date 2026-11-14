"""Tests of qwen_facts.py with a fake server (no model)."""

import csv
import json
from unittest import mock

import qwen_facts as qf


def rows(n_science, n_other):
    meta = lambda t: repr({"topic": t, "answer_type": "Person"})
    return ([{"metadata": meta(qf.SCIENCE), "problem": f"s{i}?", "answer": f"S{i}"} for i in range(n_science)] +
            [{"metadata": meta("Art"), "problem": f"o{i}?", "answer": f"O{i}"} for i in range(n_other)])


def test_the_pick_is_fixed_and_split_by_topic(tmp_path):
    picked = qf.pick(rows(150, 100))
    assert len(picked) == 200 and sum(r["problem"].startswith("s") for r in picked) == 120
    assert 40 < sum(r["problem"].startswith("s") for r in picked[:80]) < 56   # the topics are mixed
    assert picked == qf.pick(list(reversed(rows(150, 100))))   # the order of the file does not matter
    f = tmp_path / "q.csv"
    with f.open("w", newline="") as out:
        w = csv.DictWriter(out, ["metadata", "problem", "answer"])
        w.writeheader()
        w.writerows(rows(130, 90))
    assert [r["problem"] for r in qf.questions(f)] == [r["problem"] for r in qf.pick(rows(130, 90))]


def test_letter():
    assert qf.letter(" b\n", "ABC") == "B"
    assert qf.letter("**A**", "ABC") == "A"
    assert qf.letter("Grade: A", "ABC") == "A"    # a word first: the last standalone letter
    assert qf.letter("Both turns are fine.", "ABT") is None
    assert qf.letter("Turn B is better. B", "ABT") == "B"
    assert qf.letter("", "ABT") is None and qf.letter(None, "ABT") is None


def test_answer_then_grade(tmp_path):
    def fake(base, path, body, timeout=None):
        text = body["messages"][0]["content"]
        if "Gold target" in text:   # the grader: right if the answer holds the target
            target = text.split("Gold target: ")[1].split("\n")[0]
            answer = text.split("Predicted answer: ")[1].split("\n\nReply")[0]
            return {"choices": [{"message": {"content": "A" if target in answer else "C" if "know" in answer else "B"}}]}
        reply = {"s0?": "It is S0.", "s1?": "It is X.", "s2?": "I don't know.", "s3?": ""}[text]
        return {"choices": [{"message": {"content": reply}, "finish_reason": "stop" if reply else "length"}],
                "usage": {"completion_tokens": 9}}
    qs = rows(4, 0)
    with mock.patch.object(qf, "post", fake):
        qf.answer("u", "m", qs, tmp_path / "a.jsonl")
        qf.answer("u", "m", qs, tmp_path / "a.jsonl")   # a second run adds nothing
        qf.grade("u", "g", tmp_path / "a.jsonl", tmp_path / "g.jsonl")
    assert len(qf.load(tmp_path / "a.jsonl")) == 4
    assert [r["grade"] for r in qf.load(tmp_path / "g.jsonl")] == ["CORRECT", "INCORRECT", "NOT_ATTEMPTED", "NOT_ATTEMPTED"]


def test_judge_maps_the_letter_to_the_model_in_both_orders(tmp_path):
    session = {"id": "s", "system_prompt": "sys", "segments": [[{"role": "user", "content": "do it"},
                                                                 {"role": "assistant", "content": "x", "target": True}]]}
    rec = lambda text: {"session": "s", "segment": 0, "index": 1, "turn": {"role": "assistant", "content": text}}
    (tmp_path / "ref.jsonl").write_text(json.dumps(rec("old")) + "\n")
    (tmp_path / "new.jsonl").write_text(json.dumps(rec("new")) + "\n")
    seen = []

    def fake(base, path, body, timeout=None):   # the judge prefers the turn "new", wherever it stands
        text = body["messages"][0]["content"]
        seen.append(text)
        return {"choices": [{"message": {"content": "A" if "Turn A:\nnew" in text else "B"}}]}
    with mock.patch.object(qf, "post", fake):
        qf.judge("u", "g", [session], tmp_path / "ref.jsonl", tmp_path / "new.jsonl", tmp_path / "j.jsonl")
    assert [r["winner"] for r in qf.load(tmp_path / "j.jsonl")] == ["new", "new"]
    assert "[user] do it" in seen[0] and "[system] sys" in seen[0]


def test_render_shows_text_and_tool_calls_not_reasoning():
    turn = {"content": "Run it.", "reasoning_content": "secret",
            "tool_calls": [{"function": {"name": "terminal", "arguments": '{"command": "ls"}'}}]}
    assert qf.render(turn) == 'Run it.\n[tool call] terminal({"command": "ls"})'
    assert qf.render({"content": ""}) == "(empty)"


def test_the_report_counts_the_common_questions(tmp_path, capsys):
    g = lambda q, grade: json.dumps({"question": q, "topic": qf.SCIENCE, "grade": grade}) + "\n"
    (tmp_path / "a.jsonl").write_text(g("q1", "CORRECT") + g("q2", "INCORRECT") + g("q3", "INCORRECT"))
    (tmp_path / "b.jsonl").write_text(g("q1", "CORRECT") + g("q2", "CORRECT"))
    qf.report([tmp_path / "a.jsonl", tmp_path / "b.jsonl"])
    out = capsys.readouterr().out.splitlines()
    assert "2 questions; correct 1, incorrect 1" in out[0] and "2 questions; correct 2, incorrect 0" in out[1]


def test_the_judge_reads_the_reasoning_when_the_content_is_empty(tmp_path):
    session = {"id": "s", "segments": [[{"role": "user", "content": "go"}, {"role": "assistant", "content": "x"}]]}
    rec = {"session": "s", "segment": 0, "index": 1, "turn": {"role": "assistant", "content": "same"}}
    (tmp_path / "r.jsonl").write_text(json.dumps(rec) + "\n")
    fake = lambda *a, **k: {"choices": [{"message": {"content": None, "reasoning_content": "They match. So: T"}}]}
    with mock.patch.object(qf, "post", fake):
        qf.judge("u", "g", [session], tmp_path / "r.jsonl", tmp_path / "r.jsonl", tmp_path / "j.jsonl")
    assert [r["winner"] for r in qf.load(tmp_path / "j.jsonl")] == ["tie", "tie"]
