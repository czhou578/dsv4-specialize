"""The factual and side-by-side test of a pruned Qwen (plans/PLAN_QWEN_PRUNE.md §7.5).

- ``--answer``: a model answers N questions of SimpleQA (short facts with one known answer; a public set of OpenAI,
  in ``qwen-data/simpleqa/``). The pick is fixed: 120 questions of the topic "Science and technology" and 80 of the
  other topics, by a hash of the question. Each answer uses the thinking-mode sampling of the Qwen model card and a
  fixed seed per question. No tools: the test measures what the model knows, not what it can look up.
- ``--grade``: a grader model (the live DeepSeek) grades each answer against the known answer: CORRECT, INCORRECT or
  NOT_ATTEMPTED (the official grading rules of SimpleQA, in short). An INCORRECT answer is a hallucination.
- ``--judge REF NEW``: a blind side-by-side. REF and NEW are files of ``qwen_gate.py --record`` on the same held-out
  turns. On 30 of them (a fixed pick), the judge sees the end of the context and both visible replies, in both orders,
  and names the better reply or a tie. The judge thinks (about 45 s a call); the grader does not.
- ``--report``: the totals of the graded and judged files.
"""

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.home() / "Projects/stock-research-model-eval"))
from model_eval.replay import post  # noqa: E402

import qwen_gate  # noqa: E402

SCIENCE = "Science and technology"
GRADES = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}
GRADER = """Grade a predicted answer against the gold target of a question.
- CORRECT: the predicted answer contains the gold target (or an equal statement: another spelling, a fuller name,
  a number with the same value) and says nothing that contradicts it. Hedging is fine if the gold target is the answer.
- INCORRECT: the predicted answer states a fact that contradicts the gold target, or names a wrong answer, even with
  a hedge such as "I think" or "possibly". A list of guesses that includes a wrong answer as the answer is INCORRECT.
- NOT_ATTEMPTED: the predicted answer does not give the gold target and does not contradict it, for example "I don't
  know", or a request for more context.

Question: {question}
Gold target: {target}
Predicted answer: {answer}

Reply with one letter: A for CORRECT, B for INCORRECT, C for NOT_ATTEMPTED."""
JUDGE = """Two assistants wrote the next turn of the same conversation. The assistant is an agent for a software
developer: it writes code, runs tools, searches the web and researches stocks. Judge which turn is better for the
user: correct facts and code, no invented details, the right tool with the right arguments, and a clear answer.
Length is not a quality. If both turns are equally good, say so.

The end of the conversation:
{context}

Turn A:
{a}

Turn B:
{b}

Think briefly. Then reply with one letter only: A if turn A is better, B if turn B is better, T if they are equally
good."""


def load(path):
    return qwen_gate.load(path) if Path(path).exists() else []


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def pick(rows, science=120, other=80):
    """The fixed pick: ``science`` questions of the science topic and ``other`` of the rest, then all in hash order, so
    that a run cut short by its time limit still holds both kinds."""
    topic = lambda r: ast.literal_eval(r["metadata"]).get("topic")
    order = sorted(rows, key=lambda r: h(r["problem"]))
    picked = [r for r in order if topic(r) == SCIENCE][:science] + [r for r in order if topic(r) != SCIENCE][:other]
    return sorted(picked, key=lambda r: h("order:" + r["problem"]))


def questions(path):
    with Path(path).open(newline="") as f:
        return pick(list(csv.DictReader(f)))


def answer(base, model, rows, out, max_tokens=2048):
    done = {r["question"] for r in load(out)}
    with Path(out).open("a") as f:
        for r in rows:
            if r["problem"] in done:
                continue
            row = {"question": r["problem"], "target": r["answer"], "topic": ast.literal_eval(r["metadata"]).get("topic")}
            try:
                resp = post(base, "/v1/chat/completions", {
                    "model": model, "max_tokens": max_tokens, "seed": int(h(r["problem"])[:8], 16), **qwen_gate.SAMPLING,
                    "messages": [{"role": "user", "content": r["problem"]}]})
                m = resp["choices"][0]["message"]
                row.update(answer=m.get("content") or "", finish_reason=resp["choices"][0]["finish_reason"],
                           completion_tokens=resp["usage"]["completion_tokens"])
            except Exception as exc:   # a failed question is recorded, the run goes on
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            f.write(json.dumps(row) + "\n")
            f.flush()


def letter(text, allowed):
    """The letter of a reply: its first character if that is a lone allowed letter, else the last standalone capital
    letter of the reply that is allowed, else None."""
    s = (text or "").strip().strip("*").strip()
    if s and s[0].upper() in allowed and not s[1:2].isalpha():
        return s[:1].upper()
    found = [c for c in re.findall(r"\b([A-Z])\b", s) if c in allowed]
    return found[-1] if found else None


def grade(base, model, path, out):
    done = {r["question"] for r in load(out)}
    with Path(out).open("a") as f:
        for r in load(path):
            if r["question"] in done:
                continue
            g = {"question": r["question"], "topic": r["topic"]}
            if "error" in r:
                g["grade"] = "ERROR"
            elif not r["answer"].strip():   # the reasoning used every token: no answer
                g["grade"] = "NOT_ATTEMPTED"
            else:
                resp = post(base, "/v1/chat/completions", {"model": model, "max_tokens": 20, "temperature": 0,
                    "chat_template_kwargs": {"thinking": False},   # 6 of 6 known cases right with no thinking
                    "messages": [{"role": "user", "content": GRADER.format(question=r["question"], target=r["target"],
                                                                           answer=r["answer"][-4000:])}]})
                g["grade"] = GRADES.get(letter(resp["choices"][0]["message"].get("content"), "ABC"), "UNGRADED")
            f.write(json.dumps(g) + "\n")
            f.flush()


def render(turn):
    """The visible part of a turn: its text and its tool calls (not its reasoning)."""
    calls = [f"[tool call] {c['function']['name']}({c['function']['arguments']})" for c in turn.get("tool_calls") or []]
    return "\n".join([turn.get("content") or "", *calls]).strip() or "(empty)"


def context_tail(session, seg, i, chars=12000):
    lines = []
    for m in qwen_gate.messages(session, seg, i):
        text = m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
        if m.get("tool_calls"):
            text = render(m)
        lines.append(f"[{m['role']}] {text}")
    system = lines[0][:2000] + "\n...\n" if lines and lines[0].startswith("[system]") else ""
    return system + "\n".join(lines[1 if system else 0:])[-chars:]


def judge(base, model, sessions, ref_path, new_path, out, n=30):
    by_id = {s["id"]: s for s in sessions}
    key = lambda r: (r["session"], r["segment"], r["index"])
    ref, new = ({key(r): r for r in load(p)} for p in (ref_path, new_path))
    done = {(r["session"], r["segment"], r["index"], r["order"]) for r in load(out)}
    with Path(out).open("a") as f:
        for k in sorted(ref.keys() & new.keys(), key=lambda k: h(f"judge:{k}"))[:n]:   # a fixed pick of n turns
            ctx = context_tail(by_id[k[0]], by_id[k[0]]["segments"][k[1]], k[2])
            for order in ("ref_first", "new_first"):
                if (*k, order) in done:
                    continue
                a, b = (ref[k], new[k]) if order == "ref_first" else (new[k], ref[k])
                resp = post(base, "/v1/chat/completions", {"model": model, "max_tokens": 8000, "temperature": 0,
                    "messages": [{"role": "user", "content": JUDGE.format(context=ctx, a=render(a["turn"]), b=render(b["turn"]))}]})
                m = resp["choices"][0]["message"]   # an empty content: the letter is at the end of the reasoning
                reply = m.get("content") or (m.get("reasoning_content") or m.get("reasoning") or "")[-300:]
                v = letter(reply, "ABT")
                winner = {"T": "tie", None: "unjudged"}.get(v) or (("ref" if v == "A" else "new") if order == "ref_first"
                                                                   else ("new" if v == "A" else "ref"))
                f.write(json.dumps({"session": k[0], "segment": k[1], "index": k[2], "order": order, "winner": winner,
                                    "reply": reply[-200:]}) + "\n")
                f.flush()


def report(paths):
    """The totals of each file. Graded files count only the questions that every graded file holds (a model can stop
    at its time limit)."""
    graded = [{r["question"] for r in load(p)} for p in paths if load(p) and "grade" in load(p)[0]]
    common = set.intersection(*graded) if graded else set()
    for p in paths:
        rows = load(p)
        if rows and "grade" in rows[0]:
            rows = [r for r in rows if r["question"] in common]
            c = Counter(r["grade"] for r in rows)
            attempted = c["CORRECT"] + c["INCORRECT"]
            sci = Counter(r["grade"] for r in rows if r["topic"] == SCIENCE)
            print(f"{p}: {len(rows)} questions; correct {c['CORRECT']}, incorrect {c['INCORRECT']}, not attempted "
                  f"{c['NOT_ATTEMPTED']}, other {len(rows) - attempted - c['NOT_ATTEMPTED']}; incorrect of attempted "
                  f"{c['INCORRECT'] / max(attempted, 1):.1%}; science: correct {sci['CORRECT']}, incorrect {sci['INCORRECT']}")
        else:
            c = Counter(r["winner"] for r in rows)
            print(f"{p}: {len(rows)} verdicts; new wins {c['new']}, reference wins {c['ref']}, ties {c['tie']}, "
                  f"unjudged {c['unjudged']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    here = Path(__file__).resolve().parent
    ap.add_argument("--base-url", default="http://127.0.0.1:8889")
    ap.add_argument("--model", default="second-model-fit")
    ap.add_argument("--data", type=Path, default=here / "qwen-data")
    ap.add_argument("--questions", type=Path, default=here / "qwen-data/simpleqa/simple_qa_test_set.csv")
    ap.add_argument("--out", type=Path)
    cmd = ap.add_mutually_exclusive_group(required=True)
    cmd.add_argument("--answer", action="store_true")
    cmd.add_argument("--grade", type=Path, metavar="ANSWERS")
    cmd.add_argument("--judge", nargs=2, type=Path, metavar=("REF", "NEW"))
    cmd.add_argument("--report", nargs="+", type=Path)
    a = ap.parse_args(argv)
    if a.answer:
        answer(a.base_url, a.model, questions(a.questions), a.out)
        print(f"{sum('error' not in r for r in load(a.out))} answers, {sum('error' in r for r in load(a.out))} errors -> {a.out}")
    elif a.grade:
        grade(a.base_url, a.model, a.grade, a.out)
        report([a.out])
    elif a.judge:
        judge(a.base_url, a.model, qwen_gate.load(a.data / "sessions.jsonl"), *a.judge, a.out)
        report([a.out])
    else:
        report(a.report)


if __name__ == "__main__":
    main()
