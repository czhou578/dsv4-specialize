"""The replay gate of the Qwen prune (plans/PLAN_QWEN_PRUNE.md §4, G1 and G2).

- ``--record``: the full model writes one turn for each of N held-out targets of ``qwen-data`` (the context is the
  segment up to the target), at the sampling settings of the fast route with a fixed seed. The turns are the recorded
  set.
- ``--score``: a model scores the recorded turns. One request sends the context and the recorded turn as the last
  message with ``prompt_logprobs``; the tokens after the context are the turn. A row holds the NLL of the turn, its
  token count, the tokens that rank first, the decision (the first token after ``</think>`` ranks first; the first
  token of the turn when it has no reasoning) and the tool names (every token of each ``"name": "..."`` ranks first).
- ``--compare REF NEW``: the totals of both files on their common turns, and the gates: the NLL per token at most +3%,
  the tool names at least 98% of the reference, the decisions down by no more than 3.

The reference of a window is the full model, scored in that window.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

sys.path.insert(0, str(Path.home() / "Projects/stock-research-model-eval"))
from model_eval.replay import post  # noqa: E402

NAME = re.compile(r'"name": ?"([^"]+)"')
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}   # the thinking-mode settings of the Qwen model card
STRIP = ("target", "reasoning_content")


def load(path):
    with Path(path).open() as f:   # not splitlines(): a JSON string can hold a Unicode line separator
        return [json.loads(line) for line in f]


def messages(session, seg, i):
    """The system prompt (if any) and the segment up to message ``i``, without the data fields."""
    system = [{"role": "system", "content": session["system_prompt"]}] if session.get("system_prompt") else []
    return system + [{k: v for k, v in m.items() if k not in STRIP} for m in seg[:i]]


def pick(sessions, n, max_chars=240_000, per_segment=2):
    """``n`` held-out targets: one source and project after another in a fixed hash order, with a context of at most
    ``max_chars`` characters. Every segment gives one target before any segment gives a second one."""
    groups = defaultdict(list)
    for s in sessions:
        if s["split"] != "held_out":
            continue
        for g, seg in enumerate(s["segments"]):
            sizes, total = [], 0
            for m in seg:
                sizes.append(total)
                total += len(json.dumps(m))
            targets = [i for i, m in enumerate(seg) if m.get("target") and 0 < i and sizes[i] <= max_chars]
            h = lambda i: hashlib.sha256(f"{s['id']}:{g}:{i}".encode()).hexdigest()
            groups[(s["source"], s["skill"])] += [(j, h(i), s["id"], g, i) for j, i in enumerate(sorted(targets, key=h)[:per_segment])]
    rounds = [[r for row in zip_longest(*(sorted(x for x in v if x[0] == j) for _, v in sorted(groups.items()))) for r in row if r]
              for j in range(per_segment)]
    return [(sid, g, i) for _, _, sid, g, i in [r for round_ in rounds for r in round_][:n]]


def record(base, model, sessions, tools, n, out, max_tokens=800):
    by_id = {s["id"]: s for s in sessions}
    done = {(r["session"], r["segment"], r["index"]) for r in load(out)} if Path(out).exists() else set()
    with Path(out).open("a") as f:
        for sid, g, i in pick(sessions, n):
            if (sid, g, i) in done:
                continue
            seed = int(hashlib.sha256(f"{sid}:{g}:{i}".encode()).hexdigest()[:8], 16)
            resp = post(base, "/v1/chat/completions", {
                "model": model, "tools": tools, "max_tokens": max_tokens, "seed": seed, **SAMPLING,
                "messages": messages(by_id[sid], by_id[sid]["segments"][g], i)})
            m = resp["choices"][0]["message"]
            turn = {"role": "assistant", "content": m.get("content") or "",
                    **({"reasoning_content": r} if (r := m.get("reasoning_content") or m.get("reasoning")) else {}),
                    **({"tool_calls": m["tool_calls"]} if m.get("tool_calls") else {})}
            f.write(json.dumps({"session": sid, "segment": g, "index": i, "skill": by_id[sid]["skill"], "turn": turn,
                                "completion_tokens": resp["usage"]["completion_tokens"],
                                "finish_reason": resp["choices"][0]["finish_reason"]}) + "\n")
            f.flush()


def score_tokens(toks, end_think):
    """The measures of the turn tokens ``[(token id, info)]``."""
    cut = next((k + 1 for k, (t, _) in enumerate(toks) if t == end_think), 0)
    text, offsets = "", []
    for _, info in toks:
        offsets.append(len(text))
        text += info.get("decoded_token") or ""
    names = []
    for m in NAME.finditer(text):
        a, b = m.span(1)
        inside = [info for (_, info), o in zip(toks, offsets) if o < b and o + len(info.get("decoded_token") or "") > a]
        names.append(all(info["rank"] == 1 for info in inside))
    return {"tokens": len(toks), "nll": -sum(info["logprob"] for _, info in toks),
            "top1": sum(info["rank"] == 1 for _, info in toks), "reasoning_tokens": cut,
            "decision_ok": cut < len(toks) and toks[cut][1]["rank"] == 1,
            "tool_calls": len(names), "tool_names_ok": sum(names)}


def score(base, model, sessions, tools, recorded, out):
    by_id = {s["id"]: s for s in sessions}
    end_think = post(base, "/tokenize", {"model": model, "prompt": "</think>", "add_special_tokens": False})["tokens"][0]
    done = {(r["session"], r["segment"], r["index"]) for r in load(out)} if Path(out).exists() else set()
    with Path(out).open("a") as f:
        for r in recorded:
            key = (r["session"], r["segment"], r["index"])
            if key in done:
                continue
            row, began = {"session": key[0], "segment": key[1], "index": key[2], "skill": r["skill"]}, time.monotonic()
            try:
                context = messages(by_id[key[0]], by_id[key[0]]["segments"][key[1]], key[2])
                start = len(post(base, "/tokenize", {"model": model, "messages": context, "tools": tools,
                                                     "add_generation_prompt": True})["tokens"])
                turn = {**r["turn"], **({"reasoning": r["turn"]["reasoning_content"]} if "reasoning_content" in r["turn"] else {})}
                resp = post(base, "/v1/chat/completions", {
                    "model": model, "tools": tools, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0,
                    "add_generation_prompt": False, "messages": [*context, turn]})
                toks = [(int(t), info) for p in resp["prompt_logprobs"][start:] if p for t, info in p.items()]
                row.update(score_tokens(toks, end_think))
            except Exception as exc:   # a failed turn is recorded, the run goes on
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            row["seconds"] = round(time.monotonic() - began, 1)
            f.write(json.dumps(row) + "\n")
            f.flush()


def totals(rows):
    t = defaultdict(float)
    for r in rows:
        for k in ("tokens", "nll", "top1", "decision_ok", "tool_calls", "tool_names_ok"):
            t[k] += r[k]
    return {"turns": len(rows), "nll": t["nll"] / max(t["tokens"], 1), "top1": t["top1"] / max(t["tokens"], 1),
            "decisions": int(t["decision_ok"]), "tool_names": t["tool_names_ok"] / t["tool_calls"] if t["tool_calls"] else 1.0}


def compare(ref_path, new_path):
    """The totals of both files on the turns that both scored, and the gates of §4."""
    key = lambda r: (r["session"], r["segment"], r["index"])
    ref, new = ({key(r): r for r in load(p) if "error" not in r and r["tokens"]} for p in (ref_path, new_path))
    common = sorted(ref.keys() & new.keys())
    a, b = totals([ref[k] for k in common]), totals([new[k] for k in common])
    gates = {"nll at most +3%": b["nll"] <= a["nll"] * 1.03,
             "tool names at least 98% of the reference": b["tool_names"] >= a["tool_names"] * 0.98,
             "decisions down by no more than 3": b["decisions"] >= a["decisions"] - 3}
    return a, b, gates


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    here = Path(__file__).resolve().parent
    ap.add_argument("--base-url", default="http://127.0.0.1:8889")
    ap.add_argument("--model", default="second-model-fit")
    ap.add_argument("--data", type=Path, default=here / "qwen-data")
    ap.add_argument("--recorded", type=Path, default=here / "results/qwen-recorded.jsonl")
    ap.add_argument("--out", type=Path, help="the score file of --score")
    ap.add_argument("--n", type=int, default=60, help="the turns of --record")
    cmd = ap.add_mutually_exclusive_group(required=True)
    cmd.add_argument("--record", action="store_true")
    cmd.add_argument("--score", action="store_true")
    cmd.add_argument("--compare", nargs=2, metavar=("REF", "NEW"), type=Path)
    a = ap.parse_args(argv)
    if a.compare:
        ref, new, gates = compare(*a.compare)
        for k in ref:
            print(f"{k:<12}{ref[k]:>12.4f}{new[k]:>12.4f}")
        print(f"nll change  {100 * (new['nll'] / ref['nll'] - 1):+.2f}%")
        for name, ok in gates.items():
            print(f"{'PASS' if ok else 'FAIL'}  {name}")
        return
    sessions = load(a.data / "sessions.jsonl")
    tools = json.loads((a.data / "tools.json").read_text())["tools"]
    if a.record:
        record(a.base_url, a.model, sessions, tools, a.n, a.recorded)
        print(f"{len(load(a.recorded))} recorded turns -> {a.recorded}")
    else:
        score(a.base_url, a.model, sessions, tools, load(a.recorded), a.out)
        rows = load(a.out)
        print(f"{sum('error' not in r for r in rows)} scored turns, {sum('error' in r for r in rows)} errors -> {a.out}")


if __name__ == "__main__":
    main()
