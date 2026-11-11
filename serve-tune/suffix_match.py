"""Count the suffix matches in the bot's recorded sessions (PLAN_SERVE_TUNE.md §8.1).

The script tokenizes the sessions of ``model_eval/data/rank/sessions.jsonl`` and puts them in one token stream,
the oldest ``started_at`` first. Each message is one piece. An output piece is a turn that the model wrote:
an assistant message with ``"target"`` (``export_sessions.py``), as the DeepSeek V4 encoder writes it after
``<｜Assistant｜><think>``: the reasoning, ``</think>``, the content, the DSML tool calls and the end token.
All other pieces are prompt. A compaction copy shows no reasoning, as in the prompt.

At each output position i, the search history is the stream before i: the session's prompt, its earlier
output and the older sessions. The scan finds the longest suffix of the stream before i (4 to ``MAX_M``
tokens) that occurs earlier, and takes its most recent occurrence. The tokens after it are the draft. When
the occurrence overlaps the suffix, the draft repeats with the period of the overlap, so it never reads
the token at i. The correct count at depth d is the leading run of draft tokens that equal the next d
output tokens; the run stops at the end of the output piece. The scan sorts a rolling hash of each window
of m tokens: in a group of equal hashes, the previous window is the most recent occurrence.

The gate: share × (suffix mean − DSpark mean) ÷ (1 + DSpark mean) ≥ 10%. The DSpark mean at depth d is the
sum of the live acceptance per position (0.80, 0.58, 0.39, 0.24, 0.14); the plan uses 2.14 at d = 5. The
estimate is an upper bound, because DSpark also does well on copied text.
"""

import argparse
import json
from pathlib import Path

import numpy as np

RANK = Path.home() / "Projects/stock-research-model-eval/model_eval/data/rank"
TOKENIZER = Path.home() / "Projects/DeepSeek-v4-Flash-One-DGX-Spark/data/tp1-k160t5/tokenizer.json"
OUT = Path.home() / "Projects/dsv4-specialize/results/suffix-match.json"
MAX_M, BINS, GATE = 32, (4, 8, 16), 0.10
DSPARK_MEAN = {3: 1.77, 4: 2.01, 5: 2.14}
DSML, EOS = "｜DSML｜", "<｜end▁of▁sentence｜>"
HASH_BASE = np.uint64(0x9E3779B97F4A7C15)


def dsml(calls):
    """The tool calls as the DeepSeek V4 encoder writes them (``encode_arguments_to_dsml``)."""
    invokes = []
    for c in calls:
        args = json.loads(c["function"]["arguments"] or "{}")
        params = "\n".join(f'<{DSML}parameter name="{k}" string="{str(isinstance(v, str)).lower()}">'
                           f'{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}</{DSML}parameter>'
                           for k, v in args.items())
        invokes.append(f'<{DSML}invoke name="{c["function"]["name"]}">\n{params}\n</{DSML}invoke>')
    return f"\n\n<{DSML}tool_calls>\n" + "\n".join(invokes) + f"\n</{DSML}tool_calls>" if invokes else ""


def pieces(session, tools_text=""):
    """The ``(text, is output)`` pieces of one session, in order."""
    yield (session["system_prompt"] or "") + tools_text, False
    for m in (m for seg in session["segments"] for m in seg):
        if m["role"] == "user":
            yield "<｜User｜>" + m["content"], False
        elif m["role"] == "tool":
            yield f"<tool_result>{m['content']}</tool_result>", False
        else:
            out = bool(m.get("target"))
            yield "<｜Assistant｜><think>", False
            think = (m.get("reasoning_content") or "") + "</think>" if out else ""
            yield think + m["content"] + dsml(m.get("tool_calls", [])) + EOS, out


def stream(sessions, encode, tools_text=""):
    """The tokens of the sessions, oldest first, the output mask, and for each token the end of its piece and
    the start of its session."""
    parts = [(t, o, k) for k, s in enumerate(sorted(sessions, key=lambda s: s["started_at"]))
             for t, o in pieces(s, tools_text)]
    ids = encode([t for t, _, _ in parts])
    sizes = np.array([len(x) for x in ids], np.int64)
    ends, sid = np.cumsum(sizes), np.array([k for _, _, k in parts], np.int64)
    first = (ends - sizes)[np.searchsorted(sid, sid)]
    return (np.concatenate([np.asarray(x, np.int64) for x in ids]),
            np.repeat([o for _, o, _ in parts], sizes).astype(bool), np.repeat(ends, sizes), np.repeat(first, sizes))


def scan(tok, pos, max_m=MAX_M):
    """For each position in ``pos``: the length of the longest suffix of ``tok[:pos]`` that occurs earlier
    (0 below ``min(BINS)`` tokens), and the end of its most recent occurrence (-1 for no match)."""
    h = np.zeros(len(tok) + 1, np.uint64)            # h[e]: the hash of tok[e - m:e]
    t = tok.astype(np.uint64) + np.uint64(1)
    length, end = np.zeros(len(pos), np.int64), np.full(len(pos), -1, np.int64)
    for m in range(1, max_m + 1):
        h[m:] = h[m - 1:-1] * HASH_BASE + t[m - 1:]
        if m < min(BINS):
            continue
        order = np.argsort(h[m:], kind="stable") + m  # the window ends, by hash, then by end
        same = h[order[1:]] == h[order[:-1]]
        prev = np.full(len(tok) + 1, -1, np.int64)
        prev[order[1:][same]] = order[:-1][same]
        e = prev[pos]
        ok = (pos >= m) & (e >= 0)
        for k in range(1, m + 1):                     # remove the hash collisions
            ok &= tok[np.where(ok, e - k, 0)] == tok[np.where(ok, pos - k, 0)]
        length[ok], end[ok] = m, e[ok]
    return length, end


def correct(tok, pos, stop, end, d):
    """The leading run of draft tokens that equal the next ``d`` output tokens. The draft after the occurrence
    that ends at ``end`` repeats with the period ``pos - end``. ``stop`` ends the output."""
    run, period = end >= 0, np.maximum(pos - end, 1)
    count = np.zeros(len(pos), np.int64)
    for k in range(d):
        draft = tok[np.where(run, end + k % period, 0)]
        run &= (pos + k < stop) & (draft == tok[np.minimum(pos + k, len(tok) - 1)])
        count += run
    return count


def report(tok, out, stop, first, max_m=MAX_M):
    """Per (m, d): the share, the mean correct count, the gate estimate and the share from older sessions."""
    pos = np.flatnonzero(out)
    length, end = scan(tok, pos, max_m)
    counts = {d: correct(tok, pos, stop[pos], end, d) for d in DSPARK_MEAN}
    rows = []
    for m in BINS:
        sel = length >= m
        share = float(sel.mean()) if len(pos) else 0.0
        older = float((end[sel] <= first[pos[sel]]).mean()) if sel.any() else 0.0
        for d, ds in DSPARK_MEAN.items():
            mean = float(counts[d][sel].mean()) if sel.any() else 0.0
            rows.append({"m": m, "d": d, "share": round(share, 4), "mean": round(mean, 4),
                         "gate": round(share * (mean - ds) / (1 + ds), 4), "older": round(older, 4)})
    return {"output_positions": len(pos), "search_tokens": len(tok), "max_m": max_m,
            "dspark_mean": DSPARK_MEAN, "rows": rows, "gate_pass": any(r["gate"] >= GATE for r in rows)}


def encoder(path):
    from tokenizers import Tokenizer  # only the server image has it; the tests do not need it
    tk = Tokenizer.from_file(str(path))
    return lambda texts: [e.ids for e in tk.encode_batch(texts, add_special_tokens=False)]


def main(argv=None):
    """Run it with ``python -I`` in the server image, with no GPU and read-only mounts."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sessions", type=Path, default=RANK / "sessions.jsonl")
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args(argv)
    sessions = [json.loads(line) for line in a.sessions.open()]
    tools = a.sessions.with_name("tools.json")
    tools_text = "".join("\n\n" + json.dumps(t["function"], ensure_ascii=False)
                         for t in (json.loads(tools.read_text())["tools"] if tools.exists() else []))
    res = {"sessions": len(sessions), **report(*stream(sessions, encoder(a.tokenizer), tools_text))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    print("   m  d   share    mean    gate   older")
    for r in res["rows"]:
        print(f"{r['m']:>4}{r['d']:>3}{r['share']:>8.3f}{r['mean']:>8.3f}{r['gate']:>8.3f}{r['older']:>8.3f}")


if __name__ == "__main__":
    main()
