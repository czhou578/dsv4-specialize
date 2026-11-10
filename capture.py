"""Expert capture: count the routed experts of the bot's traffic (PLAN_MODEL_SPECIALIZE.md §6.2, bot repo).

Boot ``compose-capture.yml`` (Model Runner V1, ``--enable-return-routed-experts``)
on port 8890. For each segment of the calibration split, one ``/v1/chat/completions`` request sends the
context and the segment's last target turn, as the replay does, with ``max_tokens: 1``. The reply's
``choices[0].routed_experts`` is a base64 ``.npy`` array of shape (prompt tokens, layers, top-k): the
experts that the router picked for each prompt token.

- The rows of the shared prefix (system prompt and tools) count once, the first time the prefix appears;
  otherwise each request would count it again.
- A row of six equal ids is a token that the capture did not fill. It does not count.

Each segment writes one row of counts to ``--out``, so a rerun skips the finished segments. ``--rank``
sums the rows into ``expert_rank_bot.json`` (§6.5): per layer, every expert with its count and share,
most used first, and the counts per skill.

The router weights (plans/PLAN_BETTER_RANK.md §4): the patched capture container (``compose-capture.yml``)
sums, per layer and expert, the times an expert was chosen and the router weight it got, over every token
the model computes, and writes the sums to ``routing_sums.npz`` after each step. With ``--sums`` the client
reads the file before and after each request; the difference is the request's own routing. A cached prefix
is not computed again, so it counts once by itself. ``--rank --order weight`` orders the experts by the
weight sum; ``--order skill_max`` by the largest share an expert has within any one skill (the floor of §5).
"""

import argparse
import base64
import hashlib
import io
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

# The replay stays in the bot repo, and the capture sends the replay's own request: import it from the eval worktree.
EVAL = Path.home() / "Projects/stock-research-model-eval"
sys.path.insert(0, str(EVAL))
from model_eval.replay import DATA, SOURCES, load, post, request_body  # noqa: E402

EXPERTS = 216


def segments(sessions, split="calibration", sources=SOURCES):
    """``(session, segment index, index of the last target)`` of each bot segment of the split."""
    for s in sessions:
        if s["split"] != split or s["source"] not in sources:
            continue
        for g, seg in enumerate(s["segments"]):
            last = max((i for i, m in enumerate(seg) if m.get("target")), default=None)
            if last is not None:
                yield s, g, last


def capture_body(model, session, seg, i, tools, effort):
    """The replay request without prompt logprobs, with the token ids returned."""
    body = request_body(model, session, seg, i, tools, effort)
    del body["prompt_logprobs"]
    if not session["system_prompt"]:
        body["messages"] = body["messages"][1:]
    body["return_token_ids"] = True
    return body


def decode(b64):
    return np.load(io.BytesIO(base64.b64decode(b64)))


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def read_sums(path):
    """The sums file of the patched capture container, or None if there is none yet."""
    path = Path(path)
    if not path.exists():
        return None
    for _ in range(20):  # a rename is atomic, but the file can be mid-write on another filesystem
        try:
            with np.load(path) as z:  # the prof_* arrays are there with ROUTING_PROFILE=1 (PLAN_FEWER_EXPERTS.md §5)
                return {k: z[k].copy() for k in z.files}
        except (ValueError, EOFError, OSError):
            time.sleep(0.1)
    raise RuntimeError(f"{path}: cannot read the sums file")


def sums_delta(before, after):
    """The request's own routing: ``after - before`` of the sums, rounded for the JSONL row."""
    if before is None:
        before = {k: np.zeros_like(v) for k, v in after.items()}
    return {"wcount": (after["count"] - before["count"]).astype(np.int64).tolist(),
            "weight": np.round(after["weight"] - before["weight"], 5).tolist(),
            "computed_tokens": (after["tokens"] - before["tokens"]).astype(np.int64).tolist(),
            "steps": int(after["steps"] - before["steps"]),
            **{k: np.round(after[k] - before.get(k, 0), 3).tolist() for k in after if k.startswith("prof_")}}


def count(routed, start=0, experts=EXPERTS):
    """Per-layer expert counts of the rows from ``start`` on, and the number of unfilled rows."""
    rows = routed[start:].astype(np.int64)
    filled = ~(rows == rows[:, :, :1]).all(axis=2)          # (tokens, layers)
    counts = np.stack([np.bincount(rows[filled[:, l], l].ravel(), minlength=experts)
                       for l in range(rows.shape[1])])
    return counts, int((~filled).sum())


def capture_one(base, model, s, g, i, tools, effort, seen_prefixes, sums=None):
    seg = s["segments"][g]
    body = capture_body(model, s, seg, i, tools, effort)
    before = read_sums(sums) if sums else None
    resp = post(base, "/v1/chat/completions", body)
    choice = resp["choices"][0]
    if not choice.get("routed_experts"):
        raise RuntimeError("no routed_experts in the reply: boot compose-capture.yml")
    routed = decode(choice["routed_experts"])
    prompt = resp["prompt_token_ids"]
    if routed.shape[0] != len(prompt):
        raise ValueError(f"{routed.shape[0]} routing rows for {len(prompt)} prompt tokens")
    system = [m for m in body["messages"] if m["role"] == "system"]
    # /tokenize ignores reasoning_effort, and the template puts the effort line first: pass it as a
    # template kwarg, or the default ("max") line makes the common prefix 1 token (2026-11-03).
    shared = post(base, "/tokenize", {"model": model, "messages": system or [{"role": "system", "content": ""}],
                                      "tools": tools, "add_generation_prompt": False,
                                      "chat_template_kwargs": {"reasoning_effort": effort}})["tokens"]
    prefix = common_prefix(shared, prompt)
    key = hashlib.sha256(json.dumps(prompt[:prefix]).encode()).hexdigest()
    new_prefix = key not in seen_prefixes
    counts, unfilled = count(routed, 0 if new_prefix else prefix)
    row = {"prompt_tokens": len(prompt), "prefix_tokens": prefix, "prefix": key,
           "prefix_counted": new_prefix, "layers": routed.shape[1], "top_k": routed.shape[2],
           "unfilled_rows": unfilled, "counts": counts.tolist()}
    if sums:
        after = read_sums(sums)
        for _ in range(50):  # the flush of the last step is synchronous; this is a guard only
            if after is not None and (before is None or after["steps"] > before["steps"]):
                break
            time.sleep(0.2)
            after = read_sums(sums)
        if after is None or (before is not None and after["steps"] == before["steps"]):
            raise RuntimeError("the sums file did not change: is the patched container running?")
        row.update(sums_delta(before, after))
    return row


def run(base, model, out, limit=None, effort="medium", data=DATA, sources=SOURCES, sums=None):
    sessions = [json.loads(line) for line in (data / "sessions.jsonl").open()]
    tools = json.loads((data / "tools.json").read_text())["tools"]
    todo = list(segments(sessions, sources=sources))[:limit]
    rows = load(out) if out.exists() else []
    done = {(r["session"], r["segment"]) for r in rows if "error" not in r}
    seen = {r["prefix"] for r in rows if r.get("prefix_counted")}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for n, (s, g, i) in enumerate(todo, 1):
            if (s["id"], g) in done:
                continue
            row = {"session": s["id"], "segment": g, "index": i, "skill": s["skill"]}
            began = time.monotonic()
            try:
                row.update(capture_one(base, model, s, g, i, tools, effort, seen, sums))
                if row["prefix_counted"]:
                    seen.add(row["prefix"])
            except Exception as exc:  # a failed segment is recorded, the run goes on
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            row["seconds"] = round(time.monotonic() - began, 1)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{n}/{len(todo)}] {s['skill']:<13} {row.get('prompt_tokens', '-'):>7} tokens "
                  f"{row['seconds']:>6} s {row.get('unfilled_rows', '')} {row.get('error', '')}", flush=True)


ORDERS = ("count", "weight", "skill_max", "skill_floor")
SCORES = {"count": "count of routed tokens per expert (§6.2 step 1); ids are the current ids 0-215",
          "weight": "sum of the router weights per expert (PLAN_BETTER_RANK.md §5); ids are the current ids 0-215",
          "skill_max": "largest share of the router weight an expert has within any one skill with at least "
                       "--min-skill-tokens computed tokens (PLAN_BETTER_RANK.md §5)",
          "skill_floor": "the weight order with the floor per skill (PLAN_BETTER_RANK.md §5): an expert's best rank "
                         "by weight over the whole data and over each skill with at least --min-skill-tokens computed "
                         "tokens, so that an expert drops only if it is below the cut in every skill; ties by weight"}


def rank(rows, order="count", min_skill_tokens=50000):
    """``expert_rank_bot.json``: per layer, every expert with its scores, most important first by ``order``.

    ``count`` and ``share`` come from the ids of the reply (every row). ``weight``, ``weight_share`` and
    ``skill_max`` come from the sums file (rows with ``weight``); without such rows, those fields are 0 and the
    order falls back to ``count``."""
    rows = [r for r in rows if "error" not in r]
    if not rows:
        raise ValueError("no captured segments")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    total = sum(np.array(r["counts"]) for r in rows)
    weighted = [r for r in rows if "weight" in r]
    nan_slots = 0
    if weighted:
        # A NaN router weight (a padded row of a graph-replayed step) makes a slot's sum NaN from that row on. The
        # rows before it are good: the weight of such a slot is the sum of its finite rows, scaled by the count ratio.
        w = np.array([r["weight"] for r in weighted], dtype=float)
        c = np.array([r["wcount"] for r in weighted], dtype=float)
        finite = np.isfinite(w)
        nan_slots = int((~finite).any(axis=0).sum())
        c_fin = (c * finite).sum(axis=0)
        weight = np.where(finite, w, 0.0).sum(axis=0) * np.where(c_fin > 0, c.sum(axis=0) / np.maximum(c_fin, 1), 0.0)
    else:
        weight = np.zeros_like(total, dtype=float)
    per_skill = defaultdict(lambda: {"count": 0, "weight": 0, "tokens": 0})
    for r in rows:
        p = per_skill[r["skill"]]
        p["count"] = p["count"] + np.array(r["counts"])
        if "weight" in r:
            p["weight"] = p["weight"] + np.nan_to_num(np.array(r["weight"], dtype=float), nan=0.0)
            p["tokens"] += int(r["computed_tokens"][0])
    skill_max = np.zeros_like(total, dtype=float)
    best_rank = np.full(total.shape, total.shape[1], dtype=float)  # the rank position, 0 = most important

    def positions(score):
        pos = np.empty_like(score)
        for l in range(score.shape[0]):
            pos[l, np.argsort(-score[l], kind="stable")] = np.arange(score.shape[1])
        return pos
    if weighted:
        best_rank = positions(weight)
    for p in per_skill.values():
        if p["tokens"] >= min_skill_tokens:
            w = np.asarray(p["weight"], dtype=float)
            skill_max = np.maximum(skill_max, w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12))
            best_rank = np.minimum(best_rank, positions(w))
    if not weighted and order != "count":
        order = "count"
    # skill_floor: the best rank first (ascending), the weight as the tie-break
    floor_key = -(best_rank * (weight.max() + 1.0)) + weight
    key = {"count": total, "weight": weight, "skill_max": skill_max, "skill_floor": floor_key}[order]
    layers = {}
    for l, c in enumerate(total):
        ids = np.argsort(-key[l], kind="stable")
        layers[str(l)] = [{"id": int(e), "count": int(c[e]), "share": round(float(c[e] / c.sum()), 6),
                           "weight": round(float(weight[l][e]), 3),
                           "weight_share": round(float(weight[l][e] / max(weight[l].sum(), 1e-12)), 6),
                           "skill_max": round(float(skill_max[l][e]), 6),
                           "best_rank": int(best_rank[l][e])} for e in ids]
    return {"segments": len(rows), "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "unfilled_rows": sum(r["unfilled_rows"] for r in rows), "weighted_segments": len(weighted),
            "nan_slots": nan_slots,
            "order": order, "min_skill_tokens": min_skill_tokens, "score": SCORES[order], "layers": layers,
            "per_skill": {k: {"count": np.asarray(v["count"]).tolist(),
                              "weight": np.round(np.asarray(v["weight"], dtype=float), 3).tolist(),
                              "tokens": v["tokens"]} for k, v in sorted(per_skill.items())}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8890")
    parser.add_argument("--model", help="served model name (default: the server's first model)")
    parser.add_argument("--out", type=Path, help="counts JSONL; a rerun skips the segments in it")
    parser.add_argument("--limit", type=int, help="capture the first N segments only")
    parser.add_argument("--effort", default="medium", help="reasoning_effort, as the bot sends it")
    parser.add_argument("--rank", nargs=2, metavar=("COUNTS", "RANK_JSON"), type=Path,
                        help="sum a counts JSONL into the rank file and exit")
    parser.add_argument("--order", choices=ORDERS, default="count", help="the order of the rank file")
    parser.add_argument("--min-skill-tokens", type=int, default=50000,
                        help="a skill below this many computed tokens does not set skill_max")
    parser.add_argument("--data", type=Path, default=DATA, help="the folder with sessions.jsonl and tools.json")
    parser.add_argument("--sources", default=",".join(SOURCES),
                        help="session sources to capture, comma-separated (e2e and cron for the rank data)")
    parser.add_argument("--sums", type=Path, help="the routing_sums.npz of the patched capture container")
    args = parser.parse_args(argv)
    if args.rank:
        result = rank(load(args.rank[0]), args.order, args.min_skill_tokens)
        args.rank[1].write_text(json.dumps(result))
        print(f"{result['segments']} segments ({result['weighted_segments']} with weights), "
              f"{result['prompt_tokens']} prompt tokens, {result['unfilled_rows']} unfilled rows, "
              f"order {result['order']} -> {args.rank[1]}")
        return
    if not args.out:
        parser.error("--out is required unless --rank is given")
    with urllib.request.urlopen(args.base_url + "/v1/models", timeout=30) as resp:
        model = args.model or json.load(resp)["data"][0]["id"]
    run(args.base_url, model, args.out, args.limit, args.effort, args.data,
        tuple(args.sources.split(",")), args.sums)


if __name__ == "__main__":
    main()
