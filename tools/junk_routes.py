"""The route capture of the junk prompts (plans/PLAN_JUNK_CAPTURE.md §3 and §4).

  smoke    [--base-url U] [--model M] [--sums S]
  capture  --out F [--base-url U] [--model M] [--prompts P] [--rounds 3] [--pilot DIR] [--seed 0] [--sums S]
  report   [--capture F] [--out-dir D] [--k160 PRUNE.json] [--md PRUNE.json] [--bot F]
  select   [--report F] [--cold F] [--out F]

Boot compose-capture.yml with compose-capture-junk.yml (Model Runner V1, --enable-return-routed-experts, a 6.5 GiB KV
pin) on port 8890. Each reply then holds choices[0].routed_experts: a base64 .npy array of the experts that the router
picked, per token, layer and slot. `smoke` sends one short raw prompt and checks the array and the sums file.

`capture` sends one request at a time, each with a new cache_salt, so that every prompt is a cold prefill:

- probe: the probes and the greedy prompt of the prompts file (results/junk-bisect/prompts.json), raw, through
  /v1/completions, ROUNDS times. The rounds show how stable the routing is.
- pilot: the prompt of the last turn of each DeepSeek transcript (deepseek-*-s<seed>) of the pilot folder, through
  /v1/chat/completions with the tools of agent-eval. The messages are rebuilt as agent_eval.run_task sent them, except
  that a tool result is the clipped text of the transcript (at most 2,000 characters), as in the probe prompts.

Each request writes one JSONL row to --out: the counts per layer and expert over the whole prompt and over the last
TAIL rows, the ids of the last row (the probe position), and with --sums the router weight sums of the request. The ids
of every row go to <out>.ids/<key>.npy (uint8). A rerun skips the rows that have no error.

`report` applies §3 to the rows and writes report.md and report.json. For each probe (summed over the rounds), each
pilot family, the held-out pilot tasks (all but C2 and C3) and the bot baseline (results/capture-rank.jsonl), it gives
the share of the picks and of the router weight on the experts that K 160 removed (in all layers, in the hash layers
0-2, in layers 3-42) and on the experts that the MD step removed after K 160 ("MD-extra"), the same shares over the
last TAIL rows, and the removed picks at the probe position. Then the fixed rules: G1 (c3 has at least twice the bot
share of MD-extra picks), G2 (c2 has at least twice the bot share of K 160 picks in layers 3-42), the set X and G3 (X
holds at least half of c3's MD-extra weight). X: the MD-extra experts in the order of their weight share in their
layer on the held-out tasks, until X holds X_SHARE of the pool's share, at most X_MAX. `select` writes the drop file of
MD + X: the drops of results/cold-md.json without X, for mask_checkpoint.py --drop-file.
"""

import argparse
import json
import sys
import time
import types
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

from capture import EXPERTS, count, decode, post, read_sums, sums_delta

S = Path(__file__).resolve().parent.parent
PROMPTS = S / "results/junk-bisect/prompts.json"
PILOT = S / "results/agent-eval/pilot"
AGENT_EVAL = Path.home() / "Projects/dsv4-specialize-agent-eval/agent-eval"
TAIL = 256          # the rows before the probe position: the quote and the JSON around it
LAYERS, TOP_K = 43, 6
# The smoke prompt must run eagerly: a step of 24 tokens or fewer replays a CUDA graph (MAX_CUDAGRAPH_CAPTURE_SIZE of
# compose-capture.yml), and the patch writes the sums of such a step only at the start of the next step. "Say OK." alone
# (about 4 tokens) failed the smoke of 2026-11-16 22:11 for that reason. This prompt has more than 100 tokens.
SMOKE_PROMPT = " ".join(["Say OK."] * 40)
K160 = Path.home() / "Projects/DeepSeek-v4-Flash-One-DGX-Spark/data/tp1-k160t5/PRUNE.json"
MD = S / "tp1-prunemd5-k149/PRUNE.json"
BOT = S / "results/capture-rank.jsonl"
COLD_MD = S / "results/cold-md.json"
OUT_DIR = S / "results/junk-capture"
HASH_LAYERS = 3     # layers 0-2 route by a hash of the token id
HELD_OUT = ("C2", "C3")   # the pilot tasks of the probe prompts c2 and c3
X_SHARE, X_MAX = 0.5, 113  # half of the pool's held-out share, at most 1 GiB (9.04 MiB per expert)
MIB_PER_EXPERT = 9.04


def served_model(base):
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as resp:
        return json.load(resp)["data"][0]["id"]


def agent_eval():
    if str(AGENT_EVAL) not in sys.path:
        sys.path.insert(0, str(AGENT_EVAL))
    try:
        import jsonschema  # noqa: F401
    except ImportError:   # the numpy venv lacks it; only checks.validate_call uses it, and the capture never calls it
        sys.modules["jsonschema"] = types.ModuleType("jsonschema")
    import agent_eval as ae
    return ae


def pilot_messages(transcript, ae):
    """The messages of the request of the last turn, as agent_eval.run_task sent them (tool results clipped)."""
    task = ae.load_task(transcript["task"])
    msgs = [{"role": "system", "content": ae.SYSTEM}, *ae.padding(transcript["ctx"]),
            {"role": "user", "content": task["prompt"]}]
    for i, rec in enumerate(transcript["turns"][:-1]):
        calls = [{"id": f"c{i}_{j}", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                 for j, c in enumerate(rec["calls"])]
        msgs.append({"role": "assistant", "content": rec["content"], "reasoning": rec["reasoning"] or None,
                     **({"tool_calls": calls} if calls else {})})
        if not calls:
            msgs.append({"role": "user", "content": ae.NUDGE})
        for j, c in enumerate(rec["calls"]):
            msgs.append({"role": "tool", "tool_call_id": f"c{i}_{j}", "content": c["result"]})
    return msgs


def jobs(prompts=None, rounds=3, pilot=None, seed=0):
    """``(key, fields of the row, route, body)`` of every request, the probes first."""
    if prompts:
        data = json.loads(Path(prompts).read_text())
        items = [(p["name"], p["prompt"]) for p in data["probes"]] + [("greedy", data["greedy"]["prompt"])]
        for r in range(1, rounds + 1):
            for name, prompt in items:
                yield (f"probe-{name}-r{r}", {"kind": "probe", "name": name, "round": r}, "/v1/completions",
                       {"prompt": prompt, "add_special_tokens": False})
    if pilot:
        ae = agent_eval()
        for path in sorted(Path(pilot).glob(f"deepseek-*-s{seed}/transcript.json")):
            t = json.loads(path.read_text())
            if t["turns"]:
                fields = {"kind": "pilot", "name": t["task"], "family": t["task"].rstrip("0123456789")}
                yield (f"pilot-{t['task']}-s{seed}", fields, "/v1/chat/completions",
                       {"messages": pilot_messages(t, ae), "tools": ae.tools.SCHEMAS})


def sums_after(path, before):
    """The sums file once it holds the steps of the request (the flush is synchronous; this is a guard only)."""
    for _ in range(50):
        after = read_sums(path)
        if after is not None and (before is None or after["steps"] > before["steps"]):
            return after
        time.sleep(0.2)
    raise RuntimeError("the sums file did not change: is the patched capture container running?")


def routes(resp):
    """The routing rows of the prompt tokens: the reply also holds the row of the generated token."""
    choice = resp["choices"][0]
    if not choice.get("routed_experts"):
        raise RuntimeError("no routed_experts in the reply: boot compose-capture.yml")
    routed, n = decode(choice["routed_experts"]), resp["usage"]["prompt_tokens"]
    if routed.ndim != 3 or routed.shape[0] < n or routed.shape[1:] != (LAYERS, TOP_K):
        raise ValueError(f"routing rows of shape {routed.shape} for {n} prompt tokens")
    return routed[:n]


def capture_one(base, model, route, body, sums=None):
    """One cold request; the row of its routing, and its ids."""
    before = read_sums(sums) if sums else None
    t0 = time.monotonic()
    resp = post(base, route, {"model": model, **body, "max_tokens": 1, "temperature": 0,
                              "cache_salt": uuid.uuid4().hex})
    routed = routes(resp)
    n = routed.shape[0]
    counts, unfilled = count(routed, 0, EXPERTS)
    tail, _ = count(routed, max(0, n - TAIL), EXPERTS)
    row = {"prompt_tokens": n, "unfilled_rows": unfilled, "counts": counts.tolist(), "tail_counts": tail.tolist(),
           "last": routed[-1].tolist(), "seconds": round(time.monotonic() - t0, 2)}
    if sums:
        row.update(sums_delta(before, sums_after(sums, before)))
    return row, routed.astype(np.uint8)


def capture(base, out, model=None, prompts=None, rounds=3, pilot=None, seed=0, sums=None):
    out, model = Path(out), model or served_model(base)
    ids = out.with_name(out.name + ".ids")
    ids.mkdir(parents=True, exist_ok=True)
    old = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    done = {r["key"] for r in old if "error" not in r}
    rows = errors = tokens = 0
    for key, fields, route, body in jobs(prompts, rounds, pilot, seed):
        if key in done:
            continue
        try:
            row, routed = capture_one(base, model, route, body, sums)
            np.save(ids / f"{key}.npy", routed)
            tokens += row["prompt_tokens"]
        except (urllib.error.URLError, OSError, RuntimeError, ValueError, KeyError) as e:
            row, errors = {"error": f"{type(e).__name__}: {e}"[:500]}, errors + 1
        with out.open("a") as f:
            f.write(json.dumps({"key": key, **fields, **row}) + "\n")
        rows += 1
        print(f"{key}: {row.get('prompt_tokens', row.get('error'))}", flush=True)
    print(f"capture: {rows} rows, {errors} errors, {tokens} prompt tokens")
    return errors


def smoke(base, model=None, sums=None):
    """One short raw prompt: the reply must hold the routing rows of every token, and the sums file must move."""
    try:
        row, routed = capture_one(base, model or served_model(base), "/v1/completions",
                                  {"prompt": SMOKE_PROMPT, "add_special_tokens": False}, sums)
    except (urllib.error.URLError, OSError, RuntimeError, ValueError, KeyError) as e:
        print(f"smoke FAIL: {type(e).__name__}: {e}")
        return 1
    extra = f", {row['steps']} steps in the sums" if sums else ""
    print(f"smoke OK: {routed.shape[0]} rows of {routed.shape[1]} layers x {routed.shape[2]} ids{extra}")
    return 0


def removed_sets(k160=K160, md=MD):
    """Boolean (43, 216) masks: the experts that K 160 removed, and the ones that the MD step removed after it."""
    keep = [{int(l): set(v) for l, v in json.loads(Path(f).read_text())["keep"].items()} for f in (k160, md)]
    r160 = np.array([[e not in keep[0][l] for e in range(EXPERTS)] for l in range(LAYERS)])
    rmd = np.array([[e in keep[0][l] and e not in keep[1][l] for e in range(EXPERTS)] for l in range(LAYERS)])
    return r160, rmd


def shares(a, r160, rmd):
    """The shares of a (43, 216) array of picks or weights on the removed sets."""
    a = np.asarray(a, dtype=float)
    part = lambda mask, rows: float(a[rows][mask[rows]].sum() / a[rows].sum()) if a[rows].sum() else float("nan")
    return {"k160": part(r160, slice(None)), "k160_hash": part(r160, slice(0, HASH_LAYERS)),
            "k160_rest": part(r160, slice(HASH_LAYERS, None)), "md": part(rmd, slice(None))}


def total(rows, key):
    return sum(np.asarray(r[key], dtype=float) for r in rows)


def basis(rows):
    """The weight sums when every row has them, else the picks."""
    return "weight" if all("weight" in r for r in rows) else "counts"


def group(rows, r160, rmd):
    """The shares of a group of capture rows, and its mean removed picks at the probe position (of 43 x 6)."""
    g = {"rows": len(rows), "tokens": sum(r["prompt_tokens"] for r in rows),
         "picks": shares(total(rows, "counts"), r160, rmd), "tail": shares(total(rows, "tail_counts"), r160, rmd)}
    if basis(rows) == "weight":
        g["weight"] = shares(total(rows, "weight"), r160, rmd)
    layer = np.arange(LAYERS)[:, None]
    g["last_k160"] = float(np.mean([r160[layer, np.asarray(r["last"])].sum() for r in rows]))
    g["last_md"] = float(np.mean([rmd[layer, np.asarray(r["last"])].sum() for r in rows]))
    return g


def bot_group(path, r160, rmd):
    """The bot baseline: the capture rows of capture.py; the weight from the rows with no NaN slot."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    clean = [w for w in (np.asarray(r["weight"], dtype=float) for r in rows if r.get("weight") is not None)
             if np.isfinite(w).all()]
    g = {"rows": len(rows), "tokens": sum(r["prompt_tokens"] for r in rows),
         "picks": shares(total(rows, "counts"), r160, rmd)}
    if clean:
        g["weight"], g["weight_rows"] = shares(sum(clean), r160, rmd), len(clean)
    return g


def choose_x(rows, rmd, share=X_SHARE, cap=X_MAX):
    """X: the MD-extra experts in the order of their share in their layer on the rows, until X holds `share` of the
    pool's share, at most `cap` experts."""
    key = basis(rows)
    a = total(rows, key)
    s = a / np.maximum(a.sum(axis=1, keepdims=True), 1e-12)
    pool = sorted(((float(s[l, e]), int(l), int(e)) for l, e in zip(*np.nonzero(rmd))), reverse=True)
    whole, x, got = sum(p[0] for p in pool), [], 0.0
    for v, l, e in pool:
        if got >= share * whole or len(x) >= cap:
            break
        x.append([l, e])
        got += v
    return {"basis": key, "experts": x, "pool": len(pool), "covered": got / whole if whole else float("nan")}


def coverage(rows, x, rmd):
    """The share of the rows' MD-extra weight (or picks) that falls on X."""
    a, xm = total(rows, basis(rows)), np.zeros_like(rmd)
    for l, e in x:
        xm[l, e] = True
    return float(a[xm].sum() / a[rmd].sum()) if a[rmd].sum() else float("nan")


def decision(g1, g2, g3):
    text = ("G1 and G3 pass: build MD + X (§7), and run window 2 with all four points." if g1 and g3 else
            "G1 passes, G3 fails: the c3 experts are specific to c3. Window 2 runs stock5 only, or waits." if g1 else
            "G1 fails: the MD step does not remove experts that c3 needs. Do not build MD + X.")
    return text + (" G2 passes: note it for c2; stock5 still decides." if g2 else "")


def report(capture, out_dir=OUT_DIR, k160=K160, md=MD, bot=BOT):
    r160, rmd = removed_sets(k160, md)
    rows = [json.loads(line) for line in Path(capture).read_text().splitlines()]
    done = {r["key"]: r for r in rows if "error" not in r}            # a rerun appends: the last good row counts
    probes = [r for r in done.values() if r["kind"] == "probe"]
    pilot = [r for r in done.values() if r["kind"] == "pilot"]
    held = [r for r in pilot if r["name"] not in HELD_OUT]
    parts = [(f"probe {n}", [r for r in probes if r["name"] == n]) for n in dict.fromkeys(r["name"] for r in probes)]
    parts += [(f"pilot {f}", [r for r in pilot if r["family"] == f]) for f in sorted({r["family"] for r in pilot})]
    parts += [("pilot all", pilot), ("pilot held-out", held)]
    groups = {name: group(rs, r160, rmd) for name, rs in parts if rs}
    groups["bot"] = bot = bot_group(bot, r160, rmd)
    c2, c3 = groups.get("probe c2"), groups.get("probe c3")
    g1 = bool(c3) and max(c3["picks"]["md"], c3["tail"]["md"]) >= 2 * bot["picks"]["md"]
    g2 = bool(c2) and max(c2["picks"]["k160_rest"], c2["tail"]["k160_rest"]) >= 2 * bot["picks"]["k160_rest"]
    x = choose_x(held, rmd) if held else {"basis": None, "experts": [], "pool": int(rmd.sum()), "covered": float("nan")}
    cov = coverage([r for r in probes if r["name"] == "c3"], x["experts"], rmd) if c3 and x["experts"] else float("nan")
    res = {"capture": str(capture), "rows": len(rows), "missing": sorted({r["key"] for r in rows} - set(done)),
           "groups": groups, "x": x, "rules": {"G1": g1, "G2": g2, "G3": cov >= 0.5, "g3_coverage": cov},
           "decision": decision(g1, g2, cov >= 0.5)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(res, indent=1))
    (out_dir / "report.md").write_text(markdown(res))
    print(res["decision"])
    return res


def markdown(res):
    pct = lambda v: "–" if v is None or v != v else f"{100 * v:.2f}%"
    g_bot, x, rules = res["groups"]["bot"], res["x"], res["rules"]
    lines = ["# Junk route capture: report", "",
             f"Capture: `{res['capture']}`, {res['rows']} rows; keys with no good row: {len(res['missing'])}. "
             f"Picks are the top-6 slots of the stock router; the weight is the router weight of the sums file.", "",
             "| Group | Rows | Tokens | K160 picks | K160, layers 0-2 | K160, layers 3-42 | K160 3-42, last 256 "
             "| MD-extra picks | MD-extra, last 256 | K160 weight | MD-extra weight "
             "| Removed picks at the probe position (K160 / MD-extra, of 258) |", "|---" * 12 + "|"]
    for name, g in res["groups"].items():
        p, t, w = g["picks"], g.get("tail", {}), g.get("weight", {})
        last = f"{g['last_k160']:.1f} / {g['last_md']:.1f}" if "last_k160" in g else "–"
        lines.append(f"| {name} | {g['rows']} | {g['tokens']:,} | {pct(p['k160'])} | {pct(p['k160_hash'])} | "
                     f"{pct(p['k160_rest'])} | {pct(t.get('k160_rest'))} | {pct(p['md'])} | {pct(t.get('md'))} | "
                     f"{pct(w.get('k160'))} | {pct(w.get('md'))} | {last} |")
    yes = lambda v: "PASS" if v else "FAIL"
    lines += ["", "## The rules (plans/PLAN_JUNK_CAPTURE.md §3)", "",
              f"- G1 (c3, MD-extra picks of the prompt or of its last 256 rows, at least "
              f"{pct(2 * g_bot['picks']['md'])}, twice the bot share): {yes(rules['G1'])}",
              f"- G2 (c2, K160 picks in layers 3-42, at least {pct(2 * g_bot['picks']['k160_rest'])}): "
              f"{yes(rules['G2'])}",
              f"- X: {len(x['experts'])} of the {x['pool']} MD-extra experts "
              f"({len(x['experts']) * MIB_PER_EXPERT / 1024:.2f} GiB), chosen by {x['basis']} on the held-out "
              f"pilot tasks; they hold {pct(x['covered'])} of the pool's share there",
              f"- G3 (X holds at least 50% of c3's MD-extra {x['basis'] or 'weight'}): {pct(rules['g3_coverage'])}, "
              f"{yes(rules['G3'])}", "", f"**Decision (§6):** {res['decision']}", ""]
    return "\n".join(lines)


def select(report_json=OUT_DIR / "report.json", cold=COLD_MD, out=S / "results/cold-mdx.json"):
    """The drop file of MD + X: the drops of cold-md.json without the experts of X."""
    x = json.loads(Path(report_json).read_text())["x"]["experts"]
    drops = {int(l): set(v) for l, v in json.loads(Path(cold).read_text()).items()}
    for l, e in x:
        if e not in drops[l]:
            raise ValueError(f"layer {l}: expert {e} is not a drop of {cold}")
        drops[l].discard(e)
    Path(out).write_text(json.dumps({str(l): sorted(v) for l, v in sorted(drops.items())}))
    print(f"select: {len(x)} experts back ({len(x) * MIB_PER_EXPERT / 1024:.2f} GiB) in {out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "capture"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", default="http://127.0.0.1:8890")
        p.add_argument("--model", help="the served model name (default: the server's first model)")
        p.add_argument("--sums", type=Path, help="the routing_sums.npz of the patched capture container")
    c = sub.choices["capture"]
    c.add_argument("--out", type=Path, required=True, help="the rows JSONL; a rerun skips the rows in it")
    c.add_argument("--prompts", type=Path, help=f"the probe prompts (for example {PROMPTS})")
    c.add_argument("--rounds", type=int, default=3)
    c.add_argument("--pilot", type=Path, help=f"the folder of the pilot transcripts (for example {PILOT})")
    c.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("report")
    r.add_argument("--capture", type=Path, default=OUT_DIR / "capture.jsonl")
    r.add_argument("--out-dir", type=Path, default=OUT_DIR)
    r.add_argument("--k160", type=Path, default=K160, help="PRUNE.json of the K 160 prune")
    r.add_argument("--md", type=Path, default=MD, help="PRUNE.json of the MD prune")
    r.add_argument("--bot", type=Path, default=BOT, help="the bot capture rows of capture.py")
    s = sub.add_parser("select")
    s.add_argument("--report", type=Path, default=OUT_DIR / "report.json")
    s.add_argument("--cold", type=Path, default=COLD_MD)
    s.add_argument("--out", type=Path, default=S / "results/cold-mdx.json")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        report(a.capture, a.out_dir, a.k160, a.md, a.bot)
        return 0
    if a.cmd == "select":
        return select(a.report, a.cold, a.out)
    if a.cmd == "smoke":
        return smoke(a.base_url, a.model, a.sums)
    if not (a.prompts or a.pilot):
        ap.error("capture needs --prompts, --pilot or both")
    return 1 if capture(a.base_url, a.out, a.model, a.prompts, a.rounds, a.pilot, a.seed, a.sums) else 0


if __name__ == "__main__":
    sys.exit(main())
