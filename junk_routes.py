"""The route capture of the junk prompts (plans/PLAN_JUNK_CAPTURE.md §3 and §4).

  smoke    [--base-url U] [--model M] [--sums S]
  capture  --out F [--base-url U] [--model M] [--prompts P] [--rounds 3] [--pilot DIR] [--seed 0] [--sums S]

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

S = Path(__file__).resolve().parent
PROMPTS = S / "results/junk-bisect/prompts.json"
PILOT = S / "results/agent-eval/pilot"
AGENT_EVAL = Path.home() / "Projects/dsv4-specialize-agent-eval/agent-eval"
TAIL = 256          # the rows before the probe position: the quote and the JSON around it
LAYERS, TOP_K = 43, 6
# The smoke prompt must run eagerly: a step of 24 tokens or fewer replays a CUDA graph (MAX_CUDAGRAPH_CAPTURE_SIZE of
# compose-capture.yml), and the patch writes the sums of such a step only at the start of the next step. "Say OK." alone
# (about 4 tokens) failed the smoke of 2026-11-16 22:11 for that reason. This prompt has more than 100 tokens.
SMOKE_PROMPT = " ".join(["Say OK."] * 40)


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
    a = ap.parse_args(argv)
    if a.cmd == "smoke":
        return smoke(a.base_url, a.model, a.sums)
    if not (a.prompts or a.pilot):
        ap.error("capture needs --prompts, --pilot or both")
    return 1 if capture(a.base_url, a.out, a.model, a.prompts, a.rounds, a.pilot, a.seed, a.sums) else 0


if __name__ == "__main__":
    sys.exit(main())
