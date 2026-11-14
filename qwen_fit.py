"""The client of qwen_fit.sh (plans/PLAN_SECOND_MODEL_FIT.md §4): the smoke requests, the long prompt, and the memory
report of one point.

- ``--smoke``: one short request and one tool call (``tool_choice`` auto). The last line is PASS or FAIL.
- ``--long N``: one chat request with a prompt of about N tokens and 500 output tokens: the peak of a full context.
  The prompt repeats one paragraph; ``/tokenize`` gives its size.
- ``--report``: the numbers of §4.1 from the samples of the window, as JSON in ``--out`` and as text lines.
  The overhead X = the available memory before the boot - its lowest value - the loaded weights - the KV pin.
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

PARAGRAPH = ("The company reported revenue of 4.2 billion dollars for the quarter, up 12 percent from a year ago. "
             "Operating margin was 31.5 percent, and free cash flow reached 1.1 billion dollars. Management raised "
             "the full-year guidance and named supply limits in two product lines as the main risk. ")
TOOL = {"type": "function", "function": {
    "name": "get_quote", "description": "Get the latest price of a stock ticker.",
    "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}}


def post(base, path, body, timeout=3600):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def smoke(base, model):
    """``{check: bool}`` of the two smoke requests."""
    short = post(base, "/v1/chat/completions", {"model": model, "max_tokens": 200, "temperature": 0,
                                                "messages": [{"role": "user", "content": "Reply with the word OK."}]})
    msg = short["choices"][0]["message"]
    call = post(base, "/v1/chat/completions", {"model": model, "max_tokens": 600, "temperature": 0, "tools": [TOOL],
                                               "messages": [{"role": "user", "content":
                                                             "Use the tool to get the latest price of NVDA."}]})
    calls = call["choices"][0]["message"].get("tool_calls") or []
    return {"short": bool((msg.get("content") or "") + (msg.get("reasoning_content") or msg.get("reasoning") or "")),
            "tool_call": bool(calls) and calls[0]["function"]["name"] == "get_quote"}


def long_text(base, model, tokens):
    """A text of about ``tokens`` tokens (at most), from the size of one paragraph."""
    one = len(post(base, "/tokenize", {"model": model, "prompt": PARAGRAPH * 50})["tokens"]) / 50
    return PARAGRAPH * int((tokens - 200) / one)


def long_prompt(base, model, tokens, max_tokens=500):
    text = long_text(base, model, tokens)
    began = time.monotonic()
    resp = post(base, "/v1/chat/completions", {
        "model": model, "max_tokens": max_tokens, "temperature": 0,
        "messages": [{"role": "user", "content": text + "\nSummarize the report above in three sentences."}]})
    return {"prompt_tokens": resp["usage"]["prompt_tokens"], "completion_tokens": resp["usage"]["completion_tokens"],
            "seconds": round(time.monotonic() - began, 1)}


def rate(path):
    """The decode rate of the measured rows of a ``model_eval.speed`` file: all decoded tokens / all decode time. The
    summary of that tool fails on a server with no draft (it needs the draft counters)."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("run", 0) >= 1 and r.get("decode_tok_s")]
    tokens = sum(r["completion_tokens"] - 1 for r in rows)
    return tokens / sum((r["completion_tokens"] - 1) / r["decode_tok_s"] for r in rows), len(rows)


def report(before_mb, healthy_mb, pin_bytes, mem_log, server_log, gpu=None, healthy_at=0):
    """The memory numbers of one point, in GiB. ``mem_log`` holds ``epoch available_mb`` lines. The peak of the
    report counts the samples from ``healthy_at`` on; the lowest sample before it is the peak of the load."""
    samples = [(int(a), int(b)) for a, b in (line.split() for line in Path(mem_log).read_text().splitlines() if line.strip())]
    low = min([m for e, m in samples if e >= healthy_at] or [healthy_mb])
    load_low = min([m for e, m in samples if e < healthy_at] or [healthy_mb])
    log = Path(server_log).read_text(errors="replace")
    loaded = [float(x) for x in re.findall(r"Model loading took ([0-9.]+) ?GiB", log)]
    kv = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", log)
    gib = lambda mb: mb / 1024
    out = {"available_before": round(gib(before_mb), 2), "available_healthy": round(gib(healthy_mb), 2),
           "available_lowest": round(gib(low), 2), "used_at_peak": round(gib(before_mb - low), 2),
           "weights": max(loaded) if loaded else None, "kv_pin": round(pin_bytes / 2**30, 2),
           "kv_tokens": int(kv[-1].replace(",", "")) if kv else None, "load_peak": round(gib(before_mb - load_low), 2)}
    if out["weights"] is not None:
        out["overhead_x"] = round(out["used_at_peak"] - out["weights"] - out["kv_pin"], 2)
        out["overhead_idle"] = round(gib(before_mb - healthy_mb) - out["weights"] - out["kv_pin"], 2)
    if gpu and Path(gpu).exists():
        out["gpu_process_mib"] = [int(m) for m in re.findall(r", (\d+) MiB", Path(gpu).read_text())]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8889")
    ap.add_argument("--model", default="qwen3.8-27b-fit")
    cmd = ap.add_mutually_exclusive_group(required=True)
    cmd.add_argument("--smoke", action="store_true")
    cmd.add_argument("--long", type=int, metavar="TOKENS")
    cmd.add_argument("--report", action="store_true")
    cmd.add_argument("--rate", type=Path, metavar="SPEED_JSONL")
    for name, kind in (("before", int), ("healthy", int), ("healthy-at", int), ("pin", int), ("mem-log", Path), ("server-log", Path),
                       ("gpu", Path), ("out", Path)):
        ap.add_argument(f"--{name}", type=kind)
    a = ap.parse_args(argv)
    if a.smoke:
        checks = smoke(a.base_url, a.model)
        print(json.dumps(checks))
        print("PASS" if all(checks.values()) else "FAIL", "smoke:", " ".join(f"{k} {'pass' if v else 'fail'}"
                                                                           for k, v in checks.items()))
        raise SystemExit(0 if all(checks.values()) else 1)
    if a.long:
        print(json.dumps(long_prompt(a.base_url, a.model, a.long)))
        return
    if a.rate:
        print("%.2f tok/s over %d rows" % rate(a.rate))
        return
    r = report(a.before, a.healthy, a.pin, a.mem_log, a.server_log, a.gpu, a.healthy_at or 0)
    if a.out:
        a.out.write_text(json.dumps(r, indent=1) + "\n")
    print(f"weights {r['weights']} GiB; KV pin {r['kv_pin']} GiB ({r['kv_tokens']} tokens); used at the peak "
          f"{r['used_at_peak']} GiB; lowest available {r['available_lowest']} GiB; peak of the load {r['load_peak']} GiB")
    print(f"overhead X {r.get('overhead_x')} GiB at the peak, {r.get('overhead_idle')} GiB after the boot")


if __name__ == "__main__":
    main()
