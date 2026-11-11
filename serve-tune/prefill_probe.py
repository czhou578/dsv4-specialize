"""The prefill probe and the smoke requests of PLAN_SERVE_TUNE.md §6.3.

- Cold: a prompt of about N tokens. It starts with a nonce, so the prefix cache cannot hit. A secret code near the
  start and a question at the end check the output (a broken prefill kernel loses the code).
- Warm: a cold prompt of BASE tokens, then the same prompt plus EXTRA new tokens. The second request must hit the
  prefix cache: it prefills EXTRA tokens on top of a deep context, which is the path of the locked MLA workspace.
- Smoke: a reply with a JSON schema, and two tool calls (tool_choice auto and required).

Each prefill request records the time to the first token and the lowest available memory while it runs. The result
is one JSON object (stdout, and --out); stderr gets a one-line summary. The exit code is 0 only if all checks pass.

  python prefill_probe.py --base-url http://127.0.0.1:8890 --cold 8000 32000 128000 --warm 200000 8000 --smoke
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ERRORS = (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError, AttributeError,
          ZeroDivisionError)
QUOTE = {"type": "object", "properties": {"ticker": {"type": "string"}, "price": {"type": "number"}},
         "required": ["ticker", "price"], "additionalProperties": False}
TOOL = {"type": "function", "function": {
    "name": "get_quote", "description": "Get the last price of a stock.",
    "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}}
NO_THINKING = {"thinking": False}


def get(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as resp:
        return json.load(resp)


def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def stream(base, body, timeout=3600):
    """Yield (seconds since the request, event) for each server-sent event of a chat completion."""
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                yield time.perf_counter() - start, json.loads(line[6:])


def healthy(base):
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def mem_available_mib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return -1


class LowestMemory:
    """The lowest available memory (MiB) while the block runs, read once per second."""

    def __enter__(self):
        self.low, self._stop = mem_available_mib(), threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def _watch(self):
        while not self._stop.wait(1.0):
            self.low = min(self.low, mem_available_mib())

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.low = min(self.low, mem_available_mib())


def prompt(nonce, units, extra=0):
    """The probe text and its secret code: `units` filler words, then `extra` other filler words."""
    code = f"QX-{nonce % 100000:05d}"
    text = (f"Probe {nonce}. The secret code is {code}. Read the filler, then answer the last line.\n"
            + " x" * units + " y" * extra + "\nWhat is the secret code? Reply with the code only.")
    return text, code


def calibrate(base, model, nonce):
    """(tokens of a prompt with no filler, tokens per filler word), from two /tokenize calls."""
    def count(units):
        body = {"model": model, "messages": [{"role": "user", "content": prompt(nonce, units)[0]}],
                "add_special_tokens": True, "chat_template_kwargs": NO_THINKING}
        return int(post(base, "/tokenize", body)["count"])
    empty = count(0)
    return empty, (count(4096) - empty) / 4096


def ask(base, model, text, max_tokens=24):
    """Stream one request; the time to the first content token, the token counts and the lowest memory."""
    body = {"model": model, "messages": [{"role": "user", "content": text}], "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "max_completion_tokens": max_tokens,
            "chat_template_kwargs": NO_THINKING}
    first, parts, usage = None, [], {}
    with LowestMemory() as memory:
        for at, event in stream(base, body):
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    first = at if first is None else first
                    parts.append(piece)
    tokens = int(usage.get("prompt_tokens", 0))
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    return {"prompt_tokens": tokens, "cached_tokens": cached,
            "ttft_s": None if first is None else round(first, 3),
            "prefill_tok_s": round((tokens - cached) / first, 1) if first else None,
            "reply": "".join(parts)[:80], "min_avail_mib": memory.low}


def checked(base, run):
    """Run one check; an exception or a server that is not healthy afterwards fails it."""
    try:
        result = run()
    except ERRORS as error:
        result = {"pass": False, "error": f"{type(error).__name__}: {error}"[:300]}
    result["healthy"] = healthy(base)
    result["pass"] = result["pass"] and result["healthy"]
    return result


def cold(base, model, tokens, nonce, cal):
    def run():
        text, code = prompt(nonce, max(0, round((tokens - cal[0]) / cal[1])))
        r = ask(base, model, text)
        r.update(target=tokens, recall=code in r["reply"])
        r["pass"] = r["recall"] and r["cached_tokens"] < 256
        return r
    return checked(base, run)


def warm(base, model, tokens, extra, nonce, cal):
    def run():
        units = max(0, round((tokens - cal[0]) / cal[1]))
        text, code = prompt(nonce, units)
        first = ask(base, model, text)
        second = ask(base, model, prompt(nonce, units, round(extra / cal[1]))[0])
        hit = second["cached_tokens"] >= 0.9 * first["prompt_tokens"]
        recall = code in first["reply"] and code in second["reply"]
        return {"first": first, "second": second, "hit": hit, "recall": recall, "pass": hit and recall}
    return checked(base, run)


def smoke(base, model):
    def chat(content, **extra):
        body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0,
                "max_completion_tokens": 256, "chat_template_kwargs": NO_THINKING, **extra}
        return post(base, "/v1/chat/completions", body)["choices"][0]["message"]

    def json_schema():
        schema = {"type": "json_schema", "json_schema": {"name": "quote", "schema": QUOTE, "strict": True}}
        obj = json.loads(chat("Give the ticker NVDA with the price 123.4 as JSON.", response_format=schema)["content"])
        return {"pass": isinstance(obj.get("ticker"), str) and isinstance(obj.get("price"), (int, float))}

    def tool_call(choice):
        calls = chat("What is the last price of NVDA? Use the tool.", tools=[TOOL],
                     tool_choice=choice).get("tool_calls") or []
        args_ok = all(isinstance(json.loads(c["function"]["arguments"]), dict) for c in calls)
        return {"calls": len(calls), "pass": args_ok and (bool(calls) or choice == "auto")}

    checks = {"json_schema": checked(base, json_schema),
              "tool_auto": checked(base, lambda: tool_call("auto")),
              "tool_required": checked(base, lambda: tool_call("required"))}
    return {**checks, "pass": all(c["pass"] for c in checks.values())}


def summary(result):
    words = [f"cold {r.get('target')}: {r.get('prompt_tokens')} tok, ttft {r.get('ttft_s')} s, "
             f"{r.get('prefill_tok_s')} tok/s, min {r.get('min_avail_mib')} MiB, {'pass' if r['pass'] else 'FAIL'}"
             for r in result["cold"]]
    w = result["warm"]
    if w:
        s = w.get("second", {})
        words.append(f"warm: cached {s.get('cached_tokens')} of {w.get('first', {}).get('prompt_tokens')}, "
                     f"ttft {s.get('ttft_s')} s, {'pass' if w['pass'] else 'FAIL'}")
    if result["smoke"]:
        words.append("smoke: " + " ".join(f"{k} {'pass' if v['pass'] else 'FAIL'}"
                                          for k, v in result["smoke"].items() if k != "pass"))
    return f"{'PASS' if result['pass'] else 'FAIL'}; " + "; ".join(words)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8890")
    parser.add_argument("--model", help="served model name (default: the server's first model)")
    parser.add_argument("--cold", type=int, nargs="*", default=[], metavar="TOKENS")
    parser.add_argument("--warm", type=int, nargs=2, metavar=("BASE", "EXTRA"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    base = args.base_url
    model = args.model or get(base, "/v1/models")["data"][0]["id"]
    nonce = time.time_ns()
    cal = calibrate(base, model, nonce) if args.cold or args.warm else None
    result = {"model": model, "calibration": cal,
              "cold": [cold(base, model, n, nonce + 1 + i, cal) for i, n in enumerate(args.cold)],
              "warm": warm(base, model, *args.warm, nonce, cal) if args.warm else None,
              "smoke": smoke(base, model) if args.smoke else None}
    parts = result["cold"] + [p for p in (result["warm"], result["smoke"]) if p]
    result["pass"] = all(p["pass"] for p in parts)
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    print(summary(result), file=sys.stderr)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
