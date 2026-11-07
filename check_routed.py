"""PLAN_MODEL_SPECIALIZE.md §5.2 check: no dropped expert appears in the routing of a mask test boot.

Boot the mask with ``compose-capture.yml`` (Model Runner V2 rejects the capture flag, so
``compose-mask.yml`` cannot do it): ``CAPTURE_DIR=$PWD/tp1-mask-k192 docker compose -f compose-capture.yml up -d``.
Then run ``python check_routed.py --keep 192``. It sends a few prompts to the test server and decodes
``choices[0].routed_experts``: a base64 ``.npy`` array of shape (prompt + output tokens - 1,
43 layers, 6), one row per token that went through the model (vLLM ``chat_completion/protocol.py``).

The capture buffer starts at zero in each step, so a layer whose router never captures shows rows
of six equal ids. The check passes when:

- every layer captured: no row of six equal ids in layers 3-42;
- no dropped id appears in layers 3-42;
- in layers 0-2, each row holds the experts of the mask's ``tid2eid`` row for its token. This also
  shows that the server loaded the edited table.
"""

import argparse
import base64
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

import mask_checkpoint as mc

PROMPTS = (
    "List the three largest risks in the 10-K of a semiconductor equipment maker, one line each.",
    "Write a Python function that computes the 14-day RSI of a list of closing prices.",
    "Explain the difference between the P/E ratio and the EV/EBITDA multiple in four sentences.",
    "A stock trades at $156.74 after a 4.4% weekly gain. What was the price a week ago? Show the steps.",
)


def decode(b64):
    return np.load(io.BytesIO(base64.b64decode(b64)))


def check(routed, token_ids, drops, tid2eid):
    """Problems per layer for one reply. ``routed``: (tokens - 1, layers, top-k); ``token_ids``: the
    prompt and output ids; ``drops``: {layer: dropped ids}; ``tid2eid``: {hash layer: table}."""
    rows = routed.shape[0]
    if rows != len(token_ids) - 1:
        raise ValueError(f"{rows} routing rows for {len(token_ids)} tokens")
    tokens = np.asarray(token_ids[:rows])
    out = {}
    for layer in range(routed.shape[1]):
        ids = routed[:, layer, :].astype(np.int64)
        if layer in tid2eid:
            want = np.sort(tid2eid[layer][tokens], axis=1)
            problems = {"hash_mismatch": int((np.sort(ids, axis=1) != want).any(axis=1).sum())}
        else:
            problems = {"uncaptured": int((ids == ids[:, :1]).all(axis=1).sum()),
                        "dropped": int(np.isin(ids, drops.get(layer, [])).sum())}
        out[layer] = {"distinct": len(np.unique(ids)), **{k: v for k, v in problems.items() if v}}
    return out


def post(base, body):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--keep", type=int, required=True, help="K of the mask folder")
    parser.add_argument("--mask", type=Path, help="default: tp1-mask-k{K} next to this script")
    parser.add_argument("--base-url", default="http://127.0.0.1:8890")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    folder = args.mask or mc.OUT_ROOT / f"tp1-mask-k{args.keep}"
    record = json.loads((folder / "MASK.json").read_text())
    drops = {int(k): v["drop"] for k, v in record["layers"].items()}
    weight_map = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    tid2eid = {layer: mc.read_tensor(folder / weight_map[name], name)
               for layer, name in ((i, f"layers.{i}.ffn.gate.tid2eid") for i in range(mc.HASH_LAYERS))}
    with urllib.request.urlopen(args.base_url + "/v1/models", timeout=30) as resp:
        model = json.load(resp)["data"][0]["id"]
    failed, total = False, 0
    for prompt in PROMPTS:
        reply = post(args.base_url, {"model": model, "messages": [{"role": "user", "content": prompt}],
                                     "max_tokens": args.max_tokens, "temperature": 0,
                                     "return_token_ids": True})
        choice = reply["choices"][0]
        if not choice.get("routed_experts"):
            sys.exit("no routed_experts in the reply: boot compose-capture.yml")
        routed = decode(choice["routed_experts"])
        result = check(routed, reply["prompt_token_ids"] + choice["token_ids"], drops, tid2eid)
        bad = {layer: r for layer, r in result.items() if len(r) > 1}
        failed |= bool(bad)
        total += routed.shape[0]
        distinct = [r["distinct"] for layer, r in result.items() if layer >= mc.HASH_LAYERS]
        print(f"{routed.shape[0]:>5} rows, {routed.shape[1]} layers; experts used per layer "
              f"{min(distinct)}-{max(distinct)}; problems: {bad or 'none'}")
    print(f"{'FAIL' if failed else 'PASS'}: {total} rows; K={record['keep']}, no dropped expert "
          f"may appear and the hash layers must match the mask's tid2eid")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
