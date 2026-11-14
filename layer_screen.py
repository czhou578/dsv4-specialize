"""The client of the probe boot (plans/PLAN_FEWER_LAYERS.md §6.3 and §7): pick the data, measure the Block Influence
(BI) of each block, screen skip sets by their NLL, and run the greedy rounds.

The probe boot (``skip_probe.sh``: eager, ``LAYER_PROBE=1``, the prefix cache off) reads the skip set from
``<skip-dir>/skip.json`` at the start of each model step when the file changes, and writes the probe sums to
``<skip-dir>/layer_probe.npz`` after each step. The client writes the file before each request and reads the sums
before and after it: the difference is the request's own sums.

- ``--pick``: the screen set (about 150k prompt tokens, short targets) and the probe set (about 500k tokens), from the
  calibration split of the replay's data. The two sets share no segment, and no held-out (replay) session is in them.
- ``--bi``: one pass over the probe set with a skip set: the BI of each block, 1 minus the mean cosine between the
  state before and after the block. The shared prefix of a pass counts once (``count_from`` in the skip file).
- ``--screen``: the NLL of the screen set for each skip set, scored as the replay scores a target.
- ``--greedy``: the order of §7.2: R, the BI pass, round 1, the Gram sums, rounds 2 and 3, R2. Each result is one
  JSONL row in ``--log``; a rerun skips the rows that are there. A step that cannot end before ``--deadline`` is
  skipped, and R2 keeps its time.
- ``--report``: the tables of §7.5 and §7.6 from the log.

The heal fit (``--fit``, §9) comes with the heal patch, only if Phase B or C asks for it.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

EVAL = Path.home() / "Projects/stock-research-model-eval"
sys.path.insert(0, str(EVAL))
from model_eval.replay import DATA, SOURCES, post, request_body, score  # noqa: E402

KINDS = ("A", "F", "L")      # the columns of layer_probe.npz (image-patch/layer_skip.py)
FIRST, LAST = 3, 39          # the candidate layers: not the hash layers 0-2, not 40-42 (the draft reads them)
CHARS_PER_TOKEN = 4.0        # json chars of the messages and tools per prompt token: 3.77 to 4.44 (capture-k216)
ROUND1 = {"L": 10, "F": 10, "A": 8}


def canon(blocks):
    """The canonical text of a skip set, by layer (as layer_skip.canon): ``"F12,L17"``, or ``""`` for no skip."""
    return ",".join(sorted(blocks, key=lambda b: (int(b[1:]), b[0])))


def blocks_of(skip):
    return [b for b in skip.split(",") if b]


def clash(block, skip):
    """True when ``block`` cannot join ``skip``: its layer is in the set already."""
    return any(b[1:] == block[1:] for b in blocks_of(skip))


# --- the data ---------------------------------------------------------------------------------------------------

def estimate(session, seg, i, tools_chars):
    chars = len(session["system_prompt"] or "") + tools_chars + sum(len(json.dumps(m)) for m in seg[: i + 1])
    return int(chars / CHARS_PER_TOKEN)


def pick(sessions, tools, screen_tokens=150_000, probe_tokens=500_000, screen_max=30_000, probe_max=70_000):
    """The screen set and the probe set: ``[session, segment, index, estimated prompt tokens]`` each, one target per
    segment, one skill after another. The screen set takes the longest target turns of each skill first (the most
    scored tokens for its prompt tokens), the probe set a fixed hash order of the other segments."""
    names = {t["function"]["name"] for t in tools}
    held = {s["id"] for s in sessions if s["split"] != "calibration"}
    tools_chars = len(json.dumps(tools))
    by_skill = defaultdict(list)
    for s in sessions:
        if s["split"] != "calibration" or s["source"] not in SOURCES:
            continue
        for g, seg in enumerate(s["segments"]):
            for i, m in enumerate(seg):
                calls = {c["function"]["name"] for c in m.get("tool_calls", [])}
                if m.get("target") and calls <= names:
                    turn = len(m.get("reasoning_content") or "") + len(json.dumps(m.get("content") or "")) + \
                        len(json.dumps(m.get("tool_calls", [])))
                    h = hashlib.sha256(f"{s['id']}:{g}:{i}".encode()).hexdigest()
                    by_skill[s["skill"]].append((s["id"], g, i, estimate(s, seg, i, tools_chars), turn, h))
    used, out = set(), {}
    for name, budget, cap in (("screen", screen_tokens, screen_max), ("probe", probe_tokens, probe_max)):
        order = (lambda r: (-r[4], r[5])) if name == "screen" else (lambda r: r[5])
        out[name], total, queues = [], 0, {k: sorted(v, key=order) for k, v in sorted(by_skill.items())}
        while total < budget and any(queues.values()):
            for q in queues.values():
                while q and (q[0][:2] in used or q[0][3] > cap):
                    q.pop(0)
                if q and total < budget:
                    row = q.pop(0)
                    used.add(row[:2])
                    out[name].append(list(row[:4]))
                    total += row[3]
    assert not {r[0] for rs in out.values() for r in rs} & held
    return out


# --- the skip file and the probe sums ----------------------------------------------------------------------------

class SkipFile:
    """``<dir>/skip.json``, written atomically, with a new mtime on each write (the patch reads it on a change)."""

    def __init__(self, folder):
        self.path, self.last = Path(folder) / "skip.json", 0
        self.flush = int(time.time() * 1000)   # a new start value per client, so a restarted client still flushes

    def write(self, skip="", count_from=0, sums=True, gram=(), flush=False):
        self.flush += bool(flush)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"skip": skip, "count_from": count_from, "sums": sums, "gram": list(gram),
                                   "gram_flush": self.flush}))
        mtime = max(tmp.stat().st_mtime_ns, self.last + 1_000_000)   # a coarse clock can repeat an mtime
        os.utime(tmp, ns=(mtime, mtime))
        os.replace(tmp, self.path)
        self.last = mtime


def read_probe(folder):
    path = Path(folder) / "layer_probe.npz"
    for _ in range(50):   # the rename is atomic; this guards a read during a write on another filesystem
        try:
            with np.load(path) as z:
                return {k: z[k].copy() for k in z.files}
        except FileNotFoundError:
            return None
        except (ValueError, EOFError, OSError):
            time.sleep(0.1)
    raise RuntimeError(f"{path}: cannot read the probe sums")


def check(after, skip, resp):
    """The checks of §6.2 and §7.5 after a request: the patch runs ``skip``, read it with no error, and the prefix cache
    gave no token."""
    if after is None:
        raise RuntimeError("no layer_probe.npz: is the probe boot (skip_probe.sh) running?")
    if str(after["error"]):
        raise RuntimeError(f"the patch refused the skip file: {after['error']}")
    if str(after["skip"]) != (skip or "none"):
        raise RuntimeError(f"the patch runs {after['skip']}, not {skip or 'none'}")
    cached = ((resp.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    if cached:
        raise RuntimeError(f"{cached} cached prompt tokens: the prefix cache must be off in the probe boot")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    here = Path(__file__).resolve().parent
    ap.add_argument("--sets", type=Path, default=here / "results/skip-sets.json")
    cmd = ap.add_mutually_exclusive_group(required=True)
    for name in ("pick",):
        cmd.add_argument(f"--{name}", action="store_true")
    a = ap.parse_args(argv)
    if a.pick:
        sessions = [json.loads(line) for line in (DATA / "sessions.jsonl").open()]
        sets = pick(sessions, json.loads((DATA / "tools.json").read_text())["tools"])
        a.sets.write_text(json.dumps(sets))
        for k, v in sets.items():
            print(f"{k}: {len(v)} targets, about {sum(r[3] for r in v)} prompt tokens -> {a.sets}")
        return


if __name__ == "__main__":
    main()
