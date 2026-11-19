"""The client of colinfer_floor.sh (colinfer docs/history/PLAN_QWEN35_MOE.md §5, step 2): the long prompt, the memory
snapshot of a colinfer server, and the report of the window.

- ``--long N``: one chat request with a prompt of about N tokens and 200 output tokens. colinfer has no ``/tokenize``;
  ``/v1/messages/count_tokens`` gives the size of the paragraph.
- ``--snap``: one JSON line with the memory of the server now: the fall of the available memory since ``--before``,
  the bytes of the torch allocator (``/v1/status``), the resident memory of process ``--pid`` (Pss, anonymous and
  file-backed) and the memory of its CUDA context (nvidia-smi).
- ``--report DIR``: the snapshots of the points of ``--points``, in GB (1e9 bytes, as in the colinfer docs), as text
  lines and as JSON in ``--out``. ``outside`` is the fall minus the bytes that the torch allocator holds: the CUDA
  context, the host memory of the process and the pinned buffers.
"""

import argparse
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from qwen_fit import PARAGRAPH, post

GB, MIB = 10**9, 2**20
PARTS = ("fall", "allocated", "reserved", "anon", "file", "gpu")


def long_text(base, model, tokens):
    """A text of about ``tokens`` tokens (at most), from the size of 50 paragraphs."""
    count = lambda n: post(base, "/v1/messages/count_tokens",
                           {"model": model, "messages": [{"role": "user", "content": PARAGRAPH * n}]})["input_tokens"]
    return PARAGRAPH * int((tokens - 200) / ((count(51) - count(1)) / 50))


def long_prompt(base, model, tokens, max_tokens=200):
    text = long_text(base, model, tokens)
    began = time.monotonic()
    resp = post(base, "/v1/chat/completions", {
        "model": model, "max_tokens": max_tokens, "temperature": 0,
        "messages": [{"role": "user", "content": text + "\nSummarize the report above in three sentences."}]})
    return {"prompt_tokens": resp["usage"]["prompt_tokens"], "completion_tokens": resp["usage"]["completion_tokens"],
            "seconds": round(time.monotonic() - began, 1)}


def pss(text):
    """``(anonymous, file-backed)`` bytes of a ``smaps_rollup`` text, by Pss."""
    f = {k: int(v) * 1024 for k, v in re.findall(r"^(Pss_Anon|Pss_File|Pss_Shmem):\s+(\d+) kB", text, re.M)}
    return f.get("Pss_Anon", 0), f.get("Pss_File", 0) + f.get("Pss_Shmem", 0)


def gpu_mib(text, pid):
    """The MiB of process ``pid`` in ``nvidia-smi --query-compute-apps=pid,used_memory`` (csv, no units); 0 if absent."""
    return next((int(m) for p, m in re.findall(r"^(\d+), (\d+)\s*$", text, re.M) if int(p) == pid), 0)


def snapshot(base, pid, when, before_mb):
    """The memory of the server now, in bytes. ``before_mb`` is the available memory before the boot (``free -m``)."""
    with urllib.request.urlopen(base + "/v1/status", timeout=30) as resp:
        torch = json.load(resp)["memory_gb"]
    avail_kb = int(re.search(r"^MemAvailable:\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.M).group(1))
    anon, file = pss(Path(f"/proc/{pid}/smaps_rollup").read_text())
    smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {"when": when, "epoch": int(time.time()), "before_mb": before_mb, "fall": before_mb * MIB - avail_kb * 1024,
            "allocated": int(torch["allocated"] * GB), "reserved": int(torch["reserved"] * GB), "anon": anon, "file": file,
            "gpu": gpu_mib(smi, pid) * MIB}


def report(results, points):
    """The rows of the report, in GB: one for each snapshot of a point, and one for its peak (the lowest sample of the
    available memory from the first snapshot on)."""
    rows = []
    for p in points:
        f = Path(results) / f"colinfer-floor-{p}-snap.jsonl"
        snaps = [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []
        for s in snaps:
            rows.append({"point": p, "when": s["when"], **{k: round(s[k] / GB, 2) for k in PARTS},
                         "outside": round((s["fall"] - s["reserved"]) / GB, 2)})
        mem = Path(results) / f"colinfer-floor-{p}-mem.log"
        if snaps and mem.exists():
            samples = [(int(a), int(b)) for a, b in (line.split() for line in mem.read_text().splitlines() if line.strip())]
            low = min([m for e, m in samples if e >= snaps[0]["epoch"]], default=None)
            if low is not None:
                rows.append({"point": p, "when": "peak", "fall": round((snaps[0]["before_mb"] - low) * MIB / GB, 2)})
    return rows


def line(r):
    if r["when"] == "peak":
        return f"{r['point']} peak: fall {r['fall']:.2f} GB (the lowest sample of the available memory)"
    return (f"{r['point']} {r['when']}: fall {r['fall']:.2f} GB; torch {r['allocated']:.2f} allocated, {r['reserved']:.2f} reserved; "
            f"outside the allocator {r['outside']:.2f}; process {r['anon']:.2f} anonymous, {r['file']:.2f} file-backed; "
            f"GPU process {r['gpu']:.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8002")
    ap.add_argument("--model", default="colinfer-floor")
    cmd = ap.add_mutually_exclusive_group(required=True)
    cmd.add_argument("--long", type=int, metavar="TOKENS")
    cmd.add_argument("--snap", action="store_true")
    cmd.add_argument("--report", type=Path, metavar="RESULTS_DIR")
    ap.add_argument("--pid", type=int)
    ap.add_argument("--when", default="ready")
    ap.add_argument("--before", type=int, help="the available memory before the boot, in MB of free -m")
    ap.add_argument("--points", default="f1 f2 f3 f4")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.long:
        print(json.dumps(long_prompt(a.base_url, a.model, a.long)))
    elif a.snap:
        print(json.dumps(snapshot(a.base_url, a.pid, a.when, a.before)))
    else:
        rows = report(a.report, a.points.split())
        if a.out:
            a.out.write_text(json.dumps(rows, indent=1) + "\n")
        print("\n".join(line(r) for r in rows))


if __name__ == "__main__":
    main()
