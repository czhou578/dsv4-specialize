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


class Client:
    def __init__(self, base, model, skip_dir, data=DATA, effort="medium"):
        self.base, self.model, self.dir, self.effort = base, model, Path(skip_dir), effort
        with (data / "sessions.jsonl").open() as f:
            self.sessions = {s["id"]: s for s in map(json.loads, f)}
        self.tools = json.loads((data / "tools.json").read_text())["tools"]
        self.file = SkipFile(skip_dir)
        self._ids = None

    def body(self, sid, g, i, logprobs):
        s = self.sessions[sid]
        body = request_body(self.model, s, s["segments"][g], i, self.tools, self.effort)
        if not s["system_prompt"]:
            body["messages"] = body["messages"][1:]
        if not logprobs:
            del body["prompt_logprobs"]
        return body

    def tokenize(self, messages):
        return post(self.base, "/tokenize", {"model": self.model, "messages": messages, "tools": self.tools,
                                             "add_generation_prompt": False,
                                             "chat_template_kwargs": {"reasoning_effort": self.effort}})["tokens"]

    def bi(self, items, skip="", gram=()):
        """One pass over ``items`` with ``skip``: the summed cos, rel and tokens of each block (the shared prefix of
        the pass counts once)."""
        seen, total = set(), None
        for sid, g, i, _ in items:
            body = self.body(sid, g, i, logprobs=False)
            prompt = self.tokenize(body["messages"])
            system = [m for m in body["messages"] if m["role"] == "system"] or [{"role": "system", "content": ""}]
            shared = self.tokenize(system)
            n = next((k for k, (a, b) in enumerate(zip(shared, prompt)) if a != b), min(len(shared), len(prompt)))
            key = hashlib.sha256(json.dumps(prompt[:n]).encode()).hexdigest()
            self.file.write(skip, count_from=n if key in seen else 0, gram=gram)
            seen.add(key)
            before = read_probe(self.dir)
            resp = post(self.base, "/v1/chat/completions", body)
            after = read_probe(self.dir)
            check(after, skip, resp)
            delta = {k: after[k] - (before[k] if before is not None else 0) for k in ("cos", "rel", "tokens")}
            total = delta if total is None else {k: total[k] + delta[k] for k in total}
        return {k: v.tolist() for k, v in total.items()}

    def screen(self, items, skip=""):
        """The NLL of the screen set under ``skip``, summed as the replay sums its targets."""
        if self._ids is None:
            self._ids = [post(self.base, "/tokenize", {"model": self.model, "prompt": t, "add_special_tokens": False})
                         ["tokens"][0] for t in ("<think>", "</think>")]
        self.file.write(skip, sums=False)
        nll = tokens = 0.0
        for n, (sid, g, i, _) in enumerate(items):
            resp = post(self.base, "/v1/chat/completions", self.body(sid, g, i, logprobs=True))
            if n == 0:
                check(read_probe(self.dir), skip, resp)
            row = score(resp["prompt_logprobs"], *self._ids)
            nll += row["reasoning_nll"] + row["answer_nll"]
            tokens += row["reasoning_tokens"] + row["answer_tokens"]
        return {"nll": nll / tokens, "scored_tokens": int(tokens)}

    def gram(self, items, names, out):
        """The Gram sums of ``names`` over ``items`` with no skip, then the flush to ``out`` (one short request)."""
        self.bi(items, "", gram=names)
        self.file.write("", sums=False, gram=names, flush=True)
        post(self.base, "/v1/chat/completions", {"model": self.model, "max_tokens": 1,
                                                 "messages": [{"role": "user", "content": "OK"}]})
        self.file.write("", sums=True)
        with np.load(self.dir / "layer_gram.npz") as z:
            if int(z["flush"]) != self.file.flush or not all(f"xtx_{n}" in z.files for n in names):
                raise RuntimeError(f"layer_gram.npz has flush {int(z['flush'])} and {sorted(z.files)}")
        shutil.copy(self.dir / "layer_gram.npz", out)


# --- the BI table --------------------------------------------------------------------------------------------------

def bi_table(sums):
    """``{block: (BI, relative change)}`` of every block with sums: BI = 1 - mean cosine."""
    cos, rel, tok = (np.asarray(sums[k]) for k in ("cos", "rel", "tokens"))
    return {f"{kind}{layer}": (1 - cos[layer, c] / tok[layer, c], rel[layer, c] / tok[layer, c])
            for layer in range(cos.shape[0]) for c, kind in enumerate(KINDS) if tok[layer, c] > 0}


# --- the greedy rounds of §7.2 to §7.4 --------------------------------------------------------------------------

def candidates(table, kinds, skip=""):
    """The blocks of the candidate layers in ``kinds`` that can join ``skip``, lowest BI first."""
    return sorted((b for b in table if b[0] in kinds and FIRST <= int(b[1:]) <= LAST and not clash(b, skip)),
                  key=lambda b: (table[b][0], b))


def round1(table):
    """The 10 L, 10 F and 8 A blocks with the lowest BI, one kind after another by rank, so that a short window drops
    the last ranks of each kind."""
    ranked = {k: candidates(table, k)[:n] for k, n in ROUND1.items()}
    return [r[j] for j in range(max(ROUND1.values())) for r in ranked.values() if j < len(r)]


class Greedy:
    """The steps of §7.2 with the log, the resume and the deadline (a Unix time). The Gram sums and rounds 2 and 3 run
    only while the best L or F set rises by at most ``stop`` percent (1%: the stop row of §7.6 after the refresh)."""

    def __init__(self, client, sets, log, deadline, stop=1.0, gram_out=None, clock=time.time):
        self.c, self.sets, self.log, self.deadline, self.stop = client, sets, Path(log), deadline, stop
        self.clock = clock
        self.gram_out = gram_out or self.log.with_name(self.log.stem + "-gram.npz")
        self.rows = {r["key"]: r for r in map(json.loads, self.log.open())} if self.log.exists() else {}
        self.cost = {"screen": 240.0, "bi": 600.0, "gram": 720.0}   # seconds; replaced by the first measurement

    def step(self, key, kind, fn, reserve=True, **fields):
        """Run ``fn`` once for ``key``, unless the log has it or the deadline is too near (R2 keeps its time)."""
        if key in self.rows:
            return self.rows[key]
        need = self.cost[kind] + (self.cost["screen"] if reserve else 0)
        if self.clock() + need > self.deadline:
            print(f"skip {key}: needs {need:.0f} s, the deadline is in {self.deadline - self.clock():.0f} s", flush=True)
            return None
        began = self.clock()
        row = {"key": key, "kind": kind, **fields, **fn()}
        row["seconds"] = self.cost[kind] = round(self.clock() - began, 1)
        self.rows[key] = row
        with self.log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "sums"})[:300], flush=True)
        return row

    def screen(self, key, skip, r, reserve=True, **fields):
        def fn():
            res = self.c.screen(self.sets["screen"], skip)
            return {**res, "rise": 100 * (res["nll"] / r["nll"] - 1)} if r else res
        return self.step(key, "screen", fn, reserve, set=skip, **fields)

    def bi(self, key, skip):
        row = self.step(key, "bi", lambda: {"sums": self.c.bi(self.sets["probe"], skip)}, set=skip)
        return bi_table(row["sums"]) if row else None

    def gram(self, names):
        def fn():
            self.c.gram(self.sets["probe"], names, self.gram_out)
            return {"blocks": names, "file": str(self.gram_out)}
        return self.step("gram", "gram", fn)

    def run(self):
        r = self.screen("R", "", None)
        if r is None:
            return
        table = self.bi("bi-0", "")
        singles = [s for s in (self.screen(f"r1-{b}", b, r, round=1, bi=table[b][0]) for b in round1(table or {}))
                   if s]
        a_best = sorted((s for s in singles if s["set"][0] == "A"), key=lambda s: s["rise"])[:6]
        if len(a_best) == 6:
            self.screen("r1-SA", canon(s["set"] for s in a_best), r, round=1)
        lf = sorted((s for s in singles if s["set"][0] in "LF"), key=lambda s: s["rise"])
        if lf and lf[0]["rise"] <= self.stop:
            self.gram([s["set"] for s in lf[:3]])
        base = lf[0] if lf else None
        for n in (2, 3):
            if base is None or base["rise"] > self.stop:
                break
            table = self.bi(f"bi-{n - 1}", base["set"])
            if table is None:
                break
            sets = [self.screen(f"r{n}-{b}", canon([*blocks_of(base["set"]), b]), r, round=n, bi=table[b][0])
                    for b in candidates(table, "LF", base["set"])[:6]]
            base = min((s for s in sets if s), key=lambda s: s["rise"], default=None)
        self.screen("R2", "", r, reserve=False)


# --- the report of §7.5 and §7.6 ----------------------------------------------------------------------------------

def spearman(x, y):
    rx, ry = (np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float) for v in (x, y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def decide(best, sa):
    """The row of the §7.6 table (thresholds of the refresh, §0) for the best single L or F rise and the A set."""
    if best is None:
        lines = ["No single L or F block was screened: no decision."]
    elif best > 1:
        lines = [f"The best single L or F block rises by {best:+.2f}% (more than 1%): stop the memory track."]
    elif best > 0.25:
        lines = [f"The best single L or F block rises by {best:+.2f}% (0.25% to 1%): it needs a heal; "
                 "the heal pull request, then Phase C with S0, S1 and S1h."]
    else:
        lines = [f"The best single L or F block rises by {best:+.2f}% (at most 0.25%): Phase C with S0, S1 and S1h, "
                 "and S2 and S3 if each set rises by at most 0.35%."]
    if sa is not None:
        lines.append(f"The set of 6 A blocks rises by {sa:+.2f}%: " + ("add SA to Phase C." if sa <= 0.25 else
                                                                        "no SA point."))
    return lines


def report(rows):
    rows = {r["key"]: r for r in rows}
    out = ["# The probe window: layer screen report", ""]
    r, r2 = rows.get("R"), rows.get("R2")
    if r:
        out.append(f"- R: NLL {r['nll']:.4f} over {r['scored_tokens']} scored tokens ({r['seconds']:.0f} s).")
    if r and r2:
        d = 100 * (r2["nll"] / r["nll"] - 1)
        out.append(f"- R2: NLL {r2['nll']:.4f}, {d:+.3f}% against R: the screen is "
                   f"{'stable' if abs(d) <= 0.1 else 'NOT stable (more than 0.1%); its numbers do not count'}.")
    table = bi_table(rows["bi-0"]["sums"]) if "bi-0" in rows else {}
    singles = sorted((x for x in rows.values() if x.get("round") == 1 and "," not in x["set"]), key=lambda x: x["rise"])
    if singles:
        bis = [table[x["set"]][0] for x in singles if x["set"] in table]
        rho = spearman(bis, [x["rise"] for x in singles if x["set"] in table])
        out += ["", "## Round 1: single blocks", "", "| Block | BI | Rise |", "|---|---|---|"]
        out += [f"| {x['set']} | {table.get(x['set'], (float('nan'),))[0]:.4f} | {x['rise']:+.2f}% |" for x in singles]
        out += ["", f"Rank correlation of BI and the rise: {rho:.2f}" + (" (below 0.5: BI ranks this model badly; "
                "a second probe window screens the other L and F blocks first)" if rho < 0.5 else "") + "."]
    for n in (2, 3):
        sets = sorted((x for x in rows.values() if x.get("round") == n), key=lambda x: x["rise"])
        if sets:
            out += ["", f"## Round {n}: sets of {n}", "", "| Set | BI of the new block | Rise |", "|---|---|---|"]
            out += [f"| {x['set']} | {x['bi']:.4f} | {x['rise']:+.2f}% |" for x in sets]
    if table:
        out += ["", "## BI with no skip (layers 3 to 39, lowest first)", "", "| Block | BI | Relative change |",
                "|---|---|---|"]
        out += [f"| {b} | {table[b][0]:.4f} | {table[b][1]:.4f} |" for b in candidates(table, "AFL")]
    lf = [x["rise"] for x in singles if x["set"][0] in "LF"]
    sa = rows.get("r1-SA")
    out += ["", "## The decision (§7.6)", ""] + [f"- {x}" for x in decide(min(lf) if lf else None,
                                                                          sa["rise"] if sa else None)]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    here = Path(__file__).resolve().parent
    ap.add_argument("--base-url", default="http://127.0.0.1:8890")
    ap.add_argument("--model", help="served model name (default: the server's first model)")
    ap.add_argument("--skip-dir", type=Path, default=here / "skip-out")
    ap.add_argument("--sets", type=Path, default=here / "results/skip-sets.json")
    ap.add_argument("--log", type=Path, default=here / "results/skip-probe.jsonl")
    ap.add_argument("--deadline", type=float, default=float("inf"), help="--greedy ends before this Unix time")
    ap.add_argument("--stop", type=float, default=1.0, help="the limit of the rise (%%) for the Gram sums and rounds 2-3")
    ap.add_argument("--set", default="", help="the skip set of --bi and --screen, for example L17,F21")
    cmd = ap.add_mutually_exclusive_group(required=True)
    for name in ("pick", "bi", "screen", "greedy", "report"):
        cmd.add_argument(f"--{name}", action="store_true")
    a = ap.parse_args(argv)
    if a.pick:
        sessions = [json.loads(line) for line in (DATA / "sessions.jsonl").open()]
        sets = pick(sessions, json.loads((DATA / "tools.json").read_text())["tools"])
        a.sets.write_text(json.dumps(sets))
        for k, v in sets.items():
            print(f"{k}: {len(v)} targets, about {sum(r[3] for r in v)} prompt tokens -> {a.sets}")
        return
    if a.report:
        print(report(map(json.loads, a.log.open())), end="")
        return
    sets = json.loads(a.sets.read_text())
    if a.model is None:
        with urllib.request.urlopen(a.base_url + "/v1/models", timeout=30) as resp:
            a.model = json.load(resp)["data"][0]["id"]
    client = Client(a.base_url, a.model, a.skip_dir)
    if a.bi:
        print(json.dumps(bi_table(client.bi(sets["probe"], canon(blocks_of(a.set))))))
    elif a.screen:
        print(json.dumps(client.screen(sets["screen"], canon(blocks_of(a.set)))))
    else:
        Greedy(client, sets, a.log, a.deadline, a.stop).run()


if __name__ == "__main__":
    main()
