"""The rank that counts coding traffic (plans/PLAN_CODE_RANK.md §5).

  code_rank.py --code CAPTURE [--pilot F] [--bot F] [--k160 F] [--md F] [--cold F] [--out-dir D]

The inputs:
- the coding capture of window 1: capture.py rows of qwen-data (claude, repo), with the counts and, with --sums, the
  router weight per layer and expert;
- the pilot capture of the junk night (junk_routes.py rows): the 40 pilot tasks and the probes c2 and c3;
- the bot capture (results/capture-rank.jsonl), the keep lists of K 160 and MD, and the drop file of MD.
The default paths are those of the main tree, also when a worktree runs the tool.

The pool is every expert that MD drops in layers 3-42; layers 0-2 route by a hash and keep their drops. The order is the
coding weight share of the expert in its layer, largest first. MD plus the first n experts of the order is one point
of the curve: the dropped share of the bot weight, of the pilot weight (held out from the rank) and of the c2 and c3
weight. The candidates:
- code1: the smallest n where c3 drops at most C3_MAX (9.37%, the level of K 160, where c3 was clean);
- code2: the smallest n where c2 drops at most C2_MAX (7.5%, well below the 9.91% of K 160, where c2 was junk).
n is at most N_MAX (486 experts, 4.29 GiB: the memory of K 160). One candidate when both n are equal. Each candidate
gets a drop file (MD's drops without its n experts) and its mean K, for mask_checkpoint.py --drop-file.
It writes code-rank.md, code-rank.json and cold-<name>.json in the out folder.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import junk_routes as jr

C3_MAX, C2_MAX, N_MAX = 0.0937, 0.075, 486
MAIN = Path.home() / "Projects/dsv4-specialize"
GRID = (0, 56, 113, 170, 226, 283, 339, 400, 486)


def weights(path, kind=None, name=None):
    """The summed weight (else the counts) of the good rows of a capture file. A row with a NaN weight slot (the bot
    capture of 2026-11-06) is left out of the weight."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    rows = [r for r in rows if "error" not in r and (kind is None or r.get("kind") == kind)
            and (name is None or r.get("name") == name)]
    key = "weight" if rows and all(r.get("weight") is not None for r in rows) else "counts"
    if key == "weight":
        rows = [r for r in rows if np.isfinite(np.asarray(r["weight"], dtype=float)).all()]
    return sum(np.asarray(r[key], dtype=float) for r in rows), len(rows), key


def curve(order, md_drop, arrays):
    """The dropped share of each array after each count of added experts (0 to len(order))."""
    out = {}
    for name, a in arrays.items():
        added = np.concatenate([[0.0], np.cumsum([a[l, e] for l, e in order])])
        out[name] = (a[md_drop].sum() - added) / a.sum()
    return out


def rank(code, pilot, bot, k160, md, cold, out_dir, n_max=N_MAX):
    r160, rmd = jr.removed_sets(k160, md)
    md_drop = r160 | rmd
    a_code, n_code, basis = weights(code)
    arrays = {"bot": weights(bot)[0], "pilot": weights(pilot, "pilot")[0],
              "c2": weights(pilot, "probe", "c2")[0], "c3": weights(pilot, "probe", "c3")[0]}
    share = a_code / np.maximum(a_code.sum(axis=1, keepdims=True), 1e-12)
    pool = [(l, e) for l in range(jr.HASH_LAYERS, jr.LAYERS) for e in range(jr.EXPERTS) if md_drop[l, e]]
    order = sorted(pool, key=lambda le: (-share[le], le))[:n_max]
    c = curve(order, md_drop, arrays)
    k160_ref = {name: float(a[r160].sum() / a.sum()) for name, a in arrays.items()}
    cold = {int(l): set(v) for l, v in json.loads(Path(cold).read_text()).items()}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = []
    for name, probe, limit in (("code1", "c3", C3_MAX), ("code2", "c2", C2_MAX)):
        hit = np.nonzero(c[probe] <= limit)[0]
        n = int(hit[0]) if len(hit) else len(order)
        if any(x["added"] == n for x in cands):
            continue
        drops = {l: set(v) for l, v in cold.items()}
        for l, e in order[:n]:
            drops[l].discard(e)
        path = out_dir / f"cold-{name}.json"
        path.write_text(json.dumps({str(l): sorted(v) for l, v in sorted(drops.items())}))
        cands.append({"name": name, "probe": probe, "limit": limit, "added": n, "reached": bool(len(hit)),
                      "gib": round(n * jr.MIB_PER_EXPERT / 1024, 2), "drop": str(path),
                      "k": round(sum(jr.EXPERTS - len(v) for v in drops.values()) / jr.LAYERS)})
    points = sorted({min(n, len(order)) for n in GRID} | {x["added"] for x in cands})
    res = {"coding_rows": n_code, "basis": basis, "pool": len(pool),
           "curve": [{"added": n, "gib": round(n * jr.MIB_PER_EXPERT / 1024, 2),
                      **{k: float(v[n]) for k, v in c.items()}} for n in points],
           "k160": k160_ref, "candidates": cands}
    (out_dir / "code-rank.json").write_text(json.dumps(res, indent=1))
    (out_dir / "code-rank.md").write_text(markdown(res))
    print("candidates: " + ", ".join(f"{x['name']} +{x['added']} ({x['gib']} GiB)" for x in cands))
    return res


def markdown(res):
    pct = lambda v: f"{100 * v:.2f}%"
    lines = ["# The rank that counts coding traffic", "",
             f"Coding capture: {res['coding_rows']} rows, ranked by {res['basis']}. Pool: {res['pool']} experts that "
             f"MD drops in layers 3-42. The columns give the dropped share of the weight.", "",
             "| Added to MD | Memory | Bot | Pilot (held out) | c2 | c3 |", "|---|---|---|---|---|---|"]
    for p in res["curve"]:
        lines.append(f"| {p['added']} | +{p['gib']:.2f} GiB | {pct(p['bot'])} | {pct(p['pilot'])} | {pct(p['c2'])} | "
                     f"{pct(p['c3'])} |")
    k = res["k160"]
    lines += [f"| K 160 (reference) | +4.29 GiB | {pct(k['bot'])} | {pct(k['pilot'])} | {pct(k['c2'])} | "
              f"{pct(k['c3'])} |", "", "## The candidates", ""]
    for x in res["candidates"]:
        lines.append(f"- {x['name']}: +{x['added']} experts (+{x['gib']} GiB, mean K {x['k']}): {x['probe']} drops at "
                     f"most {pct(x['limit'])}{'' if x['reached'] else ' NOT REACHED within the cap'}. `{x['drop']}`")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", type=Path, required=True, help="the capture.py rows of the coding capture")
    ap.add_argument("--pilot", type=Path, default=MAIN / "results/junk-capture/capture.jsonl")
    ap.add_argument("--bot", type=Path, default=MAIN / "results/capture-rank.jsonl")
    ap.add_argument("--k160", type=Path, default=jr.K160)
    ap.add_argument("--md", type=Path, default=MAIN / "tp1-prunemd5-k149/PRUNE.json")
    ap.add_argument("--cold", type=Path, default=MAIN / "results/cold-md.json")
    ap.add_argument("--out-dir", type=Path, default=MAIN / "results/code-rank")
    a = ap.parse_args(argv)
    rank(a.code, a.pilot, a.bot, a.k160, a.md, a.cold, a.out_dir)


if __name__ == "__main__":
    main()
