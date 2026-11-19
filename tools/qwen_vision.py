"""The image test of a pruned Qwen (plans/PLAN_QWEN_PRUNE.md §8.4): does K 208 still read screenshots and diagrams?

The rank of the prune counted text tokens only, so the experts that image tokens use were never counted. This tool:

- ``--build``: renders 12 images with headless Chromium into ``data/qwen-data/vision/`` (not in git): 4 code screenshots,
  3 charts, 3 UI screens and 2 diagrams, each from an HTML string of this file, and writes ``questions.json``: 3
  questions per image with a known short answer that is visible in the image. It also renders the hard set
  (``hard.json``, 5 images): a full-HD screenshot of dense code, a table of 16 rows, a chart of two series with no
  value labels, a state diagram with labeled arrows, and a dense dark settings panel.
- ``--answer``: a model answers each question about its image (one image per request, the sampling of the fast
  route, a fixed seed per question). The rows have the format of ``qwen_facts.py``.
- ``--grade ANSWERS``: the grader model (the live DeepSeek, text only) grades each answer against the known answer,
  with the grader of ``qwen_facts.py``.
- ``--report FILES``: correct answers per kind of image for each graded file, and the questions where the files differ.
"""

import argparse
import base64
import glob
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import qwen_facts
import qwen_gate

CSS = ("body{margin:0;background:#fff;font-family:'DejaVu Sans',Arial,sans-serif;color:#1a1f2e}"
       "pre{margin:0;padding:22px 26px;font:16px/1.5 'DejaVu Sans Mono',monospace;background:#0f1422;color:#d7def0;min-height:100vh}"
       ".ln{color:#5b6788}.kw{color:#c792ea}.st{color:#c3e88d}.add{background:#12351f;color:#9be9a8;display:block}"
       ".del{background:#3d1418;color:#ffa198;display:block}.card{margin:40px;padding:28px;border:1px solid #cfd6e6;border-radius:10px;width:560px}"
       "label{display:block;margin:14px 0 4px;font-size:14px;color:#56617b}.field{border:1px solid #b9c2d6;border-radius:6px;padding:8px 10px;width:300px}"
       ".btn{display:inline-block;margin:22px 10px 0 0;padding:9px 20px;border-radius:6px;border:1px solid #b9c2d6}"
       ".primary{background:#1d63c4;color:#fff;border-color:#1d63c4}table{border-collapse:collapse;margin:40px;font-size:17px}"
       "td,th{border:1px solid #cfd6e6;padding:10px 22px;text-align:left}th{background:#eef1f8}")


def code(lines, classes=None):
    rows = []
    for i, line in enumerate(lines, 1):
        text = line.replace("&", "&amp;").replace("<", "&lt;")
        cls = (classes or {}).get(i)
        rows.append(f'<span class="{cls}"><span class="ln">{i:>3}  </span>{text}</span>' if cls else f'<span class="ln">{i:>3}  </span>{text}')
    return "<pre>" + "\n".join(rows) + "</pre>"


def bars(title, unit, data):
    top = max(v for _, v in data)
    out = [f'<svg width="760" height="420" xmlns="http://www.w3.org/2000/svg"><text x="30" y="36" font-size="20" font-weight="700">{title}</text>',
           f'<text x="30" y="62" font-size="14" fill="#56617b">{unit}</text><line x1="60" y1="360" x2="720" y2="360" stroke="#56617b"/>']
    for i, (name, v) in enumerate(data):
        h, x = 260 * v / top, 90 + i * 125
        out.append(f'<rect x="{x}" y="{360 - h:.0f}" width="70" height="{h:.0f}" rx="4" fill="#5b45d6"/>'
                   f'<text x="{x + 35}" y="{350 - h:.0f}" font-size="15" text-anchor="middle">{v}</text>'
                   f'<text x="{x + 35}" y="384" font-size="15" text-anchor="middle">{name}</text>')
    return "".join(out) + "</svg>"


def line_chart(title, points, limit):
    xs = [100 + i * 180 for i in range(len(points))]
    y = lambda v: 340 - v * 24
    path = " ".join(f'{"M" if i == 0 else "L"}{x} {y(v):.0f}' for i, (x, (_, v)) in enumerate(zip(xs, points)))
    out = [f'<svg width="760" height="420" xmlns="http://www.w3.org/2000/svg"><text x="30" y="36" font-size="20" font-weight="700">{title}</text>',
           f'<line x1="60" y1="340" x2="720" y2="340" stroke="#56617b"/><line x1="60" y1="{y(limit):.0f}" x2="720" y2="{y(limit):.0f}" stroke="#c42b2b" stroke-dasharray="6 5" stroke-width="2"/>',
           f'<text x="716" y="{y(limit) - 8:.0f}" font-size="15" fill="#c42b2b" text-anchor="end">limit {limit}%</text><path d="{path}" fill="none" stroke="#0b7f86" stroke-width="3"/>']
    for x, (name, v) in zip(xs, points):
        out.append(f'<circle cx="{x}" cy="{y(v):.0f}" r="6" fill="#0b7f86"/><text x="{x}" y="{y(v) - 14:.0f}" font-size="15" text-anchor="middle">{v}%</text>'
                   f'<text x="{x}" y="366" font-size="15" text-anchor="middle">{name}</text>')
    return "".join(out) + "</svg>"


def boxes(items, y=150, w=150, arrows=True):
    out = ['<svg width="1000" height="360" xmlns="http://www.w3.org/2000/svg"><defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
           '<path d="M0 0 L10 4 L0 8 z" fill="#1a1f2e"/></marker></defs>']
    for i, name in enumerate(items):
        x = 30 + i * (w + 40)
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="60" rx="8" fill="#eef1f8" stroke="#1a1f2e"/><text x="{x + w / 2}" y="{y + 36}" font-size="17" text-anchor="middle">{name}</text>')
        if arrows and i:
            out.append(f'<line x1="{x - 40}" y1="{y + 30}" x2="{x - 2}" y2="{y + 30}" stroke="#1a1f2e" stroke-width="2" marker-end="url(#a)"/>')
    return "".join(out) + "</svg>"


ROUTER = ('<svg width="900" height="420" xmlns="http://www.w3.org/2000/svg"><defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
          '<path d="M0 0 L10 4 L0 8 z" fill="#1a1f2e"/></marker></defs>'
          + "".join(f'<rect x="{x}" y="{y}" width="230" height="64" rx="8" fill="{c}" stroke="#1a1f2e"/><text x="{x + 115}" y="{y + 28}" font-size="17" text-anchor="middle">{a}</text>'
                    f'<text x="{x + 115}" y="{y + 50}" font-size="14" fill="#56617b" text-anchor="middle">{b}</text>'
                    for x, y, c, a, b in ((30, 178, "#fff", "Client", "VS Code"), (330, 178, "#eef1f8", "Router", "port 8010"),
                                          (640, 70, "#e9e5fb", "DeepSeek", "port 8888, reasoning"), (640, 286, "#dff3f4", "Qwen", "port 8889, fast")))
          + '<line x1="260" y1="210" x2="328" y2="210" stroke="#1a1f2e" stroke-width="2" marker-end="url(#a)"/>'
            '<line x1="560" y1="196" x2="638" y2="110" stroke="#1a1f2e" stroke-width="2" marker-end="url(#a)"/>'
            '<line x1="560" y1="224" x2="638" y2="310" stroke="#1a1f2e" stroke-width="2" marker-end="url(#a)"/></svg>')

# (name, kind, width, height, html, [(question, known answer)])
IMAGES = [
    ("code-python", "code", 900, 380, code([
        "def retry(fn, attempts=4, delay=0.25):", "    last = None", "    for i in range(attempts):", "        try:", "            return fn()",
        "        except TimeoutError as exc:", "            last = exc", "            time.sleep(delay * 2 ** i)",
        '    raise RuntimeError("gave up after 4 tries") from last']),
     [("What is the default value of the parameter attempts?", "4"), ("Which exception type does the code catch?", "TimeoutError"),
      ("What is the exact text of the error message that the function raises?", "gave up after 4 tries")]),
    ("code-shell", "code", 900, 400, code([
        "#!/usr/bin/env bash", "set -euo pipefail", "PORT=${PORT:-8889}", "LOG=/var/log/qwen/server.log", "for i in 1 2 3; do",
        '  curl -fsS "http://127.0.0.1:$PORT/health" && exit 0', "  sleep 7", "done", 'echo "not healthy after 21 s" >&2', "exit 3"]),
     [("What is the default port in this script?", "8889"), ("With which exit code does the script end when the server is not healthy?", "3"),
      ("How many seconds does the script sleep between two tries?", "7")]),
    ("code-diff", "code", 900, 330, code([
        "@@ -12,7 +12,7 @@ def lowest(rows):", "-    floor = 2048", "+    floor = 4096", "     for r in rows:",
        '-        if r.mb < floor: stop("qwen")', '+        if r.mb < floor: stop("qwen", grace=15)', "     return floor"],
        {2: "del", 3: "add", 5: "del", 6: "add"}),
     [("In this diff, what is the new value of floor?", "4096"), ("What was the old value of floor before the change?", "2048"),
      ("Which keyword argument does the diff add to the call of stop?", "grace=15")]),
    ("code-traceback", "code", 1000, 300, code([
        "Traceback (most recent call last):", '  File "/srv/app/server.py", line 61, in main', "    model = load(path)",
        '  File "/srv/app/loader.py", line 805, in weight_loader', "    expert_data = param.data[expert_id]",
        "IndexError: index 208 is out of bounds for dimension 0 with size 208"]),
     [("On which line of loader.py does the error happen?", "805"), ("What is the type of the exception?", "IndexError"),
      ("Which function of server.py calls load?", "main")]),
    ("chart-bars", "chart", 780, 440, bars("Requests per day", "requests", [("Mon", 181), ("Tue", 240), ("Wed", 96), ("Thu", 310), ("Fri", 205)]),
     [("Which day has the most requests?", "Thu (Thursday)"), ("How many requests does Wed have?", "96"), ("Which day has the fewest requests?", "Wed (Wednesday)")]),
    ("chart-line", "chart", 780, 440, line_chart("Quality loss against experts kept", [("K 256", 0), ("K 208", 4.2), ("K 192", 8.0), ("K 176", 11.1)], 3),
     [("What is the loss at K 192?", "8.0%"), ("What is the value of the limit drawn as a dashed line?", "3%"), ("Which point has a loss of 11.1%?", "K 176")]),
    ("chart-memory", "chart", 780, 440, bars("Memory at the peak", "GiB", [("full", 29.6), ("K 208", 26.5), ("K 208 + 4-bit draft", 25.4)]),
     [("How many GiB does the full model use?", "29.6"), ("Which bar is the lowest?", "K 208 + 4-bit draft"), ("What is the unit of this chart?", "GiB")]),
    ("ui-settings", "ui", 700, 520, '<div class="card"><h2>Server settings</h2><label>Model</label><div class="field">Qwen3.6-35B-A3B &#9662;</div>'
     '<label>Context length</label><div class="field">128000</div><p>&#9745; Enable draft model</p><p>&#9744; Vision input</p>'
     '<span class="btn primary">Save</span><span class="btn">Reset</span></div>',
     [("Which option is not checked?", "Vision input"), ("What value does the Context length field hold?", "128000"),
      ("What is the label of the blue button?", "Save")]),
    ("ui-table", "ui", 760, 360, "<table><tr><th>Server</th><th>Port</th><th>Status</th><th>tok/s</th></tr><tr><td>deepseek</td><td>8888</td><td>healthy</td><td>36.9</td></tr>"
     "<tr><td>qwen</td><td>8889</td><td>healthy</td><td>99.1</td></tr><tr><td>embed</td><td>8891</td><td>down</td><td>0</td></tr></table>",
     [("What is the status of the embed server?", "down"), ("Which port does the qwen server use?", "8889"), ("Which server has the highest tok/s?", "qwen")]),
    ("ui-dialog", "ui", 700, 380, '<div class="card"><h2 style="color:#c42b2b">Deploy failed</h2><p>Container exited with code 137 (out of memory).</p>'
     '<p style="font-size:13px;color:#56617b">Run ID: 7f3a-22c9</p><span class="btn primary">Retry</span><span class="btn">View logs</span></div>',
     [("With which code did the container exit?", "137"), ("What is the Run ID?", "7f3a-22c9"), ("What are the labels of the two buttons?", "Retry and View logs")]),
    ("diagram-pipeline", "diagram", 1020, 380, boxes(["capture", "rank", "prune", "check", "gate"]),
     [("Which step comes directly after rank?", "prune"), ("How many boxes does the diagram have?", "5"), ("What is the last step?", "gate")]),
    ("diagram-router", "diagram", 920, 440, ROUTER,
     [("Which port does the Router use?", "8010"), ("Which model is labeled fast?", "Qwen"), ("How many arrows leave the Router box?", "2")]),
]


# The hard set: dense, small or unlabeled content, as in a full-screen screenshot
STEPS = [(i, round(0.05 + 0.03 * ((i * 7) % 11), 2), (i * 13) % 29 + 2, (i * 5) % 9, 30 + (i * 17) % 23) for i in range(1, 19)]
DENSE = [line for i, s, off, lo, hi in STEPS for line in (f"def step_{i:02d}(x, scale={s}):", f"    y = x * scale + {off}", f"    return clamp(y, {lo}, {hi})")]
HOSTS = [(f"node-{i:02d}", ("us-west", "us-east", "eu-central")[i % 3], 11 + (i * 37) % 83, round(3.5 + (i * 2.3) % 27, 1), 40 + (i * 53) % 190,
          "degraded" if i == 9 else "ok") for i in range(1, 17)]
SERIES = {"alpha": [20, 35, 50, 80, 60, 45], "beta": [70, 55, 40, 30, 65, 90]}


def two_lines():
    x, y = (lambda i: 110 + i * 150), (lambda v: 420 - v * 3.4)
    out = ['<svg width="960" height="500" xmlns="http://www.w3.org/2000/svg"><text x="30" y="34" font-size="20" font-weight="700">Throughput by week</text>']
    for v in range(0, 101, 20):
        out.append(f'<line x1="80" y1="{y(v):.0f}" x2="900" y2="{y(v):.0f}" stroke="#d8ddea"/><text x="70" y="{y(v) + 5:.0f}" font-size="14" text-anchor="end">{v}</text>')
    for i in range(6):
        out.append(f'<text x="{x(i)}" y="446" font-size="14" text-anchor="middle">week {i + 1}</text>')
    for (name, vals), c, dash in zip(SERIES.items(), ("#5b45d6", "#c2410c"), ("", ' stroke-dasharray="8 5"')):
        out.append(f'<path d="{" ".join(f"{chr(77 if i == 0 else 76)}{x(i)} {y(v):.0f}" for i, v in enumerate(vals))}" fill="none" stroke="{c}" stroke-width="3"{dash}/>')
        out += [f'<circle cx="{x(i)}" cy="{y(v):.0f}" r="5" fill="{c}"/>' for i, v in enumerate(vals)]
    out.append('<line x1="640" y1="30" x2="680" y2="30" stroke="#5b45d6" stroke-width="3"/><text x="688" y="35" font-size="15">alpha</text>'
               '<line x1="770" y1="30" x2="810" y2="30" stroke="#c2410c" stroke-width="3" stroke-dasharray="8 5"/><text x="818" y="35" font-size="15">beta</text>')
    return "".join(out) + "</svg>"


def states():
    pos = {"idle": (40, 200), "boot": (250, 200), "serving": (470, 200), "draining": (700, 200), "stopped": (930, 200), "failed": (360, 400)}
    out = ['<svg width="1120" height="520" xmlns="http://www.w3.org/2000/svg"><defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
           '<path d="M0 0 L10 4 L0 8 z" fill="#1a1f2e"/></marker></defs>']
    for name, (x, y) in pos.items():
        out.append(f'<rect x="{x}" y="{y}" width="140" height="56" rx="28" fill="#eef1f8" stroke="#1a1f2e"/><text x="{x + 70}" y="{y + 34}" font-size="17" text-anchor="middle">{name}</text>')
    edge = lambda d, lx, ly, label: out.append(f'<path d="{d}" fill="none" stroke="#1a1f2e" stroke-width="2" marker-end="url(#a)"/><text x="{lx}" y="{ly}" font-size="14" fill="#56617b" text-anchor="middle">{label}</text>')
    edge("M180 228 L248 228", 214, 218, "start")
    edge("M390 228 L468 228", 429, 218, "healthy")
    edge("M610 228 L698 228", 654, 218, "low memory")
    edge("M840 228 L928 228", 884, 218, "done")
    edge("M300 256 L390 398", 312, 340, "timeout")
    edge("M440 398 L350 258", 432, 322, "retry")
    edge("M540 256 L470 398", 548, 340, "crash")
    edge("M1000 200 Q660 20 322 198", 660, 96, "restart")
    return "".join(out) + "</svg>"


SETTINGS = [("Prefix caching", "on"), ("Chunked prefill", "on"), ("Speculative decoding", "on"), ("Vision input", "off"), ("Auto tool choice", "on"),
            ("Max tokens", "4096"), ("Temperature", "0.6"), ("Top-p", "0.95"), ("Top-k", "20"), ("Context length", "128000"), ("Draft tokens", "3"), ("Log requests", "off")]
HARD = [
    ("hard-code-dense", "code", 1920, 1080, '<pre style="font-size:13px;line-height:1.38;columns:2;column-gap:60px">' + code(DENSE)[5:],
     [("What is the default value of scale in the function step_11?", str(STEPS[10][1])), ("Which number does step_05 add to x * scale?", str(STEPS[4][2])),
      (f"Which function returns clamp(y, {STEPS[13][3]}, {STEPS[13][4]})?", "step_14")]),
    ("hard-table", "ui", 1100, 760, "<table style='font-size:14px;margin:20px'><tr><th>Host</th><th>Region</th><th>CPU %</th><th>Mem GiB</th><th>p95 ms</th><th>Status</th></tr>"
     + "".join("<tr>" + "".join(f"<td style='padding:6px 18px'>{c}</td>" for c in h) + "</tr>" for h in HOSTS) + "</table>",
     [("What is the p95 ms value of the host node-12?", str(HOSTS[11][4])), ("Which host has the status degraded?", "node-09"),
      ("How many GiB of memory does node-07 use?", str(HOSTS[6][3]))]),
    ("hard-chart-two-lines", "chart", 980, 520, two_lines(),
     [("Which series is higher in week 5, alpha or beta?", "beta"), ("In which week does the series alpha reach its highest value?", "week 4"),
      ("About which value does the series beta have in week 3?", "about 40 (any value from 36 to 44 is correct)")]),
    ("hard-diagram-states", "diagram", 1140, 540, states(),
     [("Which state follows serving on the event low memory?", "draining"), ("Which event leads from boot to failed?", "timeout"),
      ("How many arrows end at the state boot?", "3")]),
    ("hard-ui-dark", "ui", 760, 620, '<div style="background:#12151f;color:#cdd3e4;min-height:100vh;padding:26px 34px;font-size:13px"><b style="font-size:15px">Runtime options</b>'
     + "".join(f'<div style="display:flex;justify-content:space-between;width:420px;padding:9px 0;border-bottom:1px solid #2a3044"><span>{k}</span>'
               f'<span style="color:{"#7fd99a" if v == "on" else "#8a93a8" if v == "off" else "#e6c07b"}">{v}</span></div>' for k, v in SETTINGS) + "</div>",
     [("Is the option Vision input on or off?", "off"), ("What is the value of Max tokens?", "4096"), ("Which option has the value 0.95?", "Top-p")]),
]


def chromium():
    found = sorted(glob.glob(str(Path.home() / ".cache/ms-playwright/chromium-*/chrome-linux/chrome")))
    if not found:
        raise SystemExit("no headless Chromium: install Playwright's chromium, or pass --chromium")
    return found[-1]


def build(out, chrome=None):
    out.mkdir(parents=True, exist_ok=True)
    chrome, questions = chrome or chromium(), {"questions.json": [], "hard.json": []}
    with tempfile.TemporaryDirectory() as tmp:
        for target, (name, kind, w, h, html, qa) in [("questions.json", i) for i in IMAGES] + [("hard.json", i) for i in HARD]:
            page = Path(tmp) / f"{name}.html"
            page.write_text(f"<!doctype html><meta charset=utf-8><style>{CSS}</style>{html}")
            subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{h + 90}",
                            f"--screenshot={out / (name + '.png')}", f"file://{page}"], check=True, capture_output=True, timeout=120)
            questions[target] += [{"image": f"{name}.png", "kind": kind, "question": q, "answer": a} for q, a in qa]
    for target, rows in questions.items():
        (out / target).write_text(json.dumps(rows, indent=1))
    return len(IMAGES) + len(HARD), sum(len(v) for v in questions.values())


def answer(base, model, folder, out, limit=None, max_tokens=1500, questions="questions.json"):
    done = {r["question"] for r in qwen_facts.load(out)}
    with Path(out).open("a") as f:
        for q in json.loads((folder / questions).read_text())[:limit]:
            key = f"[{q['image']}] {q['question']}"
            if key in done:
                continue
            row = {"question": key, "target": q["answer"], "topic": q["kind"]}
            try:
                data = base64.b64encode((folder / q["image"]).read_bytes()).decode()
                resp = qwen_facts.post(base, "/v1/chat/completions", {
                    "model": model, "max_tokens": max_tokens, "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), **qwen_gate.SAMPLING,
                    "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
                                                              {"type": "text", "text": q["question"] + " Answer in one short sentence."}]}]})
                row.update(answer=resp["choices"][0]["message"].get("content") or "", finish_reason=resp["choices"][0]["finish_reason"],
                           prompt_tokens=resp["usage"]["prompt_tokens"], completion_tokens=resp["usage"]["completion_tokens"])
            except Exception as exc:   # a failed question is recorded, the run goes on
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            f.write(json.dumps(row) + "\n")
            f.flush()


def report(paths):
    """Correct answers per kind for each graded file (on the questions that all files hold), and the differences."""
    files = [{r["question"]: r for r in qwen_facts.load(p)} for p in paths]
    common = sorted(set.intersection(*(set(f) for f in files)))
    kinds = sorted({files[0][q]["topic"] for q in common})
    for p, f in zip(paths, files):
        c = Counter(f[q]["topic"] for q in common if f[q]["grade"] == "CORRECT")
        print(f"{p}: correct {sum(c.values())} of {len(common)}; " + ", ".join(f"{k} {c[k]} of {sum(f[q]['topic'] == k for q in common)}" for k in kinds))
    for q in common:
        grades = [f[q]["grade"] for f in files]
        if len(set(grades)) > 1:
            print(f"  differs: {q}: {' / '.join(grades)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--base-url", default="http://127.0.0.1:8889")
    ap.add_argument("--model", default="second-model-fit")
    ap.add_argument("--images", type=Path, default=here / "data/qwen-data/vision")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--limit", type=int, help="--answer: only the first N questions")
    ap.add_argument("--questions", default="questions.json", help="--answer: the question file (hard.json: the hard set)")
    ap.add_argument("--chromium", help="--build: the Chromium binary (default: Playwright's)")
    cmd = ap.add_mutually_exclusive_group(required=True)
    cmd.add_argument("--build", action="store_true")
    cmd.add_argument("--answer", action="store_true")
    cmd.add_argument("--grade", type=Path, metavar="ANSWERS")
    cmd.add_argument("--report", nargs="+", type=Path)
    a = ap.parse_args(argv)
    if a.build:
        print("%d images, %d questions -> %s" % (*build(a.images, a.chromium), a.images))
    elif a.answer:
        answer(a.base_url, a.model, a.images, a.out, a.limit, questions=a.questions)
        rows = qwen_facts.load(a.out)
        print(f"{sum('error' not in r for r in rows)} answers, {sum('error' in r for r in rows)} errors -> {a.out}")
    elif a.grade:
        qwen_facts.grade(a.base_url, a.model, a.grade, a.out)
        report([a.out])
    else:
        report(a.report)


if __name__ == "__main__":
    sys.exit(main())
