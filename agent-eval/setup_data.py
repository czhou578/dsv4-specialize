"""Builds the frozen data of the task set (plans/PLAN_AGENT_CAPACITY.md §4) in DATA, out of git.

- repo/: this repo at the commit REPO_COMMIT (git archive), for the lookup tasks and the padding.
- papers/<name>.txt: the text of each paper of sources/papers.json (an arXiv HTML page or a PDF).
- web/pages.jsonl: the page set of sources/pages.json. A Wikipedia URL gets its plain text from the API. An entry
  with "status": 404 is a dead page: search finds its title and text, and fetch gives HTTP 404.

It fetches only what is missing, so a second run changes nothing. data_manifest.json (in git) holds the SHA-256 of
each file: --record writes it, and without --record the script compares the files with it.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from agent_eval import AE, DATA, S

REPO_COMMIT = "5a6f085"
UA = {"User-Agent": "agent-eval-setup/0.1"}
MANIFEST = AE / "data_manifest.json"
SKIP = {"script", "style", "nav", "header", "footer", "noscript", "svg", "button", "form"}
BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "table", "pre", "figcaption",
         "blockquote", "dd", "dt", "article", "caption"}


class Text(HTMLParser):
    """The visible text of an HTML page; a math element gives its alttext (the TeX source on arXiv)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.math = [], 0, 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip += 1
        elif tag == "math":
            self.math += 1
            if not self.skip and self.math == 1:
                self.out.append(f" {dict(attrs).get('alttext', '')} ")
        elif tag in BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "math":
            self.math = max(0, self.math - 1)
        elif tag in BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip and not self.math:
            self.out.append(data)


def html_text(html: str) -> str:
    p = Text()
    p.feed(html)
    lines = (" ".join(x.split()) for x in "".join(p.out).splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def get(url: str, tries: int = 6) -> bytes:
    """The body of a URL, at most one request a second. HTTP 429 and 5xx get a retry with backoff (Retry-After first)."""
    for n in range(tries):
        time.sleep(1)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if (e.code != 429 and e.code < 500) or n == tries - 1:
                raise
            wait = e.headers.get("Retry-After", "")
            time.sleep(int(wait) if wait.isdigit() else 10 * 2 ** n)


def wikipedia(url: str) -> tuple[str, str]:
    title = urllib.parse.unquote(url.rsplit("/wiki/", 1)[1])
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "titles": title,
                                "format": "json", "formatversion": 2, "redirects": 1})
    page = json.loads(get(f"https://en.wikipedia.org/w/api.php?{q}"))["query"]["pages"][0]
    return page["title"], page["extract"].strip() + "\n"


def page_text(url: str) -> tuple[str, str]:
    if "wikipedia.org/wiki/" in url:
        return wikipedia(url)
    html = get(url).decode("utf-8", errors="replace")
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    return (html_text(m.group(1)).strip() if m else url), html_text(html)


def paper_text(src: dict) -> str:
    raw = get(src["url"])
    if src.get("kind") != "pdf":
        return html_text(raw.decode("utf-8", errors="replace"))
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(raw)
        f.flush()
        return subprocess.run(["pdftotext", "-layout", f.name, "-"], capture_output=True, text=True, check=True).stdout


def build(only: str = ""):
    repo = DATA / "repo"
    if only in ("", "repo") and not repo.exists():
        repo.mkdir(parents=True)
        tar = subprocess.run(["git", "-C", str(S), "archive", REPO_COMMIT], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(repo)], input=tar, check=True)
    papers = AE / "sources/papers.json"
    for name, src in (json.loads(papers.read_text()) if papers.exists() and only in ("", "papers") else {}).items():
        out = DATA / "papers" / f"{name}.txt"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(paper_text(src))
            print(f"paper {name}: {out.stat().st_size} bytes")
    pages_src, out = AE / "sources/pages.json", DATA / "web/pages.jsonl"
    if only not in ("", "pages"):
        return
    have = {}
    if out.exists():
        have = {p["url"]: p for p in map(json.loads, out.read_text().splitlines())}
    rows = []
    out.parent.mkdir(parents=True, exist_ok=True)
    for src in json.loads(pages_src.read_text()) if pages_src.exists() else []:
        if src["url"] in have:
            rows.append(have[src["url"]])
            continue
        if src.get("status") == 404:
            rows.append({"url": src["url"], "title": src["title"], "text": src["text"], "status": 404})
        else:
            title, text = page_text(src["url"])
            rows.append({"url": src["url"], "title": title, "text": text})
            print(f"page {src['url']}: {len(text)} characters")
        have[src["url"]] = rows[-1]   # save after each new page, so a stop loses nothing
        out.write_text("".join(json.dumps(p) + "\n" for p in have.values()))
    if rows:   # the final file: the order of sources/pages.json, without pages that left the list
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))


def digests() -> dict:
    files = [p for p in sorted(DATA.rglob("*")) if p.is_file() and "repo" not in p.relative_to(DATA).parts[:1]]
    return {str(p.relative_to(DATA)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="store_true", help="write data_manifest.json from the files")
    ap.add_argument("--only", choices=["repo", "papers", "pages"], default="", help="build one part")
    a = ap.parse_args()
    build(a.only)
    now = digests()
    if a.record:
        MANIFEST.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n")
        print(f"recorded {len(now)} files")
        return
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    bad = sorted(k for k in old.keys() | now.keys() if old.get(k) != now.get(k))
    print("the data matches the manifest" if not bad else f"differs from the manifest: {', '.join(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
