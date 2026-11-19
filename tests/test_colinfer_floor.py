"""colinfer_floor.py and the dry run of colinfer_floor.sh: the parsers, the long prompt size, the report arithmetic, and
the window script under bash with stub boots and a fake client (COLINFER_STUB). No server, no GPU."""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import colinfer_floor as cf

HERE = Path(__file__).resolve().parent.parent   # the repo root
SNAP = {"before_mb": 118000, "fall": 30 * cf.GB, "allocated": 26 * cf.GB, "reserved": 27 * cf.GB, "anon": 2 * cf.GB,
        "file": cf.GB // 2, "gpu": 28 * cf.GB}
# The fake Python logs its arguments; the report runs on the real Python.
FAKE = "#!/usr/bin/env bash\ncase \" $* \" in *\" --report \"*) exec %s \"$@\";; esac\necho \"FAKE $*\" >> $STUB_DIR/calls.log\n" % sys.executable
STUB = r'''
R=$STUB_DIR PY=$STUB_DIR/python
stop_main() { log "STUB stop_main"; }
restore() { log "STUB restore"; }
wait_mem() { :; }
crunning() { return 1; }
cstop() { CPID=; log "STUB cstop"; }
avail_mb() { echo 118000; }
sample_start() { printf '1 118000\n9999999999 86000\n' > "$1"; }
sample_stop() { :; }
launch() {
  log "STUB launch $1 ${FLAGS[$1]}"
  case " ${STUB_FAIL:-} " in *" $1 "*) echo "RuntimeError: boom" > "$2"; return 1;; esac
  echo "[engine] ready: 14 s; 26.0 GB allocated, 27.0 GB reserved; x" > "$2"
}
snap() { echo "{\"when\": \"$2\", \"epoch\": 5, $STUB_SNAP}" >> $R/colinfer-floor-$1-snap.jsonl; }
'''


class ClientTest(unittest.TestCase):
    def test_the_parsers(self):
        smaps = "Rss:  3260420 kB\nPss:  3036534 kB\nPss_Anon:  2436772 kB\nPss_File:  535818 kB\nPss_Shmem:  63944 kB\n"
        self.assertEqual(cf.pss(smaps), (2436772 * 1024, (535818 + 63944) * 1024))
        self.assertEqual(cf.pss(""), (0, 0))
        smi = "3383498, 25827\n4100, 312\n"
        self.assertEqual((cf.gpu_mib(smi, 3383498), cf.gpu_mib(smi, 4100), cf.gpu_mib(smi, 410), cf.gpu_mib("4100, [N/A]\n", 4100)),
                         (25827, 312, 0, 0))

    def test_the_long_prompt_stays_under_its_token_count(self):
        def fake(base, path, body, timeout=None):   # 40 tokens per paragraph, 12 tokens of chat template
            self.assertEqual(path, "/v1/messages/count_tokens")
            return {"input_tokens": 12 + 40 * body["messages"][0]["content"].count(cf.PARAGRAPH)}
        with mock.patch.object(cf, "post", fake):
            text = cf.long_text("http://x", "m", 100000)
        self.assertEqual(text.count(cf.PARAGRAPH), (100000 - 200) // 40)

    def test_the_report_gives_each_snapshot_and_the_peak_in_gb(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "colinfer-floor-f1-snap.jsonl").write_text(json.dumps({"when": "ready", "epoch": 5, **SNAP}) + "\n" + json.dumps(
                {"when": "long", "epoch": 9, **SNAP, "fall": 32 * cf.GB, "reserved": 28 * cf.GB}) + "\n")
            (d / "colinfer-floor-f1-mem.log").write_text("1 118000\n3 60000\n6 88000\n8 86000\n")   # 3: before the ready snapshot
            rows = cf.report(d, ["f1", "f2"])                                                       # f2 has no files
        self.assertEqual([(r["point"], r["when"]) for r in rows], [("f1", "ready"), ("f1", "long"), ("f1", "peak")])
        self.assertEqual((rows[0]["fall"], rows[0]["reserved"], rows[0]["outside"], rows[0]["gpu"]), (30.0, 27.0, 3.0, 28.0))
        self.assertEqual(rows[1]["outside"], 4.0)
        self.assertAlmostEqual(rows[2]["fall"], (118000 - 86000) * cf.MIB / cf.GB, places=2)
        self.assertIn("f1 ready: fall 30.00 GB; torch 26.00 allocated, 27.00 reserved; outside the allocator 3.00", cf.line(rows[0]))
        self.assertIn("f1 peak: fall 33.55 GB", cf.line(rows[2]))


class WindowTest(unittest.TestCase):
    def run_script(self, boot_by=60, **env):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        (d / "stub.sh").write_text(STUB)
        (d / "python").write_text(FAKE)
        (d / "python").chmod(0o755)
        t = lambda m: time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + 60 * m))
        for f in ("colinfer/.venv/bin/python", "colinfer/engine/server/api.py"):   # the files of a colinfer checkout that the script checks
            (d / f).parent.mkdir(parents=True, exist_ok=True)
            (d / f).write_text("")
        full = {**os.environ, "COLINFER_STUB": str(d / "stub.sh"), "STUB_DIR": str(d), "LOG": str(d / "window.log"), "C": str(d / "colinfer"),
                "STUB_SNAP": json.dumps(SNAP)[1:-1], "START_AFTER": t(-1), "BOOT_BY": t(boot_by), "RESTORE_AT": t(240), **env}
        proc = subprocess.run(["bash", str(HERE / "scripts/colinfer_floor.sh")], env=full, capture_output=True, text=True, timeout=120)
        calls = (d / "calls.log").read_text() if (d / "calls.log").exists() else ""
        return proc.returncode, (d / "window.log").read_text(), calls, d

    def test_a_missing_checkout_ends_before_any_server_action(self):
        rc, log, _, _ = self.run_script(C=str(HERE / "no-such-checkout"))
        self.assertEqual(rc, 1)
        self.assertIn("ERROR: no ", log)
        self.assertNotIn("STUB stop_main", log)

    def test_a_normal_window(self):
        rc, log, calls, d = self.run_script()
        self.assertEqual(rc, 0, log)
        base = "--slots 1 --max-seq-len 131072 --decode-weights checkpoint --no-vision"
        for p, flags in (("f1", "--spec none --no-prefix-caching"), ("f2", "--spec none --session-cache-gb 0"), ("f3", "--spec none"),
                         ("f4", "--spec mtp --session-cache-gb 0")):
            self.assertIn(f"STUB launch {p} {base} {flags}\n", log)
        self.assertEqual(log.count("STUB stop_main"), 4)
        self.assertEqual((calls.count("--smoke"), calls.count("--long 100000")), (3, 3))   # f3 sends no request
        whens = lambda p: [json.loads(x)["when"] for x in (d / f"colinfer-floor-{p}-snap.jsonl").read_text().splitlines()]
        self.assertEqual((whens("f1"), whens("f3")), (["ready", "smoke", "long"], ["ready"]))
        self.assertIn("colinfer-f1 healthy: 118000 MB available; [engine] ready: 14 s; 26.0 GB allocated", log)
        self.assertIn("report: f3 ready: fall 30.00 GB; torch 26.00 allocated, 27.00 reserved; outside the allocator 3.00", log)
        self.assertIn("report: f4 peak: fall 33.55 GB", log)
        self.assertEqual(len(json.loads((d / "colinfer-floor.json").read_text())), 3 * 4 + 2)   # 3 x (3 snapshots, 1 peak), f3: 1 + 1
        self.assertLess(log.rindex("STUB cstop"), log.index("STUB restore"))
        self.assertLess(log.index("STUB restore"), log.index("report: f1 ready"))            # the report runs after the restore
        self.assertIn("colinfer floor window done", log)

    def test_failed_boots_skip_the_point_and_two_in_a_row_restore(self):
        rc, log, calls, _ = self.run_script(STUB_FAIL="f2")
        self.assertEqual(rc, 0, log)
        self.assertIn("ERROR: colinfer-f2 did not boot: RuntimeError: boom", log)
        self.assertIn("STUB launch f3", log)
        self.assertEqual(calls.count("--smoke"), 2)
        rc, log, _, _ = self.run_script(STUB_FAIL="f1 f2")
        self.assertEqual(rc, 1)
        self.assertIn("two failed boots in a row: restore", log)
        self.assertNotIn("STUB launch f3", log)
        self.assertIn("STUB restore", log)

    def test_a_past_boot_deadline_ends_before_any_server_action(self):
        rc, log, _, _ = self.run_script(boot_by=-1)
        self.assertEqual(rc, 1)
        self.assertIn("no boot", log)
        self.assertNotIn("STUB stop_main", log)

    def test_the_process_check_matches_a_server_only(self):
        pat = subprocess.run(["bash", "-c", f"source <(grep '^CPAT=' {HERE / 'scripts/colinfer_floor.sh'}); echo \"$CPAT\""],
                             capture_output=True, text=True).stdout.strip()
        v = "/home/u/Projects/colin-inference-engine/.venv/bin/python"
        for real in (f"{v} -m engine.server --host 127.0.0.1 --port 8002 --slots 1", f"{v} -m engine.server"):
            self.assertTrue(re.search(pat, real), real)
        for other in (f"{v} -m engine.server.gateway --host 127.0.0.1 --port 8010", f"flock -n /x/gpu-model.lock {v} -m engine.server --port 8002",
                      'bash -c pgrep -f "python -m engine.server "', "tmux new -d -s x ./colinfer_floor.sh"):
            self.assertFalse(re.search(pat, other), other)


if __name__ == "__main__":
    unittest.main()
