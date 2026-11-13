"""The dry run of skip_probe.sh (plans/PLAN_FEWER_LAYERS.md §6.5): the script runs under bash with stub boots and a
stub client (SKIP_STUB), and the tests read its log. No server, no GPU."""

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUB = r'''
R=$STUB_DIR OUT=$STUB_DIR/out
stop_main() { log "STUB stop_main"; }
down() { log "STUB down"; }
restore() { log "STUB restore"; }
memwatch_start() { :; }
memwatch_stop() { :; }
boot() {
  log "STUB boot $1 compose $TEST_COMPOSE"
  case " ${STUB_BOOT_FAIL:-} " in *" $1 "*) return 1;; esac
  [ "${STUB_NO_LINE:-0}" = 1 ] && { echo "no line" > $R/boot-$1.log; return 0; }
  echo "WARNING layer skip: set none (run-time file $SKIP_FILE); skipped weights 0.00 GiB; heal none; probe on" > $R/boot-$1.log
}
client() {
  log "STUB client"
  $PY - "$PLOG" <<'PY'
import json, sys
cos = [[100 * (1 - (0.01 + l / 1000 + k / 100)) for k in range(3)] for l in range(43)]
rows = [{"key": "R", "kind": "screen", "set": "", "nll": 0.5, "scored_tokens": 10, "seconds": 1},
        {"key": "bi-0", "kind": "bi", "set": "", "sums": {"cos": cos, "rel": [[1, 1, 1]] * 43,
                                                          "tokens": [[100, 100, 100]] * 43}, "seconds": 1},
        {"key": "r1-F3", "kind": "screen", "set": "F3", "round": 1, "bi": 0.023, "nll": 0.51, "rise": 2.0,
         "scored_tokens": 10, "seconds": 1},
        {"key": "R2", "kind": "screen", "set": "", "nll": 0.5001, "scored_tokens": 10, "seconds": 1, "rise": 0.02}]
open(sys.argv[1], "a").write("".join(json.dumps(r) + "\n" for r in rows))
PY
}
'''


class ProbeScriptTest(unittest.TestCase):
    def run_script(self, **env):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "stub.sh").write_text(STUB)
        (d / "skip-sets.json").write_text(json.dumps({"screen": [], "probe": []}))
        t = lambda m: time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + 60 * m))
        full = {**os.environ, "SKIP_STUB": str(d / "stub.sh"), "STUB_DIR": str(d), "LOG": str(d / "window.log"),
                "START_AFTER": t(-1), "BOOT_BY": t(60), "SPEED_BY": t(60), "RESTORE_AT": t(240), **env}
        proc = subprocess.run(["bash", str(HERE / "skip_probe.sh")], env=full, capture_output=True, text=True,
                              timeout=120)
        self.addCleanup(self.tmp.cleanup)
        return proc.returncode, (d / "window.log").read_text(), d

    def test_a_normal_window(self):
        rc, log, d = self.run_script()
        self.assertEqual(rc, 0, log)
        for line in ["STUB stop_main", "STUB boot skip-probe compose", "STUB client", "STUB down", "STUB restore",
                     "decision: The best single L or F block rises by +2.00% (more than 1%): stop the memory track",
                     "R2: NLL 0.5001, +0.020% against R: the screen is stable", "probe window done"]:
            self.assertIn(line, log)
        self.assertLess(log.index("STUB down"), log.index("STUB restore"))
        self.assertTrue((d / "skip-probe-report.md").exists())

    def test_the_v1_boot_after_a_failed_v2_boot(self):
        rc, log, _ = self.run_script(STUB_BOOT_FAIL="skip-probe")
        self.assertIn("one boot with Model Runner V1", log)
        self.assertIn("compose-skip.yml -f compose-skip-v1.yml", log.split("STUB boot skip-probe-v1")[1].splitlines()[0])
        self.assertIn("STUB client", log)

    def test_two_failed_boots_restore_with_no_client(self):
        rc, log, _ = self.run_script(STUB_BOOT_FAIL="skip-probe skip-probe-v1")
        self.assertEqual(rc, 1)
        self.assertIn("no probe boot (rc 1): restore", log)
        self.assertNotIn("STUB client", log)
        self.assertIn("STUB restore", log)

    def test_no_probe_line_stops_the_test_server_and_restores(self):
        rc, log, _ = self.run_script(STUB_NO_LINE="1")
        self.assertEqual(rc, 1)
        self.assertIn("the boot log has no probe line", log)
        self.assertNotIn("STUB client", log)
        self.assertLess(log.index("STUB down"), log.index("STUB restore"))

    def test_an_old_client_log_stops_the_script_before_any_server_action(self):
        old = tempfile.TemporaryDirectory()
        self.addCleanup(old.cleanup)
        (Path(old.name) / "skip-probe.jsonl").write_text("{}\n")
        (Path(old.name) / "skip-sets.json").write_text("{}")
        rc, log, _ = self.run_script(STUB_DIR=old.name)
        self.assertEqual(rc, 1)
        self.assertIn("skip-probe.jsonl exists", log)
        self.assertNotIn("STUB stop_main", log)


if __name__ == "__main__":
    unittest.main()
