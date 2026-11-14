"""The dry runs of skip_probe.sh and skip_replay.sh (plans/PLAN_FEWER_LAYERS.md §6.5): the script runs under bash with stub boots and a
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
  { echo "WARNING layer skip: set none (run-time file $SKIP_FILE); registered weights 0.00 GiB (EXL3 experts not counted); heal none; probe on"
    echo "INFO Initializing a V1 LLM engine with config: enable_prefix_caching=${STUB_CACHE:-False}, x"; } > $R/boot-$1.log
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

    def test_a_prefix_cache_on_stops_the_test_server_and_restores(self):
        rc, log, _ = self.run_script(STUB_CACHE="True")
        self.assertEqual(rc, 1)
        self.assertIn("or the prefix cache is on: restore", log)
        self.assertNotIn("STUB client", log)

    def test_an_old_client_log_stops_the_script_before_any_server_action(self):
        old = tempfile.TemporaryDirectory()
        self.addCleanup(old.cleanup)
        (Path(old.name) / "skip-probe.jsonl").write_text("{}\n")
        (Path(old.name) / "skip-sets.json").write_text("{}")
        rc, log, _ = self.run_script(STUB_DIR=old.name)
        self.assertEqual(rc, 1)
        self.assertIn("skip-probe.jsonl exists", log)
        self.assertNotIn("STUB stop_main", log)


REPLAY_STUB = r"""
R=$STUB_DIR
stop_main() { log "STUB stop_main"; }
down() { log "STUB down $TUNE_LABEL"; }
restore() { log "STUB restore"; }
smoke() { log "STUB smoke $1"; }
speed() { log "STUB speed $1"; }
boot() {
  log "STUB boot $1 skip=${LAYER_SKIP:-none} heal=${LAYER_HEAL:-}"
  case " ${STUB_BOOT_FAIL:-} " in *" $1 "*) return 1;; esac
  local s=${LAYER_SKIP:-none} h=none
  [ " ${STUB_WRONG_LINE:-} " = " $1 " ] && s=L99
  [ -n "${LAYER_HEAL:-}" ] && h="F24 sha256 abc"
  { echo "WARNING layer skip: set $s (fixed); registered weights 0.02 GiB (EXL3 experts not counted); heal $h; probe off"
    [ " ${STUB_CACHE:-} " = " $1 " ] && echo "INFO Directly load AOT compilation from path /cache/x"; } > $R/boot-$1.log
  return 0
}
replay() {
  local p=${1%-[ab]}; p=${p#skip-}; local f=STUB_F_$p
  log "STUB replay $1"
  $PY -c "
import json, sys
f = float(sys.argv[2])
rows = [{'session': 's', 'segment': 0, 'index': i, 'reasoning_tokens': 100, 'answer_tokens': 100,
         'reasoning_nll': 44.4 * f, 'answer_nll': 44.4 * f, 'decision_ok': i < 41, 'tool_calls': 1,
         'tool_names_ok': 1} for i in range(60)]
open(sys.argv[1], 'w').write(''.join(json.dumps(r) + chr(10) for r in rows))" $R/$1.jsonl ${!f:-1.0}
}
"""


class ReplayScriptTest(unittest.TestCase):
    LIVE = 0.4562 / 0.4440   # S0 at the live replays

    def run_script(self, **env):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        (d / "stub.sh").write_text(REPLAY_STUB)
        (d / "heal.npz").write_bytes(b"x")
        rows = [{"session": "s", "segment": 0, "index": i, "reasoning_tokens": 100, "answer_tokens": 100,
                 "reasoning_nll": 44.4, "answer_nll": 44.4, "decision_ok": i < 41, "tool_calls": 1, "tool_names_ok": 1}
                for i in range(60)]
        (d / "baseline-k216.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        t = lambda m: time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + 60 * m))
        full = {**os.environ, "SKIP_STUB": str(d / "stub.sh"), "STUB_DIR": str(d), "LOG": str(d / "window.log"),
                "START_AFTER": t(-1), "BOOT_BY": t(600), "SPEED_BY": t(600), "RESTORE_AT": t(900),
                "S1_SET": "F24", "HEAL_FILE": str(d / "heal.npz"), "STUB_F_s0": str(self.LIVE), **env}
        proc = subprocess.run(["bash", str(HERE / "skip_replay.sh")], env=full, capture_output=True, text=True,
                              timeout=120)
        return proc.returncode, (d / "window.log").read_text(), d

    def test_a_pass_a_fail_and_the_speed_rule(self):
        rc, log, _ = self.run_script(STUB_F_s1="1.05", STUB_F_s1h="1.025")
        self.assertEqual(rc, 0, log)
        self.assertIn("S0 check: mean NLL 0.4562 is +0.00% against the live replays", log)
        self.assertIn("skip-s1: mean NLL 0.4662 (+5.00% against K 216; the gate +3%)", log)
        self.assertRegex(log, r"skip-s1: .*: FAIL")
        self.assertRegex(log, r"skip-s1h: .*: PASS")
        self.assertIn("STUB boot skip-s1h skip=F24 heal=/skip/heal.npz", log)
        self.assertEqual([x.split()[-1] for x in log.splitlines() if "STUB speed" in x], ["skip-s0", "skip-s1h"])
        self.assertIn("skip-s1h: PASS; mean NLL 0.4551, -0.24% against S0", log)

    def test_a_failed_boot_a_wrong_line_and_a_cached_graph_skip_the_point(self):
        for env, text in [({"STUB_BOOT_FAIL": "skip-s1"}, None), ({"STUB_WRONG_LINE": "skip-s1"}, "no line"),
                          ({"STUB_CACHE": "skip-s1"}, "a graph from the compile cache")]:
            rc, log, _ = self.run_script(**env)
            self.assertEqual(rc, 0, log)
            self.assertNotIn("STUB replay skip-s1-a", log)
            self.assertIn("STUB replay skip-s1h-a", log)              # one failed boot: the window goes on
            if text:
                self.assertIn(text, log)
                self.assertIn("STUB down skip-s1", log)

    def test_the_s0_check_stops_the_window(self):
        rc, log, _ = self.run_script(STUB_F_s0=str(self.LIVE * 1.02))
        self.assertEqual(rc, 1)
        self.assertIn("S0 check fails: restore and stop", log)
        self.assertNotIn("STUB boot skip-s1", log)

    def test_old_replay_files_stop_the_script_before_any_server_action(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        Path(tmp.name, "skip-s1-a.jsonl").write_text("{}\n")
        Path(tmp.name, "baseline-k216.jsonl").write_text("{}\n")
        rc, log, _ = self.run_script(STUB_DIR=tmp.name)
        self.assertEqual(rc, 1)
        self.assertIn("old replay files of s1", log)
        self.assertNotIn("STUB stop_main", log)


if __name__ == "__main__":
    unittest.main()
