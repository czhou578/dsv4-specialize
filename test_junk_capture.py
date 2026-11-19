"""Dry runs of junk_capture.sh (plans/PLAN_JUNK_CAPTURE.md §5) with stub servers and client (JUNKCAP_STUB).
No model server, no GPU, no docker."""

import datetime
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

STUB = r'''
R=$STUB_DIR S=$STUB_DIR
stop_main() { log "STUB stop_main"; return ${STUB_STOP:-0}; }
boot() { log "STUB boot $1"; echo "GPU KV cache size: ${STUB_KV:-70,514} tokens" > $R/boot-$1.log; return ${STUB_BOOT:-0}; }
down() { log "STUB down"; }
restore() { log "STUB restore"; }
proc_text() { printf '%s\n' "${STUB_PROC:---enable-return-routed-experts --kv-cache-memory-bytes 6979321856 --max-model-len 131072}" \
  "MODEL_DIR=${STUB_MODEL:-$SERVER/data/tp1}"; }
smoke_test() { log "STUB smoke"; return ${STUB_SMOKE:-0}; }
client() { log "STUB client"; }
memguard_start() { log "STUB memguard"; }
memguard_stop() { :; }
qwen_up() { [ "${STUB_QWEN:-1}" = 1 ]; }
stop_qwen() { log "STUB stop_qwen"; STUB_QWEN=0; }
start_qwen() { log "STUB start_qwen"; }
'''


class WindowTest(unittest.TestCase):
    def run_script(self, minutes=(-1, 20, 50), **env):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = self.d = Path(tmp.name)
        (d / "stub.sh").write_text(STUB)
        for name in ("prompts.json", "junk_routes.py", "capture.py", "compose/compose-capture.yml",
                     "compose/compose-capture-junk.yml", "sums.npz"):
            (d / name).parent.mkdir(exist_ok=True)
            (d / name).write_text("x")
        (d / "pilot").mkdir()
        (d / "data").mkdir()
        for name in ("sessions.jsonl", "tools.json"):
            (d / "data" / name).write_text("x")
        t = lambda m: time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + 60 * m))
        full = {**{k: v for k, v in os.environ.items() if k != "TMUX"}, "JOB_WRAP": "0",
                "JUNKCAP_STUB": str(d / "stub.sh"), "STUB_DIR": str(d), "LOG": str(d / "w.log"),
                "OUT": str(d / "out"), "PROMPTS": str(d / "prompts.json"), "PILOT": str(d / "pilot"),
                "SUMS": str(d / "sums.npz"), "CRON_OK": "1", "START_AFTER": t(minutes[0]),
                "BOOT_BY": t(minutes[1]), "RESTORE_AT": t(minutes[2]),
                **{k: v.replace("{d}", str(d)) for k, v in env.items()}}
        proc = subprocess.run(["bash", str(HERE / "junk_capture.sh")], env=full, capture_output=True, text=True,
                              timeout=120)
        return proc.returncode, (d / "w.log").read_text()

    def assertOrder(self, log, *lines):
        at = [log.index(line) for line in lines]
        self.assertEqual(at, sorted(at), log)

    def test_a_normal_window(self):
        rc, log = self.run_script()
        self.assertEqual(rc, 0, log)
        self.assertOrder(log, "STUB stop_qwen", "STUB stop_main", "STUB memguard", "STUB boot capture-junk",
                         "STUB smoke", "STUB client", "STUB down", "STUB restore", "STUB start_qwen",
                         "junk capture done: 0 rows, 0 errors")
        self.assertNotIn("ERROR", log)
        self.assertFalse((self.d / "sums.npz").exists())        # moved, so the first delta starts from zero
        self.assertEqual(len(list((self.d / "out").glob("routing_sums-before-*.npz"))), 1)

    def test_a_failed_check_skips_the_capture_and_restores(self):
        for env, error in [({"STUB_KV": "473,251"}, "a KV cache of 473251 tokens is not the 6.5 GiB pin"),
                           ({"STUB_MODEL": "/x/data/tp1-k160t5"}, "/models/tp1 is not"),
                           ({"STUB_PROC": "--enable-return-routed-experts --max-model-len 131072"},
                            "the server lacks '--kv-cache-memory-bytes 6979321856'"),
                           ({"STUB_SMOKE": "1"}, "a check failed: no capture")]:
            with self.subTest(env=env):
                rc, log = self.run_script(**env)
                self.assertIn(error, log)
                self.assertNotIn("STUB client", log)
                self.assertOrder(log, "STUB down", "STUB restore", "STUB start_qwen")

    def test_a_failed_boot_or_stop_restores_with_no_capture(self):
        for env, line in [({"STUB_BOOT": "1"}, "STUB boot capture-junk"),
                          ({"STUB_STOP": "9"}, "safety failure before the boot")]:
            with self.subTest(env=env):
                rc, log = self.run_script(**env)
                self.assertIn(line, log)
                self.assertNotIn("STUB smoke", log)
                self.assertOrder(log, line, "STUB restore", "STUB start_qwen")

    def test_the_rank_client_runs_with_its_data(self):
        rc, log = self.run_script(CLIENT="rank", DATA="{d}/data")
        self.assertEqual(rc, 0, log)
        self.assertOrder(log, "STUB smoke", "STUB client", "STUB restore")

    def test_no_qwen_no_restart_unless_asked(self):
        rc, log = self.run_script(STUB_QWEN="0")
        self.assertNotIn("STUB stop_qwen", log)
        self.assertNotIn("STUB start_qwen", log)
        rc, log = self.run_script(STUB_QWEN="0", QWEN_AFTER="1")
        self.assertIn("STUB start_qwen", log)

    def test_no_server_action_before_a_bad_window(self):
        day = datetime.date.today() + datetime.timedelta(days=1)
        while day.isoweekday() > 5:
            day += datetime.timedelta(days=1)
        cron = int((datetime.datetime.combine(day, datetime.time(7, 0)) - datetime.datetime.now()).total_seconds() // 60)
        for kw, message in [({"minutes": (-30, -10, 30)}, "no window, no server action"),
                            ({"minutes": (-1, 20, cron), "CRON_OK": "0"}, "touches the cron window"),
                            ({"PROMPTS": "/no/such/prompts.json"}, "ERROR: no /no/such/prompts.json"),
                            ({"CLIENT": "rank", "DATA": "/no/such/data"}, "ERROR: no /no/such/data/sessions.jsonl"),
                            ({"CLIENT": "other"}, "ERROR: unknown CLIENT other")]:
            with self.subTest(kw=kw):
                rc, log = self.run_script(**kw)
                self.assertEqual(rc, 1)
                self.assertIn(message, log)
                self.assertNotIn("STUB stop", log)


if __name__ == "__main__":
    unittest.main()
