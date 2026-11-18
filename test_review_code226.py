"""Dry runs of review_code226.sh (plans/PLAN_CODE226_DEPLOY.md §5) with stub servers and clients (REVIEW_STUB).
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
stop_main() { log "STUB stop_main"; }
boot() {
  local n=$(( $(cat $R/tries-$1 2>/dev/null || echo 0) + 1 )); echo $n > $R/tries-$1
  log "STUB boot $1 len=$CAP_LEN pin=$TUNE_KV_BYTES try=$n"
  if [ "$1" = rv-code226 ] && [ "${STUB_KVFAIL:-0}" = 1 ] && [ $n = 1 ]; then
    echo "ValueError: 6.71 GiB KV cache is needed" > $R/boot-$1.log; return 1; fi
  if [[ " ${STUB_BOOTFAIL:-} " == *" $1 "* ]]; then
    echo "boot error" > $R/boot-$1.log; return 1; fi
  printf 'Model loading took 71.78 GiB\nper-layer experts: 15 counts\nGPU KV cache size: 170,000 tokens\n' > $R/boot-$1.log
}
down() { log "STUB down"; }
restore() { log "STUB restore"; }
replay() { log "STUB replay $1"; }
speed() { log "STUB speed $1 runs=$BOT_RUNS"; }
probe() { log "STUB $2 $1 ${*:3}"; }
qwen_up() { [ "${STUB_QWEN:-1}" = 1 ]; }
stop_qwen() { log "STUB stop_qwen"; STUB_QWEN=0; }
start_qwen() { log "STUB start_qwen"; }
'''


class ReviewTest(unittest.TestCase):
    def run_script(self, minutes=(-1, 60, 150), **env):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        (d / "stub.sh").write_text(STUB)
        for f in ("compose-cap.yml", "compose-perlayer.yml", "image-patch/perlayer/exl3.py", "baseline-k216.jsonl",
                  "image-patch/perlayer/deepseek_v4_model.py", "serve-tune/prefill_probe.py",
                  "tp1-prunemd5-k149/config.json", "tp1-prunecode226t5-k154/config.json"):
            (d / f).parent.mkdir(parents=True, exist_ok=True)
            (d / f).write_text("x")
        t = lambda m: time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + 60 * m))
        full = {**{k: v for k, v in os.environ.items() if k != "TMUX"}, "JOB_WRAP": "0",
                "REVIEW_STUB": str(d / "stub.sh"), "STUB_DIR": str(d), "LOG": str(d / "w.log"), "CRON_OK": "1",
                "START_AFTER": t(minutes[0]), "BOOT_BY": t(minutes[1]), "SPEED_BY": t(minutes[1]),
                "RESTORE_AT": t(minutes[2]), **env}
        proc = subprocess.run(["bash", str(HERE / "review_code226.sh")], env=full, capture_output=True, text=True,
                              timeout=120)
        return proc.returncode, (d / "w.log").read_text()

    def assertOrder(self, log, *lines):
        at = [log.index(line) for line in lines]
        self.assertEqual(at, sorted(at), log)

    def test_a_normal_window(self):
        rc, log = self.run_script()
        self.assertEqual(rc, 0, log)
        self.assertOrder(log, "STUB stop_qwen", "STUB boot rv-md len=256000 pin=7784628224", "STUB smoke rv-md",
                         "STUB replay rv-md-a", "STUB speed rv-md runs=3", "STUB down",
                         "rv-code226: folder tp1-prunecode226t5-k154, max-model-len 160000, KV pin 7431600000 bytes",
                         "STUB boot rv-code226 len=160000 pin=7431600000", "STUB replay rv-code226-a",
                         "STUB replay rv-code226-b", "STUB speed rv-code226 runs=5",
                         "STUB probe rv-code226 --cold 8000 32000 128000", "STUB restore", "STUB start_qwen",
                         "rv-code226 boot: GPU KV cache size: 170,000 tokens", "review window done")
        self.assertNotIn("STUB replay rv-md-b", log)
        self.assertNotIn("STUB probe rv-md", log)

    def test_a_small_kv_pin_gets_one_more_boot_at_the_need_plus_4_percent(self):
        rc, log = self.run_script(STUB_KVFAIL="1")
        pin = int(6.71 * 1.04 * 2 ** 30)
        self.assertIn(f"the KV pin was too small (the need is 6.71 GiB): one more boot with {pin} bytes", log)
        self.assertOrder(log, "STUB boot rv-code226 len=160000 pin=7431600000 try=1",
                         f"STUB boot rv-code226 len=160000 pin={pin} try=2", "STUB replay rv-code226-a")
        self.assertIn(f"KV pin {pin};", log)

    def test_two_failed_boots_restore_and_no_client_runs(self):
        rc, log = self.run_script(STUB_BOOTFAIL="rv-md rv-code226")
        self.assertIn("two failed boots in a row: restore", log)
        self.assertNotIn("STUB replay", log)
        self.assertOrder(log, "STUB restore", "STUB start_qwen")

    def test_no_server_action_before_a_bad_window(self):
        day = datetime.date.today() + datetime.timedelta(days=1)
        while day.isoweekday() > 5:
            day += datetime.timedelta(days=1)
        cron = int((datetime.datetime.combine(day, datetime.time(7, 0)) - datetime.datetime.now()).total_seconds() // 60)
        for kw, message in [({"minutes": (-30, -10, 30)}, "no window, no server action"),
                            ({"minutes": (-1, 20, cron), "CRON_OK": "0"}, "touches the cron window")]:
            with self.subTest(kw=kw):
                rc, log = self.run_script(**kw)
                self.assertEqual(rc, 1)
                self.assertIn(message, log)
                self.assertNotIn("STUB stop", log)


if __name__ == "__main__":
    unittest.main()
