"""qwen_fit.py and the dry run of qwen_fit.sh (plans/PLAN_SECOND_MODEL_FIT.md §4.3): the report arithmetic, the long
prompt size, and the window script under bash with stub boots and stub clients (QWEN_STUB). No server, no GPU."""

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import qwen_fit

HERE = Path(__file__).resolve().parent
STUB = r'''
R=$STUB_DIR
stop_main() { log "STUB stop_main"; }
restore() { log "STUB restore"; }
wait_mem() { :; }
qrunning() { return 1; }
qstop() { sample_stop; QPID=; log "STUB qstop"; }
avail_mb() { echo ${STUB_AVAIL:-118000}; }
gpu_free() { echo "110.0 of 121.7 GiB"; }
sample_start() { printf '1 118000\n2 84000\n9999999999 80000\n' > "$1"; }
sample_stop() { :; }
N=0
launch() {
  N=$((N + 1)); log "STUB launch $1 pin=$PIN budget=${BUDGET[$1]} draft=${DRAFT[$1]}"
  case " ${STUB_FAIL:-} " in *" $1 "*) echo "RuntimeError: boom" > "$2"; return 1;; esac
  if [ "${STUB_SMALL_PIN:-}" = "$1" ] && [ "$PIN" = "$PIN0" ]; then
    echo "ValueError: 8.80 GiB KV cache is needed, which is larger than the available KV cache memory (8.32 GiB)" > "$2"; return 1
  fi
  printf 'INFO Model loading took 20.1 GiB memory and 7 seconds\nINFO GPU KV cache size: 266,240 tokens\n' > "$2"
}
tests() { log "STUB tests $1"; }
nvidia-smi() { echo "4242, 31000 MiB"; }
'''


class ReportTest(unittest.TestCase):
    def test_the_overhead_is_the_peak_use_minus_weights_and_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "mem.log").write_text("1 118000\n2 76000\n5 86000\n6 84952\n7 90000\n")   # 2: the load peak
            (d / "speed.jsonl").write_text("".join(json.dumps(x) + "\n" for x in [
                {"run": 0, "completion_tokens": 1, "decode_tok_s": None}, {"run": 1, "completion_tokens": 101, "decode_tok_s": 10.0},
                {"run": 1, "completion_tokens": 301, "decode_tok_s": 30.0}]))
            self.assertEqual(qwen_fit.rate(d / "speed.jsonl"), (20.0, 2))
            (d / "server.log").write_text("x Model loading took 19.83 GiB memory\ny GPU KV cache size: 266,240 tokens\n")
            (d / "gpu.txt").write_text("4242, 30500 MiB\n")
            r = qwen_fit.report(118000, 88000, 8933531976, d / "mem.log", d / "server.log", d / "gpu.txt", healthy_at=5)
        self.assertAlmostEqual(r["load_peak"], 42000 / 1024, places=2)
        self.assertEqual((r["weights"], r["kv_pin"], r["kv_tokens"], r["gpu_process_mib"]), (19.83, 8.32, 266240, [30500]))
        self.assertAlmostEqual(r["used_at_peak"], (118000 - 84952) / 1024, places=2)
        self.assertAlmostEqual(r["overhead_x"], (118000 - 84952) / 1024 - 19.83 - 8.32, places=2)
        self.assertAlmostEqual(r["overhead_idle"], 30000 / 1024 - 19.83 - 8.32, places=2)

    def test_the_long_prompt_stays_under_its_token_count(self):
        def fake(base, path, body, timeout=None):   # 40 tokens per paragraph
            return {"tokens": [0] * (40 * body["prompt"].count(qwen_fit.PARAGRAPH))}
        with mock.patch.object(qwen_fit, "post", fake):
            text = qwen_fit.long_text("http://x", "m", 250000)
        self.assertEqual(text.count(qwen_fit.PARAGRAPH), (250000 - 200) // 40)


class WindowTest(unittest.TestCase):
    def run_script(self, boot_by=60, **env):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        (d / "stub.sh").write_text(STUB)
        t = lambda m: time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + 60 * m))
        full = {**os.environ, "QWEN_STUB": str(d / "stub.sh"), "STUB_DIR": str(d), "LOG": str(d / "window.log"),
                "START_AFTER": t(-1), "BOOT_BY": t(boot_by), "SPEED_BY": t(boot_by), "RESTORE_AT": t(240), **env}
        proc = subprocess.run(["bash", str(HERE / "qwen_fit.sh")], env=full, capture_output=True, text=True, timeout=120)
        return proc.returncode, (d / "window.log").read_text(), d

    def test_a_normal_window(self):
        rc, log, d = self.run_script()
        self.assertEqual(rc, 0, log)
        self.assertIn("STUB launch f1 pin=8933531976 budget=2048 draft=0", log)
        self.assertIn("STUB launch f2 pin=8933531976 budget=8192 draft=0", log)
        self.assertIn("STUB launch f3 pin=8933531976 budget=2048 draft=1", log)
        self.assertIn("qwen-f1: overhead X 8.69 GiB at the peak", log)       # (118000 - 80000) / 1024 - 20.1 - 8.32
        self.assertEqual(json.loads((d / "qwen-fit-f1.json").read_text())["kv_tokens"], 266240)
        self.assertEqual(log.count("STUB stop_main"), 3)
        self.assertLess(log.rindex("STUB qstop"), log.index("STUB restore"))
        self.assertIn("qwen fit window done", log)

    def test_a_small_pin_gets_one_more_boot_with_the_need_plus_4_percent(self):
        rc, log, _ = self.run_script(STUB_SMALL_PIN="f1", POINTS="f1 f2")
        self.assertEqual(rc, 0, log)
        pin = int(8.80 * 1.04 * 2**30)
        self.assertIn(f"the need is 8.80 GiB: one more boot with {pin} bytes", log)
        self.assertIn(f"STUB launch f2 pin={pin}", log)                       # the later points keep the larger pin
        self.assertIn("STUB tests f1", log)

    def test_failed_boots_skip_the_point_and_two_in_a_row_restore(self):
        rc, log, _ = self.run_script(STUB_FAIL="f2")
        self.assertEqual(rc, 0, log)
        self.assertIn("ERROR: qwen-f2 did not boot", log)
        self.assertNotIn("STUB tests f2", log)
        self.assertIn("STUB tests f3", log)
        rc, log, _ = self.run_script(STUB_FAIL="f1 f2")
        self.assertEqual(rc, 1)
        self.assertIn("two failed boots in a row: restore", log)
        self.assertNotIn("STUB launch f3", log)
        self.assertIn("STUB restore", log)

    def test_the_process_check_matches_a_server_only(self):
        pat = subprocess.run(["bash", "-c", f"source <(grep '^QPAT=' {HERE / 'qwen_fit.sh'}); echo \"$QPAT\""],
                             capture_output=True, text=True).stdout.strip()
        import re
        real = "/home/u/venvs/vllm033/bin/python3 /home/u/venvs/vllm033/bin/vllm serve nvidia/Qwen3.8-27B-NVFP4 --port 8889"
        self.assertTrue(re.search(pat, real))
        for other in ['bash -c pgrep -f "vllm serve" | head', "tmux new -d -s x ./qwen_fit.sh", "grep vllm serve log",
                      "/usr/bin/python3 -m vllm.entrypoints.cli.main serve /models/tp1"]:
            self.assertFalse(re.search(pat, other), other)

    def test_a_past_boot_deadline_ends_before_any_server_action(self):
        rc, log, _ = self.run_script(boot_by=-1)
        self.assertEqual(rc, 1)
        self.assertIn("no boot", log)
        self.assertNotIn("STUB stop_main", log)


if __name__ == "__main__":
    unittest.main()
