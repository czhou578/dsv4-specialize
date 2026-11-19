"""Dry runs of cap_switch.sh (the DeepSeek context cap of plans/PLAN_QWEN_PRUNE.md §8) with stub servers.
No docker, no change to the live server or to Hermes."""

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # the repo root
HERMES = """model: x
compression:
  enabled: true
  progress_notices: false
  threshold: 0.5
  target_ratio: 0.2
custom_providers:
  - name: Local (127.0.0.1:8888)
    models:
      deepseek-v4-flash-0731-k160t5:
        context_length: 256000
"""


CAP_STUB = r'''
healthy() { [ "$(cat $STUB_DIR/state)" != down ]; }
busy() { echo 0; }
max_len() { cat $STUB_DIR/len; }
compose() { echo "compose $(basename $1) ${*:2}" >> $STUB_DIR/calls; }
docker() { case "$1" in config) return 0;; inspect) echo true;; logs) cat $STUB_DIR/logs 2>/dev/null;; esac; }
systemctl() { echo active; }
curl() { echo '{"choices":[{"message":{"content":"OK."}}]}'; }
N=0
wait_healthy() {  # the boots of the stub: the outcomes in $STUB_DIR/boots, one per boot
  N=$((N + 1)); local r=$(sed -n "${N}p" $STUB_DIR/boots)
  [ "$r" = ok ] && { echo up > $STUB_DIR/state; echo 160000 > $STUB_DIR/len; return 0; }
  echo down > $STUB_DIR/state; return 1
}
'''


def cap(tmp, boots, logs=""):
    server = tmp / "server"
    server.mkdir(exist_ok=True)
    src = Path("/home/colin-spark/Projects/DeepSeek-v4-Flash-One-DGX-Spark/compose-md149t5.yml")
    (server / "compose-md149t5.yml").write_text(src.read_text())
    (tmp / "hermes.yaml").write_text(HERMES)
    (tmp / "stub.sh").write_text(CAP_STUB)
    (tmp / "boots").write_text("\n".join(boots) + "\n")
    (tmp / "logs").write_text(logs)
    (tmp / "state").write_text("up")
    (tmp / "len").write_text("256000")
    (tmp / "calls").write_text("")
    env = {**os.environ, "CAP_STUB": str(tmp / "stub.sh"), "STUB_DIR": str(tmp), "SERVER": str(server),
           "HERMES": str(tmp / "hermes.yaml"), "LOG": str(tmp / "cap.log"), "PIDFILE": str(tmp / "none.pid")}
    proc = subprocess.run(["bash", str(HERE / "scripts/cap_switch.sh")], env=env, capture_output=True, text=True, timeout=120)
    return proc.returncode, (tmp / "cap.log").read_text(), (tmp / "calls").read_text().splitlines()


def test_the_cap_switch_boots_at_160k_and_changes_hermes(tmp_path):
    rc, log, calls = cap(tmp_path, ["ok"])
    assert rc == 0, log
    assert calls == ["compose compose-md149t5-c160.yml up -d"]
    new = (tmp_path / "server/compose-md149t5-c160.yml").read_text()
    assert 'MAX_MODEL_LEN: "160000"' in new and "--kv-cache-memory-bytes 7431600000" in new
    h = (tmp_path / "hermes.yaml").read_text()
    assert "context_length: 160000" in h and "  threshold: 0.8" in h and "target_ratio: 0.2" in h
    assert list(tmp_path.glob("hermes.yaml.bak.*_cap160")) and "hermes-gateway restarted: active" in log


def test_a_small_pin_gets_one_more_boot_with_the_need(tmp_path):
    rc, log, calls = cap(tmp_path, ["fail", "ok"], logs="ValueError: 6.70 GiB KV cache is needed, which is larger\n")
    assert rc == 0, log
    pin = int(6.70 * 1.04 * 2**30)
    assert f"one more boot with {pin} bytes" in log
    assert f"--kv-cache-memory-bytes {pin}" in (tmp_path / "server/compose-md149t5-c160.yml").read_text()
    assert calls == ["compose compose-md149t5-c160.yml up -d"] * 2


def test_a_failed_switch_rolls_back_and_leaves_hermes(tmp_path):
    rc, log, calls = cap(tmp_path, ["fail", "ok"])   # no "needed" line: no second boot, the rollback
    assert rc == 1
    assert calls == ["compose compose-md149t5-c160.yml up -d", "compose compose-md149t5.yml up -d"]
    assert "ROLLBACK DONE" in log and (tmp_path / "hermes.yaml").read_text() == HERMES
