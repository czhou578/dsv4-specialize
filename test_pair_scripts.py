"""Dry runs of pair_watch.sh and pair_run.sh (Phase D of plans/PLAN_QWEN_PRUNE.md §8) with stub servers.
No GPU, no docker, no change to the live server or to Hermes."""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fake_server(tmp, port):
    """A process whose command line looks like a vLLM 0.33 server on ``port``, in its own process group."""
    qv = tmp / "venv"
    qv.mkdir(exist_ok=True)
    (qv / "python").unlink(missing_ok=True)
    (qv / "python").symlink_to(sys.executable)
    return subprocess.Popen([str(qv / "python"), "-c", "import time; time.sleep(60)", "serve", "--port", str(port), "x"],
                            start_new_session=True), qv


def watch(tmp, avail, pid, qv):
    (tmp / "pid").write_text(str(pid))
    env = {**os.environ, "WATCH_ONCE": "1", "AVAIL_CMD": f"echo {avail}", "PIDFILE": str(tmp / "pid"), "QV": str(qv),
           "LOG": str(tmp / "watch.log"), "MEM": str(tmp / "mem.log")}
    subprocess.run(["bash", str(HERE / "pair_watch.sh")], env=env, check=True, capture_output=True)
    time.sleep(0.3)


def test_the_watch_stops_qwen_below_the_floor_and_only_qwen(tmp_path):
    qwen, qv = fake_server(tmp_path, 8889)
    other, _ = fake_server(tmp_path, 8888)   # DeepSeek shows on the host as a vLLM process on port 8888
    try:
        watch(tmp_path, 5000, qwen.pid, qv)
        assert qwen.poll() is None                       # above the floor: nothing happens
        watch(tmp_path, 1500, other.pid, qv)
        assert other.poll() is None                      # below the floor, but the pid is not Qwen's: never killed
        assert "no Qwen server to stop" in (tmp_path / "watch.log").read_text()
        watch(tmp_path, 1500, qwen.pid, qv)
        assert qwen.wait(timeout=5) is not None          # below the floor: Qwen is stopped
        assert "the Qwen server (pid" in (tmp_path / "watch.log").read_text()
        assert len((tmp_path / "mem.log").read_text().splitlines()) == 3
    finally:
        for p in (qwen, other):
            p.kill()


PAIR_STUB = r'''
healthy() { case "$1" in *8888) return 0;; *) [ -s "$PIDFILE" ];; esac; }
tmux() { [ -e $STUB_DIR/watch_on ]; }
avail_mb() { echo 30000; }
boot() { sleep 60 & echo $! > "$PIDFILE"; return 0; }
for t in idle qlong dlong qspeed dspeed both; do eval "t_$t() { echo $t >> $STUB_DIR/ran; }"; done
rate() { echo "99.0 tok/s"; }
'''


def pair(tmp, watch_on, keep="0"):
    (tmp / "stub.sh").write_text(PAIR_STUB)
    (tmp / "watch_on").unlink(missing_ok=True)
    if watch_on:
        (tmp / "watch_on").write_text("")
    (tmp / "mem.log").write_text(f"{int(time.time())} 29000\n")
    for d in ("qwen-k208", "qwen-draft-nvfp4"):
        assert (HERE / d / "config.json").exists()
    env = {**os.environ, "PAIR_STUB": str(tmp / "stub.sh"), "STUB_DIR": str(tmp), "TAG": "test", "KEEP": keep,
           "LOG": str(tmp / "pair.log"), "MEM": str(tmp / "mem.log"), "PIDFILE": str(tmp / "q.pid")}
    proc = subprocess.run(["bash", str(HERE / "pair_run.sh")], env=env, capture_output=True, text=True, timeout=120)
    return proc.returncode, (tmp / "pair.log").read_text()


def test_phase_d_needs_the_watch(tmp_path):
    rc, log = pair(tmp_path, watch_on=False)
    assert rc == 1 and "the memory watch (tmux pair-watch) is not running" in log
    assert not (tmp_path / "ran").exists()


def test_phase_d_runs_the_tests_in_order(tmp_path):
    rc, log = pair(tmp_path, watch_on=True)
    assert rc == 0, log
    assert (tmp_path / "ran").read_text().split() == ["idle", "qlong", "dlong", "qspeed", "dspeed", "both"]
    assert "pair-test idle: exit 0; lowest available" in log and "Qwen stopped" in log
