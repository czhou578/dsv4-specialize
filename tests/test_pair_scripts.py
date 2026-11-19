"""Dry runs of pair_watch.sh and pair_run.sh (Phase D of plans/PLAN_QWEN_PRUNE.md §8) with stub servers.
No GPU, no docker, no change to the live server or to Hermes."""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # the repo root


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
    subprocess.run(["bash", str(HERE / "scripts/pair_watch.sh")], env=env, check=True, capture_output=True)
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
        rows = [line.split() for line in (tmp_path / "mem.log").read_text().splitlines()]
        assert [r[1] for r in rows] == ["5000", "1500", "1500"] and all(len(r) == 3 and r[2].isdigit() for r in rows)
    finally:
        for p in (qwen, other):
            p.kill()


PAIR_STUB = r'''
healthy() { case "$1" in *8888) return 0;; *) [ -s "$PIDFILE" ];; esac; }
tmux() { [ -e $STUB_DIR/watch_on ]; }
avail_mb() { echo 30000; }
boot() { setsid sleep 60 & echo $! > "$PIDFILE"; return 0; }   # a child of the script in its own group, as the real server
for t in idle qlong dlong qspeed dspeed; do eval "t_$t() { echo $t >> $STUB_DIR/ran; }"; done
speed_of() { echo "$3" >> $STUB_DIR/speed; }   # the real t_both runs: it must not wait for the server child
rate() { echo "99.0 tok/s"; }
'''


def pair(tmp, watch_on, keep="0", env=None):
    (tmp / "stub.sh").write_text(PAIR_STUB)
    (tmp / "watch_on").unlink(missing_ok=True)
    if watch_on:
        (tmp / "watch_on").write_text("")
    (tmp / "mem.log").write_text(f"{int(time.time())} 29000\n")
    (tmp / "swap.log").write_text(f"{int(time.time())} 29000 1200 8100\n")
    for d in ("qwen-k208", "qwen-draft-nvfp4", "results"):
        (tmp / "s" / d).mkdir(parents=True, exist_ok=True)
    for d in ("qwen-k208", "qwen-draft-nvfp4"):
        (tmp / "s" / d / "config.json").write_text("{}")
    env = {**os.environ, "PAIR_STUB": str(tmp / "stub.sh"), "STUB_DIR": str(tmp), "TAG": "test", "KEEP": keep, "PAIR_S": str(tmp / "s"),
           "LOG": str(tmp / "pair.log"), "MEM": str(tmp / "mem.log"), "PIDFILE": str(tmp / "q.pid"), "SWAPLOG": str(tmp / "swap.log"), **(env or {})}
    proc = subprocess.run(["bash", str(HERE / "scripts/pair_run.sh")], env=env, capture_output=True, text=True, timeout=120)
    return proc.returncode, (tmp / "pair.log").read_text()


def test_phase_d_needs_the_watch(tmp_path):
    rc, log = pair(tmp_path, watch_on=False)
    assert rc == 1 and "the memory watch (tmux pair-watch) is not running" in log
    assert not (tmp_path / "ran").exists()


def test_phase_d_runs_the_tests_in_order(tmp_path):
    rc, log = pair(tmp_path, watch_on=True)
    assert rc == 0, log
    assert (tmp_path / "ran").read_text().split() == ["idle", "qlong", "dlong", "qspeed", "dspeed"]
    assert sorted(Path(x).name for x in (tmp_path / "speed").read_text().split()) == ["pair-test-bothd", "pair-test-bothq"]
    assert "pair-test idle: exit 0; lowest available" in log and "Qwen stopped" in log
    assert "lowest swap-free 8100 MB" in log and "(--language-model-only)" in log


def test_phase_d_with_vision_uses_the_two_cuts(tmp_path):
    rc, log = pair(tmp_path, watch_on=True, env={"VISION": "1", "TESTS": "idle"})
    assert rc == 0, log
    assert '--limit-mm-per-prompt {"image":1,"video":0} --mm-processor-cache-gb 0 --mm-processor-kwargs {"max_pixels":3686400}' in log
    assert not (tmp_path / "speed").exists()


def test_phase_d_two_streams(tmp_path):
    (tmp_path / "s" / "results").mkdir(parents=True)
    names = ["qn-1", "qn-2", "qn-3", "qnd-1", "qnd-2", "qnd-3", "qnd-ds"]
    for name in names:   # the stub speed_of writes no file: give the rate lines their inputs
        (tmp_path / "s" / "results" / f"pair-test-{name}.jsonl").write_text("{}\n")
    rc, log = pair(tmp_path, watch_on=True, env={"SEQS": "3", "KV_BYTES": "3575989770", "TESTS": "idle qn qnd"})
    assert rc == 0, log
    assert "max-num-seqs 3, KV cache 3575989770 bytes (--language-model-only)" in log and "pair-test qn: exit 0" in log and "pair-test qnd: exit 0" in log
    assert sorted(Path(x).name for x in (tmp_path / "speed").read_text().split()) == sorted(f"pair-test-{n}" for n in names)
    assert "qn, 3 streams on Qwen: streams 99.0 99.0 99.0 tok/s, sum 297 tok/s" in log
    assert "DeepSeek beside them 99.0 tok/s" in log
