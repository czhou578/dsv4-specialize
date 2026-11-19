"""The sandbox of the run tool (plans/PLAN_AGENT_CAPACITY.md §5).

Each command runs in a new Docker container of the image ``agent-eval-sandbox:1``:

- no network (``--network none``), a read-only root, and a tmpfs on /tmp;
- the workspace of the run on /work, read-write, as the user of this process;
- 1 GiB of memory with no swap, 256 processes and 4 CPUs;
- ``timeout`` in the container, and ``docker kill`` from the host when the container does not stop.

At most ``SLOTS`` commands run at the same time in this process, so that the builds use at most 2 GiB beside the
model servers. The time in the queue does not count against the limit of a command.
"""

import os
import subprocess
import threading
import time
import uuid

IMAGE = os.environ.get("AGENT_EVAL_IMAGE", "agent-eval-sandbox:1")
SLOTS = threading.BoundedSemaphore(int(os.environ.get("AGENT_EVAL_SLOTS", "2")))
MAX_OUT = 10_000   # characters of output that a tool result keeps: the head and the tail


def clip(text: str, limit: int = MAX_OUT) -> str:
    """The text, or its head and tail with a note of the characters that it leaves out."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - limit} characters left out ...]\n{text[-half:]}"


def docker_cmd(workdir: str, command: str, timeout: int, name: str, memory: str = "1g") -> list[str]:
    return ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,exec,size=1g", "--memory", memory, "--memory-swap", memory,
            "--pids-limit", "256", "--cpus", "4", "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{os.path.abspath(workdir)}:/work", "-w", "/work", IMAGE,
            "timeout", "-k", "5", str(timeout), "bash", "-c", command]


def run(workdir: str, command: str, timeout: int = 120) -> dict:
    """Run one bash command in /work. Returns the exit code, the clipped output, the seconds and the queue seconds.

    Exit code 124 or 137 means the time limit or the memory limit stopped the command."""
    t0 = time.monotonic()
    with SLOTS:
        t1 = time.monotonic()
        name = f"agent-eval-{uuid.uuid4().hex[:12]}"
        try:
            p = subprocess.run(docker_cmd(workdir, command, timeout, name), capture_output=True, text=True,
                               errors="replace", timeout=timeout + 30)
            code, out = p.returncode, p.stdout + p.stderr
        except subprocess.TimeoutExpired as e:
            subprocess.run(["docker", "kill", name], capture_output=True)
            code, out = 137, f"{e.stdout or ''}{e.stderr or ''}\n[the container did not stop: killed]"
    return {"exit": code, "output": clip(out), "seconds": round(time.monotonic() - t1, 2),
            "queue_seconds": round(t1 - t0, 2)}
