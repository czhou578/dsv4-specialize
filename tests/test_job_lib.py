"""job_lib.sh and the job calls of run_lib.sh, with the job command of machine-dash and a temporary jobs folder.
Skipped when the job command is missing."""
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # the repo root
JOB = shutil.which("job") or str(Path.home() / "Projects" / "machine-dash" / "bin" / "job")


@unittest.skipUnless(os.access(JOB, os.X_OK), "needs the job command of machine-dash")
class JobLibTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.jobs, self.out = self.tmp / "jobs", self.tmp / "out.txt"
        (self.tmp / "bin").mkdir()
        (self.tmp / "bin" / "job").symlink_to(JOB)

    def run_script(self, body, tmux=True, path=None, **env):
        """Run demo.sh with the arguments 'a b' and 'c'. Returns the exit code and the status records."""
        script = self.tmp / "demo.sh"
        script.write_text("#!/usr/bin/env bash\nset -u\n" + body)
        full = {k: v for k, v in os.environ.items() if k not in ("TMUX", "MACHINE_DASH_JOB", "JOB_WRAP", "JOB_STEPS")}
        full.update(PATH=path or f"{self.tmp / 'bin'}:{os.environ['PATH']}", MACHINE_DASH_JOBS=str(self.jobs), **env)
        if tmux:
            full["TMUX"] = "/tmp/tmux-test/default,1,0"
        proc = subprocess.run(["bash", str(script), "a b", "c"], env=full, capture_output=True, text=True, timeout=60)
        records = [json.loads(p.read_text()) for p in sorted(self.jobs.glob("*.json"))] if self.jobs.exists() else []
        return proc.returncode, records

    def test_wrap_in_tmux(self):
        body = (f'source {HERE}/scripts/job_lib.sh\njob_wrap --log "{self.tmp}/demo.log" -- "$@"\n'
                f'job_say set --phase working\nprintf "%s|" "$@" > {self.out}\nexit 3\n')
        code, records = self.run_script(body)
        self.assertEqual((code, len(records), self.out.read_text()), (3, 1, "a b|c|"))
        rec = records[0]
        self.assertEqual((rec["name"], rec["exit"], rec["phase"], rec["log"]), ("demo", 3, "working", f"{self.tmp}/demo.log"))
        self.assertEqual(rec["cmd"], f"bash {self.tmp}/demo.sh 'a b' c")        # pgrep -f '^bash <script>'
        code, records = self.run_script(body.replace("source", "JOB_NAME='my demo'\nsource"))
        self.assertEqual(sorted(r["name"] for r in records), ["demo", "my demo"])

    def test_no_wrap(self):
        body = f'source {HERE}/scripts/job_lib.sh\njob_wrap -- "$@"\njob_say step 1\nprintf "%s|" "$@" > {self.out}\n'
        for kwargs in ({"tmux": False}, {"JOB_WRAP": "0"}, {"path": "/usr/bin:/bin"}):
            code, records = self.run_script(body, **kwargs)
            self.assertEqual((code, records, self.out.read_text()), (0, [], "a b|c|"), kwargs)

    def test_run_lib(self):
        now = int(time.time())
        env = {"LOG": str(self.tmp / "night.log"), "RESTORE_TAG": "t", "START_AFTER": f"@{now + 3600}",
               "BOOT_BY": f"@{now + 5400}", "SPEED_BY": f"@{now + 5400}", "RESTORE_AT": f"@{now + 7200}",
               "JOB_STEPS": "4"}
        body = (f'source {HERE}/scripts/run_lib.sh\ncp "$MACHINE_DASH_JOB" {self.out}\njob_phase "boot a-k1"\n'
                'point() { return 0; }\nrun_point a 1 replay\njob_phase "replay b-k2"\n')
        code, records = self.run_script(body, **env)
        waiting = json.loads(self.out.read_text())
        self.assertEqual((waiting["wait_until"], waiting["phase"]), (now + 3600, f"until @{now + 3600}"))
        rec = records[0]
        self.assertEqual((code, rec["until"], rec["steps"], rec["step"], rec["phase"], rec["wait_start"]),
                         (0, now + 7200, 4, 1, "replay b-k2", None))
        self.assertEqual((rec["log"], rec["exit"]), (env["LOG"], 0))
        code, records = self.run_script(body.replace(f'cp "$MACHINE_DASH_JOB" {self.out}\n', ""), tmux=False, **env)
        self.assertEqual((code, len(records)), (0, 1))                       # outside tmux: no second record


if __name__ == "__main__":
    unittest.main()
