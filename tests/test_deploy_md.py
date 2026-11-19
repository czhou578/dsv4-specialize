"""deploy_md.sh: the profiles and the Hermes change of the cap (plans/PLAN_CODE226_DEPLOY.md §4). No server action:
the tests run only the modes vars and hermes-cap (with NO_RESTART=1, on a copy of a config file)."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # the repo root
CONFIG = """custom_providers:
  - name: Local (127.0.0.1:8888)
    models:
      deepseek-v4-flash-0731-k160t5:
        context_length: 256000
  - name: qwen
    models:
      second-model-fit:
        context_length: 128000
  - name: auto
    models:
      auto:
        context_length: 256000
compression:
  enabled: true
  progress_notices: false
  threshold: 0.5
  target_ratio: 0.2
"""


class DeployTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)

    def run_mode(self, mode, **env):
        full = {**{k: v for k, v in os.environ.items() if k != "TMUX"}, "JOB_WRAP": "0",
                "LOG": str(self.d / "deploy.log"), **env}
        p = subprocess.run(["bash", str(HERE / "scripts/deploy_md.sh"), mode], env=full, capture_output=True, text=True,
                           timeout=60)
        return p.returncode, p.stdout + p.stderr

    def test_the_profiles(self):
        rc, out = self.run_mode("vars")
        self.assertEqual(rc, 0, out)
        server = "/home/colin-spark/Projects/DeepSeek-v4-Flash-One-DGX-Spark"
        for s in ("DEPLOY=md149t5", f"NEW={server}/compose-md149t5.yml", f"MAIN_COMPOSE={server}/compose-k160t5.yml",
                  "EXPECT_GIB=69.79", "MIN_KV=256000", "CAP=none"):
            self.assertIn(s, out)
        rc, out = self.run_mode("vars", DEPLOY="code226t5", EXPECT_GIB="71.80")
        self.assertEqual(rc, 0, out)
        for s in (f"NEW={server}/compose-code226t5.yml", f"MAIN_COMPOSE={server}/compose-md149t5.yml",
                  "FOLDER_SRC=/home/colin-spark/Projects/dsv4-specialize/data/tp1-prunecode226t5-k154",
                  f"FOLDER={server}/data/tp1-code226t5", "MAIN_CONTAINER=deepseek-v4-flash-md149t5-deepseek-v4-flash-1",
                  "EXPECT_GIB=71.80", "MIN_KV=160000", "CAP=160000"):
            self.assertIn(s, out)
        rc, out = self.run_mode("vars", DEPLOY="other")
        self.assertEqual(rc, 2)
        self.assertIn("unknown DEPLOY other", out)

    def test_hermes_cap_changes_two_lengths_and_the_threshold_with_a_backup(self):
        cfg = self.d / "config.yaml"
        cfg.write_text(CONFIG)
        rc, out = self.run_mode("hermes-cap", DEPLOY="code226t5", HERMES=str(cfg), NO_RESTART="1")
        self.assertEqual(rc, 0, out + (self.d / "deploy.log").read_text())
        t = cfg.read_text()
        self.assertEqual((t.count("context_length: 160000"), t.count("context_length: 128000")), (2, 1))
        self.assertIn("threshold: 0.8", t)
        backups = list(self.d.glob("config.yaml.bak.*_code226t5"))
        self.assertEqual([b.read_text() for b in backups], [CONFIG])
        rc, out = self.run_mode("hermes-cap", DEPLOY="code226t5", HERMES=str(cfg), NO_RESTART="1")
        self.assertEqual(rc, 1)                                       # nothing left to change: refused, restored
        self.assertEqual(cfg.read_text(), t)
        self.assertIn("the Hermes change failed", (self.d / "deploy.log").read_text())

    def test_hermes_cap_does_nothing_without_a_cap(self):
        cfg = self.d / "config.yaml"
        cfg.write_text(CONFIG)
        rc, out = self.run_mode("hermes-cap", HERMES=str(cfg), NO_RESTART="1")
        self.assertEqual((rc, cfg.read_text(), list(self.d.glob("config.yaml.bak.*"))), (0, CONFIG, []))


if __name__ == "__main__":
    unittest.main()
