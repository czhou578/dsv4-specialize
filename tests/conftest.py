"""The tools are scripts in the repo root, not a package: put the root on the import path of the tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
