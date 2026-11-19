"""The tools are scripts in tools/, not a package: put that folder on the import path of the tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
