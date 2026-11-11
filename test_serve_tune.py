"""Runs the tests of serve-tune/ under python -m unittest in the repo root. The folder name has a hyphen, so unittest
cannot import the folder as a package. This file puts the folder on the path and imports its test classes. The image
test (serve-tune/test_fp8_head.py) needs a GPU: serve-tune/test_fp8_head.sh runs it in the server image.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "serve-tune"))
from test_alloc_k import AllocTest  # noqa: E402,F401
from test_prefill_probe import ProbeTest  # noqa: E402,F401
from test_suffix_match import SuffixMatchTest  # noqa: E402,F401
