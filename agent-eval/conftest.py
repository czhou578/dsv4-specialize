import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

collect_ignore = ["tasks"]   # the visible and hidden tests of the tasks run in the sandbox, not here
