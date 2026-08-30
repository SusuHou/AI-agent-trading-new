"""Make ``dgj`` importable when tests run from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The readable step-by-step oracle, if present next to this project.
STEPS_ROOT = ROOT.parent / "vibe_replication"
STEPS_DIR = STEPS_ROOT / "steps"
HAVE_STEPS = STEPS_DIR.is_dir()
