#!/bin/bash
# 2026-10-21: hold the pending entry jobs and diagnose the Python environment. Changes nothing else.
set -uo pipefail
for n in entryC0 entryS1 entryS2 entryS3; do scancel -u "$USER" -t PENDING -n "$n" && echo "held: cancelled pending $n"; done
sleep 3; echo "=== queue ==="; squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
echo "=== modules ==="; module -t list 2>&1 | tr '\n' ' '; echo
V="$HOME/venv-aitrading"
echo "=== venv ==="; ls -l "$V/bin/python"*; cat "$V/pyvenv.cfg"
echo "=== venv site-packages (numpy/numba) ==="; ls -d "$V"/lib/python*/site-packages/{numpy,numba,llvmlite}* 2>&1 | head
echo "=== ~/.local (numpy/numba) ==="; ls -ld --time-style=+%Y-%m-%d_%H:%M "$HOME"/.local/lib/python*/site-packages/{numpy,numba,llvmlite}* 2>&1 | head -20
source "$V/bin/activate"
for mode in user nouser; do
  echo "=== import test ($mode site) ==="
  if [[ $mode == nouser ]]; then export PYTHONNOUSERSITE=1; else unset PYTHONNOUSERSITE; fi
  python - <<'PY' 2>&1 | tail -4
import sys; print(sys.executable, sys.version.split()[0], "user site:", __import__("site").ENABLE_USER_SITE)
try:
    import numpy; print("numpy", numpy.__version__, numpy.__file__)
    import numba; print("numba", numba.__version__, numba.__file__)
except Exception as e: print("IMPORT FAILED:", type(e).__name__, e)
PY
done
echo "=== pip in venv (no user site) ==="; PYTHONNOUSERSITE=1 python -m pip list 2>/dev/null | grep -iE "^(numpy|numba|llvmlite|scipy) " || echo "none"
echo "=== runtime identity recorded by the lone runs ==="
python - <<'PY' 2>/dev/null || true
import json, numpy as np
d = np.load("/home/" + __import__("os").environ["USER"] + "/scratch/ai-trading-runs/entry_s3_low_b5e-7/ckpt_0000.npz")
print(sorted(d.files)[:3])
PY
grep -h "Numba\|NumPy" "$HOME"/scratch/AI-agent-trading-new/03_Code/dgj_sim/logs/*.out 2>/dev/null | sort | uniq -c | tail -5
echo DONE
