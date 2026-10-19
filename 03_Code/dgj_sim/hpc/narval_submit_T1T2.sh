#!/bin/bash
# 2026-10-19: diagnostics T1 (lone-trader initial Q) and T2 (no lagged price), 100 lone incumbents each,
# paired with entry_s3_low_b5e-7 (same seed and cell otherwise); then read all three.
set -euo pipefail
source "$HOME/venv-aitrading/bin/activate"
cd "$HOME/scratch/AI-agent-trading-entry" && git fetch -q origin && git reset -q --hard origin/entry-timing
HEAD=$(git rev-parse --short HEAD); echo "entry checkout at $HEAD"
cd 03_Code/dgj_sim
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH="$PWD"
python -m unittest tests.test_entry 2>&1 | tail -2
RUNS="$HOME/scratch/ai-trading-runs"
COMMON="--noise-std 0.1 --beta 5e-7 --action-grid wide --floor 0.01 --entry-on-convergence --alone-cap 2000000000 --stop-after-alone"
for arm in T1 T2; do
  D="$RUNS/entry_s3_low_b5e-7_$arm"
  if [[ -d "$D" ]]; then echo "$arm: folder exists, not resubmitted"; continue; fi
  EXTRA=$([[ $arm == T1 ]] && echo "--incumbent-init lone" || echo "--no-lagged-price")
  mkdir -p "$D"
  J=$(OUT="$D" SEED=20261019 WORK_PERIODS=2000000000 CELL_ARGS="$COMMON $EXTRA --label entry_s3_low_$arm" \
      sbatch --parsable --time=03:00:00 --job-name="entry$arm" hpc/submit_entry.slurm)
  { echo "submitted_utc: $(date -u --iso-8601=seconds)"; echo "job_id: $J"; echo "commit: $HEAD"; echo "arm: $arm"; } >> "$D/submission_receipt.txt"
  echo "SUBMITTED $arm job=$J"
done
echo "=== QUEUE ==="
squeue -u "$USER" -h -o "%i %j %T" | awk '{split($1,a,"_"); print a[1], $2, $3}' | sort | uniq -c
echo "=== READ (base; T1/T2 when finished) ==="
python hpc/read_entry_alone.py "$RUNS/entry_s3_low_b5e-7" "$RUNS/entry_s3_low_b5e-7_T1" "$RUNS/entry_s3_low_b5e-7_T2" 2>&1 | grep -v "^OpenBLAS"
echo DONE
