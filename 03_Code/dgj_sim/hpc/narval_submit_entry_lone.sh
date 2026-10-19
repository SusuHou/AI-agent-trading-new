#!/bin/bash
# Entry timing, measure-first run (2026-10-19): 100 lone incumbents, low noise, wide grid, floor 0.01,
# beta 5e-7, stage-3 cell with --stop-after-alone (alone cap 2e9). Separate checkout; existing runs untouched.
set -euo pipefail
source "$HOME/venv-aitrading/bin/activate"
CO="$HOME/scratch/AI-agent-trading-entry"
if [[ ! -d "$CO" ]]; then
  git clone -q --branch entry-timing https://github.com/SusuHou/AI-agent-trading-new.git "$CO"
fi
cd "$CO" && git fetch -q origin && git checkout -q entry-timing && git reset -q --hard origin/entry-timing
HEAD=$(git rev-parse --short HEAD); echo "entry checkout at $HEAD"
cd 03_Code/dgj_sim
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH="$PWD"
python -m unittest tests.test_entry 2>&1 | tail -3

RUNS="$HOME/scratch/ai-trading-runs"
D="$RUNS/entry_s3_low_b5e-7"
if squeue -u "$USER" -h -o "%Z %j" | grep -q "entry"; then echo "STOP: entry jobs already queued"; exit 1; fi
mkdir -p "$D"
J=$(OUT="$D" SEED=20261019 WORK_PERIODS=2000000000 \
    CELL_ARGS="--noise-std 0.1 --beta 5e-7 --action-grid wide --floor 0.01 --entry-on-convergence --alone-cap 2000000000 --stop-after-alone --label entry_s3_low" \
    sbatch --parsable --time=03:00:00 hpc/submit_entry.slurm)
{ echo "submitted_utc: $(date -u --iso-8601=seconds)"; echo "job_id: $J"; echo "commit: $HEAD"; echo "mode: stop-after-alone (measure first)"; } >> "$D/submission_receipt.txt"
echo "SUBMITTED entry lone run job=$J into $D"

echo "=== QUEUE (job, state, count) ==="
squeue -u "$USER" -h -o "%i %T" | awk '{split($1,a,"_"); print a[1], $2}' | sort | uniq -c
echo "=== midterm resubmissions ==="
for d in x0_pv x3a_cartel x4_irf_pv; do
  echo "$d results: $(ls "$RUNS/$d"/session_*.npz 2>/dev/null | wc -l)  checkpoints: $(ls "$RUNS/$d"/ckpt_*.npz 2>/dev/null | wc -l)"
done
echo DONE
