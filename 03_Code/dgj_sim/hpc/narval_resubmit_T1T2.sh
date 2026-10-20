#!/bin/bash
# 2026-10-19: T1/T2 (jobs 6438470/6438471) failed in 1-3 s: venv not active (no numba). submit_entry.slurm now
# activates it. Cancel the old read job, resubmit any arm with no progress files, queue a new read job.
set -euo pipefail
source "$HOME/venv-aitrading/bin/activate"
cd "$HOME/scratch/AI-agent-trading-entry" && git fetch -q origin && git reset -q --hard origin/entry-timing
HEAD=$(git rev-parse --short HEAD); echo "entry checkout at $HEAD"
cd 03_Code/dgj_sim && mkdir -p logs
grep -q venv-aitrading hpc/submit_entry.slurm || { echo "STOP: slurm script does not activate the venv"; exit 1; }
scancel 6438672 2>/dev/null && echo "cancelled old read job 6438672" || true
RUNS="$HOME/scratch/ai-trading-runs"
COMMON="--noise-std 0.1 --beta 5e-7 --action-grid wide --floor 0.01 --entry-on-convergence --alone-cap 2000000000 --stop-after-alone"
DEPS=""
for arm in T1 T2; do
  D="$RUNS/entry_s3_low_b5e-7_$arm"
  if squeue -u "$USER" -h -o "%j" | grep -qx "entry$arm"; then echo "STOP: entry$arm still in queue"; exit 1; fi
  if ls "$D"/progress_*.json >/dev/null 2>&1; then echo "$arm: has progress files, not resubmitted"; continue; fi
  EXTRA=$([[ $arm == T1 ]] && echo "--incumbent-init lone" || echo "--no-lagged-price")
  mkdir -p "$D"
  J=$(OUT="$D" SEED=20261019 WORK_PERIODS=2000000000 CELL_ARGS="$COMMON $EXTRA --label entry_s3_low_$arm" \
      sbatch --parsable --time=03:00:00 --job-name="entry$arm" hpc/submit_entry.slurm)
  { echo "resubmitted_utc: $(date -u --iso-8601=seconds)"; echo "job_id: $J"; echo "commit: $HEAD"; echo "reason: first submission failed (venv not active)"; } >> "$D/submission_receipt.txt"
  echo "RESUBMITTED $arm job=$J"
  DEPS="$DEPS:$J"
done
if [[ -n "$DEPS" ]]; then
  R=$(sbatch --parsable --dependency="afterany$DEPS" hpc/read_entry_alone.slurm)
  echo "read job $R (after$DEPS) -> $RUNS/entry_alone_read.txt"
fi
sleep 20
squeue -u "$USER" -h -o "%i %j %T" | awk '{split($1,a,"_"); print a[1], $2, $3}' | sort | uniq -c
echo DONE
