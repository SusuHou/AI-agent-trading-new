#!/bin/bash
# Short login: submit T1/T2 (skips arms already submitted), then a read job that runs after them.
# Results: $HOME/scratch/ai-trading-runs/entry_alone_read.txt
set -euo pipefail
cd "$HOME/scratch/AI-agent-trading-entry" && git fetch -q origin && git reset -q --hard origin/entry-timing
HEAD=$(git rev-parse --short HEAD); echo "entry checkout at $HEAD"
cd 03_Code/dgj_sim && mkdir -p logs
RUNS="$HOME/scratch/ai-trading-runs"
COMMON="--noise-std 0.1 --beta 5e-7 --action-grid wide --floor 0.01 --entry-on-convergence --alone-cap 2000000000 --stop-after-alone"
DEPS=""
for arm in T1 T2; do
  D="$RUNS/entry_s3_low_b5e-7_$arm"
  if [[ -f "$D/submission_receipt.txt" ]]; then
    J=$(grep -h job_id "$D/submission_receipt.txt" | tail -1 | awk '{print $2}')
    echo "$arm: already submitted (job $J)"
  else
    EXTRA=$([[ $arm == T1 ]] && echo "--incumbent-init lone" || echo "--no-lagged-price")
    mkdir -p "$D"
    J=$(OUT="$D" SEED=20261019 WORK_PERIODS=2000000000 CELL_ARGS="$COMMON $EXTRA --label entry_s3_low_$arm" \
        sbatch --parsable --time=03:00:00 --job-name="entry$arm" hpc/submit_entry.slurm)
    { echo "submitted_utc: $(date -u --iso-8601=seconds)"; echo "job_id: $J"; echo "commit: $HEAD"; echo "arm: $arm"; } >> "$D/submission_receipt.txt"
    echo "SUBMITTED $arm job=$J"
  fi
  DEPS="$DEPS:$J"
done
R=$(sbatch --parsable --dependency="afterany$DEPS" hpc/read_entry_alone.slurm)
echo "read job $R (runs after$DEPS); results -> $RUNS/entry_alone_read.txt"
squeue -u "$USER" -h -o "%i %j %T" | awk '{split($1,a,"_"); print a[1], $2, $3}' | sort | uniq -c
echo DONE
