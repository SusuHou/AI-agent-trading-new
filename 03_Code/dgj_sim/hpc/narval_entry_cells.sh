#!/bin/bash
# Entry-timing cells, low noise, beta 5e-7, wide grid, floor 0.01, Dou initial Q for both (Question 181 "a").
# Submits/resumes every session without a session_K.json result; skips a cell while its tasks are queued.
# Safe to rerun (resumes paused sessions from checkpoints). Then prints the summary.
set -euo pipefail
source "$HOME/venv-aitrading/bin/activate"
export PYTHONNOUSERSITE=1   # ignore ~/.local packages
cd "$HOME/scratch/AI-agent-trading-entry" && git fetch -q origin && git reset -q --hard origin/entry-timing
HEAD=$(git rev-parse --short HEAD); echo "entry checkout at $HEAD"
cd 03_Code/dgj_sim && mkdir -p logs
RUNS="$HOME/scratch/ai-trading-runs"
COMMON="--noise-std 0.1 --beta 5e-7 --action-grid wide --floor 0.01"
if [[ "${CANCEL_PENDING:-0}" == "1" ]]; then
  scancel -u "$USER" -t PENDING -n entryC0,entryS1,entryS2,entryS3 && echo "cancelled pending entry tasks"
  sleep 5
fi
submit () {  # $1 name, $2 dir, $3 cell args
  local NAME="$1" D="$2" ARGS="$3"
  if [[ -n "$(squeue -u "$USER" -h -n "$NAME" -o %i)" ]]; then echo "$NAME: tasks in queue, skipped"; return; fi
  mkdir -p "$D"
  local MISSING
  MISSING=$(python - "$D" <<'PY'
import os, sys
d = sys.argv[1]
print(",".join(str(k) for k in range(100) if not os.path.exists(os.path.join(d, f"session_{k:04d}.json"))))
PY
)
  if [[ -z "$MISSING" ]]; then echo "$NAME: 100/100 complete"; return; fi
  local J N
  N=$(echo "$MISSING" | tr ',' '\n' | wc -l)
  J=$(OUT="$D" SEED=20261019 WORK_PERIODS=20000000000 CELL_ARGS="$COMMON $ARGS" \
      sbatch --parsable --array="$MISSING" --time=08:00:00 --job-name="$NAME" hpc/submit_entry.slurm)
  { echo "submitted_utc: $(date -u --iso-8601=seconds)"; echo "job_id: $J"; echo "commit: $HEAD"; echo "sessions: $MISSING"; } >> "$D/submission_receipt.txt"
  echo "$NAME: submitted $N sessions as job $J"
}
submit entryC0 "$RUNS/entry_c0_low_b5e-7" "--entry-period 0 --label entry_c0_low"
submit entryS1 "$RUNS/entry_s1_low_b5e-7" "--entry-period 46000000 --label entry_s1_low"
submit entryS2 "$RUNS/entry_s2_low_b5e-7" "--entry-period 92000000 --label entry_s2_low"
submit entryS3 "$RUNS/entry_s3_low_b5e-7" "--entry-on-convergence --alone-cap 2000000000 --label entry_s3_low"
sleep 20
squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
echo "=== SUMMARY (complete sessions only) ==="
python hpc/read_entry_cells.py "$RUNS/entry_c0_low_b5e-7" "$RUNS/entry_s1_low_b5e-7" "$RUNS/entry_s2_low_b5e-7" "$RUNS/entry_s3_low_b5e-7"
echo DONE
