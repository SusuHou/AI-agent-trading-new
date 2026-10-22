#!/bin/bash
# Tests of why the late entrant gains (Question 187, author: "b"):
#   A  = S2 (entry at 92M) with the incumbent's exploration reset at entry   -> entry_s2_low_b5e-7_A, vs S2
#   C  = S3 with lone-trader initial values for the incumbent (continues T1) -> entry_s3_low_b5e-7_T1, vs S3
# Submits only sessions without a result and not already queued; safe to rerun. Prints both comparisons.
set -euo pipefail
source "$HOME/venv-aitrading/bin/activate"
VPY="$HOME/venv-aitrading/bin/python"
cd "$HOME/scratch/AI-agent-trading-entry" && git fetch -q origin && git reset -q --hard origin/entry-timing
HEAD=$(git rev-parse --short HEAD); echo "entry checkout at $HEAD"
cd 03_Code/dgj_sim && mkdir -p logs
R="$HOME/scratch/ai-trading-runs"
COMMON="--noise-std 0.1 --beta 5e-7 --action-grid wide --floor 0.01"
submit () {  # $1 name, $2 dir, $3 cell args
  local NAME="$1" D="$2" ARGS="$3" QUEUED MISSING J N
  mkdir -p "$D"
  QUEUED=$(squeue -u "$USER" -h -n "$NAME" -r -o %K | paste -sd, -)
  MISSING=$("$VPY" -I - "$D" "$QUEUED" <<'PY'
import os, sys
d, q = sys.argv[1], {int(x) for x in sys.argv[2].split(",") if x.strip().isdigit()}
print(",".join(str(k) for k in range(100) if k not in q and not os.path.exists(os.path.join(d, f"session_{k:04d}.json"))))
PY
)
  [[ -n "$QUEUED" ]] && echo "$NAME: tasks already queued/running (left alone)"
  if [[ -z "$MISSING" ]]; then echo "$NAME: nothing to submit"; return; fi
  N=$(echo "$MISSING" | tr ',' ' ' | wc -w)
  J=$(OUT="$D" SEED=20261019 WORK_PERIODS=20000000000 CELL_ARGS="$COMMON $ARGS" \
      sbatch --parsable --array="$MISSING" --time=08:00:00 --job-name="$NAME" hpc/submit_entry.slurm)
  { echo "submitted_utc: $(date -u --iso-8601=seconds)"; echo "job_id: $J"; echo "commit: $HEAD"; echo "sessions: $MISSING"; } >> "$D/submission_receipt.txt"
  echo "$NAME: submitted $N sessions as job $J"
}
submit entryA2   "$R/entry_s2_low_b5e-7_A"  "--entry-period 92000000 --reset-incumbent-at-entry --label entry_s2_low_A"
submit entryS3T1 "$R/entry_s3_low_b5e-7_T1" "--entry-on-convergence --alone-cap 2000000000 --incumbent-init lone --label entry_s3_low_T1"
sleep 15
squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
echo "=== TEST A: S2 vs S2 with incumbent reset ==="
"$VPY" -I hpc/read_entry_cells.py "$R/entry_s2_low_b5e-7" "$R/entry_s2_low_b5e-7_A"
echo "=== TEST C: S3 vs S3 with lone-trader incumbent ==="
"$VPY" -I hpc/read_entry_cells.py "$R/entry_s3_low_b5e-7" "$R/entry_s3_low_b5e-7_T1"
echo DONE
