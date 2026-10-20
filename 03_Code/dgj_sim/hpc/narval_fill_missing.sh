#!/bin/bash
# Rerun only the lone-incumbent sessions that have no "alone_done" progress file (cell.json race, fixed in
# run_entry_cli). Skips an arm while any of its tasks is still queued/running. Safe to run repeatedly.
set -euo pipefail
source "$HOME/venv-aitrading/bin/activate"
cd "$HOME/scratch/AI-agent-trading-entry" && git fetch -q origin && git reset -q --hard origin/entry-timing
HEAD=$(git rev-parse --short HEAD); echo "entry checkout at $HEAD"
cd 03_Code/dgj_sim && mkdir -p logs
RUNS="$HOME/scratch/ai-trading-runs"
COMMON="--noise-std 0.1 --beta 5e-7 --action-grid wide --floor 0.01 --entry-on-convergence --alone-cap 2000000000 --stop-after-alone"
for j in $(squeue -u "$USER" -h -n entryread -o "%i"); do scancel "$j" && echo "cancelled pending read job $j"; done
DEPS=""
for arm in base T1 T2; do
  case $arm in
    base) D="$RUNS/entry_s3_low_b5e-7"; NAME=entry; EXTRA="--label entry_s3_low";;
    T1)   D="$RUNS/entry_s3_low_b5e-7_T1"; NAME=entryT1; EXTRA="--incumbent-init lone --label entry_s3_low_T1";;
    T2)   D="$RUNS/entry_s3_low_b5e-7_T2"; NAME=entryT2; EXTRA="--no-lagged-price --label entry_s3_low_T2";;
  esac
  if [[ -n "$(squeue -u "$USER" -h -n "$NAME" -o %i)" ]]; then echo "$arm: tasks still in queue, skipped"; continue; fi
  MISSING=$(python - "$D" <<'PY'
import json, os, sys
d = sys.argv[1]; out = []
for k in range(100):
    p = os.path.join(d, f"progress_{k:04d}.json")
    try: ok = json.load(open(p))["status"] == "alone_done"
    except Exception: ok = False
    if not ok: out.append(str(k))
print(",".join(out))
PY
)
  if [[ -z "$MISSING" ]]; then echo "$arm: 100/100 done"; continue; fi
  N=$(echo "$MISSING" | tr ',' '\n' | wc -l)
  J=$(OUT="$D" SEED=20261019 WORK_PERIODS=2000000000 CELL_ARGS="$COMMON $EXTRA" \
      sbatch --parsable --array="$MISSING" --time=03:00:00 --job-name="$NAME" hpc/submit_entry.slurm)
  { echo "fill_missing_utc: $(date -u --iso-8601=seconds)"; echo "job_id: $J"; echo "commit: $HEAD"; echo "sessions: $MISSING"; } >> "$D/submission_receipt.txt"
  echo "$arm: resubmitted $N missing sessions as job $J"
  DEPS="$DEPS:$J"
done
for j in $(squeue -u "$USER" -h -n entryT1,entryT2,entry -o "%A" | sort -u); do DEPS="$DEPS:$j"; done
R=$(sbatch --parsable ${DEPS:+--dependency="afterany$DEPS"} hpc/read_entry_alone.slurm)
echo "read job $R ${DEPS:+(after$DEPS)} -> $RUNS/entry_alone_read.txt"
sleep 20
squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
echo DONE
