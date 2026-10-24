#!/bin/bash
# X0-iv (2026-10-23, author: "a, do the Atanasova-style bins"): X0-pv's cell with --price-grid per_value_interval,
# branch interval-bins (f4d4ef6), seed 20261017 (paired with X0-pv on common random numbers), sessions 0-99.
# Separate checkout ~/scratch/AI-agent-trading-iv. Submits only missing sessions; safe to rerun. Then queues a read
# job (after the array) writing ~/scratch/ai-trading-runs/x0_iv_read.txt, and prints that file if it exists.
set -euo pipefail
source "$HOME/venv-aitrading/bin/activate"; export PYTHONNOUSERSITE=1
CO="$HOME/scratch/AI-agent-trading-iv"; R="$HOME/scratch/ai-trading-runs"; D="$R/x0_iv"
[[ -d "$CO" ]] || git clone -q --branch interval-bins https://github.com/SusuHou/AI-agent-trading-new.git "$CO"
( cd "$CO" && git fetch -q origin && git checkout -q interval-bins && git reset -q --hard origin/interval-bins )
HEAD=$(cd "$CO" && git rev-parse --short HEAD); echo "interval-bins checkout at $HEAD"
cd "$CO/03_Code/dgj_sim" && mkdir -p logs
python -m unittest tests.test_interval_bins 2>&1 | tail -2
mkdir -p "$D"
QUEUED=$(squeue -u "$USER" -h -n x0_iv -r -o %K | paste -sd, -)
MISSING=$(python - "$D" "$QUEUED" <<'PY'
import os, sys
d, q = sys.argv[1], {int(x) for x in sys.argv[2].split(",") if x.strip().isdigit()}
print(",".join(str(k) for k in range(100) if k not in q and not os.path.exists(os.path.join(d, f"session_{k:04d}.npz"))))
PY
)
J=""
if [[ -n "$QUEUED" ]]; then echo "x0_iv: tasks already queued/running (left alone)"; fi
if [[ -n "$MISSING" ]]; then
  J=$(OUT="$D" SEED=20261017 WORK_PERIODS=20000000000 \
      CELL_ARGS="--noise-std 0.1 --speculators 2 --rho 0.95 --xi 500 --convergence-periods 1000000 --measurement-periods 100000 --price-grid per_value_interval --prehistory nash --label x0_iv --irf-paths 0" \
      sbatch --parsable --array="$MISSING%100" --job-name=x0_iv hpc/submit_array.slurm)
  { echo "submitted_utc: $(date -u --iso-8601=seconds)"; echo "job_id: $J"; echo "commit: $HEAD"; echo "sessions: $MISSING"; } >> "$D/submission_receipt.txt"
  echo "x0_iv: submitted $(echo "$MISSING" | tr ',' ' ' | wc -w) sessions as job $J"
else
  echo "x0_iv: nothing to submit"
fi
for j in $(squeue -u "$USER" -h -n x0ivread -o %i); do scancel "$j"; done
DEP=$(squeue -u "$USER" -h -n x0_iv -o %A | sort -u | paste -sd: -)
RJ=$(cd "$HOME/scratch/AI-agent-trading-entry/03_Code/dgj_sim" && git pull -q; cd "$HOME/scratch/AI-agent-trading-entry/03_Code/dgj_sim" && sbatch --parsable ${DEP:+--dependency=afterany:$DEP} hpc/read_x0_iv.slurm)
echo "read job $RJ ${DEP:+(after $DEP)} -> $R/x0_iv_read.txt"
sleep 10; squeue -u "$USER" -h -r -o "%j %T" | sort | uniq -c
echo "=== last read ==="; cat "$R/x0_iv_read.txt" 2>/dev/null || echo "no read yet"
echo DONE
