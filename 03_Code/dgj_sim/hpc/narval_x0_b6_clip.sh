#!/bin/bash
# 2026-10-24, author (via the writing session): "2和3都想试试".
# #2 X0-b6: X0-pv's cell with --beta 5e-6 (branch x0-beta, NEW checkout ~/scratch/AI-agent-trading-beta), seed 20261017,
#    sessions 0-99 -> ~/scratch/ai-trading-runs/x0_b6; paired read vs X0-pv after the array -> x0_b6_read.txt.
# #3 clipping diagnostic (read-only) as a Slurm job -> clip_share.txt.
# Safe to rerun: submits only missing sessions; re-queues the reads; prints the latest outputs.
set -euo pipefail
source "$HOME/venv-aitrading/bin/activate"; export PYTHONNOUSERSITE=1
CO="$HOME/scratch/AI-agent-trading-beta"; R="$HOME/scratch/ai-trading-runs"; D="$R/x0_b6"
ENTRY="$HOME/scratch/AI-agent-trading-entry/03_Code/dgj_sim"
[[ -d "$CO" ]] || git clone -q --branch x0-beta https://github.com/SusuHou/AI-agent-trading-new.git "$CO"
( cd "$CO" && git fetch -q origin && git checkout -q x0-beta && git reset -q --hard origin/x0-beta )
HEAD=$(cd "$CO" && git rev-parse --short HEAD); echo "x0-beta checkout at $HEAD"
cd "$CO/03_Code/dgj_sim" && mkdir -p logs
PYTHONPATH=. python -m unittest tests.test_interval_bins tests.test_run_session_cli 2>&1 | tail -2
mkdir -p "$D"
QUEUED=$(squeue -u "$USER" -h -n x0_b6 -r -o %K | paste -sd, -)
MISSING=$(python - "$D" "$QUEUED" <<'PY'
import os, sys
d, q = sys.argv[1], {int(x) for x in sys.argv[2].split(",") if x.strip().isdigit()}
print(",".join(str(k) for k in range(100) if k not in q and not os.path.exists(os.path.join(d, f"session_{k:04d}.npz"))))
PY
)
[[ -n "$QUEUED" ]] && echo "x0_b6: tasks already queued/running (left alone)"
if [[ -n "$MISSING" ]]; then
  J=$(OUT="$D" SEED=20261017 WORK_PERIODS=20000000000 \
      CELL_ARGS="--noise-std 0.1 --speculators 2 --rho 0.95 --xi 500 --convergence-periods 1000000 --measurement-periods 100000 --price-grid per_value --prehistory nash --label x0_b6 --irf-paths 0 --beta 5e-6" \
      sbatch --parsable --array="$MISSING%100" --job-name=x0_b6 hpc/submit_array.slurm)
  { echo "submitted_utc: $(date -u --iso-8601=seconds)"; echo "job_id: $J"; echo "commit: $HEAD"; echo "sessions: $MISSING"; } >> "$D/submission_receipt.txt"
  echo "x0_b6: submitted $(echo "$MISSING" | tr ',' ' ' | wc -w) sessions as job $J"
else
  echo "x0_b6: nothing to submit"
fi
( cd "$ENTRY" && git pull -q )
for j in $(squeue -u "$USER" -h -n pairread,clipshare -o %i); do scancel "$j"; done
DEP=$(squeue -u "$USER" -h -n x0_b6 -o %A | sort -u | paste -sd: -)
RJ=$(cd "$ENTRY" && A_CO="$HOME/scratch/AI-agent-trading-new" A_DIR="$R/x0_pv" B_CO="$CO" B_DIR="$D" OUTF="$R/x0_b6_read.txt" LABEL="X0-b6 (beta 5e-6) vs X0-pv" \
     sbatch --parsable ${DEP:+--dependency=afterany:$DEP} hpc/read_pair.slurm)
CJ=$(cd "$ENTRY" && sbatch --parsable hpc/clip_share.slurm)
echo "x0_b6 read job $RJ ${DEP:+(after $DEP)} -> $R/x0_b6_read.txt;  clip job $CJ -> $R/clip_share.txt"
sleep 10; squeue -u "$USER" -h -r -o "%j %T" | sort | uniq -c
echo "=== last x0_b6 read ==="; cat "$R/x0_b6_read.txt" 2>/dev/null || echo "no read yet"
echo "=== last clip_share ==="; cat "$R/clip_share.txt" 2>/dev/null || echo "no clip output yet"
echo DONE
