#!/bin/bash
# 2026-10-24, author: "1 和 2" (Question 190).
# #1 E3 beta grid (decided S8, Question 162 "c"): beta in {5e-8, 5e-6, 5e-5} x {C0, S1, S2, S3}, low noise, wide menu,
#    floor 0.01, Dou initial Q; S1/S2 at the periods where eps reaches 0.1 / 0.01 (n_v ln(1/eps)/beta, rounded to the
#    chunk); S3 on incumbent convergence; joint training capped 2e10 after entry (flagged).
# #2 menu test: simultaneous start, floor 0.01, menu "aligned_fine" (step = gap/12, cartel and Nash on the grid)
#    with 15 actions (m1, Dou-like range) and 43 actions (m2, reaches the monopolist).
# Submits only sessions without a result and not already queued; safe to rerun (resumes paused sessions).
set -euo pipefail
source "$HOME/venv-aitrading/bin/activate"; export PYTHONNOUSERSITE=1
VPY="$HOME/venv-aitrading/bin/python"
cd "$HOME/scratch/AI-agent-trading-entry" && git fetch -q origin && git reset -q --hard origin/entry-timing
HEAD=$(git rev-parse --short HEAD); echo "entry checkout at $HEAD"
cd 03_Code/dgj_sim && mkdir -p logs
R="$HOME/scratch/ai-trading-runs"
LOW="--noise-std 0.1 --action-grid wide --floor 0.01 --joint-cap 20000000000"
submit () {  # $1 name, $2 dir, $3 chunk, $4 cell args
  local NAME="$1" D="$2" CH="$3" ARGS="$4" QUEUED MISSING J N
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
  J=$(OUT="$D" SEED=20261019 WORK_PERIODS=20000000000 CHUNK="$CH" CELL_ARGS="$ARGS" \
      sbatch --parsable --array="$MISSING" --time=08:00:00 --job-name="$NAME" hpc/submit_entry.slurm)
  { echo "submitted_utc: $(date -u --iso-8601=seconds)"; echo "job_id: $J"; echo "commit: $HEAD"; echo "chunk: $CH"; echo "sessions: $MISSING"; } >> "$D/submission_receipt.txt"
  echo "$NAME: submitted $N sessions as job $J"
}
# beta = 5e-8 (eps 0.1 at 460.5M, 0.01 at 921.0M)
submit eB8C0 "$R/entry_c0_low_b5e-8" 1000000 "$LOW --beta 5e-8 --entry-period 0         --label entry_c0_low_b5e-8"
submit eB8S1 "$R/entry_s1_low_b5e-8" 1000000 "$LOW --beta 5e-8 --entry-period 461000000 --label entry_s1_low_b5e-8"
submit eB8S2 "$R/entry_s2_low_b5e-8" 1000000 "$LOW --beta 5e-8 --entry-period 921000000 --label entry_s2_low_b5e-8"
submit eB8S3 "$R/entry_s3_low_b5e-8" 1000000 "$LOW --beta 5e-8 --entry-on-convergence --alone-cap 5000000000 --label entry_s3_low_b5e-8"
# beta = 5e-6 (eps 0.1 at 4.6M, 0.01 at 9.2M)
submit eB6C0 "$R/entry_c0_low_b5e-6" 100000 "$LOW --beta 5e-6 --entry-period 0       --label entry_c0_low_b5e-6"
submit eB6S1 "$R/entry_s1_low_b5e-6" 100000 "$LOW --beta 5e-6 --entry-period 4600000 --label entry_s1_low_b5e-6"
submit eB6S2 "$R/entry_s2_low_b5e-6" 100000 "$LOW --beta 5e-6 --entry-period 9200000 --label entry_s2_low_b5e-6"
submit eB6S3 "$R/entry_s3_low_b5e-6" 100000 "$LOW --beta 5e-6 --entry-on-convergence --alone-cap 2000000000 --label entry_s3_low_b5e-6"
# beta = 5e-5 (eps 0.1 at 0.46M, 0.01 at 0.92M)
submit eB5C0 "$R/entry_c0_low_b5e-5" 100000 "$LOW --beta 5e-5 --entry-period 0      --label entry_c0_low_b5e-5"
submit eB5S1 "$R/entry_s1_low_b5e-5" 100000 "$LOW --beta 5e-5 --entry-period 500000 --label entry_s1_low_b5e-5"
submit eB5S2 "$R/entry_s2_low_b5e-5" 100000 "$LOW --beta 5e-5 --entry-period 900000 --label entry_s2_low_b5e-5"
submit eB5S3 "$R/entry_s3_low_b5e-5" 100000 "$LOW --beta 5e-5 --entry-on-convergence --alone-cap 2000000000 --label entry_s3_low_b5e-5"
# menu test (simultaneous start, beta 5e-7, floor 0.01)
submit eM1 "$R/entry_c0_low_alignfine15" 1000000 "--noise-std 0.1 --beta 5e-7 --floor 0.01 --entry-period 0 --action-grid aligned_fine --n-actions 15 --label entry_c0_low_alignfine15"
submit eM2 "$R/entry_c0_low_alignfine43" 1000000 "--noise-std 0.1 --beta 5e-7 --floor 0.01 --entry-period 0 --action-grid aligned_fine --n-actions 43 --label entry_c0_low_alignfine43"
sleep 15
squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
echo "=== MENU TEST vs C0 (wide menu, 0.850) and a2 (Dou menu, 0.385) ==="
"$VPY" -I hpc/read_entry_cells.py "$R/entry_c0_low_b5e-7" "$R/entry_c0_low_dou_f001" "$R/entry_c0_low_alignfine15" "$R/entry_c0_low_alignfine43"
for b in 5e-8 5e-6 5e-5; do
  echo "=== BETA $b: entry stages vs its own control ==="
  "$VPY" -I hpc/read_entry_cells.py "$R/entry_c0_low_b$b" "$R/entry_s1_low_b$b" "$R/entry_s2_low_b$b" "$R/entry_s3_low_b$b"
done
echo DONE
