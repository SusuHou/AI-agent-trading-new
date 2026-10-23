#!/bin/bash
# 2026-10-22, author approved "c，两个都做" (via the writing session):
# (a) decomposition of the C0 jump (0.404 replication -> 0.850): simultaneous start, low noise, paired with C0 (seed 20261019)
#     a0 Dou menu + no floor | a1 wide menu + no floor | a2 Dou menu + floor 0.01   (C0 = wide + floor 0.01)
# (b) high noise (sigma_u = 100): C0, S1 (46M), S2 (92M), S3 (on incumbent convergence, alone cap 2e9);
#     joint training capped 5e9 periods after entry (capped sessions measured frozen and flagged; Atanasova B.2).
# Submits only sessions without a result and not already queued; safe to rerun (resumes paused sessions).
set -euo pipefail
source "$HOME/venv-aitrading/bin/activate"
VPY="$HOME/venv-aitrading/bin/python"
cd "$HOME/scratch/AI-agent-trading-entry" && git fetch -q origin && git reset -q --hard origin/entry-timing
HEAD=$(git rev-parse --short HEAD); echo "entry checkout at $HEAD"
cd 03_Code/dgj_sim && mkdir -p logs
R="$HOME/scratch/ai-trading-runs"
LOW="--noise-std 0.1 --beta 5e-7 --entry-period 0"
HIGH="--noise-std 100 --beta 5e-7 --action-grid wide --floor 0.01 --joint-cap 5000000000"
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
  J=$(OUT="$D" SEED=20261019 WORK_PERIODS=20000000000 CELL_ARGS="$ARGS" \
      sbatch --parsable --array="$MISSING" --time=08:00:00 --job-name="$NAME" hpc/submit_entry.slurm)
  { echo "submitted_utc: $(date -u --iso-8601=seconds)"; echo "job_id: $J"; echo "commit: $HEAD"; echo "sessions: $MISSING"; } >> "$D/submission_receipt.txt"
  echo "$NAME: submitted $N sessions as job $J"
}
submit entryLa0 "$R/entry_c0_low_dou_f0"   "$LOW --action-grid dou  --floor 0    --label entry_c0_low_dou_f0"
submit entryLa1 "$R/entry_c0_low_wide_f0"  "$LOW --action-grid wide --floor 0    --label entry_c0_low_wide_f0"
submit entryLa2 "$R/entry_c0_low_dou_f001" "$LOW --action-grid dou  --floor 0.01 --label entry_c0_low_dou_f001"
submit entryHC0 "$R/entry_c0_high_b5e-7" "$HIGH --entry-period 0        --label entry_c0_high"
submit entryHS1 "$R/entry_s1_high_b5e-7" "$HIGH --entry-period 46000000 --label entry_s1_high"
submit entryHS2 "$R/entry_s2_high_b5e-7" "$HIGH --entry-period 92000000 --label entry_s2_high"
submit entryHS3 "$R/entry_s3_high_b5e-7" "$HIGH --entry-on-convergence --alone-cap 2000000000 --label entry_s3_high"
sleep 15
squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
echo "=== (a) DECOMPOSITION vs C0 (wide + floor) ==="
"$VPY" -I hpc/read_entry_cells.py "$R/entry_c0_low_b5e-7" "$R/entry_c0_low_dou_f0" "$R/entry_c0_low_wide_f0" "$R/entry_c0_low_dou_f001"
echo "=== (b) HIGH NOISE vs high-noise C0 ==="
"$VPY" -I hpc/read_entry_cells.py "$R/entry_c0_high_b5e-7" "$R/entry_s1_high_b5e-7" "$R/entry_s2_high_b5e-7" "$R/entry_s3_high_b5e-7"
echo "=== speed (periods/s) from running sessions' progress files ==="
cat "$R"/entry_*high*/progress_*.json 2>/dev/null | grep -o '"rate_periods_per_s": [0-9.]*' | awk '{s+=$2; n++} END {if (n) printf "%d files, mean %.0f periods/s\n", n, s/n; else print "none yet"}'
echo DONE
