#!/bin/bash
# Best-response (equilibrium) tests, claim C2 (S8 item 5; requested 2026-11-01 via the session "btw").
# Reruns the four core low-noise entry cells (same cells and seeds, so the sessions reproduce exactly) with
# --br-test: after convergence each trader's exact best response to the other's greedy policy is computed
# with the market maker frozen at its converged rule. Branch br-test, NEW checkout ~/scratch/AI-agent-trading-br.
# Batches of 25 (Narval's per-user job cap); safe to rerun (submits only missing, not-queued sessions).
set -uo pipefail
source "$HOME/venv-aitrading/bin/activate"; export PYTHONNOUSERSITE=1
VPY="$HOME/venv-aitrading/bin/python"
R="$HOME/scratch/ai-trading-runs"
CO="$HOME/scratch/AI-agent-trading-br"
[[ -d "$CO" ]] || git clone -q --branch br-test https://github.com/SusuHou/AI-agent-trading-new.git "$CO"
( cd "$CO" && git fetch -q origin && git checkout -q br-test && git reset -q --hard origin/br-test )
W="$CO/03_Code/dgj_sim"; mkdir -p "$W/logs"
echo "br-test checkout at $(cd "$CO" && git rev-parse --short HEAD)"
( cd "$W" && PYTHONPATH=. timeout 300 python -m unittest tests.test_best_response 2>&1 | tail -2 ) || echo "quick test skipped"
LIMIT=0
missing () {
  local Q
  Q=$(squeue -u "$USER" -h -n "$1" -r -o %K | paste -sd, -)
  "$VPY" -I - "$2" "$Q" <<'PY'
import os, sys
d, q = sys.argv[1], {int(x) for x in sys.argv[2].split(",") if x.strip().isdigit()}
print(" ".join(str(k) for k in range(100) if k not in q and not os.path.exists(os.path.join(d, f"session_{k:04d}.json"))))
PY
}
submit_batches () {  # $1 name, $2 dir, $3 cell args
  local NAME="$1" D="$2" ARGS="$3" IDS B J n=0
  mkdir -p "$D"
  if [[ $LIMIT == 1 ]]; then echo "$NAME: waiting (queue limit reached earlier in this run)"; return; fi
  IDS=$(missing "$NAME" "$D")
  if [[ -z "$IDS" ]]; then echo "$NAME: nothing to submit"; return; fi
  set -- $IDS
  while [[ $# -gt 0 ]]; do
    B=$(echo "${@:1:25}" | tr ' ' ',')
    if [[ $# -gt 25 ]]; then shift 25; else shift $#; fi
    J=$(cd "$W" && OUT="$D" SEED=20261019 WORK_PERIODS=20000000000 CELL_ARGS="$ARGS" \
        sbatch --parsable --array="$B" --time=08:00:00 --job-name="$NAME" hpc/submit_entry.slurm 2>&1 | grep -v NOTE)
    if [[ "$J" =~ ^[0-9]+$ ]]; then
      n=$(( n + $(echo "$B" | tr ',' ' ' | wc -w) ))
      { echo "submitted_utc: $(date -u --iso-8601=seconds)"; echo "job_id: $J"; echo "sessions: $B"; } >> "$D/submission_receipt.txt"
      echo "$NAME: batch job $J ($B)"
    else
      echo "$NAME: refused ($J) -> the rest waits for the next run"; LIMIT=1; break
    fi
  done
  echo "$NAME: submitted $n sessions this run"
}
LOWE="--noise-std 0.1 --beta 5e-7 --action-grid wide --floor 0.01 --br-test"
submit_batches brC0 "$R/entry_c0_low_b5e-7_br" "$LOWE --entry-period 0 --label entry_c0_low"
submit_batches brS1 "$R/entry_s1_low_b5e-7_br" "$LOWE --entry-period 46000000 --label entry_s1_low"
submit_batches brS2 "$R/entry_s2_low_b5e-7_br" "$LOWE --entry-period 92000000 --label entry_s2_low"
submit_batches brS3 "$R/entry_s3_low_b5e-7_br" "$LOWE --entry-on-convergence --alone-cap 2000000000 --label entry_s3_low"
sleep 10
echo "=== queue (tasks per job name) ==="; squeue -u "$USER" -h -r -o "%j %T" | sort | uniq -c
echo "=== dC vs the core cells (should reproduce) ==="
for c in c0 s1 s2 s3; do "$VPY" -I "$W/hpc/read_entry_cells.py" "$R/entry_${c}_low_b5e-7" "$R/entry_${c}_low_b5e-7_br"; done
echo "=== best-response tests ==="
"$VPY" -I "$W/hpc/read_br.py" "$R/entry_c0_low_b5e-7_br" "$R/entry_s1_low_b5e-7_br" "$R/entry_s2_low_b5e-7_br" "$R/entry_s3_low_b5e-7_br"
echo DONE
