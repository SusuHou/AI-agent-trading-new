#!/bin/bash
# 2026-10-26, author (via the writing session): mechanism tests M1-M3, all now.
# M1 impulse responses after entry: C0/S1/S2/S3 low noise as the core cells + --irf-paths 10000 (new dirs *_irf;
#    branch mechanism-tests, NEW checkout ~/scratch/AI-agent-trading-mech).
# M2 high-noise baseline + memory test (Dou menu, run_session_cli, x0-beta checkout): X0-high (IRF 10000), NC-high.
# M3 take-turns exploration (Lambin), simultaneous start, low (IRF 10000) and high noise (joint cap 5e9).
# Narval caps queued jobs per user: sessions go in batches of 25, in priority order M1 -> M2 -> M3; at the first
# refusal the rest wait for the next run. Safe to rerun (submits only missing, not-queued sessions).
set -uo pipefail
source "$HOME/venv-aitrading/bin/activate"; export PYTHONNOUSERSITE=1
VPY="$HOME/venv-aitrading/bin/python"
R="$HOME/scratch/ai-trading-runs"
MECH="$HOME/scratch/AI-agent-trading-mech"; BETA="$HOME/scratch/AI-agent-trading-beta"
ENTRY="$HOME/scratch/AI-agent-trading-entry/03_Code/dgj_sim"
[[ -d "$MECH" ]] || git clone -q --branch mechanism-tests https://github.com/SusuHou/AI-agent-trading-new.git "$MECH"
( cd "$MECH" && git fetch -q origin && git checkout -q mechanism-tests && git reset -q --hard origin/mechanism-tests )
( cd "$ENTRY" && git pull -q )
echo "mechanism checkout at $(cd "$MECH" && git rev-parse --short HEAD); x0-beta checkout at $(cd "$BETA" && git rev-parse --short HEAD)"
mkdir -p "$MECH/03_Code/dgj_sim/logs" "$BETA/03_Code/dgj_sim/logs"
LIMIT=0

missing () {  # $1 job name, $2 dir, $3 result extension (json|npz) -> space-separated session ids
  local Q
  Q=$(squeue -u "$USER" -h -n "$1" -r -o %K | paste -sd, -)
  "$VPY" -I - "$2" "$Q" "$3" <<'PY'
import os, sys
d, q, ext = sys.argv[1], {int(x) for x in sys.argv[2].split(",") if x.strip().isdigit()}, sys.argv[3]
print(" ".join(str(k) for k in range(100) if k not in q and not os.path.exists(os.path.join(d, f"session_{k:04d}.{ext}"))))
PY
}

submit_batches () {  # $1 name, $2 dir, $3 workdir, $4 slurm script, $5 seed, $6 cell args, $7 ext
  local NAME="$1" D="$2" WD="$3" SCRIPT="$4" SEED_="$5" ARGS="$6" EXT="$7" IDS B J n=0
  mkdir -p "$D"
  if [[ $LIMIT == 1 ]]; then echo "$NAME: waiting (queue limit reached earlier in this run)"; return; fi
  IDS=$(missing "$NAME" "$D" "$EXT")
  if [[ -z "$IDS" ]]; then echo "$NAME: nothing to submit"; return; fi
  set -- $IDS
  while [[ $# -gt 0 ]]; do
    B=$(echo "${@:1:25}" | tr ' ' ',')
    if [[ $# -gt 25 ]]; then shift 25; else shift $#; fi
    J=$(cd "$WD" && OUT="$D" SEED="$SEED_" WORK_PERIODS=20000000000 CELL_ARGS="$ARGS" \
        sbatch --parsable --array="$B" --time=08:00:00 --job-name="$NAME" "$SCRIPT" 2>&1 | grep -v NOTE)
    if [[ "$J" =~ ^[0-9]+$ ]]; then
      n=$(( n + $(echo "$B" | tr ',' ' ' | wc -w) ))
      { echo "submitted_utc: $(date -u --iso-8601=seconds)"; echo "job_id: $J"; echo "sessions: $B"; } >> "$D/submission_receipt.txt"
      echo "$NAME: batch job $J ($B)"
    else
      echo "$NAME: refused ($J) -> the rest waits for the next run"
      LIMIT=1
      break
    fi
  done
  echo "$NAME: submitted $n sessions this run"
}

MW="$MECH/03_Code/dgj_sim"; BW="$BETA/03_Code/dgj_sim"
LOWE="--noise-std 0.1 --beta 5e-7 --action-grid wide --floor 0.01"
X0H="--noise-std 100 --speculators 2 --rho 0.95 --xi 500 --convergence-periods 1000000 --measurement-periods 100000 --price-grid per_value --prehistory nash"
# M1: same cells as the core low-noise entry cells (same labels -> same keys), plus impulse responses
submit_batches mC0irf "$R/entry_c0_low_b5e-7_irf" "$MW" hpc/submit_entry.slurm 20261019 "$LOWE --entry-period 0 --irf-paths 10000 --label entry_c0_low" json
submit_batches mS1irf "$R/entry_s1_low_b5e-7_irf" "$MW" hpc/submit_entry.slurm 20261019 "$LOWE --entry-period 46000000 --irf-paths 10000 --label entry_s1_low" json
submit_batches mS2irf "$R/entry_s2_low_b5e-7_irf" "$MW" hpc/submit_entry.slurm 20261019 "$LOWE --entry-period 92000000 --irf-paths 10000 --label entry_s2_low" json
submit_batches mS3irf "$R/entry_s3_low_b5e-7_irf" "$MW" hpc/submit_entry.slurm 20261019 "$LOWE --entry-on-convergence --alone-cap 2000000000 --irf-paths 10000 --label entry_s3_low" json
# M2: run_session_cli via submit_array.slurm (venv and PYTHONNOUSERSITE are exported to the job)
submit_batches x0high "$R/x0_high" "$BW" hpc/submit_array.slurm 20261017 "$X0H --label x0_high --irf-paths 10000" npz
submit_batches nchigh "$R/nc_high" "$BW" hpc/submit_array.slurm 20261017 "$X0H --no-lagged-price --label nc_high --irf-paths 0" npz
# M3: take-turns exploration, simultaneous start
submit_batches mTurnL "$R/entry_c0_low_turns"  "$MW" hpc/submit_entry.slurm 20261019 "$LOWE --entry-period 0 --explore-turns --irf-paths 10000 --label entry_c0_low_turns" json
submit_batches mTurnH "$R/entry_c0_high_turns" "$MW" hpc/submit_entry.slurm 20261019 "--noise-std 100 --beta 5e-7 --action-grid wide --floor 0.01 --joint-cap 5000000000 --entry-period 0 --explore-turns --label entry_c0_high_turns" json
sleep 10
echo "=== queue (tasks per job name) ==="
squeue -u "$USER" -h -r -o "%j %T" | sort | uniq -c
echo "=== M1: dC vs the core cells (same cells and seeds: should reproduce) ==="
for c in c0 s1 s2 s3; do "$VPY" -I "$ENTRY/hpc/read_entry_cells.py" "$R/entry_${c}_low_b5e-7" "$R/entry_${c}_low_b5e-7_irf"; done
echo "=== M1 and M3-low: mechanisms ==="
"$VPY" -I "$ENTRY/hpc/read_irf_counts.py" "$R/entry_c0_low_b5e-7_irf" "$R/entry_s1_low_b5e-7_irf" "$R/entry_s2_low_b5e-7_irf" "$R/entry_s3_low_b5e-7_irf" "$R/entry_c0_low_turns"
echo "=== M3: take-turns vs simultaneous controls ==="
"$VPY" -I "$ENTRY/hpc/read_entry_cells.py" "$R/entry_c0_low_b5e-7" "$R/entry_c0_low_turns"
"$VPY" -I "$ENTRY/hpc/read_entry_cells.py" "$R/entry_c0_high_b5e-7" "$R/entry_c0_high_turns"
echo "=== M2: paired read job (after the X0-high / NC-high arrays) ==="
for j in $(squeue -u "$USER" -h -n pairread -o %i); do scancel "$j"; done
DEP=$(squeue -u "$USER" -h -n x0high,nchigh -o %A | sort -u | paste -sd: -)
RJ=$(cd "$ENTRY" && A_CO="$BETA" A_DIR="$R/x0_high" B_CO="$BETA" B_DIR="$R/nc_high" OUTF="$R/x0_high_nc_read.txt" LABEL="NC-high vs X0-high" \
     sbatch --parsable ${DEP:+--dependency=afterany:$DEP} hpc/read_pair.slurm 2>&1 | grep -v NOTE)
echo "M2 read job $RJ -> $R/x0_high_nc_read.txt"
cat "$R/x0_high_nc_read.txt" 2>/dev/null || echo "no M2 read yet"
echo DONE
