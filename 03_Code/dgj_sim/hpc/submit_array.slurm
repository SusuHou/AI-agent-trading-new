#!/bin/bash
#SBATCH --job-name=dgj
#SBATCH --array=0-999%200          # 1,000 sessions, at most 200 running at once
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=08:00:00            # per session; the run resumes from its checkpoint if killed
#SBATCH --output=logs/%x_%A_%a.out
#
# Usage:  sbatch hpc/submit_array.slurm
# Edit the CELL_ARGS line (or export CELL_ARGS before sbatch) to pick the experiment cell.
# Each array task runs ONE session and writes <OUT>/session_<k>.npz. Rerunning the same
# sbatch after a timeout resumes every unfinished session from its ckpt_<k>.npz.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# ---- environment ------------------------------------------------------------
# module load python/3.12            # adjust to the cluster; needs numpy + numba
# source .venv/bin/activate          # e.g. python -m venv .venv && pip install numpy numba
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${USER}_${SLURM_ARRAY_JOB_ID}"   # node-local, avoids shared-FS cache races
export PYTHONPATH="$PWD"

# ---- experiment cell --------------------------------------------------------
OUT="${OUT:-outputs/low_noise_per_value}"
CELL_ARGS="${CELL_ARGS:---noise-std 0.1 --price-grid per_value --prehistory nash --label low_noise}"
MAX_PERIODS="${MAX_PERIODS:-5000000000}"     # 5B cap ~ 1.5-3 h at ~1 us/period; report censoring

python -m dgj.experiments.run_session_cli \
    --out "$OUT" --session "$SLURM_ARRAY_TASK_ID" --seed "${SEED:-20260828}" \
    $CELL_ARGS --max-periods "$MAX_PERIODS" --chunk-size 1000000 --checkpoint-every 50
