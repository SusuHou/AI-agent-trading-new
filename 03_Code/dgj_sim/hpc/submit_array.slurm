#!/bin/bash
#SBATCH --job-name=dgj
#SBATCH --array=0-999%200          # 1,000 sessions, at most 200 running at once
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=08:00:00            # one computing shift, not a scientific session cap
#SBATCH --account=def-cbravo_cpu
#SBATCH --signal=USR1@300           # ask Python to checkpoint five minutes before walltime
#SBATCH --output=logs/%x_%A_%a.out
#
# Usage:  sbatch hpc/submit_array.slurm
# Edit the CELL_ARGS line (or export CELL_ARGS before sbatch) to pick the experiment cell.
# Each array task advances ONE session for a finite work slice. A session file is
# written only after genuine convergence. Rerunning the same array skips completed
# sessions and resumes every unfinished session from ckpt_<k>.npz.
# / 每个 task 只推进一个 session 的一段计算；真正收敛后才写正式结果。

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# ---- environment ------------------------------------------------------------
# module load python/3.12            # adjust to the cluster; needs numpy + numba
# source .venv/bin/activate          # e.g. python -m venv .venv && pip install numpy numba
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/numba_cache_${USER}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="$PWD"

# Verify on the allocated compute node, not only on the login node.
python -c 'from dgj._jit import HAVE_NUMBA; import numpy, numba; print("compute-node NumPy", numpy.__version__, "Numba", numba.__version__, "enabled", HAVE_NUMBA); assert HAVE_NUMBA'

# ---- experiment cell --------------------------------------------------------
OUT="${OUT:-outputs/low_noise_per_value}"
CELL_ARGS="${CELL_ARGS:---noise-std 0.1 --price-grid per_value --prehistory nash --label low_noise}"
WORK_PERIODS="${WORK_PERIODS:-5000000000}"   # additional work in this job only; cumulative session is uncapped
REVIEW_ARGS=()
if [[ "${ALLOW_AFTER_REVIEW:-0}" == "1" ]]; then
    REVIEW_ARGS+=(--allow-after-diagnostic-review)
fi

# Exit 75 means a valid checkpoint was written but the scientific session is
# incomplete. Convert that safe pause to Slurm success; real errors stay nonzero.
set +e
srun python -m dgj.experiments.run_session_cli \
    --out "$OUT" --session "$SLURM_ARRAY_TASK_ID" --seed "${SEED:-20260828}" \
    $CELL_ARGS --work-periods "$WORK_PERIODS" --chunk-size 1000000 --checkpoint-every 50 \
    "${REVIEW_ARGS[@]}"
status=$?
set -e

if [[ "$status" -eq 75 ]]; then
    echo "Safe incomplete pause: checkpoint retained; resubmit this session. / 安全暂停：请续跑。"
    exit 0
fi
exit "$status"
