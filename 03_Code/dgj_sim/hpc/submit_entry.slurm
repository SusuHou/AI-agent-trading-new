#!/bin/bash
#SBATCH --job-name=entry
#SBATCH --array=0-99%100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=08:00:00            # one computing shift, not a scientific cap
#SBATCH --account=def-cbravo_cpu
#SBATCH --signal=USR1@300          # checkpoint five minutes before walltime
#SBATCH --output=logs/%x_%A_%a.out
#
# Entry-timing sessions (dgj.experiments.run_entry_cli). Set OUT, SEED and CELL_ARGS before sbatch.
# Exit 75 = safe pause with a checkpoint; resubmitting the same array resumes it.
# / 进入时机实验；退出码 75 表示已安全保存，可续跑。

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/numba_cache_${USER}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="$PWD"
python -c 'from dgj._jit import HAVE_NUMBA; assert HAVE_NUMBA'

: "${OUT:?set OUT}" "${SEED:?set SEED}" "${CELL_ARGS:?set CELL_ARGS}"
WORK_PERIODS="${WORK_PERIODS:-20000000000}"

set +e
srun python -m dgj.experiments.run_entry_cli \
    --out "$OUT" --session "$SLURM_ARRAY_TASK_ID" --seed "$SEED" \
    $CELL_ARGS --work-periods "$WORK_PERIODS" --chunk-size 1000000 --checkpoint-every 200
status=$?
set -e
if [[ "$status" -eq 75 ]]; then
    echo "Safe pause: checkpoint retained; resubmit. / 安全暂停：请续跑。"
    exit 0
fi
exit "$status"
