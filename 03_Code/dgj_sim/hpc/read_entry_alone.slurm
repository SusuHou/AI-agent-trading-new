#!/bin/bash
#SBATCH --job-name=entryread
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --account=def-cbravo_cpu
#SBATCH --output=logs/%x_%j.out
# Tests + read of the lone-incumbent runs; writes $RUNS/entry_alone_read.txt (read-only on results).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$HOME/venv-aitrading/bin/activate"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH="$PWD"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_cache_read_$SLURM_JOB_ID"
RUNS="$HOME/scratch/ai-trading-runs"
OUTF="$RUNS/entry_alone_read.txt"
{
  echo "read at $(date -u --iso-8601=seconds), commit $(git rev-parse --short HEAD)"
  python -m unittest tests.test_entry 2>&1 | tail -2
  python hpc/read_entry_alone.py "$RUNS/entry_s3_low_b5e-7" "$RUNS/entry_s3_low_b5e-7_T1" "$RUNS/entry_s3_low_b5e-7_T2" 2>&1 | grep -v "^OpenBLAS"
  echo DONE-READ
} > "$OUTF.tmp" 2>&1
mv "$OUTF.tmp" "$OUTF"
