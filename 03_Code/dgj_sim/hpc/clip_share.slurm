#!/bin/bash
#SBATCH --job-name=clipshare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --account=def-cbravo_cpu
#SBATCH --output=logs/%x_%j.out
# Read-only price-clipping diagnostic (#3): runs hpc/clip_share.py from the x0-beta checkout. Writes $R/clip_share.txt.
set -uo pipefail
source "$HOME/venv-aitrading/bin/activate"; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R="$HOME/scratch/ai-trading-runs"
cd "$HOME/scratch/AI-agent-trading-beta/03_Code/dgj_sim"
{ echo "clip_share at $(date -u --iso-8601=seconds), commit $(git rev-parse --short HEAD)"
  for d in x0_pv x0_iv x3a_cartel; do python -c "import json;print('$d cell.json price_grid =', json.load(open('$R/$d/cell.json'))['cell'].get('price_grid','MISSING'))"; done
  PYTHONPATH=. python hpc/clip_share.py "$R/x0_pv" "$R/x0_iv" "$R/x3a_cartel"
  echo DONE-CLIP; } > "$R/clip_share.txt.tmp" 2>&1
mv "$R/clip_share.txt.tmp" "$R/clip_share.txt"
