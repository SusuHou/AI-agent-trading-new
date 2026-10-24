#!/bin/bash
#SBATCH --job-name=x0ivread
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --account=def-cbravo_cpu
#SBATCH --output=logs/%x_%j.out
# X0-iv (interval bins) vs X0-pv: each cell aggregated with the checkout that produced it (fingerprints),
# per-session Delta^C paired on common random numbers. Writes $R/x0_iv_read.txt. Read-only on results.
set -uo pipefail
source "$HOME/venv-aitrading/bin/activate"; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R="$HOME/scratch/ai-trading-runs"; T="$SLURM_TMPDIR"; OUTF="$R/x0_iv_read.txt"
dump () {  # $1 checkout, $2 cell dir, $3 json out
  ( cd "$1/03_Code/dgj_sim" && PYTHONPATH="$PWD:$PWD/hpc" python - "$2" "$3" <<'PY'
import json, os, sys
from dgj.config import ExperimentCell, PaperParameters
from dgj.experiments.run_cell import aggregate
d, out = sys.argv[1], sys.argv[2]
c = json.load(open(os.path.join(d, "cell.json")))["cell"]
kw = dict(parameters=PaperParameters(**c["parameters"]), label=c["label"], prehistory=c["prehistory"],
          price_mapping=c["price_mapping"], price_grid=c["price_grid"],
          training_tie_rule=c["training_tie_rule"], measurement_tie_rule=c["measurement_tie_rule"])
per = aggregate(ExperimentCell(**kw), d)["per_session"]
json.dump({int(r["file"][8:12]): [r["delta_c"], r["converged_at"]] for r in per}, open(out, "w"))
PY
  )
}
{
  echo "read at $(date -u --iso-8601=seconds)"
  dump "$HOME/scratch/AI-agent-trading-new" "$R/x0_pv" "$T/pv.json" || echo "x0_pv read failed"
  dump "$HOME/scratch/AI-agent-trading-iv"  "$R/x0_iv" "$T/iv.json" || echo "x0_iv read failed"
  python -I - "$T/pv.json" "$T/iv.json" <<'PY'
import json, math, statistics as st, sys
pv = {int(k): v for k, v in json.load(open(sys.argv[1])).items()}
iv = {int(k): v for k, v in json.load(open(sys.argv[2])).items()}
for name, d in (("X0-pv (point grid)", pv), ("X0-iv (interval bins)", iv)):
    x = [v[0] for v in d.values()]; tc = [v[1] / 1e9 for v in d.values()]
    print(f"{name}: n={len(x)} mean dC {st.mean(x):.3f} SE {st.stdev(x)/math.sqrt(len(x)):.3f}  median T_c {st.median(tc):.2f}B max {max(tc):.2f}B")
common = sorted(set(pv) & set(iv))
if len(common) > 1:
    diff = [iv[k][0] - pv[k][0] for k in common]
    m, se = st.mean(diff), st.stdev(diff) / math.sqrt(len(diff))
    lo, hi = m - 2.576 * se, m + 2.576 * se
    verdict = "POINTS TO A CAUSE" if (m >= 0.10 and lo > 0) else ("RULED OUT" if lo <= 0 <= hi else "differs, below 0.10")
    print(f"X0-iv - X0-pv: n={len(common)} diff {m:+.3f} SE {se:.3f} 99% CI [{lo:+.3f}, {hi:+.3f}] -> {verdict}")
PY
  echo DONE-READ
} > "$OUTF.tmp" 2>&1
mv "$OUTF.tmp" "$OUTF"
