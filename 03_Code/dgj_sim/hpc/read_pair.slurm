#!/bin/bash
#SBATCH --job-name=pairread
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --account=def-cbravo_cpu
#SBATCH --output=logs/%x_%j.out
# Paired read of two replication-type cells (run_session_cli results), each aggregated with the checkout that
# produced it. Env: A_CO A_DIR B_CO B_DIR OUTF LABEL. Writes OUTF: B - A, 99% CI, 0.10 rule. Read-only on results.
set -uo pipefail
source "$HOME/venv-aitrading/bin/activate"; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
T="${SLURM_TMPDIR:-/tmp}"
dump () {
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
json.dump({int(r["file"][8:12]): [r["delta_c"], r["converged_at"], r.get("mechanism")] for r in per}, open(out, "w"))
PY
  )
}
{
  echo "read at $(date -u --iso-8601=seconds) — $LABEL"
  dump "$A_CO" "$A_DIR" "$T/a.json" || echo "A read failed"
  dump "$B_CO" "$B_DIR" "$T/b.json" || echo "B read failed"
  python -I - "$T/a.json" "$T/b.json" "$(basename "$A_DIR")" "$(basename "$B_DIR")" <<'PY'
import json, math, statistics as st, sys
a = {int(k): v for k, v in json.load(open(sys.argv[1])).items()}
b = {int(k): v for k, v in json.load(open(sys.argv[2])).items()}
for name, d in ((sys.argv[3], a), (sys.argv[4], b)):
    x = [v[0] for v in d.values()]; tc = [v[1] / 1e9 for v in d.values()]
    print(f"{name}: n={len(x)} mean dC {st.mean(x):.3f} SE {st.stdev(x)/math.sqrt(len(x)):.3f}  median T_c {st.median(tc):.2f}B max {max(tc):.2f}B")
    mech = [v[2] for v in d.values() if len(v) > 2 and v[2]]
    if mech:
        from collections import Counter
        print(f"   IRF mechanisms: {dict(Counter(mech))}")
common = sorted(set(a) & set(b))
if len(common) > 1:
    diff = [b[k][0] - a[k][0] for k in common]
    m, se = st.mean(diff), st.stdev(diff) / math.sqrt(len(diff))
    lo, hi = m - 2.576 * se, m + 2.576 * se
    verdict = "POINTS TO A CAUSE" if (m >= 0.10 and lo > 0) else ("RULED OUT" if lo <= 0 <= hi else "differs, below 0.10")
    print(f"{sys.argv[4]} - {sys.argv[3]}: n={len(common)} diff {m:+.3f} SE {se:.3f} 99% CI [{lo:+.3f}, {hi:+.3f}] -> {verdict}")
PY
  echo DONE-READ
} > "$OUTF.tmp" 2>&1
mv "$OUTF.tmp" "$OUTF"
