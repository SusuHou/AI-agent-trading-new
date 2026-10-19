#!/bin/bash
# Read-only (2026-10-19): (1) entry measure-first run, (2) midterm verdicts under the 0.10 rule.
set -uo pipefail
source "$HOME/venv-aitrading/bin/activate"
RUNS="$HOME/scratch/ai-trading-runs"
echo "=== QUEUE ==="
squeue -u "$USER" -h -o "%i %T" | awk '{split($1,a,"_"); print a[1], $2}' | sort | uniq -c

echo "=== (1) ENTRY LONE INCUMBENTS ==="
python - "$RUNS/entry_s3_low_b5e-7" <<'PY'
import glob, json, os, sys, numpy as np
d = sys.argv[1]
mult = None
done, tc, capped, running, modes, means = 0, [], 0, 0, [], []
for p in sorted(glob.glob(os.path.join(d, "progress_*.json"))):
    j = json.load(open(p)); mult = np.array(j["multipliers"])
    if j["status"] == "alone_done":
        done += 1
        if j["alone_capped"]: capped += 1
        else: tc.append(j["alone_converged_at"])
        ck = p.replace("progress_", "ckpt_").replace(".json", ".npz")
        if os.path.exists(ck):
            pol = np.load(ck)["policy"][0]
            idx = np.array([(int(m) & -int(m)).bit_length() - 1 for m in pol])
            modes.append(int(np.bincount(idx, minlength=15).argmax())); means.append(float(mult[idx].mean()))
    else:
        running += 1
print(f"alone_done={done} (converged {len(tc)}, capped {capped}); other progress files={running}")
if tc:
    tc = np.array(tc) / 1e6
    print(f"alone convergence (M periods): min {tc.min():.0f}  p25 {np.percentile(tc,25):.0f}  median {np.median(tc):.0f}  p75 {np.percentile(tc,75):.0f}  p90 {np.percentile(tc,90):.0f}  max {tc.max():.0f}")
if modes:
    print("menu (chi):", np.round(mult, 1).tolist(), " cartel=idx1 Nash=idx5 monopolist=idx13")
    print("modal greedy action of incumbent, count by index:", np.bincount(modes, minlength=15).tolist())
    m = np.array(means); print(f"mean greedy chi over states: median {np.median(m):.1f}  min {m.min():.1f}  max {m.max():.1f}")
PY

echo "=== (2) MIDTERM VERDICTS (0.10 rule; paired; Bonferroni over 5 => 99% CI) ==="
cd "$HOME/scratch/AI-agent-trading-new/03_Code/dgj_sim" && git rev-parse --short HEAD
python - "$RUNS" <<'PY'
import json, os, sys, numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "hpc"))
from dgj.config import ExperimentCell, PaperParameters
from dgj.experiments.run_cell import aggregate
from collections import Counter
runs = sys.argv[1]; res = {}
for d in ("x0_pv", "x3a_cartel", "x4_irf_pv"):
    path = os.path.join(runs, d)
    c = json.load(open(os.path.join(path, "cell.json")))["cell"]
    cell = ExperimentCell(parameters=PaperParameters(**c["parameters"]), label=c["label"], prehistory=c["prehistory"],
                          price_mapping=c["price_mapping"], price_grid=c["price_grid"],
                          training_tie_rule=c["training_tie_rule"], measurement_tie_rule=c["measurement_tie_rule"])
    per = aggregate(cell, path)["per_session"]
    dc = {int(r["file"][8:12]): r["delta_c"] for r in per}
    v = np.array(list(dc.values())); tc = np.array([r["converged_at"] for r in per], float)
    print(f"{d}: n={len(v)} mean dC={v.mean():.3f} sd={v.std(ddof=1):.3f} median T_c={np.median(tc)/1e9:.2f}B max T_c={tc.max()/1e9:.2f}B")
    mech = [r.get("mechanism") for r in per if r.get("mechanism")]
    if mech: print("   IRF mechanisms:", dict(Counter(mech)))
    res[d] = dc
def paired(a, b):
    common = sorted(set(res[a]) & set(res[b]))
    diff = np.array([res[b][i] - res[a][i] for i in common])
    m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
    lo, hi = m - 2.576 * se, m + 2.576 * se
    verdict = "POINTS TO A CAUSE" if (m >= 0.10 and lo > 0) else ("RULED OUT" if lo <= 0 <= hi else "differs, below 0.10")
    print(f"{b} - {a}: n={len(diff)} mean {m:+.4f} SE {se:.4f} 99% CI [{lo:+.4f}, {hi:+.4f}] max|diff| {np.abs(diff).max():.4f} -> {verdict}")
paired("x0_pv", "x3a_cartel")
paired("x0_pv", "x4_irf_pv")
PY
echo DONE-READ
