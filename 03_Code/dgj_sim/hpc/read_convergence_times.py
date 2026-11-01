"""Convergence-time distribution of finished run_session_cli cells (read-only).

    python hpc/read_convergence_times.py DIR [DIR ...]

Reads converged_at from each session_K.npz and reports n, min, p10, median, p90, max (periods), and the
exploration rate Dou's schedule gives at that point, eps = exp(-beta * T_c / n_v) (values are equally likely,
so visits per value ~ T_c / n_v; no floor in these cells). beta and n_v are read from cell.json.
"""
import glob, json, math, os, sys

import numpy as np

for d in sys.argv[1:]:
    c = json.load(open(os.path.join(d, "cell.json")))["cell"]["parameters"]
    beta, n_v = c["exploration_decay"], c["num_value_points"]
    tc = []
    for f in glob.glob(os.path.join(d, "session_*.npz")):
        with np.load(f, allow_pickle=False) as z:
            tc.append(int(z["converged_at"]))
    name = os.path.basename(d.rstrip("/"))
    if not tc:
        print(f"{name}: no finished sessions"); continue
    t = np.array(sorted(tc), float)
    q = {k: float(np.percentile(t, k)) for k in (10, 50, 90)}
    eps = lambda x: math.exp(-beta * x / n_v)
    print(f"{name}: n={len(t)}  T_c min {t[0]/1e6:.1f}M  p10 {q[10]/1e6:.1f}M  median {q[50]/1e6:.1f}M  "
          f"p90 {q[90]/1e6:.1f}M  max {t[-1]/1e6:.1f}M")
    print(f"   eps at T_c (exp(-beta T/n_v), beta={beta:g}): min-T {eps(t[0]):.3g}  p10 {eps(q[10]):.3g}  "
          f"median {eps(q[50]):.3g}  p90 {eps(q[90]):.3g}  max-T {eps(t[-1]):.3g}")
    print(f"   sessions with eps > 0.1 at convergence: {int((np.exp(-beta * t / n_v) > 0.1).sum())}; "
          f"> 0.01: {int((np.exp(-beta * t / n_v) > 0.01).sum())}")
