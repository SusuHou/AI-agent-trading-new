"""Summarize entry-timing cells: Delta^C (mean and per agent), paired differences vs the control.

    python hpc/read_entry_cells.py CONTROL_DIR CELL_DIR [CELL_DIR ...]

Rule written before the runs (D5): the prior (late entry lowers collusion) is refuted if late-entry
Delta^C is within 0.10 of the control or higher. Paired over common sessions; 99% CI shown.
"""
import glob, json, os, sys
import numpy as np


def load(d):
    out = {}
    for p in glob.glob(os.path.join(d, "session_*.json")):
        j = json.load(open(p))
        if j.get("status") == "complete":
            out[int(os.path.basename(p)[8:12])] = j
    return out


cells = [(os.path.basename(d.rstrip("/")), load(d)) for d in sys.argv[1:]]
ctrl_name, ctrl = cells[0]
for name, res in cells:
    if not res:
        print(f"=== {name}: 0 complete"); continue
    dc = np.array([r["delta_c"] for r in res.values()])
    a0 = np.array([r["delta_by_agent"][0] for r in res.values()])
    a1 = np.array([r["delta_by_agent"][1] for r in res.values()])
    tc = np.array([r["converged_at"] for r in res.values()], float) / 1e9
    ent = np.array([r["entry_period"] for r in res.values()], float) / 1e6
    chi = np.array([r["mean_order_multiplier_by_agent"] for r in res.values()])
    print(f"=== {name}: complete {len(res)}  entry (M) median {np.median(ent):.0f}  joint convergence (B) median {np.median(tc):.2f} max {tc.max():.2f}")
    print(f"  dC mean {dc.mean():.3f} sd {dc.std(ddof=1) if len(dc) > 1 else 0:.3f} | incumbent {a0.mean():.3f} | entrant {a1.mean():.3f}"
          f" | traded chi incumbent {chi[:, 0].mean():.1f} entrant {chi[:, 1].mean():.1f}")
    if name != ctrl_name:
        common = sorted(set(res) & set(ctrl))
        if len(common) > 1:
            diff = np.array([res[k]["delta_c"] - ctrl[k]["delta_c"] for k in common])
            m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
            print(f"  vs {ctrl_name}: n={len(common)} diff {m:+.3f} SE {se:.3f} 99% CI [{m-2.576*se:+.3f}, {m+2.576*se:+.3f}]"
                  f" -> prior {'REFUTED' if m > -0.10 else 'not refuted'} (rule: refuted if diff > -0.10)")
