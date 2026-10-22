"""Summarize entry-timing cells: Delta^C (mean and per agent), paired differences vs the control.

    python hpc/read_entry_cells.py CONTROL_DIR CELL_DIR [CELL_DIR ...]

Standard library only (runs on the login node without numpy).
Rule written before the runs (D5): the prior (late entry lowers collusion) is refuted if late-entry
Delta^C is within 0.10 of the control or higher. Paired over common sessions; 99% CI shown.
"""
import glob, json, math, os, statistics as st, sys


def load(d):
    out = {}
    for p in glob.glob(os.path.join(d, "session_*.json")):
        j = json.load(open(p))
        if j.get("status") == "complete":
            out[int(os.path.basename(p)[8:12])] = j
    return out


def sd(x):
    return st.stdev(x) if len(x) > 1 else 0.0


cells = [(os.path.basename(d.rstrip("/")), load(d)) for d in sys.argv[1:]]
ctrl_name, ctrl = cells[0]
for name, res in cells:
    if not res:
        print(f"=== {name}: 0 complete"); continue
    v = list(res.values())
    dc = [r["delta_c"] for r in v]
    a0 = [r["delta_by_agent"][0] for r in v]
    a1 = [r["delta_by_agent"][1] for r in v]
    tc = [r["converged_at"] / 1e9 for r in v]
    ent = [r["entry_period"] / 1e6 for r in v]
    chi0 = [r["mean_order_multiplier_by_agent"][0] for r in v]
    chi1 = [r["mean_order_multiplier_by_agent"][1] for r in v]
    print(f"=== {name}: complete {len(v)}  entry (M) median {st.median(ent):.0f}  "
          f"joint convergence (B) median {st.median(tc):.2f} max {max(tc):.2f}")
    print(f"  dC mean {st.mean(dc):.3f} sd {sd(dc):.3f} | incumbent {st.mean(a0):.3f} | entrant {st.mean(a1):.3f}"
          f" | traded chi incumbent {st.mean(chi0):.1f} entrant {st.mean(chi1):.1f}")
    if name != ctrl_name:
        common = sorted(set(res) & set(ctrl))
        if len(common) > 1:
            diff = [res[k]["delta_c"] - ctrl[k]["delta_c"] for k in common]
            m, se = st.mean(diff), sd(diff) / math.sqrt(len(diff))
            print(f"  vs {ctrl_name}: n={len(common)} diff {m:+.3f} SE {se:.3f} 99% CI [{m-2.576*se:+.3f}, {m+2.576*se:+.3f}]"
                  f" -> prior {'REFUTED' if m > -0.10 else 'not refuted'} (rule: refuted if diff > -0.10)")
