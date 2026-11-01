"""Summarize best-response tests (C2) per cell and trader (stdlib only).

    python -I hpc/read_br.py DIR [DIR ...]

Per agent (0 = incumbent, 1 = entrant): sessions with a result; share "Nash on path" (learned action optimal
in every visited state); share subgame perfect (optimal in every state); mean share of visited states where
the learned action is optimal (visit-weighted); on-path deviation gain (median, IQR) in %.
"""
import glob, json, os, statistics as st, sys


def q(x, p):
    x = sorted(x)
    k = (len(x) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(x) - 1)
    return x[lo] + (x[hi] - x[lo]) * (k - lo)


for d in sys.argv[1:]:
    rows = []
    for p in glob.glob(os.path.join(d, "session_*.json")):
        j = json.load(open(p))
        if j.get("status") == "complete" and j.get("br"):
            rows.append(j)
    name = os.path.basename(d.rstrip("/"))
    if not rows:
        print(f"=== {name}: no BR results yet")
        continue
    print(f"=== {name}: n={len(rows)}  dC mean {st.mean(r['delta_c'] for r in rows):.3f}")
    for i, who in ((0, "incumbent"), (1, "entrant")):
        a = [r["br"]["agents"][i] for r in rows]
        g = [100 * x["deviation_gain_on_path"] for x in a]
        print(f"  {who:9s}: Nash on path {sum(x['nash_on_path'] for x in a)}/{len(a)}  "
              f"subgame perfect {sum(x['subgame_perfect'] for x in a)}/{len(a)}  "
              f"optimal share (visit-weighted) {st.mean(x['share_optimal_visit_weighted'] for x in a):.3f}  "
              f"deviation gain % median {q(g, .5):.3f} [IQR {q(g, .25):.3f}, {q(g, .75):.3f}]")
