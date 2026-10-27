"""Mechanism counts and per-agent impulse responses for entry cells run with --irf-paths (stdlib only).

    python -I hpc/read_irf_counts.py DIR [DIR ...]
"""
import glob, json, os, statistics as st, sys
from collections import Counter

for d in sys.argv[1:]:
    rows = []
    for p in glob.glob(os.path.join(d, "session_*.json")):
        j = json.load(open(p))
        if j.get("status") == "complete" and j.get("irf"):
            rows.append(j["irf"])
    name = os.path.basename(d.rstrip("/"))
    if not rows:
        print(f"=== {name}: no IRF results yet")
        continue
    c = Counter(r["mechanism"] for r in rows)
    rc = [r["response_vs_control"] for r in rows]
    print(f"=== {name}: n={len(rows)} mechanisms {dict(c)}")
    print("  mean response vs control: incumbent {:+.2e} entrant {:+.2e}".format(
        st.mean(x[0] for x in rc), st.mean(x[1] for x in rc)))
