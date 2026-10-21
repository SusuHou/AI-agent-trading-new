"""Export complete entry-timing sessions as one CSV on stdout (no numpy needed).

    python -I hpc/export_entry_csv.py DIR [DIR ...] > entry_sessions.csv
"""
import csv, glob, json, os, sys

COLS = ["cell", "session", "entry_period", "alone_converged_at", "alone_capped", "converged_at",
        "delta_c", "delta_incumbent", "delta_entrant", "chi_incumbent", "chi_entrant",
        "profit_incumbent", "profit_entrant", "profit_nash", "profit_cartel"]
w = csv.writer(sys.stdout, lineterminator="\n")
w.writerow(COLS)
for d in sys.argv[1:]:
    cell = os.path.basename(d.rstrip("/"))
    for p in sorted(glob.glob(os.path.join(d, "session_*.json"))):
        j = json.load(open(p))
        if j.get("status") != "complete":
            continue
        w.writerow([cell, int(os.path.basename(p)[8:12]), j["entry_period"], j.get("alone_converged_at"),
                    j.get("alone_capped"), j["converged_at"], j["delta_c"], *j["delta_by_agent"],
                    *j["mean_order_multiplier_by_agent"], *j["mean_actual_profit"],
                    j["mean_nash_profit"], j["mean_cartel_profit"]])
