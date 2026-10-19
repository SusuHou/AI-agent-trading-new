"""Read lone-incumbent (alone-phase) results: convergence time and what the incumbent actually trades.

    python hpc/read_entry_alone.py DIR [DIR ...]

Uses progress_K.json["incumbent_play"] when present; otherwise rebuilds the session from
ckpt_K.npz and computes it (frozen greedy play, 100k periods, on copies of the state).
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dgj.config import ExperimentCell, PaperParameters
from dgj.game.entry_session import EntryCell, EntrySession


def cell_from_json(path):
    j = json.load(open(path)); c = j["entry_cell"]; e = c["entry"]
    base = ExperimentCell(parameters=PaperParameters(**c["parameters"]), label=c["label"], prehistory=c["prehistory"],
                          price_mapping=c["price_mapping"], price_grid=c["price_grid"],
                          training_tie_rule=c["training_tie_rule"], measurement_tie_rule=c["measurement_tie_rule"])
    cell = EntryCell(base=base, action_grid=e["action_grid"], exploration_floor=e["exploration_floor"],
                     entry_period=e["entry_period"], entry_on_convergence=e["entry_on_convergence"],
                     alone_cap=e["alone_cap"], incumbent_init=e.get("incumbent_init", "dou"),
                     lagged_price=e.get("lagged_price", True))
    if cell.key() != j["entry_cell_key"]:
        raise ValueError(f"cell key mismatch in {path}")
    return cell, j["seed"], j["chunk_size"]


for d in sys.argv[1:]:
    cell, seed, chunk = cell_from_json(os.path.join(d, "cell.json"))
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "progress_*.json"))):
        j = json.load(open(p))
        if j["status"] != "alone_done":
            continue
        play = j.get("incumbent_play")
        if play is None:
            k = int(os.path.basename(p)[9:13])
            s = EntrySession(cell, k, seed)
            s.load_entry_checkpoint(p.replace("progress_", "ckpt_").replace(".json", ".npz"), chunk_size=chunk)
            play = s.incumbent_play()
        rows.append((j["alone_converged_at"], j["alone_capped"], play["chi_hat"], play["modal_action"]))
    print(f"=== {os.path.basename(d)}: alone_done {len(rows)}")
    if not rows:
        continue
    tc = np.array([r[0] for r in rows if r[0] is not None], float) / 1e6
    chi = np.array([r[2] for r in rows])
    print(f"  capped {sum(r[1] for r in rows)}; alone convergence (M) median {np.median(tc):.0f} "
          f"[p10 {np.percentile(tc,10):.0f}, p90 {np.percentile(tc,90):.0f}]")
    print(f"  traded chi_hat: mean {chi.mean():.1f} sd {chi.std(ddof=1) if len(chi) > 1 else 0:.1f} "
          f"median {np.median(chi):.1f} [min {chi.min():.1f}, max {chi.max():.1f}]  (cartel 125, Nash 166.7, monopolist 250)")
    print("  modal action counts:", np.bincount([r[3] for r in rows], minlength=15).tolist())
