"""Run N_sim independent sessions of one cell and aggregate. / 一个实验单元的全部 session。

    py -3 -m dgj.experiments.run_cell --sessions 8 --workers 4 --max-periods 20000000 --out outputs/baseline

Each session writes ``session_<k>.npz`` (rows + manifest) so a crash loses one
session, not the cell, and metrics can be recomputed without re-simulating.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os

import numpy as np

from dgj.config import ExperimentCell, PaperParameters
from dgj.game.session import Session, build_grids
from dgj.metrics import collusion, market_quality, trading_policy


def run_session(cell: ExperimentCell, session_index: int, experiment_seed: int, out_dir: str,
                max_periods: int | None, chunk_size: int) -> str:
    path = os.path.join(out_dir, f"session_{session_index:04d}.npz")
    if os.path.exists(path):
        return path
    session = Session(cell, session_index, experiment_seed)
    result = session.run(chunk_size=chunk_size, max_periods=max_periods)
    np.savez_compressed(
        path,
        rows=result.measurement_rows,
        converged_at=result.converged_at,
        manifest=json.dumps(result.manifest),
    )
    return path


def run_cell(cell: ExperimentCell, n_sessions: int, experiment_seed: int, out_dir: str, *,
             workers: int = 1, max_periods: int | None = None, chunk_size: int = 1_000_000) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cell.json"), "w", encoding="utf-8") as f:
        json.dump({"cell": cell.to_dict(), "cell_key": cell.key(), "experiment_seed": experiment_seed}, f, indent=2)
    args = [(cell, k, experiment_seed, out_dir, max_periods, chunk_size) for k in range(n_sessions)]
    if workers <= 1:
        return [run_session(*a) for a in args]
    paths = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_session, *a) for a in args]
        for fut in as_completed(futures):
            paths.append(fut.result())
    return sorted(paths)


def aggregate(cell: ExperimentCell, out_dir: str) -> dict:
    """Per-session metrics and cell-level summary (mean, 1st/99th percentile)."""
    grids = build_grids(cell)
    p = cell.parameters
    per_session = []
    for name in sorted(os.listdir(out_dir)):
        if not name.startswith("session_") or not name.endswith(".npz"):
            continue
        data = np.load(os.path.join(out_dir, name))
        rows = data["rows"]
        c = collusion.compute(rows, grids, p.num_speculators, p.value_mean)
        t = trading_policy.compute(rows, p.num_speculators, grids.discrete_value_std, p.noise_std)
        liq = market_quality.liquidity(rows, p.investor_slope)
        mis = market_quality.mispricing(rows, p.num_speculators, t.average_intensity, p.value_mean)
        manifest = json.loads(str(data["manifest"])) if "manifest" in data else {}
        per_session.append({
            "file": name,
            "mechanism": manifest.get("irf", {}).get("mechanism"),
            "liquidity": liq.mean_liquidity,
            "mispricing": mis.mean_mispricing_absolute,
            "converged_at": int(data["converged_at"]),
            "delta_c": c.delta_c,
            "profit_gain_vs_nash": c.profit_gain_vs_nash,
            "chi_hat": t.average_intensity,
            "price_informativeness": t.price_informativeness,
            "mean_lambda_hat": float(rows[:, 4].mean()),
        })
    if not per_session:
        return {"sessions": 0}
    deltas = np.array([s["delta_c"] for s in per_session])
    summary = {
        "sessions": len(per_session),
        "delta_c_mean": float(deltas.mean()),
        "delta_c_p01": float(np.percentile(deltas, 1)),
        "delta_c_p99": float(np.percentile(deltas, 99)),
        "chi_hat_mean": float(np.mean([s["chi_hat"] for s in per_session])),
        "chi_nash": grids.nash.intensity,
        "chi_cartel": grids.cartel.intensity,
        "informativeness_mean": float(np.mean([s["price_informativeness"] for s in per_session])),
        "converged_at_median": float(np.median([s["converged_at"] for s in per_session])),
        "liquidity_mean": float(np.mean([s["liquidity"] for s in per_session])),
        "mispricing_mean": float(np.mean([s["mispricing"] for s in per_session])),
        "mechanism_shares": {m: sum(1 for s in per_session if s["mechanism"] == m) / len(per_session)
                             for m in ("price_trigger", "over_pruning", "unclassified")},
        "per_session": per_session,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--max-periods", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--convergence-periods", type=int, default=None)
    parser.add_argument("--measurement-periods", type=int, default=None)
    parser.add_argument("--out", type=str, default="outputs/cell")
    a = parser.parse_args()
    changes = {"noise_std": a.noise_std}
    if a.convergence_periods:
        changes["convergence_periods"] = a.convergence_periods
    if a.measurement_periods:
        changes["measurement_periods"] = a.measurement_periods
    cell = ExperimentCell(parameters=PaperParameters(**changes), label=f"noise_{a.noise_std}")
    paths = run_cell(cell, a.sessions, a.seed, a.out, workers=a.workers, max_periods=a.max_periods, chunk_size=a.chunk_size)
    summary = aggregate(cell, a.out)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_session"}, indent=2))
    print(f"{len(paths)} sessions written to {a.out}")


if __name__ == "__main__":
    main()
