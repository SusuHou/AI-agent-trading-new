"""Run ONE session as a standalone job (HPC array task). / 单个 session 的作业入口。

    py -3 -m dgj.experiments.run_session_cli --out outputs/low_noise --session 17 \
        --noise-std 0.1 --price-grid per_value --max-periods 5000000000 --checkpoint-every 20

* writes  <out>/cell.json           (once, describes the cell)
          <out>/ckpt_<k>.npz         (every --checkpoint-every chunks; resumed automatically)
          <out>/session_<k>.npz      (rows + manifest; converged_at = -1 if the cap was hit)
* exit code 0 on success, 3 if the cap was hit without convergence (rows still written), 1 on error.

Resume: rerun the same command; if ckpt_<k>.npz exists it is loaded and training continues
from the saved period with the saved RNG states, so a walltime kill costs at most one
checkpoint interval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from dgj.config import ExperimentCell, PaperParameters
from dgj.game import irf
from dgj.game.session import Session


def build_cell(a: argparse.Namespace) -> ExperimentCell:
    changes = {"noise_std": a.noise_std, "num_speculators": a.speculators, "discount_factor": a.rho,
               "investor_slope": a.xi}
    if a.convergence_periods:
        changes["convergence_periods"] = a.convergence_periods
    if a.measurement_periods:
        changes["measurement_periods"] = a.measurement_periods
    return ExperimentCell(parameters=PaperParameters(**changes), label=a.label,
                          prehistory=a.prehistory, price_grid=a.price_grid)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--label", default="cell")
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--speculators", type=int, default=2)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--xi", type=float, default=500.0)
    parser.add_argument("--prehistory", default="nash", choices=["nash", "cartel"])
    parser.add_argument("--price-grid", default="per_value", choices=["per_value", "global"])
    parser.add_argument("--convergence-periods", type=int, default=None)
    parser.add_argument("--measurement-periods", type=int, default=None)
    parser.add_argument("--max-periods", type=int, default=5_000_000_000)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=20, help="chunks between checkpoints")
    parser.add_argument("--irf-paths", type=int, default=irf.PAPER_PATHS, help="0 disables the IRF experiment")
    parser.add_argument("--no-measure-if-censored", action="store_true",
                        help="do not record measurement rows when the period cap is hit")
    a = parser.parse_args(argv)

    cell = build_cell(a)
    os.makedirs(a.out, exist_ok=True)
    cell_path = os.path.join(a.out, "cell.json")
    if not os.path.exists(cell_path):
        with open(cell_path, "w", encoding="utf-8") as f:
            json.dump({"cell": cell.to_dict(), "cell_key": cell.key(), "experiment_seed": a.seed}, f, indent=2)
    result_path = os.path.join(a.out, f"session_{a.session:04d}.npz")
    ckpt_path = os.path.join(a.out, f"ckpt_{a.session:04d}.npz")
    if os.path.exists(result_path):
        print(f"session {a.session}: result exists, nothing to do")
        return 0

    session = Session(cell, a.session, a.seed)
    if os.path.exists(ckpt_path):
        session.load_checkpoint(ckpt_path)
        print(f"session {a.session}: resumed at {session.periods_completed:,} periods", flush=True)

    t0 = time.perf_counter()
    converged = session.phase == "converged"
    step = a.chunk_size * a.checkpoint_every
    while not converged and session.periods_completed < a.max_periods:
        target = min(a.max_periods, session.periods_completed + step)
        converged = session.train(chunk_size=a.chunk_size, max_periods=target)
        session.save_checkpoint(ckpt_path)
        print(f"session {a.session}: {session.periods_completed/1e6:8.0f}M periods  "
              f"streak={int(session.state.cursor[3]):8d}  {time.perf_counter()-t0:7.0f}s", flush=True)

    censored = not converged
    if censored:
        print(f"session {a.session}: cap {a.max_periods:,} reached without convergence", flush=True)
        if a.no_measure_if_censored:
            return 3
        session.phase = "converged"          # measure the current greedy policy, flagged below
    fork = irf.take_fork(session)            # converged market, before measurement (paper t=0)
    rows = session.measure()
    manifest = session.manifest()
    manifest["censored"] = censored
    extra = {}
    if a.irf_paths > 0:
        baseline = irf.long_run_baseline(rows, cell.parameters.num_speculators, cell.parameters.value_mean)
        result = irf.run_irf(session, fork, baseline, paths=a.irf_paths)
        manifest["irf"] = {
            "paths": result.paths, "shock_magnitude": result.shock_magnitude, "mechanism": result.mechanism,
            "response_vs_long_run": result.response_vs_long_run, "response_vs_control": result.response_vs_control,
            "normalized_price_deviation": result.normalized_price_deviation.tolist(),
        }
        extra = {"irf_control_price": result.control_oriented_price, "irf_treatment_price": result.treatment_oriented_price}
        print(f"session {a.session}: IRF mechanism={result.mechanism} responses={result.response_vs_long_run}", flush=True)
    np.savez_compressed(result_path, rows=rows,
                        converged_at=-1 if censored else session.converged_at,
                        manifest=json.dumps(manifest), **extra)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"session {a.session}: done, converged={not censored}, T_c={session.converged_at}, "
          f"wall={time.perf_counter()-t0:.0f}s", flush=True)
    return 3 if censored else 0


if __name__ == "__main__":
    sys.exit(main())
