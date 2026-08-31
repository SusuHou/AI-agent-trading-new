"""Legacy local/debug runner; not a formal HPC experiment entry point.

/ 这是旧的本地调试 runner，不允许用于正式超算实验。

    py -3 -m dgj.experiments.run_cell --debug-only --sessions 8 --workers 4 \
        --max-periods 20000000 --out outputs/baseline

Each session writes ``session_<k>.npz`` (rows + manifest) so a crash loses one
session, not the cell, and metrics can be recomputed without re-simulating.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
import sys

import numpy as np

from dgj.config import ExperimentCell, PaperParameters
from dgj.game import protocol
from dgj.game.session import Session, build_grids
from dgj.metrics import collusion, market_quality, trading_policy
from dgj.provenance import analysis_identity, scientific_identity


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


def aggregate(cell: ExperimentCell, out_dir: str, *, progress: bool = False) -> dict:
    """Aggregate only genuinely converged, identity-matched session artifacts.

    / 只汇总真正收敛且身份匹配的 session；遇到旧的 censored 结果会直接停止，
    不再把未完成策略混进 Delta C。
    """
    grids = build_grids(cell)
    p = cell.parameters
    expected_seed = None
    expected_chunk_size = None
    identity_path = os.path.join(out_dir, "cell.json")
    if os.path.exists(identity_path):
        with open(identity_path, encoding="utf-8") as handle:
            identity_document = json.load(handle)
        expected_seed = int(identity_document["experiment_seed"])
        expected_chunk_size = identity_document.get("training_chunk_size")
    per_session = []
    session_names = sorted(
        name for name in os.listdir(out_dir)
        if name.startswith("session_") and name.endswith(".npz")
    )
    for position, name in enumerate(session_names, start=1):
        path = os.path.join(out_dir, name)
        with np.load(path, allow_pickle=False) as data:
            rows = np.array(data["rows"], copy=True)
            converged_at = int(data["converged_at"].item())
            manifest = json.loads(str(data["manifest"].item())) if "manifest" in data else {}
            result_schema = (
                int(data["result_schema_version"].item())
                if "result_schema_version" in data
                else None
            )
        session_index = int(name.removeprefix("session_").removesuffix(".npz"))
        if converged_at < 0 or manifest.get("censored") is True:
            raise ValueError(
                f"{name} is censored/unconverged; formal aggregation is forbidden"
            )
        shocks = manifest.get("shocks", {})
        if (
            manifest.get("cell_key") != cell.key()
            or manifest.get("cell") != cell.to_dict()
            or shocks.get("cell_key") != cell.key()
            or shocks.get("session_index") != session_index
            or (expected_seed is not None and shocks.get("experiment_seed") != expected_seed)
        ):
            raise ValueError(f"{name} has mismatched cell/session identity")
        expected_shape = (p.measurement_periods, protocol.row_width(p.num_speculators))
        if rows.shape != expected_shape or not np.isfinite(rows).all():
            raise ValueError(
                f"{name} has invalid rows {rows.shape}; expected {expected_shape} finite rows"
            )
        if manifest.get("converged_at") != converged_at:
            raise ValueError(f"{name} has inconsistent converged_at fields")
        if manifest.get("periods_completed") != converged_at + p.measurement_periods:
            raise ValueError(f"{name} has an inconsistent completed-period count")
        if result_schema is not None:
            if result_schema != 1:
                raise ValueError(f"{name} uses unsupported result schema {result_schema}")
            runner = manifest.get("runner", {})
            if (
                expected_chunk_size is None
                or runner.get("training_chunk_size") != int(expected_chunk_size)
            ):
                raise ValueError(f"{name} has mismatched training_chunk_size")
            for key, value in scientific_identity().items():
                if runner.get(key) != value:
                    raise ValueError(f"{name} has mismatched {key}")
        c = collusion.compute(rows, grids, p.num_speculators, p.value_mean)
        t = trading_policy.compute(rows, p.num_speculators, grids.discrete_value_std, p.noise_std)
        liq = market_quality.liquidity(rows, p.investor_slope)
        mis = market_quality.mispricing(rows, p.num_speculators, t.average_intensity, p.value_mean)
        per_session.append({
            "file": name,
            "provenance_class": (
                "current_schema" if result_schema is not None else "legacy_unversioned"
            ),
            "mechanism": manifest.get("irf", {}).get("mechanism"),
            "liquidity": liq.mean_liquidity,
            "mispricing": mis.mean_mispricing_absolute,
            "converged_at": converged_at,
            "delta_c": c.delta_c,
            "profit_gain_vs_nash": c.profit_gain_vs_nash,
            "chi_hat": t.average_intensity,
            "price_informativeness": t.price_informativeness,
            "mean_lambda_hat": float(rows[:, 4].mean()),
        })
        if progress and (position == 1 or position % 50 == 0 or position == len(session_names)):
            print(
                f"aggregation progress / 汇总进度: {position}/{len(session_names)}",
                file=sys.stderr,
                flush=True,
            )
    if not per_session:
        return {"sessions": 0}
    deltas = np.array([s["delta_c"] for s in per_session])
    mechanism_names = ("price_trigger", "over_pruning", "unclassified")
    mechanism_labels = [s["mechanism"] for s in per_session]
    complete_mechanism_evidence = all(label in mechanism_names for label in mechanism_labels)
    legacy_count = sum(
        session["provenance_class"] == "legacy_unversioned"
        for session in per_session
    )
    recovery_metadata = {}
    recovery_path = os.path.join(out_dir, "recovery_manifest.json")
    if os.path.exists(recovery_path):
        with open(recovery_path, encoding="utf-8") as handle:
            recovery_document = json.load(handle)
        recovery_metadata = {
            "legacy_source_commit_assertion": recovery_document.get(
                "legacy_source_commit_assertion", "unknown"
            ),
            "legacy_runtime_receipt": recovery_document.get(
                "legacy_runtime_receipt", "unknown"
            ),
            "legacy_provenance_status": recovery_document.get(
                "legacy_provenance_status", "unknown"
            ),
            "core_equivalence_status": recovery_document.get(
                "core_equivalence_status", "unknown"
            ),
        }
    summary = {
        # Bind every reported metric to the exact experiment identity.  Step
        # 37A must never be able to pair one cell's summary with another
        # cell.json. / 把汇总值直接绑定到实验身份，防止与错误 cell.json 配对。
        "summary_schema_version": 1,
        "cell_key": cell.key(),
        "cell": cell.to_dict(),
        "experiment_seed": expected_seed,
        "training_chunk_size": expected_chunk_size,
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
        "censored_sessions": 0,
        "legacy_unversioned_sessions": legacy_count,
        "current_schema_sessions": len(per_session) - legacy_count,
        "cohort_provenance": (
            "mixed_legacy_unversioned_and_current_schema"
            if legacy_count
            else "homogeneous_current_schema"
        ),
        "analysis_identity": analysis_identity(),
        **recovery_metadata,
        # Core low/high runs use --irf-paths 0. Do not turn missing or mixed IRF
        # evidence into shares whose numerators silently omit sessions.
        "mechanism_shares": (
            {
                name: mechanism_labels.count(name) / len(mechanism_labels)
                for name in mechanism_names
            }
            if complete_mechanism_evidence
            else None
        ),
        "mechanism_sessions": sum(label in mechanism_names for label in mechanism_labels),
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
    parser.add_argument(
        "--debug-only",
        action="store_true",
        help="acknowledge that this bounded, non-resumable runner is for local tests only",
    )
    a = parser.parse_args()
    if not a.debug_only:
        parser.error(
            "run_cell is a legacy bounded debug runner; add --debug-only for a toy run, "
            "or use dgj.experiments.run_session_cli for formal resumable experiments"
        )
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
