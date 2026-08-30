"""Run one resumable simulation session as an HPC array task.

Scientific completion and computing-job completion are deliberately separate:

* the *session* ends only after the joint greedy policy satisfies the paper's
  unchanged-policy convergence criterion;
* one Slurm *invocation* performs at most ``--work-periods`` additional periods,
  then saves a checkpoint and exits safely if the session is still training.

/ 科学上的 session 只有真正收敛才结束；单个 Slurm job 只负责有限的一段计算。
如果本次 job 结束时尚未收敛，只保存 checkpoint，不测量、不生成正式结果。

Files / 文件:
    <out>/cell.json                 immutable experiment-cell identity
    <out>/ckpt_<k>.npz              exact resumable state while incomplete
    <out>/progress_<k>.json         human-readable current status
    <out>/session_<k>.npz           written only after genuine convergence

Exit codes / 退出码:
    0   genuinely converged result exists
    2   invalid/mismatched input or artifact
    75  safe incomplete pause; checkpoint exists and may be resumed
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import signal
import sys
import tempfile
import time
from typing import Iterator

import numpy as np

from dgj.config import ExperimentCell, PaperParameters
from dgj.game import irf, protocol
from dgj.game.session import Session, atomic_savez_compressed
from dgj.players.market_maker.adaptive import C_STREAK
from dgj.provenance import scientific_identity


SAFE_INCOMPLETE_EXIT = 75
ARTIFACT_SCHEMA_VERSION = 1
DIAGNOSTIC_REVIEW_PERIODS = 50_000_000_000
_STOP_REQUESTED = False


class SessionAlreadyRunningError(RuntimeError):
    """The same session ID is owned by another live process."""


def build_cell(arguments: argparse.Namespace) -> ExperimentCell:
    changes = {
        "noise_std": arguments.noise_std,
        "num_speculators": arguments.speculators,
        "discount_factor": arguments.rho,
        "investor_slope": arguments.xi,
    }
    if arguments.convergence_periods:
        changes["convergence_periods"] = arguments.convergence_periods
    if arguments.measurement_periods:
        changes["measurement_periods"] = arguments.measurement_periods
    return ExperimentCell(
        parameters=PaperParameters(**changes),
        label=arguments.label,
        prehistory=arguments.prehistory,
        price_grid=arguments.price_grid,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_text_write(path: str, text: str) -> None:
    """Atomically replace one text receipt. / 原子写入一份文本记录。"""
    target = os.path.abspath(path)
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=directory,
            prefix=f".{os.path.basename(target)}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass


def _cell_payload(
    cell: ExperimentCell,
    experiment_seed: int,
    training_chunk_size: int,
) -> dict:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "cell": cell.to_dict(),
        "cell_key": cell.key(),
        "experiment_seed": int(experiment_seed),
        "training_chunk_size": int(training_chunk_size),
        **scientific_identity(),
    }


def _validate_cell_file(path: str, expected: dict) -> None:
    try:
        with open(path, encoding="utf-8") as handle:
            actual = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read cell identity {path!r}: {error}") from error
    for key in (
        "artifact_schema_version",
        "cell",
        "cell_key",
        "experiment_seed",
        "training_chunk_size",
        *scientific_identity().keys(),
    ):
        if actual.get(key) != expected[key]:
            raise ValueError(
                f"output directory belongs to a different experiment: {key} mismatch"
            )


def _ensure_cell_file(path: str, expected: dict) -> None:
    """Publish ``cell.json`` once, then verify it on every array task.

    A complete temporary file is hard-linked into place. Two identical array
    tasks may race safely; a different cell is rejected by the final check.
    / 多个 array task 可以安全竞争创建 cell.json；若参数不同则明确拒绝。
    """
    if not os.path.exists(path):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=directory,
                prefix=".cell.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
                json.dump(expected, temporary, indent=2, sort_keys=True)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                pass
        finally:
            if temporary_path is not None:
                try:
                    os.remove(temporary_path)
                except FileNotFoundError:
                    pass
    _validate_cell_file(path, expected)


def _write_progress(
    path: str,
    *,
    status: str,
    session: Session,
    stop_reason: str,
    invocation_start_period: int,
    work_periods: int,
    chunk_size: int,
    checkpoint_every: int,
) -> None:
    training_periods_completed = (
        int(session.converged_at)
        if session.converged_at is not None
        else session.periods_completed
    )
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": status,
        "stop_reason": stop_reason,
        "cell_key": session.cell.key(),
        "session_index": session.session_index,
        "experiment_seed": session.streams.experiment_seed,
        "phase": session.phase,
        "periods_completed": session.periods_completed,
        "training_periods_completed": training_periods_completed,
        "converged_at": session.converged_at,
        "policy_stability_streak": int(session.state.cursor[C_STREAK]),
        "invocation_start_period": invocation_start_period,
        "work_periods_requested": work_periods,
        "work_periods_completed": training_periods_completed - invocation_start_period,
        "training_chunk_size": chunk_size,
        "checkpoint_every_chunks": checkpoint_every,
        "diagnostic_review_due": session.periods_completed >= DIAGNOSTIC_REVIEW_PERIODS,
        "updated_utc": _utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        **scientific_identity(),
    }
    _atomic_text_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _validate_existing_result(
    path: str,
    *,
    cell: ExperimentCell,
    session_index: int,
    experiment_seed: int,
    training_chunk_size: int,
) -> dict:
    """Accept only a complete, genuinely converged result. / 只接受真正收敛的结果。"""
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = {
                "result_schema_version", "rows", "converged_at", "manifest"
            }.difference(data.files)
            if missing:
                raise ValueError("missing fields: " + ", ".join(sorted(missing)))
            result_schema = int(data["result_schema_version"].item())
            rows = data["rows"]
            converged_at = int(data["converged_at"].item())
            manifest = json.loads(str(data["manifest"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid result {path!r}: {error}") from error

    expected_shape = (
        cell.parameters.measurement_periods,
        protocol.row_width(cell.parameters.num_speculators),
    )
    if result_schema != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported result schema {result_schema}; expected {ARTIFACT_SCHEMA_VERSION}"
        )
    if converged_at < 0 or manifest.get("censored") is True:
        raise ValueError(
            "legacy censored result is not complete; preserve it as audit evidence "
            "and rerun this session in a new output directory"
        )
    if rows.shape != expected_shape or not np.isfinite(rows).all():
        raise ValueError(f"measurement rows have shape {rows.shape}; expected {expected_shape}")
    shocks = manifest.get("shocks", {})
    expected_identity = {
        "cell_key": cell.key(),
        "session_index": int(session_index),
        "experiment_seed": int(experiment_seed),
    }
    actual_identity = {
        "cell_key": manifest.get("cell_key"),
        "session_index": shocks.get("session_index"),
        "experiment_seed": shocks.get("experiment_seed"),
    }
    if actual_identity != expected_identity:
        raise ValueError(
            f"result identity mismatch: found {actual_identity}, expected {expected_identity}"
        )
    if manifest.get("cell") != cell.to_dict() or shocks.get("cell_key") != cell.key():
        raise ValueError("result contains an inconsistent full cell identity")
    if manifest.get("converged_at") != converged_at:
        raise ValueError("result manifest and converged_at array disagree")
    expected_periods = converged_at + cell.parameters.measurement_periods
    if manifest.get("periods_completed") != expected_periods:
        raise ValueError(
            "result period count is inconsistent with convergence plus measurement"
        )
    runner = manifest.get("runner", {})
    if runner.get("training_chunk_size") != int(training_chunk_size):
        raise ValueError("result training_chunk_size does not match this campaign")
    expected_engine = scientific_identity()
    for key, value in expected_engine.items():
        if runner.get(key) != value:
            raise ValueError(f"result {key} does not match the current scientific code")
    return manifest


def _repair_complete_progress(
    path: str,
    *,
    manifest: dict,
    session_index: int,
    experiment_seed: int,
    work_periods: int,
    chunk_size: int,
    checkpoint_every: int,
) -> None:
    """Repair a stale receipt when a valid result already exists.

    This covers a crash after atomic result publication but before the old
    checkpoint/progress cleanup. / 处理“结果已写完、清理记录前中断”的小窗口。
    """
    converged_at = int(manifest["converged_at"])
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "complete",
        "stop_reason": "validated_existing_result",
        "cell_key": manifest["cell_key"],
        "session_index": int(session_index),
        "experiment_seed": int(experiment_seed),
        "phase": "complete",
        "periods_completed": int(manifest["periods_completed"]),
        "training_periods_completed": converged_at,
        "converged_at": converged_at,
        "policy_stability_streak": None,
        "invocation_start_period": converged_at,
        "work_periods_requested": int(work_periods),
        "work_periods_completed": 0,
        "training_chunk_size": int(chunk_size),
        "checkpoint_every_chunks": int(checkpoint_every),
        "diagnostic_review_due": converged_at >= DIAGNOSTIC_REVIEW_PERIODS,
        "updated_utc": _utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        **scientific_identity(),
    }
    _atomic_text_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _stop_handler(_signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


@contextmanager
def _exclusive_session_lock(path: str) -> Iterator[None]:
    """Prevent two jobs from advancing one session at the same time.

    The operating-system lock is released automatically if Python or the node
    dies, so an empty/stale ``.lock`` file does not block a later resume.
    / 操作系统锁会在进程或节点终止时自动释放；残留的 .lock 文件本身不会
    阻止之后续跑。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    handle = open(path, "a+b")
    acquired = False
    try:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except (BlockingIOError, OSError) as error:
            raise SessionAlreadyRunningError(
                "another process already owns this session; do not overlap array submissions"
            ) from error
        yield
    finally:
        if acquired:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        else:
            handle.close()


@contextmanager
def _graceful_stop_signals() -> Iterator[None]:
    """Turn Slurm's advance warning into a checkpoint request.

    / 收到 Slurm 提前警告后，不在内核中途硬停；完成当前小段后保存 checkpoint。
    """
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    previous = {}
    names = ["SIGTERM"]
    if hasattr(signal, "SIGUSR1"):
        names.insert(0, "SIGUSR1")
    for name in names:
        signum = getattr(signal, name)
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _stop_handler)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--work-periods",
        type=int,
        default=5_000_000_000,
        help=(
            "additional training periods allowed in this invocation only; "
            "the scientific session has no cumulative period cap"
        ),
    )
    # Keep the old spelling only to produce a clear failure on a stale launcher.
    parser.add_argument("--max-periods", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="chunks between durable checkpoints",
    )
    parser.add_argument(
        "--irf-paths",
        type=int,
        default=0,
        help="0 keeps the core low/high experiment separate from later IRF analysis",
    )
    parser.add_argument(
        "--allow-after-diagnostic-review",
        action="store_true",
        help="continue a session already at/above 50B only after documented operator review",
    )
    return parser


def _main_unlocked(argv=None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.max_periods is not None:
        parser.error(
            "--max-periods was removed because it censored scientific sessions; "
            "use --work-periods for one resumable invocation"
        )
    if arguments.session < 0:
        parser.error("--session must be non-negative")
    if arguments.work_periods < 1 or arguments.chunk_size < 1 or arguments.checkpoint_every < 1:
        parser.error("--work-periods, --chunk-size, and --checkpoint-every must be positive")
    if arguments.work_periods % arguments.chunk_size != 0:
        parser.error(
            "--work-periods must be a multiple of --chunk-size so checkpoint/resume "
            "preserves the same chunked random path"
        )
    if arguments.irf_paths < 0:
        parser.error("--irf-paths must be non-negative")

    cell = build_cell(arguments)
    os.makedirs(arguments.out, exist_ok=True)
    cell_path = os.path.join(arguments.out, "cell.json")
    result_path = os.path.join(arguments.out, f"session_{arguments.session:04d}.npz")
    checkpoint_path = os.path.join(arguments.out, f"ckpt_{arguments.session:04d}.npz")
    progress_path = os.path.join(arguments.out, f"progress_{arguments.session:04d}.json")
    expected_cell = _cell_payload(cell, arguments.seed, arguments.chunk_size)

    try:
        _ensure_cell_file(cell_path, expected_cell)
        if os.path.exists(result_path):
            existing_manifest = _validate_existing_result(
                result_path,
                cell=cell,
                session_index=arguments.session,
                experiment_seed=arguments.seed,
                training_chunk_size=arguments.chunk_size,
            )
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            _repair_complete_progress(
                progress_path,
                manifest=existing_manifest,
                session_index=arguments.session,
                experiment_seed=arguments.seed,
                work_periods=arguments.work_periods,
                chunk_size=arguments.chunk_size,
                checkpoint_every=arguments.checkpoint_every,
            )
            print(f"session {arguments.session}: valid converged result already exists")
            return 0
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2

    session = Session(cell, arguments.session, arguments.seed)
    if os.path.exists(checkpoint_path):
        try:
            session.load_checkpoint(
                checkpoint_path,
                expected_training_chunk_size=arguments.chunk_size,
            )
        except ValueError as error:
            print(f"ERROR: {error}", file=sys.stderr, flush=True)
            return 2
        print(
            f"session {arguments.session}: resumed at {session.periods_completed:,} periods",
            flush=True,
        )
    if (
        session.phase == "training"
        and session.periods_completed >= DIAGNOSTIC_REVIEW_PERIODS
        and not arguments.allow_after_diagnostic_review
    ):
        print(
            "ERROR: this session reached the 50B diagnostic review point. "
            "Inspect its progress and logs; only after recording that review may "
            "you pass --allow-after-diagnostic-review.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    invocation_start = session.periods_completed
    requested_invocation_end = invocation_start + arguments.work_periods
    review_limited = (
        not arguments.allow_after_diagnostic_review
        and invocation_start < DIAGNOSTIC_REVIEW_PERIODS
        and requested_invocation_end >= DIAGNOSTIC_REVIEW_PERIODS
    )
    invocation_end = (
        DIAGNOSTIC_REVIEW_PERIODS if review_limited else requested_invocation_end
    )
    converged = session.phase == "converged"
    stop_reason = "already_converged" if converged else "work_budget_reached"
    started = time.perf_counter()
    chunks_since_checkpoint = 0

    with _graceful_stop_signals():
        while not converged and session.periods_completed < invocation_end:
            if _STOP_REQUESTED:
                stop_reason = "scheduler_signal"
                break
            # Return to Python after every random chunk so a Slurm warning is
            # noticed promptly. Durable disk writes still occur only every
            # ``checkpoint_every`` chunks (or immediately after a signal).
            # / 每个 chunk 都回到 Python 检查信号，但不必每次都写硬盘。
            target = min(invocation_end, session.periods_completed + arguments.chunk_size)
            converged = session.train(chunk_size=arguments.chunk_size, max_periods=target)
            chunks_since_checkpoint += 1
            should_checkpoint = (
                converged
                or _STOP_REQUESTED
                or chunks_since_checkpoint >= arguments.checkpoint_every
                or session.periods_completed >= invocation_end
            )
            if should_checkpoint:
                session.save_checkpoint(
                    checkpoint_path,
                    training_chunk_size=arguments.chunk_size,
                )
                chunks_since_checkpoint = 0
                print(
                    f"session {arguments.session}: "
                    f"{session.periods_completed / 1e6:8.0f}M periods  "
                    f"streak={int(session.state.cursor[C_STREAK]):8d}  "
                    f"{time.perf_counter() - started:7.0f}s",
                    flush=True,
                )
            if _STOP_REQUESTED and not converged:
                stop_reason = "scheduler_signal"
                break

    if not converged:
        if review_limited and session.periods_completed >= DIAGNOSTIC_REVIEW_PERIODS:
            stop_reason = "diagnostic_review_boundary"
        # Save once more even when the stop signal arrived between loop iterations.
        session.save_checkpoint(
            checkpoint_path,
            training_chunk_size=arguments.chunk_size,
        )
        _write_progress(
            progress_path,
            status="incomplete",
            session=session,
            stop_reason=stop_reason,
            invocation_start_period=invocation_start,
            work_periods=arguments.work_periods,
            chunk_size=arguments.chunk_size,
            checkpoint_every=arguments.checkpoint_every,
        )
        print(
            f"session {arguments.session}: safely paused ({stop_reason}) at "
            f"{session.periods_completed:,}; checkpoint retained; no result measured",
            flush=True,
        )
        if session.periods_completed >= DIAGNOSTIC_REVIEW_PERIODS:
            print(
                "WARNING: 50B diagnostic review point reached; inspect policy/streak "
                "diagnostics before continuing. This is not convergence or failure.",
                flush=True,
            )
        return SAFE_INCOMPLETE_EXIT

    # A converged checkpoint exists here. If measurement or result publication
    # is interrupted, the next invocation can replay measurement exactly.
    # Core low/high cells do not need an IRF fork. Avoid copying the whole
    # converged state unless a later mechanism experiment explicitly asks for it.
    fork = irf.take_fork(session) if arguments.irf_paths > 0 else None
    rows = session.measure()
    manifest = session.manifest()
    manifest["censored"] = False
    manifest["runner"] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "training_chunk_size": arguments.chunk_size,
        "checkpoint_every_chunks": arguments.checkpoint_every,
        "invocation_work_periods": arguments.work_periods,
        "scientific_cumulative_period_cap": None,
        "irf_paths": arguments.irf_paths,
        "continued_after_diagnostic_review": bool(
            arguments.allow_after_diagnostic_review
        ),
        **scientific_identity(),
    }
    extra = {}
    if arguments.irf_paths > 0:
        if fork is None:  # defensive; the branch above must have created it
            raise RuntimeError("IRF fork is missing")
        baseline = irf.long_run_baseline(
            rows,
            cell.parameters.num_speculators,
            cell.parameters.value_mean,
        )
        result = irf.run_irf(session, fork, baseline, paths=arguments.irf_paths)
        manifest["irf"] = {
            "paths": result.paths,
            "shock_magnitude": result.shock_magnitude,
            "mechanism": result.mechanism,
            "response_vs_long_run": result.response_vs_long_run,
            "response_vs_control": result.response_vs_control,
            "normalized_price_deviation": result.normalized_price_deviation.tolist(),
        }
        extra = {
            "irf_control_price": result.control_oriented_price,
            "irf_treatment_price": result.treatment_oriented_price,
        }
        print(
            f"session {arguments.session}: IRF mechanism={result.mechanism} "
            f"responses={result.response_vs_long_run}",
            flush=True,
        )

    atomic_savez_compressed(
        result_path,
        result_schema_version=ARTIFACT_SCHEMA_VERSION,
        rows=rows,
        converged_at=int(session.converged_at),
        manifest=json.dumps(manifest),
        **extra,
    )
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    _write_progress(
        progress_path,
        status="complete",
        session=session,
        stop_reason="scientific_convergence_reached",
        invocation_start_period=invocation_start,
        work_periods=arguments.work_periods,
        chunk_size=arguments.chunk_size,
        checkpoint_every=arguments.checkpoint_every,
    )
    print(
        f"session {arguments.session}: done, genuinely converged=True, "
        f"T_c={session.converged_at}, wall={time.perf_counter() - started:.0f}s",
        flush=True,
    )
    return 0


def main(argv=None) -> int:
    """Parse the session identity, acquire its lock, then run it.

    Parsing is repeated inside ``_main_unlocked`` so all validation and error
    messages remain in one place. / 先锁住这个 session，再执行原来的参数校验与运行。
    """
    preview = _parser().parse_args(argv)
    lock_path = os.path.join(
        preview.out,
        f"session_{preview.session:04d}.lock",
    )
    try:
        with _exclusive_session_lock(lock_path):
            return _main_unlocked(argv)
    except SessionAlreadyRunningError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
