"""Run the formal *core* low-noise and high-noise experiments.

运行正式的“核心”低噪声与高噪声实验。

This command-line program deliberately stops at Step 36E.  It trains and
measures 1,000 independent sessions in each cell, then collects the validated
session evidence.  It does **not** run the later common-shock calibration or
mechanism classification (Steps 35D--35F). / 本命令行程序故意停在 Step 36E：
每个实验单元训练并测量 1,000 个独立 session，然后汇总经过验证的 session
证据。它**不会**运行后面的共同冲击校准或机制分类（Steps 35D--35F）。

Beginner workflow / 初学者工作流程::

    py -3 -X utf8 run_formal_experiment.py init
    py -3 -X utf8 run_formal_experiment.py run-session --cell low --session-index 0
    py -3 -X utf8 run_formal_experiment.py status --cell low
    py -3 -X utf8 run_formal_experiment.py collect --cell low

On an HPC cluster, submit one ``run-session`` command per array worker.  A
worker owns exactly one session index. / 在超算上，为每个数组 worker 提交一个
``run-session`` 命令；每个 worker 只负责一个 session 编号。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Iterator, Sequence
import argparse
import csv
import json
import math
import os
import socket
import sys
import time
import uuid


PROJECT_ROOT = Path(__file__).resolve().parent
STEPS_DIRECTORY = PROJECT_ROOT / "steps"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from src.source_manifests import (
    LOADED_EXECUTION_SOURCE_SHA256,
    LOADED_RESULT_PIPELINE_SOURCE_SHA256,
    LOADED_SOURCE_SCOPE_MANIFEST_SHA256,
    SOURCE_SCOPE_MANIFEST_VERSION,
)
from step_27_convergence_tracker import PAPER_UNCHANGED_PERIODS
from step_28_session_phases import PAPER_MEASUREMENT_PERIODS
from steps.step_34_mechanism_classifier import PAPER_PATHS_PER_SESSION
from steps.step_35a_converged_market_checkpoint import (
    LOADED_IMPLEMENTATION_TREE_SHA256,
)
from steps.step_35e_cell_shock_calibration import (
    PAPER_SESSIONS_PER_EXPERIMENT_CELL,
)
from steps.step_36b_experiment_manifest import (
    PAPER_MODE,
    ExperimentCellConfig,
    ExperimentCellPlan,
    ExperimentExecutionPolicy,
    SessionTaskManifest,
    build_experiment_cell_plan,
    load_experiment_cell_plan,
    save_experiment_cell_plan,
    validate_experiment_cell_plan,
)
from steps.step_36d_single_session_training_runner import FAILED, INCOMPLETE
from steps.step_36e_complete_measurement_runner import (
    COMPLETE,
    CompleteMeasurementEvidence,
    CompleteMeasurementTaskError,
    complete_measurement_status_path,
    load_completed_measurement_evidence,
    load_measurement_status,
    run_complete_measurement_task,
)


# Formal counts are written here as explicit assertions.  If an upstream
# constant changes accidentally, this runner stops instead of silently running
# a different experiment. / 这里明确断言正式规模；如果上游常量意外改变，本
# runner 会停止，而不会悄悄运行另一套实验。
FORMAL_SESSION_COUNT = 1_000
FORMAL_STABILITY_STREAK = 1_000_000
FORMAL_MEASUREMENT_PERIODS = 100_000
FORMAL_IRF_PATHS_RESERVED = 10_000

if (
    PAPER_SESSIONS_PER_EXPERIMENT_CELL != FORMAL_SESSION_COUNT
    or PAPER_UNCHANGED_PERIODS != FORMAL_STABILITY_STREAK
    or PAPER_MEASUREMENT_PERIODS != FORMAL_MEASUREMENT_PERIODS
    or PAPER_PATHS_PER_SESSION != FORMAL_IRF_PATHS_RESERVED
):
    raise RuntimeError(
        "Imported paper counts differ from the formal core protocol. / "
        "导入的论文规模与正式核心协议不同。"
    )


FAMILY_SCHEMA_VERSION = "formal-core-family-v1-step36e-only"
RAW_ROWS_SCHEMA_VERSION = "formal-core-raw-session-rows-v1"
CORE_RECEIPT_SCHEMA_VERSION = "formal-core-cell-receipt-v1"
LOCK_SCHEMA_VERSION = "formal-core-exclusive-worker-lock-v1"

DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "results" / "formal_core_experiment"
FAMILY_MANIFEST_NAME = "formal_core_family_manifest.json"
PLAN_RELATIVE_PATHS = {
    "low": "plans/low_noise_plan.json",
    "high": "plans/high_noise_plan.json",
}
CELL_ARTIFACT_RELATIVE_PATHS = {
    "low": "cells/low_noise",
    "high": "cells/high_noise",
}
CELL_NOISE_STDS = {"low": 0.1, "high": 100.0}

# Fixed roots make every one of the 2,000 session streams reproducible.
# / 固定根种子让 2,000 个 session 的随机流都可以复现。
CELL_EXPERIMENT_SEEDS = {"low": 20_260_830_001, "high": 20_260_830_002}
CELL_IRF_SEEDS = {"low": 20_260_830_101, "high": 20_260_830_102}

DEFAULT_CHECKPOINT_INTERVAL = 1_000_000
EXIT_COMPLETE = 0
EXIT_FAILED = 2
EXIT_LOCK_CONFLICT = 73
EXIT_INCOMPLETE = 75
TYPE7_LABEL = (
    "Hyndman-Fan type 7; h=(n-1)p; cross-sectional across independent sessions"
)


@dataclass(frozen=True)
class CrossSectionalSummary:
    """One across-session summary, never a time-series percentile.

    一个跨 session 的汇总；它不是时间序列百分位数。
    """

    metric: str
    total_session_count: int
    defined_session_count: int
    undefined_session_count: int
    formal_all_session_mean: float | None
    mean_of_defined_sessions: float | None
    minimum_of_defined_sessions: float | None
    p01_type7_cross_sectional: float | None
    p50_type7_cross_sectional: float | None
    p99_type7_cross_sectional: float | None
    maximum_of_defined_sessions: float | None
    quantile_method: str = TYPE7_LABEL
    undefined_values_imputed: bool = False


@dataclass(frozen=True)
class FormalCoreSessionRow:
    """Raw values from one validated Step-36E evidence bundle.

    一个经过验证的 Step-36E evidence bundle 的原始数值。
    """

    session_index: int
    task_id: str
    task_sha256: str
    evidence_sha256: str
    result_sha256: str
    training_periods_completed: int
    measurement_periods_completed: int
    actual_profit_agent_1: float
    actual_profit_agent_2: float
    mean_actual_profit_across_agents: float
    mean_nash_profit: float
    mean_cartel_profit: float
    delta_c_unclamped: float
    trading_intensity: float
    price_informativeness: float
    average_market_liquidity: float
    reported_average_mispricing: float | None
    mispricing_defined: bool
    mispricing_requires_research_decision: bool
    mean_irf_oriented_price: float
    mean_irf_oriented_order_agent_1: float
    mean_irf_oriented_order_agent_2: float
    mean_irf_profit_agent_1: float
    mean_irf_profit_agent_2: float
    measurement_scored_fields_sha256: str
    a23_source_fingerprint_scope_resolved: bool


class SessionLockConflictError(RuntimeError):
    """Another worker already owns the same session. / 另一个 worker 已独占同一 session。"""


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    """Produce deterministic JSON. / 产生确定不变的 JSON。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _runner_sha256() -> str:
    """Fingerprint this entrypoint with checkout-independent newlines.

    使用与 Git checkout 无关的换行符，对本入口脚本本身取指纹。

    Git may check out CRLF on Windows and LF on Linux/HPC.  Like the explicit
    source manifests, this self-hash normalizes CRLF to LF so that the same
    committed runner keeps one identity on both systems. / Git 在 Windows 可
    checkout 为 CRLF，在 Linux/超算则为 LF；与明确源码清单相同，这里先把
    CRLF 规范成 LF，让同一份提交在两个系统保持同一个身份。
    """

    normalized = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return _sha256_bytes(normalized)


def _checked_relative_child(root: Path, relative_text: str) -> Path:
    """Resolve a saved relative path without allowing ``..`` escapes.

    安全解析保存的相对路径，不允许 ``..`` 越出根目录。
    """

    relative = PurePosixPath(relative_text)
    if relative.is_absolute() or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise ValueError("Unsafe relative artifact path. / 不安全的相对 artifact 路径。")
    resolved_root = root.resolve()
    target = resolved_root.joinpath(*relative.parts).resolve()
    if target == resolved_root or resolved_root not in target.parents:
        raise ValueError("Artifact path escapes its root. / artifact 路径越出根目录。")
    return target


def _immutable_write_bytes(path: Path, data: bytes) -> Path:
    """Publish complete bytes once; an existing different file is an error.

    只发布一次完整字节；若已有不同文件则报错。

    The temporary file is complete before it is hard-linked to the final name.
    This avoids presenting a half-written research artifact after a crash.
    / 临时文件写完整以后才硬链接到最终名称，避免崩溃后留下看似正式的半截文件。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return path
        raise FileExistsError(
            f"Different immutable artifact already exists: {path} / "
            f"已有不同的不可变 artifact：{path}"
        )

    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.complete"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Hard-link publication is atomic and refuses an existing target.
            # / 硬链接发布是原子的，而且拒绝覆盖现有目标。
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise FileExistsError(
                    f"Concurrent writer published different data: {path} / "
                    f"并发 writer 发布了不同数据：{path}"
                )
        except OSError:
            # Some filesystems do not support hard links. Exclusive creation is
            # a fail-closed fallback: a crash can leave an invalid file, which
            # every loader rejects rather than treating it as evidence.
            # / 某些文件系统不支持硬链接；排他创建是 fail-closed 后备方案。
            target_descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(target_descriptor, "wb") as target_handle:
                target_handle.write(data)
                target_handle.flush()
                os.fsync(target_handle.fileno())
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _immutable_write_json(path: Path, value: object) -> Path:
    text = _canonical_json(value, indent=2) + "\n"
    return _immutable_write_bytes(path, text.encode("utf-8"))


def _cell_plan(cell: str) -> ExperimentCellPlan:
    """Build one exact PAPER_MODE plan. / 建立一个精确 PAPER_MODE 计划。"""

    if cell not in CELL_NOISE_STDS:
        raise ValueError("cell must be 'low' or 'high'. / cell 必须是 low 或 high。")
    config = ExperimentCellConfig(
        mode=PAPER_MODE,
        experiment_cell_key=f"formal-core-{cell}-noise-v1",
        parameters=PaperParameters(noise_std=CELL_NOISE_STDS[cell]),
        experiment_seed=CELL_EXPERIMENT_SEEDS[cell],
        irf_experiment_seed=CELL_IRF_SEEDS[cell],
        session_count=FORMAL_SESSION_COUNT,
        convergence_periods_required=FORMAL_STABILITY_STREAK,
        measurement_periods_required=FORMAL_MEASUREMENT_PERIODS,
        irf_paths_per_session=FORMAL_IRF_PATHS_RESERVED,
        # Step 36B requires this paper-plan flag.  This core runner nevertheless
        # stops at Step 36E and publishes no mechanism classification.
        # / Step 36B 的正式计划要求此标记；但本核心 runner 仍停在 Step 36E，
        # 不发布任何机制分类。
        mechanism_analysis_enabled=True,
    )
    policy = ExperimentExecutionPolicy(
        maximum_training_periods=None,
        within_session_checkpointing_available=True,
        persisted_post_convergence_bundle_available=True,
        formal_session_runner_available=True,
        hpc_array_dispatch_available=True,
    )
    plan = build_experiment_cell_plan(config, policy)
    validate_experiment_cell_plan(plan)
    if not (
        plan.formal_mode_requested
        and plan.paper_scale_counts_requested
        and plan.uncapped_training_requested
        and plan.formal_session_runner_connected
        and plan.within_session_checkpointing_available
        and plan.persisted_post_convergence_bundle_available
        and plan.hpc_array_dispatch_available
    ):
        raise RuntimeError("Formal plan policy flags are incomplete. / 正式计划的 policy 标记不完整。")
    return plan


def _family_payload() -> dict[str, object]:
    """Build immutable family metadata for both cells. / 建立两个 cell 的不可变元数据。"""

    cells: dict[str, object] = {}
    family_session_seeds: set[int] = set()
    family_child_stream_seeds: set[int] = set()
    for cell in ("low", "high"):
        plan = _cell_plan(cell)
        for task in plan.tasks:
            family_session_seeds.add(task.seed_manifest.session_seed)
            family_child_stream_seeds.update(task.seed_manifest.child_seeds())
        cells[cell] = {
            "noise_std": CELL_NOISE_STDS[cell],
            "experiment_seed": CELL_EXPERIMENT_SEEDS[cell],
            "irf_experiment_seed": CELL_IRF_SEEDS[cell],
            "plan_relative_path": PLAN_RELATIVE_PATHS[cell],
            "artifact_relative_path": CELL_ARTIFACT_RELATIVE_PATHS[cell],
            "plan_sha256": plan.plan_sha256,
            "run_config_sha256": plan.run_config_sha256,
            "experiment_cell_sha256": plan.experiment_cell_sha256,
        }
    expected_sessions = 2 * FORMAL_SESSION_COUNT
    expected_child_streams = 7 * expected_sessions
    if len(family_session_seeds) != expected_sessions:
        raise RuntimeError("Cross-cell session-seed collision detected. / 检测到跨 cell 的 session 种子碰撞。")
    if len(family_child_stream_seeds) != expected_child_streams:
        raise RuntimeError("Cross-cell child-stream seed collision detected. / 检测到跨 cell 的子随机流种子碰撞。")
    return {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "scope": "formal core training and Step-36E measurement only",
        "runner_relative_path": "run_formal_experiment.py",
        "runner_sha256": _runner_sha256(),
        "source_scope_manifest_version": SOURCE_SCOPE_MANIFEST_VERSION,
        "source_scope_manifest_sha256": LOADED_SOURCE_SCOPE_MANIFEST_SHA256,
        "execution_source_sha256": LOADED_EXECUTION_SOURCE_SHA256,
        "result_pipeline_source_sha256": LOADED_RESULT_PIPELINE_SOURCE_SHA256,
        "implementation_tree_sha256": LOADED_IMPLEMENTATION_TREE_SHA256,
        "session_count_per_cell": FORMAL_SESSION_COUNT,
        "unique_session_seed_count_across_cells": len(family_session_seeds),
        "unique_child_stream_seed_count_across_cells": len(
            family_child_stream_seeds
        ),
        "convergence_unchanged_periods": FORMAL_STABILITY_STREAK,
        "measurement_periods_per_session": FORMAL_MEASUREMENT_PERIODS,
        "irf_paths_per_session_reserved_for_later_steps": FORMAL_IRF_PATHS_RESERVED,
        "cells": cells,
        "step35d_irf_paths_run": False,
        "step35e_common_shock_calibrated": False,
        "step35f_mechanisms_classified": False,
        "mechanism_results_included": False,
        "research_result": False,
        "paper_results_ready": False,
    }


def _family_manifest() -> dict[str, object]:
    payload = _family_payload()
    return {**payload, "family_manifest_sha256": _sha256_json(payload)}


def initialize_family(artifact_root: Path) -> dict[str, object]:
    """Save and immediately reload both plans and the family manifest.

    保存并立即重读两个计划和 family manifest。
    """

    root = artifact_root.resolve()
    for cell in ("low", "high"):
        plan = _cell_plan(cell)
        plan_path = _checked_relative_child(root, PLAN_RELATIVE_PATHS[cell])
        save_experiment_cell_plan(plan, plan_path)
        if load_experiment_cell_plan(plan_path) != plan:
            raise RuntimeError("Saved plan did not round-trip. / 保存的计划未能一致重读。")
        _checked_relative_child(root, CELL_ARTIFACT_RELATIVE_PATHS[cell]).mkdir(
            parents=True,
            exist_ok=True,
        )
    manifest = _family_manifest()
    _immutable_write_json(root / FAMILY_MANIFEST_NAME, manifest)
    return load_family_manifest(root)


def load_family_manifest(artifact_root: Path) -> dict[str, object]:
    """Load the family and reject code changes or tampering. / 读取 family，并拒绝代码变化或篡改。"""

    root = artifact_root.resolve()
    path = root / FAMILY_MANIFEST_NAME
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read a complete family manifest. / 无法读取完整 family manifest。") from error
    expected = _family_manifest()
    if loaded != expected:
        raise ValueError(
            "Family manifest was changed, or this runner/source build differs. "
            "/ family manifest 被修改，或当前 runner/源码版本不同。"
        )
    for cell in ("low", "high"):
        plan_path = _checked_relative_child(root, PLAN_RELATIVE_PATHS[cell])
        loaded_plan = load_experiment_cell_plan(plan_path)
        if loaded_plan != _cell_plan(cell):
            raise ValueError("Saved cell plan differs from the formal plan. / 已保存 cell 计划与正式计划不同。")
    return loaded


def _load_cell(artifact_root: Path, cell: str) -> tuple[ExperimentCellPlan, Path]:
    load_family_manifest(artifact_root)
    if cell not in CELL_NOISE_STDS:
        raise ValueError("cell must be 'low' or 'high'. / cell 必须是 low 或 high。")
    plan = load_experiment_cell_plan(
        _checked_relative_child(artifact_root.resolve(), PLAN_RELATIVE_PATHS[cell])
    )
    cell_root = _checked_relative_child(
        artifact_root.resolve(), CELL_ARTIFACT_RELATIVE_PATHS[cell]
    )
    return plan, cell_root


def _task_directory(cell_root: Path, task: SessionTaskManifest) -> Path:
    return _checked_relative_child(cell_root, task.relative_artifact_directory)


def _worker_lock_path(cell_root: Path, task: SessionTaskManifest) -> Path:
    return _task_directory(cell_root, task) / "formal_core_worker.lock"


@contextmanager
def _exclusive_session_claim(
    cell_root: Path,
    task: SessionTaskManifest,
) -> Iterator[Path]:
    """Own one session using an atomic ``O_EXCL`` lock.

    用原子的 ``O_EXCL`` 锁独占一个 session。

    A second worker cannot accidentally train the same session.  A lock left
    by a killed process is intentionally not guessed to be stale; a researcher
    must inspect it before manual removal. / 第二个 worker 无法误跑同一个
    session。若进程被强制杀死留下锁，程序不会猜测它已失效；研究者应检查后
    再手动移除。
    """

    lock_path = _worker_lock_path(cell_root, task)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    record = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "task_id": task.task_id,
        "task_sha256": task.task_sha256,
        "process_id": os.getpid(),
        "host": socket.gethostname(),
        "runner_sha256": _runner_sha256(),
        "claimed_unix_time": time.time(),
        "claim_token": token,
    }
    data = (_canonical_json(record, indent=2) + "\n").encode("utf-8")
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise SessionLockConflictError(
            f"Session already has a worker lock: {lock_path} / "
            f"此 session 已有 worker 锁：{lock_path}"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        yield lock_path
    finally:
        # Delete only our own lock. / 只删除本 worker 自己建立的锁。
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
            if current.get("claim_token") == token:
                lock_path.unlink()
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            pass


def run_one_session(
    artifact_root: Path,
    *,
    cell: str,
    session_index: int,
    checkpoint_interval_periods: int,
    invocation_period_budget: int | None,
    retry_failed: bool,
) -> dict[str, object]:
    """Run or exactly resume one formal session. / 运行或精确续跑一个正式 session。"""

    plan, cell_root = _load_cell(artifact_root, cell)
    if isinstance(session_index, bool) or not isinstance(session_index, int):
        raise TypeError("session_index must be an integer. / session_index 必须是整数。")
    if not 0 <= session_index < FORMAL_SESSION_COUNT:
        raise ValueError("session_index must lie in [0, 999]. / session_index 必须在 [0, 999]。")
    if (
        isinstance(checkpoint_interval_periods, bool)
        or not isinstance(checkpoint_interval_periods, int)
        or checkpoint_interval_periods < 1
    ):
        raise ValueError("checkpoint interval must be positive. / checkpoint 间隔必须为正数。")
    if invocation_period_budget is not None and (
        isinstance(invocation_period_budget, bool)
        or not isinstance(invocation_period_budget, int)
        or invocation_period_budget < 1
    ):
        raise ValueError("period budget must be positive. / period budget 必须为正数。")
    task = plan.tasks[session_index]
    with _exclusive_session_claim(cell_root, task):
        execution = run_complete_measurement_task(
            plan,
            task,
            artifact_root=cell_root,
            checkpoint_interval_periods=checkpoint_interval_periods,
            # This optional budget ends only the current worker invocation. It
            # does not alter the uncapped PAPER_MODE plan, and the next worker
            # resumes exactly from Step 36E's checkpoint. / 这个可选 budget 只
            # 结束当前 worker 调用，不修改不设上限的 PAPER_MODE 计划；下一次
            # 调用会从 Step 36E checkpoint 精确续跑。
            invocation_training_period_budget=invocation_period_budget,
            retry_failed=retry_failed,
        )
    outcome = execution.status.scientific_outcome
    return {
        "cell": cell,
        "session_index": session_index,
        "task_id": task.task_id,
        "status": outcome.status,
        "phase": outcome.phase,
        "stop_reason": outcome.stop_reason,
        "training_periods_verified": outcome.training_periods_verified,
        "measurement_rows_committed": outcome.committed_measurement_rows,
        "complete_evidence_available": execution.evidence is not None,
    }


def _classify_task_status(
    plan: ExperimentCellPlan,
    cell_root: Path,
    task: SessionTaskManifest,
) -> str:
    """Return one operational status label. / 返回一个运行状态标签。"""

    lock_exists = _worker_lock_path(cell_root, task).is_file()
    status_path = complete_measurement_status_path(
        artifact_root=cell_root,
        task=task,
    )
    if not status_path.is_file():
        return "running" if lock_exists else "pending"
    try:
        status = load_measurement_status(
            status_path,
            expected_task=task,
            expected_config=plan.config,
        )
        scientific = status.scientific_outcome.status
        if scientific == COMPLETE:
            # COMPLETE counts only after the referenced evidence also validates.
            # / 只有引用的 evidence 也通过验证，才计为 COMPLETE。
            load_completed_measurement_evidence(
                plan,
                task,
                artifact_root=cell_root,
            )
            return "complete"
        if lock_exists:
            return "running"
        if scientific == INCOMPLETE:
            return "incomplete"
        if scientific == FAILED:
            return "failed"
        return "invalid"
    except (OSError, TypeError, ValueError, RuntimeError):
        return "invalid"


def status_counts(artifact_root: Path, cell: str) -> dict[str, int]:
    """Count all 1,000 tasks without changing them. / 只读统计全部 1,000 个任务。"""

    plan, cell_root = _load_cell(artifact_root, cell)
    counts = {
        "pending": 0,
        "running": 0,
        "incomplete": 0,
        "failed": 0,
        "complete": 0,
        "invalid": 0,
    }
    for task in plan.tasks:
        counts[_classify_task_status(plan, cell_root, task)] += 1
    if sum(counts.values()) != FORMAL_SESSION_COUNT:
        raise RuntimeError("Status counts do not sum to 1,000. / 状态计数之和不是 1,000。")
    return counts


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} is not numeric. / {label} 不是数值。")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} is not finite. / {label} 不是有限数。")
    return converted


def _row_from_evidence(evidence: CompleteMeasurementEvidence) -> FormalCoreSessionRow:
    """Copy values without clipping Delta C. / 原样复制数值，不裁剪 Delta C。"""

    result = evidence.learned_session_result
    actual = result.mean_actual_profit_by_agent
    oriented_orders = result.mean_irf_oriented_order_by_agent
    irf_profits = result.mean_irf_profit_by_agent
    if len(actual) != 2 or len(oriented_orders) != 2 or len(irf_profits) != 2:
        raise ValueError("Formal core currently requires exactly two agents. / 正式核心当前要求恰好两个 agent。")
    mispricing = result.reported_average_mispricing
    mispricing_defined = mispricing is not None
    if mispricing_defined == result.mispricing_requires_research_decision:
        raise ValueError("Mispricing value and decision flag disagree. / mispricing 数值与决策标记矛盾。")

    numeric_values = {
        "actual_profit_agent_1": actual[0],
        "actual_profit_agent_2": actual[1],
        "mean_nash_profit": result.mean_nash_profit,
        "mean_cartel_profit": result.mean_cartel_profit,
        "delta_c": result.delta_c,
        "trading_intensity": result.trading_intensity,
        "price_informativeness": result.price_informativeness,
        "average_market_liquidity": result.average_market_liquidity,
        "mean_irf_oriented_price": result.mean_irf_oriented_price,
        "mean_irf_oriented_order_agent_1": oriented_orders[0],
        "mean_irf_oriented_order_agent_2": oriented_orders[1],
        "mean_irf_profit_agent_1": irf_profits[0],
        "mean_irf_profit_agent_2": irf_profits[1],
    }
    checked = {
        label: _finite_float(value, label)
        for label, value in numeric_values.items()
    }
    checked_mispricing = (
        None
        if mispricing is None
        else _finite_float(mispricing, "reported_average_mispricing")
    )
    mean_actual = (
        checked["actual_profit_agent_1"] + checked["actual_profit_agent_2"]
    ) / 2.0
    return FormalCoreSessionRow(
        session_index=evidence.session_index,
        task_id=evidence.task_id,
        task_sha256=evidence.task_sha256,
        evidence_sha256=evidence.evidence_sha256,
        result_sha256=result.result_sha256,
        training_periods_completed=result.training_periods_completed,
        measurement_periods_completed=result.measurement_periods_completed,
        actual_profit_agent_1=checked["actual_profit_agent_1"],
        actual_profit_agent_2=checked["actual_profit_agent_2"],
        mean_actual_profit_across_agents=mean_actual,
        mean_nash_profit=checked["mean_nash_profit"],
        mean_cartel_profit=checked["mean_cartel_profit"],
        # No max(0, ...) and no min(1, ...). Values outside [0,1] are data.
        # / 不做 max(0, ...) 或 min(1, ...)；超出 [0,1] 仍是数据。
        delta_c_unclamped=checked["delta_c"],
        trading_intensity=checked["trading_intensity"],
        price_informativeness=checked["price_informativeness"],
        average_market_liquidity=checked["average_market_liquidity"],
        reported_average_mispricing=checked_mispricing,
        mispricing_defined=mispricing_defined,
        mispricing_requires_research_decision=(
            result.mispricing_requires_research_decision
        ),
        mean_irf_oriented_price=checked["mean_irf_oriented_price"],
        mean_irf_oriented_order_agent_1=checked[
            "mean_irf_oriented_order_agent_1"
        ],
        mean_irf_oriented_order_agent_2=checked[
            "mean_irf_oriented_order_agent_2"
        ],
        mean_irf_profit_agent_1=checked["mean_irf_profit_agent_1"],
        mean_irf_profit_agent_2=checked["mean_irf_profit_agent_2"],
        measurement_scored_fields_sha256=(
            result.measurement_scored_fields_sha256
        ),
        a23_source_fingerprint_scope_resolved=(
            result.a23_source_fingerprint_scope_resolved
        ),
    )


def _type7(values: Sequence[float], probability: float) -> float:
    """Hyndman--Fan type-7 sample quantile. / Hyndman--Fan 第 7 型样本分位数。"""

    if not values:
        raise ValueError("A quantile needs at least one value. / 分位数至少需要一个数值。")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]. / probability 必须位于 [0,1]。")
    ordered = sorted(_finite_float(value, "quantile value") for value in values)
    h = (len(ordered) - 1) * probability
    lower = math.floor(h)
    fraction = h - lower
    if lower >= len(ordered) - 1:
        return ordered[-1]
    return ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])


def _summary(
    metric: str,
    values: Sequence[float | None],
) -> CrossSectionalSummary:
    """Summarize defined values without imputing missing mispricing.

    只汇总有定义的数值，不填补缺失的 mispricing。
    """

    defined = [
        _finite_float(value, metric)
        for value in values
        if value is not None
    ]
    total = len(values)
    undefined = total - len(defined)
    if not defined:
        return CrossSectionalSummary(
            metric=metric,
            total_session_count=total,
            defined_session_count=0,
            undefined_session_count=undefined,
            formal_all_session_mean=None,
            mean_of_defined_sessions=None,
            minimum_of_defined_sessions=None,
            p01_type7_cross_sectional=None,
            p50_type7_cross_sectional=None,
            p99_type7_cross_sectional=None,
            maximum_of_defined_sessions=None,
        )
    defined_mean = math.fsum(defined) / len(defined)
    return CrossSectionalSummary(
        metric=metric,
        total_session_count=total,
        defined_session_count=len(defined),
        undefined_session_count=undefined,
        # A formal cell mean cannot silently drop sessions.  If any value is
        # undefined (possible for mispricing), the all-session mean remains
        # None.  The defined-only mean below is explicitly diagnostic.
        # / 正式 cell 均值不能悄悄丢弃 session；只要有一个数值未定义（mispricing
        # 可能如此），全样本均值就保持 None。下面的有定义样本均值明确只是诊断。
        formal_all_session_mean=(defined_mean if undefined == 0 else None),
        mean_of_defined_sessions=defined_mean,
        minimum_of_defined_sessions=min(defined),
        p01_type7_cross_sectional=_type7(defined, 0.01),
        p50_type7_cross_sectional=_type7(defined, 0.50),
        p99_type7_cross_sectional=_type7(defined, 0.99),
        maximum_of_defined_sessions=max(defined),
    )


CSV_FIELDS = tuple(FormalCoreSessionRow.__dataclass_fields__.keys())


def _float_for_csv(value: object) -> object:
    """Use 17 significant digits so a float round-trips. / 用 17 位有效数字保证 float 可往返。"""

    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _rows_csv(rows: Sequence[FormalCoreSessionRow]) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {key: _float_for_csv(value) for key, value in asdict(row).items()}
        )
    return buffer.getvalue().encode("utf-8")


def _metric_summaries(rows: Sequence[FormalCoreSessionRow]) -> dict[str, object]:
    metric_fields = (
        "training_periods_completed",
        "mean_actual_profit_across_agents",
        "mean_nash_profit",
        "mean_cartel_profit",
        "delta_c_unclamped",
        "trading_intensity",
        "price_informativeness",
        "average_market_liquidity",
        "reported_average_mispricing",
    )
    return {
        field: asdict(_summary(field, [getattr(row, field) for row in rows]))
        for field in metric_fields
    }


def collect_cell(artifact_root: Path, cell: str) -> dict[str, object]:
    """Collect exactly 1,000 validated Step-36E results for one cell.

    为一个 cell 汇总恰好 1,000 份经过验证的 Step-36E 结果。
    """

    family = load_family_manifest(artifact_root)
    plan, cell_root = _load_cell(artifact_root, cell)
    if len(plan.tasks) != FORMAL_SESSION_COUNT:
        raise RuntimeError("Formal collection requires exactly 1,000 tasks. / 正式汇总要求恰好 1,000 个任务。")

    rows: list[FormalCoreSessionRow] = []
    evidence_hashes: set[str] = set()
    task_ids: set[str] = set()
    for expected_index, task in enumerate(plan.tasks):
        evidence = load_completed_measurement_evidence(
            plan,
            task,
            artifact_root=cell_root,
        )
        if evidence.session_index != expected_index:
            raise ValueError("Evidence order/index mismatch. / evidence 顺序或编号不一致。")
        if not evidence.learned_session_result.exact_paper_scale_counts_matched:
            raise ValueError("A session does not have exact paper-scale counts. / 某个 session 不是精确论文规模。")
        if not evidence.learned_session_result.a23_source_fingerprint_scope_resolved:
            raise ValueError(
                "A session does not carry the resolved A23 source-scope proof. "
                "/ 某个 session 没有携带已解决的 A23 源码范围证明。"
            )
        if evidence.evidence_sha256 in evidence_hashes or task.task_id in task_ids:
            raise ValueError("Duplicate evidence or task detected. / 检测到重复 evidence 或 task。")
        evidence_hashes.add(evidence.evidence_sha256)
        task_ids.add(task.task_id)
        rows.append(_row_from_evidence(evidence))

    exact_complete_cell = (
        len(rows) == FORMAL_SESSION_COUNT
        and len(evidence_hashes) == FORMAL_SESSION_COUNT
        and len(task_ids) == FORMAL_SESSION_COUNT
        and plan.config.mode == PAPER_MODE
        and plan.paper_scale_counts_requested
        and plan.uncapped_training_requested
        and plan.config.parameters.noise_std == CELL_NOISE_STDS[cell]
        and all(
            # A23 proves that the execution and result-pipeline source scopes
            # were both fingerprinted. / A23 证明执行与结果管线的源码范围都已取指纹。
            row.a23_source_fingerprint_scope_resolved
            for row in rows
        )
        and all(
            row.measurement_periods_completed == FORMAL_MEASUREMENT_PERIODS
            for row in rows
        )
    )
    if not exact_complete_cell:
        raise RuntimeError("Cell did not satisfy every formal-core condition. / cell 未满足全部正式核心条件。")

    row_dictionaries = [asdict(row) for row in rows]
    raw_rows_sha256 = _sha256_json(row_dictionaries)
    raw_envelope_without_checksum = {
        "schema_version": RAW_ROWS_SCHEMA_VERSION,
        "cell": cell,
        "noise_std": CELL_NOISE_STDS[cell],
        "plan_sha256": plan.plan_sha256,
        "row_count": len(rows),
        "ordered_by": "session_index ascending",
        "delta_c_storage": "unclamped",
        "undefined_mispricing_storage": "JSON null; never imputed",
        "rows_sha256": raw_rows_sha256,
        "rows": row_dictionaries,
    }
    raw_envelope = {
        **raw_envelope_without_checksum,
        "raw_artifact_sha256": _sha256_json(raw_envelope_without_checksum),
    }
    raw_bytes = (_canonical_json(raw_envelope, indent=2) + "\n").encode("utf-8")
    csv_bytes = _rows_csv(rows)
    raw_file_sha256 = _sha256_bytes(raw_bytes)
    csv_file_sha256 = _sha256_bytes(csv_bytes)

    collection_directory = cell_root / "core_collection"
    raw_name = f"raw_session_rows_{raw_rows_sha256[:20]}.json"
    csv_name = f"raw_session_rows_{raw_rows_sha256[:20]}.csv"
    raw_path = collection_directory / raw_name
    csv_path = collection_directory / csv_name
    _immutable_write_bytes(raw_path, raw_bytes)
    _immutable_write_bytes(csv_path, csv_bytes)

    summaries = _metric_summaries(rows)
    undefined_mispricing_count = sum(
        not row.mispricing_defined for row in rows
    )
    receipt_without_checksum = {
        "schema_version": CORE_RECEIPT_SCHEMA_VERSION,
        "cell": cell,
        "noise_std": CELL_NOISE_STDS[cell],
        "family_manifest_sha256": family["family_manifest_sha256"],
        "plan_sha256": plan.plan_sha256,
        "run_config_sha256": plan.run_config_sha256,
        "experiment_cell_sha256": plan.experiment_cell_sha256,
        "runner_sha256": _runner_sha256(),
        "source_scope_manifest_version": SOURCE_SCOPE_MANIFEST_VERSION,
        "source_scope_manifest_sha256": LOADED_SOURCE_SCOPE_MANIFEST_SHA256,
        "execution_source_sha256": LOADED_EXECUTION_SOURCE_SHA256,
        "result_pipeline_source_sha256": LOADED_RESULT_PIPELINE_SOURCE_SHA256,
        "implementation_tree_sha256": LOADED_IMPLEMENTATION_TREE_SHA256,
        "a23_source_fingerprint_scope_resolved_for_every_session": True,
        "expected_session_count": FORMAL_SESSION_COUNT,
        "validated_complete_session_count": len(rows),
        "convergence_unchanged_periods_required": FORMAL_STABILITY_STREAK,
        "measurement_periods_per_session": FORMAL_MEASUREMENT_PERIODS,
        "irf_paths_per_session_reserved_for_later_steps": FORMAL_IRF_PATHS_RESERVED,
        "raw_rows_relative_path": raw_path.relative_to(artifact_root.resolve()).as_posix(),
        "raw_rows_sha256": raw_rows_sha256,
        "raw_file_sha256": raw_file_sha256,
        "csv_relative_path": csv_path.relative_to(artifact_root.resolve()).as_posix(),
        "csv_file_sha256": csv_file_sha256,
        "delta_c_unclamped": True,
        "delta_c_values_outside_zero_one_retained": True,
        "quantiles_are_cross_sectional_across_sessions": True,
        "quantile_method": TYPE7_LABEL,
        "mispricing_undefined_session_count": undefined_mispricing_count,
        "mispricing_undefined_values_imputed": False,
        "mispricing_summary_population": (
            "formal mean uses all sessions or null; defined-only diagnostics "
            "are separately labeled"
        ),
        "mispricing_formal_mean_is_null_if_any_session_is_undefined": True,
        "metric_summaries": summaries,
        # Core evidence ends at Step 36E. / 核心证据停在 Step 36E。
        "step35d_irf_paths_run": False,
        "step35e_common_shock_calibrated": False,
        "step35f_mechanisms_classified": False,
        "mechanism_results_included": False,
        "exact_formal_core_cell_completed": exact_complete_cell,
        # This is a genuine core research result only because all 1,000 exact
        # sessions validated. It is not yet the paper's full mechanism result.
        # / 只有全部 1,000 个正式 session 验证后，它才是核心研究结果；但它
        # 仍不是论文的完整机制结果。
        "research_result": exact_complete_cell,
        "paper_results_ready": False,
    }
    receipt = {
        **receipt_without_checksum,
        "core_receipt_sha256": _sha256_json(receipt_without_checksum),
    }
    receipt_path = collection_directory / (
        f"core_receipt_{receipt['core_receipt_sha256'][:20]}.json"
    )
    _immutable_write_json(receipt_path, receipt)
    return {
        "cell": cell,
        "receipt_path": str(receipt_path),
        "raw_rows_path": str(raw_path),
        "csv_path": str(csv_path),
        "validated_complete_session_count": len(rows),
        "research_result": receipt["research_result"],
        "paper_results_ready": receipt["paper_results_ready"],
        "delta_c_summary": summaries["delta_c_unclamped"],
        "undefined_mispricing_session_count": undefined_mispricing_count,
    }


def _cells_from_argument(cell: str) -> tuple[str, ...]:
    return ("low", "high") if cell == "all" else (cell,)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be positive / 数值必须为正")
    return value


def _session_index(text: str) -> int:
    value = int(text)
    if not 0 <= value < FORMAL_SESSION_COUNT:
        raise argparse.ArgumentTypeError("session index must be 0..999 / session 编号必须为 0..999")
    return value


def _array_index(text: str) -> int:
    value = int(text)
    if not 0 <= value < 2 * FORMAL_SESSION_COUNT:
        raise argparse.ArgumentTypeError("array index must be 0..1999 / array 编号必须为 0..1999")
    return value


def _resolve_run_target(arguments: argparse.Namespace) -> tuple[str, int, int | None]:
    """Map an HPC array index deterministically to one cell/session.

    把 HPC array 编号确定性映射到一个 cell/session：0..999 是 low，
    1000..1999 是 high。
    """

    explicit_any = arguments.cell is not None or arguments.session_index is not None
    if explicit_any and (
        arguments.cell is None or arguments.session_index is None
    ):
        raise ValueError("--cell and --session-index must be supplied together. / --cell 与 --session-index 必须一起提供。")

    environment_text = os.environ.get("SLURM_ARRAY_TASK_ID")
    environment_index: int | None = None
    if environment_text is not None:
        try:
            environment_index = _array_index(environment_text)
        except (TypeError, ValueError, argparse.ArgumentTypeError) as error:
            raise ValueError("SLURM_ARRAY_TASK_ID must be an integer in 0..1999. / SLURM_ARRAY_TASK_ID 必须是 0..1999 的整数。") from error

    if explicit_any and (
        arguments.array_index is not None or environment_index is not None
    ):
        raise ValueError(
            "Explicit cell/session is mutually exclusive with array selection. "
            "/ 明确 cell/session 与 array 选择互斥。"
        )
    if explicit_any:
        return arguments.cell, arguments.session_index, None

    if (
        arguments.array_index is not None
        and environment_index is not None
        and arguments.array_index != environment_index
    ):
        raise ValueError("CLI and SLURM array indices disagree. / CLI 与 SLURM array 编号不一致。")
    selected = (
        arguments.array_index
        if arguments.array_index is not None
        else environment_index
    )
    if selected is None:
        raise ValueError(
            "Choose --cell with --session-index, --array-index, or SLURM_ARRAY_TASK_ID. "
            "/ 请选择 cell+session、array-index，或设置 SLURM_ARRAY_TASK_ID。"
        )
    if selected < FORMAL_SESSION_COUNT:
        return "low", selected, selected
    return "high", selected - FORMAL_SESSION_COUNT, selected


def _add_artifact_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="formal family directory / 正式实验 family 目录",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Formal core low/high experiment runner / 正式核心低/高噪声实验 runner"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init_parser = commands.add_parser("init", help="freeze plans / 固定实验计划")
    _add_artifact_root(init_parser)

    run_parser = commands.add_parser(
        "run-session",
        help="run exactly one session / 运行恰好一个 session",
    )
    _add_artifact_root(run_parser)
    run_parser.add_argument("--cell", choices=("low", "high"))
    run_parser.add_argument("--session-index", type=_session_index)
    run_parser.add_argument(
        "--array-index",
        type=_array_index,
        help="HPC index: 0..999 low, 1000..1999 high / HPC 编号映射",
    )
    run_parser.add_argument(
        "--period-budget",
        type=_positive_int,
        default=None,
        help="operational periods for this invocation only; plan remains uncapped / 仅限本次调用的运行期数；正式计划仍无上限",
    )
    run_parser.add_argument(
        "--checkpoint-interval",
        type=_positive_int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="completed training periods between durable checkpoints / 两次持久 checkpoint 之间的训练期数",
    )
    run_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="explicitly retry a FAILED Step-36E task / 明确重试 FAILED 的 Step-36E 任务",
    )

    status_parser = commands.add_parser("status", help="count task states / 统计任务状态")
    _add_artifact_root(status_parser)
    status_parser.add_argument("--cell", choices=("low", "high", "all"), default="all")

    collect_parser = commands.add_parser(
        "collect",
        help="collect 1,000 completed sessions / 汇总 1,000 个已完成 session",
    )
    _add_artifact_root(collect_parser)
    collect_parser.add_argument("--cell", choices=("low", "high", "all"), default="all")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one transparent command. / 分发一个透明的命令。"""

    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    root = arguments.artifact_root.resolve()
    exit_status = EXIT_COMPLETE
    if arguments.command == "init":
        manifest = initialize_family(root)
        output: object = {
            "artifact_root": str(root),
            "family_manifest": str(root / FAMILY_MANIFEST_NAME),
            "family_manifest_sha256": manifest["family_manifest_sha256"],
            "cells": ("low", "high"),
            "sessions_per_cell": FORMAL_SESSION_COUNT,
            "research_result": False,
        }
    elif arguments.command == "run-session":
        try:
            cell, session_index, array_index = _resolve_run_target(arguments)
        except ValueError as error:
            parser.error(str(error))
        if array_index is not None and arguments.period_budget is None:
            parser.error(
                "Array workers require --period-budget so they checkpoint and "
                "exit before scheduler walltime. / array worker 必须提供 "
                "--period-budget，才能在调度器时限前 checkpoint 并退出。"
            )
        try:
            output = run_one_session(
                root,
                cell=cell,
                session_index=session_index,
                checkpoint_interval_periods=arguments.checkpoint_interval,
                invocation_period_budget=arguments.period_budget,
                retry_failed=arguments.retry_failed,
            )
            output["array_index"] = array_index
            status = output["status"]
            if status == COMPLETE:
                exit_status = EXIT_COMPLETE
            elif status == INCOMPLETE:
                exit_status = EXIT_INCOMPLETE
            elif status == FAILED:
                exit_status = EXIT_FAILED
            else:
                raise RuntimeError("Unexpected Step-36E status. / 未预期的 Step-36E 状态。")
        except SessionLockConflictError as error:
            output = {
                "status": "LOCK_CONFLICT",
                "error": str(error),
                "cell": cell,
                "session_index": session_index,
                "array_index": array_index,
            }
            exit_status = EXIT_LOCK_CONFLICT
        except CompleteMeasurementTaskError as error:
            scientific = error.status.scientific_outcome
            output = {
                "status": scientific.status,
                "phase": scientific.phase,
                "stop_reason": scientific.stop_reason,
                "cell": cell,
                "session_index": session_index,
                "array_index": array_index,
                "status_path": str(error.status_path),
            }
            exit_status = EXIT_FAILED
    elif arguments.command == "status":
        output = {
            cell: status_counts(root, cell)
            for cell in _cells_from_argument(arguments.cell)
        }
    elif arguments.command == "collect":
        output = {
            cell: collect_cell(root, cell)
            for cell in _cells_from_argument(arguments.cell)
        }
    else:  # pragma: no cover - argparse guarantees a known command.
        raise RuntimeError("Unknown command. / 未知命令。")

    print(_canonical_json(output, indent=2))
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
