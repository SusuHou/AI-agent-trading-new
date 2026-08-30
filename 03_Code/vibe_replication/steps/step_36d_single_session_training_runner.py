"""Step 36D: reliably train one planned session until a safe stopping point.

第 36D 步：可靠地训练一个已规划的 session，直到安全停止点。

Run the small two-invocation demonstration / 运行两次调用的小型演示:
    py -3 -X utf8 steps/step_36d_single_session_training_runner.py

This is an operational training runner, not a paper result. It can start or
resume one Step-36B task, save exact Step-36C checkpoints at global period
multiples while retaining only the newest two safe resume points, stop honestly
at a work/cap boundary, and time the work. If training
converges, it stops before measurement row zero and returns an in-memory
Step-35A handoff for Step 36E. / 这是训练运行器，不是论文结果。它可以启动或
恢复一个 Step-36B 任务，在全局时期倍数上保存 Step-36C checkpoint，同时只保留
最新两个安全恢复点；在工作量或训练上限处诚实停止，并记录速度。若训练收敛，
它会停在第 0 条测量记录之前，
并把 Step-35A 内存交接对象留给第 36E 步。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import isfinite
from numbers import Real
from pathlib import Path, PurePosixPath
from time import perf_counter
import json
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from step_25_one_market_period import build_paper_inputs
from step_26_reproducible_random_streams import (
    RandomizedMarketSession,
    build_randomized_paper_session,
)
from step_28_session_phases import (
    MeasurementSink,
    SessionPhase,
    SessionPhaseController,
)
from steps.step_35a_converged_market_checkpoint import (
    ConvergedMarketCheckpoint,
    LOADED_IMPLEMENTATION_TREE_SHA256,
    capture_at_convergence_boundary,
)
from steps.step_36b_experiment_manifest import (
    DEBUG_MODE,
    ExperimentCellConfig,
    ExperimentCellPlan,
    ExperimentExecutionPolicy,
    SessionTaskManifest,
    build_experiment_cell_plan,
    validate_experiment_cell_plan,
    validate_session_task_for_config,
)
from steps.step_36c_exact_training_resume import (
    NO_MEASUREMENT_SINK_PROTOCOL,
    MidTrainingCheckpoint,
    MeasurementSinkFactory,
    _atomic_binary_write,
    capture_mid_training_checkpoint,
    load_mid_training_checkpoint,
    restore_mid_training_controller,
    save_mid_training_checkpoint,
)
from src.parameters import PaperParameters


RUNNER_SCHEMA_VERSION = "step36d-one-session-training-runner-v2-formal-source-scope"
SCIENTIFIC_OUTCOME_SCHEMA_VERSION = "step36d-training-scientific-outcome-v2"
ATTEMPT_SCHEMA_VERSION = "step36d-training-attempt-v2"
STATUS_FILE_NAME = "training_status.json"
CHECKPOINT_DIRECTORY_NAME = "training_checkpoints"
# Two files are deliberate: after a newly verified checkpoint is written, the
# immediately preceding durable file remains available even if pruning itself
# is interrupted.  Step 36E may additionally protect the exact pre-convergence
# file named by its immutable origin evidence. / 故意保留两份：新 checkpoint
# 已写入并核对后，即使清理过程被中断，紧邻的上一份可靠文件仍然存在。
# Step 36E 还可以额外保护其不可变 origin 证据明确引用的收敛前文件。
TRAINING_CHECKPOINTS_TO_RETAIN = 2
CHECKPOINT_FILE_PATTERN = re.compile(
    r"^training_period_(?P<period>[0-9]{12})_"
    r"(?P<digest>[0-9a-f]{16})\.checkpoint$"
)

INCOMPLETE = "INCOMPLETE"
CONVERGED = "CONVERGED"
FAILED = "FAILED"
ALLOWED_TRAINING_STATUSES = (INCOMPLETE, CONVERGED, FAILED)

FRESH_START = "fresh"
RESUMED_START = "resumed"

Timer = Callable[[], float]
PeriodCompletedHook = Callable[[SessionPhaseController], None]


@dataclass(frozen=True)
class TrainingCheckpointReference:
    """Small audited pointer to one immutable Step-36C file.

    指向一个不可修改 Step-36C 文件的小型审计记录。
    """

    relative_path: str
    period_number: int
    checkpoint_sha256: str


ConvergenceHandoffHook = Callable[
    [
        SessionPhaseController,
        ConvergedMarketCheckpoint,
        TrainingCheckpointReference | None,
    ],
    ConvergedMarketCheckpoint | None,
]


@dataclass(frozen=True)
class TrainingScientificOutcome:
    """Deterministic scientific state; timing is deliberately elsewhere.

    确定性的科学状态；非确定性的运行时间故意放在别处。
    """

    schema_version: str
    status: str
    stop_reason: str
    task_id: str
    task_sha256: str
    run_config_sha256: str
    implementation_tree_sha256: str
    session_index: int
    measurement_sink_protocol_id: str
    verified_training_periods: int
    required_unchanged_periods: int
    ending_unchanged_periods: int
    policy_change_events: int
    policy_entries_changed: int
    last_policy_change_period_index: int | None
    latest_mid_training_checkpoint: TrainingCheckpointReference | None
    convergence_period_index: int | None
    converged_checkpoint_sha256: str | None
    ready_for_measurement: bool
    measurement_periods_completed: int
    durable_convergence_bundle_available: bool
    training_stage_only: bool
    research_result: bool
    paper_results_ready: bool
    outcome_sha256: str


@dataclass(frozen=True)
class TrainingAttemptMetadata:
    """Operational facts for one invocation. / 一次调用的运行信息。"""

    schema_version: str
    attempt_number: int
    plan_sha256: str
    start_mode: str
    input_checkpoint: TrainingCheckpointReference | None
    starting_training_period: int
    successful_periods_this_attempt: int
    ending_verified_period: int
    checkpoint_interval_periods: int
    checkpoint_periods_written: tuple[int, ...]
    absolute_training_cap: int | None
    invocation_period_budget: int | None
    elapsed_seconds: float
    periods_per_second: float | None
    failure_type: str | None
    failure_message: str | None
    live_state_discarded_after_failure: bool


@dataclass(frozen=True)
class PersistedTrainingStatus:
    """One checksum-protected status JSON. / 一份带校验码的状态 JSON。"""

    schema_version: str
    scientific_outcome: TrainingScientificOutcome
    attempt: TrainingAttemptMetadata
    status_sha256: str


@dataclass(frozen=True)
class TrainingRunExecution:
    """Runtime return value; live objects are never written into status JSON.

    运行时返回值；实时对象绝不会写入状态 JSON。
    """

    status: PersistedTrainingStatus
    controller: SessionPhaseController | None
    converged_checkpoint: ConvergedMarketCheckpoint | None
    status_path: Path


class TrainingTaskExecutionError(RuntimeError):
    """Unexpected execution failure with a persisted fail-closed status.

    意外运行失败，并携带已经落盘的 fail-closed 状态。
    """

    def __init__(self, status: PersistedTrainingStatus, status_path: Path) -> None:
        self.status = status
        self.status_path = status_path
        super().__init__(
            "Training task failed; only the last durable checkpoint is trusted. "
            "/ 训练任务失败；只信任最后一个已落盘 checkpoint。"
        )


def _json_ready(value: object) -> object:
    """Convert dataclasses and tuples to canonical JSON data. / 转成规范 JSON 数据。"""

    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _sha256_json(value: object) -> str:
    """Hash deterministic JSON. / 对确定性的 JSON 取 SHA-256。"""

    encoded = json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _scientific_without_checksum(
    outcome: TrainingScientificOutcome,
) -> dict[str, object]:
    data = asdict(outcome)
    data.pop("outcome_sha256")
    return data


def _status_without_checksum(
    status: PersistedTrainingStatus,
) -> dict[str, object]:
    data = asdict(status)
    data.pop("status_sha256")
    return data


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer. / {label} 必须是正整数。")
    return value


def _validate_sink_choice(
    protocol_id: str,
    factory: MeasurementSinkFactory | None,
) -> None:
    """Bind the lifetime-immutable Step-28 sink before period zero.

    在第 0 期之前绑定 Step-28 生命周期内不可替换的 sink。
    """

    if not isinstance(protocol_id, str) or not protocol_id or protocol_id.strip() != protocol_id:
        raise ValueError("Measurement-sink protocol is invalid. / measurement sink 协议无效。")
    if factory is None:
        if protocol_id != NO_MEASUREMENT_SINK_PROTOCOL:
            raise ValueError("A real measurement protocol requires a sink factory. / 真实测量协议需要 sink factory。")
    elif not callable(factory):
        raise TypeError("measurement_sink_factory must be callable. / measurement_sink_factory 必须可调用。")
    elif protocol_id == NO_MEASUREMENT_SINK_PROTOCOL:
        raise ValueError("A real sink cannot use the debug no-sink protocol. / 真实 sink 不能使用调试 no-sink 协议。")


def _task_artifact_directory(
    artifact_root: Path,
    task: SessionTaskManifest,
) -> Path:
    """Resolve the task directory and prove it stays below artifact_root.

    解析任务目录，并证明它仍位于 artifact_root 之下。
    """

    if not isinstance(artifact_root, Path):
        raise TypeError("artifact_root must be pathlib.Path. / artifact_root 必须是 pathlib.Path。")
    relative = PurePosixPath(task.relative_artifact_directory)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError("Task artifact path is unsafe. / 任务产物路径不安全。")
    root = artifact_root.resolve()
    target = root.joinpath(*relative.parts).resolve()
    if target == root or root not in target.parents:
        raise ValueError("Task artifact path escapes artifact_root. / 任务产物路径越出 artifact_root。")
    return target


def build_fresh_training_controller(
    config: ExperimentCellConfig,
    task: SessionTaskManifest,
    *,
    measurement_sink_protocol_id: str = NO_MEASUREMENT_SINK_PROTOCOL,
    measurement_sink_factory: MeasurementSinkFactory | None = None,
) -> SessionPhaseController:
    """Build period-zero market objects only from the checked config/task.

    只根据已核对的 config/task 建立第 0 期市场对象。
    """

    validate_session_task_for_config(task, config)
    _validate_sink_choice(measurement_sink_protocol_id, measurement_sink_factory)
    value_grid, price_grid, actions, initial_q, prehistory = build_paper_inputs(
        config.parameters
    )
    session = build_randomized_paper_session(
        parameters=config.parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=actions,
        initial_q_table=initial_q,
        prehistory=prehistory,
        experiment_seed=task.seed_manifest.experiment_seed,
        experiment_cell_key=task.seed_manifest.experiment_cell_key,
        session_index=task.session_index,
    )
    measurement_sink: MeasurementSink | None = (
        None
        if measurement_sink_factory is None
        else measurement_sink_factory(session)
    )
    return SessionPhaseController.create_for_fresh_session(
        session,
        convergence_periods_required=config.convergence_periods_required,
        measurement_periods_required=config.measurement_periods_required,
        measurement_sink=measurement_sink,
    )


def _checkpoint_path(
    checkpoint_directory: Path,
    checkpoint: MidTrainingCheckpoint,
) -> Path:
    period = checkpoint.payload.period_number
    return checkpoint_directory / (
        f"training_period_{period:012d}_"
        f"{checkpoint.checkpoint_sha256[:16]}.checkpoint"
    )


def _reference_for_checkpoint(
    checkpoint: MidTrainingCheckpoint,
    path: Path,
    artifact_root: Path,
) -> TrainingCheckpointReference:
    relative_path = path.resolve().relative_to(artifact_root.resolve()).as_posix()
    return TrainingCheckpointReference(
        relative_path=relative_path,
        period_number=checkpoint.payload.period_number,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
    )


def _save_verified_training_checkpoint(
    controller: SessionPhaseController,
    *,
    task: SessionTaskManifest,
    config: ExperimentCellConfig,
    artifact_root: Path,
    checkpoint_directory: Path,
    measurement_sink_protocol_id: str,
) -> tuple[MidTrainingCheckpoint, TrainingCheckpointReference]:
    """Capture, save, reload, and compare before declaring durability.

    保存后重新读取并比较，之后才声明 checkpoint 已可靠落盘。
    """

    checkpoint = capture_mid_training_checkpoint(
        controller,
        task=task,
        expected_config=config,
        measurement_sink_protocol_id=measurement_sink_protocol_id,
    )
    path = _checkpoint_path(checkpoint_directory, checkpoint)
    save_mid_training_checkpoint(
        checkpoint,
        path,
        expected_task=task,
        expected_config=config,
        expected_measurement_sink_protocol_id=measurement_sink_protocol_id,
    )
    loaded = load_mid_training_checkpoint(
        path,
        expected_task=task,
        expected_config=config,
        expected_measurement_sink_protocol_id=measurement_sink_protocol_id,
        trusted_local_file=True,
    )
    if loaded != checkpoint:
        raise RuntimeError("Saved checkpoint does not replay exactly. / 已保存 checkpoint 无法精确重放。")
    reference = _reference_for_checkpoint(
        checkpoint,
        path,
        artifact_root,
    )
    prune_training_checkpoints(checkpoint_directory)
    return checkpoint, reference


def _parsed_checkpoint_paths(
    checkpoint_directory: Path,
) -> list[tuple[int, str, Path]]:
    """Parse every managed checkpoint name, failing closed on surprises.

    解析全部受管理的 checkpoint 文件名；遇到异常文件时拒绝继续清理。
    """

    if not isinstance(checkpoint_directory, Path):
        raise TypeError(
            "checkpoint_directory must be pathlib.Path. / "
            "checkpoint_directory 必须是 pathlib.Path。"
        )
    if not checkpoint_directory.exists():
        return []
    resolved_directory = checkpoint_directory.resolve()
    parsed: list[tuple[int, str, Path]] = []
    for path in checkpoint_directory.glob("training_period_*.checkpoint"):
        if path.is_symlink():
            raise ValueError(
                "A managed checkpoint cannot be a symbolic link. / "
                "受管理的 checkpoint 不能是符号链接。"
            )
        resolved_path = path.resolve()
        if resolved_path.parent != resolved_directory or not resolved_path.is_file():
            raise ValueError(
                "A checkpoint path escapes its managed directory. / "
                "某个 checkpoint 路径越出受管理目录。"
            )
        match = CHECKPOINT_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(
                "A checkpoint filename is malformed. / 某个 checkpoint 文件名错误。"
            )
        parsed.append(
            (int(match.group("period")), match.group("digest"), resolved_path)
        )
    return parsed


def prune_training_checkpoints(
    checkpoint_directory: Path,
    *,
    protected_checkpoint_paths: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    """Keep the newest two checkpoints plus explicitly protected evidence.

    保留最新两份 checkpoint，以及证据明确要求保护的额外文件。

    The function only removes files with the runner's exact managed filename
    pattern.  A protected path must already exist inside this exact directory;
    otherwise cleanup fails closed instead of silently losing provenance. / 此
    函数只删除完全符合 runner 命名规则的文件。受保护路径必须已经存在于这个
    精确目录内；否则清理会安全失败，而不会悄悄丢失来源证据。
    """

    if not isinstance(protected_checkpoint_paths, tuple) or any(
        not isinstance(path, Path) for path in protected_checkpoint_paths
    ):
        raise TypeError(
            "protected_checkpoint_paths must be a tuple of pathlib.Path. / "
            "protected_checkpoint_paths 必须是 pathlib.Path 元组。"
        )
    parsed = _parsed_checkpoint_paths(checkpoint_directory)
    if not parsed:
        if protected_checkpoint_paths:
            raise FileNotFoundError(
                "Protected checkpoint evidence is missing. / 受保护 checkpoint 证据丢失。"
            )
        return ()

    # Duplicate periods are scientifically ambiguous: do not choose one merely
    # by filename or deletion order. / 同一期出现多份文件在科学上含糊，不能按
    # 文件名或删除顺序擅自挑选。
    periods = [period for period, _, _ in parsed]
    if len(periods) != len(set(periods)):
        raise ValueError(
            "Several checkpoints claim the same period. / 多个 checkpoint 声称同一时期。"
        )

    resolved_directory = checkpoint_directory.resolve()
    available_paths = {path for _, _, path in parsed}
    protected: set[Path] = set()
    for candidate in protected_checkpoint_paths:
        resolved = candidate.resolve()
        if resolved.parent != resolved_directory or resolved not in available_paths:
            raise FileNotFoundError(
                "Protected checkpoint evidence is outside the directory or missing. / "
                "受保护 checkpoint 证据不在目标目录内或已经丢失。"
            )
        protected.add(resolved)

    newest = {
        path
        for _, _, path in sorted(parsed, key=lambda item: item[0])[
            -TRAINING_CHECKPOINTS_TO_RETAIN:
        ]
    }
    retained = newest | protected
    for _, _, obsolete_path in sorted(parsed, key=lambda item: item[0]):
        if obsolete_path not in retained:
            obsolete_path.unlink()
    return tuple(
        path for _, _, path in sorted(parsed, key=lambda item: item[0]) if path in retained
    )


def _discover_latest_checkpoint(
    checkpoint_directory: Path,
    *,
    task: SessionTaskManifest,
    config: ExperimentCellConfig,
    artifact_root: Path,
    measurement_sink_protocol_id: str,
) -> tuple[MidTrainingCheckpoint, TrainingCheckpointReference] | None:
    """Find the greatest global-period checkpoint, then validate its contents.

    找到全局时期最大的 checkpoint，并验证其真实内容。
    """

    if not checkpoint_directory.exists():
        return None
    parsed = _parsed_checkpoint_paths(checkpoint_directory)
    if not parsed:
        return None
    greatest_period = max(item[0] for item in parsed)
    greatest = [item for item in parsed if item[0] == greatest_period]
    if len(greatest) != 1:
        raise ValueError("Several checkpoints claim the same latest period. / 多个 checkpoint 声称同一个最新时期。")
    claimed_period, claimed_digest, path = greatest[0]
    checkpoint = load_mid_training_checkpoint(
        path,
        expected_task=task,
        expected_config=config,
        expected_measurement_sink_protocol_id=measurement_sink_protocol_id,
        trusted_local_file=True,
    )
    if (
        checkpoint.payload.period_number != claimed_period
        or checkpoint.checkpoint_sha256[:16] != claimed_digest
    ):
        raise ValueError("Checkpoint filename and payload disagree. / checkpoint 文件名与内容不一致。")
    return checkpoint, _reference_for_checkpoint(checkpoint, path, artifact_root)


def _build_scientific_outcome(
    *,
    status: str,
    stop_reason: str,
    task: SessionTaskManifest,
    protocol_id: str,
    durable_periods: int,
    required_unchanged_periods: int,
    ending_unchanged_periods: int,
    policy_change_events: int,
    policy_entries_changed: int,
    last_policy_change_period_index: int | None,
    latest_checkpoint: TrainingCheckpointReference | None,
    convergence_period_index: int | None,
    converged_checkpoint_sha256: str | None,
) -> TrainingScientificOutcome:
    if status not in ALLOWED_TRAINING_STATUSES:
        raise ValueError("Unknown training status. / 未知训练状态。")
    converged = status == CONVERGED
    draft = TrainingScientificOutcome(
        schema_version=SCIENTIFIC_OUTCOME_SCHEMA_VERSION,
        status=status,
        stop_reason=stop_reason,
        task_id=task.task_id,
        task_sha256=task.task_sha256,
        run_config_sha256=task.run_config_sha256,
        implementation_tree_sha256=task.implementation_tree_sha256,
        session_index=task.session_index,
        measurement_sink_protocol_id=protocol_id,
        verified_training_periods=durable_periods,
        required_unchanged_periods=required_unchanged_periods,
        ending_unchanged_periods=ending_unchanged_periods,
        policy_change_events=policy_change_events,
        policy_entries_changed=policy_entries_changed,
        last_policy_change_period_index=last_policy_change_period_index,
        latest_mid_training_checkpoint=latest_checkpoint,
        convergence_period_index=convergence_period_index,
        converged_checkpoint_sha256=converged_checkpoint_sha256,
        ready_for_measurement=converged,
        measurement_periods_completed=0,
        durable_convergence_bundle_available=False,
        training_stage_only=True,
        research_result=False,
        paper_results_ready=False,
        outcome_sha256="",
    )
    return replace(
        draft,
        outcome_sha256=_sha256_json(_scientific_without_checksum(draft)),
    )


def _build_status(
    scientific_outcome: TrainingScientificOutcome,
    attempt: TrainingAttemptMetadata,
) -> PersistedTrainingStatus:
    draft = PersistedTrainingStatus(
        schema_version=RUNNER_SCHEMA_VERSION,
        scientific_outcome=scientific_outcome,
        attempt=attempt,
        status_sha256="",
    )
    return replace(
        draft,
        status_sha256=_sha256_json(_status_without_checksum(draft)),
    )


def _checkpoint_reference_from_dictionary(
    data: object,
) -> TrainingCheckpointReference | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("Checkpoint reference is malformed. / checkpoint 引用格式错误。")
    return TrainingCheckpointReference(**data)


def _status_from_dictionary(data: object) -> PersistedTrainingStatus:
    if not isinstance(data, dict):
        raise ValueError("Training status must be a JSON object. / 训练状态必须是 JSON 对象。")
    try:
        scientific_data = dict(data["scientific_outcome"])
        scientific_data["latest_mid_training_checkpoint"] = (
            _checkpoint_reference_from_dictionary(
                scientific_data["latest_mid_training_checkpoint"]
            )
        )
        scientific = TrainingScientificOutcome(**scientific_data)
        attempt_data = dict(data["attempt"])
        attempt_data["input_checkpoint"] = _checkpoint_reference_from_dictionary(
            attempt_data["input_checkpoint"]
        )
        attempt_data["checkpoint_periods_written"] = tuple(
            attempt_data["checkpoint_periods_written"]
        )
        attempt = TrainingAttemptMetadata(**attempt_data)
        return PersistedTrainingStatus(
            schema_version=data["schema_version"],
            scientific_outcome=scientific,
            attempt=attempt,
            status_sha256=data["status_sha256"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Training status schema is malformed. / 训练状态结构错误。") from error


def validate_training_status(
    status: PersistedTrainingStatus,
    *,
    expected_task: SessionTaskManifest,
    expected_config: ExperimentCellConfig,
) -> None:
    """Validate checksums, identity, status logic, and finite timing.

    验证校验码、身份、状态逻辑与有限的计时数据。
    """

    validate_session_task_for_config(expected_task, expected_config)
    if not isinstance(status, PersistedTrainingStatus):
        raise TypeError("status has the wrong type. / status 类型错误。")
    if status.schema_version != RUNNER_SCHEMA_VERSION:
        raise ValueError("Training status schema is unsupported. / 训练状态格式不支持。")
    scientific = status.scientific_outcome
    attempt = status.attempt
    if scientific.schema_version != SCIENTIFIC_OUTCOME_SCHEMA_VERSION:
        raise ValueError("Scientific outcome schema is unsupported. / 科学状态格式不支持。")
    if attempt.schema_version != ATTEMPT_SCHEMA_VERSION:
        raise ValueError("Attempt schema is unsupported. / 调用状态格式不支持。")
    if _sha256_json(_scientific_without_checksum(scientific)) != scientific.outcome_sha256:
        raise ValueError("Scientific outcome checksum failed. / 科学状态校验失败。")
    if _sha256_json(_status_without_checksum(status)) != status.status_sha256:
        raise ValueError("Training status checksum failed. / 训练状态校验失败。")
    if (
        scientific.task_id != expected_task.task_id
        or scientific.task_sha256 != expected_task.task_sha256
        or scientific.run_config_sha256 != expected_task.run_config_sha256
        or scientific.implementation_tree_sha256 != LOADED_IMPLEMENTATION_TREE_SHA256
        or scientific.session_index != expected_task.session_index
    ):
        raise ValueError("Training status belongs to another task/build. / 训练状态属于另一个任务或源码版本。")
    if scientific.status not in ALLOWED_TRAINING_STATUSES:
        raise ValueError("Training status value is invalid. / 训练状态值无效。")
    integer_values = (
        scientific.verified_training_periods,
        scientific.required_unchanged_periods,
        scientific.ending_unchanged_periods,
        scientific.policy_change_events,
        scientific.policy_entries_changed,
        attempt.attempt_number,
        attempt.starting_training_period,
        attempt.successful_periods_this_attempt,
        attempt.ending_verified_period,
        attempt.checkpoint_interval_periods,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in integer_values):
        raise ValueError("A training status counter is invalid. / 某个训练状态计数无效。")
    if scientific.required_unchanged_periods != expected_config.convergence_periods_required:
        raise ValueError("Status convergence target differs from config. / 状态收敛目标与 config 不同。")
    if attempt.checkpoint_interval_periods < 1 or attempt.attempt_number < 1:
        raise ValueError("Attempt counters must be positive. / 调用计数必须为正。")
    if attempt.start_mode not in (FRESH_START, RESUMED_START):
        raise ValueError("Attempt start mode is invalid. / 调用起点模式无效。")
    if attempt.ending_verified_period != scientific.verified_training_periods:
        raise ValueError("Attempt and scientific durable periods differ. / 调用与科学状态的可靠时期不同。")
    elapsed = attempt.elapsed_seconds
    if isinstance(elapsed, bool) or not isinstance(elapsed, Real) or not isfinite(float(elapsed)) or elapsed < 0:
        raise ValueError("Elapsed time must be finite and non-negative. / 运行时间必须有限且非负。")
    if attempt.periods_per_second is not None:
        rate = attempt.periods_per_second
        if isinstance(rate, bool) or not isinstance(rate, Real) or not isfinite(float(rate)) or rate < 0:
            raise ValueError("Throughput must be finite and non-negative. / 吞吐速度必须有限且非负。")
    reference = scientific.latest_mid_training_checkpoint
    if reference is not None:
        if reference.period_number > scientific.verified_training_periods:
            raise ValueError("Checkpoint lies after the durable state. / checkpoint 晚于可靠状态。")
        if not re.fullmatch(r"[0-9a-f]{64}", reference.checkpoint_sha256):
            raise ValueError("Checkpoint digest is invalid. / checkpoint 摘要无效。")
    if scientific.status == CONVERGED:
        if (
            not scientific.ready_for_measurement
            or scientific.convergence_period_index is None
            or scientific.converged_checkpoint_sha256 is None
            or scientific.verified_training_periods < 1
        ):
            raise ValueError("Converged status lacks convergence evidence. / 收敛状态缺少证据。")
    else:
        if (
            scientific.ready_for_measurement
            or scientific.convergence_period_index is not None
            or scientific.converged_checkpoint_sha256 is not None
        ):
            raise ValueError("Non-converged status claims convergence. / 未收敛状态错误声称收敛。")
    if scientific.measurement_periods_completed != 0:
        raise ValueError("Step 36D cannot contain measurement rows. / 第 36D 步不能含测量记录。")
    if any((scientific.durable_convergence_bundle_available, scientific.research_result, scientific.paper_results_ready)):
        raise ValueError("Step 36D honesty flags are invalid. / 第 36D 步诚实标记无效。")
    failed = scientific.status == FAILED
    if failed != (attempt.failure_type is not None):
        raise ValueError("Failure status and metadata disagree. / 失败状态与元数据不一致。")
    if failed != attempt.live_state_discarded_after_failure:
        raise ValueError("Failure discard flag is inconsistent. / 失败丢弃标记不一致。")


def save_training_status(
    status: PersistedTrainingStatus,
    path: Path,
    *,
    expected_task: SessionTaskManifest,
    expected_config: ExperimentCellConfig,
) -> Path:
    """Atomically replace the operational status JSON. / 原子替换运行状态 JSON。"""

    validate_training_status(
        status,
        expected_task=expected_task,
        expected_config=expected_config,
    )
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path. / path 必须是 pathlib.Path。")
    text = json.dumps(
        _json_ready(status),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _atomic_binary_write(path, text.encode("utf-8"))
    return path


def load_training_status(
    path: Path,
    *,
    expected_task: SessionTaskManifest,
    expected_config: ExperimentCellConfig,
) -> PersistedTrainingStatus:
    """Load and validate one status file. / 读取并验证一份状态文件。"""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path. / path 必须是 pathlib.Path。")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read a complete training status. / 无法读取完整训练状态。") from error
    status = _status_from_dictionary(data)
    validate_training_status(
        status,
        expected_task=expected_task,
        expected_config=expected_config,
    )
    return status


def _read_timer(timer: Timer) -> float:
    value = timer()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("timer must return a real number. / timer 必须返回实数。")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError("timer must return a finite number. / timer 必须返回有限数。")
    return converted


def _attempt_metadata(
    *,
    attempt_number: int,
    plan: ExperimentCellPlan,
    start_mode: str,
    input_checkpoint: TrainingCheckpointReference | None,
    starting_period: int,
    successful_periods: int,
    ending_durable_period: int,
    checkpoint_interval: int,
    checkpoint_periods_written: tuple[int, ...],
    invocation_period_budget: int | None,
    elapsed_seconds: float,
    failure: Exception | None,
) -> TrainingAttemptMetadata:
    rate = (
        successful_periods / elapsed_seconds
        if successful_periods > 0 and elapsed_seconds > 0.0
        else None
    )
    return TrainingAttemptMetadata(
        schema_version=ATTEMPT_SCHEMA_VERSION,
        attempt_number=attempt_number,
        plan_sha256=plan.plan_sha256,
        start_mode=start_mode,
        input_checkpoint=input_checkpoint,
        starting_training_period=starting_period,
        successful_periods_this_attempt=successful_periods,
        ending_verified_period=ending_durable_period,
        checkpoint_interval_periods=checkpoint_interval,
        checkpoint_periods_written=checkpoint_periods_written,
        absolute_training_cap=plan.execution_policy.maximum_training_periods,
        invocation_period_budget=invocation_period_budget,
        elapsed_seconds=elapsed_seconds,
        periods_per_second=rate,
        failure_type=(None if failure is None else type(failure).__name__),
        failure_message=(None if failure is None else str(failure)[:1000]),
        live_state_discarded_after_failure=(failure is not None),
    )


def run_training_task(
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    *,
    artifact_root: Path,
    checkpoint_interval_periods: int,
    invocation_period_budget: int | None = None,
    measurement_sink_protocol_id: str = NO_MEASUREMENT_SINK_PROTOCOL,
    measurement_sink_factory: MeasurementSinkFactory | None = None,
    retry_failed: bool = False,
    timer: Timer = perf_counter,
    period_completed_hook: PeriodCompletedHook | None = None,
    convergence_handoff_hook: ConvergenceHandoffHook | None = None,
) -> TrainingRunExecution:
    """Start/resume one task and run only its Q-learning phase.

    启动或恢复一个任务，并且只运行它的 Q-learning 阶段。
    """

    validate_experiment_cell_plan(plan)
    if not isinstance(task, SessionTaskManifest) or task not in plan.tasks:
        raise ValueError("task is not a member of this experiment plan. / task 不属于该实验计划。")
    validate_session_task_for_config(task, plan.config)
    checkpoint_interval = _positive_integer(
        checkpoint_interval_periods,
        "checkpoint_interval_periods",
    )
    if invocation_period_budget is not None:
        invocation_period_budget = _positive_integer(
            invocation_period_budget,
            "invocation_period_budget",
        )
    if not isinstance(retry_failed, bool):
        raise TypeError("retry_failed must be bool. / retry_failed 必须是 bool。")
    if not callable(timer):
        raise TypeError("timer must be callable. / timer 必须可调用。")
    if period_completed_hook is not None and not callable(period_completed_hook):
        raise TypeError("period_completed_hook must be callable. / period_completed_hook 必须可调用。")
    if convergence_handoff_hook is not None and not callable(
        convergence_handoff_hook
    ):
        raise TypeError(
            "convergence_handoff_hook must be callable. / "
            "convergence_handoff_hook 必须可调用。"
        )
    _validate_sink_choice(measurement_sink_protocol_id, measurement_sink_factory)

    task_directory = _task_artifact_directory(artifact_root, task)
    checkpoint_directory = task_directory / CHECKPOINT_DIRECTORY_NAME
    status_path = task_directory / STATUS_FILE_NAME
    existing_status = (
        load_training_status(
            status_path,
            expected_task=task,
            expected_config=plan.config,
        )
        if status_path.exists()
        else None
    )
    if existing_status is not None:
        existing_scientific = existing_status.scientific_outcome
        if existing_scientific.measurement_sink_protocol_id != measurement_sink_protocol_id:
            raise ValueError("Existing status uses another measurement protocol. / 现有状态使用另一 measurement 协议。")
        if existing_scientific.status == CONVERGED:
            return TrainingRunExecution(
                status=existing_status,
                controller=None,
                converged_checkpoint=None,
                status_path=status_path,
            )
        if existing_scientific.status == FAILED and not retry_failed:
            return TrainingRunExecution(
                status=existing_status,
                controller=None,
                converged_checkpoint=None,
                status_path=status_path,
            )

    started_at = _read_timer(timer)
    discovered = _discover_latest_checkpoint(
        checkpoint_directory,
        task=task,
        config=plan.config,
        artifact_root=artifact_root,
        measurement_sink_protocol_id=measurement_sink_protocol_id,
    )
    if discovered is None:
        input_checkpoint = None
        latest_reference = None
        controller = build_fresh_training_controller(
            plan.config,
            task,
            measurement_sink_protocol_id=measurement_sink_protocol_id,
            measurement_sink_factory=measurement_sink_factory,
        )
        start_mode = FRESH_START
    else:
        input_checkpoint, latest_reference = discovered
        controller = restore_mid_training_controller(
            input_checkpoint,
            expected_task=task,
            expected_config=plan.config,
            expected_measurement_sink_protocol_id=measurement_sink_protocol_id,
            measurement_sink_factory=measurement_sink_factory,
        )
        start_mode = RESUMED_START

    starting_period = controller.training_periods_completed
    input_reference = latest_reference
    if (
        existing_status is not None
        and existing_status.scientific_outcome.latest_mid_training_checkpoint is not None
        and (
            latest_reference is None
            or latest_reference.period_number
            < existing_status.scientific_outcome.latest_mid_training_checkpoint.period_number
        )
    ):
        raise FileNotFoundError("The status points to a newer missing checkpoint. / 状态指向一个更新但丢失的 checkpoint。")

    successful_periods = 0
    checkpoint_periods_written: list[int] = []
    converged_checkpoint: ConvergedMarketCheckpoint | None = None
    stop_status: str | None = None
    stop_reason: str | None = None
    absolute_cap = plan.execution_policy.maximum_training_periods
    attempt_number = (
        1 if existing_status is None else existing_status.attempt.attempt_number + 1
    )

    try:
        while controller.phase is SessionPhase.TRAINING:
            if (
                absolute_cap is not None
                and controller.training_periods_completed >= absolute_cap
            ):
                stop_status = INCOMPLETE
                stop_reason = "absolute_training_cap_reached"
                break
            if (
                invocation_period_budget is not None
                and successful_periods >= invocation_period_budget
            ):
                stop_status = INCOMPLETE
                stop_reason = "invocation_period_budget_reached"
                break

            controller.run_next_period()
            successful_periods += 1
            if period_completed_hook is not None:
                period_completed_hook(controller)

            if controller.phase is SessionPhase.MEASUREMENT:
                if controller.measurement_periods_completed != 0:
                    raise RuntimeError("Step 36D accidentally ran a measurement row. / 第 36D 步意外运行了测量记录。")
                converged_checkpoint = capture_at_convergence_boundary(controller)
                payload = converged_checkpoint.payload
                if (
                    payload.seed_manifest != task.seed_manifest
                    or payload.parameters != plan.config.parameters
                    or payload.convergence_receipt.required_unchanged_periods
                    != plan.config.convergence_periods_required
                ):
                    raise RuntimeError("Convergence handoff identity is inconsistent. / 收敛交接身份不一致。")
                # Step 36E uses this before the terminal CONVERGED status is
                # published. If durable origin persistence fails, the normal
                # failure path trusts only the last mid-training checkpoint.
                # / Step 36E 会在发布 CONVERGED 终态前使用此钩子。
                # 若收敛原点落盘失败，普通失败路径只信最后一个
                # 训练中 checkpoint。
                if convergence_handoff_hook is not None:
                    replacement_handoff = convergence_handoff_hook(
                        controller,
                        converged_checkpoint,
                        latest_reference,
                    )
                    if replacement_handoff is not None:
                        if not isinstance(
                            replacement_handoff,
                            ConvergedMarketCheckpoint,
                        ):
                            raise TypeError(
                                "The convergence hook returned the wrong type. / "
                                "收敛钩子返回了错误类型。"
                            )
                        if replacement_handoff != converged_checkpoint:
                            raise RuntimeError(
                                "The convergence hook changed the scientific "
                                "handoff. / 收敛钩子改变了科学交接内容。"
                            )
                        # Step 35C deliberately binds downstream provenance to
                        # this exact object identity. / Step 35C 故意把下游
                        # 来源绑定在这个精确对象身份上。
                        converged_checkpoint = replacement_handoff
                stop_status = CONVERGED
                stop_reason = "policy_convergence_reached"
                break

            global_period = controller.training_periods_completed
            if global_period % checkpoint_interval == 0:
                _, latest_reference = _save_verified_training_checkpoint(
                    controller,
                    task=task,
                    config=plan.config,
                    artifact_root=artifact_root,
                    checkpoint_directory=checkpoint_directory,
                    measurement_sink_protocol_id=measurement_sink_protocol_id,
                )
                checkpoint_periods_written.append(global_period)

        if stop_status is None or stop_reason is None:
            raise RuntimeError("Training runner stopped without a classified outcome. / 训练 runner 停止但没有分类。")

        if stop_status == INCOMPLETE:
            current_period = controller.training_periods_completed
            if latest_reference is None or latest_reference.period_number != current_period:
                _, latest_reference = _save_verified_training_checkpoint(
                    controller,
                    task=task,
                    config=plan.config,
                    artifact_root=artifact_root,
                    checkpoint_directory=checkpoint_directory,
                    measurement_sink_protocol_id=measurement_sink_protocol_id,
                )
                checkpoint_periods_written.append(current_period)

        ended_at = _read_timer(timer)
        if ended_at < started_at:
            raise ValueError("timer moved backwards. / timer 倒退。")
        elapsed = ended_at - started_at
        tracker = controller.tracker
        if stop_status == CONVERGED:
            if tracker.convergence_receipt is None or converged_checkpoint is None:
                raise RuntimeError("Convergence receipt is missing. / 收敛记录丢失。")
            durable_period = controller.training_periods_completed
            convergence_period_index = (
                tracker.convergence_receipt.convergence_period_index
            )
            convergence_digest = converged_checkpoint.checkpoint_sha256
        else:
            if latest_reference is None:
                raise RuntimeError("Incomplete work lacks a durable checkpoint. / 未完成工作缺少可靠 checkpoint。")
            durable_period = latest_reference.period_number
            convergence_period_index = None
            convergence_digest = None

        scientific = _build_scientific_outcome(
            status=stop_status,
            stop_reason=stop_reason,
            task=task,
            protocol_id=measurement_sink_protocol_id,
            durable_periods=durable_period,
            required_unchanged_periods=tracker.required_unchanged_periods,
            ending_unchanged_periods=tracker.unchanged_periods,
            policy_change_events=tracker.policy_change_events,
            policy_entries_changed=tracker.policy_entries_changed,
            last_policy_change_period_index=tracker.last_policy_change_period_index,
            latest_checkpoint=latest_reference,
            convergence_period_index=convergence_period_index,
            converged_checkpoint_sha256=convergence_digest,
        )
        attempt = _attempt_metadata(
            attempt_number=attempt_number,
            plan=plan,
            start_mode=start_mode,
            input_checkpoint=input_reference,
            starting_period=starting_period,
            successful_periods=successful_periods,
            ending_durable_period=durable_period,
            checkpoint_interval=checkpoint_interval,
            checkpoint_periods_written=tuple(checkpoint_periods_written),
            invocation_period_budget=invocation_period_budget,
            elapsed_seconds=elapsed,
            failure=None,
        )
        persisted = _build_status(scientific, attempt)
        save_training_status(
            persisted,
            status_path,
            expected_task=task,
            expected_config=plan.config,
        )
        return TrainingRunExecution(
            status=persisted,
            controller=controller,
            converged_checkpoint=converged_checkpoint,
            status_path=status_path,
        )

    except Exception as error:
        # Never capture the live object here: a late kernel/observer exception
        # may leave its Q, maker, RNG, and scalar clocks between states. / 这里
        # 绝不保存实时对象：kernel/observer 的晚期异常可能让 Q、做市商、随机流
        # 与标量时钟处在不一致的中间状态。
        ended_at = _read_timer(timer)
        elapsed = max(0.0, ended_at - started_at)
        durable_period = (
            0 if latest_reference is None else latest_reference.period_number
        )
        if latest_reference is None:
            required = plan.config.convergence_periods_required
            unchanged = 0
            changes = 0
            entries_changed = 0
            last_change = None
        else:
            state = input_checkpoint.payload.tracker_state if discovered is not None else None
            # If this attempt wrote a newer checkpoint, load its tracker state.
            if latest_reference is not input_reference:
                checkpoint_path = artifact_root / PurePosixPath(
                    latest_reference.relative_path
                )
                durable_checkpoint = load_mid_training_checkpoint(
                    checkpoint_path,
                    expected_task=task,
                    expected_config=plan.config,
                    expected_measurement_sink_protocol_id=measurement_sink_protocol_id,
                    trusted_local_file=True,
                )
                state = durable_checkpoint.payload.tracker_state
            if state is None:
                raise RuntimeError("Durable tracker state is missing. / 可靠 tracker 状态丢失。") from error
            required = state.required_unchanged_periods
            unchanged = state.unchanged_periods
            changes = state.policy_change_events
            entries_changed = state.policy_entries_changed
            last_change = state.last_policy_change_period_index
        failed_scientific = _build_scientific_outcome(
            status=FAILED,
            stop_reason="unexpected_exception",
            task=task,
            protocol_id=measurement_sink_protocol_id,
            durable_periods=durable_period,
            required_unchanged_periods=required,
            ending_unchanged_periods=unchanged,
            policy_change_events=changes,
            policy_entries_changed=entries_changed,
            last_policy_change_period_index=last_change,
            latest_checkpoint=latest_reference,
            convergence_period_index=None,
            converged_checkpoint_sha256=None,
        )
        failed_attempt = _attempt_metadata(
            attempt_number=attempt_number,
            plan=plan,
            start_mode=start_mode,
            input_checkpoint=input_reference,
            starting_period=starting_period,
            successful_periods=successful_periods,
            ending_durable_period=durable_period,
            checkpoint_interval=checkpoint_interval,
            checkpoint_periods_written=tuple(checkpoint_periods_written),
            invocation_period_budget=invocation_period_budget,
            elapsed_seconds=elapsed,
            failure=error,
        )
        failed_status = _build_status(failed_scientific, failed_attempt)
        save_training_status(
            failed_status,
            status_path,
            expected_task=task,
            expected_config=plan.config,
        )
        raise TrainingTaskExecutionError(failed_status, status_path) from error


def main() -> None:
    """Demonstrate an incomplete first invocation and an exact resume.

    演示第一次未完成，以及随后精确恢复。
    """

    parameters = PaperParameters(noise_std=0.1, market_maker_window=20)
    config = ExperimentCellConfig(
        mode=DEBUG_MODE,
        experiment_cell_key="step36d-debug-low-noise",
        parameters=parameters,
        experiment_seed=36_000_001,
        irf_experiment_seed=36_000_002,
        session_count=1,
        convergence_periods_required=1,
        measurement_periods_required=3,
        irf_paths_per_session=1,
    )
    plan = build_experiment_cell_plan(
        config,
        ExperimentExecutionPolicy(maximum_training_periods=20),
    )
    task = plan.tasks[0]
    artifact_root = PROJECT_ROOT / "results" / "step36d_single_session_training_runner"

    first = run_training_task(
        plan,
        task,
        artifact_root=artifact_root,
        checkpoint_interval_periods=2,
        invocation_period_budget=1,
    )
    second = run_training_task(
        plan,
        task,
        artifact_root=artifact_root,
        checkpoint_interval_periods=2,
        invocation_period_budget=19,
    )

    print("Step 36D: one-session training runner / 第 36D 步：单 session 训练 runner")
    print(
        "First invocation / 第一次调用: "
        f"{first.status.scientific_outcome.status}, "
        f"period={first.status.scientific_outcome.verified_training_periods}"
    )
    print(
        "Second invocation / 第二次调用: "
        f"{second.status.scientific_outcome.status}, "
        f"period={second.status.scientific_outcome.verified_training_periods}"
    )
    print(
        "Periods/second in last invocation / 最后一次调用每秒时期数: "
        f"{second.status.attempt.periods_per_second}"
    )
    print(f"Status JSON / 状态文件: {second.status_path}")
    print(
        "Boundary / 边界: zero measurement rows; in-memory convergence handoff "
        "only; no paper result or HPC claim. / 测量记录为零；只有内存收敛交接；"
        "不是论文结果，也不声称已接通 HPC。"
    )


if __name__ == "__main__":
    main()
