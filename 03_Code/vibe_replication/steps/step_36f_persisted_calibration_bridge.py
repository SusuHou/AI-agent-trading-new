"""Step 36F: persist one session's verified Step-35D calibration bridge.

步骤 36F：保存一个 session 经核对的 Step-35D 校准桥。

Run / 运行:
    py -3 -X utf8 steps/step_36f_persisted_calibration_bridge.py

Why this step exists / 为什么需要这一步:
    Step 36E saves complete measurement values, but Step 35D and Step 35F
    deliberately require the *live* Step-35C scorer and the exact checkpoint
    object bound to it.  Python object identity cannot be saved to disk.
    Therefore a new process must replay the saved origin and all measurement
    rows, prove that it reproduces Step 36E exactly, and only then run Step
    35D. / Step 36E 保存了完整测量数值，但 Step 35D 与 Step 35F 故意要求
    实时 Step-35C scorer 以及与它绑定的同一个 checkpoint 对象。Python
    对象身份不能直接保存到硬盘，所以新进程必须从保存的 origin 重放全部
    测量行，证明完全复现 Step 36E，然后才能运行 Step 35D。

Scope / 范围:
    This completes one per-session bridge.  It does not yet pool 1,000
    sessions, calibrate one common cell shock, run Step 35F, or produce a paper
    result. / 本步骤只完成一个 session 的桥；尚不汇总 1,000 个 session、
    不校准统一冲击、不运行 Step 35F，也不产生论文结果。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from steps.step_35a_converged_market_checkpoint import (
    ConvergedMarketCheckpoint,
    LOADED_IMPLEMENTATION_TREE_SHA256,
)
from steps.step_35d_unshocked_t3_calibration_paths import (
    UnshockedT3SessionCalibrationReceipt,
    run_unshocked_t3_calibration_paths,
    validate_unshocked_t3_session_calibration_receipt,
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
from steps.step_36e_complete_measurement_runner import (
    COMPLETE,
    CompleteMeasurementEvidence,
    CompleteMeasurementExecution,
    MEASUREMENT_PIPELINE_PROTOCOL_ID,
    ReconstructedCompleteMeasurementRuntime,
    ArtifactReference,
    TrainingCheckpointReference,
    _digest_dataclass,
    _load_bundle,
    _safe_artifact_path,
    _safe_task_child_path,
    _save_immutable_bundle,
    _task_directory,
    load_complete_measurement_evidence,
    load_convergence_origin,
    reconstruct_verified_complete_measurement_runtime,
    run_complete_measurement_task,
)


BRIDGE_SCHEMA_VERSION = "step36f-session-calibration-bridge-v2-source-scope"
BRIDGE_FILE_MAGIC = b"VIBE_STEP36F_SESSION_CALIBRATION_BRIDGE_V2\n"
BRIDGE_DIRECTORY_NAME = "calibration_bridge"
BRIDGE_FILE_NAME = "session_calibration_bridge.bundle"

ROLE_COMPLETE_EVIDENCE = "step36e_complete_measurement_evidence"
ROLE_CONVERGENCE_ORIGIN = "step36e_convergence_origin"
ROLE_REPLAY_CHECKPOINT = "step36c_replay_checkpoint"


@dataclass(frozen=True)
class RetainedArtifactRecord:
    """One file that must remain byte-for-byte available for replay.

    为了重放而必须逐字节保留的一份文件记录。
    """

    role: str
    relative_path: str
    logical_sha256: str
    file_sha256: str
    byte_size: int


@dataclass(frozen=True)
class Step36FSessionCalibrationBridge:
    """Immutable per-session Step-36E -> Step-35D evidence bridge.

    不可修改的单 session Step-36E -> Step-35D 证据桥。
    """

    schema_version: str
    plan_sha256: str
    experiment_cell_sha256: str
    run_config_sha256: str
    task_id: str
    task_sha256: str
    session_index: int
    implementation_tree_sha256: str
    measurement_sink_protocol_id: str
    complete_evidence_reference: ArtifactReference
    convergence_origin_reference: ArtifactReference
    replay_mid_training_checkpoint: TrainingCheckpointReference | None
    retained_artifacts: tuple[RetainedArtifactRecord, ...]
    complete_evidence_sha256: str
    convergence_origin_sha256: str
    convergence_checkpoint_sha256: str
    baseline_receipt_payload_sha256: str
    baseline_scored_fields_sha256: str
    calibration_receipt: UnshockedT3SessionCalibrationReceipt
    calibration_paths_requested: int
    calibration_paths_executed: int
    calibration_irf_experiment_seed: int
    exact_complete_measurement_replay_verified: bool
    live_checkpoint_and_scorer_identity_verified: bool
    calibration_receipt_persisted_and_reloaded: bool
    retained_dependencies_verified_at_commit: bool
    calibration_restart_from_path_zero_after_failure: bool
    partial_calibration_resume_supported: bool
    per_session_step35f_reload_adapter_ready: bool
    a24_per_session_chain_complete: bool
    a24_full_cell_bridge_complete: bool
    cell_shock_calibrated: bool
    step35f_run: bool
    research_result: bool
    paper_results_ready: bool
    bridge_sha256: str


@dataclass(frozen=True)
class VerifiedStep35FSessionSource:
    """Runtime-only live inputs that a later Step-35F worker can use.

    以后 Step-35F worker 可使用的、仅存在于运行时的实时输入。
    """

    bridge: Step36FSessionCalibrationBridge
    evidence: CompleteMeasurementEvidence
    runtime: ReconstructedCompleteMeasurementRuntime

    @property
    def checkpoint(self) -> ConvergedMarketCheckpoint:
        """Return the exact live-bound checkpoint. / 返回实时绑定的精确 checkpoint。"""

        return self.runtime.convergence_checkpoint

    @property
    def baseline_scorer(self):
        """Return the genuine live Step-35C scorer. / 返回真正的实时 Step-35C scorer。"""

        return self.runtime.pipeline.irf_long_run_baseline_scorer


@dataclass(frozen=True)
class Step36FExecution:
    """One runner invocation result. / 一次 runner 调用结果。"""

    status: str
    bridge: Step36FSessionCalibrationBridge | None
    bridge_path: Path
    measurement_execution: CompleteMeasurementExecution | None
    verified_live_source: VerifiedStep35FSessionSource | None


def _is_sha256(value: object) -> bool:
    """Return whether value is lowercase SHA-256 text. / 判断是否为小写 SHA-256。"""

    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value)
    )


def _bridge_registry() -> dict[str, type]:
    """Extra explicit wire types owned by Step 36F. / Step 36F 自己的明确 wire 类型。"""

    return {
        RetainedArtifactRecord.__name__: RetainedArtifactRecord,
        Step36FSessionCalibrationBridge.__name__: (
            Step36FSessionCalibrationBridge
        ),
        UnshockedT3SessionCalibrationReceipt.__name__: (
            UnshockedT3SessionCalibrationReceipt
        ),
    }


def _bridge_path(
    artifact_root: Path,
    task: SessionTaskManifest,
) -> Path:
    """Return one deterministic immutable bridge path. / 返回确定的不可变 bridge 路径。"""

    return _safe_task_child_path(
        _task_directory(artifact_root, task),
        BRIDGE_DIRECTORY_NAME,
        BRIDGE_FILE_NAME,
    )


def _retained_record(
    *,
    role: str,
    relative_path: str,
    logical_sha256: str,
    artifact_root: Path,
) -> RetainedArtifactRecord:
    """Read and fingerprint one required artifact. / 读取并指纹化一份必需 artifact。"""

    path = _safe_artifact_path(artifact_root, relative_path)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"Required retained artifact is unavailable: {role}. / "
            f"必需保留的 artifact 不可用：{role}。"
        ) from error
    return RetainedArtifactRecord(
        role=role,
        relative_path=relative_path,
        logical_sha256=logical_sha256,
        file_sha256=sha256(payload).hexdigest(),
        byte_size=len(payload),
    )


def _build_retention_manifest(
    *,
    evidence_reference: ArtifactReference,
    origin_reference: ArtifactReference,
    replay_reference: TrainingCheckpointReference | None,
    artifact_root: Path,
) -> tuple[RetainedArtifactRecord, ...]:
    """Fingerprint the complete replay dependency set. / 指纹化完整重放依赖集合。"""

    records = [
        _retained_record(
            role=ROLE_COMPLETE_EVIDENCE,
            relative_path=evidence_reference.relative_path,
            logical_sha256=evidence_reference.content_sha256,
            artifact_root=artifact_root,
        ),
        _retained_record(
            role=ROLE_CONVERGENCE_ORIGIN,
            relative_path=origin_reference.relative_path,
            logical_sha256=origin_reference.content_sha256,
            artifact_root=artifact_root,
        ),
    ]
    if replay_reference is not None:
        records.append(
            _retained_record(
                role=ROLE_REPLAY_CHECKPOINT,
                relative_path=replay_reference.relative_path,
                logical_sha256=replay_reference.checkpoint_sha256,
                artifact_root=artifact_root,
            )
        )
    return tuple(records)


def _validate_retained_record_shape(record: RetainedArtifactRecord) -> None:
    """Validate metadata without touching the filesystem. / 不读取文件，只验证 metadata。"""

    if not isinstance(record, RetainedArtifactRecord):
        raise TypeError("Retention record has the wrong type. / 保留记录类型错误。")
    relative = PurePosixPath(record.relative_path)
    if (
        record.role not in {
            ROLE_COMPLETE_EVIDENCE,
            ROLE_CONVERGENCE_ORIGIN,
            ROLE_REPLAY_CHECKPOINT,
        }
        or relative.is_absolute()
        or any(part in ("", ".", "..") for part in relative.parts)
        or not _is_sha256(record.logical_sha256)
        or not _is_sha256(record.file_sha256)
        or isinstance(record.byte_size, bool)
        or not isinstance(record.byte_size, int)
        or record.byte_size < 1
    ):
        raise ValueError("Retention record is malformed. / 保留记录格式错误。")


def validate_step36f_session_calibration_bridge(
    bridge: Step36FSessionCalibrationBridge,
    *,
    expected_plan: ExperimentCellPlan,
    expected_task: SessionTaskManifest,
) -> None:
    """Validate the bridge's values and all honesty claims.

    验证 bridge 的数值以及全部诚实边界声明。
    """

    validate_experiment_cell_plan(expected_plan)
    validate_session_task_for_config(expected_task, expected_plan.config)
    if expected_task not in expected_plan.tasks:
        raise ValueError("task is not a member of plan. / task 不属于 plan。")
    if not isinstance(bridge, Step36FSessionCalibrationBridge):
        raise TypeError("bridge has the wrong type. / bridge 类型错误。")
    if bridge.schema_version != BRIDGE_SCHEMA_VERSION:
        raise ValueError("Bridge version is unsupported. / bridge 版本不支持。")
    if _digest_dataclass(bridge, "bridge_sha256") != bridge.bridge_sha256:
        raise ValueError("Bridge checksum failed. / bridge 校验失败。")
    if (
        bridge.plan_sha256 != expected_plan.plan_sha256
        or bridge.experiment_cell_sha256
        != expected_plan.experiment_cell_sha256
        or bridge.run_config_sha256 != expected_plan.run_config_sha256
        or bridge.task_id != expected_task.task_id
        or bridge.task_sha256 != expected_task.task_sha256
        or bridge.session_index != expected_task.session_index
        or bridge.implementation_tree_sha256
        != LOADED_IMPLEMENTATION_TREE_SHA256
        or bridge.measurement_sink_protocol_id
        != MEASUREMENT_PIPELINE_PROTOCOL_ID
    ):
        raise ValueError("Bridge belongs to another task/build. / bridge 属于另一任务或源码。")
    if not expected_plan.persisted_post_convergence_bundle_available:
        raise ValueError(
            "Plan did not authorize the persisted post-convergence bridge. / "
            "plan 没有启用收敛后持久 bridge。"
        )

    validate_unshocked_t3_session_calibration_receipt(
        bridge.calibration_receipt
    )
    receipt = bridge.calibration_receipt
    baseline = receipt.long_run_baseline_receipt
    if (
        bridge.complete_evidence_sha256
        != bridge.complete_evidence_reference.content_sha256
        or bridge.convergence_origin_sha256
        != bridge.convergence_origin_reference.content_sha256
        or bridge.convergence_checkpoint_sha256
        != receipt.checkpoint_sha256
        or bridge.baseline_receipt_payload_sha256
        != baseline.receipt_payload_sha256
        or bridge.baseline_scored_fields_sha256
        != baseline.scored_fields_sha256
        or bridge.calibration_paths_requested
        != expected_plan.config.irf_paths_per_session
        or bridge.calibration_paths_requested != receipt.paths_requested
        or bridge.calibration_paths_executed != receipt.paths_executed
        or bridge.calibration_paths_executed
        != bridge.calibration_paths_requested
        or bridge.calibration_irf_experiment_seed
        != expected_plan.config.irf_experiment_seed
        or bridge.calibration_irf_experiment_seed
        != receipt.irf_experiment_seed
        or receipt.source_seed_manifest != expected_task.seed_manifest
        or receipt.implementation_tree_sha256
        != LOADED_IMPLEMENTATION_TREE_SHA256
    ):
        raise ValueError("Bridge and Step-35D receipt disagree. / bridge 与 Step-35D receipt 不一致。")

    expected_roles = {
        ROLE_COMPLETE_EVIDENCE,
        ROLE_CONVERGENCE_ORIGIN,
    }
    if bridge.replay_mid_training_checkpoint is not None:
        expected_roles.add(ROLE_REPLAY_CHECKPOINT)
    records_by_role: dict[str, RetainedArtifactRecord] = {}
    for record in bridge.retained_artifacts:
        _validate_retained_record_shape(record)
        if record.role in records_by_role:
            raise ValueError("Retention roles must be unique. / 保留记录 role 必须唯一。")
        records_by_role[record.role] = record
    if set(records_by_role) != expected_roles:
        raise ValueError("Retention dependency set is incomplete. / 保留依赖集合不完整。")
    evidence_record = records_by_role[ROLE_COMPLETE_EVIDENCE]
    origin_record = records_by_role[ROLE_CONVERGENCE_ORIGIN]
    if (
        evidence_record.relative_path
        != bridge.complete_evidence_reference.relative_path
        or evidence_record.logical_sha256
        != bridge.complete_evidence_reference.content_sha256
        or origin_record.relative_path
        != bridge.convergence_origin_reference.relative_path
        or origin_record.logical_sha256
        != bridge.convergence_origin_reference.content_sha256
    ):
        raise ValueError("Retention references disagree with bridge. / 保留引用与 bridge 不一致。")
    replay_reference = bridge.replay_mid_training_checkpoint
    if replay_reference is not None:
        replay_record = records_by_role[ROLE_REPLAY_CHECKPOINT]
        if (
            replay_record.relative_path != replay_reference.relative_path
            or replay_record.logical_sha256
            != replay_reference.checkpoint_sha256
        ):
            raise ValueError("Replay checkpoint retention differs. / 重放 checkpoint 保留记录不同。")

    honesty_flags = (
        bridge.exact_complete_measurement_replay_verified,
        bridge.live_checkpoint_and_scorer_identity_verified,
        bridge.calibration_receipt_persisted_and_reloaded,
        bridge.retained_dependencies_verified_at_commit,
        bridge.calibration_restart_from_path_zero_after_failure,
        not bridge.partial_calibration_resume_supported,
        bridge.per_session_step35f_reload_adapter_ready,
        bridge.a24_per_session_chain_complete,
        not bridge.a24_full_cell_bridge_complete,
        not bridge.cell_shock_calibrated,
        not bridge.step35f_run,
        not bridge.research_result,
        not bridge.paper_results_ready,
    )
    if not all(honesty_flags):
        raise ValueError("Step 36F honesty flags are inconsistent. / Step 36F 诚实标记不一致。")


def _verify_retained_files(
    bridge: Step36FSessionCalibrationBridge,
    *,
    artifact_root: Path,
) -> None:
    """Require every retained file to remain byte-for-byte identical.

    要求每个保留文件仍然逐字节完全相同。
    """

    for record in bridge.retained_artifacts:
        path = _safe_artifact_path(artifact_root, record.relative_path)
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"Retained artifact is missing: {record.role}. / "
                f"保留 artifact 丢失：{record.role}。"
            ) from error
        if (
            len(payload) != record.byte_size
            or sha256(payload).hexdigest() != record.file_sha256
        ):
            raise ValueError(
                f"Retained artifact changed: {record.role}. / "
                f"保留 artifact 已改变：{record.role}。"
            )


def _load_referenced_evidence(
    bridge: Step36FSessionCalibrationBridge,
    *,
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    artifact_root: Path,
) -> CompleteMeasurementEvidence:
    """Load and cross-check the Step-36E evidence/origin chain.

    读取并交叉核对 Step-36E evidence/origin 证据链。
    """

    _verify_retained_files(bridge, artifact_root=artifact_root)
    evidence_path = _safe_artifact_path(
        artifact_root,
        bridge.complete_evidence_reference.relative_path,
    )
    evidence = load_complete_measurement_evidence(
        evidence_path,
        expected_plan=plan,
        expected_task=task,
        trusted_local_file=True,
    )
    origin_path = _safe_artifact_path(
        artifact_root,
        bridge.convergence_origin_reference.relative_path,
    )
    origin = load_convergence_origin(
        origin_path,
        expected_plan=plan,
        expected_task=task,
        trusted_local_file=True,
    )
    if (
        evidence.evidence_sha256 != bridge.complete_evidence_sha256
        or origin.origin_sha256 != bridge.convergence_origin_sha256
        or evidence.convergence_origin != origin
        or origin.checkpoint.checkpoint_sha256
        != bridge.convergence_checkpoint_sha256
        or origin.replay_mid_training_checkpoint
        != bridge.replay_mid_training_checkpoint
        or evidence.irf_long_run_baseline_receipt
        != bridge.calibration_receipt.long_run_baseline_receipt
    ):
        raise ValueError(
            "Step-36E artifacts and Step-36F bridge do not form one chain. / "
            "Step-36E artifacts与 Step-36F bridge 未形成同一证据链。"
        )
    return evidence


def save_step36f_session_calibration_bridge(
    bridge: Step36FSessionCalibrationBridge,
    path: Path,
    *,
    expected_plan: ExperimentCellPlan,
    expected_task: SessionTaskManifest,
    artifact_root: Path,
) -> Path:
    """Atomically save one immutable bridge. / 原子保存一份不可变 bridge。"""

    validate_step36f_session_calibration_bridge(
        bridge,
        expected_plan=expected_plan,
        expected_task=expected_task,
    )
    if path.resolve() != _bridge_path(artifact_root, expected_task).resolve():
        raise ValueError("Bridge path belongs to another task. / bridge 路径属于另一任务。")
    _verify_retained_files(bridge, artifact_root=artifact_root)
    return _save_immutable_bundle(bridge, path, BRIDGE_FILE_MAGIC)


def load_step36f_session_calibration_bridge(
    path: Path,
    *,
    expected_plan: ExperimentCellPlan,
    expected_task: SessionTaskManifest,
    artifact_root: Path,
    trusted_local_file: bool = False,
) -> Step36FSessionCalibrationBridge:
    """Load a bridge and verify all retained dependencies. / 读取 bridge 并核对全部保留依赖。"""

    if not trusted_local_file:
        raise ValueError(
            "Set trusted_local_file=True for project-created bundle data. / "
            "项目自建 bundle 数据需设 trusted_local_file=True。"
        )
    if path.resolve() != _bridge_path(artifact_root, expected_task).resolve():
        raise ValueError("Bridge path belongs to another task. / bridge 路径属于另一任务。")
    decoded = _load_bundle(
        path,
        BRIDGE_FILE_MAGIC,
        Step36FSessionCalibrationBridge,
        additional_registry=_bridge_registry(),
    )
    if not isinstance(decoded, Step36FSessionCalibrationBridge):
        raise RuntimeError("Decoded bridge has the wrong type. / 解码 bridge 类型错误。")
    validate_step36f_session_calibration_bridge(
        decoded,
        expected_plan=expected_plan,
        expected_task=expected_task,
    )
    _load_referenced_evidence(
        decoded,
        plan=expected_plan,
        task=expected_task,
        artifact_root=artifact_root,
    )
    return decoded


def reconstruct_verified_step35f_session_source(
    bridge: Step36FSessionCalibrationBridge,
    *,
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    artifact_root: Path,
) -> VerifiedStep35FSessionSource:
    """Recreate live Step-35F inputs and re-prove the saved Step-35D result.

    重建 Step-35F 所需实时输入，并重新证明已保存的 Step-35D 结果。
    """

    validate_step36f_session_calibration_bridge(
        bridge,
        expected_plan=plan,
        expected_task=task,
    )
    evidence = _load_referenced_evidence(
        bridge,
        plan=plan,
        task=task,
        artifact_root=artifact_root,
    )
    runtime = reconstruct_verified_complete_measurement_runtime(
        evidence,
        plan=plan,
        task=task,
        artifact_root=artifact_root,
    )
    scorer = runtime.pipeline.irf_long_run_baseline_scorer
    live_baseline = scorer.verified_live_result_for_step35d(
        runtime.convergence_checkpoint
    )
    if live_baseline != evidence.irf_long_run_baseline_receipt:
        raise RuntimeError("Live baseline differs after replay. / 重放后实时 baseline 不同。")
    repeated_receipt = run_unshocked_t3_calibration_paths(
        runtime.convergence_checkpoint,
        baseline_scorer=scorer,
        irf_experiment_seed=bridge.calibration_irf_experiment_seed,
        path_count=bridge.calibration_paths_requested,
    )
    if repeated_receipt != bridge.calibration_receipt:
        raise RuntimeError(
            "Replayed Step-35D receipt differs from the persisted bridge. / "
            "重放 Step-35D receipt 与已保存 bridge 不同。"
        )
    return VerifiedStep35FSessionSource(
        bridge=bridge,
        evidence=evidence,
        runtime=runtime,
    )


def _build_bridge(
    *,
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    measurement_execution: CompleteMeasurementExecution,
    evidence: CompleteMeasurementEvidence,
    receipt: UnshockedT3SessionCalibrationReceipt,
    retained_artifacts: tuple[RetainedArtifactRecord, ...],
) -> Step36FSessionCalibrationBridge:
    """Build one checksummed honest bridge. / 建立一份带校验且诚实的 bridge。"""

    scientific = measurement_execution.status.scientific_outcome
    evidence_reference = scientific.complete_evidence_reference
    origin_reference = scientific.convergence_origin_reference
    if evidence_reference is None or origin_reference is None:
        raise RuntimeError("Complete Step-36E references are missing. / Step-36E 完整引用丢失。")
    origin = evidence.convergence_origin
    baseline = receipt.long_run_baseline_receipt
    draft = Step36FSessionCalibrationBridge(
        schema_version=BRIDGE_SCHEMA_VERSION,
        plan_sha256=plan.plan_sha256,
        experiment_cell_sha256=plan.experiment_cell_sha256,
        run_config_sha256=plan.run_config_sha256,
        task_id=task.task_id,
        task_sha256=task.task_sha256,
        session_index=task.session_index,
        implementation_tree_sha256=LOADED_IMPLEMENTATION_TREE_SHA256,
        measurement_sink_protocol_id=MEASUREMENT_PIPELINE_PROTOCOL_ID,
        complete_evidence_reference=evidence_reference,
        convergence_origin_reference=origin_reference,
        replay_mid_training_checkpoint=(
            origin.replay_mid_training_checkpoint
        ),
        retained_artifacts=retained_artifacts,
        complete_evidence_sha256=evidence.evidence_sha256,
        convergence_origin_sha256=origin.origin_sha256,
        convergence_checkpoint_sha256=(
            origin.checkpoint.checkpoint_sha256
        ),
        baseline_receipt_payload_sha256=(
            baseline.receipt_payload_sha256
        ),
        baseline_scored_fields_sha256=baseline.scored_fields_sha256,
        calibration_receipt=receipt,
        calibration_paths_requested=receipt.paths_requested,
        calibration_paths_executed=receipt.paths_executed,
        calibration_irf_experiment_seed=receipt.irf_experiment_seed,
        exact_complete_measurement_replay_verified=True,
        live_checkpoint_and_scorer_identity_verified=True,
        calibration_receipt_persisted_and_reloaded=True,
        retained_dependencies_verified_at_commit=True,
        calibration_restart_from_path_zero_after_failure=True,
        partial_calibration_resume_supported=False,
        per_session_step35f_reload_adapter_ready=True,
        a24_per_session_chain_complete=True,
        a24_full_cell_bridge_complete=False,
        cell_shock_calibrated=False,
        step35f_run=False,
        research_result=False,
        paper_results_ready=False,
        bridge_sha256="",
    )
    return replace(
        draft,
        bridge_sha256=_digest_dataclass(draft, "bridge_sha256"),
    )


def run_and_persist_session_calibration_bridge(
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    *,
    artifact_root: Path,
    checkpoint_interval_periods: int,
    invocation_training_period_budget: int | None = None,
    retry_failed: bool = False,
) -> Step36FExecution:
    """Complete Step 36E, replay it, run real Step 35D, and persist the bridge.

    完成 Step 36E、重放核对、运行真正 Step 35D，并保存 bridge。
    """

    validate_experiment_cell_plan(plan)
    if task not in plan.tasks:
        raise ValueError("task is not a member of plan. / task 不属于 plan。")
    validate_session_task_for_config(task, plan.config)
    if not plan.persisted_post_convergence_bundle_available:
        raise ValueError(
            "Build the plan with persisted_post_convergence_bundle_available=True. / "
            "建立 plan 时请设置 persisted_post_convergence_bundle_available=True。"
        )
    path = _bridge_path(artifact_root, task)
    if path.exists():
        loaded = load_step36f_session_calibration_bridge(
            path,
            expected_plan=plan,
            expected_task=task,
            artifact_root=artifact_root,
            trusted_local_file=True,
        )
        return Step36FExecution(
            status=COMPLETE,
            bridge=loaded,
            bridge_path=path,
            measurement_execution=None,
            verified_live_source=None,
        )

    measurement = run_complete_measurement_task(
        plan,
        task,
        artifact_root=artifact_root,
        checkpoint_interval_periods=checkpoint_interval_periods,
        invocation_training_period_budget=invocation_training_period_budget,
        retry_failed=retry_failed,
    )
    if (
        measurement.status.scientific_outcome.status != COMPLETE
        or measurement.evidence is None
    ):
        return Step36FExecution(
            status=measurement.status.scientific_outcome.status,
            bridge=None,
            bridge_path=path,
            measurement_execution=measurement,
            verified_live_source=None,
        )

    evidence = measurement.evidence
    runtime = reconstruct_verified_complete_measurement_runtime(
        evidence,
        plan=plan,
        task=task,
        artifact_root=artifact_root,
    )
    scorer = runtime.pipeline.irf_long_run_baseline_scorer
    live_baseline = scorer.verified_live_result_for_step35d(
        runtime.convergence_checkpoint
    )
    if live_baseline != evidence.irf_long_run_baseline_receipt:
        raise RuntimeError("Step-35C baseline differs after replay. / 重放后 Step-35C baseline 不同。")
    receipt = run_unshocked_t3_calibration_paths(
        runtime.convergence_checkpoint,
        baseline_scorer=scorer,
        irf_experiment_seed=plan.config.irf_experiment_seed,
        path_count=plan.config.irf_paths_per_session,
    )
    validate_unshocked_t3_session_calibration_receipt(receipt)

    scientific = measurement.status.scientific_outcome
    evidence_reference = scientific.complete_evidence_reference
    origin_reference = scientific.convergence_origin_reference
    if evidence_reference is None or origin_reference is None:
        raise RuntimeError("Complete Step-36E references are missing. / Step-36E 完整引用丢失。")
    retained = _build_retention_manifest(
        evidence_reference=evidence_reference,
        origin_reference=origin_reference,
        replay_reference=(
            evidence.convergence_origin.replay_mid_training_checkpoint
        ),
        artifact_root=artifact_root,
    )
    bridge = _build_bridge(
        plan=plan,
        task=task,
        measurement_execution=measurement,
        evidence=evidence,
        receipt=receipt,
        retained_artifacts=retained,
    )
    validate_step36f_session_calibration_bridge(
        bridge,
        expected_plan=plan,
        expected_task=task,
    )
    _verify_retained_files(bridge, artifact_root=artifact_root)
    save_step36f_session_calibration_bridge(
        bridge,
        path,
        expected_plan=plan,
        expected_task=task,
        artifact_root=artifact_root,
    )
    loaded = load_step36f_session_calibration_bridge(
        path,
        expected_plan=plan,
        expected_task=task,
        artifact_root=artifact_root,
        trusted_local_file=True,
    )
    if loaded != bridge:
        raise RuntimeError("Reloaded Step-36F bridge differs. / 重读 Step-36F bridge 不同。")
    source = VerifiedStep35FSessionSource(
        bridge=loaded,
        evidence=evidence,
        runtime=runtime,
    )
    return Step36FExecution(
        status=COMPLETE,
        bridge=loaded,
        bridge_path=path,
        measurement_execution=measurement,
        verified_live_source=source,
    )


def main() -> None:
    """Run a tiny low-noise engineering demonstration. / 运行一个低噪声工程演示。"""

    config = ExperimentCellConfig(
        mode=DEBUG_MODE,
        experiment_cell_key="step36f-debug-low-noise",
        parameters=PaperParameters(noise_std=0.1, market_maker_window=20),
        experiment_seed=36_600_001,
        irf_experiment_seed=36_600_002,
        session_count=1,
        convergence_periods_required=1,
        measurement_periods_required=20,
        irf_paths_per_session=3,
    )
    plan = build_experiment_cell_plan(
        config,
        ExperimentExecutionPolicy(
            maximum_training_periods=20,
            persisted_post_convergence_bundle_available=True,
        ),
    )
    execution = run_and_persist_session_calibration_bridge(
        plan,
        plan.tasks[0],
        artifact_root=(
            PROJECT_ROOT / "results" / "step36f_persisted_calibration_bridge"
        ),
        checkpoint_interval_periods=5,
    )
    print("Step 36F: persisted calibration bridge / 已保存校准桥")
    print(f"Status / 状态: {execution.status}")
    if execution.bridge is None:
        print("Training is not complete yet. / 训练尚未完成。")
        return
    bridge = execution.bridge
    print(
        "Step-35D paths / Step-35D 路径数: "
        f"{bridge.calibration_paths_executed}"
    )
    print(
        "Mean t=3 lambda / t=3 平均 lambda: "
        f"{bridge.calibration_receipt.mean_t3_price_impact_lambda:.12f}"
    )
    print(f"Bridge / 证据文件: {execution.bridge_path}")
    print("Research result / 论文结果: False (engineering validation only / 仅工程验证)")


if __name__ == "__main__":
    main()
