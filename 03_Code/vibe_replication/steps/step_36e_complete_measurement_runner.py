"""Step 36E: complete and persist one learned session's measurement window.

第 36E 步：完成并持久化一个已学习 session 的测量窗口。

Run the small demonstration / 运行小型演示:
    py -3 -X utf8 steps/step_36e_complete_measurement_runner.py

Safety boundary / 安全边界:
    Training starts with the final measurement fan-out already attached.  At
    convergence, the exact pre-row-zero origin is written and reloaded before
    CONVERGED may be published.  A failed partial measurement is never merged
    into a result: retry deterministically replays from the recorded training
    source and reruns measurement from row zero. / 训练从一开始就连接最终
    测量 fan-out。收敛时，必须先落盘并重读核对“第 0 条测量
    之前”的精确原点，才允许发布 CONVERGED。失败的半截测量绝不
    会并入结果；重试会从记录的训练起点确定性重放，然后从第 0
    条重跑整个测量窗口。

    When an origin names a Step-36C replay checkpoint, that immutable file is
    part of the evidence-retention set. Deletion or corruption fails closed;
    Step 36F will add the formal retention manifest. / 如果 origin 指向一份
    Step-36C 重放 checkpoint，该不可变文件就是证据保留集合的一部分；删除或
    损坏都会明确失败。正式保留清单留给 Step 36F。

This is a verified per-session engineering artifact, not yet a paper result:
Step 35D's live calibration receipt, cross-session aggregation, A23, A24, and
HPC dispatch remain later work. / 这是已验证的单 session 工程产物，
仍不是论文结果；Step 35D 实时校准凭证、跨 session 汇总、A23、
A24 和超算调度均留给后续步骤。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from hashlib import sha256
from io import BytesIO
from math import fsum, isfinite
from numbers import Real
from pathlib import Path, PurePosixPath
from time import perf_counter
import json
import pickle
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from step_22_market_maker_rolling_history import MarketObservation
from step_24b_fast_rolling_ols import (
    CenteredPairStatisticsState,
    RollingMarketMakerState,
)
from step_26_reproducible_random_streams import (
    FrozenPolicyPeriodObservation,
    RandomizedMarketSession,
    SessionSeedManifest,
)
from step_27_convergence_tracker import (
    PAPER_UNCHANGED_PERIODS,
    ConvergenceReceipt,
)
from step_28_session_phases import (
    PAPER_MEASUREMENT_PERIODS,
    SessionPhase,
    SessionPhaseController,
    SessionPhaseReceipt,
)
from step_29_matched_path_collusion_profitability import (
    CollusionProfitabilityReceipt,
    MatchedPathBenchmarkCoefficients,
    MatchedPathCollusionScorer,
    build_matched_path_benchmarks,
    normalize_collusion_profitability,
)
from step_30_trading_intensity import (
    OnlineTradingIntensityScorer,
    TradingIntensityReceipt,
    build_measurement_sink_fanout,
)
from step_31_price_informativeness import (
    PriceInformativenessReceipt,
    build_price_informativeness_receipt,
    calculate_price_informativeness,
)
from step_32_market_liquidity import (
    MarketLiquidityReceipt,
    OnlineMarketLiquidityScorer,
)
from step_33_mispricing import (
    DeferredOnlineMispricingScorer,
    MispricingReceipt,
)
from steps.step_35a_converged_market_checkpoint import (
    CheckpointProtocolNotes,
    ConvergedMarketCheckpoint,
    ConvergedMarketCheckpointPayload,
    ImmutableArraySnapshot,
    LOADED_IMPLEMENTATION_TREE_SHA256,
    verify_converged_market_checkpoint,
)
from steps.step_35c_irf_long_run_baseline import (
    IRFLongRunBaselineReceipt,
    OnlineIRFLongRunBaselineScorer,
    validate_irf_long_run_baseline_receipt,
)
from steps.step_36b_experiment_manifest import (
    DEBUG_MODE,
    PAPER_PATHS_PER_SESSION,
    PAPER_SESSIONS_PER_EXPERIMENT_CELL,
    ExperimentCellConfig,
    ExperimentCellPlan,
    ExperimentExecutionPolicy,
    SessionTaskManifest,
    build_experiment_cell_plan,
    validate_experiment_cell_plan,
    validate_session_task_for_config,
)
from steps.step_36c_exact_training_resume import (
    MidTrainingCheckpoint,
    _atomic_binary_write,
    load_mid_training_checkpoint,
    restore_mid_training_controller,
)
from steps.step_36d_single_session_training_runner import (
    CONVERGED,
    FAILED,
    INCOMPLETE,
    TrainingCheckpointReference,
    TrainingRunExecution,
    TrainingTaskExecutionError,
    build_fresh_training_controller,
    prune_training_checkpoints,
    run_training_task,
)


MEASUREMENT_PIPELINE_PROTOCOL_ID = (
    "step36e-fanout-v1:step29,step30,step32,step33,step35c"
)
ORIGIN_SCHEMA_VERSION = "step36e-convergence-replay-origin-v2-source-scope"
RESULT_SCHEMA_VERSION = "step36e-learned-session-result-v2"
EVIDENCE_SCHEMA_VERSION = "step36e-complete-measurement-evidence-v2"
STATUS_SCHEMA_VERSION = "step36e-measurement-status-v2"
ATTEMPT_SCHEMA_VERSION = "step36e-measurement-attempt-v2"
SCIENTIFIC_SCHEMA_VERSION = "step36e-measurement-scientific-outcome-v2"
WIRE_SCHEMA_VERSION = "step36e-explicit-dataclass-wire-v2"

ORIGIN_FILE_MAGIC = b"VIBE_STEP36E_CONVERGENCE_ORIGIN_V2\n"
EVIDENCE_FILE_MAGIC = b"VIBE_STEP36E_MEASUREMENT_EVIDENCE_V2\n"
ORIGIN_DIRECTORY_NAME = "convergence"
MEASUREMENT_DIRECTORY_NAME = "measurement"
STATUS_FILE_NAME = "measurement_status.json"

COMPLETE = "COMPLETE"
ALLOWED_STATUSES = (INCOMPLETE, COMPLETE, FAILED)
TRAINING_PHASE = "training"
MEASUREMENT_REPLAY_REQUIRED_PHASE = "measurement_replay_required"
MEASUREMENT_COMPLETE_PHASE = "measurement_complete"

FRESH_TRAINING_START = "fresh_training"
RESUMED_TRAINING_START = "resumed_training"
CONVERGENCE_CONTINUATION_START = "convergence_continuation"
MEASUREMENT_REPLAY_START = "measurement_replay"

Timer = Callable[[], float]
MeasurementPeriodCompletedHook = Callable[
    [SessionPhaseController, FrozenPolicyPeriodObservation],
    None,
]


@dataclass(frozen=True)
class ArtifactReference:
    """Content identity and safe relative path. / 内容身份与安全相对路径。"""

    relative_path: str
    content_sha256: str


@dataclass(frozen=True)
class ConvergenceReplayOrigin:
    """Durable row-zero origin plus the exact source used for replay.

    可持久的第 0 条测量原点，以及重放所用的精确起点。
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
    replay_mid_training_checkpoint: TrainingCheckpointReference | None
    convergence_period_index: int
    origin_global_period: int
    measurement_periods_required: int
    checkpoint: ConvergedMarketCheckpoint
    restart_entire_measurement_after_interruption: bool
    mid_measurement_resume_supported: bool
    origin_sha256: str


@dataclass(frozen=True)
class LearnedSessionResult:
    """Small scientific summary derived only after all scorers finalize.

    只在所有 scorer 完成后生成的小型科学摘要。
    """

    schema_version: str
    task_id: str
    session_index: int
    noise_std: float
    training_periods_completed: int
    measurement_periods_completed: int
    mean_actual_profit_by_agent: tuple[float, ...]
    mean_nash_profit: float
    mean_cartel_profit: float
    delta_c: float
    trading_intensity: float
    price_informativeness: float
    average_market_liquidity: float
    reported_average_mispricing: float | None
    mispricing_requires_research_decision: bool
    mean_irf_oriented_price: float
    mean_irf_oriented_order_by_agent: tuple[float, ...]
    mean_irf_profit_by_agent: tuple[float, ...]
    measurement_scored_fields_sha256: str
    exact_paper_scale_counts_matched: bool
    a23_source_fingerprint_scope_resolved: bool
    a24_distributed_evidence_bridge_resolved: bool
    research_result: bool
    paper_results_ready: bool
    result_sha256: str


@dataclass(frozen=True)
class CompleteMeasurementEvidence:
    """One immutable, cross-checked completed-session artifact.

    一份不可修改、经交叉核对的完整 session 产物。
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
    convergence_origin: ConvergenceReplayOrigin
    phase_receipt: SessionPhaseReceipt
    profitability_receipt: CollusionProfitabilityReceipt
    trading_intensity_receipt: TradingIntensityReceipt
    price_informativeness_receipt: PriceInformativenessReceipt
    market_liquidity_receipt: MarketLiquidityReceipt
    mispricing_receipt: MispricingReceipt
    irf_long_run_baseline_receipt: IRFLongRunBaselineReceipt
    learned_session_result: LearnedSessionResult
    complete_measurement_rows_committed: int
    partial_measurement_rows_committed: int
    measurement_restart_not_mid_window_resume: bool
    step35d_live_calibration_receipt_persisted: bool
    a24_full_bundle_available: bool
    research_result: bool
    paper_results_ready: bool
    evidence_sha256: str


@dataclass(frozen=True)
class MeasurementScientificOutcome:
    """Deterministic status facts; elapsed time is deliberately excluded.

    确定性状态事实；运行时间被故意排除。
    """

    schema_version: str
    status: str
    phase: str
    stop_reason: str
    task_id: str
    task_sha256: str
    run_config_sha256: str
    session_index: int
    implementation_tree_sha256: str
    measurement_sink_protocol_id: str
    training_periods_verified: int
    measurement_periods_required: int
    committed_measurement_rows: int
    convergence_origin_reference: ArtifactReference | None
    complete_evidence_reference: ArtifactReference | None
    replay_entire_window_required_after_failure: bool
    mid_measurement_resume_supported: bool
    research_result: bool
    paper_results_ready: bool
    outcome_sha256: str


@dataclass(frozen=True)
class MeasurementAttemptMetadata:
    """Nondeterministic operational facts for one invocation.

    一次调用中不确定的运行信息。
    """

    schema_version: str
    attempt_number: int
    start_mode: str
    training_periods_at_start: int
    training_periods_at_end: int
    measurement_rows_delivered_this_attempt: int
    elapsed_seconds: float
    measurement_rows_per_second: float | None
    failure_type: str | None
    failure_message: str | None
    live_runtime_discarded_after_failure: bool


@dataclass(frozen=True)
class PersistedMeasurementStatus:
    """Checksum-protected mutable task pointer. / 带校验的可变任务指针。"""

    schema_version: str
    scientific_outcome: MeasurementScientificOutcome
    attempt: MeasurementAttemptMetadata
    status_sha256: str


@dataclass(frozen=True)
class CompleteMeasurementExecution:
    """Runtime result; live objects are returned only in the current process.

    运行时结果；实时对象只在当前进程返回。
    """

    status: PersistedMeasurementStatus
    evidence: CompleteMeasurementEvidence | None
    controller: SessionPhaseController | None
    pipeline: "SessionMeasurementPipeline | None"
    convergence_checkpoint: ConvergedMarketCheckpoint | None
    status_path: Path


@dataclass(frozen=True)
class ReconstructedCompleteMeasurementRuntime:
    """A verified live runtime rebuilt from persisted Step-36E evidence.

    从已保存的 Step-36E evidence 重建并核对过的实时运行对象。

    This object is deliberately runtime-only.  The controller and scorers are
    not serialized; a new process must rebuild them by deterministic replay.
    / 此对象只存在于当前运行时。controller 与 scorer 不会被序列化；新的
    Python 进程必须通过确定性重放重新建立它们。
    """

    evidence: CompleteMeasurementEvidence
    controller: SessionPhaseController
    pipeline: "SessionMeasurementPipeline"
    convergence_checkpoint: ConvergedMarketCheckpoint


class CompleteMeasurementTaskError(RuntimeError):
    """Unexpected failure with a fail-closed persisted status.

    意外失败，并已持久化 fail-closed 状态。
    """

    def __init__(self, status: PersistedMeasurementStatus, status_path: Path) -> None:
        self.status = status
        self.status_path = status_path
        super().__init__(
            "Complete measurement failed; partial rows were not committed. "
            "/ 完整测量失败；半截记录未被提交。"
        )


class _UpstreamTrainingFailure(RuntimeError):
    """Carry Step-36D's original failure label through the outer status.

    把 Step-36D 原始失败类型与消息带入外层状态。
    """

    def __init__(self, training_status: object) -> None:
        attempt = training_status.attempt
        self.reported_failure_type = (
            attempt.failure_type or "TrainingTaskExecutionError"
        )
        self.reported_failure_message = (
            attempt.failure_message
            or "Step 36D ended in FAILED without a detailed message."
        )
        super().__init__(
            f"Step 36D {self.reported_failure_type}: "
            f"{self.reported_failure_message}"
        )


@dataclass(frozen=True)
class _FinalizedPipelineReceipts:
    profitability: CollusionProfitabilityReceipt
    trading_intensity: TradingIntensityReceipt
    price_informativeness: PriceInformativenessReceipt
    market_liquidity: MarketLiquidityReceipt
    mispricing: MispricingReceipt
    irf_long_run_baseline: IRFLongRunBaselineReceipt


class SessionMeasurementPipeline:
    """All scorers attached to one period-zero session in canonical order.

    按固定顺序连接到同一个第 0 期 session 的全部 scorer。
    """

    def __init__(self, session: RandomizedMarketSession) -> None:
        if not isinstance(session, RandomizedMarketSession):
            raise TypeError("session has the wrong type. / session 类型错误。")
        if session.period_number != 0 or session.execution_mode != "training":
            raise RuntimeError(
                "Build the pipeline on a temporary period-zero training session. "
                "/ 必须在临时第 0 期训练 session 上建立管线。"
            )
        self.session = session
        self.profitability_scorer = MatchedPathCollusionScorer(
            session,
            build_matched_path_benchmarks(
                session.parameters,
                tuple(session.value_grid),
            ),
        )
        self.trading_intensity_scorer = OnlineTradingIntensityScorer(session)
        self.market_liquidity_scorer = OnlineMarketLiquidityScorer(session)
        self.mispricing_scorer = DeferredOnlineMispricingScorer(session)
        self.irf_long_run_baseline_scorer = OnlineIRFLongRunBaselineScorer(
            session
        )
        # Ordering is part of MEASUREMENT_PIPELINE_PROTOCOL_ID. / 此顺序是
        # MEASUREMENT_PIPELINE_PROTOCOL_ID 的一部分。
        self.sink = build_measurement_sink_fanout(
            self.profitability_scorer.observe,
            self.trading_intensity_scorer.observe,
            self.market_liquidity_scorer.observe,
            self.mispricing_scorer.observe,
            self.irf_long_run_baseline_scorer.observe,
        )

    def capture_and_bind_origin(
        self,
        controller: SessionPhaseController,
    ) -> ConvergedMarketCheckpoint:
        """Bind Step 35C before measurement row zero. / 在第 0 条测量前绑定 Step 35C。"""

        if controller.session is not self.session:
            raise ValueError("Pipeline belongs to another session. / 管线属于另一 session。")
        return self.irf_long_run_baseline_scorer.capture_and_bind_convergence_checkpoint(
            controller
        )

    def finalize(
        self,
        controller: SessionPhaseController,
        checkpoint: ConvergedMarketCheckpoint,
    ) -> _FinalizedPipelineReceipts:
        """Finalize dependencies only after all T rows. / 只在 T 条全部完成后结算。"""

        if controller.session is not self.session:
            raise ValueError("Pipeline belongs to another session. / 管线属于另一 session。")
        if controller.phase is not SessionPhase.COMPLETE:
            raise RuntimeError("Measurement is not complete. / 测量尚未完成。")
        profitability = self.profitability_scorer.finalize(controller)
        trading = self.trading_intensity_scorer.finalize(controller)
        informativeness = build_price_informativeness_receipt(
            self.trading_intensity_scorer,
            controller,
        )
        liquidity = self.market_liquidity_scorer.finalize(controller)
        mispricing = self.mispricing_scorer.finalize(
            self.trading_intensity_scorer,
            controller,
        )
        baseline = self.irf_long_run_baseline_scorer.finalize(controller)
        live_baseline = (
            self.irf_long_run_baseline_scorer.verified_live_result_for_step35d(
                checkpoint
            )
        )
        if live_baseline is not baseline:
            raise RuntimeError("Step 35C live result identity changed. / Step 35C 实时结果身份改变。")
        return _FinalizedPipelineReceipts(
            profitability=profitability,
            trading_intensity=trading,
            price_informativeness=informativeness,
            market_liquidity=liquidity,
            mispricing=mispricing,
            irf_long_run_baseline=baseline,
        )


class MeasurementPipelineFactory:
    """Create one fresh pipeline for each fresh/restored training runtime.

    每次新建或恢复训练 runtime 时，建立一条全新管线。
    """

    def __init__(self) -> None:
        self.current: SessionMeasurementPipeline | None = None
        self.generation_count = 0

    def __call__(self, session: RandomizedMarketSession):
        pipeline = SessionMeasurementPipeline(session)
        self.current = pipeline
        self.generation_count += 1
        return pipeline.sink


def _digest_dataclass(value: object, checksum_field: str) -> str:
    """Digest a dataclass after blanking its own checksum. / 清空自身校验后取摘要。"""

    unsigned = replace(value, **{checksum_field: ""})
    return sha256(pickle.dumps(_wire_encode(unsigned), protocol=5)).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value)
    )


def _wire_encode(value: object) -> object:
    """Encode only explicit dataclasses, tuples, lists, dictionaries and scalars.

    只编码明确的 dataclass、tuple、list、dictionary 与标量。
    """

    if is_dataclass(value) and not isinstance(value, type):
        return {
            "@kind": "dataclass",
            "@type": type(value).__name__,
            "fields": {
                item.name: _wire_encode(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, tuple):
        return {"@kind": "tuple", "items": [_wire_encode(item) for item in value]}
    if isinstance(value, list):
        return {"@kind": "list", "items": [_wire_encode(item) for item in value]}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Wire dictionaries require string keys. / wire dictionary 要求字符串 key。")
        return {
            "@kind": "dict",
            # Dictionary insertion order is an implementation detail, not a
            # scientific value.  Sorting makes equal mappings produce exactly
            # the same bytes and SHA-256 after save/load. / 字典插入顺序不是
            # 科学数值；排序后，相同 mapping 在保存/读取前后产生完全相同的
            # bytes 与 SHA-256。
            "items": {
                key: _wire_encode(value[key])
                for key in sorted(value)
            },
        }
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    raise TypeError(f"Unsupported wire value {type(value)!r}. / 不支持的 wire 值。")


class _BuiltinsOnlyUnpickler(pickle.Unpickler):
    """Reject every GLOBAL opcode. / 拒绝所有 GLOBAL opcode。"""

    def find_class(self, module: str, name: str) -> object:
        raise pickle.UnpicklingError(
            f"Global {module}.{name} is forbidden in Step 36E wire data."
        )


def _restricted_pickle_loads(data: bytes) -> object:
    return _BuiltinsOnlyUnpickler(BytesIO(data)).load()


def _dataclass_registry() -> dict[str, type]:
    """Explicit allowlist for immutable scientific data classes.

    不可修改科学数据 class 的明确允许清单。
    """

    allowed = (
        PaperParameters,
        MarketObservation,
        CenteredPairStatisticsState,
        RollingMarketMakerState,
        SessionSeedManifest,
        ConvergenceReceipt,
        SessionPhaseReceipt,
        ImmutableArraySnapshot,
        CheckpointProtocolNotes,
        ConvergedMarketCheckpointPayload,
        ConvergedMarketCheckpoint,
        TrainingCheckpointReference,
        MatchedPathBenchmarkCoefficients,
        CollusionProfitabilityReceipt,
        TradingIntensityReceipt,
        PriceInformativenessReceipt,
        MarketLiquidityReceipt,
        MispricingReceipt,
        IRFLongRunBaselineReceipt,
        ArtifactReference,
        ConvergenceReplayOrigin,
        LearnedSessionResult,
        CompleteMeasurementEvidence,
    )
    registry = {item.__name__: item for item in allowed}
    if len(registry) != len(allowed):
        raise RuntimeError("Wire dataclass names are not unique. / wire dataclass 名称不唯一。")
    return registry


def _wire_decode(
    value: object,
    additional_registry: dict[str, type] | None = None,
) -> object:
    """Rebuild only classes in the explicit allowlist. / 只重建允许清单中的 class。"""

    if not isinstance(value, dict) or "@kind" not in value:
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return value
        raise ValueError("Wire scalar is malformed. / wire 标量格式错误。")
    kind = value.get("@kind")
    if kind in ("tuple", "list"):
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("Wire sequence is malformed. / wire 序列格式错误。")
        decoded = [
            _wire_decode(item, additional_registry)
            for item in items
        ]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "dict":
        items = value.get("items")
        if not isinstance(items, dict) or any(
            not isinstance(key, str) for key in items
        ):
            raise ValueError("Wire dictionary is malformed. / wire dictionary 格式错误。")
        return {
            key: _wire_decode(item, additional_registry)
            for key, item in items.items()
        }
    if kind == "dataclass":
        name = value.get("@type")
        field_values = value.get("fields")
        registry = _dataclass_registry()
        if additional_registry is not None:
            for extra_name, extra_type in additional_registry.items():
                if (
                    extra_name in registry
                    and registry[extra_name] is not extra_type
                ):
                    raise ValueError(
                        "Additional wire type conflicts with the base registry. / "
                        "附加 wire 类型与基础清单冲突。"
                    )
                registry[extra_name] = extra_type
        if name not in registry or not isinstance(field_values, dict):
            raise ValueError("Wire dataclass is not allowed. / wire dataclass 未被允许。")
        cls = registry[name]
        expected = {item.name for item in fields(cls)}
        if set(field_values) != expected:
            raise ValueError("Wire dataclass fields differ from schema. / wire dataclass 字段与 schema 不同。")
        return cls(
            **{
                key: _wire_decode(item, additional_registry)
                for key, item in field_values.items()
            }
        )
    raise ValueError("Wire kind is unsupported. / wire kind 不支持。")


def _bundle_bytes(value: object, magic: bytes) -> bytes:
    wire = {
        "wire_schema_version": WIRE_SCHEMA_VERSION,
        "payload": _wire_encode(value),
    }
    serialized = pickle.dumps(wire, protocol=5)
    digest = sha256(serialized).hexdigest().encode("ascii")
    return magic + digest + b"\n" + serialized


def _save_immutable_bundle(value: object, path: Path, magic: bytes) -> Path:
    data = _bundle_bytes(value, magic)
    if path.exists():
        if path.read_bytes() == data:
            return path
        raise FileExistsError("A different immutable bundle already exists. / 该路径已有不同不可变 bundle。")
    _atomic_binary_write(path, data)
    return path


def _load_bundle(
    path: Path,
    magic: bytes,
    expected_type: type,
    *,
    additional_registry: dict[str, type] | None = None,
) -> object:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError("Cannot read Step 36E bundle. / 无法读取 Step 36E bundle。") from error
    if not data.startswith(magic):
        raise ValueError("Step 36E bundle header is invalid. / Step 36E bundle 文件头无效。")
    try:
        digest_bytes, serialized = data[len(magic):].split(b"\n", 1)
    except ValueError as error:
        raise ValueError("Step 36E bundle is truncated. / Step 36E bundle 被截断。") from error
    if (
        len(digest_bytes) != 64
        or sha256(serialized).hexdigest().encode("ascii") != digest_bytes
    ):
        raise ValueError("Step 36E file checksum failed. / Step 36E 文件校验失败。")
    try:
        wire = _restricted_pickle_loads(serialized)
    except Exception as error:
        raise ValueError("Step 36E built-in wire cannot be decoded. / Step 36E 内置 wire 无法解码。") from error
    if (
        not isinstance(wire, dict)
        or wire.get("wire_schema_version") != WIRE_SCHEMA_VERSION
        or "payload" not in wire
    ):
        raise ValueError("Step 36E wire schema is invalid. / Step 36E wire schema 无效。")
    decoded = _wire_decode(
        wire["payload"],
        additional_registry,
    )
    if not isinstance(decoded, expected_type):
        raise ValueError("Step 36E bundle has the wrong payload type. / Step 36E bundle 主体类型错误。")
    return decoded


def _json_ready(value: object) -> object:
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
    encoded = json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _artifact_reference_from_dictionary(
    data: object,
) -> ArtifactReference | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("Artifact reference is malformed. / artifact 引用格式错误。")
    return ArtifactReference(**data)


def _status_without_checksum(status: PersistedMeasurementStatus) -> dict[str, object]:
    data = asdict(status)
    data.pop("status_sha256")
    return data


def _scientific_without_checksum(
    outcome: MeasurementScientificOutcome,
) -> dict[str, object]:
    data = asdict(outcome)
    data.pop("outcome_sha256")
    return data


def _build_status(
    scientific: MeasurementScientificOutcome,
    attempt: MeasurementAttemptMetadata,
) -> PersistedMeasurementStatus:
    draft = PersistedMeasurementStatus(
        schema_version=STATUS_SCHEMA_VERSION,
        scientific_outcome=scientific,
        attempt=attempt,
        status_sha256="",
    )
    return replace(
        draft,
        status_sha256=_sha256_json(_status_without_checksum(draft)),
    )


def _build_scientific_outcome(
    *,
    status: str,
    phase: str,
    stop_reason: str,
    task: SessionTaskManifest,
    training_periods_verified: int,
    measurement_periods_required: int,
    committed_measurement_rows: int,
    origin_reference: ArtifactReference | None,
    evidence_reference: ArtifactReference | None,
    replay_required: bool,
) -> MeasurementScientificOutcome:
    draft = MeasurementScientificOutcome(
        schema_version=SCIENTIFIC_SCHEMA_VERSION,
        status=status,
        phase=phase,
        stop_reason=stop_reason,
        task_id=task.task_id,
        task_sha256=task.task_sha256,
        run_config_sha256=task.run_config_sha256,
        session_index=task.session_index,
        implementation_tree_sha256=LOADED_IMPLEMENTATION_TREE_SHA256,
        measurement_sink_protocol_id=MEASUREMENT_PIPELINE_PROTOCOL_ID,
        training_periods_verified=training_periods_verified,
        measurement_periods_required=measurement_periods_required,
        committed_measurement_rows=committed_measurement_rows,
        convergence_origin_reference=origin_reference,
        complete_evidence_reference=evidence_reference,
        replay_entire_window_required_after_failure=replay_required,
        mid_measurement_resume_supported=False,
        research_result=False,
        paper_results_ready=False,
        outcome_sha256="",
    )
    return replace(
        draft,
        outcome_sha256=_sha256_json(_scientific_without_checksum(draft)),
    )


def _status_from_dictionary(data: object) -> PersistedMeasurementStatus:
    if not isinstance(data, dict):
        raise ValueError("Measurement status must be a JSON object. / 测量状态必须是 JSON object。")
    try:
        scientific_data = dict(data["scientific_outcome"])
        scientific_data["convergence_origin_reference"] = (
            _artifact_reference_from_dictionary(
                scientific_data["convergence_origin_reference"]
            )
        )
        scientific_data["complete_evidence_reference"] = (
            _artifact_reference_from_dictionary(
                scientific_data["complete_evidence_reference"]
            )
        )
        scientific = MeasurementScientificOutcome(**scientific_data)
        attempt = MeasurementAttemptMetadata(**data["attempt"])
        return PersistedMeasurementStatus(
            schema_version=data["schema_version"],
            scientific_outcome=scientific,
            attempt=attempt,
            status_sha256=data["status_sha256"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Measurement status schema is malformed. / 测量状态 schema 格式错误。") from error


def validate_measurement_status(
    status: PersistedMeasurementStatus,
    *,
    expected_task: SessionTaskManifest,
    expected_config: ExperimentCellConfig,
) -> None:
    """Validate checksums, task identity, and honest completion flags.

    验证校验码、任务身份与诚实的完成标记。
    """

    validate_session_task_for_config(expected_task, expected_config)
    if not isinstance(status, PersistedMeasurementStatus):
        raise TypeError("status has the wrong type. / status 类型错误。")
    if status.schema_version != STATUS_SCHEMA_VERSION:
        raise ValueError("Measurement status version is unsupported. / 测量状态版本不支持。")
    outcome = status.scientific_outcome
    attempt = status.attempt
    if outcome.schema_version != SCIENTIFIC_SCHEMA_VERSION:
        raise ValueError("Scientific outcome version is unsupported. / 科学状态版本不支持。")
    if attempt.schema_version != ATTEMPT_SCHEMA_VERSION:
        raise ValueError("Attempt version is unsupported. / attempt 版本不支持。")
    if _sha256_json(_scientific_without_checksum(outcome)) != outcome.outcome_sha256:
        raise ValueError("Scientific outcome checksum failed. / 科学状态校验失败。")
    if _sha256_json(_status_without_checksum(status)) != status.status_sha256:
        raise ValueError("Measurement status checksum failed. / 测量状态校验失败。")
    if (
        outcome.task_id != expected_task.task_id
        or outcome.task_sha256 != expected_task.task_sha256
        or outcome.run_config_sha256 != expected_task.run_config_sha256
        or outcome.session_index != expected_task.session_index
        or outcome.implementation_tree_sha256
        != LOADED_IMPLEMENTATION_TREE_SHA256
        or outcome.measurement_sink_protocol_id
        != MEASUREMENT_PIPELINE_PROTOCOL_ID
    ):
        raise ValueError("Measurement status belongs to another task/build. / 测量状态属于另一任务或源码。")
    if outcome.status not in ALLOWED_STATUSES:
        raise ValueError("Measurement status value is invalid. / 测量状态值无效。")
    if outcome.measurement_periods_required != expected_config.measurement_periods_required:
        raise ValueError("Measurement target differs from config. / 测量目标与 config 不同。")
    counters = (
        outcome.training_periods_verified,
        outcome.measurement_periods_required,
        outcome.committed_measurement_rows,
        attempt.attempt_number,
        attempt.training_periods_at_start,
        attempt.training_periods_at_end,
        attempt.measurement_rows_delivered_this_attempt,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counters
    ):
        raise ValueError("A measurement counter is invalid. / 某个测量计数无效。")
    if attempt.attempt_number < 1:
        raise ValueError("Attempt number must be positive. / attempt number 必须为正数。")
    if attempt.start_mode not in (
        FRESH_TRAINING_START,
        RESUMED_TRAINING_START,
        CONVERGENCE_CONTINUATION_START,
        MEASUREMENT_REPLAY_START,
    ):
        raise ValueError("Attempt start mode is invalid. / attempt 起点模式无效。")
    if (
        attempt.training_periods_at_end
        < attempt.training_periods_at_start
        or outcome.training_periods_verified
        != attempt.training_periods_at_end
        or attempt.measurement_rows_delivered_this_attempt
        > outcome.measurement_periods_required
    ):
        raise ValueError("Attempt counters contradict the scientific outcome. / attempt 计数与科学状态矛盾。")
    if (
        isinstance(attempt.elapsed_seconds, bool)
        or not isinstance(attempt.elapsed_seconds, Real)
        or not isfinite(float(attempt.elapsed_seconds))
        or attempt.elapsed_seconds < 0.0
    ):
        raise ValueError("Elapsed time is invalid. / 运行时间无效。")
    if attempt.measurement_rows_per_second is not None and (
        isinstance(attempt.measurement_rows_per_second, bool)
        or not isinstance(attempt.measurement_rows_per_second, Real)
        or not isfinite(float(attempt.measurement_rows_per_second))
        or attempt.measurement_rows_per_second < 0.0
    ):
        raise ValueError("Measurement throughput is invalid. / 测量速度无效。")
    expected_rate = (
        attempt.measurement_rows_delivered_this_attempt
        / attempt.elapsed_seconds
        if attempt.measurement_rows_delivered_this_attempt > 0
        and attempt.elapsed_seconds > 0.0
        else None
    )
    if attempt.measurement_rows_per_second != expected_rate:
        raise ValueError("Measurement throughput does not match rows/time. / 测量速度与行数和时间不一致。")
    for reference in (
        outcome.convergence_origin_reference,
        outcome.complete_evidence_reference,
    ):
        if reference is not None:
            relative = PurePosixPath(reference.relative_path)
            if (
                not _is_sha256(reference.content_sha256)
                or relative.is_absolute()
                or any(part in ("", ".", "..") for part in relative.parts)
            ):
                raise ValueError("Artifact reference is invalid. / artifact 引用无效。")
    if outcome.status == COMPLETE:
        if not (
            outcome.phase == MEASUREMENT_COMPLETE_PHASE
            and outcome.committed_measurement_rows
            == outcome.measurement_periods_required
            and outcome.convergence_origin_reference is not None
            and outcome.complete_evidence_reference is not None
            and not outcome.replay_entire_window_required_after_failure
        ):
            raise ValueError("COMPLETE status lacks complete evidence. / COMPLETE 状态缺少完整证据。")
        if (
            outcome.stop_reason
            != "complete_measurement_evidence_committed"
            or attempt.measurement_rows_delivered_this_attempt
            != outcome.measurement_periods_required
        ):
            raise ValueError("COMPLETE status has inconsistent completion metadata. / COMPLETE 状态的完成 metadata 不一致。")
    elif outcome.status == INCOMPLETE:
        if not (
            outcome.phase == TRAINING_PHASE
            and outcome.stop_reason == "training_not_yet_converged"
            and outcome.committed_measurement_rows == 0
            and outcome.convergence_origin_reference is None
            and outcome.complete_evidence_reference is None
            and not outcome.replay_entire_window_required_after_failure
            and attempt.measurement_rows_delivered_this_attempt == 0
        ):
            raise ValueError("INCOMPLETE status is internally inconsistent. / INCOMPLETE 状态内部不一致。")
    else:
        has_origin = outcome.convergence_origin_reference is not None
        if not (
            outcome.stop_reason == "unexpected_exception"
            and outcome.committed_measurement_rows == 0
            and outcome.complete_evidence_reference is None
            and outcome.replay_entire_window_required_after_failure
            == has_origin
            and (
                (has_origin and outcome.phase == MEASUREMENT_REPLAY_REQUIRED_PHASE)
                or (not has_origin and outcome.phase == TRAINING_PHASE)
            )
        ):
            raise ValueError("FAILED status is internally inconsistent. / FAILED 状态内部不一致。")
    failed = outcome.status == FAILED
    if failed:
        valid_failure_metadata = (
            attempt.failure_type is not None
            and attempt.failure_message is not None
            and attempt.live_runtime_discarded_after_failure
        )
    else:
        valid_failure_metadata = (
            attempt.failure_type is None
            and attempt.failure_message is None
            and not attempt.live_runtime_discarded_after_failure
        )
    if not valid_failure_metadata:
        raise ValueError("Failure discard flag is inconsistent. / 失败丢弃标记不一致。")
    if (
        outcome.mid_measurement_resume_supported
        or outcome.research_result
        or outcome.paper_results_ready
    ):
        raise ValueError("Step 36E cannot claim a paper result. / Step 36E 不能声称论文结果。")


def save_measurement_status(
    status: PersistedMeasurementStatus,
    path: Path,
    *,
    expected_task: SessionTaskManifest,
    expected_config: ExperimentCellConfig,
) -> Path:
    validate_measurement_status(
        status,
        expected_task=expected_task,
        expected_config=expected_config,
    )
    text = json.dumps(
        _json_ready(status),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _atomic_binary_write(path, text.encode("utf-8"))
    return path


def load_measurement_status(
    path: Path,
    *,
    expected_task: SessionTaskManifest,
    expected_config: ExperimentCellConfig,
) -> PersistedMeasurementStatus:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read complete measurement status. / 无法读取完整测量状态。") from error
    status = _status_from_dictionary(data)
    validate_measurement_status(
        status,
        expected_task=expected_task,
        expected_config=expected_config,
    )
    return status


def _task_directory(artifact_root: Path, task: SessionTaskManifest) -> Path:
    if not isinstance(artifact_root, Path):
        raise TypeError("artifact_root must be pathlib.Path. / artifact_root 必须是 pathlib.Path。")
    relative = PurePosixPath(task.relative_artifact_directory)
    if relative.is_absolute() or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise ValueError("Task artifact path is unsafe. / 任务 artifact 路径不安全。")
    root = artifact_root.resolve()
    target = root.joinpath(*relative.parts).resolve()
    if target == root or root not in target.parents:
        raise ValueError("Task artifact path escapes artifact_root. / 任务 artifact 路径越界。")
    return target


def _safe_artifact_path(artifact_root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise ValueError("Artifact reference path is unsafe. / artifact 引用路径不安全。")
    root = artifact_root.resolve()
    target = root.joinpath(*relative.parts).resolve()
    if target == root or root not in target.parents:
        raise ValueError("Artifact reference escapes artifact_root. / artifact 引用路径越界。")
    return target


def _safe_task_child_path(
    task_directory: Path,
    *parts: str,
) -> Path:
    """Resolve one child and reject a nested symlink/junction escape.

    解析一个子路径，并拒绝通过内部 symlink/junction 越出 task 目录。
    """

    task_root = task_directory.resolve()
    target = task_root.joinpath(*parts).resolve()
    if target == task_root or task_root not in target.parents:
        raise ValueError(
            "Artifact child path escapes the task directory. / "
            "artifact 子路径越出 task 目录。"
        )
    return target


def _relative_reference(
    path: Path,
    artifact_root: Path,
    content_sha256: str,
) -> ArtifactReference:
    return ArtifactReference(
        relative_path=path.resolve().relative_to(
            artifact_root.resolve()
        ).as_posix(),
        content_sha256=content_sha256,
    )


def _read_timer(timer: Timer) -> float:
    value = timer()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("timer must return a real number. / timer 必须返回实数。")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError("timer must return a finite number. / timer 必须返回有限数。")
    return converted


def _build_convergence_origin(
    *,
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    checkpoint: ConvergedMarketCheckpoint,
    replay_source: TrainingCheckpointReference | None,
) -> ConvergenceReplayOrigin:
    payload = checkpoint.payload
    draft = ConvergenceReplayOrigin(
        schema_version=ORIGIN_SCHEMA_VERSION,
        plan_sha256=plan.plan_sha256,
        experiment_cell_sha256=plan.experiment_cell_sha256,
        run_config_sha256=plan.run_config_sha256,
        task_id=task.task_id,
        task_sha256=task.task_sha256,
        session_index=task.session_index,
        implementation_tree_sha256=LOADED_IMPLEMENTATION_TREE_SHA256,
        measurement_sink_protocol_id=MEASUREMENT_PIPELINE_PROTOCOL_ID,
        replay_mid_training_checkpoint=replay_source,
        convergence_period_index=(
            payload.convergence_receipt.convergence_period_index
        ),
        origin_global_period=payload.origin_global_period,
        measurement_periods_required=plan.config.measurement_periods_required,
        checkpoint=checkpoint,
        restart_entire_measurement_after_interruption=True,
        mid_measurement_resume_supported=False,
        origin_sha256="",
    )
    return replace(
        draft,
        origin_sha256=_digest_dataclass(draft, "origin_sha256"),
    )


def validate_convergence_origin(
    origin: ConvergenceReplayOrigin,
    *,
    expected_plan: ExperimentCellPlan,
    expected_task: SessionTaskManifest,
) -> None:
    """Prove a saved origin belongs to this exact planned task.

    证明保存原点属于这个精确的计划任务。
    """

    validate_experiment_cell_plan(expected_plan)
    validate_session_task_for_config(expected_task, expected_plan.config)
    if not isinstance(origin, ConvergenceReplayOrigin):
        raise TypeError("origin has the wrong type. / origin 类型错误。")
    if origin.schema_version != ORIGIN_SCHEMA_VERSION:
        raise ValueError("Convergence-origin version is unsupported. / 收敛原点版本不支持。")
    if _digest_dataclass(origin, "origin_sha256") != origin.origin_sha256:
        raise ValueError("Convergence-origin checksum failed. / 收敛原点校验失败。")
    if (
        origin.plan_sha256 != expected_plan.plan_sha256
        or origin.experiment_cell_sha256
        != expected_plan.experiment_cell_sha256
        or origin.run_config_sha256 != expected_plan.run_config_sha256
        or origin.task_id != expected_task.task_id
        or origin.task_sha256 != expected_task.task_sha256
        or origin.session_index != expected_task.session_index
        or origin.implementation_tree_sha256
        != LOADED_IMPLEMENTATION_TREE_SHA256
        or origin.measurement_sink_protocol_id
        != MEASUREMENT_PIPELINE_PROTOCOL_ID
    ):
        raise ValueError("Convergence origin belongs to another task/build. / 收敛原点属于另一任务或源码。")
    verify_converged_market_checkpoint(origin.checkpoint)
    payload = origin.checkpoint.payload
    receipt = payload.convergence_receipt
    if (
        payload.parameters != expected_plan.config.parameters
        or payload.seed_manifest != expected_task.seed_manifest
        or payload.implementation_tree_sha256
        != LOADED_IMPLEMENTATION_TREE_SHA256
        or receipt.required_unchanged_periods
        != expected_plan.config.convergence_periods_required
        or receipt.convergence_period_index
        != origin.convergence_period_index
        or receipt.training_periods_completed != origin.origin_global_period
        or payload.origin_global_period != origin.origin_global_period
        or origin.measurement_periods_required
        != expected_plan.config.measurement_periods_required
    ):
        raise ValueError("Convergence-origin scientific context is inconsistent. / 收敛原点科学环境不一致。")
    source = origin.replay_mid_training_checkpoint
    if source is not None:
        if (
            source.period_number < 1
            or source.period_number >= origin.origin_global_period
            or not _is_sha256(source.checkpoint_sha256)
        ):
            raise ValueError("Replay source is not strictly pre-convergence. / 重放起点不是严格的收敛前时点。")
    if not (
        origin.restart_entire_measurement_after_interruption
        and not origin.mid_measurement_resume_supported
    ):
        raise ValueError("Convergence-origin scope flags are dishonest. / 收敛原点范围标记不诚实。")


def save_convergence_origin(
    origin: ConvergenceReplayOrigin,
    path: Path,
    *,
    expected_plan: ExperimentCellPlan,
    expected_task: SessionTaskManifest,
) -> Path:
    validate_convergence_origin(
        origin,
        expected_plan=expected_plan,
        expected_task=expected_task,
    )
    return _save_immutable_bundle(origin, path, ORIGIN_FILE_MAGIC)


def load_convergence_origin(
    path: Path,
    *,
    expected_plan: ExperimentCellPlan,
    expected_task: SessionTaskManifest,
    trusted_local_file: bool = False,
) -> ConvergenceReplayOrigin:
    if not trusted_local_file:
        raise ValueError("Set trusted_local_file=True for project-created bundle data. / 项目自建 bundle 数据需设 trusted_local_file=True。")
    origin = _load_bundle(path, ORIGIN_FILE_MAGIC, ConvergenceReplayOrigin)
    if not isinstance(origin, ConvergenceReplayOrigin):
        raise RuntimeError("Decoded convergence origin has the wrong type. / 解码收敛原点类型错误。")
    validate_convergence_origin(
        origin,
        expected_plan=expected_plan,
        expected_task=expected_task,
    )
    return origin


def _origin_path(
    task_directory: Path,
    origin: ConvergenceReplayOrigin,
) -> Path:
    return _safe_task_child_path(
        task_directory,
        ORIGIN_DIRECTORY_NAME,
        f"origin_{origin.origin_sha256[:20]}.bundle",
    )


def _discover_origin(
    task_directory: Path,
    *,
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
) -> tuple[ConvergenceReplayOrigin, Path] | None:
    directory = _safe_task_child_path(
        task_directory,
        ORIGIN_DIRECTORY_NAME,
    )
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("origin_*.bundle"))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError("Expected exactly one convergence origin. / 应当恰好只有一个收敛原点。")
    path = candidates[0]
    resolved_directory = directory.resolve()
    resolved_path = path.resolve()
    if resolved_directory not in resolved_path.parents:
        raise ValueError("Convergence-origin file escapes its directory. / 收敛原点文件越出所属目录。")
    origin = load_convergence_origin(
        resolved_path,
        expected_plan=plan,
        expected_task=task,
        trusted_local_file=True,
    )
    expected_name = f"origin_{origin.origin_sha256[:20]}.bundle"
    if resolved_path.name != expected_name:
        raise ValueError("Origin filename and content digest disagree. / origin 文件名与内容摘要不同。")
    return origin, resolved_path


def _load_verified_replay_checkpoint(
    source: TrainingCheckpointReference,
    *,
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    artifact_root: Path,
) -> MidTrainingCheckpoint:
    """Open and verify the exact Step-36C file named by a replay reference.

    打开并核对 replay 引用指定的精确 Step-36C 文件。
    """

    source_path = _safe_artifact_path(
        artifact_root,
        source.relative_path,
    )
    checkpoint = load_mid_training_checkpoint(
        source_path,
        expected_task=task,
        expected_config=plan.config,
        expected_measurement_sink_protocol_id=(
            MEASUREMENT_PIPELINE_PROTOCOL_ID
        ),
        trusted_local_file=True,
    )
    if (
        checkpoint.payload.period_number != source.period_number
        or checkpoint.checkpoint_sha256 != source.checkpoint_sha256
    ):
        raise ValueError(
            "Replay source reference differs from its file. / "
            "重放起点引用与文件不同。"
        )
    return checkpoint


def _controller_from_replay_source(
    source: TrainingCheckpointReference | None,
    *,
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    artifact_root: Path,
    factory: MeasurementPipelineFactory,
) -> SessionPhaseController:
    """Build period zero or restore one verified pre-convergence checkpoint.

    从第 0 期新建，或恢复一份已核对的收敛前 checkpoint。
    """

    if source is None:
        return build_fresh_training_controller(
            plan.config,
            task,
            measurement_sink_protocol_id=MEASUREMENT_PIPELINE_PROTOCOL_ID,
            measurement_sink_factory=factory,
        )
    checkpoint = _load_verified_replay_checkpoint(
        source,
        plan=plan,
        task=task,
        artifact_root=artifact_root,
    )
    return restore_mid_training_controller(
        checkpoint,
        expected_task=task,
        expected_config=plan.config,
        expected_measurement_sink_protocol_id=(
            MEASUREMENT_PIPELINE_PROTOCOL_ID
        ),
        measurement_sink_factory=factory,
    )


def _persist_and_reload_convergence_origin(
    *,
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    checkpoint: ConvergedMarketCheckpoint,
    replay_source: TrainingCheckpointReference | None,
    artifact_root: Path,
    task_directory: Path,
) -> tuple[ConvergenceReplayOrigin, Path, ArtifactReference]:
    """Verify replay durability, then atomically publish and reload the origin.

    先验证重放起点确实可读取，再原子发布并重读收敛原点。
    """

    if replay_source is not None:
        _load_verified_replay_checkpoint(
            replay_source,
            plan=plan,
            task=task,
            artifact_root=artifact_root,
        )
    built = _build_convergence_origin(
        plan=plan,
        task=task,
        checkpoint=checkpoint,
        replay_source=replay_source,
    )
    path = _origin_path(task_directory, built)
    save_convergence_origin(
        built,
        path,
        expected_plan=plan,
        expected_task=task,
    )
    loaded = load_convergence_origin(
        path,
        expected_plan=plan,
        expected_task=task,
        trusted_local_file=True,
    )
    if loaded != built:
        raise RuntimeError(
            "Reloaded convergence origin differs. / 重读收敛原点不同。"
        )
    if replay_source is not None:
        # The origin is now durable, so explicitly protect the exact Step-36C
        # file it names while applying the bounded retention policy. / origin
        # 现在已经可靠落盘；执行有限保留策略时，明确保护它所引用的 Step-36C
        # 文件。
        replay_path = _safe_artifact_path(
            artifact_root,
            replay_source.relative_path,
        )
        prune_training_checkpoints(
            replay_path.parent,
            protected_checkpoint_paths=(replay_path,),
        )
    reference = _relative_reference(
        path,
        artifact_root,
        loaded.origin_sha256,
    )
    return loaded, path, reference


def _reconstruct_terminal_training_convergence(
    training_scientific: object,
    *,
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    artifact_root: Path,
    factory: MeasurementPipelineFactory,
) -> tuple[
    SessionPhaseController,
    SessionMeasurementPipeline,
    ConvergedMarketCheckpoint,
    TrainingCheckpointReference | None,
]:
    """Recover a Step-36D terminal status created before 36E saved an origin.

    恢复一个在 Step 36E 保存 origin 之前就已终止的 Step-36D 状态。

    Step 36D intentionally stores no live Python objects.  We therefore replay
    its exact latest pre-convergence checkpoint (or period zero), and require
    both convergence period and checkpoint digest to match the terminal status.
    / Step 36D 故意不保存 live Python 对象；因此这里从其精确的收敛前
    checkpoint（或第 0 期）重放，并要求收敛时期与 checkpoint 摘要都和
    终态记录完全一致。
    """

    source = training_scientific.latest_mid_training_checkpoint
    target_period = training_scientific.verified_training_periods
    expected_digest = training_scientific.converged_checkpoint_sha256
    if expected_digest is None:
        raise RuntimeError("Terminal convergence digest is missing. / 终态收敛摘要丢失。")
    controller = _controller_from_replay_source(
        source,
        plan=plan,
        task=task,
        artifact_root=artifact_root,
        factory=factory,
    )
    while controller.phase is SessionPhase.TRAINING:
        if controller.training_periods_completed >= target_period:
            raise RuntimeError(
                "Terminal convergence could not be replayed at its recorded period. / "
                "无法在记录时期重放终态收敛。"
            )
        controller.run_next_period()
    if (
        controller.phase is not SessionPhase.MEASUREMENT
        or controller.measurement_periods_completed != 0
        or controller.training_periods_completed != target_period
        or factory.current is None
    ):
        raise RuntimeError("Terminal convergence replay is inconsistent. / 终态收敛重放不一致。")
    checkpoint = factory.current.capture_and_bind_origin(controller)
    if checkpoint.checkpoint_sha256 != expected_digest:
        raise RuntimeError(
            "Terminal convergence replay differs from Step 36D. / "
            "终态收敛重放与 Step 36D 不同。"
        )
    return controller, factory.current, checkpoint, source


def _replay_to_convergence_origin(
    origin: ConvergenceReplayOrigin,
    *,
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    artifact_root: Path,
) -> tuple[
    SessionPhaseController,
    SessionMeasurementPipeline,
    ConvergedMarketCheckpoint,
]:
    """Rebuild fresh scorers and deterministically replay to the saved origin.

    重建全新 scorer，并确定性重放到已保存原点。
    """

    validate_convergence_origin(
        origin,
        expected_plan=plan,
        expected_task=task,
    )
    factory = MeasurementPipelineFactory()
    source = origin.replay_mid_training_checkpoint
    controller = _controller_from_replay_source(
        source,
        plan=plan,
        task=task,
        artifact_root=artifact_root,
        factory=factory,
    )
    while controller.phase is SessionPhase.TRAINING:
        if controller.training_periods_completed >= origin.origin_global_period:
            raise RuntimeError("Replay did not converge at the recorded period. / 重放未在记录时期收敛。")
        controller.run_next_period()
    if (
        controller.phase is not SessionPhase.MEASUREMENT
        or controller.measurement_periods_completed != 0
        or controller.training_periods_completed != origin.origin_global_period
        or factory.current is None
    ):
        raise RuntimeError("Replayed convergence boundary is inconsistent. / 重放收敛边界不一致。")
    replay_checkpoint = factory.current.capture_and_bind_origin(controller)
    if replay_checkpoint != origin.checkpoint:
        raise RuntimeError("Replay convergence state differs from saved origin. / 重放收敛状态与已保存原点不同。")
    return controller, factory.current, replay_checkpoint


def _validate_common_measurement_receipts(
    evidence: CompleteMeasurementEvidence,
    *,
    expected_plan: ExperimentCellPlan,
    expected_task: SessionTaskManifest,
) -> None:
    config = expected_plan.config
    origin = evidence.convergence_origin
    parameters = config.parameters
    value_grid = origin.checkpoint.payload.value_grid
    number_of_agents = parameters.num_speculators
    validate_convergence_origin(
        origin,
        expected_plan=expected_plan,
        expected_task=expected_task,
    )
    validate_irf_long_run_baseline_receipt(
        evidence.irf_long_run_baseline_receipt
    )
    phase = evidence.phase_receipt
    receipts = (
        evidence.profitability_receipt,
        evidence.trading_intensity_receipt,
        evidence.price_informativeness_receipt,
        evidence.market_liquidity_receipt,
        evidence.mispricing_receipt,
        evidence.irf_long_run_baseline_receipt,
    )
    expected_count = config.measurement_periods_required
    first_global = origin.origin_global_period
    last_global = first_global + expected_count - 1
    for receipt in receipts:
        if (
            receipt.measurement_periods_scored != expected_count
            or receipt.first_measurement_index != 0
            or receipt.last_measurement_index != expected_count - 1
            or receipt.first_global_period_index != first_global
            or receipt.last_global_period_index != last_global
        ):
            raise ValueError("A scorer receipt has inconsistent measurement bounds. / 某 scorer receipt 的测量边界不一致。")
    if (
        phase.measurement_periods_required != expected_count
        or phase.measurement_periods_completed != expected_count
        or phase.measurement_first_period_index != first_global
        or phase.measurement_last_period_index != last_global
        or phase.total_session_periods_completed != first_global + expected_count
        or phase.convergence_receipt != origin.checkpoint.payload.convergence_receipt
        or evidence.irf_long_run_baseline_receipt.session_phase_receipt
        != phase
        or not phase.q_learning_disabled
        or not phase.exploration_disabled
        or not phase.market_maker_remained_adaptive
        or phase.exact_tie_rule
        != origin.checkpoint.payload.exact_tie_rule
    ):
        raise ValueError("Step 28 and scorer measurement receipts disagree. / Step 28 与 scorer 测量 receipt 不同。")
    manifests = (
        evidence.profitability_receipt.session_seed_manifest,
        evidence.trading_intensity_receipt.session_seed_manifest,
        evidence.price_informativeness_receipt.session_seed_manifest,
        evidence.market_liquidity_receipt.session_seed_manifest,
        evidence.mispricing_receipt.session_seed_manifest,
        evidence.irf_long_run_baseline_receipt.session_seed_manifest,
    )
    if any(manifest != expected_task.seed_manifest for manifest in manifests):
        raise ValueError("A scorer receipt belongs to another seed manifest. / 某 scorer receipt 属于另一种子。")

    profitability = evidence.profitability_receipt
    trading = evidence.trading_intensity_receipt
    informativeness = evidence.price_informativeness_receipt
    liquidity = evidence.market_liquidity_receipt
    mispricing = evidence.mispricing_receipt
    irf = evidence.irf_long_run_baseline_receipt

    # Every scorer must describe the same economic environment.  Checking only
    # row counts would allow individually plausible receipts from different
    # sessions to be wrapped together. / 每个 scorer 必须描述同一个经济环境；
    # 只检查行数会让来自不同 session、各自看似合理的 receipt 被错误拼在一起。
    if (
        trading.number_of_agents != number_of_agents
        or informativeness.number_of_agents != number_of_agents
        or mispricing.number_of_agents != number_of_agents
        or irf.number_of_agents != number_of_agents
        or len(profitability.mean_actual_profits) != number_of_agents
        or len(profitability.delta_by_agent) != number_of_agents
        or len(trading.intercept_by_agent) != number_of_agents
        or len(trading.slope_by_agent) != number_of_agents
        or len(trading.theory_restriction_residual_by_agent)
        != number_of_agents
        or len(informativeness.slope_by_agent) != number_of_agents
        or len(mispricing.slope_by_agent) != number_of_agents
        or len(irf.mean_oriented_order_by_agent) != number_of_agents
        or len(irf.mean_raw_order_by_agent) != number_of_agents
        or len(irf.mean_profit_by_agent) != number_of_agents
    ):
        raise ValueError("Scorer agent dimensions disagree. / scorer 的 agent 维度不同。")
    if (
        trading.parameter_snapshot != parameters
        or liquidity.parameter_snapshot != parameters
        or mispricing.parameter_snapshot != parameters
        or irf.parameter_snapshot != parameters
        or origin.checkpoint.payload.parameters != parameters
        or trading.value_grid_snapshot != value_grid
        or informativeness.value_grid != value_grid
        or mispricing.value_grid_snapshot != value_grid
        or irf.value_grid_snapshot != value_grid
        or informativeness.value_grid_points != len(value_grid)
        or liquidity.investor_slope_xi != parameters.investor_slope
        or informativeness.continuous_value_std_parameter
        != parameters.value_std
        or informativeness.noise_std != parameters.noise_std
        or irf.value_mean_parameter != parameters.value_mean
        or mispricing.value_mean != parameters.value_mean
    ):
        raise ValueError("Scorer parameter/grid provenance disagrees. / scorer 的参数或网格来源不同。")

    # Step 29 can be reconstructed from its compact receipt. / Step 29 可由
    # 自己的紧凑 receipt 重新计算。
    expected_benchmarks = build_matched_path_benchmarks(
        parameters,
        value_grid,
    )
    if (
        trading.average_trading_intensity
        != fsum(trading.slope_by_agent) / number_of_agents
        or trading.discrete_value_std_snapshot
        != expected_benchmarks.discrete_fundamental_std
        or trading.value_mean_parameter != parameters.value_mean
        or not trading.unrestricted_intercept_estimated
        or not trading.actual_raw_orders_used
    ):
        raise ValueError("Step 30 receipt is internally inconsistent. / Step 30 receipt 内部不一致。")
    normalized = normalize_collusion_profitability(
        profitability.mean_actual_profits,
        profitability.mean_nash_profit,
        profitability.mean_cartel_profit,
    )
    if (
        profitability.benchmark_coefficients != expected_benchmarks
        or profitability.normalization_denominator
        != normalized.normalization_denominator
        or profitability.denominator_numerical_floor
        != normalized.denominator_numerical_floor
        or profitability.delta_by_agent != normalized.delta_by_agent
        or profitability.delta_c != normalized.delta_c
        or not profitability.theoretical_actions_remained_continuous
        or not profitability.adaptive_ols_excluded_from_benchmarks
    ):
        raise ValueError("Step 29 receipt is internally inconsistent. / Step 29 receipt 内部不一致。")

    # Step 31 and Step 33 are derived from the very same Step-30 slopes; they
    # are not independent estimates. / Step 31 与 Step 33 都来自同一份
    # Step-30 斜率，并不是彼此独立的估计。
    expected_information = calculate_price_informativeness(
        number_of_agents,
        trading.average_trading_intensity,
        trading.discrete_value_std_snapshot,
        parameters.noise_std,
    )
    if (
        informativeness.slope_by_agent != trading.slope_by_agent
        or informativeness.average_trading_intensity
        != trading.average_trading_intensity
        or informativeness.aggregate_informed_slope
        != expected_information.aggregate_informed_slope
        or informativeness.discrete_value_std
        != expected_information.discrete_value_std
        or informativeness.standard_deviation_ratio
        != expected_information.standard_deviation_ratio
        or informativeness.informed_flow_variance
        != expected_information.informed_flow_variance
        or informativeness.noise_order_variance
        != expected_information.noise_order_variance
        or informativeness.price_informativeness
        != expected_information.price_informativeness
        or informativeness.source_step30_estimator_version
        != trading.estimator_version
        or informativeness.source_step30_regression_specification
        != trading.regression_specification
        or not informativeness.uses_discrete_value_grid_std
        or not informativeness.uses_configured_noise_std
        or mispricing.slope_by_agent != trading.slope_by_agent
        or mispricing.average_trading_intensity
        != trading.average_trading_intensity
        or mispricing.aggregate_informed_slope
        != number_of_agents * trading.average_trading_intensity
        or mispricing.source_step30_estimator_version
        != trading.estimator_version
        or mispricing.source_step30_regression_specification
        != trading.regression_specification
    ):
        raise ValueError("Steps 30, 31, and 33 do not share one fitted policy. / Steps 30、31、33 未共享同一拟合策略。")
    reported_sum_is_none = mispricing.reported_mispricing_sum is None
    reported_average_is_none = (
        mispricing.reported_average_mispricing is None
    )
    if (
        reported_sum_is_none != reported_average_is_none
        or mispricing.requires_explicit_research_decision
        != (reported_sum_is_none and reported_average_is_none)
        or (
            not reported_sum_is_none
            and mispricing.reported_average_mispricing
            != mispricing.reported_mispricing_sum / expected_count
        )
    ):
        raise ValueError("Step 33 research-decision flag disagrees with its reported values. / Step 33 研究决策标记与报告值不一致。")
    if (
        irf.source_checkpoint_sha256
        != origin.checkpoint.checkpoint_sha256
        or irf.source_implementation_tree_sha256
        != LOADED_IMPLEMENTATION_TREE_SHA256
    ):
        raise ValueError("Step 35C provenance differs from the convergence origin. / Step 35C 来源与收敛原点不同。")


def _exact_paper_scale_counts_matched(config: ExperimentCellConfig) -> bool:
    """Return whether every count needed by this step is paper scale.

    返回本步骤涉及的全部数量是否都达到论文正式规模。
    """

    return (
        config.convergence_periods_required == PAPER_UNCHANGED_PERIODS
        and config.measurement_periods_required == PAPER_MEASUREMENT_PERIODS
        and config.session_count == PAPER_SESSIONS_PER_EXPERIMENT_CELL
        and config.irf_paths_per_session == PAPER_PATHS_PER_SESSION
        and config.mechanism_analysis_enabled
    )


def _build_learned_result(
    *,
    task: SessionTaskManifest,
    config: ExperimentCellConfig,
    origin: ConvergenceReplayOrigin,
    receipts: _FinalizedPipelineReceipts,
) -> LearnedSessionResult:
    exact_paper_counts = _exact_paper_scale_counts_matched(config)
    draft = LearnedSessionResult(
        schema_version=RESULT_SCHEMA_VERSION,
        task_id=task.task_id,
        session_index=task.session_index,
        noise_std=config.parameters.noise_std,
        training_periods_completed=origin.origin_global_period,
        measurement_periods_completed=config.measurement_periods_required,
        mean_actual_profit_by_agent=(
            receipts.profitability.mean_actual_profits
        ),
        mean_nash_profit=receipts.profitability.mean_nash_profit,
        mean_cartel_profit=receipts.profitability.mean_cartel_profit,
        delta_c=receipts.profitability.delta_c,
        trading_intensity=(
            receipts.trading_intensity.average_trading_intensity
        ),
        price_informativeness=(
            receipts.price_informativeness.price_informativeness
        ),
        average_market_liquidity=(
            receipts.market_liquidity.average_market_liquidity
        ),
        reported_average_mispricing=(
            receipts.mispricing.reported_average_mispricing
        ),
        mispricing_requires_research_decision=(
            receipts.mispricing.requires_explicit_research_decision
        ),
        mean_irf_oriented_price=(
            receipts.irf_long_run_baseline.mean_oriented_price
        ),
        mean_irf_oriented_order_by_agent=(
            receipts.irf_long_run_baseline.mean_oriented_order_by_agent
        ),
        mean_irf_profit_by_agent=(
            receipts.irf_long_run_baseline.mean_profit_by_agent
        ),
        measurement_scored_fields_sha256=(
            receipts.irf_long_run_baseline.scored_fields_sha256
        ),
        exact_paper_scale_counts_matched=exact_paper_counts,
        # A23 is resolved by the versioned, explicit execution/result source
        # manifests stored in every Step-36B plan and task. / A23 已由写入每份
        # Step-36B plan/task 的带版本、明确 execution/result 源码清单解决。
        a23_source_fingerprint_scope_resolved=True,
        a24_distributed_evidence_bridge_resolved=False,
        research_result=False,
        paper_results_ready=False,
        result_sha256="",
    )
    return replace(
        draft,
        result_sha256=_digest_dataclass(draft, "result_sha256"),
    )


def _build_complete_evidence(
    *,
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    origin: ConvergenceReplayOrigin,
    phase_receipt: SessionPhaseReceipt,
    receipts: _FinalizedPipelineReceipts,
) -> CompleteMeasurementEvidence:
    result = _build_learned_result(
        task=task,
        config=plan.config,
        origin=origin,
        receipts=receipts,
    )
    draft = CompleteMeasurementEvidence(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        plan_sha256=plan.plan_sha256,
        experiment_cell_sha256=plan.experiment_cell_sha256,
        run_config_sha256=plan.run_config_sha256,
        task_id=task.task_id,
        task_sha256=task.task_sha256,
        session_index=task.session_index,
        implementation_tree_sha256=LOADED_IMPLEMENTATION_TREE_SHA256,
        measurement_sink_protocol_id=MEASUREMENT_PIPELINE_PROTOCOL_ID,
        convergence_origin=origin,
        phase_receipt=phase_receipt,
        profitability_receipt=receipts.profitability,
        trading_intensity_receipt=receipts.trading_intensity,
        price_informativeness_receipt=receipts.price_informativeness,
        market_liquidity_receipt=receipts.market_liquidity,
        mispricing_receipt=receipts.mispricing,
        irf_long_run_baseline_receipt=(
            receipts.irf_long_run_baseline
        ),
        learned_session_result=result,
        complete_measurement_rows_committed=(
            plan.config.measurement_periods_required
        ),
        partial_measurement_rows_committed=0,
        measurement_restart_not_mid_window_resume=True,
        step35d_live_calibration_receipt_persisted=False,
        a24_full_bundle_available=False,
        research_result=False,
        paper_results_ready=False,
        evidence_sha256="",
    )
    return replace(
        draft,
        evidence_sha256=_digest_dataclass(draft, "evidence_sha256"),
    )


def validate_complete_measurement_evidence(
    evidence: CompleteMeasurementEvidence,
    *,
    expected_plan: ExperimentCellPlan,
    expected_task: SessionTaskManifest,
) -> None:
    """Validate the complete bundle and every cross-component boundary.

    验证完整 bundle 以及所有跨组件边界。
    """

    validate_experiment_cell_plan(expected_plan)
    validate_session_task_for_config(expected_task, expected_plan.config)
    if not isinstance(evidence, CompleteMeasurementEvidence):
        raise TypeError("evidence has the wrong type. / evidence 类型错误。")
    if evidence.schema_version != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Evidence version is unsupported. / evidence 版本不支持。")
    if _digest_dataclass(evidence, "evidence_sha256") != evidence.evidence_sha256:
        raise ValueError("Complete-evidence checksum failed. / 完整 evidence 校验失败。")
    if (
        evidence.plan_sha256 != expected_plan.plan_sha256
        or evidence.experiment_cell_sha256
        != expected_plan.experiment_cell_sha256
        or evidence.run_config_sha256 != expected_plan.run_config_sha256
        or evidence.task_id != expected_task.task_id
        or evidence.task_sha256 != expected_task.task_sha256
        or evidence.session_index != expected_task.session_index
        or evidence.implementation_tree_sha256
        != LOADED_IMPLEMENTATION_TREE_SHA256
        or evidence.measurement_sink_protocol_id
        != MEASUREMENT_PIPELINE_PROTOCOL_ID
    ):
        raise ValueError("Evidence belongs to another task/build. / evidence 属于另一任务或源码。")
    _validate_common_measurement_receipts(
        evidence,
        expected_plan=expected_plan,
        expected_task=expected_task,
    )
    result = evidence.learned_session_result
    if (
        result.schema_version != RESULT_SCHEMA_VERSION
        or _digest_dataclass(result, "result_sha256") != result.result_sha256
        or result.task_id != expected_task.task_id
        or result.session_index != expected_task.session_index
        or result.noise_std != expected_plan.config.parameters.noise_std
        or result.training_periods_completed
        != evidence.convergence_origin.origin_global_period
        or result.measurement_periods_completed
        != expected_plan.config.measurement_periods_required
        or result.mean_actual_profit_by_agent
        != evidence.profitability_receipt.mean_actual_profits
        or result.mean_nash_profit
        != evidence.profitability_receipt.mean_nash_profit
        or result.mean_cartel_profit
        != evidence.profitability_receipt.mean_cartel_profit
        or result.delta_c != evidence.profitability_receipt.delta_c
        or result.trading_intensity
        != evidence.trading_intensity_receipt.average_trading_intensity
        or result.price_informativeness
        != evidence.price_informativeness_receipt.price_informativeness
        or result.average_market_liquidity
        != evidence.market_liquidity_receipt.average_market_liquidity
        or result.reported_average_mispricing
        != evidence.mispricing_receipt.reported_average_mispricing
        or result.mispricing_requires_research_decision
        != evidence.mispricing_receipt.requires_explicit_research_decision
        or result.mean_irf_oriented_price
        != evidence.irf_long_run_baseline_receipt.mean_oriented_price
        or result.mean_irf_oriented_order_by_agent
        != evidence.irf_long_run_baseline_receipt.mean_oriented_order_by_agent
        or result.mean_irf_profit_by_agent
        != evidence.irf_long_run_baseline_receipt.mean_profit_by_agent
        or result.measurement_scored_fields_sha256
        != evidence.irf_long_run_baseline_receipt.scored_fields_sha256
        or result.exact_paper_scale_counts_matched
        != _exact_paper_scale_counts_matched(expected_plan.config)
    ):
        raise ValueError("Learned result and component receipts disagree. / 学习结果与组件 receipt 不同。")
    if not (
        evidence.complete_measurement_rows_committed
        == expected_plan.config.measurement_periods_required
        and evidence.partial_measurement_rows_committed == 0
        and evidence.measurement_restart_not_mid_window_resume
        and not evidence.step35d_live_calibration_receipt_persisted
        and not evidence.a24_full_bundle_available
        and not evidence.research_result
        and not evidence.paper_results_ready
        and result.a23_source_fingerprint_scope_resolved
        and not result.a24_distributed_evidence_bridge_resolved
        and not result.research_result
        and not result.paper_results_ready
    ):
        raise ValueError("Step 36E honesty flags are inconsistent. / Step 36E 诚实标记不一致。")


def save_complete_measurement_evidence(
    evidence: CompleteMeasurementEvidence,
    path: Path,
    *,
    expected_plan: ExperimentCellPlan,
    expected_task: SessionTaskManifest,
) -> Path:
    validate_complete_measurement_evidence(
        evidence,
        expected_plan=expected_plan,
        expected_task=expected_task,
    )
    return _save_immutable_bundle(evidence, path, EVIDENCE_FILE_MAGIC)


def load_complete_measurement_evidence(
    path: Path,
    *,
    expected_plan: ExperimentCellPlan,
    expected_task: SessionTaskManifest,
    trusted_local_file: bool = False,
) -> CompleteMeasurementEvidence:
    if not trusted_local_file:
        raise ValueError("Set trusted_local_file=True for project-created bundle data. / 项目自建 bundle 数据需设 trusted_local_file=True。")
    evidence = _load_bundle(
        path,
        EVIDENCE_FILE_MAGIC,
        CompleteMeasurementEvidence,
    )
    if not isinstance(evidence, CompleteMeasurementEvidence):
        raise RuntimeError("Decoded evidence has the wrong type. / 解码 evidence 类型错误。")
    validate_complete_measurement_evidence(
        evidence,
        expected_plan=expected_plan,
        expected_task=expected_task,
    )
    return evidence


def _evidence_path(
    task_directory: Path,
    evidence: CompleteMeasurementEvidence,
) -> Path:
    return _safe_task_child_path(
        task_directory,
        MEASUREMENT_DIRECTORY_NAME,
        f"evidence_{evidence.evidence_sha256[:20]}.bundle",
    )


def _attempt_metadata(
    *,
    attempt_number: int,
    start_mode: str,
    training_start: int,
    training_end: int,
    rows: int,
    elapsed: float,
    failure: Exception | None,
) -> MeasurementAttemptMetadata:
    rate = rows / elapsed if rows > 0 and elapsed > 0.0 else None
    failure_type = (
        None
        if failure is None
        else getattr(
            failure,
            "reported_failure_type",
            type(failure).__name__,
        )
    )
    failure_message = (
        None
        if failure is None
        else str(
            getattr(
                failure,
                "reported_failure_message",
                str(failure),
            )
        )[:1000]
    )
    return MeasurementAttemptMetadata(
        schema_version=ATTEMPT_SCHEMA_VERSION,
        attempt_number=attempt_number,
        start_mode=start_mode,
        training_periods_at_start=training_start,
        training_periods_at_end=training_end,
        measurement_rows_delivered_this_attempt=rows,
        elapsed_seconds=elapsed,
        measurement_rows_per_second=rate,
        failure_type=failure_type,
        failure_message=failure_message,
        live_runtime_discarded_after_failure=(failure is not None),
    )


def _load_complete_from_status(
    status: PersistedMeasurementStatus,
    *,
    artifact_root: Path,
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
) -> CompleteMeasurementEvidence:
    reference = status.scientific_outcome.complete_evidence_reference
    origin_reference = status.scientific_outcome.convergence_origin_reference
    if reference is None:
        raise ValueError("COMPLETE status lacks evidence reference. / COMPLETE 状态缺少 evidence 引用。")
    if origin_reference is None:
        raise ValueError("COMPLETE status lacks origin reference. / COMPLETE 状态缺少 origin 引用。")
    path = _safe_artifact_path(artifact_root, reference.relative_path)
    evidence = load_complete_measurement_evidence(
        path,
        expected_plan=plan,
        expected_task=task,
        trusted_local_file=True,
    )
    if evidence.evidence_sha256 != reference.content_sha256:
        raise ValueError("Status and evidence digest disagree. / status 与 evidence 摘要不同。")
    if path.name != f"evidence_{evidence.evidence_sha256[:20]}.bundle":
        raise ValueError("Evidence filename and content digest disagree. / evidence 文件名与内容摘要不同。")
    origin_path = _safe_artifact_path(
        artifact_root,
        origin_reference.relative_path,
    )
    origin = load_convergence_origin(
        origin_path,
        expected_plan=plan,
        expected_task=task,
        trusted_local_file=True,
    )
    if (
        origin.origin_sha256 != origin_reference.content_sha256
        or origin != evidence.convergence_origin
        or origin_path.name != f"origin_{origin.origin_sha256[:20]}.bundle"
    ):
        raise ValueError("Status, origin, and evidence do not form one chain. / status、origin 与 evidence 未形成同一证据链。")
    return evidence


def complete_measurement_status_path(
    *,
    artifact_root: Path,
    task: SessionTaskManifest,
) -> Path:
    """Return the deterministic Step-36E status path for one task.

    返回一个任务确定不变的 Step-36E status 路径。
    """

    return _safe_task_child_path(
        _task_directory(artifact_root, task),
        MEASUREMENT_DIRECTORY_NAME,
        STATUS_FILE_NAME,
    )


def load_completed_measurement_evidence(
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    *,
    artifact_root: Path,
) -> CompleteMeasurementEvidence:
    """Load one authoritative COMPLETE result without rerunning the market.

    不重新运行市场，读取一个权威的 COMPLETE 结果。

    A formal core aggregator uses this read-only boundary to collect Delta C
    and market-quality metrics. It refuses missing, incomplete, failed, stale,
    or cross-task artifacts. / 正式核心汇总器通过这个只读边界收集 Delta C
    与市场质量指标；缺失、未完成、失败、过期或跨任务 artifact 都会被拒绝。
    """

    validate_experiment_cell_plan(plan)
    if task not in plan.tasks:
        raise ValueError("task is not a member of plan. / task 不属于 plan。")
    validate_session_task_for_config(task, plan.config)
    status_path = complete_measurement_status_path(
        artifact_root=artifact_root,
        task=task,
    )
    if not status_path.is_file():
        raise FileNotFoundError(
            "Complete measurement status is missing. / 完整测量 status 丢失。"
        )
    status = load_measurement_status(
        status_path,
        expected_task=task,
        expected_config=plan.config,
    )
    if status.scientific_outcome.status != COMPLETE:
        raise ValueError(
            "Measurement task is not COMPLETE. / 测量任务尚未 COMPLETE。"
        )
    return _load_complete_from_status(
        status,
        artifact_root=artifact_root,
        plan=plan,
        task=task,
    )


def reconstruct_verified_complete_measurement_runtime(
    evidence: CompleteMeasurementEvidence,
    *,
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    artifact_root: Path,
) -> ReconstructedCompleteMeasurementRuntime:
    """Rebuild the exact live scorer/checkpoint chain in a new process.

    在新的 Python 进程中重建完全相同的实时 scorer/checkpoint 证据链。

    Equality of saved numbers is not enough for Step 35D: its provenance rule
    also requires the real scorer that observed all measurement rows and the
    exact checkpoint object bound to that scorer.  This helper recreates those
    live objects, reruns all ``T`` rows, and accepts them only if the complete
    evidence is exactly equal to the persisted Step-36E evidence. / 仅有保存
    数值相等还不足以进入 Step 35D；来源规则还要求真正看过全部测量行的 scorer，
    以及与它绑定的同一个 checkpoint 对象。本函数重新建立这些实时对象、重跑
    全部 ``T`` 行，并且只有完整 evidence 与已保存 Step-36E evidence 完全相等
    时才接受。
    """

    validate_complete_measurement_evidence(
        evidence,
        expected_plan=plan,
        expected_task=task,
    )
    controller, pipeline, checkpoint = _replay_to_convergence_origin(
        evidence.convergence_origin,
        plan=plan,
        task=task,
        artifact_root=artifact_root,
    )
    rows_replayed = 0
    while controller.phase is SessionPhase.MEASUREMENT:
        observation = controller.run_next_period()
        if not isinstance(observation, FrozenPolicyPeriodObservation):
            raise RuntimeError(
                "Replayed measurement observation is missing. / "
                "重放测量 observation 丢失。"
            )
        rows_replayed += 1
    if (
        controller.phase is not SessionPhase.COMPLETE
        or controller.final_receipt is None
        or rows_replayed != plan.config.measurement_periods_required
    ):
        raise RuntimeError(
            "Complete measurement replay did not reproduce exactly T rows. / "
            "完整测量重放没有精确复现 T 条记录。"
        )
    receipts = pipeline.finalize(controller, checkpoint)
    rebuilt = _build_complete_evidence(
        plan=plan,
        task=task,
        origin=evidence.convergence_origin,
        phase_receipt=controller.final_receipt,
        receipts=receipts,
    )
    validate_complete_measurement_evidence(
        rebuilt,
        expected_plan=plan,
        expected_task=task,
    )
    if rebuilt != evidence:
        raise RuntimeError(
            "Replayed complete evidence differs from persisted Step 36E evidence. / "
            "重放的完整 evidence 与已保存 Step 36E evidence 不同。"
        )
    return ReconstructedCompleteMeasurementRuntime(
        evidence=evidence,
        controller=controller,
        pipeline=pipeline,
        convergence_checkpoint=checkpoint,
    )


def _persist_failure_status(
    *,
    error: Exception,
    existing_status: PersistedMeasurementStatus | None,
    status_path: Path,
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    origin: ConvergenceReplayOrigin | None,
    origin_reference: ArtifactReference | None,
    start_mode: str,
    training_start: int,
    training_end: int,
    rows: int,
    elapsed: float,
) -> PersistedMeasurementStatus:
    attempt_number = (
        1 if existing_status is None else existing_status.attempt.attempt_number + 1
    )
    scientific = _build_scientific_outcome(
        status=FAILED,
        phase=(
            MEASUREMENT_REPLAY_REQUIRED_PHASE
            if origin is not None
            else TRAINING_PHASE
        ),
        stop_reason="unexpected_exception",
        task=task,
        training_periods_verified=(
            training_end
            if origin is None
            else origin.origin_global_period
        ),
        measurement_periods_required=plan.config.measurement_periods_required,
        committed_measurement_rows=0,
        origin_reference=origin_reference,
        evidence_reference=None,
        replay_required=(origin is not None),
    )
    attempt = _attempt_metadata(
        attempt_number=attempt_number,
        start_mode=start_mode,
        training_start=training_start,
        training_end=training_end,
        rows=rows,
        elapsed=elapsed,
        failure=error,
    )
    persisted = _build_status(scientific, attempt)
    save_measurement_status(
        persisted,
        status_path,
        expected_task=task,
        expected_config=plan.config,
    )
    return persisted


def run_complete_measurement_task(
    plan: ExperimentCellPlan,
    task: SessionTaskManifest,
    *,
    artifact_root: Path,
    checkpoint_interval_periods: int,
    invocation_training_period_budget: int | None = None,
    retry_failed: bool = False,
    timer: Timer = perf_counter,
    measurement_period_completed_hook: (
        MeasurementPeriodCompletedHook | None
    ) = None,
) -> CompleteMeasurementExecution:
    """Train/resume one task, then commit exactly one complete T-row result.

    训练/续跑一个任务，然后只提交一份完整 T 条结果。

    A failure during measurement discards every live scorer. An explicit retry
    reconstructs fresh scorers and replays from the durable origin. / 测量中
    失败会丢弃全部实时 scorer。明确重试时会重建 scorer，并从
    可持久原点重放。
    """

    validate_experiment_cell_plan(plan)
    if not isinstance(task, SessionTaskManifest) or task not in plan.tasks:
        raise ValueError("task is not a member of this plan. / task 不属于该计划。")
    validate_session_task_for_config(task, plan.config)
    if (
        isinstance(checkpoint_interval_periods, bool)
        or not isinstance(checkpoint_interval_periods, int)
        or checkpoint_interval_periods < 1
    ):
        raise ValueError("checkpoint_interval_periods must be positive. / checkpoint_interval_periods 必须为正数。")
    if invocation_training_period_budget is not None and (
        isinstance(invocation_training_period_budget, bool)
        or not isinstance(invocation_training_period_budget, int)
        or invocation_training_period_budget < 1
    ):
        raise ValueError("invocation_training_period_budget must be positive. / invocation_training_period_budget 必须为正数。")
    if not isinstance(retry_failed, bool):
        raise TypeError("retry_failed must be bool. / retry_failed 必须是 bool。")
    if not callable(timer):
        raise TypeError("timer must be callable. / timer 必须可调用。")
    if (
        measurement_period_completed_hook is not None
        and not callable(measurement_period_completed_hook)
    ):
        raise TypeError("measurement_period_completed_hook must be callable. / measurement hook 必须可调用。")

    task_directory = _task_directory(artifact_root, task)
    status_path = _safe_task_child_path(
        task_directory,
        MEASUREMENT_DIRECTORY_NAME,
        STATUS_FILE_NAME,
    )
    existing_status = (
        load_measurement_status(
            status_path,
            expected_task=task,
            expected_config=plan.config,
        )
        if status_path.exists()
        else None
    )
    if existing_status is not None:
        if existing_status.scientific_outcome.status == COMPLETE:
            evidence = _load_complete_from_status(
                existing_status,
                artifact_root=artifact_root,
                plan=plan,
                task=task,
            )
            return CompleteMeasurementExecution(
                status=existing_status,
                evidence=evidence,
                controller=None,
                pipeline=None,
                convergence_checkpoint=None,
                status_path=status_path,
            )
        if (
            existing_status.scientific_outcome.status == FAILED
            and not retry_failed
        ):
            return CompleteMeasurementExecution(
                status=existing_status,
                evidence=None,
                controller=None,
                pipeline=None,
                convergence_checkpoint=None,
                status_path=status_path,
            )

    # Load and validate immutable origins before running a timer, sink, or
    # market period. Corruption therefore fails without mutating artifacts.
    # / 在调用 timer、sink 或运行市场期之前，先读取并验证
    # 不可变原点；因此损坏会在不改动 artifact 的情况下失败。
    discovered_origin = _discover_origin(
        task_directory,
        plan=plan,
        task=task,
    )
    origin = None if discovered_origin is None else discovered_origin[0]
    origin_path = None if discovered_origin is None else discovered_origin[1]
    origin_reference = (
        None
        if origin is None or origin_path is None
        else _relative_reference(
            origin_path,
            artifact_root,
            origin.origin_sha256,
        )
    )

    started_at = _read_timer(timer)
    controller: SessionPhaseController | None = None
    pipeline: SessionMeasurementPipeline | None = None
    convergence_checkpoint: ConvergedMarketCheckpoint | None = None
    training_start = 0
    training_end = 0
    rows_delivered = 0
    start_mode = FRESH_TRAINING_START
    attempt_number = (
        1 if existing_status is None else existing_status.attempt.attempt_number + 1
    )

    try:
        if origin is not None:
            source = origin.replay_mid_training_checkpoint
            training_start = 0 if source is None else source.period_number
            # The durable origin already proves the verified training boundary,
            # even if reconstructing a disposable live runtime later fails. /
            # 即使之后重建临时 live runtime 失败，可靠 origin 已经证明训练边界。
            training_end = origin.origin_global_period
            (
                controller,
                pipeline,
                convergence_checkpoint,
            ) = _replay_to_convergence_origin(
                origin,
                plan=plan,
                task=task,
                artifact_root=artifact_root,
            )
            start_mode = MEASUREMENT_REPLAY_START
        else:
            factory = MeasurementPipelineFactory()
            captured: dict[str, object] = {}

            def persist_origin_before_converged_status(
                converged_controller: SessionPhaseController,
                runner_checkpoint: ConvergedMarketCheckpoint,
                replay_source: TrainingCheckpointReference | None,
            ) -> ConvergedMarketCheckpoint:
                nonlocal origin, origin_path, origin_reference
                if factory.current is None:
                    raise RuntimeError("Measurement pipeline was not created. / 测量管线未建立。")
                bound_checkpoint = factory.current.capture_and_bind_origin(
                    converged_controller
                )
                if bound_checkpoint != runner_checkpoint:
                    raise RuntimeError("Step 35C and Step 36D convergence checkpoints differ. / Step 35C 与 Step 36D 收敛 checkpoint 不同。")
                (
                    origin,
                    origin_path,
                    origin_reference,
                ) = _persist_and_reload_convergence_origin(
                    plan=plan,
                    task=task,
                    checkpoint=bound_checkpoint,
                    replay_source=replay_source,
                    artifact_root=artifact_root,
                    task_directory=task_directory,
                )
                captured["pipeline"] = factory.current
                captured["checkpoint"] = bound_checkpoint
                return bound_checkpoint

            try:
                training_execution: TrainingRunExecution = run_training_task(
                    plan,
                    task,
                    artifact_root=artifact_root,
                    checkpoint_interval_periods=checkpoint_interval_periods,
                    invocation_period_budget=invocation_training_period_budget,
                    measurement_sink_protocol_id=(
                        MEASUREMENT_PIPELINE_PROTOCOL_ID
                    ),
                    measurement_sink_factory=factory,
                    retry_failed=retry_failed,
                    # Step 36D records its own operational attempt time. The
                    # injected Step-36E timer is deliberately read only at this
                    # outer runner's start/end, which makes timing tests exact.
                    # / Step 36D 自己记录训练 attempt 时间；注入的 Step-36E
                    # timer 只在外层 runner 起止时读取。
                    timer=perf_counter,
                    convergence_handoff_hook=(
                        persist_origin_before_converged_status
                    ),
                )
            except TrainingTaskExecutionError as upstream_error:
                upstream_status = upstream_error.status
                training_start = (
                    upstream_status.attempt.starting_training_period
                )
                training_end = (
                    upstream_status.scientific_outcome.verified_training_periods
                )
                start_mode = (
                    FRESH_TRAINING_START
                    if upstream_status.attempt.start_mode == "fresh"
                    else RESUMED_TRAINING_START
                )
                raise _UpstreamTrainingFailure(upstream_status) from upstream_error
            training_scientific = training_execution.status.scientific_outcome
            training_start = (
                training_execution.status.attempt.starting_training_period
            )
            training_end = training_scientific.verified_training_periods
            start_mode = (
                FRESH_TRAINING_START
                if training_execution.status.attempt.start_mode == "fresh"
                else RESUMED_TRAINING_START
            )
            terminal_convergence_replayed = False
            if training_scientific.status == FAILED:
                raise _UpstreamTrainingFailure(training_execution.status)
            if (
                training_scientific.status == CONVERGED
                and training_execution.controller is None
                and training_execution.converged_checkpoint is None
            ):
                # A user may have run Step 36D separately before invoking 36E.
                # Its terminal JSON has no live Python objects, so deterministically
                # rebuild the same boundary and then create the missing durable
                # origin. / 用户可能先单独运行了 Step 36D；终态 JSON 不含 live
                # Python 对象，因此这里确定性重建同一边界，再补建可靠 origin。
                (
                    rebuilt_controller,
                    rebuilt_pipeline,
                    rebuilt_checkpoint,
                    replay_source,
                ) = _reconstruct_terminal_training_convergence(
                    training_scientific,
                    plan=plan,
                    task=task,
                    artifact_root=artifact_root,
                    factory=factory,
                )
                (
                    origin,
                    origin_path,
                    origin_reference,
                ) = _persist_and_reload_convergence_origin(
                    plan=plan,
                    task=task,
                    checkpoint=rebuilt_checkpoint,
                    replay_source=replay_source,
                    artifact_root=artifact_root,
                    task_directory=task_directory,
                )
                captured["pipeline"] = rebuilt_pipeline
                captured["checkpoint"] = rebuilt_checkpoint
                training_execution = replace(
                    training_execution,
                    controller=rebuilt_controller,
                    converged_checkpoint=rebuilt_checkpoint,
                )
                terminal_convergence_replayed = True
            if training_scientific.status == INCOMPLETE:
                ended_at = _read_timer(timer)
                if ended_at < started_at:
                    raise ValueError("timer moved backwards. / timer 倒退。")
                scientific = _build_scientific_outcome(
                    status=INCOMPLETE,
                    phase=TRAINING_PHASE,
                    stop_reason="training_not_yet_converged",
                    task=task,
                    training_periods_verified=training_end,
                    measurement_periods_required=(
                        plan.config.measurement_periods_required
                    ),
                    committed_measurement_rows=0,
                    origin_reference=None,
                    evidence_reference=None,
                    replay_required=False,
                )
                attempt = _attempt_metadata(
                    attempt_number=attempt_number,
                    start_mode=start_mode,
                    training_start=training_start,
                    training_end=training_end,
                    rows=0,
                    elapsed=ended_at - started_at,
                    failure=None,
                )
                persisted = _build_status(scientific, attempt)
                save_measurement_status(
                    persisted,
                    status_path,
                    expected_task=task,
                    expected_config=plan.config,
                )
                return CompleteMeasurementExecution(
                    status=persisted,
                    evidence=None,
                    controller=training_execution.controller,
                    pipeline=factory.current,
                    convergence_checkpoint=None,
                    status_path=status_path,
                )
            if (
                training_scientific.status != CONVERGED
                or training_execution.controller is None
                or training_execution.converged_checkpoint is None
                or origin is None
                or origin_path is None
                or origin_reference is None
                or captured.get("pipeline") is not factory.current
                or captured.get("checkpoint")
                is not training_execution.converged_checkpoint
            ):
                raise RuntimeError("Converged training lacks its durable live handoff. / 收敛训练缺少可持久实时交接。")
            controller = training_execution.controller
            pipeline = factory.current
            convergence_checkpoint = (
                training_execution.converged_checkpoint
            )
            training_end = origin.origin_global_period
            start_mode = (
                MEASUREMENT_REPLAY_START
                if terminal_convergence_replayed
                else CONVERGENCE_CONTINUATION_START
            )

        if (
            origin is None
            or origin_reference is None
            or controller is None
            or pipeline is None
            or convergence_checkpoint is None
            or controller.phase is not SessionPhase.MEASUREMENT
            or controller.measurement_periods_completed != 0
        ):
            raise RuntimeError("Measurement origin is not ready. / 测量原点尚未就绪。")

        while controller.phase is SessionPhase.MEASUREMENT:
            observation = controller.run_next_period()
            if not isinstance(observation, FrozenPolicyPeriodObservation):
                raise RuntimeError("Measurement observation is missing. / 测量 observation 丢失。")
            rows_delivered += 1
            if measurement_period_completed_hook is not None:
                measurement_period_completed_hook(controller, observation)
        if (
            controller.phase is not SessionPhase.COMPLETE
            or controller.final_receipt is None
            or rows_delivered != plan.config.measurement_periods_required
        ):
            raise RuntimeError("Measurement did not complete exactly T rows. / 测量未精确完成 T 条。")

        receipts = pipeline.finalize(
            controller,
            convergence_checkpoint,
        )
        evidence = _build_complete_evidence(
            plan=plan,
            task=task,
            origin=origin,
            phase_receipt=controller.final_receipt,
            receipts=receipts,
        )
        validate_complete_measurement_evidence(
            evidence,
            expected_plan=plan,
            expected_task=task,
        )
        evidence_path = _evidence_path(task_directory, evidence)
        save_complete_measurement_evidence(
            evidence,
            evidence_path,
            expected_plan=plan,
            expected_task=task,
        )
        loaded_evidence = load_complete_measurement_evidence(
            evidence_path,
            expected_plan=plan,
            expected_task=task,
            trusted_local_file=True,
        )
        if loaded_evidence != evidence:
            raise RuntimeError("Reloaded evidence differs. / 重读 evidence 不同。")
        evidence_reference = _relative_reference(
            evidence_path,
            artifact_root,
            evidence.evidence_sha256,
        )
        ended_at = _read_timer(timer)
        if ended_at < started_at:
            raise ValueError("timer moved backwards. / timer 倒退。")
        scientific = _build_scientific_outcome(
            status=COMPLETE,
            phase=MEASUREMENT_COMPLETE_PHASE,
            stop_reason="complete_measurement_evidence_committed",
            task=task,
            training_periods_verified=origin.origin_global_period,
            measurement_periods_required=(
                plan.config.measurement_periods_required
            ),
            committed_measurement_rows=(
                plan.config.measurement_periods_required
            ),
            origin_reference=origin_reference,
            evidence_reference=evidence_reference,
            replay_required=False,
        )
        attempt = _attempt_metadata(
            attempt_number=attempt_number,
            start_mode=start_mode,
            training_start=training_start,
            training_end=origin.origin_global_period,
            rows=rows_delivered,
            elapsed=ended_at - started_at,
            failure=None,
        )
        persisted = _build_status(scientific, attempt)
        save_measurement_status(
            persisted,
            status_path,
            expected_task=task,
            expected_config=plan.config,
        )
        return CompleteMeasurementExecution(
            status=persisted,
            evidence=loaded_evidence,
            controller=controller,
            pipeline=pipeline,
            convergence_checkpoint=convergence_checkpoint,
            status_path=status_path,
        )

    except Exception as error:
        # Never serialize a partly advanced controller or scorer. / 绝不
        # 序列化已部分推进的 controller 或 scorer。
        try:
            ended_at = _read_timer(timer)
            elapsed = max(0.0, ended_at - started_at)
        except Exception:
            elapsed = 0.0
        if origin is None:
            rediscovered = _discover_origin(
                task_directory,
                plan=plan,
                task=task,
            )
            if rediscovered is not None:
                origin, origin_path = rediscovered
                origin_reference = _relative_reference(
                    origin_path,
                    artifact_root,
                    origin.origin_sha256,
                )
        failed_status = _persist_failure_status(
            error=error,
            existing_status=existing_status,
            status_path=status_path,
            plan=plan,
            task=task,
            origin=origin,
            origin_reference=origin_reference,
            start_mode=start_mode,
            training_start=training_start,
            training_end=training_end,
            rows=rows_delivered,
            elapsed=elapsed,
        )
        raise CompleteMeasurementTaskError(
            failed_status,
            status_path,
        ) from error


def main() -> None:
    """Run one tiny complete low-noise measurement demonstration.

    运行一个小型完整低噪声测量演示。
    """

    parameters = PaperParameters(noise_std=0.1, market_maker_window=20)
    config = ExperimentCellConfig(
        mode=DEBUG_MODE,
        experiment_cell_key="step36e-debug-low-noise",
        parameters=parameters,
        experiment_seed=36_100_001,
        irf_experiment_seed=36_100_002,
        session_count=1,
        convergence_periods_required=1,
        measurement_periods_required=20,
        irf_paths_per_session=1,
    )
    plan = build_experiment_cell_plan(
        config,
        ExperimentExecutionPolicy(maximum_training_periods=20),
    )
    task = plan.tasks[0]
    artifact_root = (
        PROJECT_ROOT / "results" / "step36e_complete_measurement_runner"
    )
    execution = run_complete_measurement_task(
        plan,
        task,
        artifact_root=artifact_root,
        checkpoint_interval_periods=2,
    )
    outcome = execution.status.scientific_outcome
    result = (
        None
        if execution.evidence is None
        else execution.evidence.learned_session_result
    )
    print("Step 36E: complete measurement runner / 第 36E 步：完整测量 runner")
    print(f"Status / 状态: {outcome.status}")
    print(f"Training periods / 训练期数: {outcome.training_periods_verified}")
    print(f"Committed measurement rows / 已提交测量条数: {outcome.committed_measurement_rows}")
    if result is not None:
        print(f"Delta C / 合谋利润指标: {result.delta_c:.9f}")
        print(f"Trading intensity / 交易强度: {result.trading_intensity:.9f}")
    print(f"Status JSON / 状态文件: {execution.status_path}")
    print(
        "Boundary / 边界: complete per-session measurement evidence; "
        "Step 35D/A24/HPC and paper-result flags remain false. / 已完成单 "
        "session 测量证据；Step 35D/A24/HPC 与论文结果标记仍为 false。"
    )


if __name__ == "__main__":
    main()
