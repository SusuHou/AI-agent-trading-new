"""Step 36C: save, load, and exactly resume one mid-training session.

第 36C 步：保存、读取并精确续跑一个训练中的 session。

Run the small demonstration / 运行小型演示:
    py -3 -X utf8 steps/step_36c_exact_training_resume.py

Core validation / 核心验证:
    continuous 40 periods == 19 periods + save/load + 21 periods
    / 连续运行 40 期 == 运行 19 期 + 保存/读取 + 再运行 21 期

Strict boundary / 严格边界:
    This checkpoint supports a clean between-period boundary while the
    Q-learners are still training. It does not yet resume a partly consumed
    measurement window, run SLURM, or create a research result. / 本 checkpoint
    支持 Q-learner 仍在训练时的完整期际边界；尚不支持测量窗口中途恢复、不启动
    SLURM，也不产生科研结果。

Security boundary / 安全边界:
    The compact disk codec uses pickle only for built-in data. A SHA-256
    envelope detects accidental corruption before unpickling, but it is not
    authentication. Load only files created by this project and pass the
    explicit ``trusted_local_file=True`` acknowledgement. / 磁盘格式只用 pickle
    保存内置数据。SHA-256 外壳会在反序列化前发现意外损坏，但不构成身份认证；
    只读取本项目自己生成的文件，并明确传入 ``trusted_local_file=True``。
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from numbers import Real
from pathlib import Path
import os
import pickle
import platform
import random
import struct
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from step_21_two_independent_q_traders import InformedQTrader
from step_22_market_maker_rolling_history import MarketObservation
from step_24b_fast_rolling_ols import (
    CenteredPairStatisticsState,
    RollingMarketMakerOLS,
    RollingMarketMakerState,
)
from step_25_one_market_period import build_paper_inputs
from step_26_reproducible_random_streams import (
    RandomizedMarketSession,
    SessionRandomStreams,
    SessionSeedManifest,
    build_randomized_paper_session,
)
from step_27_convergence_tracker import (
    PolicyConvergenceTracker,
    PolicyConvergenceTrackerState,
)
from step_28_session_phases import (
    MeasurementSink,
    SessionPhase,
    SessionPhaseController,
)
from steps.step_36b_experiment_manifest import (
    DEBUG_MODE,
    ExperimentCellConfig,
    ExperimentExecutionPolicy,
    SessionTaskManifest,
    build_experiment_cell_plan,
    validate_session_task_for_config,
)
from steps.step_35a_converged_market_checkpoint import (
    ImmutableArraySnapshot,
    LOADED_IMPLEMENTATION_TREE_SHA256,
)


CHECKPOINT_SCHEMA_VERSION = "step36c-mid-training-checkpoint-v2-layered-source-scope"
WIRE_SCHEMA_VERSION = "step36c-builtins-wire-v2"
FILE_MAGIC = b"VIBE_STEP36C_TRAINING_CHECKPOINT_V2\n"
NO_MEASUREMENT_SINK_PROTOCOL = "none-debug-only-v1"
DEFAULT_DEMO_SPLIT_PERIOD = 19
DEFAULT_DEMO_TOTAL_PERIODS = 40

MeasurementSinkFactory = Callable[[RandomizedMarketSession], MeasurementSink]


@dataclass(frozen=True)
class TrainingCheckpointProtocolNotes:
    """Machine-readable honesty labels. / 机器可读的诚实边界标签。"""

    paper_requires_training_until_convergence: bool = True
    paper_specifies_checkpoint_file_format: bool = False
    checkpoint_captured_only_between_completed_periods: bool = True
    q_tables_preserved_at_float64_precision: bool = True
    all_seven_rng_states_preserved: bool = True
    exact_rolling_ols_state_preserved: bool = True
    convergence_streak_and_policy_masks_preserved: bool = True
    callbacks_and_controller_tokens_rebuilt_not_serialized: bool = True
    mid_measurement_resume_supported: bool = False
    formal_session_runner_connected: bool = False
    hpc_dispatch_connected: bool = False
    research_result: bool = False


@dataclass(frozen=True)
class MidTrainingCheckpointPayload:
    """All immutable data needed to continue one training session exactly.

    精确继续一个训练 session 所需的全部不可修改数据。
    """

    schema_version: str
    python_version: str
    python_implementation: str
    numpy_version: str
    platform_system: str
    platform_machine: str
    native_byteorder: str
    implementation_tree_sha256: str
    task_id: str
    task_sha256: str
    run_config_sha256: str
    parameters: PaperParameters
    value_grid: tuple[float, ...]
    price_grid: tuple[tuple[float, ...], ...]
    action_multipliers: tuple[float, ...]
    seed_manifest: SessionSeedManifest
    initial_state_indexes: tuple[int, int, int]
    trader_names: tuple[str, str]
    period_number: int
    previous_price: float
    previous_value: float
    current_value: float
    shared_value_visit_counts: tuple[int, ...]
    q_tables: tuple[ImmutableArraySnapshot, ImmutableArraySnapshot]
    all_seven_rng_states: tuple[object, ...]
    market_maker_state: RollingMarketMakerState
    tracker_state: PolicyConvergenceTrackerState
    measurement_periods_required: int
    measurement_sink_protocol_id: str
    source_controller_phase: str
    source_execution_mode: str
    source_full_q_validation_count: int
    protocol_notes: TrainingCheckpointProtocolNotes


@dataclass(frozen=True)
class MidTrainingCheckpoint:
    """Payload plus a corruption-detection checksum. / 主体加损坏检测校验码。"""

    payload: MidTrainingCheckpointPayload
    checkpoint_sha256: str


def _payload_digest(payload: MidTrainingCheckpointPayload) -> str:
    """Hash only built-in wire data so direct-script and imported runs agree.

    只哈希内置 wire 数据，使直接运行脚本与导入模块得到相同摘要。
    """

    digest = sha256()
    _update_canonical_digest(digest, asdict(payload))
    return digest.hexdigest()


def _update_canonical_digest(digest: object, value: object) -> None:
    """Hash values, not Python object-alias patterns. / 哈希数值内容，而不是对象共享方式。"""

    def add(tag: bytes, payload: bytes = b"") -> None:
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    if value is None:
        add(b"N")
    elif isinstance(value, bool):
        add(b"B", b"1" if value else b"0")
    elif isinstance(value, int):
        add(b"I", str(value).encode("ascii"))
    elif isinstance(value, float):
        add(b"F", struct.pack(">d", value))
    elif isinstance(value, str):
        add(b"S", value.encode("utf-8"))
    elif isinstance(value, bytes):
        add(b"Y", value)
    elif isinstance(value, tuple):
        add(b"T", len(value).to_bytes(8, "big"))
        for item in value:
            _update_canonical_digest(digest, item)
    elif isinstance(value, list):
        add(b"L", len(value).to_bytes(8, "big"))
        for item in value:
            _update_canonical_digest(digest, item)
    elif isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Checkpoint dictionary keys must be strings. / checkpoint 字典键必须是字符串。")
        add(b"D", len(value).to_bytes(8, "big"))
        for key in sorted(value):
            _update_canonical_digest(digest, key)
            _update_canonical_digest(digest, value[key])
    else:
        raise TypeError(
            f"Unsupported checkpoint digest type {type(value).__name__}. / "
            "checkpoint 摘要遇到不支持的类型。"
        )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _first_builtin_difference(
    left: object,
    right: object,
    path: str = "checkpoint",
) -> str:
    """Return one compact path for an internal parity failure. / 返回内部一致性失败的首个路径。"""

    if type(left) is not type(right):
        return f"{path}: {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: dictionary keys differ"
        for key in left:
            difference = _first_builtin_difference(
                left[key],
                right[key],
                f"{path}.{key}",
            )
            if difference:
                return difference
        return ""
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return f"{path}: lengths differ"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = _first_builtin_difference(
                left_item,
                right_item,
                f"{path}[{index}]",
            )
            if difference:
                return difference
        return ""
    if isinstance(left, float):
        if struct.pack(">d", left) != struct.pack(">d", right):
            return f"{path}: float bits differ"
        return ""
    if left != right:
        return f"{path}: values differ"
    return ""


def _validate_sink_protocol(
    controller: SessionPhaseController,
    protocol_id: str,
) -> None:
    """Keep a no-sink debug checkpoint distinct from a real metric pipeline.

    区分“无 sink 调试 checkpoint”与真实指标管线。
    """

    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError("measurement_sink_protocol_id cannot be empty. / measurement sink 协议标签不能为空。")
    if protocol_id.strip() != protocol_id:
        raise ValueError("measurement_sink_protocol_id cannot have outer spaces. / measurement sink 协议标签首尾不能有空格。")
    if controller.measurement_sink is None:
        if protocol_id != NO_MEASUREMENT_SINK_PROTOCOL:
            raise ValueError("A controller without a sink must use the debug no-sink protocol. / 无 sink controller 必须使用调试 no-sink 协议。")
    elif protocol_id == NO_MEASUREMENT_SINK_PROTOCOL:
        raise ValueError("A real sink cannot use the no-sink protocol label. / 真实 sink 不能使用 no-sink 协议标签。")


def _validate_capture_boundary(
    controller: SessionPhaseController,
    task: SessionTaskManifest,
    expected_config: ExperimentCellConfig,
    protocol_id: str,
) -> None:
    """Require a complete, causal, between-period training boundary.

    要求一个完整、因果正确的训练期际边界。
    """

    if not isinstance(controller, SessionPhaseController):
        raise TypeError("controller must be SessionPhaseController. / controller 类型错误。")
    validate_session_task_for_config(task, expected_config)
    _validate_sink_protocol(controller, protocol_id)
    session = controller.session
    tracker = controller.tracker
    if task.seed_manifest != session.streams.manifest:
        raise ValueError("Checkpoint task and live session seed identities differ. / checkpoint 任务与实时 session 种子身份不同。")
    if controller.phase is not SessionPhase.TRAINING or tracker.converged:
        raise RuntimeError("Checkpoint capture requires unfinished TRAINING. / checkpoint 必须在尚未收敛的 TRAINING 阶段保存。")
    if session.execution_mode != "training" or session.frozen_draw_source_mode is not None:
        raise RuntimeError("Live session is not in ordinary training mode. / 实时 session 不在普通训练模式。")
    if session.period_number < 1:
        raise RuntimeError("Run at least one complete period before checkpointing. / 保存前至少完整运行一期。")
    if session.parameters != expected_config.parameters:
        raise ValueError(
            "Live market parameters differ from the expected experiment cell. / "
            "实时市场参数与预期实验单元不同。"
        )
    if tracker.required_unchanged_periods != expected_config.convergence_periods_required:
        raise ValueError(
            "Live convergence target differs from the expected experiment cell. / "
            "实时收敛目标与预期实验单元不同。"
        )
    if controller.measurement_periods_required != expected_config.measurement_periods_required:
        raise ValueError(
            "Live measurement target differs from the expected experiment cell. / "
            "实时测量目标与预期实验单元不同。"
        )
    expected_value_grid, expected_price_grid, expected_action_multipliers, _, _ = (
        build_paper_inputs(expected_config.parameters)
    )
    if (
        session.value_grid != expected_value_grid
        or session.price_grid != expected_price_grid
        or session.action_multipliers != expected_action_multipliers
    ):
        raise ValueError(
            "Live grids differ from the deterministic paper inputs for this cell. / "
            "实时网格与该实验单元的确定性论文输入不同。"
        )
    if (
        session.period_number != tracker.periods_observed
        or session.period_number != sum(session.shared_value_visit_counts)
    ):
        raise RuntimeError("Session, tracker, and visit-counter clocks disagree. / session、tracker 与访问计数时钟不一致。")
    if controller.measurement_periods_completed != 0:
        raise RuntimeError("A training checkpoint cannot contain measurement rows. / 训练 checkpoint 不能含测量记录。")
    if (
        controller.measurement_first_period_index is not None
        or controller.final_receipt is not None
        or controller.failure_period_index is not None
    ):
        raise RuntimeError("Training controller contains later-phase state. / 训练 controller 含有后续阶段状态。")
    if session.full_q_validation_count != 1:
        raise RuntimeError("Q-tables must have exactly one full validation. / Q 表必须恰好完成一次全面检查。")
    if any(not trader.q_table.flags.writeable for trader in session.traders):
        raise RuntimeError("Training Q-tables must remain writable. / 训练 Q 表必须可写。")
    if not session.market_maker.is_full:
        raise RuntimeError("Market-maker history must be a full rolling window. / 做市商历史必须是完整滚动窗口。")
    if (
        session.market_maker.successful_append_count
        != session.parameters.market_maker_window + session.period_number
    ):
        raise RuntimeError("Market-maker append accounting is inconsistent. / 做市商追加计数不一致。")
    latest = session.market_maker.snapshot()[-1]
    if (
        latest.fundamental_value_v != session.previous_value
        or latest.market_price_p != session.previous_price
    ):
        raise RuntimeError("Latest maker row and market position disagree. / 最新做市商记录与市场位置不同。")
    observer = session.after_q_update_observer
    if (
        getattr(observer, "__self__", None) is not tracker
        or getattr(observer, "__func__", None)
        is not PolicyConvergenceTracker.observe_after_q_update
    ):
        raise RuntimeError("The convergence observer is not correctly bound. / 收敛 observer 绑定不正确。")
    if getattr(tracker, "_attached_session", None) is not session:
        raise RuntimeError("Tracker is not attached to this session. / tracker 未连接到此 session。")
    if getattr(session, "_phase_controller_token", None) is not getattr(
        controller,
        "_controller_token",
        None,
    ):
        raise RuntimeError("Controller ownership token is inconsistent. / controller 所有权 token 不一致。")
    if getattr(session, "_active_frozen_supplied_path", None) is not None:
        raise RuntimeError("An active disposable path cannot be checkpointed. / 活动短路径不能保存 checkpoint。")
    if session.market_maker.has_active_append_transaction:
        raise RuntimeError(
            "An active market-maker rollback transaction cannot be checkpointed. / "
            "做市商仍有活动回滚事务时不能保存 checkpoint。"
        )


def _source_audit(controller: SessionPhaseController) -> tuple[object, ...]:
    """Read exact causal state without retaining a period trace. / 读取精确因果状态。"""

    session = controller.session
    return (
        controller.phase,
        controller.measurement_periods_completed,
        controller.measurement_first_period_index,
        controller.final_receipt,
        controller.failure_period_index,
        session.period_number,
        session.previous_price,
        session.previous_value,
        session.current_value,
        tuple(session.shared_value_visit_counts),
        deepcopy(session.all_random_states()),
        session.market_maker.export_state(),
        tuple(
            (
                trader.q_table.dtype.str,
                trader.q_table.shape,
                trader.q_table.tobytes(order="C"),
                bool(trader.q_table.flags.writeable),
            )
            for trader in session.traders
        ),
        controller.tracker.export_training_state(),
    )


def capture_mid_training_checkpoint(
    controller: SessionPhaseController,
    *,
    task: SessionTaskManifest,
    expected_config: ExperimentCellConfig,
    measurement_sink_protocol_id: str = NO_MEASUREMENT_SINK_PROTOCOL,
) -> MidTrainingCheckpoint:
    """Capture one immutable checkpoint without changing the live session.

    在不改变实时 session 的情况下保存一个不可修改 checkpoint。
    """

    _validate_capture_boundary(
        controller,
        task,
        expected_config,
        measurement_sink_protocol_id,
    )
    before = _source_audit(controller)
    session = controller.session
    payload = MidTrainingCheckpointPayload(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        numpy_version=np.__version__,
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        native_byteorder=sys.byteorder,
        implementation_tree_sha256=LOADED_IMPLEMENTATION_TREE_SHA256,
        task_id=task.task_id,
        task_sha256=task.task_sha256,
        run_config_sha256=task.run_config_sha256,
        parameters=session.parameters,
        value_grid=tuple(session.value_grid),
        price_grid=tuple(tuple(float(price) for price in row) for row in session.price_grid),
        action_multipliers=tuple(session.action_multipliers),
        seed_manifest=session.streams.manifest,
        initial_state_indexes=tuple(session.initial_state_indexes),
        trader_names=(session.traders[0].name, session.traders[1].name),
        period_number=session.period_number,
        previous_price=session.previous_price,
        previous_value=session.previous_value,
        current_value=session.current_value,
        shared_value_visit_counts=tuple(session.shared_value_visit_counts),
        q_tables=(
            ImmutableArraySnapshot.capture(session.traders[0].q_table),
            ImmutableArraySnapshot.capture(session.traders[1].q_table),
        ),
        all_seven_rng_states=deepcopy(session.all_random_states()),
        market_maker_state=session.market_maker.export_state(),
        tracker_state=controller.tracker.export_training_state(),
        measurement_periods_required=controller.measurement_periods_required,
        measurement_sink_protocol_id=measurement_sink_protocol_id,
        source_controller_phase=controller.phase.value,
        source_execution_mode=session.execution_mode,
        source_full_q_validation_count=session.full_q_validation_count,
        protocol_notes=TrainingCheckpointProtocolNotes(),
    )
    checkpoint = MidTrainingCheckpoint(
        payload=payload,
        checkpoint_sha256=_payload_digest(payload),
    )
    after = _source_audit(controller)
    if before != after:
        raise RuntimeError("Checkpoint capture mutated the live session. / 保存 checkpoint 改变了实时 session。")
    verify_mid_training_checkpoint(
        checkpoint,
        expected_task=task,
        expected_config=expected_config,
        expected_measurement_sink_protocol_id=measurement_sink_protocol_id,
    )
    return checkpoint


def _validate_environment(payload: MidTrainingCheckpointPayload) -> None:
    if payload.python_version != platform.python_version():
        raise RuntimeError("Exact resume requires the saved Python version. / 精确续跑需要保存时的 Python 版本。")
    if payload.python_implementation != platform.python_implementation():
        raise RuntimeError("Exact resume requires the saved Python implementation. / 精确续跑需要保存时的 Python 实现。")
    if payload.numpy_version != np.__version__:
        raise RuntimeError("Exact resume requires the saved NumPy version. / 精确续跑需要保存时的 NumPy 版本。")
    if (
        payload.platform_system != platform.system()
        or payload.platform_machine != platform.machine()
        or payload.native_byteorder != sys.byteorder
    ):
        raise RuntimeError("Exact resume requires the saved platform and byte order. / 精确续跑需要保存时的平台与字节序。")
    if payload.implementation_tree_sha256 != LOADED_IMPLEMENTATION_TREE_SHA256:
        raise RuntimeError("Exact resume requires the saved checked source build. / 精确续跑需要保存时已核对的源码版本。")


def verify_mid_training_checkpoint(
    checkpoint: MidTrainingCheckpoint,
    *,
    expected_task: SessionTaskManifest,
    expected_config: ExperimentCellConfig,
    expected_measurement_sink_protocol_id: str,
) -> None:
    """Validate provenance, corruption checks, and causal accounting.

    核对来源、损坏校验以及因果计数。
    """

    validate_session_task_for_config(expected_task, expected_config)
    if not isinstance(checkpoint, MidTrainingCheckpoint):
        raise TypeError("checkpoint has the wrong type. / checkpoint 类型错误。")
    payload = checkpoint.payload
    if not isinstance(payload, MidTrainingCheckpointPayload):
        raise TypeError("checkpoint payload has the wrong type. / checkpoint 主体类型错误。")
    if payload.schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Checkpoint schema is unsupported. / checkpoint 格式不支持。")
    if not _is_sha256(checkpoint.checkpoint_sha256) or _payload_digest(payload) != checkpoint.checkpoint_sha256:
        raise ValueError("Checkpoint payload checksum failed. / checkpoint 主体校验失败。")
    _validate_environment(payload)
    if (
        payload.task_id != expected_task.task_id
        or payload.task_sha256 != expected_task.task_sha256
        or payload.run_config_sha256 != expected_task.run_config_sha256
        or payload.seed_manifest != expected_task.seed_manifest
    ):
        raise ValueError("Checkpoint belongs to another experiment task. / checkpoint 属于另一个实验任务。")
    if payload.parameters != expected_config.parameters:
        raise ValueError(
            "Checkpoint parameters differ from the expected experiment cell. / "
            "checkpoint 参数与预期实验单元不同。"
        )
    if payload.measurement_periods_required != expected_config.measurement_periods_required:
        raise ValueError(
            "Checkpoint measurement target differs from the expected experiment cell. / "
            "checkpoint 测量目标与预期实验单元不同。"
        )
    if payload.tracker_state.required_unchanged_periods != expected_config.convergence_periods_required:
        raise ValueError(
            "Checkpoint convergence target differs from the expected experiment cell. / "
            "checkpoint 收敛目标与预期实验单元不同。"
        )
    expected_value_grid, expected_price_grid, expected_action_multipliers, _, _ = (
        build_paper_inputs(expected_config.parameters)
    )
    if (
        payload.value_grid != expected_value_grid
        or payload.price_grid != expected_price_grid
        or payload.action_multipliers != expected_action_multipliers
    ):
        raise ValueError(
            "Checkpoint grids differ from the deterministic paper inputs for this cell. / "
            "checkpoint 网格与该实验单元的确定性论文输入不同。"
        )
    if payload.measurement_sink_protocol_id != expected_measurement_sink_protocol_id:
        raise ValueError("Measurement-sink protocol differs from the expected task pipeline. / measurement sink 协议与预期任务管线不同。")
    if (
        payload.source_controller_phase != SessionPhase.TRAINING.value
        or payload.source_execution_mode != "training"
        or payload.source_full_q_validation_count != 1
    ):
        raise ValueError("Checkpoint was not captured from ordinary training. / checkpoint 不是从普通训练阶段保存的。")
    if isinstance(payload.period_number, bool) or not isinstance(payload.period_number, int) or payload.period_number < 1:
        raise ValueError("Saved period_number is invalid. / 保存的 period_number 无效。")
    if (
        payload.tracker_state.periods_observed != payload.period_number
        or sum(payload.shared_value_visit_counts) != payload.period_number
    ):
        raise ValueError("Saved session clocks are inconsistent. / 保存的 session 时钟不一致。")
    if payload.measurement_periods_required < 1:
        raise ValueError("Saved measurement target must be positive. / 保存的测量目标必须为正。")
    scalars = (payload.previous_price, payload.previous_value, payload.current_value)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in scalars):
        raise TypeError("Saved market scalars must be real. / 保存的市场标量必须是实数。")
    if not all(isfinite(float(value)) for value in scalars):
        raise ValueError("Saved market scalars must be finite. / 保存的市场标量必须有限。")
    if payload.previous_value not in payload.value_grid or payload.current_value not in payload.value_grid:
        raise ValueError("Saved values lie outside V. / 保存价值不在 V 中。")
    if len(payload.q_tables) != 2 or len(payload.trader_names) != 2:
        raise ValueError("Exactly two trader records are required. / 必须恰好保存两位 trader。")
    if not isinstance(payload.all_seven_rng_states, tuple) or len(payload.all_seven_rng_states) != 7:
        raise ValueError("Exactly seven RNG states are required. / 必须恰好保存七条随机状态。")
    for state in payload.all_seven_rng_states:
        probe = random.Random()
        try:
            probe.setstate(deepcopy(state))
        except (TypeError, ValueError) as error:
            raise ValueError("A saved RNG state is invalid. / 某条保存随机状态无效。") from error

    q_tables = tuple(snapshot.restore(writeable=True) for snapshot in payload.q_tables)
    if any(not snapshot.source_was_writeable for snapshot in payload.q_tables):
        raise ValueError("Saved training Q-tables were not writable. / 保存的训练 Q 表并非可写。")
    manifest = payload.seed_manifest
    traders = (
        InformedQTrader(
            payload.trader_names[0],
            q_tables[0],
            manifest.trader_1_mode_seed,
            manifest.trader_1_action_tie_seed,
        ),
        InformedQTrader(
            payload.trader_names[1],
            q_tables[1],
            manifest.trader_2_mode_seed,
            manifest.trader_2_action_tie_seed,
        ),
    )
    # Rebuilding the tracker checks every saved mask against the Q argmax set.
    # / 重建 tracker 会逐项核对保存的 mask 与 Q 最优动作集合。
    PolicyConvergenceTracker.from_training_state(traders, payload.tracker_state)

    maker = RollingMarketMakerOLS.from_state(payload.market_maker_state)
    if (
        maker.successful_append_count
        != payload.parameters.market_maker_window + payload.period_number
    ):
        raise ValueError("Saved maker append count is inconsistent. / 保存的做市商追加计数不一致。")
    latest = maker.snapshot()[-1]
    if (
        latest.fundamental_value_v != payload.previous_value
        or latest.market_price_p != payload.previous_price
    ):
        raise ValueError("Saved maker history and market position disagree. / 保存的做市商历史与市场位置不同。")
    notes = payload.protocol_notes
    if not isinstance(notes, TrainingCheckpointProtocolNotes) or not all(
        (
            notes.paper_requires_training_until_convergence,
            not notes.paper_specifies_checkpoint_file_format,
            notes.checkpoint_captured_only_between_completed_periods,
            notes.q_tables_preserved_at_float64_precision,
            notes.all_seven_rng_states_preserved,
            notes.exact_rolling_ols_state_preserved,
            notes.convergence_streak_and_policy_masks_preserved,
            notes.callbacks_and_controller_tokens_rebuilt_not_serialized,
            not notes.mid_measurement_resume_supported,
            not notes.formal_session_runner_connected,
            not notes.hpc_dispatch_connected,
            not notes.research_result,
        )
    ):
        raise ValueError("Checkpoint scope labels are inconsistent. / checkpoint 范围标签不一致。")


def restore_mid_training_controller(
    checkpoint: MidTrainingCheckpoint,
    *,
    expected_task: SessionTaskManifest,
    expected_config: ExperimentCellConfig,
    expected_measurement_sink_protocol_id: str,
    measurement_sink_factory: MeasurementSinkFactory | None = None,
) -> SessionPhaseController:
    """Build detached runtime objects and reconnect them at the saved period.

    建立相互独立的新运行对象，并在保存时期重新连接。
    """

    verify_mid_training_checkpoint(
        checkpoint,
        expected_task=expected_task,
        expected_config=expected_config,
        expected_measurement_sink_protocol_id=expected_measurement_sink_protocol_id,
    )
    payload = checkpoint.payload
    if expected_measurement_sink_protocol_id == NO_MEASUREMENT_SINK_PROTOCOL:
        if measurement_sink_factory is not None:
            raise ValueError("Debug no-sink restore cannot accept a sink factory. / 调试 no-sink 恢复不能接收 sink factory。")
    elif measurement_sink_factory is None or not callable(measurement_sink_factory):
        raise ValueError("A real measurement protocol requires a sink factory. / 真实测量协议需要 sink factory。")

    q_tables = tuple(snapshot.restore(writeable=True) for snapshot in payload.q_tables)
    manifest = payload.seed_manifest
    traders = (
        InformedQTrader(
            payload.trader_names[0],
            q_tables[0],
            manifest.trader_1_mode_seed,
            manifest.trader_1_action_tie_seed,
        ),
        InformedQTrader(
            payload.trader_names[1],
            q_tables[1],
            manifest.trader_2_mode_seed,
            manifest.trader_2_action_tie_seed,
        ),
    )
    session = RandomizedMarketSession(
        parameters=payload.parameters,
        value_grid=payload.value_grid,
        price_grid=payload.price_grid,
        action_multipliers=payload.action_multipliers,
        traders=traders,
        market_maker=RollingMarketMakerOLS.from_state(payload.market_maker_state),
        shared_value_visit_counts=list(payload.shared_value_visit_counts),
        streams=SessionRandomStreams(manifest),
        initial_state_indexes=payload.initial_state_indexes,
    )
    measurement_sink = (
        None
        if measurement_sink_factory is None
        else measurement_sink_factory(session)
    )
    tracker = PolicyConvergenceTracker.from_training_state(
        session.traders,
        payload.tracker_state,
    )
    controller = SessionPhaseController.create_for_restored_training_session(
        session,
        tracker,
        measurement_periods_required=payload.measurement_periods_required,
        measurement_sink=measurement_sink,
    )
    controller.install_restored_training_position(
        period_number=payload.period_number,
        previous_price=payload.previous_price,
        previous_value=payload.previous_value,
        current_value=payload.current_value,
        all_seven_rng_states=deepcopy(payload.all_seven_rng_states),
    )

    # Recapture is an end-to-end postcondition: every causal field must rebuild
    # the exact original payload. / 再次保存作为端到端后置条件：每个因果字段都必须
    # 精确重建原主体。
    replay = capture_mid_training_checkpoint(
        controller,
        task=expected_task,
        expected_config=expected_config,
        measurement_sink_protocol_id=expected_measurement_sink_protocol_id,
    )
    if replay != checkpoint:
        difference = _first_builtin_difference(
            asdict(checkpoint),
            asdict(replay),
        )
        raise RuntimeError(
            "Restored checkpoint does not exactly replay at "
            f"{difference}. / 恢复后的 checkpoint 无法精确重放。"
        )
    return controller


def _wire_dictionary(checkpoint: MidTrainingCheckpoint) -> dict[str, object]:
    """Return only built-in values; never pickle live classes/callbacks.

    只返回内置值；绝不 pickle 实时 class 或 callback。
    """

    return {
        "wire_schema_version": WIRE_SCHEMA_VERSION,
        "checkpoint": asdict(checkpoint),
    }


def _array_snapshot_from_dictionary(data: dict[str, object]) -> ImmutableArraySnapshot:
    copied = dict(data)
    copied["shape"] = tuple(copied["shape"])
    return ImmutableArraySnapshot(**copied)


def _tracker_state_from_dictionary(data: dict[str, object]) -> PolicyConvergenceTrackerState:
    copied = dict(data)
    copied["policy_mask_shape"] = tuple(copied["policy_mask_shape"])
    return PolicyConvergenceTrackerState(**copied)


def _maker_state_from_dictionary(data: dict[str, object]) -> RollingMarketMakerState:
    copied = dict(data)
    copied["rows"] = tuple(
        MarketObservation(**row) for row in copied["rows"]
    )
    copied["demand_statistics"] = CenteredPairStatisticsState(
        **copied["demand_statistics"]
    )
    copied["value_statistics"] = CenteredPairStatisticsState(
        **copied["value_statistics"]
    )
    return RollingMarketMakerState(**copied)


def _checkpoint_from_wire_dictionary(wire: object) -> MidTrainingCheckpoint:
    """Reconstruct current classes from built-in wire data. / 从内置 wire 数据重建当前 class。"""

    if not isinstance(wire, dict) or wire.get("wire_schema_version") != WIRE_SCHEMA_VERSION:
        raise ValueError("Checkpoint wire schema is invalid. / checkpoint wire 格式无效。")
    checkpoint_data = wire.get("checkpoint")
    if not isinstance(checkpoint_data, dict):
        raise ValueError("Checkpoint wire payload is missing. / checkpoint wire 主体缺失。")
    payload_data = checkpoint_data.get("payload")
    if not isinstance(payload_data, dict):
        raise ValueError("Checkpoint payload is malformed. / checkpoint 主体格式错误。")
    data = dict(payload_data)
    try:
        data["parameters"] = PaperParameters(**data["parameters"])
        data["value_grid"] = tuple(data["value_grid"])
        data["price_grid"] = tuple(tuple(row) for row in data["price_grid"])
        data["action_multipliers"] = tuple(data["action_multipliers"])
        data["seed_manifest"] = SessionSeedManifest(**data["seed_manifest"])
        data["initial_state_indexes"] = tuple(data["initial_state_indexes"])
        data["trader_names"] = tuple(data["trader_names"])
        data["shared_value_visit_counts"] = tuple(data["shared_value_visit_counts"])
        data["q_tables"] = tuple(
            _array_snapshot_from_dictionary(item) for item in data["q_tables"]
        )
        data["all_seven_rng_states"] = tuple(data["all_seven_rng_states"])
        data["market_maker_state"] = _maker_state_from_dictionary(
            data["market_maker_state"]
        )
        data["tracker_state"] = _tracker_state_from_dictionary(
            data["tracker_state"]
        )
        data["protocol_notes"] = TrainingCheckpointProtocolNotes(
            **data["protocol_notes"]
        )
        payload = MidTrainingCheckpointPayload(**data)
        checkpoint = MidTrainingCheckpoint(
            payload=payload,
            checkpoint_sha256=checkpoint_data["checkpoint_sha256"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Checkpoint built-in data is malformed. / checkpoint 内置数据格式错误。") from error
    return checkpoint


def _file_bytes(checkpoint: MidTrainingCheckpoint) -> bytes:
    serialized = pickle.dumps(_wire_dictionary(checkpoint), protocol=5)
    serialized_sha256 = sha256(serialized).hexdigest().encode("ascii")
    return FILE_MAGIC + serialized_sha256 + b"\n" + serialized


def _atomic_staging_path(path: Path, data: bytes) -> Path:
    """Return the short deterministic sibling used for one atomic write.

    返回一次原子写入所用的短、确定性 sibling 路径。
    """

    staging_token = sha256(
        path.name.encode("utf-8") + b"\0" + data
    ).hexdigest()[:12]
    return path.with_name(f".w-{staging_token}.tmp")


def _atomic_binary_write(path: Path, data: bytes) -> None:
    """Write beside the target, close it, then atomically replace it.

    在目标旁写完并关闭文件，再原子替换。

    POSIX hosts additionally call ``fsync`` before replacement. On Windows,
    Python 3.13 ``fsync`` can block indefinitely on this workspace filesystem,
    so close + atomic replace + the outer checksum is the fail-closed path.
    / POSIX 主机还会在替换前调用 ``fsync``。本工作区的 Windows 文件系统上，
    Python 3.13 的 ``fsync`` 可能无限阻塞，因此 Windows 使用“关闭文件 + 原子
    替换 + 外层校验码”；若写入不完整，读取时会明确拒绝。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # Python 3.13's NamedTemporaryFile can spin while retrying denied random
    # names on this managed Windows filesystem. A digest-named sibling is
    # deterministic, collision-resistant, and still supports O_EXCL. / 本托管
    # Windows 文件系统可能让 Python 3.13 的 NamedTemporaryFile 在随机文件名
    # 被拒后不断重试。使用内容摘要命名的 sibling 仍可独占创建并避免碰撞。
    # Keep the staging basename short.  Appending a suffix to the already long
    # paper-checkpoint filename can cross legacy Windows MAX_PATH even when the
    # final target itself is valid.  The token binds both target name and
    # content, so distinct writes in one directory still do not share a stage.
    # / 暂存 basename 必须短；若在本来就很长的论文 checkpoint 文件名后继续
    # 加后缀，最终目标虽合法，暂存路径却可能越过 Windows MAX_PATH。token 同时
    # 绑定目标名与内容，因此同一目录中的不同写入不会共用暂存文件。
    temporary_path = _atomic_staging_path(path, data)
    created_here = False
    try:
        with temporary_path.open("xb") as temporary_file:
            created_here = True
            temporary_file.write(data)
            temporary_file.flush()
            if os.name != "nt":
                os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        created_here = False
    except FileExistsError as error:
        # A process may have died after fully closing the stage but before the
        # atomic replace.  If and only if every byte already equals this exact
        # requested write, safely finish the interrupted replace.  Partial or
        # foreign bytes still fail closed. / 进程可能在完整关闭暂存文件之后、原子
        # 替换之前退出。只有暂存内容与本次请求逐字节完全一致时，才安全完成上次
        # 替换；半截或其他内容仍明确失败。
        try:
            staged_data = temporary_path.read_bytes()
        except OSError as read_error:
            raise FileExistsError(
                "A staging file exists but cannot be verified. / "
                "暂存文件存在但无法核对。"
            ) from read_error
        if staged_data != data:
            raise FileExistsError(
                "A different or partial staging file already exists. / "
                "已存在不同或不完整的暂存文件。"
            ) from error
        os.replace(temporary_path, path)
    finally:
        if created_here and temporary_path.exists():
            temporary_path.unlink()


def save_mid_training_checkpoint(
    checkpoint: MidTrainingCheckpoint,
    path: Path,
    *,
    expected_task: SessionTaskManifest,
    expected_config: ExperimentCellConfig,
    expected_measurement_sink_protocol_id: str,
) -> Path:
    """Atomically save an immutable period checkpoint. / 原子保存不可修改的时期 checkpoint。"""

    verify_mid_training_checkpoint(
        checkpoint,
        expected_task=expected_task,
        expected_config=expected_config,
        expected_measurement_sink_protocol_id=expected_measurement_sink_protocol_id,
    )
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path. / path 必须是 pathlib.Path。")
    expected_bytes = _file_bytes(checkpoint)
    if path.exists():
        if path.read_bytes() == expected_bytes:
            return path
        raise FileExistsError("A different checkpoint already exists at this path. / 此路径已存在另一个 checkpoint。")
    _atomic_binary_write(path, expected_bytes)
    return path


def load_mid_training_checkpoint(
    path: Path,
    *,
    expected_task: SessionTaskManifest,
    expected_config: ExperimentCellConfig,
    expected_measurement_sink_protocol_id: str,
    trusted_local_file: bool = False,
) -> MidTrainingCheckpoint:
    """Check the byte envelope first, then rebuild and validate the checkpoint.

    先核对字节外壳，再重建并验证 checkpoint。
    """

    if not trusted_local_file:
        raise ValueError("Refusing pickle data without trusted_local_file=True. / 未明确 trusted_local_file=True，拒绝读取 pickle 数据。")
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path. / path 必须是 pathlib.Path。")
    try:
        file_data = path.read_bytes()
    except OSError as error:
        raise ValueError("Cannot read checkpoint file. / 无法读取 checkpoint 文件。") from error
    if not file_data.startswith(FILE_MAGIC):
        raise ValueError("Checkpoint file header is invalid. / checkpoint 文件头无效。")
    remainder = file_data[len(FILE_MAGIC):]
    try:
        digest_bytes, serialized = remainder.split(b"\n", 1)
    except ValueError as error:
        raise ValueError("Checkpoint file is truncated. / checkpoint 文件被截断。") from error
    if (
        len(digest_bytes) != 64
        or any(character not in b"0123456789abcdef" for character in digest_bytes)
        or sha256(serialized).hexdigest().encode("ascii") != digest_bytes
    ):
        raise ValueError("Checkpoint file-byte checksum failed. / checkpoint 文件字节校验失败。")
    try:
        wire = pickle.loads(serialized)
    except Exception as error:
        raise ValueError("Trusted checkpoint data could not be decoded. / 可信 checkpoint 数据无法解码。") from error
    checkpoint = _checkpoint_from_wire_dictionary(wire)
    verify_mid_training_checkpoint(
        checkpoint,
        expected_task=expected_task,
        expected_config=expected_config,
        expected_measurement_sink_protocol_id=expected_measurement_sink_protocol_id,
    )
    return checkpoint


def _build_demo_controller(
    config: ExperimentCellConfig,
    task: SessionTaskManifest,
) -> SessionPhaseController:
    """Build one demo runtime directly from its checked cell config.

    直接根据已核对的实验单元配置建立一份演示 runtime。
    """

    parameters = config.parameters
    value_grid, price_grid, action_multipliers, initial_q, prehistory = (
        build_paper_inputs(parameters)
    )
    session = build_randomized_paper_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        initial_q_table=initial_q,
        prehistory=prehistory,
        experiment_seed=task.seed_manifest.experiment_seed,
        experiment_cell_key=task.seed_manifest.experiment_cell_key,
        session_index=task.session_index,
    )
    return SessionPhaseController.create_for_fresh_session(
        session,
        convergence_periods_required=config.convergence_periods_required,
        measurement_periods_required=config.measurement_periods_required,
        measurement_sink=None,
    )


def main() -> None:
    """Prove 40 uninterrupted periods equal a real disk interruption.

    证明连续 40 期与真实磁盘中断续跑完全相同。
    """

    parameters = PaperParameters(noise_std=0.1)
    config = ExperimentCellConfig(
        mode=DEBUG_MODE,
        experiment_cell_key="step36c-debug-low-noise",
        parameters=parameters,
        experiment_seed=20260829,
        irf_experiment_seed=20260830,
        session_count=1,
        convergence_periods_required=1_000_000,
        measurement_periods_required=10,
        irf_paths_per_session=1,
        mechanism_analysis_enabled=True,
    )
    plan = build_experiment_cell_plan(
        config,
        ExperimentExecutionPolicy(
            maximum_training_periods=DEFAULT_DEMO_TOTAL_PERIODS
        ),
    )
    task = plan.tasks[0]
    uninterrupted = _build_demo_controller(config, task)
    split = _build_demo_controller(config, task)

    for _ in range(DEFAULT_DEMO_TOTAL_PERIODS):
        uninterrupted.run_next_period()
    for _ in range(DEFAULT_DEMO_SPLIT_PERIOD):
        split.run_next_period()

    checkpoint = capture_mid_training_checkpoint(
        split,
        task=task,
        expected_config=config,
    )
    output_path = (
        PROJECT_ROOT
        / "results"
        / "step36c_exact_training_resume"
        / (
            f"period_{checkpoint.payload.period_number:09d}_"
            f"{checkpoint.checkpoint_sha256[:12]}.checkpoint"
        )
    )
    save_mid_training_checkpoint(
        checkpoint,
        output_path,
        expected_task=task,
        expected_config=config,
        expected_measurement_sink_protocol_id=NO_MEASUREMENT_SINK_PROTOCOL,
    )
    loaded = load_mid_training_checkpoint(
        output_path,
        expected_task=task,
        expected_config=config,
        expected_measurement_sink_protocol_id=NO_MEASUREMENT_SINK_PROTOCOL,
        trusted_local_file=True,
    )
    resumed = restore_mid_training_controller(
        loaded,
        expected_task=task,
        expected_config=config,
        expected_measurement_sink_protocol_id=NO_MEASUREMENT_SINK_PROTOCOL,
    )
    for _ in range(DEFAULT_DEMO_TOTAL_PERIODS - DEFAULT_DEMO_SPLIT_PERIOD):
        resumed.run_next_period()

    uninterrupted_final = capture_mid_training_checkpoint(
        uninterrupted,
        task=task,
        expected_config=config,
    )
    resumed_final = capture_mid_training_checkpoint(
        resumed,
        task=task,
        expected_config=config,
    )
    assert uninterrupted_final == resumed_final

    print("Step 36C: exact training resume / 第 36C 步：精确训练续跑")
    print(f"Continuous periods / 连续运行期数: {DEFAULT_DEMO_TOTAL_PERIODS}")
    print(f"Saved after period / 保存时点: {DEFAULT_DEMO_SPLIT_PERIOD}")
    print(
        "Resumed periods / 恢复后运行期数: "
        f"{DEFAULT_DEMO_TOTAL_PERIODS - DEFAULT_DEMO_SPLIT_PERIOD}"
    )
    print(f"Saved file / 保存文件: {output_path}")
    print(f"Saved bytes / 保存字节数: {output_path.stat().st_size:,}")
    print(
        "Exact parity / 精确一致: Q tables, seven RNG states, market maker, "
        "state, visits, and convergence tracker all match. / Q 表、七条随机流、"
        "做市商、市场状态、访问计数与收敛 tracker 全部一致。"
    )
    print(
        "Boundary / 边界: training checkpoint verified; no measurement, HPC, "
        "or research result yet. / 已验证训练 checkpoint；尚无测量、HPC 或科研结果。"
    )


if __name__ == "__main__":
    main()
