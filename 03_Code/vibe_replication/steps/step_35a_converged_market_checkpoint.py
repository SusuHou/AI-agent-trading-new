"""Step 35A: capture and restore a converged market without contamination.

步骤 35A：无损保存并恢复一个已经收敛的市场，避免实验组互相污染。

Run / 运行:
    py -3 -X utf8 steps/step_35a_converged_market_checkpoint.py

What this small step does / 这一小步做什么:
    1. Wait until Step 28 has completed the convergence period and has frozen
       the learned greedy policy, but before the first measurement period.
       / 等第 28 步完成收敛期并冻结已学策略，但尚未运行第一条测量记录。
    2. Copy every piece of mutable state that can affect the future.
       / 复制所有会影响未来的可变状态。
    3. Restore two detached sessions and prove that advancing one does not
       change the other or the original. / 恢复两个脱离原 controller 的 session，
       并证明推进一个不会改变另一个或原 session。

Scope boundary / 本步边界:
    No impulse, treatment, or control rule is applied here. Step 35B will use
    these safe branches to define paired paths. / 本步不施加冲击，也不定义实验组
    或对照组；第 35B 步才会利用这些安全分支建立配对路径。

Paper facts versus implementation choices / 原文事实与实现选择:
    The paper starts impulse-response paths after convergence and retains the
    learned outcome. It does not fully specify which terminal state, rolling
    OLS memory, or RNG state is carried forward. We therefore make a lossless
    checkpoint first so later reset/freeze alternatives remain testable. The
    parity restore in this step uses the existing frozen-policy boundary.
    / 原文要求在收敛后开始脉冲响应路径并保留学习结果，但没有完整说明终点状态、
    滚动 OLS 记忆或随机流如何继承。因此我们先做无损快照，让之后仍可检验不同的
    重置/冻结方案；本步的一致性恢复使用现有的固定策略边界。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite, prod
from pathlib import Path
import platform
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
from src.source_manifests import (
    LOADED_COMBINED_SESSION_SOURCE_SHA256,
    LOADED_EXECUTION_SOURCE_SHA256,
    LOADED_RESULT_PIPELINE_SOURCE_SHA256,
    LOADED_SOURCE_SCOPE_MANIFEST_SHA256,
    SOURCE_SCOPE_MANIFEST_VERSION,
)
from step_21_two_independent_q_traders import InformedQTrader
from step_22_market_maker_rolling_history import MarketObservation
from step_24b_fast_rolling_ols import (
    RollingMarketMakerOLS,
    RollingMarketMakerState,
)
from step_25_one_market_period import build_paper_inputs
from step_14_state_representation import validate_price_grids_by_value
from step_26_reproducible_random_streams import (
    RandomizedMarketSession,
    SessionRandomStreams,
    SessionSeedManifest,
    build_randomized_paper_session,
)
from step_27_convergence_tracker import ConvergenceReceipt
from step_28_session_phases import (
    EXACT_TIE_RULE,
    SessionPhase,
    SessionPhaseController,
)


CHECKPOINT_SCHEMA_VERSION = "step35a-lossless-value-specific-grid-v4-layered-source-scope"

# Backward-compatible combined name used by existing checkpoint fields. Its
# meaning is now a versioned combination of two explicit source manifests,
# rather than every Python file under src/steps. Adding a later plot or formal
# coordinator therefore cannot invalidate multi-day training. / 现有 checkpoint
# 字段继续使用这个 combined 名称；但其含义已经改为两个带版本明确清单的组合，
# 而不是 src/steps 下所有 Python 文件。以后新增绘图或正式协调器不会使多日训练
# 无故失效。
LOADED_IMPLEMENTATION_TREE_SHA256 = (
    LOADED_COMBINED_SESSION_SOURCE_SHA256
)


@dataclass(frozen=True)
class ImmutableArraySnapshot:
    """A NumPy array stored as immutable metadata plus immutable bytes.

    把 NumPy 数组保存为不可修改的说明信息和 bytes。

    ``frozen=True`` alone is not enough if a dataclass contains a writable
    NumPy array. Bytes solve that problem and prevent two restored branches
    from sharing array memory. / 若冻结 dataclass 里面仍放可写 NumPy 数组，
    它并不真正安全。保存为 bytes 后，两个恢复分支也不会共享数组内存。
    """

    shape: tuple[int, ...]
    dtype_string: str
    c_order_bytes: bytes
    payload_sha256: str
    source_was_writeable: bool

    @classmethod
    def capture(cls, array: np.ndarray) -> "ImmutableArraySnapshot":
        """Copy one array without changing it. / 在不改变原数组的前提下复制它。"""

        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a NumPy array. / array 必须是 NumPy 数组。")
        payload = array.tobytes(order="C")
        return cls(
            shape=tuple(int(size) for size in array.shape),
            dtype_string=array.dtype.str,
            c_order_bytes=payload,
            payload_sha256=sha256(payload).hexdigest(),
            source_was_writeable=bool(array.flags.writeable),
        )

    def restore(self, *, writeable: bool) -> np.ndarray:
        """Validate the bytes, then return a new independent array.

        检查 bytes 后，返回一张全新且内存独立的数组。
        """

        if not isinstance(writeable, bool):
            raise TypeError("writeable must be bool. / writeable 必须是布尔值。")
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in self.shape
        ):
            raise ValueError("Saved array shape is invalid. / 保存的数组形状无效。")
        try:
            dtype = np.dtype(self.dtype_string)
        except TypeError as error:
            raise ValueError("Saved NumPy dtype is invalid. / 保存的 NumPy dtype 无效。") from error
        expected_bytes = prod(self.shape) * dtype.itemsize
        if len(self.c_order_bytes) != expected_bytes:
            raise ValueError("Saved array byte length is wrong. / 保存的数组字节长度错误。")
        if sha256(self.c_order_bytes).hexdigest() != self.payload_sha256:
            raise ValueError("Saved array checksum failed. / 保存的数组校验失败。")
        restored = np.frombuffer(self.c_order_bytes, dtype=dtype).reshape(
            self.shape,
            order="C",
        ).copy()
        restored.flags.writeable = writeable
        return restored


@dataclass(frozen=True)
class CheckpointProtocolNotes:
    """Machine-readable boundary between paper facts and our choices.

    用机器可读的方式区分“原文事实”和“我们的复现选择”。
    """

    paper_requires_post_convergence_origin: bool = True
    paper_requires_retaining_learned_outcome: bool = True
    paper_specifies_exact_capture_period: bool = False
    paper_specifies_start_state_carry_rule: bool = False
    paper_specifies_q_update_or_freeze_rule: bool = False
    paper_specifies_maker_history_carry_rule: bool = False
    paper_specifies_maker_update_or_freeze_rule: bool = False
    paper_specifies_rng_reset_rule: bool = False
    paper_specifies_common_random_numbers: bool = False
    paper_specifies_control_branch: bool = False
    replication_preserves_full_q_tables: bool = True
    replication_preserves_all_seven_rng_states: bool = True
    replication_preserves_exact_maker_numerical_state: bool = True
    replication_restores_detached_frozen_branch: bool = True
    capture_verified_source_not_mutated: bool = True


@dataclass(frozen=True)
class ConvergedMarketCheckpointPayload:
    """All immutable data needed to reconstruct one converged market.

    重建一个已收敛市场所需的全部不可修改数据。
    """

    schema_version: str
    python_version: str
    python_implementation: str
    numpy_version: str
    platform_system: str
    platform_machine: str
    native_byteorder: str
    implementation_tree_sha256: str
    parameters: PaperParameters
    value_grid: tuple[float, ...]
    price_grid: tuple[tuple[float, ...], ...]
    action_multipliers: tuple[float, ...]
    seed_manifest: SessionSeedManifest
    convergence_receipt: ConvergenceReceipt
    exact_tie_rule: str
    initial_state_indexes: tuple[int, int, int]
    origin_global_period: int
    irf_relative_origin: int
    previous_price: float
    previous_value: float
    current_value: float
    shared_value_visit_counts: tuple[int, ...]
    trader_names: tuple[str, str]
    q_tables: tuple[ImmutableArraySnapshot, ImmutableArraySnapshot]
    converged_policy_masks: ImmutableArraySnapshot
    frozen_policy_action_indexes: ImmutableArraySnapshot
    all_seven_rng_states: tuple[object, ...]
    market_maker_state: RollingMarketMakerState
    source_execution_mode: str
    source_controller_phase: str
    source_full_q_validation_count: int
    protocol_notes: CheckpointProtocolNotes


@dataclass(frozen=True)
class ConvergedMarketCheckpoint:
    """Payload plus a digest that detects accidental replacement/corruption.

    数据主体加总校验码，用来发现意外替换或损坏。
    """

    payload: ConvergedMarketCheckpointPayload
    checkpoint_sha256: str


def _payload_digest(payload: ConvergedMarketCheckpointPayload) -> str:
    """Hash values rather than Python's invisible object-sharing layout.

    对实际数值内容计算哈希，而不是对 Python 看不见的对象共享结构计算哈希。

    ``pickle`` preserves whether two equal objects happen to share the same
    memory reference.  Save/load can legitimately change that sharing without
    changing one economic value.  A research checkpoint digest must therefore
    depend only on the explicit data values. / ``pickle`` 会记录两个相等对象是否
    恰好共享同一内存引用；保存再读取可能改变这种共享方式，却没有改变任何经济
    数值。因此科研 checkpoint 的摘要只能取决于明确的数据内容。
    """

    digest = sha256()
    _update_canonical_digest(digest, asdict(payload))
    return digest.hexdigest()


def _update_canonical_digest(digest: object, value: object) -> None:
    """Add one value with explicit type and length tags. / 用明确类型和长度标签加入一个值。"""

    def add(tag: bytes, data: bytes = b"") -> None:
        digest.update(tag)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

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
            raise TypeError(
                "Checkpoint dictionary keys must be strings. / "
                "checkpoint 字典键必须是字符串。"
            )
        add(b"D", len(value).to_bytes(8, "big"))
        for key in sorted(value):
            _update_canonical_digest(digest, key)
            _update_canonical_digest(digest, value[key])
    else:
        raise TypeError(
            f"Unsupported checkpoint digest type {type(value).__name__}. / "
            "checkpoint 摘要遇到不支持的类型。"
        )


def _verify_checkpoint(checkpoint: ConvergedMarketCheckpoint) -> None:
    """Reject wrong versions or any payload changed after capture.

    拒绝错误版本，以及保存后被替换过的任何数据。
    """

    if not isinstance(checkpoint, ConvergedMarketCheckpoint):
        raise TypeError("checkpoint has the wrong type. / checkpoint 类型错误。")
    payload = checkpoint.payload
    if not isinstance(payload, ConvergedMarketCheckpointPayload):
        raise TypeError("checkpoint payload has the wrong type. / checkpoint 主体类型错误。")
    if payload.schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Checkpoint schema version is unsupported. / checkpoint 格式版本不支持。")
    if not isinstance(payload.price_grid, tuple) or any(
        not isinstance(row, tuple) for row in payload.price_grid
    ):
        raise ValueError(
            "Checkpoint price grids must be immutable nested tuples. / "
            "checkpoint 价格网格必须是不可修改的嵌套 tuple。"
        )
    validated_price_grids = validate_price_grids_by_value(
        payload.price_grid,
        payload.parameters.num_value_points,
        payload.parameters.num_price_points,
    )
    if validated_price_grids != payload.price_grid:
        raise ValueError(
            "Checkpoint price-grid encoding is not canonical. / "
            "checkpoint 价格网格编码不是规范格式。"
        )
    if payload.python_version != platform.python_version():
        raise RuntimeError(
            "Exact continuation requires the saved Python version. / "
            "精确续跑需要使用保存时的 Python 版本。"
        )
    if payload.numpy_version != np.__version__:
        raise RuntimeError(
            "Exact continuation requires the saved NumPy version. / "
            "精确续跑需要使用保存时的 NumPy 版本。"
        )
    if payload.python_implementation != platform.python_implementation():
        raise RuntimeError(
            "Exact continuation requires the saved Python implementation. / "
            "精确续跑需要使用保存时的 Python 实现。"
        )
    if (
        payload.platform_system != platform.system()
        or payload.platform_machine != platform.machine()
        or payload.native_byteorder != sys.byteorder
    ):
        raise RuntimeError(
            "Exact continuation requires the saved platform and byte order. / "
            "精确续跑需要使用保存时的平台与字节序。"
        )
    if (
        payload.implementation_tree_sha256
        != LOADED_IMPLEMENTATION_TREE_SHA256
    ):
        raise RuntimeError(
            "Exact continuation requires the saved checked source build. / "
            "精确续跑需要使用保存时已核对的源码版本。"
        )
    if _payload_digest(payload) != checkpoint.checkpoint_sha256:
        raise ValueError("Checkpoint checksum failed. / checkpoint 总校验失败。")


def verify_converged_market_checkpoint(
    checkpoint: ConvergedMarketCheckpoint,
) -> None:
    """Public read-only validation used by later replication steps.

    供后续复现步骤使用的公开只读验证入口。
    """

    _verify_checkpoint(checkpoint)


def _capture_source_audit(controller: SessionPhaseController) -> tuple[object, ...]:
    """Read a compact exact fingerprint without mutating the source.

    读取一份精确指纹，但不改变原 session。
    """

    session = controller.session
    return (
        controller.phase,
        controller.measurement_periods_completed,
        controller.measurement_first_period_index,
        session.period_number,
        session.execution_mode,
        session.frozen_draw_source_mode,
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
        session.frozen_policy_action_indexes_snapshot().tobytes(order="C"),
    )


def _validate_convergence_boundary(controller: SessionPhaseController) -> None:
    """Require the one safe between-period capture point.

    只允许在一个安全的“两期之间”时点保存。
    """

    if not isinstance(controller, SessionPhaseController):
        raise TypeError("controller must be SessionPhaseController. / controller 类型错误。")
    session = controller.session
    tracker = controller.tracker
    if controller.phase is not SessionPhase.MEASUREMENT:
        raise RuntimeError("Capture requires the measurement boundary. / 必须在测量阶段边界保存。")
    if controller.measurement_periods_completed != 0:
        raise RuntimeError("Capture must precede the first measurement row. / 必须在第一条测量记录之前保存。")
    if session.execution_mode != "measurement":
        raise RuntimeError("The learned policy is not frozen. / 已学策略尚未冻结。")
    if session.frozen_draw_source_mode is not None:
        raise RuntimeError(
            "The first frozen period has already selected a draw source. / "
            "固定阶段已经开始并选定了抽样来源。"
        )
    if tracker.convergence_receipt is None or tracker.converged_policy_masks is None:
        raise RuntimeError("Convergence evidence is missing. / 收敛证据丢失。")
    receipt = tracker.convergence_receipt
    expected_origin = receipt.convergence_period_index + 1
    if (
        session.period_number != expected_origin
        or controller.measurement_first_period_index != expected_origin
    ):
        raise RuntimeError("The convergence boundary is off by one. / 收敛边界出现一期偏差。")
    if session.full_q_validation_count != 1:
        raise RuntimeError("The source Q-tables lack one validation pass. / 源 Q 表未恰好完成一次全面检查。")
    if any(trader.q_table.flags.writeable for trader in session.traders):
        raise RuntimeError("Converged Q-tables must be read-only. / 收敛后的 Q 表必须只读。")
    if not session.market_maker.is_full:
        raise RuntimeError("The rolling market-maker window is not full. / 做市商滚动窗口未满。")
    expected_appends = session.parameters.market_maker_window + session.period_number
    if session.market_maker.successful_append_count != expected_appends:
        raise RuntimeError("Market-maker append accounting is inconsistent. / 做市商追加计数不一致。")
    latest_row = session.market_maker.snapshot()[-1]
    if (
        latest_row.fundamental_value_v != session.previous_value
        or latest_row.market_price_p != session.previous_price
    ):
        raise RuntimeError("Latest market-maker row does not match the completed period. / 最新历史行与刚完成时期不一致。")
    if session.previous_value not in session.value_grid or session.current_value not in session.value_grid:
        raise RuntimeError("Saved values are outside V. / 要保存的价值不在 V 网格中。")
    if not isfinite(session.previous_price):
        raise RuntimeError("Saved previous price is not finite. / 要保存的上一期价格不是有限数。")


def capture_at_convergence_boundary(
    controller: SessionPhaseController,
) -> ConvergedMarketCheckpoint:
    """Capture a lossless immutable checkpoint and prove capture is read-only.

    保存无损不可修改 checkpoint，并证明保存动作没有改变源 session。
    """

    _validate_convergence_boundary(controller)
    before = _capture_source_audit(controller)
    session = controller.session
    tracker = controller.tracker
    if tracker.convergence_receipt is None or tracker.converged_policy_masks is None:
        raise RuntimeError("Convergence evidence disappeared. / 收敛证据意外消失。")

    payload = ConvergedMarketCheckpointPayload(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        numpy_version=np.__version__,
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        native_byteorder=sys.byteorder,
        implementation_tree_sha256=LOADED_IMPLEMENTATION_TREE_SHA256,
        parameters=session.parameters,
        value_grid=tuple(session.value_grid),
        price_grid=tuple(
            tuple(float(price) for price in row)
            for row in session.price_grid
        ),
        action_multipliers=tuple(session.action_multipliers),
        seed_manifest=session.streams.manifest,
        convergence_receipt=tracker.convergence_receipt,
        exact_tie_rule=EXACT_TIE_RULE,
        initial_state_indexes=tuple(session.initial_state_indexes),
        origin_global_period=session.period_number,
        irf_relative_origin=0,
        previous_price=session.previous_price,
        previous_value=session.previous_value,
        current_value=session.current_value,
        shared_value_visit_counts=tuple(session.shared_value_visit_counts),
        trader_names=(session.traders[0].name, session.traders[1].name),
        q_tables=(
            ImmutableArraySnapshot.capture(session.traders[0].q_table),
            ImmutableArraySnapshot.capture(session.traders[1].q_table),
        ),
        converged_policy_masks=ImmutableArraySnapshot.capture(
            tracker.converged_policy_masks
        ),
        frozen_policy_action_indexes=ImmutableArraySnapshot.capture(
            session.frozen_policy_action_indexes_snapshot()
        ),
        all_seven_rng_states=deepcopy(session.all_random_states()),
        market_maker_state=session.market_maker.export_state(),
        source_execution_mode=session.execution_mode,
        source_controller_phase=controller.phase.value,
        source_full_q_validation_count=session.full_q_validation_count,
        protocol_notes=CheckpointProtocolNotes(),
    )
    checkpoint = ConvergedMarketCheckpoint(
        payload=payload,
        checkpoint_sha256=_payload_digest(payload),
    )
    after = _capture_source_audit(controller)
    if before != after:
        raise RuntimeError("Checkpoint capture mutated the source. / 保存 checkpoint 改变了源 session。")
    _verify_checkpoint(checkpoint)
    return checkpoint


def _restore_rng_states(
    session: RandomizedMarketSession,
    states: tuple[object, ...],
) -> None:
    """Restore all seven states into newly created RNG objects only.

    把七条状态恢复到刚建立的新随机对象中，不接触原 session。
    """

    if not isinstance(states, tuple) or len(states) != 7:
        raise ValueError("Exactly seven RNG states are required. / 必须恰好保存七条随机状态。")
    generators = (
        session.streams.initial_state_generator,
        session.streams.value_generator,
        session.streams.noise_generator,
        session.traders[0].mode_random_generator,
        session.traders[0].action_random_generator,
        session.traders[1].mode_random_generator,
        session.traders[1].action_random_generator,
    )
    if len({id(generator) for generator in generators}) != 7:
        raise RuntimeError("Restored RNG objects are not independent. / 恢复后的随机对象没有相互独立。")
    for generator, state in zip(generators, states, strict=True):
        generator.setstate(deepcopy(state))


def restore_detached_frozen_branch(
    checkpoint: ConvergedMarketCheckpoint,
) -> RandomizedMarketSession:
    """Restore one independent frozen-policy branch with no old callbacks.

    恢复一个独立的固定策略分支，不复制旧 controller 或 callback。
    """

    _verify_checkpoint(checkpoint)
    payload = checkpoint.payload
    if payload.source_execution_mode != "measurement" or payload.source_controller_phase != SessionPhase.MEASUREMENT.value:
        raise ValueError("Checkpoint was not captured at the frozen boundary. / checkpoint 不是在固定策略边界保存的。")
    if payload.exact_tie_rule != EXACT_TIE_RULE or payload.irf_relative_origin != 0:
        raise ValueError("Checkpoint policy convention is unsupported. / checkpoint 策略约定不支持。")
    if payload.source_full_q_validation_count != 1:
        raise ValueError("Checkpoint Q-validation provenance is invalid. / checkpoint 的 Q 检查来源无效。")

    q_tables = tuple(
        snapshot.restore(writeable=True) for snapshot in payload.q_tables
    )
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
    streams = SessionRandomStreams(manifest)
    maker = RollingMarketMakerOLS.from_state(payload.market_maker_state)
    branch = RandomizedMarketSession(
        parameters=payload.parameters,
        value_grid=payload.value_grid,
        price_grid=payload.price_grid,
        action_multipliers=payload.action_multipliers,
        traders=traders,
        market_maker=maker,
        shared_value_visit_counts=list(payload.shared_value_visit_counts),
        streams=streams,
        initial_state_indexes=payload.initial_state_indexes,
    )
    branch.period_number = payload.origin_global_period
    branch.previous_price = payload.previous_price
    branch.previous_value = payload.previous_value
    branch.current_value = payload.current_value
    _restore_rng_states(branch, payload.all_seven_rng_states)
    masks = payload.converged_policy_masks.restore(writeable=False)
    branch.begin_frozen_greedy_measurement(masks)

    # Postconditions / 恢复后的核对条件。
    expected_actions = payload.frozen_policy_action_indexes.restore(
        writeable=False
    )
    actual_actions = branch.frozen_policy_action_indexes_snapshot()
    if not np.array_equal(actual_actions, expected_actions):
        raise RuntimeError("Restored frozen actions differ. / 恢复后的固定动作不同。")
    if branch.all_random_states() != payload.all_seven_rng_states:
        raise RuntimeError("Restored RNG states differ. / 恢复后的随机状态不同。")
    if branch.market_maker.export_state() != payload.market_maker_state:
        raise RuntimeError("Restored market-maker state differs. / 恢复后的做市商状态不同。")
    if tuple(branch.shared_value_visit_counts) != payload.shared_value_visit_counts:
        raise RuntimeError("Restored visit counts differ. / 恢复后的访问计数不同。")
    for restored_trader, saved_q in zip(branch.traders, payload.q_tables, strict=True):
        restored_q = saved_q.restore(writeable=False)
        if not np.array_equal(restored_trader.q_table, restored_q):
            raise RuntimeError("Restored Q-table differs. / 恢复后的 Q 表不同。")
        if restored_trader.q_table.flags.writeable:
            raise RuntimeError("Restored converged Q-table is writable. / 恢复后的收敛 Q 表仍可写。")
    if branch.after_q_update_observer is not None:
        raise RuntimeError("An old observer leaked into the branch. / 旧 observer 泄漏到恢复分支。")
    return branch


def restore_two_independent_branches(
    checkpoint: ConvergedMarketCheckpoint,
) -> tuple[RandomizedMarketSession, RandomizedMarketSession]:
    """Restore two equal-state branches with separate mutable ownership.

    恢复两个状态相同、但所有可变对象各自独立的分支。
    """

    first = restore_detached_frozen_branch(checkpoint)
    second = restore_detached_frozen_branch(checkpoint)
    if first is second or first.market_maker is second.market_maker:
        raise RuntimeError("Branches share mutable objects. / 两个分支共享了可变对象。")
    if first.shared_value_visit_counts is second.shared_value_visit_counts:
        raise RuntimeError("Branches share visit counters. / 两个分支共享访问计数。")
    for first_trader, second_trader in zip(first.traders, second.traders, strict=True):
        if first_trader is second_trader or np.shares_memory(
            first_trader.q_table,
            second_trader.q_table,
        ):
            raise RuntimeError("Branches share trader memory. / 两个分支共享 trader 内存。")
    first_generators = (
        first.streams.initial_state_generator,
        first.streams.value_generator,
        first.streams.noise_generator,
        first.traders[0].mode_random_generator,
        first.traders[0].action_random_generator,
        first.traders[1].mode_random_generator,
        first.traders[1].action_random_generator,
    )
    second_generators = (
        second.streams.initial_state_generator,
        second.streams.value_generator,
        second.streams.noise_generator,
        second.traders[0].mode_random_generator,
        second.traders[0].action_random_generator,
        second.traders[1].mode_random_generator,
        second.traders[1].action_random_generator,
    )
    if any(left is right for left, right in zip(first_generators, second_generators, strict=True)):
        raise RuntimeError("Branches share an RNG object. / 两个分支共享随机对象。")
    return first, second


def _build_demo_boundary() -> SessionPhaseController:
    """Create a one-period convergence boundary for this validation demo.

    为本步验证建立一个“一期即收敛”的小演示边界；不是论文实验结果。
    """

    parameters = PaperParameters()
    value_grid, price_grid, actions, initial_q, prehistory = build_paper_inputs(
        parameters
    )
    stable_q = np.zeros_like(initial_q)
    stable_q[:, 0] = 1_000_000_000.0
    session = build_randomized_paper_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=actions,
        initial_q_table=stable_q,
        prehistory=prehistory,
        experiment_seed=20_260_829,
        experiment_cell_key="step35a_demo_only",
        session_index=0,
    )
    controller = SessionPhaseController.create_for_fresh_session(
        session,
        convergence_periods_required=1,
        measurement_periods_required=2,
    )
    result = controller.run_next_period()
    if result is not None or controller.phase is not SessionPhase.MEASUREMENT:
        raise RuntimeError("Demo failed to reach the capture boundary. / 演示未到达保存边界。")
    return controller


def main() -> None:
    """Demonstrate exact parity and branch independence. / 演示精确一致和分支独立。"""

    controller = _build_demo_boundary()
    source_before = _capture_source_audit(controller)
    checkpoint = capture_at_convergence_boundary(controller)
    branch_a, branch_b = restore_two_independent_branches(checkpoint)

    branch_b_before = (
        branch_b.period_number,
        branch_b.all_random_states(),
        branch_b.market_maker.export_state(),
    )
    first_a = branch_a.run_next_frozen_policy_period()
    branch_b_after_a_moves = (
        branch_b.period_number,
        branch_b.all_random_states(),
        branch_b.market_maker.export_state(),
    )
    if branch_b_before != branch_b_after_a_moves:
        raise RuntimeError("Advancing branch A changed branch B. / 推进 A 分支改变了 B 分支。")
    if _capture_source_audit(controller) != source_before:
        raise RuntimeError("Advancing a branch changed the source. / 推进分支改变了源 session。")
    first_b = branch_b.run_next_frozen_policy_period()
    if first_a != first_b:
        raise RuntimeError("Equal restored branches produced different paths. / 相同恢复分支产生了不同路径。")

    print("Step 35A: converged-market checkpoint / 步骤 35A：收敛市场快照")
    print(f"Global origin period / 原 session 起点期数: {checkpoint.payload.origin_global_period}")
    print(f"IRF local origin / 脉冲响应局部起点: {checkpoint.payload.irf_relative_origin}")
    print(f"Saved market-maker rows / 保存的做市商历史行: {len(checkpoint.payload.market_maker_state.rows):,}")
    print(f"Saved RNG states / 保存的随机流状态: {len(checkpoint.payload.all_seven_rng_states)}")
    print(f"Checkpoint digest / 快照校验码: {checkpoint.checkpoint_sha256[:16]}...")
    print("First restored periods match exactly / 两个恢复分支的第一期完全一致")
    print("Branch independence passed / 分支互不污染验证通过")
    print("Original session remained unchanged / 原 session 保持不变")
    print("No shock was applied yet / 本步尚未施加冲击")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
