"""Step 35F: run paired response paths and classify each trained session.

第 35F 步：运行配对反应路径，并逐个分类已经训练完成的 session。

Run / 运行:
    py -3 -X utf8 steps/step_35f_paired_response_and_classification.py

Paper order / 论文顺序:
    1. Step 35E chooses ONE shock magnitude for an experiment cell.
       / 第 35E 步为整个实验单元选择一个统一冲击幅度。
    2. Every trained session runs 10,000 paired continuations from its own
       convergence checkpoint. / 每个已训练 session 从自己的收敛快照运行
       10,000 条配对续接路径。
    3. Treatment alone receives the adverse noise shock at local t=3; both
       branches continue to local t=4. / 只有实验组在局部 t=3 接受逆向噪声
       冲击；两组都继续运行到局部 t=4。
    4. Within EACH session, average t=4 oriented orders across paths, normalize
       by that session's Step-35C long-run order means, then classify the
       session. / 在每个 session 内先平均 t=4 方向调整订单，再除以该 session
       自己在第 35C 步得到的长期订单均值，最后分类该 session。
    5. Across 1,000 sessions, report the shares of labels and audit that the
       executed treatment paths actually achieve the 1.2% cell-level price
       target. / 在 1,000 个 session 之间统计各标签占比，并核对实际执行的
       实验路径是否在实验单元层面实现 1.2% 价格目标。

Important non-equivalences / 重要的“不等价”:
    We do NOT classify individual paths and vote; we do NOT use the paired
    treatment-control order difference as the paper's primary response; and we
    do NOT classify one pooled experiment-cell response. / 我们不会逐路径分类
    后投票；不会把实验组减对照组的订单差当作原文主要反应；也不会把整个
    实验单元混成一个反应再分类。

Replication choices / 复现选择:
    The paper does not disclose checkpoint reset, common-random-number, or
    finite-sample calibration mechanics.  We inherit the disclosed Step-35A-E
    choices: Q/policy are frozen, each branch's rolling OLS remains adaptive,
    common ordinary draws are supplied to both branches, and every path is
    rolled back exactly to the same convergence checkpoint. / 原文没有公开快照
    重置、共同随机数和有限样本校准的具体实现。本步沿用第 35A-E 步已经公开
    的复现选择：冻结 Q/policy；每个分支的滚动 OLS 继续独立更新；两组使用相同
    的普通随机抽样；每条路径结束后精确回滚到同一收敛快照。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import fsum, isclose, isfinite
from numbers import Integral, Real
from pathlib import Path
import pickle
import sys
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from step_24_adaptive_market_maker_price import calculate_adaptive_price_impact
from step_26_reproducible_random_streams import RandomizedMarketSession
from steps.step_34_mechanism_classifier import (
    PAPER_CLASSIFIER_AGENTS,
    PAPER_PATHS_PER_SESSION,
    PAPER_RESPONSE_PERIOD,
    PAPER_SHOCK_PERIOD,
    PAPER_TARGET_PRICE_DEVIATION,
    CollusionMechanism,
    MechanismClassification,
    NormalizedOrderResponse,
    add_adverse_shock_to_noise,
    calculate_normalized_order_response,
    classify_normalized_order_responses,
    orient_order,
    orient_price,
)
from steps.step_35a_converged_market_checkpoint import (
    ConvergedMarketCheckpoint,
    restore_two_independent_branches,
    verify_converged_market_checkpoint,
)
from steps.step_35b_paired_irf_path import (
    IRF_LOCAL_PERIODS,
    PAIRED_PATH_SEED_VERSION,
    PairedPathSeedManifest,
    VerifiedPairedPathScheduleContext,
    build_paired_path_draw_schedule_from_verified_context,
    derive_paired_path_seed_manifest_from_verified_context,
    prepare_verified_paired_path_schedule_context,
    _completed_row,
)
from steps.step_35c_irf_long_run_baseline import (
    OnlineIRFLongRunBaselineScorer,
    validate_irf_long_run_baseline_receipt,
)
from steps.step_35d_unshocked_t3_calibration_paths import (
    UNSHOCKED_T3_DIGEST_DOMAIN,
    UnshockedT3SessionCalibrationReceipt,
    _completed_path_payload,
    validate_unshocked_t3_session_calibration_receipt,
)
from steps.step_35e_cell_shock_calibration import (
    ExperimentCellShockCalibrationReceipt,
    calibrate_experiment_cell_uniform_shock,
    validate_experiment_cell_shock_calibration_receipt,
)


STEP35F_PROTOCOL_VERSION = "step35f-paired-response-classification-v1"
STEP35F_PATH_DIGEST_DOMAIN = b"vibe-replication.step35f.paired-path.v1\0"
STEP35F_SESSION_RECEIPT_DOMAIN = b"vibe-replication.step35f.session-receipt.v1\0"
STEP35F_CELL_RECEIPT_DOMAIN = b"vibe-replication.step35f.cell-receipt.v1\0"
STEP35F_CONTEXT_DOMAIN = b"vibe-replication.step35f.verified-cell-context.v1\0"
STEP35F_CELL_SESSION_DIGEST_DOMAIN = b"vibe-replication.step35f.cell-sessions.v1\0"
PAIRED_RESPONSE_PERIOD_COUNT = len(IRF_LOCAL_PERIODS)


def _finite_real(number: float, label: str) -> float:
    """Return one finite non-Boolean float. / 返回有限且不是布尔值的浮点数。"""

    if isinstance(number, bool) or not isinstance(number, Real):
        raise TypeError(f"{label} must be a real number. / {label} 必须是实数。")
    converted = float(number)
    if not isfinite(converted):
        raise ValueError(f"{label} must be finite. / {label} 必须是有限数。")
    return converted


def _positive_count(number: int, label: str, maximum: int) -> int:
    """Validate a small debug or paper-scale count. / 检查调试或论文规模计数。"""

    if isinstance(number, bool) or not isinstance(number, Integral):
        raise TypeError(f"{label} must be an integer. / {label} 必须是整数。")
    checked = int(number)
    if not 1 <= checked <= maximum:
        raise ValueError(
            f"{label} must lie in [1, {maximum}]. / {label} 必须位于 [1, {maximum}]。"
        )
    return checked


def _is_sha256_text(value: object) -> bool:
    """Return whether a value is lowercase SHA-256 text. / 判断是否为小写 SHA-256。"""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _next_neumaier_sum(
    running_sum: float,
    compensation: float,
    value: float,
) -> tuple[float, float]:
    """Accurately add one number while keeping constant memory.

    用固定内存更准确地累加一个数。普通 ``sum`` 在一千万项时可能积累较多
    浮点误差；Neumaier 补偿和额外保存一个修正量。
    """

    candidate_sum = running_sum + value
    if abs(running_sum) >= abs(value):
        candidate_compensation = compensation + running_sum - candidate_sum + value
    else:
        candidate_compensation = compensation + value - candidate_sum + running_sum
    if not isfinite(candidate_sum) or not isfinite(candidate_compensation):
        raise OverflowError("A Step-35F sum overflowed. / 第 35F 步求和溢出。")
    return candidate_sum, candidate_compensation


def _mean(compensated_sum: tuple[float, float], count: int) -> float:
    """Finish one compensated mean. / 完成一个补偿均值。"""

    if count < 1:
        raise RuntimeError("Cannot average zero paths. / 不能对零条路径求平均。")
    answer = (compensated_sum[0] + compensated_sum[1]) / count
    if not isfinite(answer):
        raise OverflowError("A Step-35F mean is not finite. / 第 35F 步均值不是有限数。")
    return answer


@dataclass(frozen=True)
class VerifiedStep35FCellContext:
    """A rechecked link between Step 35E and all of its Step-35D sources.

    重新核对过的第 35E 步与全部第 35D 步来源之间的连接。

    ``frozen=True`` prevents accidental editing; it does not make the market
    agents frozen. / ``frozen=True`` 只防止凭证被意外改写，并不代表市场中的
    agent 不能变化。
    """

    protocol_version: str
    cell_calibration: ExperimentCellShockCalibrationReceipt
    ordered_source_receipts: tuple[UnshockedT3SessionCalibrationReceipt, ...]
    ordered_session_indexes: tuple[int, ...]
    source_manifest_sha256: str
    exact_recalibration_match_verified: bool
    debug_context: bool
    formal_context: bool


def _context_manifest_digest(
    calibration: ExperimentCellShockCalibrationReceipt,
    receipts: Sequence[UnshockedT3SessionCalibrationReceipt],
) -> str:
    """Bind one calibration to the exact ordered source receipts. / 绑定校准与来源。"""

    digest = sha256(STEP35F_CONTEXT_DOMAIN)
    digest.update(bytes.fromhex(calibration.receipt_payload_sha256))
    for receipt in receipts:
        digest.update(receipt.source_seed_manifest.session_index.to_bytes(8, "big"))
        digest.update(bytes.fromhex(receipt.receipt_payload_sha256))
        digest.update(bytes.fromhex(receipt.checkpoint_sha256))
    return digest.hexdigest()


def prepare_verified_step35f_cell_context(
    cell_calibration: ExperimentCellShockCalibrationReceipt,
    source_session_receipts: Sequence[UnshockedT3SessionCalibrationReceipt],
) -> VerifiedStep35FCellContext:
    """Rebuild Step 35E and require an exact match before any shocked path.

    在运行任何受冲击路径前，重新执行第 35E 步并要求结果完全相同。

    A Step-35E receipt stores an aggregate digest, not a list that proves an
    individual worker's membership.  Supplying all ordered Step-35D receipts
    closes that provenance gap. / 第 35E 步只保存聚合摘要，不能单独证明某个
    worker 属于该实验单元；因此这里重新提供全部第 35D 步 receipt 来补齐来源链。
    """

    validate_experiment_cell_shock_calibration_receipt(cell_calibration)
    if isinstance(source_session_receipts, (str, bytes)) or not isinstance(
        source_session_receipts, Sequence
    ):
        raise TypeError("source_session_receipts must be a sequence. / 来源 receipts 必须是序列。")
    supplied = tuple(source_session_receipts)
    if len(supplied) != cell_calibration.sessions_expected:
        raise ValueError("The Step-35D source count differs from Step 35E. / 第35D来源数量与第35E步不一致。")
    for receipt in supplied:
        validate_unshocked_t3_session_calibration_receipt(receipt)
    ordered = tuple(
        sorted(supplied, key=lambda item: item.source_seed_manifest.session_index)
    )
    try:
        rebuilt = calibrate_experiment_cell_uniform_shock(
            ordered,
            expected_session_count=cell_calibration.sessions_expected,
            target_normalized_price_level_deviation=(
                cell_calibration.arithmetic.target_normalized_price_level_deviation
            ),
        )
    except ArithmeticError as error:
        raise ValueError(
            "The supplied sources cannot rebuild this Step-35E calibration. / "
            "提供的来源不能重建这份第 35E 步校准。"
        ) from error
    if rebuilt != cell_calibration:
        raise ValueError(
            "The supplied Step-35D sources do not exactly rebuild Step 35E. / "
            "提供的第 35D 步来源不能精确重建第 35E 步。"
        )
    indexes = tuple(
        receipt.source_seed_manifest.session_index for receipt in ordered
    )
    formal = cell_calibration.ready_for_formal_step35f
    context = VerifiedStep35FCellContext(
        protocol_version=STEP35F_PROTOCOL_VERSION,
        cell_calibration=cell_calibration,
        ordered_source_receipts=ordered,
        ordered_session_indexes=indexes,
        source_manifest_sha256=_context_manifest_digest(cell_calibration, ordered),
        exact_recalibration_match_verified=True,
        debug_context=not formal,
        formal_context=formal,
    )
    validate_verified_step35f_cell_context(context)
    return context


def validate_verified_step35f_cell_context(
    context: VerifiedStep35FCellContext,
) -> None:
    """Reject a changed or stale prepared context. / 拒绝被修改或过期的 context。"""

    if not isinstance(context, VerifiedStep35FCellContext):
        raise TypeError("context has the wrong type. / context 类型错误。")
    if context.protocol_version != STEP35F_PROTOCOL_VERSION:
        raise ValueError("Context version is unsupported. / context 版本不支持。")
    validate_experiment_cell_shock_calibration_receipt(context.cell_calibration)
    for receipt in context.ordered_source_receipts:
        validate_unshocked_t3_session_calibration_receipt(receipt)
    indexes = tuple(
        receipt.source_seed_manifest.session_index
        for receipt in context.ordered_source_receipts
    )
    expected_digest = _context_manifest_digest(
        context.cell_calibration,
        context.ordered_source_receipts,
    )
    try:
        rebuilt_calibration = calibrate_experiment_cell_uniform_shock(
            context.ordered_source_receipts,
            expected_session_count=context.cell_calibration.sessions_expected,
            target_normalized_price_level_deviation=(
                context.cell_calibration.arithmetic.target_normalized_price_level_deviation
            ),
        )
    except ArithmeticError as error:
        raise ValueError("Context sources cannot rebuild Step 35E. / context 来源不能重建第35E步。") from error
    formal = context.cell_calibration.ready_for_formal_step35f
    if (
        context.ordered_session_indexes != indexes
        or indexes != tuple(range(context.cell_calibration.sessions_expected))
        or context.source_manifest_sha256 != expected_digest
        or rebuilt_calibration != context.cell_calibration
        or not context.exact_recalibration_match_verified
        or context.debug_context != (not formal)
        or context.formal_context != formal
    ):
        raise ValueError("Step-35F context is inconsistent. / 第 35F 步 context 不一致。")


def _source_for_session(
    context: VerifiedStep35FCellContext,
    session_index: int,
) -> UnshockedT3SessionCalibrationReceipt:
    """Return one canonical session source. / 返回一个标准 session 来源。"""

    checked = _positive_count(
        session_index + 1,
        "session_index + 1",
        context.cell_calibration.sessions_expected,
    ) - 1
    receipt = context.ordered_source_receipts[checked]
    if receipt.source_seed_manifest.session_index != checked:
        raise RuntimeError("Context session lookup failed. / context 的 session 查找失败。")
    return receipt


@dataclass(frozen=True)
class PairedT4PathResult:
    """One compact path result, discarded after online aggregation.

    一条紧凑的路径结果；在线汇总后立即丢弃。
    """

    protocol_version: str
    checkpoint_sha256: str
    path_index: int
    seed_manifest: PairedPathSeedManifest
    executed_local_periods: tuple[int, ...]
    t3_price_impact_lambda_hat: float
    control_t3_oriented_price: float
    treatment_t3_oriented_price: float
    paired_t3_oriented_price_increment: float
    expected_t3_oriented_price_increment: float
    control_t4_oriented_orders: tuple[float, float]
    treatment_t4_oriented_orders: tuple[float, float]
    path_fields_sha256: str
    replayed_step35d_control_path_fields_sha256: str
    pre_shock_exact_parity_verified: bool
    common_draws_verified: bool
    shock_applied_exactly_once: bool
    each_branch_t4_used_own_t3_history: bool
    control_rollback_periods_verified: int
    treatment_rollback_periods_verified: int


def _prior_ols_tuple(observation: object) -> tuple[float, float, float, float]:
    """Extract the OLS coefficients recorded on one observation. / 提取一期 OLS 系数。"""

    return (
        observation.xi_0_hat,
        observation.xi_1_hat,
        observation.gamma_0_hat,
        observation.gamma_1_hat,
    )


class ReusablePairedT4Workspace:
    """Restore two branches once and roll both back after every path.

    只恢复一对分支一次，并在每条路径后把两者都精确回滚。

    The workspace is deliberately single-threaded.  Step 36 should parallelize
    different sessions across processes. / 此工作区故意是单线程；第 36 步应在
    不同进程中并行不同 session。
    """

    def __init__(
        self,
        checkpoint: ConvergedMarketCheckpoint,
        *,
        baseline_scorer: OnlineIRFLongRunBaselineScorer,
        context: VerifiedStep35FCellContext,
    ) -> None:
        validate_verified_step35f_cell_context(context)
        verify_converged_market_checkpoint(checkpoint)
        if not isinstance(baseline_scorer, OnlineIRFLongRunBaselineScorer):
            raise TypeError("baseline_scorer has the wrong type. / baseline_scorer 类型错误。")
        session_index = checkpoint.payload.seed_manifest.session_index
        source = _source_for_session(context, session_index)
        live_baseline = baseline_scorer.verified_live_result_for_step35d(checkpoint)
        validate_irf_long_run_baseline_receipt(live_baseline)
        if (
            checkpoint.checkpoint_sha256 != source.checkpoint_sha256
            or checkpoint.payload.seed_manifest != source.source_seed_manifest
            or live_baseline != source.long_run_baseline_receipt
            or source.irf_experiment_seed != context.cell_calibration.irf_experiment_seed
            or source.path_seed_derivation_version != PAIRED_PATH_SEED_VERSION
        ):
            raise ValueError(
                "Checkpoint, live baseline, Step 35D, and Step 35E do not identify "
                "the same session. / checkpoint、实时 baseline、第35D步与第35E步不是同一 session。"
            )
        if (
            checkpoint.payload.parameters.num_speculators != PAPER_CLASSIFIER_AGENTS
            or len(checkpoint.payload.q_tables) != PAPER_CLASSIFIER_AGENTS
            or live_baseline.number_of_agents != PAPER_CLASSIFIER_AGENTS
            or context.cell_calibration.number_of_agents != PAPER_CLASSIFIER_AGENTS
        ):
            raise ValueError(
                "Appendix Section 4.5 classification requires exactly two informed agents. / "
                "附录第 4.5 节的分类要求恰好两个知情 agent。"
            )
        self.checkpoint = checkpoint
        self.context = context
        self.source_receipt = source
        self.baseline_receipt = live_baseline
        self.absolute_noise_shock = context.cell_calibration.selected_absolute_noise_shock
        self.schedule_context: VerifiedPairedPathScheduleContext = (
            prepare_verified_paired_path_schedule_context(
                checkpoint,
                irf_experiment_seed=source.irf_experiment_seed,
            )
        )
        self._control, self._treatment = restore_two_independent_branches(checkpoint)
        if self._control.market_maker is self._treatment.market_maker:
            raise RuntimeError("Paired branches share a market maker. / 配对分支共享做市商。")
        self._initial_rng = (
            self._control.all_random_states(),
            self._treatment.all_random_states(),
        )
        self._initial_q = (
            tuple(sha256(t.q_table.tobytes(order="C")).hexdigest() for t in self._control.traders),
            tuple(sha256(t.q_table.tobytes(order="C")).hexdigest() for t in self._treatment.traders),
        )
        self._initial_policy = (
            sha256(self._control.frozen_policy_action_indexes_snapshot().tobytes(order="C")).hexdigest(),
            sha256(self._treatment.frozen_policy_action_indexes_snapshot().tobytes(order="C")).hexdigest(),
        )
        self._initial_counts = (
            tuple(self._control.shared_value_visit_counts),
            tuple(self._treatment.shared_value_visit_counts),
        )
        self.paths_completed = 0
        self.rollbacks_completed = 0
        self._poisoned = False
        self._closed = False
        self.verify_exact_checkpoint_reset()

    @property
    def is_poisoned(self) -> bool:
        """Whether an interrupted path made reuse unsafe. / 是否因中断而不再安全。"""

        return self._poisoned

    def _branch_cheap_reset_check(self, branch: RandomizedMarketSession) -> None:
        """Check constant-time reset facts for one branch. / 检查一个分支的回滚事实。"""

        payload = self.checkpoint.payload
        if (
            branch.period_number != payload.origin_global_period
            or branch.previous_price != payload.previous_price
            or branch.previous_value != payload.previous_value
            or branch.current_value != payload.current_value
            or branch.frozen_draw_source_mode is not None
            or branch.market_maker.successful_append_count
            != payload.market_maker_state.successful_append_count
            or branch.market_maker.resynchronization_count
            != payload.market_maker_state.resynchronization_count
        ):
            raise RuntimeError("A paired branch did not return to t=0. / 某个配对分支没有回到 t=0。")

    def verify_exact_checkpoint_reset(self) -> None:
        """Perform the expensive exact audit at safe boundaries. / 在安全边界完整核对。"""

        for branch in (self._control, self._treatment):
            self._branch_cheap_reset_check(branch)
            if branch.market_maker.export_state() != self.checkpoint.payload.market_maker_state:
                raise RuntimeError("Market-maker rollback is not exact. / 做市商回滚不精确。")
        current_rng = (
            self._control.all_random_states(),
            self._treatment.all_random_states(),
        )
        current_q = (
            tuple(sha256(t.q_table.tobytes(order="C")).hexdigest() for t in self._control.traders),
            tuple(sha256(t.q_table.tobytes(order="C")).hexdigest() for t in self._treatment.traders),
        )
        current_policy = (
            sha256(self._control.frozen_policy_action_indexes_snapshot().tobytes(order="C")).hexdigest(),
            sha256(self._treatment.frozen_policy_action_indexes_snapshot().tobytes(order="C")).hexdigest(),
        )
        current_counts = (
            tuple(self._control.shared_value_visit_counts),
            tuple(self._treatment.shared_value_visit_counts),
        )
        if (
            current_rng != self._initial_rng
            or current_q != self._initial_q
            or current_policy != self._initial_policy
            or current_counts != self._initial_counts
            or any(t.q_table.flags.writeable for b in (self._control, self._treatment) for t in b.traders)
        ):
            raise RuntimeError("Frozen learning or RNG state changed. / 冻结学习或随机状态发生变化。")

    def run_path(self, path_index: int) -> PairedT4PathResult:
        """Run one paired t=1..4 path, then roll both branches back.

        运行一条配对的 t=1..4 路径，然后回滚两个分支。
        """

        if self._closed:
            raise RuntimeError("The workspace is closed. / 工作区已经关闭。")
        if self._poisoned:
            raise RuntimeError("The workspace is poisoned and must be rebuilt. / 工作区已失效，必须重建。")
        schedule = build_paired_path_draw_schedule_from_verified_context(
            self.schedule_context,
            path_index=path_index,
        )
        control_token = None
        treatment_token = None
        shock_count = 0
        pre_shock_parity = True
        t3_lambda: float | None = None
        control_t3_price: float | None = None
        treatment_t3_price: float | None = None
        control_t4_orders: tuple[float, float] | None = None
        treatment_t4_orders: tuple[float, float] | None = None
        t4_own_history = False
        control_t3_observation = None
        treatment_t3_observation = None
        compact_period_fields: list[tuple[object, ...]] = []
        # These first-three-period control observations reproduce the exact
        # Step-35D digest, not merely the same two means. / 这三个对照 observation
        # 会重建第 35D 步的精确摘要，而不只是比较两个均值。
        control_step35d_observations: list[object] = []
        try:
            control_token = self._control.begin_reversible_frozen_supplied_path(
                max_periods=PAIRED_RESPONSE_PERIOD_COUNT
            )
            treatment_token = self._treatment.begin_reversible_frozen_supplied_path(
                max_periods=PAIRED_RESPONSE_PERIOD_COUNT
            )
            for offset, local_period in enumerate(IRF_LOCAL_PERIODS):
                ordinary_noise = schedule.ordinary_noise_orders_u[offset]
                next_value_index = schedule.next_value_indexes[offset]
                if self._control.current_value != self._treatment.current_value:
                    raise RuntimeError("Paired current values diverged. / 配对分支的当前价值不同。")
                current_value = self._control.current_value
                if local_period == PAPER_RESPONSE_PERIOD:
                    if (
                        control_t3_observation is None
                        or treatment_t3_observation is None
                        or self._control.market_maker.snapshot()[-1]
                        != _completed_row(control_t3_observation)
                        or self._treatment.market_maker.snapshot()[-1]
                        != _completed_row(treatment_t3_observation)
                    ):
                        raise RuntimeError(
                            "A named branch does not carry its own completed t=3 row. / "
                            "某个命名分支没有携带自己的完整 t=3 记录。"
                        )
                    t4_own_history = True
                control_prior = self._control.market_maker.estimates()
                treatment_prior = self._treatment.market_maker.estimates()
                treatment_noise = ordinary_noise
                signed_shock = 0.0
                if local_period == PAPER_SHOCK_PERIOD:
                    applied = add_adverse_shock_to_noise(
                        ordinary_noise,
                        current_value,
                        self.checkpoint.payload.parameters.value_mean,
                        self.absolute_noise_shock,
                    )
                    treatment_noise = applied.noise_order_used_for_pricing
                    signed_shock = applied.signed_adverse_shock
                    shock_count += 1

                control_observation = self._control.run_next_frozen_policy_period_with_supplied_draws(
                    noise_order_u=ordinary_noise,
                    next_value_index=next_value_index,
                )
                treatment_observation = self._treatment.run_next_frozen_policy_period_with_supplied_draws(
                    noise_order_u=treatment_noise,
                    next_value_index=next_value_index,
                )
                expected_global_period = (
                    self.checkpoint.payload.origin_global_period + local_period - 1
                )
                if (
                    control_observation.period_number != expected_global_period
                    or treatment_observation.period_number != expected_global_period
                    or control_observation.noise_order_u != ordinary_noise
                    or treatment_observation.noise_order_u != treatment_noise
                    or control_observation.next_value_index != next_value_index
                    or treatment_observation.next_value_index != next_value_index
                ):
                    raise RuntimeError("A paired schedule was not used exactly. / 配对抽样计划没有被精确使用。")
                control_prior_tuple = (
                    control_prior.xi_0_hat,
                    control_prior.xi_1_hat,
                    control_prior.gamma_0_hat,
                    control_prior.gamma_1_hat,
                )
                treatment_prior_tuple = (
                    treatment_prior.xi_0_hat,
                    treatment_prior.xi_1_hat,
                    treatment_prior.gamma_0_hat,
                    treatment_prior.gamma_1_hat,
                )
                if (
                    _prior_ols_tuple(control_observation) != control_prior_tuple
                    or _prior_ols_tuple(treatment_observation) != treatment_prior_tuple
                ):
                    raise RuntimeError("A branch did not price from its own prior OLS. / 某分支没有使用自己的先前 OLS。")
                expected_control_lambda = calculate_adaptive_price_impact(
                    control_prior,
                    self.checkpoint.payload.parameters.pricing_error_weight,
                )
                expected_treatment_lambda = calculate_adaptive_price_impact(
                    treatment_prior,
                    self.checkpoint.payload.parameters.pricing_error_weight,
                )
                if (
                    control_observation.price_impact_lambda_hat != expected_control_lambda
                    or treatment_observation.price_impact_lambda_hat != expected_treatment_lambda
                ):
                    raise RuntimeError("A recorded lambda is inconsistent with prior OLS. / 记录的 lambda 与先前 OLS 不一致。")
                if local_period < PAPER_SHOCK_PERIOD and control_observation != treatment_observation:
                    pre_shock_parity = False
                    raise RuntimeError("Branches diverged before the shock. / 分支在冲击之前已分化。")

                control_oriented_price = orient_price(
                    control_observation.continuous_price_p,
                    current_value,
                    self.checkpoint.payload.parameters.value_mean,
                )
                treatment_oriented_price = orient_price(
                    treatment_observation.continuous_price_p,
                    current_value,
                    self.checkpoint.payload.parameters.value_mean,
                )
                control_oriented_orders = tuple(
                    orient_order(order, current_value, self.checkpoint.payload.parameters.value_mean)
                    for order in control_observation.raw_orders_x
                )
                treatment_oriented_orders = tuple(
                    orient_order(order, current_value, self.checkpoint.payload.parameters.value_mean)
                    for order in treatment_observation.raw_orders_x
                )
                compact_period_fields.append(
                    (
                        local_period,
                        ordinary_noise.hex(),
                        treatment_noise.hex(),
                        signed_shock.hex(),
                        next_value_index,
                        control_observation.current_state_indexes,
                        treatment_observation.current_state_indexes,
                        control_observation.action_indexes,
                        treatment_observation.action_indexes,
                        control_observation.price_impact_lambda_hat.hex(),
                        treatment_observation.price_impact_lambda_hat.hex(),
                        control_observation.continuous_price_p.hex(),
                        treatment_observation.continuous_price_p.hex(),
                        tuple(float(x).hex() for x in control_oriented_orders),
                        tuple(float(x).hex() for x in treatment_oriented_orders),
                    )
                )
                if local_period <= PAPER_SHOCK_PERIOD:
                    control_step35d_observations.append(control_observation)

                if local_period == PAPER_SHOCK_PERIOD:
                    pre_noise_control = (
                        control_observation.current_state_indexes,
                        control_observation.current_value_index,
                        control_observation.fundamental_value_v,
                        control_observation.action_indexes,
                        control_observation.raw_orders_x,
                        control_observation.xi_0_hat,
                        control_observation.xi_1_hat,
                        control_observation.gamma_0_hat,
                        control_observation.gamma_1_hat,
                        control_observation.price_impact_lambda_hat,
                    )
                    pre_noise_treatment = (
                        treatment_observation.current_state_indexes,
                        treatment_observation.current_value_index,
                        treatment_observation.fundamental_value_v,
                        treatment_observation.action_indexes,
                        treatment_observation.raw_orders_x,
                        treatment_observation.xi_0_hat,
                        treatment_observation.xi_1_hat,
                        treatment_observation.gamma_0_hat,
                        treatment_observation.gamma_1_hat,
                        treatment_observation.price_impact_lambda_hat,
                    )
                    if pre_noise_control != pre_noise_treatment:
                        raise RuntimeError("Shock-period branches differed before noise. / 冲击期分支在噪声之前不同。")
                    t3_lambda = control_observation.price_impact_lambda_hat
                    if t3_lambda <= 0.0:
                        raise RuntimeError("Shock-period lambda must be positive. / 冲击期 lambda 必须为正。")
                    control_t3_price = control_oriented_price
                    treatment_t3_price = treatment_oriented_price
                    control_t3_observation = control_observation
                    treatment_t3_observation = treatment_observation
                    actual_increment = treatment_t3_price - control_t3_price
                    expected_increment = t3_lambda * self.absolute_noise_shock
                    if not isclose(actual_increment, expected_increment, rel_tol=1e-12, abs_tol=1e-12):
                        raise RuntimeError("The t=3 price-shock identity failed. / t=3 价格冲击恒等式失败。")

                if local_period == PAPER_RESPONSE_PERIOD:
                    control_t4_orders = control_oriented_orders  # type: ignore[assignment]
                    treatment_t4_orders = treatment_oriented_orders  # type: ignore[assignment]

            if (
                shock_count != 1
                or t3_lambda is None
                or control_t3_price is None
                or treatment_t3_price is None
                or control_t4_orders is None
                or treatment_t4_orders is None
            ):
                raise RuntimeError("A required Step-35F period result is missing. / 第35F步缺少必要时期结果。")
            expected_increment = t3_lambda * self.absolute_noise_shock
            actual_increment = treatment_t3_price - control_t3_price
            path_payload = repr(
                (
                    self.checkpoint.checkpoint_sha256,
                    path_index,
                    schedule.seed_manifest,
                    tuple(compact_period_fields),
                    t3_lambda.hex(),
                    control_t3_price.hex(),
                    treatment_t3_price.hex(),
                    tuple(float(x).hex() for x in control_t4_orders),
                    tuple(float(x).hex() for x in treatment_t4_orders),
                )
            ).encode("ascii") + b"\0"
            control_step35d_payload = _completed_path_payload(
                checkpoint_sha256=self.checkpoint.checkpoint_sha256,
                path_index=path_index,
                seed_manifest=schedule.seed_manifest,
                ordinary_noise_orders=tuple(
                    schedule.ordinary_noise_orders_u[:PAPER_SHOCK_PERIOD]
                ),
                next_value_indexes=tuple(
                    schedule.next_value_indexes[:PAPER_SHOCK_PERIOD]
                ),
                observations=tuple(control_step35d_observations),  # type: ignore[arg-type]
            )
            result_without_rollback = PairedT4PathResult(
                protocol_version=STEP35F_PROTOCOL_VERSION,
                checkpoint_sha256=self.checkpoint.checkpoint_sha256,
                path_index=path_index,
                seed_manifest=schedule.seed_manifest,
                executed_local_periods=IRF_LOCAL_PERIODS,
                t3_price_impact_lambda_hat=t3_lambda,
                control_t3_oriented_price=control_t3_price,
                treatment_t3_oriented_price=treatment_t3_price,
                paired_t3_oriented_price_increment=actual_increment,
                expected_t3_oriented_price_increment=expected_increment,
                control_t4_oriented_orders=control_t4_orders,
                treatment_t4_oriented_orders=treatment_t4_orders,
                path_fields_sha256=sha256(
                    STEP35F_PATH_DIGEST_DOMAIN + path_payload
                ).hexdigest(),
                replayed_step35d_control_path_fields_sha256=sha256(
                    UNSHOCKED_T3_DIGEST_DOMAIN + control_step35d_payload
                ).hexdigest(),
                pre_shock_exact_parity_verified=pre_shock_parity,
                common_draws_verified=True,
                shock_applied_exactly_once=True,
                each_branch_t4_used_own_t3_history=t4_own_history,
                control_rollback_periods_verified=0,
                treatment_rollback_periods_verified=0,
            )
        except BaseException:
            self._poisoned = True
            for branch, token in (
                (self._treatment, treatment_token),
                (self._control, control_token),
            ):
                if token is None:
                    continue
                try:
                    branch.rollback_reversible_frozen_supplied_path(token)
                except BaseException:
                    pass
            raise

        try:
            if treatment_token is None or control_token is None:
                raise RuntimeError("A reversible transaction token is missing. / 可回滚事务 token 丢失。")
            treatment_rolled = self._treatment.rollback_reversible_frozen_supplied_path(
                treatment_token
            )
            control_rolled = self._control.rollback_reversible_frozen_supplied_path(
                control_token
            )
            if (
                treatment_rolled != PAIRED_RESPONSE_PERIOD_COUNT
                or control_rolled != PAIRED_RESPONSE_PERIOD_COUNT
            ):
                raise RuntimeError("Rollback period count is wrong. / 回滚时期数错误。")
            self._branch_cheap_reset_check(self._control)
            self._branch_cheap_reset_check(self._treatment)
        except BaseException:
            self._poisoned = True
            raise
        self.paths_completed += 1
        self.rollbacks_completed += 2
        return replace(
            result_without_rollback,
            control_rollback_periods_verified=control_rolled,
            treatment_rollback_periods_verified=treatment_rolled,
        )

    def close_and_verify(self) -> None:
        """Run one final exact reset audit and close. / 最后完整核对并关闭。"""

        if self._poisoned:
            raise RuntimeError("A poisoned workspace cannot finalize. / 已失效工作区不能完成。")
        if not self._closed:
            self.verify_exact_checkpoint_reset()
            self._closed = True


class OnlinePairedT4Moments:
    """Constant-memory, exact-order reducer for one session.

    一个 session 的固定内存、严格按路径顺序汇总器。
    """

    def __init__(
        self,
        schedule_context: VerifiedPairedPathScheduleContext,
        *,
        selected_absolute_noise_shock: float,
    ) -> None:
        if not isinstance(schedule_context, VerifiedPairedPathScheduleContext):
            raise TypeError("schedule_context has the wrong type. / schedule_context 类型错误。")
        self.schedule_context = schedule_context
        self.selected_absolute_noise_shock = _finite_real(
            selected_absolute_noise_shock,
            "selected_absolute_noise_shock",
        )
        if self.selected_absolute_noise_shock <= 0.0:
            raise ValueError("Shock magnitude must be positive. / 冲击幅度必须为正。")
        self.count = 0
        self._lambda_sum = (0.0, 0.0)
        self._control_t3_price_sum = (0.0, 0.0)
        self._treatment_t3_price_sum = (0.0, 0.0)
        self._control_t4_order_sums = [(0.0, 0.0) for _ in range(PAPER_CLASSIFIER_AGENTS)]
        self._treatment_t4_order_sums = [(0.0, 0.0) for _ in range(PAPER_CLASSIFIER_AGENTS)]
        self.maximum_absolute_t3_identity_error = 0.0
        self._digest = sha256(STEP35F_PATH_DIGEST_DOMAIN)
        self._step35d_control_digest = sha256(UNSHOCKED_T3_DIGEST_DOMAIN)

    def add(self, result: PairedT4PathResult) -> None:
        """Consume exactly the next canonical path. / 读取恰好下一条标准路径。"""

        expected_manifest = derive_paired_path_seed_manifest_from_verified_context(
            self.schedule_context,
            path_index=self.count,
        )
        if (
            not isinstance(result, PairedT4PathResult)
            or result.protocol_version != STEP35F_PROTOCOL_VERSION
            or result.path_index != self.count
            or result.checkpoint_sha256 != self.schedule_context.checkpoint_sha256
            or result.seed_manifest != expected_manifest
            or result.executed_local_periods != IRF_LOCAL_PERIODS
            or not result.pre_shock_exact_parity_verified
            or not result.common_draws_verified
            or not result.shock_applied_exactly_once
            or not result.each_branch_t4_used_own_t3_history
            or result.control_rollback_periods_verified != PAIRED_RESPONSE_PERIOD_COUNT
            or result.treatment_rollback_periods_verified != PAIRED_RESPONSE_PERIOD_COUNT
        ):
            raise ValueError("A paired path result is invalid or out of order. / 配对路径结果无效或顺序错误。")
        values = (
            result.t3_price_impact_lambda_hat,
            result.control_t3_oriented_price,
            result.treatment_t3_oriented_price,
            *result.control_t4_oriented_orders,
            *result.treatment_t4_oriented_orders,
        )
        if any(not isfinite(float(value)) for value in values):
            raise ValueError("A paired path contains a nonfinite value. / 配对路径含非有限数。")
        if (
            len(result.control_t4_oriented_orders) != PAPER_CLASSIFIER_AGENTS
            or len(result.treatment_t4_oriented_orders) != PAPER_CLASSIFIER_AGENTS
            or not _is_sha256_text(result.path_fields_sha256)
            or not _is_sha256_text(
                result.replayed_step35d_control_path_fields_sha256
            )
        ):
            raise ValueError("A paired path vector or digest is invalid. / 配对路径向量或摘要无效。")
        actual_increment_from_prices = (
            result.treatment_t3_oriented_price
            - result.control_t3_oriented_price
        )
        expected_increment_from_inputs = (
            result.t3_price_impact_lambda_hat
            * self.selected_absolute_noise_shock
        )
        if (
            result.t3_price_impact_lambda_hat <= 0.0
            or result.paired_t3_oriented_price_increment
            != actual_increment_from_prices
            or result.expected_t3_oriented_price_increment
            != expected_increment_from_inputs
            or not isclose(
                actual_increment_from_prices,
                expected_increment_from_inputs,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("A paired path's t=3 shock arithmetic is inconsistent. / 配对路径的 t=3 冲击计算不一致。")
        identity_error = (
            result.paired_t3_oriented_price_increment
            - result.expected_t3_oriented_price_increment
        )
        self._lambda_sum = _next_neumaier_sum(*self._lambda_sum, result.t3_price_impact_lambda_hat)
        self._control_t3_price_sum = _next_neumaier_sum(*self._control_t3_price_sum, result.control_t3_oriented_price)
        self._treatment_t3_price_sum = _next_neumaier_sum(*self._treatment_t3_price_sum, result.treatment_t3_oriented_price)
        for agent in range(PAPER_CLASSIFIER_AGENTS):
            self._control_t4_order_sums[agent] = _next_neumaier_sum(
                *self._control_t4_order_sums[agent],
                result.control_t4_oriented_orders[agent],
            )
            self._treatment_t4_order_sums[agent] = _next_neumaier_sum(
                *self._treatment_t4_order_sums[agent],
                result.treatment_t4_oriented_orders[agent],
            )
        self.maximum_absolute_t3_identity_error = max(
            self.maximum_absolute_t3_identity_error,
            abs(identity_error),
        )
        self._digest.update(
            repr((result.path_index, result.path_fields_sha256)).encode("ascii") + b"\0"
        )
        # Match OnlineUnshockedT3Moments.add byte-for-byte. / 与第 35D 步在线
        # 汇总器逐字节使用相同的摘要输入。
        self._step35d_control_digest.update(
            repr(
                (
                    result.path_index,
                    result.replayed_step35d_control_path_fields_sha256,
                    result.t3_price_impact_lambda_hat.hex(),
                    result.control_t3_oriented_price.hex(),
                )
            ).encode("ascii")
            + b"\0"
        )
        self.count += 1

    def summarize(
        self,
    ) -> tuple[
        float,
        float,
        float,
        tuple[float, float],
        tuple[float, float],
        float,
        str,
        str,
    ]:
        """Return session means and the executed-path digest. / 返回 session 均值与摘要。"""

        return (
            _mean(self._lambda_sum, self.count),
            _mean(self._control_t3_price_sum, self.count),
            _mean(self._treatment_t3_price_sum, self.count),
            tuple(_mean(item, self.count) for item in self._control_t4_order_sums),  # type: ignore[return-value]
            tuple(_mean(item, self.count) for item in self._treatment_t4_order_sums),  # type: ignore[return-value]
            self.maximum_absolute_t3_identity_error,
            self._digest.hexdigest(),
            self._step35d_control_digest.hexdigest(),
        )


@dataclass(frozen=True)
class Step35FSessionReceipt:
    """One session's averaged paths and mechanism classification.

    一个 session 的路径均值与机制分类凭证。
    """

    protocol_version: str
    checkpoint_sha256: str
    source_session_index: int
    source_step35d_receipt_sha256: str
    source_step35d_executed_path_fields_sha256: str
    source_baseline_receipt_sha256: str
    cell_calibration_receipt_sha256: str
    cell_context_manifest_sha256: str
    irf_experiment_seed: int
    path_seed_derivation_version: str
    selected_absolute_noise_shock: float
    paths_requested: int
    paths_executed: int
    first_path_index: int
    last_path_index: int
    mean_t3_price_impact_lambda: float
    mean_control_t3_oriented_price: float
    mean_treatment_t3_oriented_price: float
    session_long_run_mean_oriented_price: float
    session_achieved_normalized_price_level_deviation: float
    mean_control_t4_oriented_order_by_agent: tuple[float, float]
    mean_treatment_t4_oriented_order_by_agent: tuple[float, float]
    session_long_run_mean_oriented_order_by_agent: tuple[float, float]
    normalized_order_response_by_agent: tuple[float, float]
    paired_control_sensitivity_response_by_agent: tuple[float, float]
    normalized_response_details: tuple[NormalizedOrderResponse, NormalizedOrderResponse]
    mechanism_classification: MechanismClassification
    mechanism_label: CollusionMechanism
    maximum_absolute_t3_identity_error: float
    executed_path_fields_sha256: str
    replayed_step35d_executed_path_fields_sha256: str
    source_control_t3_moments_exactly_reproduced: bool
    same_path_schedules_as_step35d_verified: bool
    pre_shock_parity_all_paths_verified: bool
    shock_once_all_paths_verified: bool
    independent_adaptive_ols_histories_verified: bool
    frozen_learning_state_verified: bool
    exact_checkpoint_reset_after_batch_verified: bool
    constant_memory_online_aggregation_verified: bool
    raw_path_results_retained: int
    source_session_full_restores: int
    successful_branch_transaction_rollbacks: int
    paper_10000_paths_verified: bool
    paper_scale_long_run_source_verified: bool
    formal_cell_calibration_context_verified: bool
    session_classification_computed: bool
    formal_session_classification_ready: bool
    paper_primary_long_run_denominator_used: bool
    treatment_control_response_is_sensitivity_only: bool
    paper_classifies_session_not_paths: bool
    paper_specifies_reset_and_common_random_number_protocol: bool
    checksum_detects_stale_replacement_not_authentication: bool
    receipt_payload_sha256: str


def _session_receipt_digest(receipt: Step35FSessionReceipt) -> str:
    """Checksum all session fields except the checksum. / 校验 session 的全部字段。"""

    unsigned = replace(receipt, receipt_payload_sha256="")
    return sha256(
        STEP35F_SESSION_RECEIPT_DOMAIN + pickle.dumps(unsigned, protocol=5)
    ).hexdigest()


def validate_step35f_session_receipt(receipt: Step35FSessionReceipt) -> None:
    """Recompute formulas and reject exaggerated claims. / 重算公式并拒绝夸大声明。"""

    if not isinstance(receipt, Step35FSessionReceipt):
        raise TypeError("receipt has the wrong type. / receipt 类型错误。")
    if receipt.protocol_version != STEP35F_PROTOCOL_VERSION:
        raise ValueError("Session receipt version is unsupported. / session receipt 版本不支持。")
    if not _is_sha256_text(receipt.receipt_payload_sha256) or _session_receipt_digest(receipt) != receipt.receipt_payload_sha256:
        raise ValueError("Session receipt checksum failed. / session receipt 校验失败。")
    for digest in (
        receipt.checkpoint_sha256,
        receipt.source_step35d_receipt_sha256,
        receipt.source_step35d_executed_path_fields_sha256,
        receipt.source_baseline_receipt_sha256,
        receipt.cell_calibration_receipt_sha256,
        receipt.cell_context_manifest_sha256,
        receipt.executed_path_fields_sha256,
        receipt.replayed_step35d_executed_path_fields_sha256,
    ):
        if not _is_sha256_text(digest):
            raise ValueError("Session receipt contains an invalid digest. / session receipt 含无效摘要。")
    count = _positive_count(receipt.paths_executed, "paths_executed", PAPER_PATHS_PER_SESSION)
    if (
        isinstance(receipt.irf_experiment_seed, bool)
        or not isinstance(receipt.irf_experiment_seed, int)
        or not 0 <= receipt.irf_experiment_seed < 2**64
    ):
        raise ValueError("IRF experiment seed is outside uint64. / IRF 实验 seed 超出 uint64。")
    canonical = (
        receipt.paths_requested == count
        and receipt.first_path_index == 0
        and receipt.last_path_index == count - 1
    )
    if not canonical:
        raise ValueError("Session paths are not canonical 0..N-1. / session 路径不是标准的 0..N-1。")
    if len(receipt.session_long_run_mean_oriented_order_by_agent) != PAPER_CLASSIFIER_AGENTS:
        raise ValueError("Exactly two agent denominators are required. / 必须有两个 agent 分母。")
    expected_details = tuple(
        calculate_normalized_order_response(
            agent + 1,
            receipt.session_long_run_mean_oriented_order_by_agent[agent],
            receipt.mean_treatment_t4_oriented_order_by_agent[agent],
        )
        for agent in range(PAPER_CLASSIFIER_AGENTS)
    )
    expected_responses = tuple(item.normalized_response for item in expected_details)
    expected_sensitivity = tuple(
        (
            receipt.mean_treatment_t4_oriented_order_by_agent[agent]
            - receipt.mean_control_t4_oriented_order_by_agent[agent]
        )
        / receipt.session_long_run_mean_oriented_order_by_agent[agent]
        for agent in range(PAPER_CLASSIFIER_AGENTS)
    )
    expected_classification = classify_normalized_order_responses(expected_responses)
    finite_values = (
        receipt.selected_absolute_noise_shock,
        receipt.mean_t3_price_impact_lambda,
        receipt.mean_control_t3_oriented_price,
        receipt.mean_treatment_t3_oriented_price,
        receipt.session_long_run_mean_oriented_price,
        receipt.session_achieved_normalized_price_level_deviation,
        *receipt.mean_control_t4_oriented_order_by_agent,
        *receipt.mean_treatment_t4_oriented_order_by_agent,
        *receipt.session_long_run_mean_oriented_order_by_agent,
        *receipt.normalized_order_response_by_agent,
        *receipt.paired_control_sensitivity_response_by_agent,
        receipt.maximum_absolute_t3_identity_error,
    )
    if any(not isfinite(float(value)) for value in finite_values):
        raise ValueError("Session receipt contains a nonfinite value. / session receipt 含非有限数。")
    if receipt.selected_absolute_noise_shock <= 0.0 or receipt.mean_t3_price_impact_lambda <= 0.0:
        raise ValueError("Shock magnitude and mean lambda must be positive. / 冲击幅度与平均 lambda 必须为正。")
    actual_mean_increment = (
        receipt.mean_treatment_t3_oriented_price
        - receipt.mean_control_t3_oriented_price
    )
    expected_mean_increment = (
        receipt.mean_t3_price_impact_lambda
        * receipt.selected_absolute_noise_shock
    )
    if receipt.session_long_run_mean_oriented_price <= 0.0:
        raise ValueError("Session long-run oriented price must be positive. / session 长期方向价格必须为正。")
    expected_level = (
        receipt.mean_treatment_t3_oriented_price
        - receipt.session_long_run_mean_oriented_price
    ) / receipt.session_long_run_mean_oriented_price
    paper_paths = count == PAPER_PATHS_PER_SESSION
    formal = (
        paper_paths
        and receipt.paper_scale_long_run_source_verified
        and receipt.formal_cell_calibration_context_verified
    )
    if (
        receipt.path_seed_derivation_version != PAIRED_PATH_SEED_VERSION
        or receipt.normalized_response_details != expected_details
        or receipt.normalized_order_response_by_agent != expected_responses
        or receipt.paired_control_sensitivity_response_by_agent != expected_sensitivity
        or receipt.mechanism_classification != expected_classification
        or receipt.mechanism_label != expected_classification.mechanism
        or receipt.session_achieved_normalized_price_level_deviation
        != expected_level
        or not isclose(
            actual_mean_increment,
            expected_mean_increment,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or receipt.replayed_step35d_executed_path_fields_sha256
        != receipt.source_step35d_executed_path_fields_sha256
        or receipt.maximum_absolute_t3_identity_error > 1e-12
    ):
        raise ValueError("Session response arithmetic is inconsistent. / session 反应计算不一致。")
    logical_claims = (
        receipt.source_control_t3_moments_exactly_reproduced,
        receipt.same_path_schedules_as_step35d_verified,
        receipt.pre_shock_parity_all_paths_verified,
        receipt.shock_once_all_paths_verified,
        receipt.independent_adaptive_ols_histories_verified,
        receipt.frozen_learning_state_verified,
        receipt.exact_checkpoint_reset_after_batch_verified,
        receipt.constant_memory_online_aggregation_verified,
        receipt.raw_path_results_retained == 0,
        receipt.source_session_full_restores == 2,
        receipt.successful_branch_transaction_rollbacks == 2 * count,
        receipt.paper_10000_paths_verified == paper_paths,
        receipt.session_classification_computed,
        receipt.formal_session_classification_ready == formal,
        receipt.paper_primary_long_run_denominator_used,
        receipt.treatment_control_response_is_sensitivity_only,
        receipt.paper_classifies_session_not_paths,
        not receipt.paper_specifies_reset_and_common_random_number_protocol,
        receipt.checksum_detects_stale_replacement_not_authentication,
    )
    if not all(logical_claims):
        raise ValueError("Session receipt claims are inconsistent. / session receipt 声明不一致。")


def run_step35f_session_response_paths(
    checkpoint: ConvergedMarketCheckpoint,
    *,
    baseline_scorer: OnlineIRFLongRunBaselineScorer,
    context: VerifiedStep35FCellContext,
    path_count: int | None = None,
) -> Step35FSessionReceipt:
    """Run and classify one session after averaging its paired paths.

    平均一个 session 的配对路径后，再对该 session 分类。
    """

    validate_verified_step35f_cell_context(context)
    session_index = checkpoint.payload.seed_manifest.session_index
    source = _source_for_session(context, session_index)
    requested = source.paths_executed if path_count is None else _positive_count(
        path_count, "path_count", PAPER_PATHS_PER_SESSION
    )
    if requested != source.paths_executed:
        raise ValueError(
            "Step 35F must rerun exactly the same canonical path count as Step 35D. / "
            "第 35F 步必须重跑与第 35D 步完全相同的标准路径数。"
        )
    workspace = ReusablePairedT4Workspace(
        checkpoint,
        baseline_scorer=baseline_scorer,
        context=context,
    )
    moments = OnlinePairedT4Moments(
        workspace.schedule_context,
        selected_absolute_noise_shock=workspace.absolute_noise_shock,
    )
    for path_index in range(requested):
        moments.add(workspace.run_path(path_index))
    workspace.close_and_verify()
    (
        mean_lambda,
        mean_control_p3,
        mean_treatment_p3,
        mean_control_x4,
        mean_treatment_x4,
        maximum_identity_error,
        executed_digest,
        replayed_step35d_digest,
    ) = moments.summarize()
    if (
        mean_lambda != source.mean_t3_price_impact_lambda
        or mean_control_p3 != source.mean_unshocked_t3_oriented_price
        or replayed_step35d_digest != source.executed_path_fields_sha256
    ):
        raise RuntimeError(
            "Step 35F did not exactly reproduce Step-35D control moments. / "
            "第 35F 步没有精确重现第 35D 步的对照统计量。"
        )
    baseline_orders = tuple(workspace.baseline_receipt.mean_oriented_order_by_agent)
    if len(baseline_orders) != PAPER_CLASSIFIER_AGENTS:
        raise ValueError("Appendix 4.5 requires exactly two agents. / 附录4.5要求恰好两个 agent。")
    details = tuple(
        calculate_normalized_order_response(
            agent + 1,
            baseline_orders[agent],
            mean_treatment_x4[agent],
        )
        for agent in range(PAPER_CLASSIFIER_AGENTS)
    )
    responses = tuple(item.normalized_response for item in details)
    sensitivity = tuple(
        (mean_treatment_x4[agent] - mean_control_x4[agent]) / baseline_orders[agent]
        for agent in range(PAPER_CLASSIFIER_AGENTS)
    )
    classification = classify_normalized_order_responses(responses)
    long_run_price = workspace.baseline_receipt.mean_oriented_price
    session_level = (mean_treatment_p3 - long_run_price) / long_run_price
    paper_paths = requested == PAPER_PATHS_PER_SESSION
    paper_baseline = workspace.baseline_receipt.paper_scale_thresholds_and_provenance_verified
    receipt = Step35FSessionReceipt(
        protocol_version=STEP35F_PROTOCOL_VERSION,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        source_session_index=session_index,
        source_step35d_receipt_sha256=source.receipt_payload_sha256,
        source_step35d_executed_path_fields_sha256=(
            source.executed_path_fields_sha256
        ),
        source_baseline_receipt_sha256=workspace.baseline_receipt.receipt_payload_sha256,
        cell_calibration_receipt_sha256=context.cell_calibration.receipt_payload_sha256,
        cell_context_manifest_sha256=context.source_manifest_sha256,
        irf_experiment_seed=source.irf_experiment_seed,
        path_seed_derivation_version=source.path_seed_derivation_version,
        selected_absolute_noise_shock=context.cell_calibration.selected_absolute_noise_shock,
        paths_requested=requested,
        paths_executed=moments.count,
        first_path_index=0,
        last_path_index=moments.count - 1,
        mean_t3_price_impact_lambda=mean_lambda,
        mean_control_t3_oriented_price=mean_control_p3,
        mean_treatment_t3_oriented_price=mean_treatment_p3,
        session_long_run_mean_oriented_price=long_run_price,
        session_achieved_normalized_price_level_deviation=session_level,
        mean_control_t4_oriented_order_by_agent=mean_control_x4,
        mean_treatment_t4_oriented_order_by_agent=mean_treatment_x4,
        session_long_run_mean_oriented_order_by_agent=baseline_orders,  # type: ignore[arg-type]
        normalized_order_response_by_agent=responses,  # type: ignore[arg-type]
        paired_control_sensitivity_response_by_agent=sensitivity,  # type: ignore[arg-type]
        normalized_response_details=details,  # type: ignore[arg-type]
        mechanism_classification=classification,
        mechanism_label=classification.mechanism,
        maximum_absolute_t3_identity_error=maximum_identity_error,
        executed_path_fields_sha256=executed_digest,
        replayed_step35d_executed_path_fields_sha256=replayed_step35d_digest,
        source_control_t3_moments_exactly_reproduced=True,
        same_path_schedules_as_step35d_verified=True,
        pre_shock_parity_all_paths_verified=True,
        shock_once_all_paths_verified=True,
        independent_adaptive_ols_histories_verified=True,
        frozen_learning_state_verified=True,
        exact_checkpoint_reset_after_batch_verified=True,
        constant_memory_online_aggregation_verified=True,
        raw_path_results_retained=0,
        source_session_full_restores=2,
        successful_branch_transaction_rollbacks=workspace.rollbacks_completed,
        paper_10000_paths_verified=paper_paths,
        paper_scale_long_run_source_verified=paper_baseline,
        formal_cell_calibration_context_verified=context.formal_context,
        session_classification_computed=True,
        formal_session_classification_ready=(
            paper_paths and paper_baseline and context.formal_context
        ),
        paper_primary_long_run_denominator_used=True,
        treatment_control_response_is_sensitivity_only=True,
        paper_classifies_session_not_paths=True,
        paper_specifies_reset_and_common_random_number_protocol=False,
        checksum_detects_stale_replacement_not_authentication=True,
        receipt_payload_sha256="",
    )
    receipt = replace(receipt, receipt_payload_sha256=_session_receipt_digest(receipt))
    validate_step35f_session_receipt(receipt)
    return receipt


@dataclass(frozen=True)
class Step35FCellReceipt:
    """Cell-level price audit and shares of session-level labels.

    实验单元层面的价格核对，以及 session 分类占比。
    """

    protocol_version: str
    cell_calibration_receipt_sha256: str
    cell_context_manifest_sha256: str
    ordered_session_response_receipts_sha256: str
    sessions_expected: int
    sessions_received: int
    ordered_session_indexes: tuple[int, ...]
    total_control_paths_executed: int
    total_treatment_paths_executed: int
    pooled_mean_control_t3_oriented_price: float
    pooled_mean_treatment_t3_oriented_price: float
    pooled_mean_actual_t3_price_impact_lambda: float
    pooled_long_run_mean_oriented_price: float
    achieved_normalized_treatment_price_level_deviation: float
    target_normalized_treatment_price_level_deviation: float
    price_target_absolute_error: float
    price_trigger_session_count: int
    over_pruning_session_count: int
    unclassified_session_count: int
    price_trigger_session_share: float
    over_pruning_session_share: float
    unclassified_session_share: float
    session_normalized_order_responses: tuple[tuple[float, float], ...]
    session_mechanism_labels: tuple[CollusionMechanism, ...]
    exact_paper_target_achieved_on_executed_paths: bool
    every_session_classified_after_path_aggregation: bool
    labels_aggregated_as_shares_not_reclassified_from_pooled_response: bool
    paper_1000_sessions_verified: bool
    paper_10000_paths_per_session_verified: bool
    formal_cell_calibration_context_verified: bool
    all_session_receipts_formal_ready: bool
    formal_paper_mechanism_result_ready: bool
    paper_specifies_session_pooling_and_finite_sample_calibration: bool
    checksum_detects_stale_replacement_not_authentication: bool
    receipt_payload_sha256: str


def _ordered_session_response_digest(
    receipts: Sequence[Step35FSessionReceipt],
) -> str:
    """Bind ordered session response receipts. / 绑定有序的 session 反应凭证。"""

    digest = sha256(STEP35F_CELL_SESSION_DIGEST_DOMAIN)
    for receipt in receipts:
        digest.update(receipt.source_session_index.to_bytes(8, "big"))
        digest.update(bytes.fromhex(receipt.receipt_payload_sha256))
    return digest.hexdigest()


def _cell_receipt_digest(receipt: Step35FCellReceipt) -> str:
    """Checksum all cell fields except the checksum. / 校验实验单元的全部字段。"""

    unsigned = replace(receipt, receipt_payload_sha256="")
    return sha256(
        STEP35F_CELL_RECEIPT_DOMAIN + pickle.dumps(unsigned, protocol=5)
    ).hexdigest()


def validate_step35f_cell_receipt(receipt: Step35FCellReceipt) -> None:
    """Reject changed shares, levels, or formal claims. / 拒绝被修改的占比、水平或声明。"""

    if not isinstance(receipt, Step35FCellReceipt):
        raise TypeError("receipt has the wrong type. / receipt 类型错误。")
    if receipt.protocol_version != STEP35F_PROTOCOL_VERSION:
        raise ValueError("Cell receipt version is unsupported. / cell receipt 版本不支持。")
    if not _is_sha256_text(receipt.receipt_payload_sha256) or _cell_receipt_digest(receipt) != receipt.receipt_payload_sha256:
        raise ValueError("Cell receipt checksum failed. / cell receipt 校验失败。")
    for digest in (
        receipt.cell_calibration_receipt_sha256,
        receipt.cell_context_manifest_sha256,
        receipt.ordered_session_response_receipts_sha256,
    ):
        if not _is_sha256_text(digest):
            raise ValueError("Cell receipt contains an invalid digest. / cell receipt 含无效摘要。")
    count = _positive_count(receipt.sessions_received, "sessions_received", 1_000)
    if (
        receipt.sessions_expected != count
        or receipt.ordered_session_indexes != tuple(range(count))
        or len(receipt.session_normalized_order_responses) != count
        or len(receipt.session_mechanism_labels) != count
    ):
        raise ValueError("Cell session coverage is inconsistent. / 实验单元的 session 覆盖不一致。")
    numeric_fields = (
        receipt.pooled_mean_control_t3_oriented_price,
        receipt.pooled_mean_treatment_t3_oriented_price,
        receipt.pooled_mean_actual_t3_price_impact_lambda,
        receipt.pooled_long_run_mean_oriented_price,
        receipt.achieved_normalized_treatment_price_level_deviation,
        receipt.target_normalized_treatment_price_level_deviation,
        receipt.price_target_absolute_error,
        receipt.price_trigger_session_share,
        receipt.over_pruning_session_share,
        receipt.unclassified_session_share,
        *(value for pair in receipt.session_normalized_order_responses for value in pair),
    )
    if any(not isfinite(float(value)) for value in numeric_fields):
        raise ValueError("Cell receipt contains a nonfinite number. / cell receipt 含非有限数。")
    if (
        receipt.pooled_mean_actual_t3_price_impact_lambda <= 0.0
        or receipt.pooled_long_run_mean_oriented_price <= 0.0
        or receipt.price_target_absolute_error < 0.0
        or isinstance(receipt.total_control_paths_executed, bool)
        or not isinstance(receipt.total_control_paths_executed, int)
        or isinstance(receipt.total_treatment_paths_executed, bool)
        or not isinstance(receipt.total_treatment_paths_executed, int)
        or receipt.total_control_paths_executed < 1
        or receipt.total_treatment_paths_executed < 1
    ):
        raise ValueError("Cell numeric domains are invalid. / cell 数值定义域无效。")
    label_counts = (
        receipt.session_mechanism_labels.count(CollusionMechanism.PRICE_TRIGGER),
        receipt.session_mechanism_labels.count(CollusionMechanism.OVER_PRUNING),
        receipt.session_mechanism_labels.count(CollusionMechanism.UNCLASSIFIED),
    )
    expected_labels = tuple(
        classify_normalized_order_responses(response).mechanism
        for response in receipt.session_normalized_order_responses
    )
    if expected_labels != receipt.session_mechanism_labels:
        raise ValueError("A stored label differs from its two session responses. / 某标签与该 session 的两个反应不一致。")
    achieved = (
        receipt.pooled_mean_treatment_t3_oriented_price
        - receipt.pooled_long_run_mean_oriented_price
    ) / receipt.pooled_long_run_mean_oriented_price
    error = achieved - receipt.target_normalized_treatment_price_level_deviation
    target_hit = isclose(
        achieved,
        receipt.target_normalized_treatment_price_level_deviation,
        rel_tol=0.0,
        abs_tol=2e-12,
    )
    paper_sessions = count == 1_000
    paper_paths = (
        receipt.total_control_paths_executed
        == receipt.total_treatment_paths_executed
        == count * PAPER_PATHS_PER_SESSION
    )
    if (
        label_counts
        != (
            receipt.price_trigger_session_count,
            receipt.over_pruning_session_count,
            receipt.unclassified_session_count,
        )
        or receipt.price_trigger_session_share != label_counts[0] / count
        or receipt.over_pruning_session_share != label_counts[1] / count
        or receipt.unclassified_session_share != label_counts[2] / count
        or receipt.achieved_normalized_treatment_price_level_deviation != achieved
        or receipt.target_normalized_treatment_price_level_deviation
        != PAPER_TARGET_PRICE_DEVIATION
        or receipt.price_target_absolute_error != abs(error)
        or receipt.exact_paper_target_achieved_on_executed_paths != target_hit
        or receipt.paper_1000_sessions_verified != paper_sessions
        or receipt.paper_10000_paths_per_session_verified != paper_paths
        or receipt.formal_paper_mechanism_result_ready
        != (
            target_hit
            and paper_sessions
            and paper_paths
            and receipt.formal_cell_calibration_context_verified
            and receipt.all_session_receipts_formal_ready
        )
    ):
        raise ValueError("Cell arithmetic or formal claims are inconsistent. / cell 计算或正式声明不一致。")
    if not (
        receipt.every_session_classified_after_path_aggregation
        and receipt.labels_aggregated_as_shares_not_reclassified_from_pooled_response
        and not receipt.paper_specifies_session_pooling_and_finite_sample_calibration
        and receipt.checksum_detects_stale_replacement_not_authentication
    ):
        raise ValueError("Cell methodology claims are inconsistent. / cell 方法声明不一致。")


def aggregate_step35f_experiment_cell(
    context: VerifiedStep35FCellContext,
    session_receipts: Sequence[Step35FSessionReceipt],
) -> Step35FCellReceipt:
    """Audit the cell price target and count session-level labels.

    核对实验单元价格目标，并统计 session 层面的标签。
    """

    validate_verified_step35f_cell_context(context)
    if isinstance(session_receipts, (str, bytes)) or not isinstance(session_receipts, Sequence):
        raise TypeError("session_receipts must be a sequence. / session receipts 必须是序列。")
    supplied = tuple(session_receipts)
    if len(supplied) != context.cell_calibration.sessions_expected:
        raise ValueError("The response session count differs from Step 35E. / 反应 session 数与第35E步不同。")
    for receipt in supplied:
        validate_step35f_session_receipt(receipt)
    ordered = tuple(sorted(supplied, key=lambda item: item.source_session_index))
    indexes = tuple(receipt.source_session_index for receipt in ordered)
    if indexes != tuple(range(context.cell_calibration.sessions_expected)):
        raise ValueError("Response sessions must cover canonical indexes. / 反应 sessions 必须覆盖标准编号。")
    for receipt, source in zip(ordered, context.ordered_source_receipts, strict=True):
        if (
            receipt.checkpoint_sha256 != source.checkpoint_sha256
            or receipt.source_step35d_receipt_sha256 != source.receipt_payload_sha256
            or receipt.source_step35d_executed_path_fields_sha256
            != source.executed_path_fields_sha256
            or receipt.source_baseline_receipt_sha256 != source.baseline_receipt_payload_sha256
            or receipt.irf_experiment_seed != source.irf_experiment_seed
            or receipt.cell_calibration_receipt_sha256
            != context.cell_calibration.receipt_payload_sha256
            or receipt.cell_context_manifest_sha256 != context.source_manifest_sha256
            or receipt.paths_executed != source.paths_executed
            or receipt.selected_absolute_noise_shock
            != context.cell_calibration.selected_absolute_noise_shock
        ):
            raise ValueError("A response receipt does not match its Step-35D source. / 某反应凭证与其第35D来源不匹配。")

    total_paths = sum(receipt.paths_executed for receipt in ordered)
    pooled_control_p3 = fsum(
        receipt.mean_control_t3_oriented_price * receipt.paths_executed
        for receipt in ordered
    ) / total_paths
    pooled_treatment_p3 = fsum(
        receipt.mean_treatment_t3_oriented_price * receipt.paths_executed
        for receipt in ordered
    ) / total_paths
    pooled_lambda = fsum(
        receipt.mean_t3_price_impact_lambda * receipt.paths_executed
        for receipt in ordered
    ) / total_paths
    calibration = context.cell_calibration
    if (
        not isclose(
            pooled_control_p3,
            calibration.pooled_unshocked_t3_mean_oriented_price,
            rel_tol=0.0,
            abs_tol=2e-12,
        )
        or not isclose(
            pooled_lambda,
            calibration.pooled_actual_t3_mean_price_impact_lambda,
            rel_tol=0.0,
            abs_tol=2e-12,
        )
    ):
        raise RuntimeError("Executed control paths do not reproduce Step 35E. / 实际对照路径没有重现第35E步。")
    achieved = (
        pooled_treatment_p3 - calibration.pooled_long_run_mean_oriented_price
    ) / calibration.pooled_long_run_mean_oriented_price
    error = achieved - PAPER_TARGET_PRICE_DEVIATION
    target_hit = isclose(achieved, PAPER_TARGET_PRICE_DEVIATION, rel_tol=0.0, abs_tol=2e-12)
    labels = tuple(receipt.mechanism_label for receipt in ordered)
    responses = tuple(receipt.normalized_order_response_by_agent for receipt in ordered)
    trigger_count = labels.count(CollusionMechanism.PRICE_TRIGGER)
    pruning_count = labels.count(CollusionMechanism.OVER_PRUNING)
    unclassified_count = labels.count(CollusionMechanism.UNCLASSIFIED)
    count = len(ordered)
    paper_sessions = count == 1_000
    paper_paths = all(receipt.paper_10000_paths_verified for receipt in ordered)
    formal = (
        target_hit
        and paper_sessions
        and paper_paths
        and context.formal_context
        and all(receipt.formal_session_classification_ready for receipt in ordered)
    )
    receipt = Step35FCellReceipt(
        protocol_version=STEP35F_PROTOCOL_VERSION,
        cell_calibration_receipt_sha256=calibration.receipt_payload_sha256,
        cell_context_manifest_sha256=context.source_manifest_sha256,
        ordered_session_response_receipts_sha256=_ordered_session_response_digest(ordered),
        sessions_expected=context.cell_calibration.sessions_expected,
        sessions_received=count,
        ordered_session_indexes=indexes,
        total_control_paths_executed=total_paths,
        total_treatment_paths_executed=total_paths,
        pooled_mean_control_t3_oriented_price=pooled_control_p3,
        pooled_mean_treatment_t3_oriented_price=pooled_treatment_p3,
        pooled_mean_actual_t3_price_impact_lambda=pooled_lambda,
        pooled_long_run_mean_oriented_price=calibration.pooled_long_run_mean_oriented_price,
        achieved_normalized_treatment_price_level_deviation=achieved,
        target_normalized_treatment_price_level_deviation=PAPER_TARGET_PRICE_DEVIATION,
        price_target_absolute_error=abs(error),
        price_trigger_session_count=trigger_count,
        over_pruning_session_count=pruning_count,
        unclassified_session_count=unclassified_count,
        price_trigger_session_share=trigger_count / count,
        over_pruning_session_share=pruning_count / count,
        unclassified_session_share=unclassified_count / count,
        session_normalized_order_responses=responses,
        session_mechanism_labels=labels,
        exact_paper_target_achieved_on_executed_paths=target_hit,
        every_session_classified_after_path_aggregation=True,
        labels_aggregated_as_shares_not_reclassified_from_pooled_response=True,
        paper_1000_sessions_verified=paper_sessions,
        paper_10000_paths_per_session_verified=paper_paths,
        formal_cell_calibration_context_verified=context.formal_context,
        all_session_receipts_formal_ready=all(
            receipt.formal_session_classification_ready for receipt in ordered
        ),
        formal_paper_mechanism_result_ready=formal,
        paper_specifies_session_pooling_and_finite_sample_calibration=False,
        checksum_detects_stale_replacement_not_authentication=True,
        receipt_payload_sha256="",
    )
    receipt = replace(receipt, receipt_payload_sha256=_cell_receipt_digest(receipt))
    validate_step35f_cell_receipt(receipt)
    return receipt


def main() -> None:
    """Run one short genuine wiring demo; this is not a paper result.

    运行一个真实但很短的接线演示；这不是论文结果。
    """

    from step_28_session_phases import SessionPhase
    from steps.step_35c_irf_long_run_baseline import _build_demo_controller
    from steps.step_35d_unshocked_t3_calibration_paths import (
        run_unshocked_t3_calibration_paths,
    )

    controller, scorer = _build_demo_controller()
    while controller.phase is SessionPhase.TRAINING:
        if controller.training_periods_completed >= 5:
            raise TimeoutError("Debug convergence was not reached. / 调试收敛尚未达到。")
        controller.run_next_period()
    checkpoint = scorer.capture_and_bind_convergence_checkpoint(controller)
    controller.run_until_complete()
    scorer.finalize(controller)
    source = run_unshocked_t3_calibration_paths(
        checkpoint,
        baseline_scorer=scorer,
        irf_experiment_seed=20_260_835,
        path_count=100,
    )
    calibration = calibrate_experiment_cell_uniform_shock(
        (source,),
        expected_session_count=1,
    )
    context = prepare_verified_step35f_cell_context(calibration, (source,))
    session_result = run_step35f_session_response_paths(
        checkpoint,
        baseline_scorer=scorer,
        context=context,
    )
    cell_result = aggregate_step35f_experiment_cell(context, (session_result,))
    print("Step 35F: paired t=4 response / 第 35F 步：配对 t=4 反应")
    print(f"Debug paths / 调试路径数: {session_result.paths_executed}")
    print(f"Common shock magnitude / 统一冲击幅度: {session_result.selected_absolute_noise_shock:.9f}")
    print(
        "Executed cell price deviation / 实际实验单元价格偏差: "
        f"{cell_result.achieved_normalized_treatment_price_level_deviation:.9%}"
    )
    print(
        "Agent responses / 两位 agent 反应: "
        f"{session_result.normalized_order_response_by_agent}"
    )
    print(f"Provisional label / 调试标签: {session_result.mechanism_label.value}")
    print("Paper scale / 论文规模: False (1 session x 100 paths / 1个session乘100条路径)")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
