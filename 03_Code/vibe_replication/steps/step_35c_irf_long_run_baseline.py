"""Step 35C: collect one session's long-run IRF denominators.

步骤 35C：收集一个 session 的长期 IRF 分母。

Run / 运行:
    py -3 -X utf8 steps/step_35c_irf_long_run_baseline.py

Why this comes before 10,000 shocked paths / 为什么它必须先于一万条冲击路径:
    Figure 3 normalizes prices, profits, and orders by long-run expectations. The paper
    defines the transformations but does not state how those expectations are
    estimated. Our disclosed baseline uses the same session's post-convergence
    frozen-policy measurement window (100,000 periods in paper mode). It never
    substitutes Step 35B's unshocked control branch for a long-run denominator.
    / 图 3 用长期期望标准化价格、利润与订单。原文定义了转换，却没有说明期望如何
    估计。本复现公开选择同一 session 收敛后的固定策略测量窗口（论文模式为
    100,000 期）。第 35B 步的无冲击对照分支绝不会被偷换成长期分母。

This step only supplies per-session baseline statistics. It runs zero IRF
paths, applies zero shocks, and issues no mechanism classification. / 本步骤只
提供逐 session 的基准统计：不运行 IRF 路径、不施加冲击、也不进行机制分类。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import isclose, isfinite
from numbers import Integral, Real
from pathlib import Path
import pickle
import sys
from types import MethodType

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from step_05_speculator_profit import calculate_profit
from step_25_one_market_period import build_paper_inputs
from step_26_reproducible_random_streams import (
    FrozenPolicyPeriodObservation,
    RandomizedMarketSession,
    SessionSeedManifest,
    build_randomized_paper_session,
)
from step_28_session_phases import (
    PAPER_MEASUREMENT_PERIODS,
    SessionPhase,
    SessionPhaseController,
    SessionPhaseReceipt,
)
from step_27_convergence_tracker import PAPER_UNCHANGED_PERIODS
from step_30_trading_intensity import MeasurementSinkFanout
from steps.step_34_mechanism_classifier import (
    PAPER_ORIENTED_ORDER_FORMULA,
    PAPER_ORIENTED_PRICE_FORMULA,
    orient_order,
    orient_price,
    orientation_sign,
)
from steps.step_35a_converged_market_checkpoint import (
    ConvergedMarketCheckpoint,
    capture_at_convergence_boundary,
    verify_converged_market_checkpoint,
)


IRF_LONG_RUN_BASELINE_VERSION = "step35c-session-long-run-baseline-v2"
IRF_EXPECTATION_ESTIMATOR = (
    "same session's post-convergence frozen-policy measurement window"
)
OBSERVATION_DIGEST_DOMAIN = b"vibe-replication.step35c.scored-fields.v2\0"


def _finite_real(number: float, label: str) -> float:
    """Return one finite non-Boolean real. / 返回有限、非布尔实数。"""

    if isinstance(number, bool) or not isinstance(number, Real):
        raise TypeError(f"{label} must be a real number. / {label} 必须是实数。")
    converted = float(number)
    if not isfinite(converted):
        raise ValueError(f"{label} must be finite. / {label} 必须是有限数。")
    return converted


def _next_neumaier_sum(
    running_sum: float,
    compensation: float,
    value: float,
) -> tuple[float, float]:
    """Prepare one constant-memory compensated-sum update.

    准备一次固定内存的补偿求和更新。
    """

    candidate_sum = running_sum + value
    if abs(running_sum) >= abs(value):
        candidate_compensation = compensation + (
            running_sum - candidate_sum + value
        )
    else:
        candidate_compensation = compensation + (
            value - candidate_sum + running_sum
        )
    if not isfinite(candidate_sum) or not isfinite(candidate_compensation):
        raise OverflowError("Long-run sum overflowed. / 长期统计求和溢出。")
    return candidate_sum, candidate_compensation


def _sum_value(state: tuple[float, float]) -> float:
    """Read one compensated total. / 读取一个补偿求和总值。"""

    total = state[0] + state[1]
    if not isfinite(total):
        raise OverflowError("Compensated total overflowed. / 补偿总值溢出。")
    return total


@dataclass(frozen=True)
class IRFLongRunMomentSummary:
    """Pure means prepared before session/provenance checks.

    在 session 与来源核对之前得到的纯统计均值。
    """

    observations: int
    number_of_agents: int
    mean_oriented_price: float
    mean_centered_raw_price: float
    mean_oriented_order_by_agent: tuple[float, ...]
    mean_raw_order_by_agent: tuple[float, ...]
    mean_profit_by_agent: tuple[float, ...]
    mean_price_impact_lambda: float
    minimum_price_impact_lambda: float
    nonpositive_price_impact_count: int
    value_above_mean_count: int
    value_below_mean_count: int
    value_equal_mean_count: int
    oriented_before_averaging: bool


class OnlineIRFLongRunMoments:
    """Constant-memory means for Figure-3 normalization ingredients.

    用固定内存计算图 3 标准化所需均值。
    """

    def __init__(self, number_of_agents: int) -> None:
        if (
            isinstance(number_of_agents, bool)
            or not isinstance(number_of_agents, Integral)
            or int(number_of_agents) < 1
        ):
            raise ValueError("number_of_agents must be positive. / agent 数量必须为正。")
        self.number_of_agents = int(number_of_agents)
        self.count = 0
        self._oriented_price_sum = (0.0, 0.0)
        self._centered_raw_price_sum = (0.0, 0.0)
        self._oriented_order_sums = [
            (0.0, 0.0) for _ in range(self.number_of_agents)
        ]
        self._raw_order_sums = [
            (0.0, 0.0) for _ in range(self.number_of_agents)
        ]
        self._profit_sums = [
            (0.0, 0.0) for _ in range(self.number_of_agents)
        ]
        self._lambda_sum = (0.0, 0.0)
        self.minimum_price_impact_lambda = float("inf")
        self.nonpositive_price_impact_count = 0
        self.value_above_mean_count = 0
        self.value_below_mean_count = 0
        self.value_equal_mean_count = 0

    def audit_state(self) -> tuple[object, ...]:
        """Return an immutable state tuple for atomicity tests.

        返回不可修改状态元组，用于原子性测试。
        """

        return (
            self.count,
            self._oriented_price_sum,
            self._centered_raw_price_sum,
            tuple(self._oriented_order_sums),
            tuple(self._raw_order_sums),
            tuple(self._profit_sums),
            self._lambda_sum,
            self.minimum_price_impact_lambda,
            self.nonpositive_price_impact_count,
            self.value_above_mean_count,
            self.value_below_mean_count,
            self.value_equal_mean_count,
        )

    def add(
        self,
        *,
        fundamental_value_v: float,
        continuous_price_p: float,
        raw_orders_x: tuple[float, ...],
        profits: tuple[float, ...],
        price_impact_lambda_hat: float,
        value_mean: float,
    ) -> None:
        """Validate a row completely, then commit every moment together.

        先完整检查一行，再一次性提交全部统计量。
        """

        value = _finite_real(fundamental_value_v, "fundamental_value_v")
        price = _finite_real(continuous_price_p, "continuous_price_p")
        price_impact = _finite_real(
            price_impact_lambda_hat,
            "price_impact_lambda_hat",
        )
        mean_value = _finite_real(value_mean, "value_mean")
        if len(raw_orders_x) != self.number_of_agents:
            raise ValueError("There must be one order per agent. / 每位 agent 必须有一个订单。")
        if len(profits) != self.number_of_agents:
            raise ValueError("There must be one profit per agent. / 每位 agent 必须有一个利润。")
        orders = tuple(
            _finite_real(order, f"raw_order_{index + 1}")
            for index, order in enumerate(raw_orders_x)
        )
        checked_profits = tuple(
            _finite_real(profit, f"profit_{index + 1}")
            for index, profit in enumerate(profits)
        )

        direction = orientation_sign(value, mean_value)
        oriented_price = orient_price(price, value, mean_value)
        oriented_orders = tuple(
            orient_order(order, value, mean_value) for order in orders
        )
        centered_raw_price = price - mean_value
        if not isfinite(centered_raw_price):
            raise OverflowError("Centered raw price overflowed. / 原始中心化价格溢出。")

        # Prepare all candidate sums first. No object state changes above this
        # line. / 先准备全部候选求和；此行之前不会修改对象状态。
        candidate_oriented_price = _next_neumaier_sum(
            *self._oriented_price_sum,
            oriented_price,
        )
        candidate_centered_price = _next_neumaier_sum(
            *self._centered_raw_price_sum,
            centered_raw_price,
        )
        candidate_oriented_orders = [
            _next_neumaier_sum(*state, order)
            for state, order in zip(
                self._oriented_order_sums,
                oriented_orders,
                strict=True,
            )
        ]
        candidate_raw_orders = [
            _next_neumaier_sum(*state, order)
            for state, order in zip(
                self._raw_order_sums,
                orders,
                strict=True,
            )
        ]
        candidate_profits = [
            _next_neumaier_sum(*state, profit)
            for state, profit in zip(
                self._profit_sums,
                checked_profits,
                strict=True,
            )
        ]
        candidate_lambda = _next_neumaier_sum(
            *self._lambda_sum,
            price_impact,
        )

        self._oriented_price_sum = candidate_oriented_price
        self._centered_raw_price_sum = candidate_centered_price
        self._oriented_order_sums = candidate_oriented_orders
        self._raw_order_sums = candidate_raw_orders
        self._profit_sums = candidate_profits
        self._lambda_sum = candidate_lambda
        self.minimum_price_impact_lambda = min(
            self.minimum_price_impact_lambda,
            price_impact,
        )
        self.nonpositive_price_impact_count += int(price_impact <= 0.0)
        self.value_above_mean_count += int(direction > 0)
        self.value_below_mean_count += int(direction < 0)
        self.value_equal_mean_count += int(direction == 0)
        self.count += 1

    def summarize(self) -> IRFLongRunMomentSummary:
        """Return means without mutating the online accumulator.

        返回均值，不修改在线累加器。
        """

        if self.count < 1:
            raise RuntimeError("No long-run observations were added. / 尚未加入长期观测。")
        divisor = float(self.count)
        mean_oriented_price = _sum_value(self._oriented_price_sum) / divisor
        mean_centered_price = _sum_value(self._centered_raw_price_sum) / divisor
        mean_oriented_orders = tuple(
            _sum_value(state) / divisor for state in self._oriented_order_sums
        )
        mean_raw_orders = tuple(
            _sum_value(state) / divisor for state in self._raw_order_sums
        )
        mean_profits = tuple(
            _sum_value(state) / divisor for state in self._profit_sums
        )
        mean_lambda = _sum_value(self._lambda_sum) / divisor
        all_means = (
            mean_oriented_price,
            mean_centered_price,
            *mean_oriented_orders,
            *mean_raw_orders,
            *mean_profits,
            mean_lambda,
        )
        if not all(isfinite(number) for number in all_means):
            raise OverflowError("A long-run mean is not finite. / 某个长期均值不是有限数。")
        return IRFLongRunMomentSummary(
            observations=self.count,
            number_of_agents=self.number_of_agents,
            mean_oriented_price=mean_oriented_price,
            mean_centered_raw_price=mean_centered_price,
            mean_oriented_order_by_agent=mean_oriented_orders,
            mean_raw_order_by_agent=mean_raw_orders,
            mean_profit_by_agent=mean_profits,
            mean_price_impact_lambda=mean_lambda,
            minimum_price_impact_lambda=self.minimum_price_impact_lambda,
            nonpositive_price_impact_count=self.nonpositive_price_impact_count,
            value_above_mean_count=self.value_above_mean_count,
            value_below_mean_count=self.value_below_mean_count,
            value_equal_mean_count=self.value_equal_mean_count,
            oriented_before_averaging=True,
        )


@dataclass(frozen=True)
class IRFLongRunBaselineReceipt:
    """Immutable, session-bound Step-35C baseline result.

    不可修改、绑定一个 session 的第 35C 步基准结果。
    """

    estimator_version: str
    expectation_estimator: str
    measurement_periods_scored: int
    first_measurement_index: int
    last_measurement_index: int
    first_global_period_index: int
    last_global_period_index: int
    number_of_agents: int
    mean_oriented_price: float
    mean_centered_raw_price: float
    mean_oriented_order_by_agent: tuple[float, ...]
    mean_raw_order_by_agent: tuple[float, ...]
    mean_profit_by_agent: tuple[float, ...]
    mean_price_impact_lambda: float
    minimum_price_impact_lambda: float
    nonpositive_price_impact_count: int
    value_above_mean_count: int
    value_below_mean_count: int
    value_equal_mean_count: int
    value_mean_parameter: float
    parameter_snapshot: PaperParameters
    value_grid_snapshot: tuple[float, ...]
    session_seed_manifest: SessionSeedManifest
    session_phase_receipt: SessionPhaseReceipt
    scored_fields_sha256: str
    source_checkpoint_sha256: str
    source_implementation_tree_sha256: str
    oriented_price_formula: str
    oriented_order_formula: str
    paper_defines_expectations_but_not_estimator: bool
    same_session_post_convergence_window_is_replication_interpretation: bool
    per_session_denominator_is_replication_interpretation: bool
    oriented_before_averaging: bool
    unshocked_control_used_as_denominator: bool
    per_session_denominator_not_pooled_across_sessions: bool
    measurement_sink_delivery_verified: bool
    exact_convergence_checkpoint_provenance_verified: bool
    same_session_scored_sample_provenance_verified: bool
    complete_raw_observation_digest_included: bool
    paper_post_convergence_measurement_length_100000_verified: bool
    paper_links_that_window_to_irf_denominator: bool
    paper_convergence_threshold_1000000_verified: bool
    paper_scale_thresholds_and_provenance_verified: bool
    ready_for_price_normalization: bool
    ready_for_order_normalization: bool
    ready_for_profit_normalization: bool
    ready_for_all_figure3_normalizations: bool
    long_run_lambdas_all_positive: bool
    long_run_lambda_is_diagnostic_not_shock_calibration_input: bool
    ready_for_shock_calibration: bool
    irf_paths_aggregated: int
    shock_applied: bool
    classification_ready: bool
    checksum_detects_stale_replacement_not_authentication: bool
    standalone_receipt_authenticates_streamed_means: bool
    step35d_must_use_live_scorer_and_matching_checkpoint: bool
    receipt_payload_sha256: str


def _digest_scored_fields(
    measurement_index: int,
    observation: FrozenPolicyPeriodObservation,
) -> bytes:
    """Encode only the denominator fields that this scorer validates and uses.

    只确定性编码本 scorer 已验证且实际使用的分母字段。
    """

    fields: tuple[object, ...] = (
        measurement_index,
        observation.period_number,
        observation.current_value_index,
        observation.fundamental_value_v.hex(),
        tuple(int(index) for index in observation.action_indexes),
        tuple(float(order).hex() for order in observation.raw_orders_x),
        observation.price_impact_lambda_hat.hex(),
        observation.continuous_price_p.hex(),
        tuple(float(profit).hex() for profit in observation.profits),
    )
    return repr(fields).encode("ascii") + b"\0"


def _measurement_sink_key(sink: object) -> tuple[str, int, int | None]:
    """Normalize a callable identity exactly as the official Step-30 fan-out.

    按照正式 Step-30 fan-out 的同一规则规范化 callable 身份。
    """

    if isinstance(sink, MethodType):
        return ("bound_method", id(sink.__self__), id(sink.__func__))
    return ("callable", id(sink), None)


def _controller_delivers_to_scorer(
    controller: SessionPhaseController,
    scorer: "OnlineIRFLongRunBaselineScorer",
) -> bool:
    """Check direct registration or authenticated Step-30 fan-out membership.

    检查 scorer 是直接注册的 sink，或正式 Step-30 fan-out 的成员。
    """

    registered_sink = controller.measurement_sink
    if registered_sink is None:
        return False
    scorer_key = _measurement_sink_key(scorer.observe)
    if _measurement_sink_key(registered_sink) == scorer_key:
        return True
    if type(registered_sink) is MeasurementSinkFanout:
        return registered_sink.contains_sink(scorer.observe)
    return False


def _receipt_payload_digest(receipt: IRFLongRunBaselineReceipt) -> str:
    """Checksum every receipt field except the checksum itself.

    对 receipt 中除校验码本身外的全部字段求校验码。
    """

    unsigned = replace(receipt, receipt_payload_sha256="")
    return sha256(pickle.dumps(unsigned, protocol=5)).hexdigest()


def _is_sha256_text(value: object) -> bool:
    """Return whether a value is 64 lowercase hexadecimal characters.

    判断一个值是否为 64 位小写十六进制 SHA-256 文本。
    """

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_irf_long_run_baseline_receipt(
    receipt: IRFLongRunBaselineReceipt,
) -> None:
    """Reject a changed or internally inconsistent Step-35C receipt.

    拒绝被替换或内部不一致的第 35C 步 receipt。

    This validator cannot recreate 100,000 discarded streaming rows. Its
    unkeyed checksum detects an ordinary stale ``dataclasses.replace`` copy,
    but is not cryptographic authentication because anyone can recompute it.
    Step 35D must consume the live scorer result and the exact matching
    checkpoint. / 本验证器无法重新生成已经流式丢弃的十万行。无密钥校验码可以
    发现普通、过期的 ``dataclasses.replace`` 副本，但任何人都能重新计算它，
    因而不构成密码学认证。第 35D 步必须使用实时 scorer 结果及精确匹配的 checkpoint。
    """

    if not isinstance(receipt, IRFLongRunBaselineReceipt):
        raise TypeError("receipt has the wrong type. / receipt 类型错误。")
    if receipt.estimator_version != IRF_LONG_RUN_BASELINE_VERSION:
        raise ValueError("Baseline receipt version is unsupported. / 基准 receipt 版本不支持。")
    if not _is_sha256_text(receipt.receipt_payload_sha256):
        raise ValueError("Receipt checksum has the wrong format. / receipt 校验码格式错误。")
    if _receipt_payload_digest(receipt) != receipt.receipt_payload_sha256:
        raise ValueError("Baseline receipt checksum failed. / 基准 receipt 校验失败。")
    if not _is_sha256_text(receipt.scored_fields_sha256):
        raise ValueError("Scored-fields digest is invalid. / 计分字段摘要无效。")
    if not _is_sha256_text(receipt.source_checkpoint_sha256):
        raise ValueError("Checkpoint digest is invalid. / checkpoint 摘要无效。")
    if not _is_sha256_text(receipt.source_implementation_tree_sha256):
        raise ValueError("Implementation digest is invalid. / 源码摘要无效。")

    count = receipt.measurement_periods_scored
    if (
        isinstance(count, bool)
        or not isinstance(count, Integral)
        or int(count) < 1
        or receipt.first_measurement_index != 0
        or receipt.last_measurement_index != int(count) - 1
        or receipt.last_global_period_index - receipt.first_global_period_index + 1
        != int(count)
    ):
        raise ValueError("Measurement boundaries are inconsistent. / 测量边界不一致。")
    if receipt.number_of_agents < 1:
        raise ValueError("Agent count is invalid. / agent 数量无效。")
    vector_fields = (
        receipt.mean_oriented_order_by_agent,
        receipt.mean_raw_order_by_agent,
        receipt.mean_profit_by_agent,
    )
    if any(len(values) != receipt.number_of_agents for values in vector_fields):
        raise ValueError("A receipt vector has the wrong length. / receipt 向量长度错误。")
    scalar_values = (
        receipt.mean_oriented_price,
        receipt.mean_centered_raw_price,
        receipt.mean_price_impact_lambda,
        receipt.minimum_price_impact_lambda,
        receipt.value_mean_parameter,
        *receipt.mean_oriented_order_by_agent,
        *receipt.mean_raw_order_by_agent,
        *receipt.mean_profit_by_agent,
        *receipt.value_grid_snapshot,
    )
    if any(not isfinite(float(value)) for value in scalar_values):
        raise ValueError("Receipt contains a nonfinite value. / receipt 含非有限数。")
    if (
        receipt.value_above_mean_count
        + receipt.value_below_mean_count
        + receipt.value_equal_mean_count
        != count
        or not 0 <= receipt.nonpositive_price_impact_count <= count
    ):
        raise ValueError("Diagnostic counts are inconsistent. / 诊断计数不一致。")

    price_ready = receipt.mean_oriented_price > 0.0
    order_ready = all(
        value > 0.0 for value in receipt.mean_oriented_order_by_agent
    )
    profit_ready = all(value != 0.0 for value in receipt.mean_profit_by_agent)
    lambda_positive = (
        receipt.mean_price_impact_lambda > 0.0
        and receipt.minimum_price_impact_lambda > 0.0
        and receipt.nonpositive_price_impact_count == 0
    )
    paper_measurement_length = (
        count == PAPER_MEASUREMENT_PERIODS
        and receipt.session_phase_receipt.measurement_periods_required
        == PAPER_MEASUREMENT_PERIODS
        and receipt.session_phase_receipt.measurement_periods_completed
        == PAPER_MEASUREMENT_PERIODS
    )
    paper_convergence = (
        receipt.session_phase_receipt.convergence_receipt.required_unchanged_periods
        == PAPER_UNCHANGED_PERIODS
    )
    provenance = (
        receipt.measurement_sink_delivery_verified
        and receipt.exact_convergence_checkpoint_provenance_verified
    )
    expected_scale_and_provenance = (
        paper_measurement_length and paper_convergence and provenance
    )
    logical_checks = (
        receipt.expectation_estimator == IRF_EXPECTATION_ESTIMATOR,
        receipt.oriented_price_formula == PAPER_ORIENTED_PRICE_FORMULA,
        receipt.oriented_order_formula == PAPER_ORIENTED_ORDER_FORMULA,
        receipt.paper_defines_expectations_but_not_estimator,
        receipt.same_session_post_convergence_window_is_replication_interpretation,
        receipt.per_session_denominator_is_replication_interpretation,
        receipt.oriented_before_averaging,
        not receipt.unshocked_control_used_as_denominator,
        receipt.per_session_denominator_not_pooled_across_sessions,
        receipt.same_session_scored_sample_provenance_verified == provenance,
        not receipt.complete_raw_observation_digest_included,
        receipt.paper_post_convergence_measurement_length_100000_verified
        == paper_measurement_length,
        not receipt.paper_links_that_window_to_irf_denominator,
        receipt.paper_convergence_threshold_1000000_verified == paper_convergence,
        receipt.paper_scale_thresholds_and_provenance_verified
        == expected_scale_and_provenance,
        receipt.ready_for_price_normalization == price_ready,
        receipt.ready_for_order_normalization == order_ready,
        receipt.ready_for_profit_normalization == profit_ready,
        receipt.ready_for_all_figure3_normalizations
        == (price_ready and order_ready and profit_ready),
        receipt.long_run_lambdas_all_positive == lambda_positive,
        receipt.long_run_lambda_is_diagnostic_not_shock_calibration_input,
        not receipt.ready_for_shock_calibration,
        receipt.irf_paths_aggregated == 0,
        not receipt.shock_applied,
        not receipt.classification_ready,
        receipt.checksum_detects_stale_replacement_not_authentication,
        not receipt.standalone_receipt_authenticates_streamed_means,
        receipt.step35d_must_use_live_scorer_and_matching_checkpoint,
    )
    if not all(logical_checks):
        raise ValueError("Baseline receipt claims are inconsistent. / 基准 receipt 声明不一致。")


class OnlineIRFLongRunBaselineScorer:
    """Session-bound Step-28 sink that owns the long-run denominator sample.

    绑定 Step-28 session、拥有长期分母样本的在线 scorer。
    """

    def __init__(self, session: RandomizedMarketSession) -> None:
        if not isinstance(session, RandomizedMarketSession):
            raise TypeError("session has the wrong type. / session 类型错误。")
        if session.period_number != 0 or session.execution_mode != "training":
            raise RuntimeError("Attach the scorer to a fresh session. / 请把 scorer 连接到新 session。")
        self._session = session
        self._parameter_snapshot = session.parameters
        self._value_grid_snapshot = tuple(float(value) for value in session.value_grid)
        self._seed_manifest_snapshot = session.streams.manifest
        self._moments = OnlineIRFLongRunMoments(
            session.parameters.num_speculators
        )
        self._digest = sha256(OBSERVATION_DIGEST_DOMAIN)
        self.rows_scored = 0
        self.first_global_period_index: int | None = None
        self.last_global_period_index: int | None = None
        self._bound_controller: SessionPhaseController | None = None
        self._source_checkpoint: ConvergedMarketCheckpoint | None = None
        self._final_receipt: IRFLongRunBaselineReceipt | None = None

    def capture_and_bind_convergence_checkpoint(
        self,
        controller: SessionPhaseController,
    ) -> ConvergedMarketCheckpoint:
        """Capture the exact Step-35A origin before accepting measurement rows.

        在接收测量记录前，保存并绑定精确的 Step-35A 收敛起点。

        The returned checkpoint must be reused by Step 35D for its IRF branches.
        / 返回的 checkpoint 必须由第 35D 步复用于 IRF 分支。
        """

        if not isinstance(controller, SessionPhaseController):
            raise TypeError("controller has the wrong type. / controller 类型错误。")
        if controller.session is not self._session:
            raise ValueError("controller belongs to another session. / controller 属于另一个 session。")
        if self.rows_scored != 0 or self._final_receipt is not None:
            raise RuntimeError("Bind before the first baseline row. / 必须在第一条基准记录前绑定。")
        if self._source_checkpoint is not None:
            if controller is not self._bound_controller:
                raise RuntimeError("The scorer is already bound elsewhere. / scorer 已绑定到其他 controller。")
            return self._source_checkpoint
        if not _controller_delivers_to_scorer(controller, self):
            raise RuntimeError(
                "The controller's registered sink does not contain this scorer. "
                "/ controller 注册的 sink 不包含本 scorer。"
            )
        checkpoint = capture_at_convergence_boundary(controller)
        verify_converged_market_checkpoint(checkpoint)
        if (
            checkpoint.payload.parameters != self._parameter_snapshot
            or checkpoint.payload.value_grid != self._value_grid_snapshot
            or checkpoint.payload.seed_manifest != self._seed_manifest_snapshot
        ):
            raise RuntimeError("Checkpoint context differs from the scorer. / checkpoint 环境与 scorer 不一致。")
        self._bound_controller = controller
        self._source_checkpoint = checkpoint
        return checkpoint

    def observe(
        self,
        measurement_index: int,
        observation: FrozenPolicyPeriodObservation,
    ) -> None:
        """Validate and add one actual Step-28 measurement row.

        检查并加入一条真实的 Step-28 测量记录。
        """

        if self._final_receipt is not None:
            raise RuntimeError("The scorer is already final. / scorer 已经完成。")
        if self._source_checkpoint is None or self._bound_controller is None:
            raise RuntimeError(
                "Capture and bind the convergence checkpoint before observing rows. "
                "/ 接收记录前必须保存并绑定收敛 checkpoint。"
            )
        if (
            self._bound_controller.phase is not SessionPhase.MEASUREMENT
            or self._bound_controller.measurement_periods_completed
            != self.rows_scored
        ):
            raise RuntimeError(
                "Rows are accepted only during the controller's live sink call. "
                "/ 只在 controller 实时调用 sink 时接收记录。"
            )
        if (
            isinstance(measurement_index, bool)
            or not isinstance(measurement_index, Integral)
            or int(measurement_index) != self.rows_scored
        ):
            raise ValueError("Measurement indexes must be consecutive from zero. / 测量编号必须从零连续。")
        if not isinstance(observation, FrozenPolicyPeriodObservation):
            raise TypeError("observation has the wrong type. / observation 类型错误。")
        if (
            isinstance(observation.period_number, bool)
            or not isinstance(observation.period_number, Integral)
            or int(observation.period_number) < 0
        ):
            raise ValueError("period_number is invalid. / period_number 无效。")
        period_number = int(observation.period_number)
        if self._session.period_number != period_number + 1:
            raise RuntimeError(
                "The observation is not the session's just-completed period. "
                "/ observation 不是该 session 刚完成的时期。"
            )
        if (
            self.last_global_period_index is not None
            and period_number != self.last_global_period_index + 1
        ):
            raise ValueError("Global periods must be consecutive. / 全局时期必须连续。")

        value_index = observation.current_value_index
        if (
            isinstance(value_index, bool)
            or not isinstance(value_index, Integral)
            or not 0 <= int(value_index) < len(self._value_grid_snapshot)
        ):
            raise ValueError("current_value_index is invalid. / 当前价值编号无效。")
        value_index = int(value_index)
        expected_value = self._value_grid_snapshot[value_index]
        if not isclose(
            observation.fundamental_value_v,
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("v_t does not match its grid index. / v_t 与网格编号不一致。")

        if len(observation.action_indexes) != self._moments.number_of_agents:
            raise ValueError("There must be one action per agent. / 每位 agent 必须有一个动作。")
        available_orders = self._session.orders_by_value_and_action[value_index]
        checked_action_indexes: list[int] = []
        for action_index in observation.action_indexes:
            if (
                isinstance(action_index, bool)
                or not isinstance(action_index, Integral)
                or not 0 <= int(action_index) < len(available_orders)
            ):
                raise ValueError("An action index is invalid. / 某个动作编号无效。")
            checked_action_indexes.append(int(action_index))
        if len(observation.raw_orders_x) != self._moments.number_of_agents:
            raise ValueError("There must be one raw order per agent. / 每位 agent 必须有一个原始订单。")
        for order, action_index in zip(
            observation.raw_orders_x,
            checked_action_indexes,
            strict=True,
        ):
            checked_order = _finite_real(order, "raw_order")
            if not isclose(
                checked_order,
                available_orders[action_index],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("Raw order disagrees with value/action. / 原始订单与价值或动作不一致。")

        price = _finite_real(observation.continuous_price_p, "continuous_price_p")
        price_impact = _finite_real(
            observation.price_impact_lambda_hat,
            "price_impact_lambda_hat",
        )
        total_flow = _finite_real(observation.total_order_flow_y, "total_order_flow_y")
        gamma_0 = _finite_real(observation.gamma_0_hat, "gamma_0_hat")
        expected_price = gamma_0 + price_impact * total_flow
        if not isfinite(expected_price) or not isclose(
            price,
            expected_price,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("Price disagrees with its recorded adaptive rule. / 价格与记录的自适应规则不一致。")
        if len(observation.profits) != self._moments.number_of_agents:
            raise ValueError("There must be one profit per agent. / 每位 agent 必须有一个利润。")
        for profit, order in zip(
            observation.profits,
            observation.raw_orders_x,
            strict=True,
        ):
            checked_profit = _finite_real(profit, "profit")
            expected_profit = calculate_profit(
                observation.fundamental_value_v,
                price,
                float(order),
            )
            if not isclose(
                checked_profit,
                expected_profit,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("Profit disagrees with (v-p)x. / 利润与 (v-p)x 不一致。")

        # ``add`` prepares every numerical update before committing it. The
        # digest copy is likewise committed only after the row succeeds. / add
        # 会先准备全部数值更新；digest 副本也只在本行成功后提交。
        candidate_digest = self._digest.copy()
        candidate_digest.update(_digest_scored_fields(int(measurement_index), observation))
        self._moments.add(
            fundamental_value_v=observation.fundamental_value_v,
            continuous_price_p=price,
            raw_orders_x=tuple(observation.raw_orders_x),
            profits=tuple(observation.profits),
            price_impact_lambda_hat=price_impact,
            value_mean=self._parameter_snapshot.value_mean,
        )
        self._digest = candidate_digest
        if self.rows_scored == 0:
            self.first_global_period_index = period_number
        self.last_global_period_index = period_number
        self.rows_scored += 1

    def finalize(
        self,
        controller: SessionPhaseController,
    ) -> IRFLongRunBaselineReceipt:
        """Issue one receipt only after the bound measurement completes.

        只有绑定的测量完成后才签发凭证。
        """

        if not isinstance(controller, SessionPhaseController):
            raise TypeError("controller has the wrong type. / controller 类型错误。")
        if controller.session is not self._session:
            raise ValueError("controller belongs to another session. / controller 属于另一个 session。")
        if controller is not self._bound_controller or self._source_checkpoint is None:
            raise RuntimeError("The convergence checkpoint is not bound. / 收敛 checkpoint 尚未绑定。")
        if not _controller_delivers_to_scorer(controller, self):
            raise RuntimeError("The registered measurement sink changed. / 注册的测量 sink 已改变。")
        verify_converged_market_checkpoint(self._source_checkpoint)
        if self._final_receipt is not None:
            return self._final_receipt
        if controller.phase is not SessionPhase.COMPLETE or controller.final_receipt is None:
            raise RuntimeError("The bound measurement is not complete. / 绑定的测量尚未完成。")
        phase_receipt = controller.final_receipt
        if (
            self.rows_scored != phase_receipt.measurement_periods_completed
            or self.rows_scored != controller.measurement_periods_required
            or self.first_global_period_index
            != phase_receipt.measurement_first_period_index
            or self.last_global_period_index
            != phase_receipt.measurement_last_period_index
        ):
            raise RuntimeError("Baseline rows do not match the measurement receipt. / 基准记录与测量凭证不一致。")
        if (
            self._session.parameters != self._parameter_snapshot
            or tuple(self._session.value_grid) != self._value_grid_snapshot
            or self._session.streams.manifest != self._seed_manifest_snapshot
        ):
            raise RuntimeError("Session context changed after attachment. / 连接后 session 环境发生改变。")

        summary = self._moments.summarize()
        price_ready = summary.mean_oriented_price > 0.0
        order_ready = all(
            order_mean > 0.0
            for order_mean in summary.mean_oriented_order_by_agent
        )
        profit_ready = all(
            profit_mean != 0.0
            for profit_mean in summary.mean_profit_by_agent
        )
        long_run_lambdas_positive = (
            summary.mean_price_impact_lambda > 0.0
            and summary.minimum_price_impact_lambda > 0.0
            and summary.nonpositive_price_impact_count == 0
        )
        paper_measurement_length = (
            self.rows_scored == PAPER_MEASUREMENT_PERIODS
            and phase_receipt.measurement_periods_required
            == PAPER_MEASUREMENT_PERIODS
        )
        paper_convergence = (
            phase_receipt.convergence_receipt.required_unchanged_periods
            == PAPER_UNCHANGED_PERIODS
        )
        receipt = IRFLongRunBaselineReceipt(
            estimator_version=IRF_LONG_RUN_BASELINE_VERSION,
            expectation_estimator=IRF_EXPECTATION_ESTIMATOR,
            measurement_periods_scored=self.rows_scored,
            first_measurement_index=0,
            last_measurement_index=self.rows_scored - 1,
            first_global_period_index=self.first_global_period_index,  # type: ignore[arg-type]
            last_global_period_index=self.last_global_period_index,  # type: ignore[arg-type]
            number_of_agents=summary.number_of_agents,
            mean_oriented_price=summary.mean_oriented_price,
            mean_centered_raw_price=summary.mean_centered_raw_price,
            mean_oriented_order_by_agent=(
                summary.mean_oriented_order_by_agent
            ),
            mean_raw_order_by_agent=summary.mean_raw_order_by_agent,
            mean_profit_by_agent=summary.mean_profit_by_agent,
            mean_price_impact_lambda=summary.mean_price_impact_lambda,
            minimum_price_impact_lambda=(
                summary.minimum_price_impact_lambda
            ),
            nonpositive_price_impact_count=(
                summary.nonpositive_price_impact_count
            ),
            value_above_mean_count=summary.value_above_mean_count,
            value_below_mean_count=summary.value_below_mean_count,
            value_equal_mean_count=summary.value_equal_mean_count,
            value_mean_parameter=self._parameter_snapshot.value_mean,
            parameter_snapshot=self._parameter_snapshot,
            value_grid_snapshot=self._value_grid_snapshot,
            session_seed_manifest=self._seed_manifest_snapshot,
            session_phase_receipt=phase_receipt,
            scored_fields_sha256=self._digest.hexdigest(),
            source_checkpoint_sha256=(
                self._source_checkpoint.checkpoint_sha256
            ),
            source_implementation_tree_sha256=(
                self._source_checkpoint.payload.implementation_tree_sha256
            ),
            oriented_price_formula=PAPER_ORIENTED_PRICE_FORMULA,
            oriented_order_formula=PAPER_ORIENTED_ORDER_FORMULA,
            paper_defines_expectations_but_not_estimator=True,
            same_session_post_convergence_window_is_replication_interpretation=True,
            per_session_denominator_is_replication_interpretation=True,
            oriented_before_averaging=True,
            unshocked_control_used_as_denominator=False,
            per_session_denominator_not_pooled_across_sessions=True,
            measurement_sink_delivery_verified=True,
            exact_convergence_checkpoint_provenance_verified=True,
            same_session_scored_sample_provenance_verified=True,
            complete_raw_observation_digest_included=False,
            paper_post_convergence_measurement_length_100000_verified=(
                paper_measurement_length
            ),
            paper_links_that_window_to_irf_denominator=False,
            paper_convergence_threshold_1000000_verified=paper_convergence,
            paper_scale_thresholds_and_provenance_verified=(
                paper_measurement_length and paper_convergence
            ),
            ready_for_price_normalization=price_ready,
            ready_for_order_normalization=order_ready,
            ready_for_profit_normalization=profit_ready,
            ready_for_all_figure3_normalizations=(
                price_ready and order_ready and profit_ready
            ),
            long_run_lambdas_all_positive=long_run_lambdas_positive,
            long_run_lambda_is_diagnostic_not_shock_calibration_input=True,
            ready_for_shock_calibration=False,
            irf_paths_aggregated=0,
            shock_applied=False,
            classification_ready=False,
            checksum_detects_stale_replacement_not_authentication=True,
            standalone_receipt_authenticates_streamed_means=False,
            step35d_must_use_live_scorer_and_matching_checkpoint=True,
            receipt_payload_sha256="",
        )
        receipt = replace(
            receipt,
            receipt_payload_sha256=_receipt_payload_digest(receipt),
        )
        validate_irf_long_run_baseline_receipt(receipt)
        self._final_receipt = receipt
        return receipt

    def verified_live_result_for_step35d(
        self,
        checkpoint: ConvergedMarketCheckpoint,
    ) -> IRFLongRunBaselineReceipt:
        """Return the live final result bound to this exact checkpoint object.

        返回与这个 checkpoint 对象精确绑定的实时最终结果，供第 35D 步使用。

        A copied receipt can carry a recomputed unkeyed checksum, so Step 35D
        must ask the scorer that actually observed the Step-28 stream.  Object
        identity here is deliberate: it joins the discarded streaming sample
        to the checkpoint captured immediately before that sample began. / 复制
        receipt 后可以重算无密钥 checksum，因此第 35D 步必须向真正接收过
        Step-28 数据流的 scorer 取结果。这里故意核对对象身份，用来把已经流式
        丢弃的样本与样本开始前保存的 checkpoint 连接起来。
        """

        if not isinstance(checkpoint, ConvergedMarketCheckpoint):
            raise TypeError("checkpoint has the wrong type. / checkpoint 类型错误。")
        if self._source_checkpoint is None or self._final_receipt is None:
            raise RuntimeError("The live baseline is not final. / 实时长期基准尚未完成。")
        if checkpoint is not self._source_checkpoint:
            raise ValueError(
                "Step 35D must reuse the exact checkpoint object bound by this scorer. / "
                "第 35D 步必须复用本 scorer 绑定的同一个 checkpoint 对象。"
            )
        if self._bound_controller is None:
            raise RuntimeError("The source controller is missing. / 来源 controller 丢失。")
        if (
            self._bound_controller.phase is not SessionPhase.COMPLETE
            or self._bound_controller.final_receipt is None
            or not _controller_delivers_to_scorer(self._bound_controller, self)
        ):
            raise RuntimeError("The live measurement provenance no longer verifies. / 实时测量来源不再通过核对。")
        verify_converged_market_checkpoint(checkpoint)
        validate_irf_long_run_baseline_receipt(self._final_receipt)
        if (
            self._final_receipt.source_checkpoint_sha256
            != checkpoint.checkpoint_sha256
            or self._final_receipt.source_implementation_tree_sha256
            != checkpoint.payload.implementation_tree_sha256
            or self._final_receipt.session_seed_manifest
            != checkpoint.payload.seed_manifest
            or self._final_receipt.parameter_snapshot != checkpoint.payload.parameters
            or self._final_receipt.value_grid_snapshot != checkpoint.payload.value_grid
        ):
            raise RuntimeError("Live baseline and checkpoint contexts differ. / 实时基准与 checkpoint 环境不同。")
        return self._final_receipt


def _build_demo_controller() -> tuple[
    SessionPhaseController,
    OnlineIRFLongRunBaselineScorer,
]:
    """Build a quick debug session; formal paper mode uses 100,000 rows.

    建立快速调试 session；正式论文模式使用 100,000 行。
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
        experiment_cell_key="step35c_demo_only",
        session_index=0,
    )
    scorer = OnlineIRFLongRunBaselineScorer(session)
    controller = SessionPhaseController.create_for_fresh_session(
        session,
        convergence_periods_required=5,
        measurement_periods_required=200,
        measurement_sink=scorer.observe,
    )
    return controller, scorer


def main() -> None:
    """Run one visible, deliberately short baseline demonstration.

    运行一个可见且刻意缩短的基准演示。
    """

    controller, scorer = _build_demo_controller()
    while controller.phase is SessionPhase.TRAINING:
        if controller.training_periods_completed >= 5:
            raise TimeoutError("Debug convergence was not reached. / 调试收敛尚未达到。")
        controller.run_next_period()
    scorer.capture_and_bind_convergence_checkpoint(controller)
    controller.run_until_complete()
    receipt = scorer.finalize(controller)
    print("Step 35C: long-run IRF baseline / 第 35C 步：长期 IRF 基准")
    print(f"Debug measurement rows / 调试测量行数: {receipt.measurement_periods_scored}")
    print(f"E[p_tilde] / 长期方向调整价格均值: {receipt.mean_oriented_price:.9f}")
    print(
        "E[x_tilde_i] / 各 agent 长期方向调整订单均值: "
        f"{receipt.mean_oriented_order_by_agent}"
    )
    print(f"E[profit_i] / 各 agent 长期利润均值: {receipt.mean_profit_by_agent}")
    print(f"E[lambda_hat] / 长期价格冲击均值: {receipt.mean_price_impact_lambda:.12f}")
    print("Control branch used as denominator / 对照分支作为分母: False")
    print(
        "Paper 100,000 measurement length verified / 论文十万期测量长度已验证: "
        f"{receipt.paper_post_convergence_measurement_length_100000_verified} "
        "(debug run / 调试运行)"
    )
    print(
        "Checkpoint + sink provenance verified / checkpoint 与 sink 来源已验证: "
        f"{receipt.same_session_scored_sample_provenance_verified}"
    )
    print(
        "Ready for shock calibration / 可用于冲击校准: "
        f"{receipt.ready_for_shock_calibration}"
    )
    print("IRF paths run / 已运行 IRF 路径: 0")
    print("Classification ready / 可以分类: False")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
