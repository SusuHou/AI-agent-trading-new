"""Step 33: compute the paper's mispricing metric.

第 33 步：计算论文的错误定价指标。

Run the hand-checkable demonstration / 运行可以手算的演示:
    py -3 -X utf8 steps/step_33_mispricing.py

Run the separate automated tests / 运行独立自动测试:
    py -3 -X utf8 -m unittest discover -s tests \
        -p "test_step33_mispricing.py" -v

Paper rule, Online Appendix equation IA.4.7 / 论文规则，在线附录 IA.4.7:

    E_t^C = |E[p_t | v_t] - v_t|
          = (1 - lambda_hat_t * I * chi_hat^C) * |v_t - v_bar|

The first line is an absolute-error definition, but the printed expansion does
NOT put an absolute value around ``1-lambda_hat_t*I*chi_hat^C``. Theoretical
equilibria make that loading non-negative; a finite learned simulation is not
automatically guaranteed to do so. We therefore calculate both expressions.
They agree whenever the loading is non-negative, and also in the special case
``v_t=v_bar`` where both terms are zero. If a negative loading meets a positive
value deviation, the receipt preserves both results but leaves the primary
metric undefined and requests an explicit research decision. / 第一行把错误定价
定义为绝对误差，但原文展开式没有在 ``1-lambda_hat_t*I*chi_hat^C`` 外加
绝对值。理论均衡会使这个系数非负，但有限期学习模拟不一定自动满足。因此代码同时
计算两个版本；系数非负时两者一致，``v_t=v_bar`` 时两项也都为零。若负系数遇到正的
价值偏差，receipt 会保留两种结果，但不擅自选择其中一个，而是要求研究者明确决定。

Why two compact arrays are needed / 为什么需要两个紧凑数组:
    ``lambda_hat_t`` is known in every measurement period, but ``chi_hat^C`` is
    the full-window Step-30 regression slope and is known only after measurement
    ends. Step 33 therefore stores only two float64 numbers per period:
    ``lambda_hat_t`` and ``|v_t-v_bar|``. It then replays those pairs once after
    Step 30 finalizes. At T=100,000 this is about 1.6 MB, not 100,000 complete
    market observations. / 每期都知道 lambda_hat_t，但 chi_hat^C 是 Step 30
    使用完整测量窗口才得到的回归斜率，结束前并不知道。因此每期只保存两个 float64：
    lambda_hat_t 与 |v_t-v_bar|，Step 30 完成后只重放一次。T=100,000 时约
    1.6 MB，并不是保存十万条完整市场记录。

Timing / 时序:
    The scorer reads the period-specific lambda already frozen in the Step-28
    observation. That is the prior-history estimate which actually priced the
    period. It never queries the live rolling market maker after the completed
    row has entered history. / scorer 读取 Step-28 observation 中冻结的逐期
    lambda；它是当期真正用于定价、由旧历史估计出的值。代码绝不在本期记录进入历史
    后重新查询 live market maker，因而没有一期前视偏差。
"""

from array import array
from dataclasses import dataclass
from math import fsum, isclose, isfinite
from numbers import Integral, Real
from pathlib import Path
from collections.abc import Sequence
import sys


if sys.version_info < (3, 13):
    raise RuntimeError(
        "Step 33 requires Python 3.13 or newer for math.fma. "
        "/ Step 33 要求 Python 3.13 或更高版本，以使用 math.fma。"
    )

from math import fma


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from step_26_reproducible_random_streams import (
    FrozenPolicyPeriodObservation,
    RandomizedMarketSession,
    SessionSeedManifest,
)
from step_28_session_phases import (
    SessionPhase,
    SessionPhaseController,
    SessionPhaseReceipt,
)
from step_30_trading_intensity import (
    ESTIMATOR_VERSION,
    REGRESSION_SPECIFICATION,
    OnlineTradingIntensityScorer,
    TradingIntensityReceipt,
)
from step_32_market_liquidity import validate_recorded_price_impact


PAPER_MISPRICING_FORMULA = (
    "E_t_C = (1 - lambda_hat_t * I * chi_hat_C) * abs(v_t - v_bar)"
)
DEFINITION_3_4_ABSOLUTE_FORMULA = (
    "abs(1 - lambda_hat_t * I * chi_hat_C) * abs(v_t - v_bar)"
)
MISPRICING_VERSION = "appendix-ia.4.7-v1-deferred-chi"
PAPER_PRINTED_AGGREGATION = "E_C = sum_{t=T_c}^{T_c+T} E_t_C"
REPLICATION_AGGREGATION = "arithmetic_mean_over_exact_step28_measurement_rows"


class UndefinedMispricingError(ArithmeticError):
    """No period-level mispricing is available to summarize. / 没有可汇总的逐期错误定价。"""


def _finite_real(number: float, label: str) -> float:
    """Return one finite, non-Boolean real number. / 返回一个有限、非布尔实数。"""

    if isinstance(number, bool) or not isinstance(number, Real):
        raise TypeError(f"{label} must be a real number. / {label} 必须是实数。")
    converted = float(number)
    if not isfinite(converted):
        raise ValueError(f"{label} must be finite. / {label} 必须是有限数。")
    return converted


def _positive_integer(number: int, label: str) -> int:
    """Return one positive, non-Boolean integer. / 返回一个正的、非布尔整数。"""

    if (
        isinstance(number, bool)
        or not isinstance(number, Integral)
        or int(number) < 1
    ):
        raise ValueError(f"{label} must be a positive integer. / {label} 必须是正整数。")
    return int(number)


def _aggregate_informed_slope(
    number_of_agents: int,
    average_trading_intensity: float,
) -> tuple[int, float, float]:
    """Validate ``I`` and ``chi``, then return ``I*chi``. / 验证 I 与 chi，再返回 I*chi。"""

    agent_count = _positive_integer(number_of_agents, "number_of_agents I")
    intensity = _finite_real(
        average_trading_intensity,
        "average_trading_intensity chi_hat_C",
    )
    aggregate_slope = agent_count * intensity
    if not isfinite(aggregate_slope):
        raise OverflowError("I * chi_hat_C overflowed. / I 与 chi_hat_C 的乘积溢出。")
    return agent_count, intensity, aggregate_slope


@dataclass(frozen=True)
class PeriodMispricingCalculation:
    """One immutable, hand-auditable IA.4.7 calculation.

    一份不可修改、可以手算核对的 IA.4.7 逐期结果。
    """

    number_of_agents: int
    average_trading_intensity: float
    aggregate_informed_slope: float
    price_impact_lambda_hat: float
    fundamental_value_v: float
    value_mean: float
    loading_factor: float
    absolute_loading_factor: float
    absolute_value_deviation: float
    paper_signed_expression: float
    definition_3_4_absolute_error: float
    paper_nonnegative_domain_satisfied: bool
    paper_formula_matches_definition: bool


def calculate_period_mispricing(
    fundamental_value_v: float,
    value_mean: float,
    price_impact_lambda_hat: float,
    number_of_agents: int,
    average_trading_intensity: float,
) -> PeriodMispricingCalculation:
    """Apply both readings of IA.4.7 to one period.

    对一个时期同时计算 IA.4.7 的原文展开式与绝对误差定义。

    Example / 例子:
        I=2, chi=0.25, lambda=0.5, v=3, v_bar=1
        loading = 1 - 0.5*2*0.25 = 0.75
        |v-v_bar| = 2
        E_t = 0.75*2 = 1.5
    """

    agent_count, intensity, aggregate_slope = _aggregate_informed_slope(
        number_of_agents,
        average_trading_intensity,
    )
    value = _finite_real(fundamental_value_v, "fundamental_value_v")
    mean = _finite_real(value_mean, "value_mean v_bar")
    lambda_hat = _finite_real(
        price_impact_lambda_hat,
        "price_impact_lambda_hat",
    )

    value_gap = value - mean
    if not isfinite(value_gap):
        raise OverflowError("v_t - v_bar overflowed. / v_t 与 v_bar 的差发生溢出。")
    deviation = abs(value_gap)

    # fma evaluates 1-lambda*(I*chi) with one final rounding. / fma 只在最后
    # 舍入一次地计算 1-lambda*(I*chi)。
    loading = fma(-lambda_hat, aggregate_slope, 1.0)
    if not isfinite(loading):
        raise OverflowError("The IA.4.7 loading overflowed. / IA.4.7 系数发生溢出。")

    paper_value = loading * deviation
    definition_value = abs(loading) * deviation
    if not isfinite(paper_value) or not isfinite(definition_value):
        raise OverflowError("The period mispricing calculation overflowed. / 逐期错误定价计算溢出。")

    # Convert -0.0 to ordinary 0.0 for clearer receipts. / 把 -0.0 转成更清楚的 0.0。
    if paper_value == 0.0:
        paper_value = 0.0
    if definition_value == 0.0:
        definition_value = 0.0

    return PeriodMispricingCalculation(
        number_of_agents=agent_count,
        average_trading_intensity=intensity,
        aggregate_informed_slope=aggregate_slope,
        price_impact_lambda_hat=lambda_hat,
        fundamental_value_v=value,
        value_mean=mean,
        loading_factor=loading,
        absolute_loading_factor=abs(loading),
        absolute_value_deviation=deviation,
        paper_signed_expression=paper_value,
        definition_3_4_absolute_error=definition_value,
        paper_nonnegative_domain_satisfied=loading >= 0.0,
        paper_formula_matches_definition=(loading >= 0.0 or deviation == 0.0),
    )


def _compensated_add(
    running_sum: float,
    compensation: float,
    value: float,
) -> tuple[float, float]:
    """Prepare one Neumaier compensated-sum update. / 准备一次 Neumaier 补偿求和更新。"""

    new_sum = running_sum + value
    if not isfinite(new_sum):
        raise OverflowError("A mispricing sum overflowed. / 错误定价求和溢出。")
    if abs(running_sum) >= abs(value):
        correction = (running_sum - new_sum) + value
    else:
        correction = (value - new_sum) + running_sum
    new_compensation = compensation + correction
    if not isfinite(new_compensation):
        raise OverflowError("Mispricing compensation overflowed. / 错误定价补偿项溢出。")
    return new_sum, new_compensation


@dataclass(frozen=True)
class MispricingPairSummary:
    """Summary created by replaying compact lambda/deviation pairs.

    重放紧凑的 lambda/deviation 数对后得到的汇总。
    """

    observations: int
    number_of_agents: int
    average_trading_intensity: float
    aggregate_informed_slope: float
    paper_signed_expression_sum: float
    paper_signed_expression_average: float
    definition_3_4_absolute_sum: float
    definition_3_4_absolute_average: float
    reported_mispricing_sum: float | None
    reported_average_mispricing: float | None
    paper_nonnegative_domain_satisfied: bool
    paper_formula_matches_definition_on_observed_path: bool
    negative_loading_period_count: int
    formula_disagreement_period_count: int
    zero_loading_period_count: int
    minimum_loading_factor: float
    maximum_loading_factor: float
    first_negative_pair_index: int | None
    first_formula_disagreement_pair_index: int | None
    minimum_loading_pair_index: int


def summarize_mispricing_pairs(
    price_impact_lambdas: Sequence[float],
    absolute_value_deviations: Sequence[float],
    number_of_agents: int,
    average_trading_intensity: float,
) -> MispricingPairSummary:
    """Replay compact inputs after the final Step-30 ``chi_hat_C`` is known.

    在最终 Step-30 ``chi_hat_C`` 已知后，重放紧凑输入。

    This public pure function makes the deferred calculation independently
    testable without constructing a market session. / 这个公开纯函数让我们无需
    建立完整市场 session，也能独立测试延迟计算。
    """

    agent_count, intensity, aggregate_slope = _aggregate_informed_slope(
        number_of_agents,
        average_trading_intensity,
    )
    try:
        lambda_count = len(price_impact_lambdas)
        deviation_count = len(absolute_value_deviations)
    except TypeError as error:
        raise TypeError("Both compact inputs must be sequences. / 两个紧凑输入都必须是序列。") from error
    if lambda_count != deviation_count:
        raise ValueError("The two compact arrays must have equal length. / 两个紧凑数组长度必须相等。")
    if lambda_count < 1:
        raise UndefinedMispricingError("At least one measurement row is required. / 至少需要一条测量记录。")

    paper_sum = 0.0
    paper_compensation = 0.0
    definition_sum = 0.0
    definition_compensation = 0.0
    negative_count = 0
    disagreement_count = 0
    zero_count = 0
    minimum_loading = float("inf")
    maximum_loading = float("-inf")
    first_negative_index: int | None = None
    first_disagreement_index: int | None = None
    minimum_loading_index = 0

    for index in range(lambda_count):
        lambda_hat = _finite_real(
            price_impact_lambdas[index],
            f"price_impact_lambdas[{index}]",
        )
        deviation = _finite_real(
            absolute_value_deviations[index],
            f"absolute_value_deviations[{index}]",
        )
        if deviation < 0.0:
            raise ValueError("Absolute value deviations cannot be negative. / 价值绝对偏差不能为负。")

        loading = fma(-lambda_hat, aggregate_slope, 1.0)
        if not isfinite(loading):
            raise OverflowError("An IA.4.7 loading overflowed. / 某一期 IA.4.7 系数溢出。")
        paper_value = loading * deviation
        definition_value = abs(loading) * deviation
        if not isfinite(paper_value) or not isfinite(definition_value):
            raise OverflowError("A period mispricing value overflowed. / 某一期错误定价值溢出。")

        paper_sum, paper_compensation = _compensated_add(
            paper_sum,
            paper_compensation,
            paper_value,
        )
        definition_sum, definition_compensation = _compensated_add(
            definition_sum,
            definition_compensation,
            definition_value,
        )

        if loading < 0.0:
            negative_count += 1
            if first_negative_index is None:
                first_negative_index = index
            # A negative loading changes the absolute-error result only when
            # the value deviation is positive. At v_t=v_bar both formulas are
            # exactly zero. / 只有价值偏差为正时，负系数才会让两种公式不同；
            # v_t=v_bar 时两式都精确为零。
            if deviation > 0.0:
                disagreement_count += 1
                if first_disagreement_index is None:
                    first_disagreement_index = index
        elif loading == 0.0:
            zero_count += 1
        if loading < minimum_loading:
            minimum_loading = loading
            minimum_loading_index = index
        maximum_loading = max(maximum_loading, loading)

    corrected_paper_sum = paper_sum + paper_compensation
    corrected_definition_sum = definition_sum + definition_compensation
    if not isfinite(corrected_paper_sum) or not isfinite(corrected_definition_sum):
        raise OverflowError("A corrected mispricing sum overflowed. / 修正后的错误定价总和溢出。")
    paper_average = corrected_paper_sum / lambda_count
    definition_average = corrected_definition_sum / lambda_count
    if not isfinite(paper_average) or not isfinite(definition_average):
        raise OverflowError("An average mispricing value overflowed. / 平均错误定价值溢出。")

    domain_satisfied = negative_count == 0
    formulas_match = disagreement_count == 0
    if formulas_match:
        # Non-negative loadings agree automatically; a negative loading also
        # agrees in a zero-deviation period because both terms are zero.
        # / 非负系数自动一致；零偏差时期即使系数为负，两项也同为零。
        if corrected_paper_sum != corrected_definition_sum:
            raise ArithmeticError("Equivalent IA.4.7 sums unexpectedly disagree. / 等价 IA.4.7 求和意外不一致。")
        reported_sum: float | None = corrected_definition_sum
        reported_average: float | None = definition_average
    else:
        reported_sum = None
        reported_average = None

    return MispricingPairSummary(
        observations=lambda_count,
        number_of_agents=agent_count,
        average_trading_intensity=intensity,
        aggregate_informed_slope=aggregate_slope,
        paper_signed_expression_sum=corrected_paper_sum,
        paper_signed_expression_average=paper_average,
        definition_3_4_absolute_sum=corrected_definition_sum,
        definition_3_4_absolute_average=definition_average,
        reported_mispricing_sum=reported_sum,
        reported_average_mispricing=reported_average,
        paper_nonnegative_domain_satisfied=domain_satisfied,
        paper_formula_matches_definition_on_observed_path=formulas_match,
        negative_loading_period_count=negative_count,
        formula_disagreement_period_count=disagreement_count,
        zero_loading_period_count=zero_count,
        minimum_loading_factor=minimum_loading,
        maximum_loading_factor=maximum_loading,
        first_negative_pair_index=first_negative_index,
        first_formula_disagreement_pair_index=first_disagreement_index,
        minimum_loading_pair_index=minimum_loading_index,
    )


@dataclass(frozen=True)
class MispricingReceipt:
    """Auditable Step-33 result for one completed seeded session.

    一个完整带种子 session 的可审计 Step-33 结果。
    """

    measurement_periods_scored: int
    first_measurement_index: int
    last_measurement_index: int
    first_global_period_index: int
    last_global_period_index: int
    number_of_agents: int
    slope_by_agent: tuple[float, ...]
    average_trading_intensity: float
    aggregate_informed_slope: float
    value_mean: float
    paper_signed_expression_sum: float
    paper_signed_expression_average: float
    definition_3_4_absolute_sum: float
    definition_3_4_absolute_average: float
    reported_mispricing_sum: float | None
    reported_average_mispricing: float | None
    paper_nonnegative_domain_satisfied: bool
    paper_formula_matches_definition_on_observed_path: bool
    requires_explicit_research_decision: bool
    negative_loading_period_count: int
    formula_disagreement_period_count: int
    zero_loading_period_count: int
    minimum_loading_factor: float
    maximum_loading_factor: float
    first_negative_global_period_index: int | None
    first_formula_disagreement_global_period_index: int | None
    minimum_loading_global_period_index: int
    parameter_snapshot: PaperParameters
    value_grid_snapshot: tuple[float, ...]
    session_seed_manifest: SessionSeedManifest
    paper_formula: str
    definition_3_4_absolute_formula: str
    calculation_version: str
    source_step30_estimator_version: str
    source_step30_regression_specification: str
    paper_printed_aggregation: str
    replication_aggregation: str
    paper_printed_first_factor_has_absolute_value: bool
    paper_prose_calls_aggregation_average: bool
    paper_printed_one_over_t: bool
    uses_full_measurement_window_chi: bool
    uses_period_specific_prior_history_lambda: bool
    uses_conditional_expected_mispricing_not_realized_price_error: bool
    uses_fused_multiply_add: bool
    compact_float64_values_buffered_before_finalize: int
    compact_buffer_bytes_before_finalize: int
    compact_storage_is_linear_in_measurement_periods: bool
    full_observation_rows_stored: bool
    compact_buffers_cleared_after_finalize: bool


class DeferredOnlineMispricingScorer:
    """Session-bound Step-28 sink that waits for Step 30's final slope.

    与指定 session 绑定、等待 Step 30 最终斜率的 Step-28 sink。
    """

    def __init__(self, session: RandomizedMarketSession) -> None:
        if not isinstance(session, RandomizedMarketSession):
            raise TypeError("session has the wrong type. / session 类型错误。")
        if session.period_number != 0 or session.execution_mode != "training":
            raise RuntimeError("Attach the scorer to a fresh training session. / 请把 scorer 连接到尚未运行的训练 session。")
        if session.parameters.num_speculators < 2:
            raise ValueError("The paper's collusion model requires I >= 2. / 论文的合谋模型要求 I >= 2。")

        self._session = session
        self._parameter_snapshot = session.parameters
        self._value_grid_snapshot = tuple(float(value) for value in session.value_grid)
        self._seed_manifest_snapshot = session.streams.manifest
        # array('d') is compact C double storage: exactly 8 bytes on supported
        # CPython builds. / array('d') 是紧凑的 C double 存储；这里每个值 8 字节。
        self._price_impact_lambdas = array("d")
        self._absolute_value_deviations = array("d")
        if self._price_impact_lambdas.itemsize != 8:
            raise RuntimeError("This platform does not provide 8-byte array('d'). / 本平台 array('d') 不是 8 字节。")
        self.rows_scored = 0
        self.first_global_period_index: int | None = None
        self.last_global_period_index: int | None = None
        self._final_receipt: MispricingReceipt | None = None
        self._final_intensity_scorer: OnlineTradingIntensityScorer | None = None

    @property
    def buffered_rows(self) -> int:
        """Number of compact period pairs currently retained. / 当前保留的紧凑时期数对数量。"""

        if len(self._price_impact_lambdas) != len(self._absolute_value_deviations):
            raise RuntimeError("The two compact buffers disagree. / 两个紧凑缓冲区长度不一致。")
        return len(self._price_impact_lambdas)

    @property
    def buffered_bytes(self) -> int:
        """Raw numeric payload size, excluding tiny array headers. / 原始数值载荷大小，不含很小的数组头。"""

        return (
            len(self._price_impact_lambdas) * self._price_impact_lambdas.itemsize
            + len(self._absolute_value_deviations)
            * self._absolute_value_deviations.itemsize
        )

    def observe(
        self,
        measurement_index: int,
        observation: FrozenPolicyPeriodObservation,
    ) -> None:
        """Validate and save only ``(lambda_hat_t, |v_t-v_bar|)``.

        验证后只保存 ``(lambda_hat_t, |v_t-v_bar|)``。
        """

        if self._final_receipt is not None:
            raise RuntimeError("This scorer is already finalized. / 这个 scorer 已经完成。")
        if (
            isinstance(measurement_index, bool)
            or not isinstance(measurement_index, Integral)
        ):
            raise TypeError("measurement_index must be an integer. / measurement_index 必须是整数。")
        index = int(measurement_index)
        if index != self.rows_scored:
            raise ValueError(
                f"Expected measurement index {self.rows_scored}, received {index}. "
                f"/ 预期测量编号 {self.rows_scored}，却收到 {index}。"
            )
        if not isinstance(observation, FrozenPolicyPeriodObservation):
            raise TypeError("observation has the wrong type. / observation 类型错误。")
        if isinstance(observation.period_number, bool) or not isinstance(
            observation.period_number,
            Integral,
        ):
            raise TypeError("period_number must be an integer. / period_number 必须是整数。")
        period_number = int(observation.period_number)
        if period_number < 0:
            raise ValueError("period_number cannot be negative. / period_number 不能为负数。")
        if (
            self.last_global_period_index is not None
            and period_number != self.last_global_period_index + 1
        ):
            raise ValueError("Global measurement periods must be consecutive. / 全局测量期必须连续。")

        value_index = observation.current_value_index
        if (
            isinstance(value_index, bool)
            or not isinstance(value_index, Integral)
            or not 0 <= int(value_index) < len(self._value_grid_snapshot)
        ):
            raise ValueError("current_value_index is invalid. / current_value_index 无效。")
        expected_value = self._value_grid_snapshot[int(value_index)]
        observed_value = _finite_real(
            observation.fundamental_value_v,
            "fundamental_value_v",
        )
        if not isclose(observed_value, expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("v_t does not match current_value_index. / v_t 与 current_value_index 不匹配。")

        recorded_lambda = validate_recorded_price_impact(
            observation,
            self._parameter_snapshot.pricing_error_weight,
        )
        value_gap = observed_value - self._parameter_snapshot.value_mean
        if not isfinite(value_gap):
            raise OverflowError("v_t - v_bar overflowed. / v_t 与 v_bar 的差发生溢出。")
        deviation = abs(value_gap)

        # Commit the two values atomically: if the second append unexpectedly
        # fails, remove the first. / 两个值原子化提交；第二次 append 若意外失败，
        # 就撤回第一次 append。
        self._price_impact_lambdas.append(recorded_lambda)
        try:
            self._absolute_value_deviations.append(deviation)
        except Exception:
            self._price_impact_lambdas.pop()
            raise

        if self.rows_scored == 0:
            self.first_global_period_index = period_number
        self.last_global_period_index = period_number
        self.rows_scored += 1

    def _validate_step30_receipt(
        self,
        receipt: TradingIntensityReceipt,
        phase_receipt: SessionPhaseReceipt,
    ) -> None:
        """Reject mixed, altered, or incomplete Step-30 provenance.

        拒绝混用、被修改或不完整的 Step-30 来源。
        """

        if not isinstance(receipt, TradingIntensityReceipt):
            raise TypeError("Step-30 receipt has the wrong type. / Step-30 receipt 类型错误。")
        if (
            receipt.measurement_periods_scored != self.rows_scored
            or receipt.measurement_periods_scored
            != phase_receipt.measurement_periods_completed
            or receipt.first_measurement_index != 0
            or receipt.last_measurement_index != self.rows_scored - 1
            or receipt.first_global_period_index != self.first_global_period_index
            or receipt.last_global_period_index != self.last_global_period_index
        ):
            raise RuntimeError("Step 28, Step 30, and Step 33 measurement bounds disagree. / Step 28、30、33 的测量边界不一致。")
        if (
            receipt.parameter_snapshot != self._parameter_snapshot
            or receipt.value_grid_snapshot != self._value_grid_snapshot
            or receipt.session_seed_manifest != self._seed_manifest_snapshot
        ):
            raise RuntimeError("Step 30 and Step 33 provenance disagree. / Step 30 与 Step 33 的数据来源不一致。")
        if receipt.number_of_agents != self._parameter_snapshot.num_speculators:
            raise RuntimeError("Step-30 agent count disagrees with the session. / Step-30 agent 数量与 session 不一致。")
        if (
            receipt.estimator_version != ESTIMATOR_VERSION
            or receipt.regression_specification != REGRESSION_SPECIFICATION
            or not receipt.unrestricted_intercept_estimated
            or not receipt.actual_raw_orders_used
        ):
            raise RuntimeError("Step-30 estimator provenance is incompatible. / Step-30 估计器来源不兼容。")
        slopes = receipt.slope_by_agent
        if len(slopes) != receipt.number_of_agents or not all(
            isfinite(slope) for slope in slopes
        ):
            raise RuntimeError("Step-30 slopes are incomplete or invalid. / Step-30 斜率不完整或无效。")
        recomputed_average = fsum(slopes) / receipt.number_of_agents
        if not isclose(
            receipt.average_trading_intensity,
            recomputed_average,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError("Step-30 average slope is internally inconsistent. / Step-30 平均斜率内部不一致。")

    def finalize(
        self,
        trading_intensity_scorer: OnlineTradingIntensityScorer,
        controller: SessionPhaseController,
    ) -> MispricingReceipt:
        """Replay compact pairs after Step 30 supplies final ``chi_hat_C``.

        Step 30 给出最终 ``chi_hat_C`` 后，重放紧凑数对。
        """

        if not isinstance(trading_intensity_scorer, OnlineTradingIntensityScorer):
            raise TypeError("trading_intensity_scorer has the wrong type. / Step-30 scorer 类型错误。")
        if not isinstance(controller, SessionPhaseController):
            raise TypeError("controller has the wrong type. / controller 类型错误。")
        if controller.session is not self._session:
            raise RuntimeError("This controller belongs to another session. / 这个 controller 属于另一个 session。")
        if getattr(trading_intensity_scorer, "_session", None) is not self._session:
            raise RuntimeError("Step 30 belongs to another session. / Step 30 属于另一个 session。")
        if controller.phase is not SessionPhase.COMPLETE:
            raise RuntimeError("Step 28 has not completed successfully. / Step 28 尚未成功完成。")
        if self._final_receipt is not None:
            if trading_intensity_scorer is not self._final_intensity_scorer:
                raise RuntimeError("This receipt was finalized from another Step-30 scorer. / 本 receipt 由另一个 Step-30 scorer 完成。")
            return self._final_receipt

        phase_receipt = controller.final_receipt
        if not isinstance(phase_receipt, SessionPhaseReceipt):
            raise RuntimeError("Step-28 completion receipt is missing. / Step-28 完成 receipt 丢失。")
        if self._session.execution_mode != "complete":
            raise RuntimeError("The market session is not complete. / 市场 session 尚未完成。")
        if (
            phase_receipt.measurement_periods_required
            != phase_receipt.measurement_periods_completed
            or self.rows_scored != phase_receipt.measurement_periods_completed
            or self.buffered_rows != self.rows_scored
        ):
            raise RuntimeError("Step-28 and Step-33 row counts disagree. / Step-28 与 Step-33 行数不一致。")
        if (
            self.first_global_period_index
            != phase_receipt.measurement_first_period_index
            or self.last_global_period_index
            != phase_receipt.measurement_last_period_index
        ):
            raise RuntimeError("Step-28 and Step-33 period bounds disagree. / Step-28 与 Step-33 时期边界不一致。")
        if self._session.parameters != self._parameter_snapshot:
            raise RuntimeError("Session parameters changed after Step 33 attached. / Step 33 连接后 session 参数发生改变。")
        live_grid = tuple(float(value) for value in self._session.value_grid)
        if live_grid != self._value_grid_snapshot:
            raise RuntimeError("The value grid changed after Step 33 attached. / Step 33 连接后价值网格发生改变。")
        if self._session.streams.manifest != self._seed_manifest_snapshot:
            raise RuntimeError("The seed manifest changed after Step 33 attached. / Step 33 连接后随机种子记录发生改变。")

        intensity_receipt = trading_intensity_scorer.finalize(controller)
        self._validate_step30_receipt(intensity_receipt, phase_receipt)
        summary = summarize_mispricing_pairs(
            self._price_impact_lambdas,
            self._absolute_value_deviations,
            self._parameter_snapshot.num_speculators,
            intensity_receipt.average_trading_intensity,
        )

        first_negative_global = (
            None
            if summary.first_negative_pair_index is None
            else self.first_global_period_index + summary.first_negative_pair_index
        )
        first_disagreement_global = (
            None
            if summary.first_formula_disagreement_pair_index is None
            else self.first_global_period_index
            + summary.first_formula_disagreement_pair_index
        )
        minimum_loading_global = (
            self.first_global_period_index + summary.minimum_loading_pair_index
        )
        bytes_before_finalize = self.buffered_bytes
        values_before_finalize = 2 * self.buffered_rows

        receipt = MispricingReceipt(
            measurement_periods_scored=self.rows_scored,
            first_measurement_index=0,
            last_measurement_index=self.rows_scored - 1,
            first_global_period_index=self.first_global_period_index,
            last_global_period_index=self.last_global_period_index,
            number_of_agents=summary.number_of_agents,
            slope_by_agent=intensity_receipt.slope_by_agent,
            average_trading_intensity=summary.average_trading_intensity,
            aggregate_informed_slope=summary.aggregate_informed_slope,
            value_mean=float(self._parameter_snapshot.value_mean),
            paper_signed_expression_sum=summary.paper_signed_expression_sum,
            paper_signed_expression_average=(
                summary.paper_signed_expression_average
            ),
            definition_3_4_absolute_sum=summary.definition_3_4_absolute_sum,
            definition_3_4_absolute_average=(
                summary.definition_3_4_absolute_average
            ),
            reported_mispricing_sum=summary.reported_mispricing_sum,
            reported_average_mispricing=summary.reported_average_mispricing,
            paper_nonnegative_domain_satisfied=(
                summary.paper_nonnegative_domain_satisfied
            ),
            paper_formula_matches_definition_on_observed_path=(
                summary.paper_formula_matches_definition_on_observed_path
            ),
            requires_explicit_research_decision=(
                not summary.paper_formula_matches_definition_on_observed_path
            ),
            negative_loading_period_count=(
                summary.negative_loading_period_count
            ),
            formula_disagreement_period_count=(
                summary.formula_disagreement_period_count
            ),
            zero_loading_period_count=summary.zero_loading_period_count,
            minimum_loading_factor=summary.minimum_loading_factor,
            maximum_loading_factor=summary.maximum_loading_factor,
            first_negative_global_period_index=first_negative_global,
            first_formula_disagreement_global_period_index=(
                first_disagreement_global
            ),
            minimum_loading_global_period_index=minimum_loading_global,
            parameter_snapshot=self._parameter_snapshot,
            value_grid_snapshot=self._value_grid_snapshot,
            session_seed_manifest=self._seed_manifest_snapshot,
            paper_formula=PAPER_MISPRICING_FORMULA,
            definition_3_4_absolute_formula=DEFINITION_3_4_ABSOLUTE_FORMULA,
            calculation_version=MISPRICING_VERSION,
            source_step30_estimator_version=ESTIMATOR_VERSION,
            source_step30_regression_specification=REGRESSION_SPECIFICATION,
            paper_printed_aggregation=PAPER_PRINTED_AGGREGATION,
            replication_aggregation=REPLICATION_AGGREGATION,
            paper_printed_first_factor_has_absolute_value=False,
            paper_prose_calls_aggregation_average=True,
            paper_printed_one_over_t=False,
            uses_full_measurement_window_chi=True,
            uses_period_specific_prior_history_lambda=True,
            uses_conditional_expected_mispricing_not_realized_price_error=True,
            uses_fused_multiply_add=True,
            compact_float64_values_buffered_before_finalize=(
                values_before_finalize
            ),
            compact_buffer_bytes_before_finalize=bytes_before_finalize,
            compact_storage_is_linear_in_measurement_periods=True,
            full_observation_rows_stored=False,
            compact_buffers_cleared_after_finalize=True,
        )

        # The frozen receipt now contains the result, so release the O(T)
        # temporary arrays before returning. / 冻结 receipt 已保存结果，因此返回前
        # 释放 O(T) 临时数组。
        self._price_impact_lambdas = array("d")
        self._absolute_value_deviations = array("d")
        self._final_intensity_scorer = trading_intensity_scorer
        self._final_receipt = receipt
        return receipt


def main() -> None:
    """Run hand arithmetic, aggregation, and ambiguity checks.

    运行手算、汇总与原文歧义检查。
    """

    hand = calculate_period_mispricing(
        fundamental_value_v=3.0,
        value_mean=1.0,
        price_impact_lambda_hat=0.5,
        number_of_agents=2,
        average_trading_intensity=0.25,
    )
    assert hand.loading_factor == 0.75
    assert hand.absolute_value_deviation == 2.0
    assert hand.paper_signed_expression == 1.5
    assert hand.definition_3_4_absolute_error == 1.5

    summary = summarize_mispricing_pairs(
        price_impact_lambdas=(0.2, 0.4, -0.2),
        absolute_value_deviations=(0.0, 2.0, 1.0),
        number_of_agents=2,
        average_trading_intensity=0.5,
    )
    assert isclose(summary.paper_signed_expression_sum, 2.4)
    assert isclose(summary.reported_average_mispricing, 0.8)

    ambiguity = calculate_period_mispricing(
        fundamental_value_v=2.0,
        value_mean=1.0,
        price_impact_lambda_hat=0.75,
        number_of_agents=2,
        average_trading_intensity=1.0,
    )
    assert ambiguity.loading_factor == -0.5
    assert ambiguity.paper_signed_expression == -0.5
    assert ambiguity.definition_3_4_absolute_error == 0.5
    assert not ambiguity.paper_nonnegative_domain_satisfied

    print("Step 33: Mispricing / 第 33 步：错误定价")
    print("Hand example / 手算例子:")
    print(f"  Loading 1-lambda*I*chi / 系数: {hand.loading_factor:.2f}")
    print(f"  |v-v_bar| / 价值绝对偏差: {hand.absolute_value_deviation:.2f}")
    print(f"  E_t^C / 逐期错误定价: {hand.paper_signed_expression:.2f}")
    print("Three-row aggregation / 三期汇总:")
    print(f"  Literal sum / 原文字面求和: {summary.paper_signed_expression_sum:.2f}")
    print(f"  Arithmetic mean / 算术平均: {summary.reported_average_mispricing:.2f}")
    print("Paper ambiguity guard / 原文歧义保护:")
    print(f"  Printed expansion / 打印展开式: {ambiguity.paper_signed_expression:.2f}")
    print(f"  Absolute-error definition / 绝对误差定义: {ambiguity.definition_3_4_absolute_error:.2f}")
    print("  Negative loading is flagged; no silent repair. / 负系数会被标记，不会静默修补。")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
