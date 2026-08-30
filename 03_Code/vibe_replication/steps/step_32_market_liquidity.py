"""Step 32: compute the paper's market-liquidity metric.

第 32 步：计算论文的市场流动性指标。

Run the hand-checkable demonstration / 运行可以手算的演示:
    py -3 -X utf8 steps/step_32_market_liquidity.py

Run the separate automated tests / 运行独立自动测试:
    py -3 -X utf8 -m unittest discover -s tests \
        -p "test_step32_market_liquidity.py" -v

Paper rule, Online Appendix equation IA.4.6 / 论文规则，在线附录 IA.4.6:

    L_t^C = 1 / abs(1 - xi * lambda_hat_t)

This is a period-level market measure. ``xi`` is the configured structural
slope of information-insensitive investor demand. ``lambda_hat_t`` is the
adaptive price impact that the market maker actually used in period t. / 这是
逐期的市场层面指标。xi 是信息不敏感投资者需求的结构参数；lambda_hat_t 是做市商
在第 t 期实际用于定价的自适应价格冲击。

Important timing / 重要时序:
    Step 28 emits the period-t lambda that was estimated from prior history
    D_t. By the time a measurement sink receives the row, the live rolling
    history already contains the completed period. Therefore this scorer uses
    ``observation.price_impact_lambda_hat`` and never asks the live market maker
    to estimate again. / Step 28 输出的是根据旧历史 D_t 得到、并在本期真正使用的
    lambda。sink 收到记录时，本期已经被加入滚动历史，所以这里必须读取 observation
    中保存的 lambda，不能再次查询 live market maker，否则会产生一期前视偏差。

Appendix aggregation ambiguity / 附录汇总歧义:
    The prose says "average market liquidity," but the displayed expression is
    a sum and omits 1/T. It also uses inclusive-looking bounds that would contain
    T+1 rows. We report BOTH the literal sum and the arithmetic mean over the
    exact T Step-28 measurement rows; the mean is our reported L^C. / 原文文字说
    “平均流动性”，但展示式只写求和、漏了 1/T，而且上下界按字面会包含 T+1 条。
    因此结果同时保存字面求和与恰好 T 条 Step-28 记录的算术平均；平均值作为 L^C。

Numerical boundary / 数值边界:
    The paper gives no software rule for xi*lambda_hat_t=1. We do not add an
    epsilon and do not clip a large valid result. Fused multiply-add computes
    ``1-xi*lambda`` with one rounding. An exact zero is recorded as infinite
    liquidity; every nonzero representable gap is retained. / 论文没有规定
    xi*lambda_hat_t=1 时的软件处理。我们不偷偷加 epsilon，也不截断有效大数；
    使用融合乘加只舍入一次。精确零记录为无穷流动性，任何可表示的非零差都保留。
"""

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Integral, Real
from pathlib import Path
import sys


if sys.version_info < (3, 13):
    raise RuntimeError(
        "Step 32 requires Python 3.13 or newer for math.fma, which prevents "
        "false singularities in IA.4.6. / Step 32 要求 Python 3.13 或更高版本，"
        "以使用 math.fma 防止 IA.4.6 出现虚假奇点。"
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


MARKET_LIQUIDITY_FORMULA = "L_t_C = 1 / abs(1 - xi * lambda_hat_t)"
MARKET_LIQUIDITY_VERSION = "appendix-ia.4.6-v1"
PAPER_PRINTED_AGGREGATION = "L_C = sum_{t=T_c}^{T_c+T} L_t_C"
REPLICATION_AGGREGATION = (
    "arithmetic_mean_over_exact_step28_measurement_rows"
)


class UndefinedMarketLiquidityError(ArithmeticError):
    """No period-level liquidity is available to summarize. / 没有可汇总的逐期流动性。"""


def _finite_real(number: float, label: str) -> float:
    """Return one finite non-Boolean real number. / 返回一个有限、非布尔实数。"""

    if isinstance(number, bool) or not isinstance(number, Real):
        raise TypeError(f"{label} must be a real number. / {label} 必须是实数。")
    converted = float(number)
    if not isfinite(converted):
        raise ValueError(f"{label} must be finite. / {label} 必须是有限数。")
    return converted


def validate_recorded_price_impact(
    observation: FrozenPolicyPeriodObservation,
    pricing_error_weight: float,
) -> float:
    """Verify that a Step-28 row's lambda follows paper equation (4.2).

    验证 Step-28 记录中的 lambda 是否遵守论文方程 (4.2)。

    Step 33 reuses this public integrity check. It reads only coefficients
    frozen in the period observation and never queries the already-updated live
    rolling market maker. / Step 33 会复用这一公开检查。它只读取逐期 observation
    中冻结的系数，绝不查询已经加入本期记录的 live rolling market maker。
    """

    if not isinstance(observation, FrozenPolicyPeriodObservation):
        raise TypeError("observation has the wrong type. / observation 类型错误。")
    xi_1_hat = _finite_real(observation.xi_1_hat, "xi_1_hat")
    gamma_1_hat = _finite_real(observation.gamma_1_hat, "gamma_1_hat")
    recorded_lambda = _finite_real(
        observation.price_impact_lambda_hat,
        "price_impact_lambda_hat",
    )
    theta = _finite_real(
        pricing_error_weight,
        "pricing_error_weight theta",
    )
    if theta <= 0.0:
        raise ValueError("theta must be strictly positive. / theta 必须严格为正。")
    try:
        numerator = theta * gamma_1_hat + xi_1_hat
        # Mirror Step 24's operation order exactly. / 与 Step 24 的运算顺序完全一致。
        denominator = theta + xi_1_hat ** 2
    except OverflowError as error:
        raise ValueError(
            "Recorded OLS coefficients overflow equation (4.2). / 已记录 OLS 系数使方程 (4.2) 溢出。"
        ) from error
    if (
        not isfinite(numerator)
        or not isfinite(denominator)
        or denominator <= 0.0
    ):
        raise ValueError(
            "Recorded OLS coefficients cannot produce a finite lambda. / 已记录 OLS 系数无法生成有限 lambda。"
        )
    expected_lambda = numerator / denominator
    # The row is created in memory by this same Step-24 arithmetic, so exact
    # equality is expected. A tolerance is dangerous near xi*lambda=1. / 本记录
    # 由同一套 Step-24 运算在内存中生成，因此应精确相等；奇点附近不能使用容差。
    if recorded_lambda != expected_lambda:
        raise ValueError(
            "Recorded lambda_hat does not match equation (4.2). / 记录的 lambda_hat 与方程 (4.2) 不一致。"
        )
    return recorded_lambda


@dataclass(frozen=True)
class PeriodMarketLiquidityCalculation:
    """One immutable, hand-auditable IA.4.6 calculation.

    一份不可修改、可以手算核对的 IA.4.6 逐期结果。
    """

    investor_slope_xi: float
    price_impact_lambda_hat: float
    rounded_xi_times_lambda_hat: float
    signed_inventory_sensitivity: float
    absolute_inventory_sensitivity: float
    market_liquidity: float
    is_exactly_singular: bool
    reciprocal_overflowed: bool


def calculate_period_market_liquidity(
    investor_slope: float,
    price_impact_lambda_hat: float,
) -> PeriodMarketLiquidityCalculation:
    """Apply equation IA.4.6 to one period's two explicit inputs.

    对一个时期的两个明确输入应用 IA.4.6。

    We return the denominator as well as its inverse so the result can be
    audited without reverse engineering. / 除了倒数，也保存分母，方便直接审计。
    """

    xi = _finite_real(investor_slope, "investor_slope xi / 投资者斜率 xi")
    lambda_hat = _finite_real(
        price_impact_lambda_hat,
        "price_impact_lambda_hat / 价格冲击 lambda_hat",
    )
    if xi < 0.0:
        raise ValueError("investor_slope xi cannot be negative. / xi 不能为负数。")

    xi_lambda = xi * lambda_hat
    if not isfinite(xi_lambda):
        raise OverflowError(
            "xi * lambda_hat overflowed. / xi 与 lambda_hat 的乘积溢出。"
        )

    # fma(-xi, lambda, 1) evaluates the product and subtraction with only one
    # final rounding. For example, naive ``1-500*0.002`` becomes 0.0, while
    # fma preserves the small nonzero binary-float gap. / fma 只在最后舍入一次；
    # 普通 ``1-500*0.002`` 会误成 0，而 fma 能保留二进制浮点的微小非零差。
    signed_sensitivity = fma(-xi, lambda_hat, 1.0)
    if not isfinite(signed_sensitivity):
        raise OverflowError(
            "The inventory-sensitivity calculation overflowed. / 库存敏感度计算溢出。"
        )
    absolute_sensitivity = abs(signed_sensitivity)
    is_exactly_singular = absolute_sensitivity == 0.0
    if is_exactly_singular:
        liquidity = float("inf")
        reciprocal_overflowed = False
    else:
        liquidity = 1.0 / absolute_sensitivity
        reciprocal_overflowed = not isfinite(liquidity)
        if reciprocal_overflowed:
            liquidity = float("inf")

    return PeriodMarketLiquidityCalculation(
        investor_slope_xi=xi,
        price_impact_lambda_hat=lambda_hat,
        rounded_xi_times_lambda_hat=xi_lambda,
        signed_inventory_sensitivity=signed_sensitivity,
        absolute_inventory_sensitivity=absolute_sensitivity,
        market_liquidity=liquidity,
        is_exactly_singular=is_exactly_singular,
        reciprocal_overflowed=reciprocal_overflowed,
    )


@dataclass(frozen=True)
class MarketLiquidityAggregate:
    """Constant-memory summary of several valid period calculations.

    多个有效逐期结果的固定内存汇总。
    """

    observations: int
    literal_liquidity_sum: float
    average_market_liquidity: float
    minimum_period_liquidity: float
    maximum_period_liquidity: float
    minimum_absolute_inventory_sensitivity: float
    infinite_period_count: int
    exact_singular_period_count: int
    reciprocal_overflow_period_count: int


class OnlineMarketLiquidityAccumulator:
    """Accumulate IA.4.6 values without storing measurement rows.

    不保存测量记录，在线累计 IA.4.6 数值。

    The two-number Neumaier sum is more accurate than repeatedly applying
    ``total += value`` while still using constant memory. / Neumaier 双数求和
    比反复 total += value 更准确，同时仍只占固定内存。
    """

    def __init__(self) -> None:
        self.count = 0
        self._running_sum = 0.0
        self._compensation = 0.0
        self.minimum_period_liquidity = float("inf")
        self.maximum_period_liquidity = float("-inf")
        self.minimum_absolute_inventory_sensitivity = float("inf")
        self.infinite_period_count = 0
        self.exact_singular_period_count = 0
        self.reciprocal_overflow_period_count = 0

    def add(self, calculation: PeriodMarketLiquidityCalculation) -> None:
        """Validate a supplied calculation, then commit it atomically.

        验证外部传入的计算结果，再原子化提交。
        """

        if not isinstance(calculation, PeriodMarketLiquidityCalculation):
            raise TypeError(
                "calculation has the wrong type. / calculation 类型错误。"
            )
        canonical = calculate_period_market_liquidity(
            calculation.investor_slope_xi,
            calculation.price_impact_lambda_hat,
        )
        if calculation != canonical:
            raise ValueError(
                "The period calculation is internally inconsistent. / 逐期计算结果内部不一致。"
            )
        self._commit_canonical(canonical)

    def add_from_inputs(
        self,
        investor_slope: float,
        price_impact_lambda_hat: float,
    ) -> PeriodMarketLiquidityCalculation:
        """Calculate once and commit; this is the efficient session path.

        只计算一次并提交；这是正式 session 使用的高效路径。
        """

        calculation = calculate_period_market_liquidity(
            investor_slope,
            price_impact_lambda_hat,
        )
        self._commit_canonical(calculation)
        return calculation

    def _commit_canonical(
        self,
        calculation: PeriodMarketLiquidityCalculation,
    ) -> None:
        """Prepare and commit a calculation created by the pure formula.

        预计算并提交由纯公式函数创建的结果。
        """

        try:
            value = float(calculation.market_liquidity)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "period market liquidity must be numeric. / 逐期市场流动性必须是数字。"
            ) from error
        if value != value or value <= 0.0:
            raise ValueError(
                "period market liquidity must be positive and not NaN. / 逐期市场流动性必须为正且不能是 NaN。"
            )
        sensitivity = _finite_real(
            calculation.absolute_inventory_sensitivity,
            "absolute inventory sensitivity / 库存绝对敏感度",
        )
        if sensitivity < 0.0:
            raise ValueError(
                "Absolute sensitivity cannot be negative. / 绝对敏感度不能为负。"
            )
        is_infinite = value == float("inf")
        if is_infinite != (
            calculation.is_exactly_singular
            or calculation.reciprocal_overflowed
        ):
            raise ValueError(
                "Infinite-liquidity status is internally inconsistent. / 无穷流动性状态内部不一致。"
            )
        if calculation.is_exactly_singular != (sensitivity == 0.0):
            raise ValueError(
                "Exact-singularity status is internally inconsistent. / 精确奇点状态内部不一致。"
            )

        # Prepare a compensated sum before mutating any field. / 先准备补偿求和，
        # 在所有计算成功前不改变任何字段。
        old_sum = self._running_sum
        if is_infinite:
            new_sum = old_sum
            new_compensation = self._compensation
        else:
            new_sum = old_sum + value
            if not isfinite(new_sum):
                raise OverflowError("Finite liquidity sum overflowed. / 有限流动性求和溢出。")
            if abs(old_sum) >= abs(value):
                correction = (old_sum - new_sum) + value
            else:
                correction = (value - new_sum) + old_sum
            new_compensation = self._compensation + correction
            if not isfinite(new_compensation):
                raise OverflowError(
                    "Liquidity compensation overflowed. / 流动性补偿项溢出。"
                )

        # Commit only after every validation succeeds. / 全部验证成功后才提交。
        self._running_sum = new_sum
        self._compensation = new_compensation
        self.minimum_period_liquidity = min(
            self.minimum_period_liquidity,
            value,
        )
        self.maximum_period_liquidity = max(
            self.maximum_period_liquidity,
            value,
        )
        self.minimum_absolute_inventory_sensitivity = min(
            self.minimum_absolute_inventory_sensitivity,
            sensitivity,
        )
        self.infinite_period_count += int(is_infinite)
        self.exact_singular_period_count += int(
            calculation.is_exactly_singular
        )
        self.reciprocal_overflow_period_count += int(
            calculation.reciprocal_overflowed
        )
        self.count += 1

    def summarize(self) -> MarketLiquidityAggregate:
        """Return the literal sum and our disclosed arithmetic mean.

        返回附录展示式的字面求和，以及我们明确采用的算术平均。
        """

        if self.count < 1:
            raise UndefinedMarketLiquidityError(
                "At least one measurement row is required. / 至少需要一条测量记录。"
            )
        if self.infinite_period_count:
            corrected_sum = float("inf")
            average = float("inf")
        else:
            corrected_sum = self._running_sum + self._compensation
            if not isfinite(corrected_sum):
                raise OverflowError("Corrected liquidity sum overflowed. / 修正后的流动性求和溢出。")
            average = corrected_sum / self.count
            if not isfinite(average) or average <= 0.0:
                raise OverflowError("Average liquidity is invalid. / 平均流动性无效。")
        return MarketLiquidityAggregate(
            observations=self.count,
            literal_liquidity_sum=corrected_sum,
            average_market_liquidity=average,
            minimum_period_liquidity=self.minimum_period_liquidity,
            maximum_period_liquidity=self.maximum_period_liquidity,
            minimum_absolute_inventory_sensitivity=(
                self.minimum_absolute_inventory_sensitivity
            ),
            infinite_period_count=self.infinite_period_count,
            exact_singular_period_count=self.exact_singular_period_count,
            reciprocal_overflow_period_count=(
                self.reciprocal_overflow_period_count
            ),
        )


@dataclass(frozen=True)
class MarketLiquidityReceipt:
    """Auditable Step-32 result for one completed seeded session.

    一个完整带种子 session 的可审计 Step-32 结果。
    """

    measurement_periods_scored: int
    first_measurement_index: int
    last_measurement_index: int
    first_global_period_index: int
    last_global_period_index: int
    investor_slope_xi: float
    literal_liquidity_sum: float
    average_market_liquidity: float
    minimum_period_liquidity: float
    maximum_period_liquidity: float
    minimum_absolute_inventory_sensitivity: float
    infinite_period_count: int
    exact_singular_period_count: int
    reciprocal_overflow_period_count: int
    first_infinite_global_period_index: int | None
    parameter_snapshot: PaperParameters
    session_seed_manifest: SessionSeedManifest
    formula: str
    calculation_version: str
    paper_printed_aggregation: str
    replication_aggregation: str
    paper_prose_calls_aggregation_average: bool
    paper_printed_one_over_t: bool
    uses_configured_structural_xi: bool
    uses_period_specific_prior_history_lambda: bool
    uses_fused_multiply_add: bool


class OnlineMarketLiquidityScorer:
    """Session-bound Step-28 sink for Appendix equation IA.4.6.

    与指定 session 绑定的 Step-28 sink，用于在线附录方程 IA.4.6。
    """

    def __init__(self, session: RandomizedMarketSession) -> None:
        if not isinstance(session, RandomizedMarketSession):
            raise TypeError("session has the wrong type. / session 类型错误。")
        if session.period_number != 0 or session.execution_mode != "training":
            raise RuntimeError(
                "Attach the scorer to a fresh training session. / 请把 scorer 连接到尚未运行的训练 session。"
            )
        self._session = session
        self._parameter_snapshot = session.parameters
        self._seed_manifest_snapshot = session.streams.manifest
        self._accumulator = OnlineMarketLiquidityAccumulator()
        self.rows_scored = 0
        self.first_global_period_index: int | None = None
        self.last_global_period_index: int | None = None
        self.first_infinite_global_period_index: int | None = None
        self._final_receipt: MarketLiquidityReceipt | None = None

    def _validated_recorded_lambda(
        self,
        observation: FrozenPolicyPeriodObservation,
    ) -> float:
        """Check that the recorded lambda follows equation (4.2).

        检查记录中的 lambda 是否遵守方程 (4.2)。

        This uses the coefficients already stored in the observation. It never
        queries the now-updated live rolling market maker. / 这里只读取 observation
        已保存的系数，绝不查询此时已经更新过的 live rolling market maker。
        """

        return validate_recorded_price_impact(
            observation,
            self._parameter_snapshot.pricing_error_weight,
        )

    def observe(
        self,
        measurement_index: int,
        observation: FrozenPolicyPeriodObservation,
    ) -> None:
        """Calculate and add one actual Step-28 period before committing.

        计算并加入一条真实 Step-28 记录；全部检查成功后才提交。
        """

        if self._final_receipt is not None:
            raise RuntimeError("This scorer is already finalized. / 这个 scorer 已经完成。")
        if (
            isinstance(measurement_index, bool)
            or not isinstance(measurement_index, Integral)
        ):
            raise TypeError(
                "measurement_index must be an integer. / measurement_index 必须是整数。"
            )
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
            raise ValueError(
                "Global measurement periods must be consecutive. / 全局测量期必须连续。"
            )

        # Prepare the complete economic calculation before changing counters.
        # / 在改变计数器之前，先完成全部经济计算。
        recorded_lambda = self._validated_recorded_lambda(observation)
        calculation = self._accumulator.add_from_inputs(
            self._parameter_snapshot.investor_slope,
            recorded_lambda,
        )

        if self.rows_scored == 0:
            self.first_global_period_index = period_number
        if (
            calculation.market_liquidity == float("inf")
            and self.first_infinite_global_period_index is None
        ):
            self.first_infinite_global_period_index = period_number
        self.last_global_period_index = period_number
        self.rows_scored += 1

    def finalize(
        self,
        controller: SessionPhaseController,
    ) -> MarketLiquidityReceipt:
        """Issue a result only for the exact successfully completed controller.

        只有完全对应且成功结束的 controller 才能生成结果。
        """

        if not isinstance(controller, SessionPhaseController):
            raise TypeError("controller has the wrong type. / controller 类型错误。")
        if controller.session is not self._session:
            raise RuntimeError(
                "This controller belongs to another session. / 这个 controller 属于另一个 session。"
            )
        if controller.phase is not SessionPhase.COMPLETE:
            raise RuntimeError("Step 28 has not completed successfully. / Step 28 尚未成功完成。")
        if self._final_receipt is not None:
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
            or self._accumulator.count != self.rows_scored
        ):
            raise RuntimeError(
                "Step-28 and Step-32 row counts disagree. / Step-28 与 Step-32 行数不一致。"
            )
        if self.first_global_period_index != phase_receipt.measurement_first_period_index:
            raise RuntimeError(
                "The first measurement period disagrees with Step 28. / 首个测量期与 Step 28 不一致。"
            )
        if self.last_global_period_index != phase_receipt.measurement_last_period_index:
            raise RuntimeError(
                "The last measurement period disagrees with Step 28. / 最后测量期与 Step 28 不一致。"
            )
        if self._session.parameters != self._parameter_snapshot:
            raise RuntimeError(
                "Session parameters changed after Step 32 attached. / Step 32 连接后 session 参数发生改变。"
            )
        if self._session.streams.manifest != self._seed_manifest_snapshot:
            raise RuntimeError(
                "The session seed manifest changed after Step 32 attached. / Step 32 连接后随机种子记录发生改变。"
            )

        aggregate = self._accumulator.summarize()
        receipt = MarketLiquidityReceipt(
            measurement_periods_scored=self.rows_scored,
            first_measurement_index=0,
            last_measurement_index=self.rows_scored - 1,
            first_global_period_index=self.first_global_period_index,
            last_global_period_index=self.last_global_period_index,
            investor_slope_xi=float(self._parameter_snapshot.investor_slope),
            literal_liquidity_sum=aggregate.literal_liquidity_sum,
            average_market_liquidity=aggregate.average_market_liquidity,
            minimum_period_liquidity=aggregate.minimum_period_liquidity,
            maximum_period_liquidity=aggregate.maximum_period_liquidity,
            minimum_absolute_inventory_sensitivity=(
                aggregate.minimum_absolute_inventory_sensitivity
            ),
            infinite_period_count=aggregate.infinite_period_count,
            exact_singular_period_count=aggregate.exact_singular_period_count,
            reciprocal_overflow_period_count=(
                aggregate.reciprocal_overflow_period_count
            ),
            first_infinite_global_period_index=(
                self.first_infinite_global_period_index
            ),
            parameter_snapshot=self._parameter_snapshot,
            session_seed_manifest=self._seed_manifest_snapshot,
            formula=MARKET_LIQUIDITY_FORMULA,
            calculation_version=MARKET_LIQUIDITY_VERSION,
            paper_printed_aggregation=PAPER_PRINTED_AGGREGATION,
            replication_aggregation=REPLICATION_AGGREGATION,
            paper_prose_calls_aggregation_average=True,
            paper_printed_one_over_t=False,
            uses_configured_structural_xi=True,
            uses_period_specific_prior_history_lambda=True,
            uses_fused_multiply_add=True,
        )
        self._final_receipt = receipt
        return receipt


def main() -> None:
    """Run direct formula, derivative, aggregation, and singularity checks.

    运行直接公式、导数、汇总和奇点检查。
    """

    xi = 2.0
    lambda_values = (0.0, 0.25, -0.25)
    calculations = tuple(
        calculate_period_market_liquidity(xi, lambda_hat)
        for lambda_hat in lambda_values
    )
    expected_liquidity = (1.0, 2.0, 2.0 / 3.0)
    for calculation, expected in zip(
        calculations,
        expected_liquidity,
        strict=True,
    ):
        assert isclose(
            calculation.market_liquidity,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    accumulator = OnlineMarketLiquidityAccumulator()
    for calculation in calculations:
        accumulator.add(calculation)
    aggregate = accumulator.summarize()
    assert isclose(aggregate.literal_liquidity_sum, 11.0 / 3.0, abs_tol=1e-12)
    assert isclose(aggregate.average_market_liquidity, 11.0 / 9.0, abs_tol=1e-12)

    # Independent derivative check for the middle period. / 对中间时期进行独立导数检查。
    informed_order = 3.0
    intercept = 1.0
    value_mean = 1.0
    lambda_hat = 0.25

    def maker_inventory(noise_order: float) -> float:
        total_flow = informed_order + noise_order
        price = intercept + lambda_hat * total_flow
        insensitive_order = -xi * (price - value_mean)
        return -(insensitive_order + total_flow)

    step = 1e-6
    numerical_sensitivity = abs(
        (maker_inventory(step) - maker_inventory(-step)) / (2.0 * step)
    )
    assert isclose(numerical_sensitivity, 0.5, rel_tol=0.0, abs_tol=1e-9)
    assert isclose(1.0 / numerical_sensitivity, 2.0, rel_tol=0.0, abs_tol=1e-8)
    assert calculate_period_market_liquidity(0.0, 999.0).market_liquidity == 1.0

    exact_singularity = calculate_period_market_liquidity(2.0, 0.5)
    assert exact_singularity.is_exactly_singular
    assert exact_singularity.market_liquidity == float("inf")
    near_singularity = calculate_period_market_liquidity(
        2.0,
        0.5000000000000001,
    )
    assert not near_singularity.is_exactly_singular
    assert isfinite(near_singularity.market_liquidity)
    # This pair exposes the ordinary multiply-then-subtract rounding trap.
    # / 这一对数字揭示普通“先乘后减”的舍入陷阱。
    fma_example = calculate_period_market_liquidity(500.0, 0.002)
    assert fma_example.signed_inventory_sensitivity != 0.0
    assert isfinite(fma_example.market_liquidity)

    print("Step 32: Market liquidity / 步骤 32：市场流动性")
    print(f"Structural investor slope xi / 结构投资者斜率 xi: {xi:.2f}")
    for lambda_hat, calculation in zip(
        lambda_values,
        calculations,
        strict=True,
    ):
        print(
            f"  lambda_hat={lambda_hat:+.2f} -> "
            f"abs(1-xi*lambda_hat)={calculation.absolute_inventory_sensitivity:.6f} "
            f"-> L_t={calculation.market_liquidity:.6f}"
        )
    print(
        "Literal printed sum / 附录展示式的字面求和: "
        f"{aggregate.literal_liquidity_sum:.6f}"
    )
    print(
        "Arithmetic mean over exactly 3 rows / 恰好 3 条记录的算术平均: "
        f"{aggregate.average_market_liquidity:.6f}"
    )
    print(
        "Finite-difference inventory sensitivity / 有限差分库存敏感度: "
        f"{numerical_sensitivity:.6f}"
    )
    print(
        "Near-singular but finite L_t / 接近奇点但仍有限的 L_t: "
        f"{near_singularity.market_liquidity:.6e}"
    )
    print(
        "FMA check for xi=500, lambda=0.002 / 融合乘加检查: "
        f"gap={fma_example.signed_inventory_sensitivity:.6e}, "
        f"L_t={fma_example.market_liquidity:.6e}"
    )
    print("Exact singularity recorded as +infinity / 精确奇点记录为正无穷")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
