"""Step 29: matched-path Nash/cartel profits and collusion profitability.

第 29 步：使用同一条随机路径计算 Nash/cartel 利润与合谋利润。

Run / 运行:
    py -3 -X utf8 steps/step_29_matched_path_collusion_profitability.py

What "matched path" means / “同一路径”是什么:
    Step 28 produces one realized fundamental value v_t and one realized noise
    order u_t.  We keep those two numbers fixed and ask three questions:

    第 28 步每期产生一个已实现的基本价值 v_t 和噪声订单 u_t。我们固定这两个数，
    再问三个问题：

        1. What profit did each learned AI actually earn?
           / 每个已学习 AI 实际赚了多少？
        2. What would one symmetric Nash trader earn on the same (v_t, u_t)?
           / 同样的 (v_t, u_t) 下，一位 Nash 交易者会赚多少？
        3. What would one perfect-cartel member earn on that same path?
           / 同一路径下，一位完全 cartel 成员会赚多少？

For B in {Nash, cartel}, the paper's per-period reconstruction is:
/ 对 B in {Nash, cartel}，论文的逐期重建公式是：

    x_t^B  = chi^B (v_t - v_bar)
    y_t^B  = I x_t^B + u_t
    p_t^B  = v_bar + lambda^B y_t^B
    pi_t^B = (v_t - p_t^B) x_t^B

The theoretical benchmark x and p remain continuous.  They are NOT rounded to
the AI's 15-action or 31-price grids, and they do NOT use the live adaptive OLS
price. / 理论基准 x 与 p 保持连续；不映射回 AI 的 15 个动作或 31 个
价格网格，也不使用实时自适应 OLS 价格。

Paper outcome / 论文指标:

    Delta_i^C = (mean(pi_i) - mean(pi^N))
                / (mean(pi^M) - mean(pi^N))
    Delta^C   = average_i(Delta_i^C)

The online scorer stores only running sums, so 100,000 measurement periods do
not create a 100,000-row in-memory table. / 在线计分器只保存累计和，因此
10 万个测量期不会在内存中建立 10 万行表格。
"""

from dataclasses import dataclass, replace
from math import fsum, isclose, isfinite, sqrt, ulp
from numbers import Integral
from pathlib import Path
import sys
from collections.abc import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from src.step01_value_grid import discrete_value_std
from step_05_speculator_profit import calculate_profit
from step_08_nash_benchmark import calculate_nash_order, calculate_nash_price
from step_09_cartel_benchmark import (
    calculate_cartel_order,
    calculate_cartel_price,
)
from step_10_fixed_point_solver import solve_benchmark_fixed_point
from step_11_benchmark_profits import (
    calculate_cartel_benchmark_profit,
    calculate_nash_benchmark_profit,
)
from step_25_one_market_period import build_paper_inputs
from step_26_reproducible_random_streams import (
    FrozenPolicyPeriodObservation,
    RandomizedMarketSession,
    SessionSeedManifest,
    build_randomized_paper_session,
)
from step_28_session_phases import (
    SessionPhase,
    SessionPhaseController,
    SessionPhaseReceipt,
)


FIXED_POINT_RESIDUAL_TOLERANCE = 1e-10


@dataclass(frozen=True)
class MatchedPathBenchmarkCoefficients:
    """The fixed theoretical coefficients used by every row in one cell.

    一个实验单元中每条记录共用的固定理论系数。

    ``nash_*`` and ``cartel_*`` are solved once, not re-estimated each period.
    / nash_* 和 cartel_* 只求解一次，不在每期重新估计。
    """

    number_of_speculators: int
    value_mean: float
    discrete_fundamental_std: float
    noise_std: float
    investor_slope: float
    pricing_error_weight: float
    nash_intensity: float
    nash_price_impact: float
    nash_fixed_point_residual: float
    cartel_intensity: float
    cartel_price_impact: float
    cartel_fixed_point_residual: float

    def __post_init__(self) -> None:
        """Reject impossible coefficients immediately. / 立即拒绝不可能的系数。"""

        I = self.number_of_speculators
        if isinstance(I, bool) or not isinstance(I, int) or I < 2:
            raise ValueError("The paper's collusion model requires I >= 2. / 论文的合谋模型要求 I >= 2。")

        finite_values = (
            self.value_mean,
            self.discrete_fundamental_std,
            self.noise_std,
            self.investor_slope,
            self.pricing_error_weight,
            self.nash_intensity,
            self.nash_price_impact,
            self.nash_fixed_point_residual,
            self.cartel_intensity,
            self.cartel_price_impact,
            self.cartel_fixed_point_residual,
        )
        if not all(isfinite(value) for value in finite_values):
            raise ValueError("All benchmark coefficients must be finite. / 所有基准系数必须有限。")
        if self.discrete_fundamental_std <= 0.0:
            raise ValueError("sigma_v_hat must be positive. / sigma_v_hat 必须大于零。")
        if self.noise_std <= 0.0 or self.investor_slope < 0.0:
            raise ValueError("sigma_u must be positive and xi non-negative. / sigma_u 必须大于零，xi 必须非负。")
        if self.pricing_error_weight <= 0.0:
            raise ValueError("theta must be positive. / theta 必须大于零。")
        if min(
            self.nash_intensity,
            self.nash_price_impact,
            self.cartel_intensity,
            self.cartel_price_impact,
        ) <= 0.0:
            raise ValueError("Benchmark chi and lambda must be positive. / 基准 chi 与 lambda 必须大于零。")

        # These identities catch accidentally swapping Nash and cartel values.
        # 这两个恒等式可以抓住误交换 Nash 与 cartel 系数的错误。
        nash_identity = (
            (I + 1) * self.nash_price_impact * self.nash_intensity
        )
        cartel_identity = (
            2.0 * I * self.cartel_price_impact * self.cartel_intensity
        )
        if not isclose(nash_identity, 1.0, rel_tol=1e-11, abs_tol=1e-11):
            raise ValueError("Nash chi/lambda identity failed. / Nash chi/lambda 恒等式失败。")
        if not isclose(cartel_identity, 1.0, rel_tol=1e-11, abs_tol=1e-11):
            raise ValueError("Cartel chi/lambda identity failed. / Cartel chi/lambda 恒等式失败。")

        # Verify the market-maker fixed point itself, not only the trading
        # identities above. / 不仅检查上述交易恒等式，还要检查做市商不动点。
        def recomputed_residual(intensity: float, price_impact: float) -> float:
            informed_flow_slope = I * intensity
            gamma = informed_flow_slope / (
                informed_flow_slope**2
                + (self.noise_std / self.discrete_fundamental_std) ** 2
            )
            implied_price_impact = (
                self.pricing_error_weight * gamma + self.investor_slope
            ) / (
                self.pricing_error_weight + self.investor_slope**2
            )
            return price_impact - implied_price_impact

        recomputed_nash = recomputed_residual(
            self.nash_intensity,
            self.nash_price_impact,
        )
        recomputed_cartel = recomputed_residual(
            self.cartel_intensity,
            self.cartel_price_impact,
        )
        for label, supplied, recomputed in (
            ("Nash", self.nash_fixed_point_residual, recomputed_nash),
            ("cartel", self.cartel_fixed_point_residual, recomputed_cartel),
        ):
            if not isclose(supplied, recomputed, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"{label} fixed-point residual was recorded incorrectly. / {label} 不动点残差记录不正确。")
            if abs(recomputed) > FIXED_POINT_RESIDUAL_TOLERANCE:
                raise ValueError(
                    f"{label} coefficients do not solve the fixed point: "
                    f"residual={recomputed!r}. / {label} 系数未解决不动点。"
                )


def build_matched_path_benchmarks(
    parameters: PaperParameters,
    value_grid: Sequence[float],
) -> MatchedPathBenchmarkCoefficients:
    """Solve the Nash/cartel coefficients using the exact discrete grid.

    使用当前离散价值网格求解 Nash/cartel 系数。

    The paper requires sigma_v_hat from the grid, not nominal sigma_v=1.
    / 论文要求使用网格的 sigma_v_hat，而不是名义上的 sigma_v=1。
    """

    if not isinstance(parameters, PaperParameters):
        raise TypeError("parameters must be PaperParameters. / parameters 类型错误。")
    grid = np.asarray(tuple(value_grid), dtype=float)
    if grid.ndim != 1 or grid.size != parameters.num_value_points:
        raise ValueError("value_grid has the wrong number of points. / value_grid 点数不正确。")
    if not np.all(np.isfinite(grid)):
        raise ValueError("Every value-grid point must be finite. / 每个价值网格点必须有限。")
    if not isclose(
        float(np.mean(grid)),
        parameters.value_mean,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("The value-grid mean does not match v_bar. / 价值网格均值与 v_bar 不一致。")

    sigma_v_hat = discrete_value_std(grid, parameters.value_mean)
    nash = solve_benchmark_fixed_point(
        "nash",
        parameters.num_speculators,
        parameters.noise_std,
        sigma_v_hat,
        parameters.investor_slope,
        parameters.pricing_error_weight,
    )
    cartel = solve_benchmark_fixed_point(
        "cartel",
        parameters.num_speculators,
        parameters.noise_std,
        sigma_v_hat,
        parameters.investor_slope,
        parameters.pricing_error_weight,
    )
    return MatchedPathBenchmarkCoefficients(
        number_of_speculators=parameters.num_speculators,
        value_mean=parameters.value_mean,
        discrete_fundamental_std=sigma_v_hat,
        noise_std=parameters.noise_std,
        investor_slope=parameters.investor_slope,
        pricing_error_weight=parameters.pricing_error_weight,
        nash_intensity=nash["intensity"],
        nash_price_impact=nash["price_impact"],
        nash_fixed_point_residual=nash["fixed_point_residual"],
        cartel_intensity=cartel["intensity"],
        cartel_price_impact=cartel["price_impact"],
        cartel_fixed_point_residual=cartel["fixed_point_residual"],
    )


@dataclass(frozen=True)
class MatchedPathPeriodScore:
    """One temporary hand-checkable score; the accumulator does not retain it.

    一条可手算核对的临时计分；累计器不保存它。
    """

    fundamental_value_v: float
    noise_order_u: float
    actual_profits: tuple[float, ...]
    nash_order_per_agent: float
    nash_total_order_flow: float
    nash_price: float
    nash_profit_per_agent: float
    cartel_order_per_agent: float
    cartel_total_order_flow: float
    cartel_price: float
    cartel_profit_per_agent: float


def calculate_matched_path_period(
    fundamental_value: float,
    noise_order: float,
    actual_profits: Sequence[float],
    benchmarks: MatchedPathBenchmarkCoefficients,
) -> MatchedPathPeriodScore:
    """Rebuild both theoretical benchmarks on one realized (v_t, u_t).

    在一组已实现的 (v_t, u_t) 上重建两个理论基准。
    """

    if not isinstance(benchmarks, MatchedPathBenchmarkCoefficients):
        raise TypeError("benchmarks has the wrong type. / benchmarks 类型错误。")
    v_t = float(fundamental_value)
    u_t = float(noise_order)
    profits = tuple(float(profit) for profit in actual_profits)
    if not isfinite(v_t) or not isfinite(u_t):
        raise ValueError("v_t and u_t must be finite. / v_t 与 u_t 必须有限。")
    if len(profits) != benchmarks.number_of_speculators:
        raise ValueError("There must be one actual profit per agent. / 每个 agent 必须有一个实际利润。")
    if not all(isfinite(profit) for profit in profits):
        raise ValueError("Every actual profit must be finite. / 每个实际利润必须有限。")

    I = benchmarks.number_of_speculators
    v_bar = benchmarks.value_mean

    nash_order = calculate_nash_order(
        v_t,
        v_bar,
        benchmarks.nash_intensity,
    )
    nash_flow = I * nash_order + u_t
    nash_price = calculate_nash_price(
        nash_flow,
        v_bar,
        benchmarks.nash_price_impact,
    )
    nash_profit = calculate_profit(v_t, nash_price, nash_order)

    cartel_order = calculate_cartel_order(
        v_t,
        v_bar,
        benchmarks.cartel_intensity,
    )
    cartel_flow = I * cartel_order + u_t
    cartel_price = calculate_cartel_price(
        cartel_flow,
        v_bar,
        benchmarks.cartel_price_impact,
    )
    cartel_profit = calculate_profit(v_t, cartel_price, cartel_order)

    calculated = (
        nash_order,
        nash_flow,
        nash_price,
        nash_profit,
        cartel_order,
        cartel_flow,
        cartel_price,
        cartel_profit,
    )
    if not all(isfinite(value) for value in calculated):
        raise OverflowError("A matched-path benchmark overflowed. / 同路径基准计算溢出。")

    return MatchedPathPeriodScore(
        fundamental_value_v=v_t,
        noise_order_u=u_t,
        actual_profits=profits,
        nash_order_per_agent=nash_order,
        nash_total_order_flow=nash_flow,
        nash_price=nash_price,
        nash_profit_per_agent=nash_profit,
        cartel_order_per_agent=cartel_order,
        cartel_total_order_flow=cartel_flow,
        cartel_price=cartel_price,
        cartel_profit_per_agent=cartel_profit,
    )


class UndefinedCollusionProfitabilityError(ArithmeticError):
    """The Nash-cartel gap cannot safely normalize this path.

    Nash-cartel 利润差无法安全地归一化这条路径。
    """


@dataclass(frozen=True)
class NormalizedCollusionProfitability:
    """The paper's normalized result after time and agent averaging.

    先对时间取平均、再对 agent 取平均后的论文归一化结果。
    """

    normalization_denominator: float
    denominator_numerical_floor: float
    delta_by_agent: tuple[float, ...]
    delta_c: float


def normalize_collusion_profitability(
    mean_actual_profits: Sequence[float],
    mean_nash_profit: float,
    mean_cartel_profit: float,
) -> NormalizedCollusionProfitability:
    """Apply IA.4.1 without clipping negative or above-one results.

    按 IA.4.1 归一化；不把负数或大于 1 的结果强行截断。

    The paper does not define a zero-denominator software rule.  We therefore
    reject a gap that is non-positive or indistinguishable from zero at
    floating-point resolution; we never add an arbitrary epsilon.
    / 论文没有规定分母为零时的软件处理。因此，若利润差非正，或在浮点
    精度下无法与零区分，我们明确报错；绝不偷偷加一个 epsilon。
    """

    actual = tuple(float(value) for value in mean_actual_profits)
    nash = float(mean_nash_profit)
    cartel = float(mean_cartel_profit)
    if not actual:
        raise ValueError("At least one agent is required. / 至少需要一个 agent。")
    if not all(isfinite(value) for value in (*actual, nash, cartel)):
        raise ValueError("All mean profits must be finite. / 所有平均利润必须有限。")

    denominator = cartel - nash
    # Use the actual profit scale so changing currency units cannot change
    # whether a denominator is accepted. ulp(0.0) is still a positive number.
    # / 使用实际利润尺度，避免只因更换货币单位就改变验收结果。
    scale = max(abs(cartel), abs(nash))
    numerical_floor = 64.0 * ulp(scale)
    if denominator <= numerical_floor:
        raise UndefinedCollusionProfitabilityError(
            "mean(pi^M)-mean(pi^N) is not materially positive: "
            f"denominator={denominator!r}, numerical_floor={numerical_floor!r}. "
            "/ mean(pi^M)-mean(pi^N) 不是可靠的正数。"
        )

    delta_by_agent = tuple(
        (agent_profit - nash) / denominator
        for agent_profit in actual
    )
    delta_c = fsum(delta_by_agent) / len(delta_by_agent)
    if not all(isfinite(value) for value in (*delta_by_agent, delta_c)):
        raise OverflowError("Delta-C overflowed. / Delta-C 计算溢出。")
    return NormalizedCollusionProfitability(
        normalization_denominator=denominator,
        denominator_numerical_floor=numerical_floor,
        delta_by_agent=delta_by_agent,
        delta_c=delta_c,
    )


class _CompensatedSum:
    """A tiny Neumaier running sum; it stores no historical rows.

    一个小型 Neumaier 补偿求和器；不保存历史行。
    """

    __slots__ = ("total", "correction")

    def __init__(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def preview_add(self, value: float) -> tuple[float, float]:
        """Calculate the next state without mutating this sum. / 先算新状态，但不修改当前求和器。"""

        new_total = self.total + value
        if abs(self.total) >= abs(value):
            new_correction = (
                self.correction + (self.total - new_total) + value
            )
        else:
            new_correction = (
                self.correction + (value - new_total) + self.total
            )
        if not isfinite(new_total) or not isfinite(new_correction):
            raise OverflowError("A running profit sum overflowed. / 利润累计和溢出。")
        return new_total, new_correction

    def commit(self, state: tuple[float, float]) -> None:
        """Commit one already-validated state. / 提交一个已验证的新状态。"""

        self.total, self.correction = state

    @property
    def value(self) -> float:
        """Return the compensated total. / 返回补偿后的累计和。"""

        result = self.total + self.correction
        if not isfinite(result):
            raise OverflowError("A compensated sum overflowed. / 补偿累计和溢出。")
        return result


@dataclass(frozen=True)
class CollusionProfitabilityReceipt:
    """Immutable final Step-29 output for one completed session.

    一个已完成 session 的不可修改 Step-29 结果。
    """

    measurement_periods_scored: int
    first_measurement_index: int
    last_measurement_index: int
    first_global_period_index: int
    last_global_period_index: int
    mean_actual_profits: tuple[float, ...]
    mean_nash_profit: float
    mean_cartel_profit: float
    normalization_denominator: float
    denominator_numerical_floor: float
    delta_by_agent: tuple[float, ...]
    delta_c: float
    theoretical_actions_remained_continuous: bool
    adaptive_ols_excluded_from_benchmarks: bool
    session_seed_manifest: SessionSeedManifest
    benchmark_coefficients: MatchedPathBenchmarkCoefficients


class MatchedPathCollusionScorer:
    """Step-28 sink that scores one session with constant memory.

    作为 Step-28 sink，以固定内存对一个 session 计分。

    Mutable running sums are appropriate while data arrive; the final receipt
    is frozen so a completed research result cannot change later.
    / 数据到达时累计和需要可变；最终 receipt 则冻结，防止已完成的研究结果后续改变。

    Chunk merging is deliberately unsupported: create one scorer for one
    session and compute Delta-C before aggregating session-level results.
    Pooling profit sums across sessions would produce a different statistic.
    / 本步明确不支持区块合并：每个 session 建立一个 scorer，先求该
    session 的 Delta-C，再汇总 session 级结果。跨 session 合并利润和会变成另一个统计量。
    """

    def __init__(
        self,
        session: RandomizedMarketSession,
        benchmarks: MatchedPathBenchmarkCoefficients,
    ) -> None:
        if not isinstance(benchmarks, MatchedPathBenchmarkCoefficients):
            raise TypeError("benchmarks has the wrong type. / benchmarks 类型错误。")
        if not isinstance(session, RandomizedMarketSession):
            raise TypeError("session has the wrong type. / session 类型错误。")
        if session.period_number != 0 or session.execution_mode != "training":
            raise RuntimeError("Attach the scorer to a fresh training session. / 请把计分器连接到尚未运行的训练 session。")

        parameters = session.parameters
        if parameters.num_speculators != benchmarks.number_of_speculators:
            raise ValueError("Session I and benchmark I disagree. / session 与基准的 I 不一致。")
        exact_pairs = (
            (parameters.value_mean, benchmarks.value_mean, "v_bar"),
            (parameters.noise_std, benchmarks.noise_std, "sigma_u"),
            (parameters.investor_slope, benchmarks.investor_slope, "xi"),
            (
                parameters.pricing_error_weight,
                benchmarks.pricing_error_weight,
                "theta",
            ),
        )
        for session_value, benchmark_value, label in exact_pairs:
            if session_value != benchmark_value:
                raise ValueError(f"Session and benchmark {label} disagree. / session 与基准的 {label} 不一致。")
        session_sigma_v_hat = discrete_value_std(
            np.asarray(session.value_grid, dtype=float),
            parameters.value_mean,
        )
        if not isclose(
            session_sigma_v_hat,
            benchmarks.discrete_fundamental_std,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Session grid and benchmark sigma_v_hat disagree. / session 网格与基准 sigma_v_hat 不一致。")

        self._session = session
        self.benchmarks = benchmarks
        self._actual_sums = [
            _CompensatedSum()
            for _ in range(benchmarks.number_of_speculators)
        ]
        self._nash_sum = _CompensatedSum()
        self._cartel_sum = _CompensatedSum()
        self.rows_scored = 0
        self.first_global_period_index: int | None = None
        self.last_global_period_index: int | None = None
        self._final_receipt: CollusionProfitabilityReceipt | None = None

    def observe(
        self,
        measurement_index: int,
        observation: FrozenPolicyPeriodObservation,
    ) -> None:
        """Validate, score, then atomically add one Step-28 observation.

        先验证与计算，再原子式加入一条 Step-28 观测。
        """

        if self._final_receipt is not None:
            raise RuntimeError("This scorer is already finalized. / 这个计分器已经完成。")
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
            raise TypeError("observation.period_number must be an integer. / observation.period_number 必须是整数。")
        period_number = int(observation.period_number)
        if period_number < 0:
            raise ValueError("period_number cannot be negative. / period_number 不能为负数。")
        if (
            self.last_global_period_index is not None
            and period_number != self.last_global_period_index + 1
        ):
            raise ValueError("Global measurement periods must be consecutive. / 全局测量期必须连续。")

        # Complete all calculations before changing even one running sum.
        # 在修改任何累计和之前，先完成整条记录的计算。
        score = calculate_matched_path_period(
            observation.fundamental_value_v,
            observation.noise_order_u,
            observation.profits,
            self.benchmarks,
        )
        actual_states = tuple(
            running_sum.preview_add(profit)
            for running_sum, profit in zip(
                self._actual_sums,
                score.actual_profits,
                strict=True,
            )
        )
        nash_state = self._nash_sum.preview_add(score.nash_profit_per_agent)
        cartel_state = self._cartel_sum.preview_add(
            score.cartel_profit_per_agent
        )

        # Only validated states reach this commit block. / 只有已验证状态才会进入提交区。
        for running_sum, state in zip(
            self._actual_sums,
            actual_states,
            strict=True,
        ):
            running_sum.commit(state)
        self._nash_sum.commit(nash_state)
        self._cartel_sum.commit(cartel_state)
        if self.rows_scored == 0:
            self.first_global_period_index = period_number
        self.last_global_period_index = period_number
        self.rows_scored += 1

    def finalize(
        self,
        controller: SessionPhaseController,
    ) -> CollusionProfitabilityReceipt:
        """Create a result only after Step 28 proves successful completion.

        只有 Step 28 证明 session 成功完成后，才生成结果。
        """

        if not isinstance(controller, SessionPhaseController):
            raise TypeError("controller has the wrong type. / controller 类型错误。")
        if controller.session is not self._session:
            raise RuntimeError("This controller belongs to another session. / 这个 controller 属于另一个 session。")
        if controller.phase is not SessionPhase.COMPLETE:
            raise RuntimeError("Step 28 has not completed successfully. / Step 28 尚未成功完成。")
        phase_receipt = controller.final_receipt
        if phase_receipt is None:
            raise RuntimeError("Step-28 completion receipt is missing. / Step-28 完成 receipt 丢失。")
        if self._final_receipt is not None:
            return self._final_receipt
        if not isinstance(phase_receipt, SessionPhaseReceipt):
            raise TypeError("phase_receipt has the wrong type. / phase_receipt 类型错误。")
        required = phase_receipt.measurement_periods_required
        completed = phase_receipt.measurement_periods_completed
        if required != completed or self.rows_scored != completed:
            raise RuntimeError("Step-28 and Step-29 row counts disagree. / Step-28 与 Step-29 的行数不一致。")
        if self.rows_scored < 1:
            raise RuntimeError("No measurement rows were scored. / 没有已计分的测量行。")
        if self.first_global_period_index != phase_receipt.measurement_first_period_index:
            raise RuntimeError("The first measurement period disagrees with Step 28. / 首个测量期与 Step 28 不一致。")
        if self.last_global_period_index != phase_receipt.measurement_last_period_index:
            raise RuntimeError("The last measurement period disagrees with Step 28. / 最后测量期与 Step 28 不一致。")

        count = self.rows_scored
        mean_actual = tuple(
            running_sum.value / count
            for running_sum in self._actual_sums
        )
        mean_nash = self._nash_sum.value / count
        mean_cartel = self._cartel_sum.value / count
        normalized = normalize_collusion_profitability(
            mean_actual,
            mean_nash,
            mean_cartel,
        )
        receipt = CollusionProfitabilityReceipt(
            measurement_periods_scored=count,
            first_measurement_index=0,
            last_measurement_index=count - 1,
            first_global_period_index=self.first_global_period_index,
            last_global_period_index=self.last_global_period_index,
            mean_actual_profits=mean_actual,
            mean_nash_profit=mean_nash,
            mean_cartel_profit=mean_cartel,
            normalization_denominator=(
                normalized.normalization_denominator
            ),
            denominator_numerical_floor=(
                normalized.denominator_numerical_floor
            ),
            delta_by_agent=normalized.delta_by_agent,
            delta_c=normalized.delta_c,
            theoretical_actions_remained_continuous=True,
            adaptive_ols_excluded_from_benchmarks=True,
            session_seed_manifest=self._session.streams.manifest,
            benchmark_coefficients=self.benchmarks,
        )
        self._final_receipt = receipt
        return receipt


def _toy_observation(
    period_number: int,
    fundamental_value: float,
    noise_order: float,
    profits: tuple[float, ...],
) -> FrozenPolicyPeriodObservation:
    """Build a minimal full Step-28 row for transparent tests only.

    仅供透明测试使用，建立一条最小但完整的 Step-28 记录。
    """

    return FrozenPolicyPeriodObservation(
        period_number=period_number,
        current_state_indexes=(0, 0, 0),
        current_state_id=0,
        current_value_index=0,
        fundamental_value_v=fundamental_value,
        action_indexes=(0, 0),
        raw_orders_x=(0.0, 0.0),
        noise_order_u=noise_order,
        total_order_flow_y=0.0,
        xi_0_hat=0.0,
        xi_1_hat=500.0,
        gamma_0_hat=1.0,
        gamma_1_hat=0.0,
        price_impact_lambda_hat=0.0,
        continuous_price_p=1.0,
        insensitive_order_z=0.0,
        profits=profits,
        next_value_index=0,
        next_state_indexes=(0, 0, 0),
        next_price_was_clipped=False,
    )


def main() -> None:
    """Run hand checks, formula oracles, and a tiny Step-28 integration.

    运行手算检查、公式 oracle，以及一个小型 Step-28 整合测试。
    """

    print("Step 29: matched-path collusion profitability / 第 29 步：同路径合谋利润")

    # ---------- 1. Exact hand calculation / 精确手算 ----------
    hand_benchmarks = MatchedPathBenchmarkCoefficients(
        number_of_speculators=2,
        value_mean=1.0,
        discrete_fundamental_std=1.0,
        # These simple-looking coefficients do solve one genuine fixed point:
        # sigma_u=2/sqrt(3), xi=1, theta=7 makes gamma^N=gamma^M=3/7.
        # / 这些简单系数确实解决一个真实不动点。
        noise_std=2.0 / sqrt(3.0),
        investor_slope=1.0,
        pricing_error_weight=7.0,
        nash_intensity=2.0 / 3.0,
        nash_price_impact=0.5,
        nash_fixed_point_residual=0.0,
        cartel_intensity=0.5,
        cartel_price_impact=0.5,
        cartel_fixed_point_residual=0.0,
    )
    positive_row = calculate_matched_path_period(
        1.3,
        0.1,
        (0.01, 0.015),
        hand_benchmarks,
    )
    negative_row = calculate_matched_path_period(
        0.7,
        -0.1,
        (0.01, 0.015),
        hand_benchmarks,
    )
    assert isclose(positive_row.nash_order_per_agent, 0.2, abs_tol=1e-12)
    assert isclose(positive_row.nash_total_order_flow, 0.5, abs_tol=1e-12)
    assert isclose(positive_row.nash_price, 1.25, abs_tol=1e-12)
    assert isclose(positive_row.nash_profit_per_agent, 0.01, abs_tol=1e-12)
    assert isclose(positive_row.cartel_order_per_agent, 0.15, abs_tol=1e-12)
    assert isclose(positive_row.cartel_total_order_flow, 0.4, abs_tol=1e-12)
    assert isclose(positive_row.cartel_price, 1.2, abs_tol=1e-12)
    assert isclose(positive_row.cartel_profit_per_agent, 0.015, abs_tol=1e-12)
    assert isclose(negative_row.nash_profit_per_agent, 0.01, abs_tol=1e-12)
    assert isclose(negative_row.cartel_profit_per_agent, 0.015, abs_tol=1e-12)

    hand_result = normalize_collusion_profitability(
        mean_actual_profits=(0.01, 0.015),
        mean_nash_profit=0.01,
        mean_cartel_profit=0.015,
    )
    assert isclose(hand_result.delta_by_agent[0], 0.0, abs_tol=1e-12)
    assert isclose(hand_result.delta_by_agent[1], 1.0, abs_tol=1e-12)
    assert isclose(hand_result.delta_c, 0.5, abs_tol=1e-12)

    # Changing only realized noise must change the reconstructed benchmarks.
    # 只改变已实现噪声，重建基准也必须改变。
    changed_noise_row = calculate_matched_path_period(
        1.3,
        -0.1,
        (0.01, 0.015),
        hand_benchmarks,
    )
    assert isclose(changed_noise_row.nash_profit_per_agent, 0.03, abs_tol=1e-12)
    assert isclose(changed_noise_row.cartel_profit_per_agent, 0.03, abs_tol=1e-12)

    # Delta is not silently clipped. / Delta 不会被偷偷截到 [0,1]。
    outside_range = normalize_collusion_profitability(
        mean_actual_profits=(-1.0, 3.0),
        mean_nash_profit=0.0,
        mean_cartel_profit=1.0,
    )
    assert outside_range.delta_by_agent == (-1.0, 3.0)
    try:
        normalize_collusion_profitability((0.0, 0.0), 1.0, 1.0)
    except UndefinedCollusionProfitabilityError:
        pass
    else:
        raise AssertionError("A zero denominator should fail. / 零分母应报错。")

    print("\nHand row (v=1.3, u=0.1) / 手算记录：")
    print(
        f"  Nash:   x={positive_row.nash_order_per_agent:.3f}, "
        f"y={positive_row.nash_total_order_flow:.3f}, "
        f"p={positive_row.nash_price:.3f}, "
        f"profit={positive_row.nash_profit_per_agent:.3f}"
    )
    print(
        f"  Cartel: x={positive_row.cartel_order_per_agent:.3f}, "
        f"y={positive_row.cartel_total_order_flow:.3f}, "
        f"p={positive_row.cartel_price:.3f}, "
        f"profit={positive_row.cartel_profit_per_agent:.3f}"
    )
    print(
        "  Two-agent Delta / 两位 agent 的 Delta: "
        f"{hand_result.delta_by_agent}; Delta^C={hand_result.delta_c:.3f}"
    )

    # ---------- 2. Paper coefficients + independent Step-11 oracle ----------
    # / 论文系数 + 独立 Step-11 oracle
    parameters = PaperParameters()
    (
        value_grid,
        price_grid,
        action_multipliers,
        initial_q_table,
        prehistory,
    ) = build_paper_inputs(parameters)
    benchmarks = build_matched_path_benchmarks(parameters, value_grid)

    symmetric_noises = (-parameters.noise_std, parameters.noise_std)
    oracle_scores = tuple(
        calculate_matched_path_period(
            value,
            noise,
            (0.0,) * parameters.num_speculators,
            benchmarks,
        )
        for value in value_grid
        for noise in symmetric_noises
    )
    matched_nash_mean = (
        fsum(score.nash_profit_per_agent for score in oracle_scores)
        / len(oracle_scores)
    )
    matched_cartel_mean = (
        fsum(score.cartel_profit_per_agent for score in oracle_scores)
        / len(oracle_scores)
    )
    closed_form_nash = calculate_nash_benchmark_profit(
        benchmarks.discrete_fundamental_std,
        parameters.num_speculators,
        benchmarks.nash_price_impact,
    )
    closed_form_cartel = calculate_cartel_benchmark_profit(
        benchmarks.discrete_fundamental_std,
        parameters.num_speculators,
        benchmarks.cartel_price_impact,
    )
    assert isclose(matched_nash_mean, closed_form_nash, rel_tol=1e-12)
    assert isclose(matched_cartel_mean, closed_form_cartel, rel_tol=1e-12)
    assert abs(benchmarks.nash_fixed_point_residual) < 1e-10
    assert abs(benchmarks.cartel_fixed_point_residual) < 1e-10

    # Repeat the same independent oracle under the paper's high-noise cell.
    # 在论文的高噪声实验单元中重复同一独立 oracle。
    high_noise_parameters = PaperParameters(noise_std=100.0)
    high_noise_benchmarks = build_matched_path_benchmarks(
        high_noise_parameters,
        value_grid,
    )
    high_noise_scores = tuple(
        calculate_matched_path_period(
            value,
            noise,
            (0.0,) * high_noise_parameters.num_speculators,
            high_noise_benchmarks,
        )
        for value in value_grid
        for noise in (-100.0, 100.0)
    )
    high_noise_nash_mean = (
        fsum(score.nash_profit_per_agent for score in high_noise_scores)
        / len(high_noise_scores)
    )
    high_noise_cartel_mean = (
        fsum(score.cartel_profit_per_agent for score in high_noise_scores)
        / len(high_noise_scores)
    )
    assert isclose(
        high_noise_nash_mean,
        calculate_nash_benchmark_profit(
            high_noise_benchmarks.discrete_fundamental_std,
            high_noise_parameters.num_speculators,
            high_noise_benchmarks.nash_price_impact,
        ),
        rel_tol=1e-12,
    )
    assert isclose(
        high_noise_cartel_mean,
        calculate_cartel_benchmark_profit(
            high_noise_benchmarks.discrete_fundamental_std,
            high_noise_parameters.num_speculators,
            high_noise_benchmarks.cartel_price_impact,
        ),
        rel_tol=1e-12,
    )

    # The paper also studies xi=0. This removes information-insensitive
    # investor demand but does not remove noise or matched-path scoring.
    # / 论文还研究 xi=0；它删去信息不敏感投资者需求，但不删去噪声或同路径计分。
    zero_xi_parameters = PaperParameters(investor_slope=0.0)
    zero_xi_benchmarks = build_matched_path_benchmarks(
        zero_xi_parameters,
        value_grid,
    )
    zero_xi_scores = tuple(
        calculate_matched_path_period(
            value,
            noise,
            (0.0,) * zero_xi_parameters.num_speculators,
            zero_xi_benchmarks,
        )
        for value in value_grid
        for noise in (-parameters.noise_std, parameters.noise_std)
    )
    zero_xi_nash_mean = (
        fsum(score.nash_profit_per_agent for score in zero_xi_scores)
        / len(zero_xi_scores)
    )
    zero_xi_cartel_mean = (
        fsum(score.cartel_profit_per_agent for score in zero_xi_scores)
        / len(zero_xi_scores)
    )
    assert isclose(
        zero_xi_nash_mean,
        calculate_nash_benchmark_profit(
            zero_xi_benchmarks.discrete_fundamental_std,
            zero_xi_parameters.num_speculators,
            zero_xi_benchmarks.nash_price_impact,
        ),
        rel_tol=1e-12,
    )
    assert isclose(
        zero_xi_cartel_mean,
        calculate_cartel_benchmark_profit(
            zero_xi_benchmarks.discrete_fundamental_std,
            zero_xi_parameters.num_speculators,
            zero_xi_benchmarks.cartel_price_impact,
        ),
        rel_tol=1e-12,
    )

    print("\nPaper low-noise coefficients / 论文低噪声系数：")
    print(
        f"  sigma_v_hat={benchmarks.discrete_fundamental_std:.6f}, "
        f"lambda^N={benchmarks.nash_price_impact:.9f}, "
        f"chi^N={benchmarks.nash_intensity:.9f}"
    )
    print(
        f"  lambda^M={benchmarks.cartel_price_impact:.9f}, "
        f"chi^M={benchmarks.cartel_intensity:.9f}"
    )
    print(
        "  Matched-path mean equals Step-11 formula / 同路径平均等于 Step-11 公式: "
        f"Nash={matched_nash_mean:.9f}, Cartel={matched_cartel_mean:.9f}"
    )
    print(
        "  High-noise oracle also matches / 高噪声 oracle 也一致: "
        f"Nash={high_noise_nash_mean:.9f}, "
        f"Cartel={high_noise_cartel_mean:.9f}"
    )
    print(
        "  xi=0 paper cell also matches / xi=0 论文单元也一致: "
        f"lambda^N={zero_xi_benchmarks.nash_price_impact:.9f}, "
        f"lambda^M={zero_xi_benchmarks.cartel_price_impact:.9f}"
    )

    # ---------- 3. Ordering and atomic rejection / 顺序与原子拒绝 ----------
    ordering_session = build_randomized_paper_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        initial_q_table=initial_q_table,
        prehistory=prehistory,
        experiment_seed=20260828,
        experiment_cell_key="step29_low_noise|ordering_test",
        session_index=0,
    )
    ordering_scorer = MatchedPathCollusionScorer(
        ordering_session,
        benchmarks,
    )
    ordering_scorer.observe(
        0,
        _toy_observation(100, 1.3, 0.1, (0.01, 0.015)),
    )
    try:
        ordering_scorer.observe(
            0,
            _toy_observation(101, 0.7, -0.1, (0.01, 0.015)),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("A duplicate measurement index should fail. / 重复测量编号应报错。")
    assert ordering_scorer.rows_scored == 1
    assert ordering_scorer.last_global_period_index == 100
    try:
        ordering_scorer.observe(
            1,
            _toy_observation(102, 0.7, -0.1, (0.01, 0.015)),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("A skipped global period should fail. / 跳过全局时期应报错。")
    assert ordering_scorer.rows_scored == 1
    assert ordering_scorer.last_global_period_index == 100
    try:
        ordering_scorer.observe(
            1,
            _toy_observation(101, 0.7, -0.1, (float("nan"), 0.015)),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("A non-finite profit should fail. / 非有限利润应报错。")
    assert ordering_scorer.rows_scored == 1
    assert ordering_scorer.last_global_period_index == 100
    try:
        ordering_scorer.observe(
            1,
            _toy_observation(101, 0.7, -0.1, (0.01,)),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("A missing agent profit should fail. / 缺少 agent 利润应报错。")
    assert ordering_scorer.rows_scored == 1
    assert ordering_scorer.last_global_period_index == 100

    # ---------- 4. Tiny real Step-28 -> Step-29 connection ----------
    # / 小型真实 Step-28 -> Step-29 连接
    stable_q_table = np.zeros_like(initial_q_table, dtype=float)
    stable_q_table[:, 0] = 1_000_000_000.0
    session = build_randomized_paper_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        initial_q_table=stable_q_table,
        prehistory=prehistory,
        experiment_seed=20260828,
        experiment_cell_key="step29_low_noise|A3=nash",
        session_index=0,
    )
    try:
        MatchedPathCollusionScorer(session, high_noise_benchmarks)
    except ValueError:
        pass
    else:
        raise AssertionError("Low-noise session with high-noise benchmarks should fail. / 低噪声 session 不能连接高噪声基准。")
    scorer = MatchedPathCollusionScorer(session, benchmarks)
    tiny_audit_scores: list[MatchedPathPeriodScore] = []

    def score_and_audit(
        measurement_index: int,
        observation: FrozenPolicyPeriodObservation,
    ) -> None:
        """Retain three rows only in this toy test, never in a paper run.

        只在这个玩具测试中保留 3 条，正式运行不保留。
        """

        independent_score = calculate_matched_path_period(
            observation.fundamental_value_v,
            observation.noise_order_u,
            observation.profits,
            benchmarks,
        )
        tiny_audit_scores.append(independent_score)
        scorer.observe(measurement_index, observation)

    controller = SessionPhaseController.create_for_fresh_session(
        session,
        convergence_periods_required=2,
        measurement_periods_required=3,
        measurement_sink=score_and_audit,
    )
    try:
        scorer.finalize(controller)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Premature finalization should fail. / 提前生成结果应报错。")
    phase_receipt = controller.run_until_complete(
        maximum_training_periods=2,
    )
    assert phase_receipt is controller.final_receipt
    result = scorer.finalize(controller)
    assert scorer.finalize(controller) is result
    assert result.measurement_periods_scored == 3
    assert result.first_global_period_index == 2
    assert result.last_global_period_index == 4
    assert result.mean_actual_profits == tuple(
        fsum(score.actual_profits[agent] for score in tiny_audit_scores) / 3
        for agent in range(parameters.num_speculators)
    )
    assert isclose(
        result.mean_nash_profit,
        fsum(score.nash_profit_per_agent for score in tiny_audit_scores) / 3,
        rel_tol=1e-14,
    )
    assert isclose(
        result.mean_cartel_profit,
        fsum(score.cartel_profit_per_agent for score in tiny_audit_scores) / 3,
        rel_tol=1e-14,
    )
    assert not any(
        name in scorer.__dict__
        for name in ("rows", "history", "observations")
    )

    # Even after valid finalization, another session's controller is rejected.
    # / 即使已成功生成结果，也会拒绝另一 session 的 controller。
    wrong_controller = SessionPhaseController.create_for_fresh_session(
        ordering_session,
        convergence_periods_required=2,
        measurement_periods_required=3,
        measurement_sink=lambda index, observation: None,
    )
    try:
        scorer.finalize(wrong_controller)
    except RuntimeError:
        pass
    else:
        raise AssertionError("A wrong-session controller should fail. / 错误 session 的 controller 应报错。")

    # End-to-end known-answer aggregation: agent 1 is assigned the reconstructed
    # Nash profit and agent 2 the reconstructed cartel profit on each of two
    # real Step-28 rows. The answer must therefore be (0, 1) and 0.5.
    # / 端到端已知答案：在两条真实 Step-28 路径上，测试中给 agent 1
    # 赋 Nash 重建利润，给 agent 2 赋 cartel 重建利润；结果必须是 (0,1) 与 0.5。
    known_session = build_randomized_paper_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        initial_q_table=stable_q_table,
        prehistory=prehistory,
        experiment_seed=20260828,
        experiment_cell_key="step29_low_noise|known_delta_test",
        session_index=0,
    )
    known_scorer = MatchedPathCollusionScorer(
        known_session,
        benchmarks,
    )

    def known_delta_sink(
        measurement_index: int,
        observation: FrozenPolicyPeriodObservation,
    ) -> None:
        path_score = calculate_matched_path_period(
            observation.fundamental_value_v,
            observation.noise_order_u,
            observation.profits,
            benchmarks,
        )
        known_scorer.observe(
            measurement_index,
            replace(
                observation,
                profits=(
                    path_score.nash_profit_per_agent,
                    path_score.cartel_profit_per_agent,
                ),
            ),
        )

    known_controller = SessionPhaseController.create_for_fresh_session(
        known_session,
        convergence_periods_required=1,
        measurement_periods_required=2,
        measurement_sink=known_delta_sink,
    )
    known_controller.run_until_complete(maximum_training_periods=1)
    known_result = known_scorer.finalize(known_controller)
    assert isclose(known_result.delta_by_agent[0], 0.0, abs_tol=1e-12)
    assert isclose(known_result.delta_by_agent[1], 1.0, abs_tol=1e-12)
    assert isclose(known_result.delta_c, 0.5, abs_tol=1e-12)

    # A successful scorer cannot accept an accidental fourth row. / 完成后不能误收第 4 条。
    try:
        scorer.observe(
            3,
            _toy_observation(5, 1.3, 0.1, (0.01, 0.015)),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("A finalized scorer should reject more rows. / 已完成计分器应拒绝新行。")

    print("\nTiny integration only (not a research result) / 仅小型整合测试（非研究结果）：")
    print(
        f"  Step-28 measurement indexes / 测量编号: "
        f"{result.first_measurement_index}..{result.last_measurement_index}"
    )
    print(
        f"  Global periods / 全局时期: "
        f"{result.first_global_period_index}..{result.last_global_period_index}"
    )
    print(
        "  Same realized (v_t,u_t) reused for both benchmarks / "
        "Nash 与 cartel 共用同一实现路径: yes / 是"
    )
    print(
        "  Period rows retained inside scorer / 计分器内保留的逐期行: 0"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
