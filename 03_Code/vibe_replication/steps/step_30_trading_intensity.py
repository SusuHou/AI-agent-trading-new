"""Step 30: estimate the post-convergence trading-intensity metric.

第 30 步：估计收敛后的交易强度指标。

Run the short, human-readable demonstration / 运行简短手算演示:
    py -3 -X utf8 steps/step_30_trading_intensity.py

Run the separate automated tests / 运行独立自动测试:
    py -3 -m unittest discover -s tests -p "test_step30_trading_intensity.py" -v

Paper rule, Online Appendix equation IA.4.4 / 论文规则，在线附录 IA.4.4:

    x_(i,t) = chi_(i,0)^C + chi_(i,1)^C * v_t + error_(i,t)
    chi_hat^C = average_i(chi_hat_(i,1)^C)

Important / 重要:
    - Run one unrestricted regression for EACH informed agent.
      / 对每个知情 agent 分别做一条非约束回归。
    - Estimate an intercept. Do not force intercept = -v_bar * slope.
      / 必须估计截距，不强制截距 = -v_bar * 斜率。
    - Use the actual raw order x_(i,t), not an action index or total flow y_t.
      / 使用实际原始订单 x_(i,t)，不使用动作编号或总订单流 y_t。
    - Consume exactly the same Step-28 measurement rows as Step 29.
      / 与 Step 29 使用完全相同的 Step-28 测量行。

The production estimator uses Welford's centered online covariance. It stores
only O(I) sufficient statistics, not 100,000 observations. / 正式估计器使用
Welford 中心化在线协方差；只保存 O(I) 个充分统计量，不保存十万行。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import fsum, isclose, isfinite, ulp
from numbers import Integral
from pathlib import Path
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
from src.step01_value_grid import discrete_value_std
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


REGRESSION_SPECIFICATION = (
    "per-agent unrestricted OLS: x_i_t = intercept_i + slope_i * v_t + error_i_t"
)
ESTIMATOR_VERSION = "welford-unrestricted-v2-context-snapshot"


class UndefinedTradingIntensityError(ArithmeticError):
    """The observed values do not identify an OLS slope.

    已观测价值无法识别 OLS 斜率。
    """


@dataclass(frozen=True)
class TradingPolicyFit:
    """One immutable unrestricted OLS result before session metadata.

    一份不含 session 元数据的不可修改非约束 OLS 结果。
    """

    observations: int
    number_of_agents: int
    sample_value_mean: float
    minimum_value: float
    maximum_value: float
    centered_value_sum_squares: float
    value_range_numerical_floor: float
    intercept_by_agent: tuple[float, ...]
    slope_by_agent: tuple[float, ...]
    average_trading_intensity: float
    theory_restriction_residual_by_agent: tuple[float, ...]


class OnlineTradingPolicyMoments:
    """Maintain stable OLS sufficient statistics without retaining rows.

    不保存逐期记录，只维护稳定的 OLS 充分统计量。

    For each agent we ultimately need / 对每个 agent 最终需要:

        slope_i = S_vx_i / S_vv
        intercept_i = mean_x_i - slope_i * mean_v
    """

    def __init__(self, number_of_agents: int) -> None:
        if (
            isinstance(number_of_agents, bool)
            or not isinstance(number_of_agents, int)
            or number_of_agents < 1
        ):
            raise ValueError("number_of_agents must be positive. / agent 数量必须为正整数。")
        self.number_of_agents = number_of_agents
        self.count = 0
        self.mean_value = 0.0
        self.centered_value_sum_squares = 0.0
        self.mean_order_by_agent = [0.0] * number_of_agents
        self.centered_value_order_sum_by_agent = [0.0] * number_of_agents
        self.minimum_value: float | None = None
        self.maximum_value: float | None = None

    def add(self, fundamental_value: float, raw_orders: Sequence[float]) -> None:
        """Prepare and atomically add one value plus all agents' orders.

        先计算并验证，再原子式加入一个价值与全部 agent 订单。
        """

        value = float(fundamental_value)
        orders = tuple(float(order) for order in raw_orders)
        if not isfinite(value):
            raise ValueError("fundamental_value must be finite. / 基本价值必须有限。")
        if len(orders) != self.number_of_agents:
            raise ValueError("There must be one raw order per agent. / 每个 agent 必须有一个原始订单。")
        if not all(isfinite(order) for order in orders):
            raise ValueError("Every raw order must be finite. / 每个原始订单必须有限。")

        new_count = self.count + 1
        value_delta = value - self.mean_value
        new_mean_value = self.mean_value + value_delta / new_count
        new_value_sum_squares = (
            self.centered_value_sum_squares
            + value_delta * (value - new_mean_value)
        )

        new_mean_orders: list[float] = []
        new_cross_sums: list[float] = []
        for agent_index, order in enumerate(orders):
            old_mean_order = self.mean_order_by_agent[agent_index]
            order_delta = order - old_mean_order
            new_mean_order = old_mean_order + order_delta / new_count
            new_cross_sum = (
                self.centered_value_order_sum_by_agent[agent_index]
                + value_delta * (order - new_mean_order)
            )
            new_mean_orders.append(new_mean_order)
            new_cross_sums.append(new_cross_sum)

        proposed_values = (
            new_mean_value,
            new_value_sum_squares,
            *new_mean_orders,
            *new_cross_sums,
        )
        if not all(isfinite(number) for number in proposed_values):
            raise OverflowError("Online OLS moments overflowed. / 在线 OLS 统计量溢出。")

        # Commit only after every proposed number is valid. / 全部新数值有效后才提交。
        self.count = new_count
        self.mean_value = new_mean_value
        self.centered_value_sum_squares = new_value_sum_squares
        self.mean_order_by_agent[:] = new_mean_orders
        self.centered_value_order_sum_by_agent[:] = new_cross_sums
        if self.minimum_value is None or value < self.minimum_value:
            self.minimum_value = value
        if self.maximum_value is None or value > self.maximum_value:
            self.maximum_value = value

    def fit(self, value_mean_parameter: float) -> TradingPolicyFit:
        """Convert the running moments into unrestricted OLS coefficients.

        把在线统计量转换为非约束 OLS 系数。
        """

        value_mean = float(value_mean_parameter)
        if not isfinite(value_mean):
            raise ValueError("value_mean_parameter must be finite. / v_bar 必须有限。")
        if self.count < 2:
            raise UndefinedTradingIntensityError("At least two rows are required. / 至少需要两条记录。")
        if self.minimum_value is None or self.maximum_value is None:
            raise RuntimeError("Value bounds are missing. / 价值边界丢失。")

        observed_range = self.maximum_value - self.minimum_value
        value_scale = max(abs(self.minimum_value), abs(self.maximum_value))
        numerical_floor = 64.0 * ulp(value_scale)
        if observed_range <= numerical_floor:
            raise UndefinedTradingIntensityError(
                "The realized fundamental values do not vary enough to identify "
                f"a slope: range={observed_range!r}, floor={numerical_floor!r}. "
                "/ 已实现基本价值变化不足，无法识别斜率。"
            )
        denominator = self.centered_value_sum_squares
        if not isfinite(denominator) or denominator <= 0.0:
            raise UndefinedTradingIntensityError("OLS value variance is not positive. / OLS 价值方差不是正数。")

        slopes = tuple(
            cross_sum / denominator
            for cross_sum in self.centered_value_order_sum_by_agent
        )
        intercepts = tuple(
            mean_order - slope * self.mean_value
            for mean_order, slope in zip(
                self.mean_order_by_agent,
                slopes,
                strict=True,
            )
        )
        average_intensity = fsum(slopes) / self.number_of_agents
        restriction_residuals = tuple(
            intercept + value_mean * slope
            for intercept, slope in zip(intercepts, slopes, strict=True)
        )
        outputs = (
            *slopes,
            *intercepts,
            average_intensity,
            *restriction_residuals,
        )
        if not all(isfinite(number) for number in outputs):
            raise OverflowError("Trading-policy coefficients overflowed. / 交易策略系数溢出。")

        return TradingPolicyFit(
            observations=self.count,
            number_of_agents=self.number_of_agents,
            sample_value_mean=self.mean_value,
            minimum_value=self.minimum_value,
            maximum_value=self.maximum_value,
            centered_value_sum_squares=denominator,
            value_range_numerical_floor=numerical_floor,
            intercept_by_agent=intercepts,
            slope_by_agent=slopes,
            average_trading_intensity=average_intensity,
            theory_restriction_residual_by_agent=restriction_residuals,
        )


def fit_trading_policy_batch_ols(
    fundamental_values: Sequence[float],
    raw_orders_by_period: Sequence[Sequence[float]],
    value_mean_parameter: float,
) -> TradingPolicyFit:
    """Independent readable NumPy OLS oracle for tests and small audits.

    供测试和小型审计使用的独立、可读 NumPy OLS oracle。

    Formal simulations use ``OnlineTradingPolicyMoments`` instead, because
    this batch function keeps every row. / 正式模拟使用在线统计量，因为此批量
    函数会保存全部记录。
    """

    values = np.asarray(tuple(fundamental_values), dtype=float)
    try:
        orders = np.asarray(tuple(tuple(row) for row in raw_orders_by_period), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("raw_orders_by_period must be a rectangular numeric table. / 订单必须是矩形数值表。") from error
    value_mean = float(value_mean_parameter)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("At least two one-dimensional values are required. / 至少需要两个一维价值。")
    if orders.ndim != 2 or orders.shape[0] != values.size or orders.shape[1] < 1:
        raise ValueError("Order rows must match values and contain agents. / 订单行必须与价值一一对应且包含 agent。")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(orders)):
        raise ValueError("All batch inputs must be finite. / 所有批量输入必须有限。")
    if not isfinite(value_mean):
        raise ValueError("value_mean_parameter must be finite. / v_bar 必须有限。")

    design = np.column_stack((np.ones(values.size), values))
    coefficients, _, rank, _ = np.linalg.lstsq(design, orders, rcond=None)
    if rank < 2:
        raise UndefinedTradingIntensityError("Batch OLS cannot identify a slope. / 批量 OLS 无法识别斜率。")
    intercepts = tuple(float(value) for value in coefficients[0, :])
    slopes = tuple(float(value) for value in coefficients[1, :])
    mean_value = float(np.mean(values))
    centered_sum_squares = float(np.sum((values - mean_value) ** 2))
    minimum_value = float(np.min(values))
    maximum_value = float(np.max(values))
    scale = max(abs(minimum_value), abs(maximum_value))
    numerical_floor = 64.0 * ulp(scale)
    return TradingPolicyFit(
        observations=int(values.size),
        number_of_agents=int(orders.shape[1]),
        sample_value_mean=mean_value,
        minimum_value=minimum_value,
        maximum_value=maximum_value,
        centered_value_sum_squares=centered_sum_squares,
        value_range_numerical_floor=numerical_floor,
        intercept_by_agent=intercepts,
        slope_by_agent=slopes,
        average_trading_intensity=fsum(slopes) / len(slopes),
        theory_restriction_residual_by_agent=tuple(
            intercept + value_mean * slope
            for intercept, slope in zip(intercepts, slopes, strict=True)
        ),
    )


@dataclass(frozen=True)
class TradingIntensityReceipt:
    """Immutable Step-30 result for one successfully completed session.

    一个成功完成 session 的不可修改 Step-30 结果。
    """

    measurement_periods_scored: int
    first_measurement_index: int
    last_measurement_index: int
    first_global_period_index: int
    last_global_period_index: int
    number_of_agents: int
    sample_value_mean: float
    minimum_value: float
    maximum_value: float
    centered_value_sum_squares: float
    value_range_numerical_floor: float
    intercept_by_agent: tuple[float, ...]
    slope_by_agent: tuple[float, ...]
    average_trading_intensity: float
    theory_restriction_residual_by_agent: tuple[float, ...]
    value_mean_parameter: float
    parameter_snapshot: PaperParameters
    value_grid_snapshot: tuple[float, ...]
    discrete_value_std_snapshot: float
    session_seed_manifest: SessionSeedManifest
    regression_specification: str
    estimator_version: str
    unrestricted_intercept_estimated: bool
    actual_raw_orders_used: bool


class OnlineTradingIntensityScorer:
    """Session-bound Step-28 sink for Appendix IA.4.4.

    与指定 session 绑定的 Step-28 sink，用于附录 IA.4.4。

    Create one scorer per session. Cross-session merging is deliberately not
    supported because the paper first computes a session-level estimate.
    / 每个 session 建立一个 scorer。本步明确不支持跨 session 合并。
    """

    def __init__(self, session: RandomizedMarketSession) -> None:
        if not isinstance(session, RandomizedMarketSession):
            raise TypeError("session has the wrong type. / session 类型错误。")
        if session.period_number != 0 or session.execution_mode != "training":
            raise RuntimeError("Attach the scorer to a fresh training session. / 请把 scorer 连接到尚未运行的训练 session。")
        if session.parameters.num_speculators < 2:
            raise ValueError("The paper's collusion model requires I >= 2. / 论文的合谋模型要求 I >= 2。")
        self._session = session
        # Freeze the economic context before period 0. The live session object
        # is mutable and could otherwise be rebound after measurement. / 在第 0
        # 期之前冻结经济环境；live session 对象可变，否则测量后可能被重新绑定。
        self._parameter_snapshot = session.parameters
        self._value_grid_snapshot = tuple(float(value) for value in session.value_grid)
        self._discrete_value_std_snapshot = discrete_value_std(
            np.asarray(self._value_grid_snapshot, dtype=float),
            self._parameter_snapshot.value_mean,
        )
        self._moments = OnlineTradingPolicyMoments(
            session.parameters.num_speculators
        )
        self.rows_scored = 0
        self.first_global_period_index: int | None = None
        self.last_global_period_index: int | None = None
        self._final_receipt: TradingIntensityReceipt | None = None

    def observe(
        self,
        measurement_index: int,
        observation: FrozenPolicyPeriodObservation,
    ) -> None:
        """Validate and add one actual Step-28 value/order row.

        验证并加入一条真实 Step-28 价值/订单记录。
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
            or not 0 <= int(value_index) < len(self._session.value_grid)
        ):
            raise ValueError("current_value_index is invalid. / current_value_index 无效。")
        expected_value = self._session.value_grid[int(value_index)]
        if not isclose(
            observation.fundamental_value_v,
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("v_t does not match current_value_index. / v_t 与 current_value_index 不匹配。")

        action_indexes = observation.action_indexes
        number_of_agents = self._moments.number_of_agents
        if len(action_indexes) != number_of_agents:
            raise ValueError("There must be one action index per agent. / 每个 agent 必须有一个动作编号。")
        available_orders = self._session.orders_by_value_and_action[int(value_index)]
        checked_action_indexes: list[int] = []
        for action_index in action_indexes:
            if (
                isinstance(action_index, bool)
                or not isinstance(action_index, Integral)
                or not 0 <= int(action_index) < len(available_orders)
            ):
                raise ValueError("An action index is invalid. / 某个动作编号无效。")
            checked_action_indexes.append(int(action_index))

        if len(observation.raw_orders_x) != number_of_agents:
            raise ValueError("There must be one raw order per agent. / 每个 agent 必须有一个原始订单。")
        for raw_order, action_index in zip(
            observation.raw_orders_x,
            checked_action_indexes,
            strict=True,
        ):
            try:
                observed_order = float(raw_order)
            except (TypeError, ValueError) as error:
                raise ValueError("Every raw order must be numeric. / 每个原始订单必须是数字。") from error
            expected_order = available_orders[action_index]
            if not isfinite(observed_order) or not isclose(
                observed_order,
                expected_order,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "raw_orders_x does not match value/action indexes. "
                    "/ raw_orders_x 与价值/动作编号不一致。"
                )

        # ``add`` validates all orders and prepares all moments before commit.
        # / add 在提交前会验证全部订单与新统计量。
        self._moments.add(
            observation.fundamental_value_v,
            observation.raw_orders_x,
        )
        if self.rows_scored == 0:
            self.first_global_period_index = period_number
        self.last_global_period_index = period_number
        self.rows_scored += 1

    def finalize(
        self,
        controller: SessionPhaseController,
    ) -> TradingIntensityReceipt:
        """Create a result only after the bound Step-28 controller completes.

        只有绑定的 Step-28 controller 成功完成后才生成结果。
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
        if (
            phase_receipt.measurement_periods_required
            != phase_receipt.measurement_periods_completed
            or self.rows_scored != phase_receipt.measurement_periods_completed
        ):
            raise RuntimeError("Step-28 and Step-30 row counts disagree. / Step-28 与 Step-30 行数不一致。")
        if self.first_global_period_index != phase_receipt.measurement_first_period_index:
            raise RuntimeError("The first measurement period disagrees with Step 28. / 首个测量期与 Step 28 不一致。")
        if self.last_global_period_index != phase_receipt.measurement_last_period_index:
            raise RuntimeError("The last measurement period disagrees with Step 28. / 最后测量期与 Step 28 不一致。")
        if self._session.parameters != self._parameter_snapshot:
            raise RuntimeError("Session parameters changed after Step 30 attached. / Step 30 连接后 session 参数发生改变。")
        live_value_grid = tuple(float(value) for value in self._session.value_grid)
        if live_value_grid != self._value_grid_snapshot:
            raise RuntimeError("The value grid changed after Step 30 attached. / Step 30 连接后价值网格发生改变。")

        fit = self._moments.fit(self._session.parameters.value_mean)
        receipt = TradingIntensityReceipt(
            measurement_periods_scored=self.rows_scored,
            first_measurement_index=0,
            last_measurement_index=self.rows_scored - 1,
            first_global_period_index=self.first_global_period_index,
            last_global_period_index=self.last_global_period_index,
            number_of_agents=fit.number_of_agents,
            sample_value_mean=fit.sample_value_mean,
            minimum_value=fit.minimum_value,
            maximum_value=fit.maximum_value,
            centered_value_sum_squares=fit.centered_value_sum_squares,
            value_range_numerical_floor=fit.value_range_numerical_floor,
            intercept_by_agent=fit.intercept_by_agent,
            slope_by_agent=fit.slope_by_agent,
            average_trading_intensity=fit.average_trading_intensity,
            theory_restriction_residual_by_agent=(
                fit.theory_restriction_residual_by_agent
            ),
            value_mean_parameter=self._session.parameters.value_mean,
            parameter_snapshot=self._parameter_snapshot,
            value_grid_snapshot=self._value_grid_snapshot,
            discrete_value_std_snapshot=self._discrete_value_std_snapshot,
            session_seed_manifest=self._session.streams.manifest,
            regression_specification=REGRESSION_SPECIFICATION,
            estimator_version=ESTIMATOR_VERSION,
            unrestricted_intercept_estimated=True,
            actual_raw_orders_used=True,
        )
        self._final_receipt = receipt
        return receipt


MeasurementSink = Callable[[int, FrozenPolicyPeriodObservation], None]


def _measurement_sink_identity(
    sink: MeasurementSink,
) -> tuple[str, int, int | None]:
    """Normalize a callable, treating repeated bound-method reads as one sink.

    规范化 callable 身份；同一绑定方法即使被多次读取，也视为同一 sink。
    """

    if isinstance(sink, MethodType):
        return ("bound_method", id(sink.__self__), id(sink.__func__))
    return ("callable", id(sink), None)


class MeasurementSinkFanout:
    """Callable fan-out whose membership is the tuple it actually executes.

    可调用 fan-out；用于成员证明的 tuple，就是它实际执行的同一个 tuple。

    A plain function cannot forge membership by attaching look-alike public
    metadata. Step 35C accepts this exact class only. / 普通函数不能通过添加
    相似的公开属性伪造成员身份；第 35C 步只接受这个精确 class。
    """

    __slots__ = ("__sinks", "__measurement_session")

    def __init__(
        self,
        sinks: tuple[MeasurementSink, ...],
        measurement_session: RandomizedMarketSession | None,
    ) -> None:
        self.__sinks = sinks
        self.__measurement_session = measurement_session

    @property
    def _measurement_session(self) -> RandomizedMarketSession | None:
        """Declare the bound session to Step 28. / 向第 28 步声明绑定 session。"""

        return self.__measurement_session

    def contains_sink(self, sink: MeasurementSink) -> bool:
        """Prove membership from the same private tuple used by ``__call__``.

        根据 ``__call__`` 使用的同一个私有 tuple 证明成员关系。
        """

        target = _measurement_sink_identity(sink)
        return any(
            _measurement_sink_identity(member) == target
            for member in self.__sinks
        )

    def __call__(
        self,
        measurement_index: int,
        observation: FrozenPolicyPeriodObservation,
    ) -> None:
        """Deliver one row to every registered sink in order.

        按注册顺序把一条记录交给每个 sink。
        """

        for sink in self.__sinks:
            sink(measurement_index, observation)


def build_measurement_sink_fanout(
    *sinks: MeasurementSink,
) -> MeasurementSink:
    """Send each Step-28 row to several sibling metric scorers in order.

    按固定顺序把每条 Step-28 记录发给多个平级指标 scorer。

    If any sink fails, the exception reaches Step 28, which marks the whole
    seeded session FAILED. Discard all partial scorers and restart that seed;
    do not keep a partially updated metric. / 任一 sink 失败时，Step 28 会把
    整个种子 session 标记为 FAILED。应丢弃全部部分结果并重跑该种子。
    """

    if not sinks:
        raise ValueError("At least one measurement sink is required. / 至少需要一个测量 sink。")
    if not all(callable(sink) for sink in sinks):
        raise TypeError("Every measurement sink must be callable. / 每个测量 sink 必须可调用。")

    # Accessing ``scorer.observe`` twice creates two temporary bound-method
    # objects. Their object IDs differ even though they call the same method on
    # the same scorer. Compare (owner, function) in that case. / 连续两次读取
    # ``scorer.observe`` 会产生两个临时绑定方法对象；虽然对象 ID 不同，它们仍是
    # 同一 scorer 的同一方法，因此要比较（拥有者，函数）。
    sink_keys: list[tuple[str, int, int | None]] = []
    bound_sessions: list[RandomizedMarketSession] = []
    for sink in sinks:
        sink_keys.append(_measurement_sink_identity(sink))
        if isinstance(sink, MethodType):
            owner_session = getattr(sink.__self__, "_session", None)
            if isinstance(owner_session, RandomizedMarketSession):
                bound_sessions.append(owner_session)

    if len(set(sink_keys)) != len(sink_keys):
        raise ValueError("The same sink cannot be registered twice. / 同一 sink 不能重复注册。")
    if bound_sessions and any(
        session is not bound_sessions[0]
        for session in bound_sessions[1:]
    ):
        raise ValueError(
            "Session-bound metric sinks must belong to the same session. "
            "/ 与 session 绑定的指标 sink 必须属于同一个 session。"
        )

    # Step 28 reads the read-only session property before claiming a session.
    # / 第 28 步在占用 session 前读取只读的 session 属性。
    measurement_session = bound_sessions[0] if bound_sessions else None
    return MeasurementSinkFanout(tuple(sinks), measurement_session)


def main() -> None:
    """Run a three-row hand example that can be checked with a calculator.

    运行一个可用计算器手算的三行例子。
    """

    values = (0.0, 1.0, 2.0)
    # Agent 1: x = 3 + 2v. Agent 2: x = -4 + 0.5v.
    # / Agent 1: x = 3 + 2v。Agent 2: x = -4 + 0.5v。
    order_rows = (
        (3.0, -4.0),
        (5.0, -3.5),
        (7.0, -3.0),
    )

    online = OnlineTradingPolicyMoments(number_of_agents=2)
    for value, orders in zip(values, order_rows, strict=True):
        online.add(value, orders)
    online_fit = online.fit(value_mean_parameter=1.0)
    batch_fit = fit_trading_policy_batch_ols(
        values,
        order_rows,
        value_mean_parameter=1.0,
    )

    assert all(
        isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(
            online_fit.intercept_by_agent,
            (3.0, -4.0),
            strict=True,
        )
    )
    assert all(
        isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(
            online_fit.slope_by_agent,
            (2.0, 0.5),
            strict=True,
        )
    )
    assert isclose(online_fit.average_trading_intensity, 1.25, abs_tol=1e-12)
    assert np.allclose(
        online_fit.intercept_by_agent,
        batch_fit.intercept_by_agent,
        atol=1e-12,
        rtol=0.0,
    )
    assert np.allclose(
        online_fit.slope_by_agent,
        batch_fit.slope_by_agent,
        atol=1e-12,
        rtol=0.0,
    )

    try:
        constant_values = OnlineTradingPolicyMoments(number_of_agents=2)
        constant_values.add(1.0, (1.0, 2.0))
        constant_values.add(1.0, (2.0, 3.0))
        constant_values.fit(value_mean_parameter=1.0)
    except UndefinedTradingIntensityError:
        pass
    else:
        raise AssertionError("Constant v must not fabricate a slope. / 恒定 v 不得伪造斜率。")

    print("Step 30: unrestricted trading-intensity OLS / 第 30 步：非约束交易强度 OLS")
    print(f"Values v_t / 基本价值: {values}")
    print(f"Agent 1 orders / Agent 1 订单: {[row[0] for row in order_rows]}")
    print(f"Agent 2 orders / Agent 2 订单: {[row[1] for row in order_rows]}")
    print(f"Estimated intercepts / 估计截距: {online_fit.intercept_by_agent}")
    print(f"Estimated slopes / 估计斜率: {online_fit.slope_by_agent}")
    print(
        "Average trading intensity chi_hat^C / 平均交易强度: "
        f"{online_fit.average_trading_intensity:.6f}"
    )
    print("Online result equals independent NumPy OLS / 在线结果等于独立 NumPy OLS")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
