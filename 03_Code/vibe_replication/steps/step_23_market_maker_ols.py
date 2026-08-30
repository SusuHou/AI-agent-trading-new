"""Step 23: estimate the market maker's two linear regressions.

步骤 23：估计做市商的两个线性回归。

Run / 运行:
    py -3 -X utf8 steps/step_23_market_maker_ols.py

Paper equations / 论文方程 (main paper, equation 4.1, PDF page 22):

    z = xi_0 - xi_1 * p + error_z
    v = gamma_0 + gamma_1 * y + error_v

Sign translation / 符号转换:
    Ordinary regression software writes z = intercept + raw_slope * p.
    The paper instead writes z = xi_0 - xi_1 * p. Therefore:

        xi_0_hat = intercept
        xi_1_hat = -raw_slope

    Never use abs(raw_slope): algebraic negation preserves an unexpected sign
    rather than hiding it. / 普通回归把第一式写成“截距 + 原始斜率乘 p”，因此
    xi_0_hat 等于截距，xi_1_hat 等于原始斜率的相反数。不能使用绝对值，否则
    会掩盖异常符号。

Explicit implementation interpretation / 明确的实现解释:
    The paper calls these linear regression models but does not specify the
    estimator, regularization, weighting, coefficient constraints, or singular
    sample handling. We use equal-weight, unregularized OLS with an intercept as
    the transparent correctness reference. / 论文称其为线性回归模型，但没有说明
    估计器、正则化、权重、系数约束或奇异样本处理。我们使用含截距、等权、无
    正则化 OLS，作为透明的正确性基准。

Timing / 时间顺序:
    This function consumes an immutable Step-22 snapshot D_t containing past
    periods only. Current period t must not enter its own regression. / 本函数
    只读取 Step 22 生成的不可变历史快照 D_t；本期数据不能参与自己的回归。

This readable calculation scans the full window. A later optimized rolling
implementation must match it before replacing it in full experiments. / 这个
可读版本会扫描整个窗口；未来的高性能滚动版本必须先与它结果一致，才能用于正式实验。
"""

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass
from math import fsum, isfinite
from numbers import Real

import numpy as np

from step_22_market_maker_rolling_history import (
    MarketMakerHistory,
    MarketObservation,
)


@dataclass(frozen=True)
class OLSLine:
    """The immutable intercept and slope of one fitted line. / 一条拟合直线的不可变结果。"""

    intercept: float
    slope: float


@dataclass(frozen=True)
class MarketMakerOLSEstimates:
    """The four paper-named coefficients estimated from one D_t snapshot.

    从同一个 D_t 快照估计出的四个论文系数。
    """

    xi_0_hat: float
    xi_1_hat: float
    gamma_0_hat: float
    gamma_1_hat: float
    sample_size: int


def _validated_real_values(
    values: Sequence[float],
    label: str,
) -> tuple[float, ...]:
    """Return finite floats with clear errors. / 检查输入并返回有限浮点数。"""

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a number sequence. / {label} 必须是数字序列。")

    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise TypeError(
            f"{label} must be a number sequence. / {label} 必须是数字序列。"
        ) from error

    converted_values: list[float] = []
    for number in raw_values:
        if isinstance(number, bool) or not isinstance(number, Real):
            raise TypeError(f"Every {label} value must be real. / 每个 {label} 数值必须是实数。")
        converted_number = float(number)
        if not isfinite(converted_number):
            raise ValueError(f"Every {label} value must be finite. / 每个 {label} 数值必须有限。")
        converted_values.append(converted_number)
    return tuple(converted_values)


def fit_ols_line(
    explanatory_values: Sequence[float],
    dependent_values: Sequence[float],
    *,
    explanatory_name: str = "x",
) -> OLSLine:
    """Fit dependent = intercept + slope * explanatory by readable OLS.

    使用可读的 OLS 公式拟合“因变量 = 截距 + 斜率 × 解释变量”。
    """

    x_values = _validated_real_values(explanatory_values, explanatory_name)
    dependent = _validated_real_values(dependent_values, "dependent variable / 因变量")

    if len(x_values) != len(dependent):
        raise ValueError("OLS columns must have equal lengths. / OLS 两列长度必须相同。")
    if len(x_values) < 2:
        raise ValueError("OLS requires at least two observations. / OLS 至少需要两条观测。")

    sample_size = len(x_values)
    x_mean = fsum(x_values) / sample_size
    dependent_mean = fsum(dependent) / sample_size
    centered_x = tuple(value - x_mean for value in x_values)
    centered_dependent = tuple(value - dependent_mean for value in dependent)

    # S_xx and S_xy are the two quantities in the hand-written OLS slope.
    # S_xx 与 S_xy 是手写 OLS 斜率公式需要的两个量。
    sum_squared_x = fsum(value * value for value in centered_x)
    sum_cross_products = fsum(
        x_deviation * dependent_deviation
        for x_deviation, dependent_deviation in zip(
            centered_x,
            centered_dependent,
            strict=True,
        )
    )

    if sum_squared_x <= 0.0:
        raise ValueError(
            f"OLS cannot identify a slope because {explanatory_name} has no variation. / "
            f"{explanatory_name} 没有变化，因此 OLS 无法识别斜率。"
        )

    slope = sum_cross_products / sum_squared_x
    intercept = dependent_mean - slope * x_mean
    if not isfinite(slope) or not isfinite(intercept):
        raise ValueError("OLS produced a non-finite coefficient. / OLS 产生了非有限系数。")
    return OLSLine(intercept=intercept, slope=slope)


def fit_market_maker_regressions(
    observations: Sequence[MarketObservation],
) -> MarketMakerOLSEstimates:
    """Estimate (xi_0, xi_1, gamma_0, gamma_1) from one past-only snapshot.

    使用一个只包含过去数据的快照，估计四个做市商系数。

    Small samples are allowed for hand validation. The later paper-mode
    environment must separately require a full T_m-row history before pricing.
    / 为了手算验证，这里允许小样本；之后的论文模式环境必须在定价前另外确认
    历史已经装满 T_m 行。
    """

    rows = tuple(observations)
    for row in rows:
        if not isinstance(row, MarketObservation):
            raise TypeError(
                "Every regression row must be MarketObservation. / "
                "每条回归记录必须是 MarketObservation。"
            )

    prices_p = tuple(row.market_price_p for row in rows)
    insensitive_orders_z = tuple(row.insensitive_order_z for row in rows)
    informed_and_noise_orders_y = tuple(
        row.informed_and_noise_order_y for row in rows
    )
    fundamental_values_v = tuple(row.fundamental_value_v for row in rows)

    demand_line = fit_ols_line(
        prices_p,
        insensitive_orders_z,
        explanatory_name="market price p / 市场价格 p",
    )
    value_line = fit_ols_line(
        informed_and_noise_orders_y,
        fundamental_values_v,
        explanatory_name="observed order flow y / 订单流 y",
    )

    # Paper: z = xi_0 - xi_1*p. Generic OLS: z = intercept + slope*p.
    # 论文写法与普通 OLS 写法之间只有这里这一处符号转换。
    xi_0_hat = demand_line.intercept
    xi_1_hat = -demand_line.slope

    return MarketMakerOLSEstimates(
        xi_0_hat=xi_0_hat,
        xi_1_hat=xi_1_hat,
        gamma_0_hat=value_line.intercept,
        gamma_1_hat=value_line.slope,
        sample_size=len(rows),
    )


def main() -> None:
    """Match hand calculations and independent NumPy OLS. / 匹配手算与 NumPy OLS。"""

    history = MarketMakerHistory(window_size=5)
    historical_rows = (
        MarketObservation(
            fundamental_value_v=0.80,
            market_price_p=0.98,
            insensitive_order_z=10.0,
            informed_and_noise_order_y=-2.0,
        ),
        MarketObservation(
            fundamental_value_v=0.90,
            market_price_p=1.00,
            insensitive_order_z=0.0,
            informed_and_noise_order_y=0.0,
        ),
        MarketObservation(
            fundamental_value_v=1.00,
            market_price_p=1.02,
            insensitive_order_z=-10.0,
            informed_and_noise_order_y=2.0,
        ),
        MarketObservation(
            fundamental_value_v=1.10,
            market_price_p=1.04,
            insensitive_order_z=-20.0,
            informed_and_noise_order_y=4.0,
        ),
    )
    for row in historical_rows:
        assert history.append(row) is None

    # Freeze D_t before current period data exist. / 在本期数据产生前冻结 D_t。
    pre_pricing_snapshot = history.snapshot()
    estimates = fit_market_maker_regressions(pre_pricing_snapshot)

    # Hand calculation for z on p / z 对 p 的手算:
    # p_bar=1.01, z_bar=-5, S_xx=0.002, S_xy=-1
    # raw slope=-500, intercept=500; paper xi_1=-raw slope=500.
    assert abs(estimates.xi_0_hat - 500.0) < 1e-10
    assert abs(estimates.xi_1_hat - 500.0) < 1e-10

    # Hand calculation for v on y / v 对 y 的手算:
    # y_bar=1, v_bar=0.95, S_xx=20, S_xy=1
    # slope=0.05, intercept=0.90.
    assert abs(estimates.gamma_0_hat - 0.90) < 1e-12
    assert abs(estimates.gamma_1_hat - 0.05) < 1e-12
    assert estimates.sample_size == 4

    # Independent check with NumPy's least-squares implementation.
    # 使用 NumPy 的最小二乘实现进行独立核对。
    prices = np.array([row.market_price_p for row in pre_pricing_snapshot])
    z_orders = np.array([row.insensitive_order_z for row in pre_pricing_snapshot])
    order_flows = np.array(
        [row.informed_and_noise_order_y for row in pre_pricing_snapshot]
    )
    values = np.array([row.fundamental_value_v for row in pre_pricing_snapshot])

    z_design = np.column_stack((np.ones(len(prices)), prices))
    value_design = np.column_stack((np.ones(len(order_flows)), order_flows))
    numpy_z_intercept, numpy_z_raw_slope = np.linalg.lstsq(
        z_design,
        z_orders,
        rcond=None,
    )[0]
    numpy_gamma_0, numpy_gamma_1 = np.linalg.lstsq(
        value_design,
        values,
        rcond=None,
    )[0]

    np.testing.assert_allclose(
        [estimates.xi_0_hat, estimates.xi_1_hat],
        [numpy_z_intercept, -numpy_z_raw_slope],
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        [estimates.gamma_0_hat, estimates.gamma_1_hat],
        [numpy_gamma_0, numpy_gamma_1],
        rtol=0.0,
        atol=1e-12,
    )

    # Current row arrives only after the already-fitted price rule is used.
    # 本期记录只在已经估计好的价格规则使用之后到达。
    current_period_row = MarketObservation(
        fundamental_value_v=1.40,
        market_price_p=1.10,
        insensitive_order_z=-50.0,
        informed_and_noise_order_y=10.0,
    )
    assert current_period_row not in pre_pricing_snapshot
    history.append(current_period_row)
    assert len(history.snapshot()) == 5
    assert len(pre_pricing_snapshot) == 4
    assert fit_market_maker_regressions(pre_pricing_snapshot) == estimates

    # Never hide an unexpected upward demand slope with abs(). If raw slope is
    # +1, the paper coefficient xi_1 must honestly be -1. / 绝不能用绝对值掩盖
    # 意外的向上需求斜率；原始斜率为 +1 时，论文系数 xi_1 必须如实等于 -1。
    upward_demand_rows = (
        MarketObservation(
            fundamental_value_v=0.0,
            market_price_p=1.0,
            insensitive_order_z=1.0,
            informed_and_noise_order_y=0.0,
        ),
        MarketObservation(
            fundamental_value_v=1.0,
            market_price_p=2.0,
            insensitive_order_z=2.0,
            informed_and_noise_order_y=1.0,
        ),
    )
    upward_demand_estimates = fit_market_maker_regressions(
        upward_demand_rows
    )
    assert abs(upward_demand_estimates.xi_1_hat - (-1.0)) < 1e-12

    # Frozen coefficients cannot be rewritten later. / 已估计系数不能被事后改写。
    try:
        setattr(estimates, "xi_1_hat", 999.0)
    except FrozenInstanceError:
        estimates_are_frozen = True
    else:
        estimates_are_frozen = False
    assert estimates_are_frozen

    # Explicit safety decisions for invalid regression samples.
    # 对无效回归样本采用明确报错。
    invalid_cases = (
        ((1.0,), (2.0,)),
        ((1.0, 1.0), (2.0, 3.0)),
        ((1.0, 2.0), (3.0,)),
        ((1.0, float("nan")), (2.0, 3.0)),
    )
    for bad_x, bad_dependent in invalid_cases:
        try:
            fit_ols_line(bad_x, bad_dependent)
        except (TypeError, ValueError):
            invalid_case_was_rejected = True
        else:
            invalid_case_was_rejected = False
        assert invalid_case_was_rejected

    try:
        fit_market_maker_regressions([(1.0, 1.0, 0.0, 0.0)])  # type: ignore[list-item]
    except TypeError:
        wrong_history_row_was_rejected = True
    else:
        wrong_history_row_was_rejected = False
    assert wrong_history_row_was_rejected

    print("Step 23: Market-maker OLS / 步骤 23：做市商 OLS")
    print("\nRegression 1 / 回归 1: z = xi_0 - xi_1 * p")
    print(f"  Ordinary raw slope / 普通回归原始斜率: {-estimates.xi_1_hat:.6f}")
    print(f"  xi_0_hat / xi_0 估计值: {estimates.xi_0_hat:.6f}")
    print(f"  xi_1_hat = -raw slope / xi_1 估计值: {estimates.xi_1_hat:.6f}")
    print("\nRegression 2 / 回归 2: v = gamma_0 + gamma_1 * y")
    print(f"  gamma_0_hat / gamma_0 估计值: {estimates.gamma_0_hat:.6f}")
    print(f"  gamma_1_hat / gamma_1 估计值: {estimates.gamma_1_hat:.6f}")
    print(f"\nPast-only sample size / 只含过去数据的样本量: {estimates.sample_size}")
    print("Readable formula matches NumPy least squares. / 可读公式与 NumPy 最小二乘一致。")
    print("Current-period data were excluded. / 本期数据没有进入自己的回归。")
    print("No adaptive lambda or price was calculated. / 尚未计算自适应 lambda 或价格。")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
