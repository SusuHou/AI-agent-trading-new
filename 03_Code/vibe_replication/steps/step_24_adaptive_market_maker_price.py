"""Step 24: calculate the adaptive market maker's price.

步骤 24：计算自适应做市商的价格。

Run / 运行:
    py -3 -X utf8 steps/step_24_adaptive_market_maker_price.py

Paper rule / 论文规则 (main paper, equation 4.2, PDF page 22):

    lambda_hat_t
        = (theta * gamma_1_hat_t + xi_1_hat_t)
          / (theta + xi_1_hat_t ** 2)

    p_hat_t(y) = gamma_0_hat_t + lambda_hat_t * y

Important boundaries / 重要边界:
    - xi_0_hat is estimated in equation (4.1) but is NOT used in equation
      (4.2). Do not invent a different price intercept. / xi_0_hat 虽在方程
      (4.1) 中被估计，但不进入方程 (4.2)；不能另造价格截距。
    - Input y is informed traders plus noise, before z exists. Never use y+z.
      / 输入 y 是“知情订单 + 噪声订单”；此时 z 尚未产生，不能使用 y+z。
    - The output is a continuous price. Do not round or clip it to price grid P.
      / 输出是连续价格，不能在这里舍入或截断到价格网格 P。
    - The current row is not appended here. Step 25 appends it only after
      price, z, and profits exist. / 本函数不加入本期历史；第 25 步将在价格、
      z 与利润产生后才加入。

The paper specifies theta>0 and uses theta=0.1 in the baseline. It does not
state price clipping, coefficient constraints, smoothing, or fallback rules.
Our finite-input checks are engineering safety decisions, not paper rules. /
论文规定 theta>0，基准 theta=0.1；没有规定价格截断、系数约束、平滑或失败
回退。这里的有限数检查属于工程安全措施，不是论文规则。
"""

from dataclasses import FrozenInstanceError, dataclass
from math import isfinite
from numbers import Real

from step_22_market_maker_rolling_history import (
    MarketMakerHistory,
    MarketObservation,
)
from step_23_market_maker_ols import (
    MarketMakerOLSEstimates,
    fit_market_maker_regressions,
)


@dataclass(frozen=True)
class AdaptivePriceQuote:
    """An immutable record of one continuous adaptive-price calculation.

    一次连续自适应定价计算的不可修改记录。
    """

    observed_order_flow_y: float
    price_impact_lambda_hat: float
    continuous_price_p_hat: float


def _finite_real(number: float, label: str) -> float:
    """Convert one finite real number with a clear error. / 检查并转换一个有限实数。"""

    if isinstance(number, bool) or not isinstance(number, Real):
        raise TypeError(f"{label} must be a real number. / {label} 必须是实数。")
    converted = float(number)
    if not isfinite(converted):
        raise ValueError(f"{label} must be finite. / {label} 必须是有限数。")
    return converted


def _validate_estimates(
    estimates: MarketMakerOLSEstimates,
) -> MarketMakerOLSEstimates:
    """Validate the complete frozen Step-23 result. / 检查 Step 23 的完整冻结结果。"""

    if not isinstance(estimates, MarketMakerOLSEstimates):
        raise TypeError(
            "estimates must be MarketMakerOLSEstimates. / "
            "estimates 必须是 MarketMakerOLSEstimates。"
        )
    for label, coefficient in (
        ("xi_0_hat", estimates.xi_0_hat),
        ("xi_1_hat", estimates.xi_1_hat),
        ("gamma_0_hat", estimates.gamma_0_hat),
        ("gamma_1_hat", estimates.gamma_1_hat),
    ):
        _finite_real(coefficient, label)
    if (
        isinstance(estimates.sample_size, bool)
        or not isinstance(estimates.sample_size, int)
        or estimates.sample_size < 2
    ):
        raise ValueError(
            "OLS sample_size must be an integer of at least two. / "
            "OLS sample_size 必须是至少为 2 的整数。"
        )
    return estimates


def calculate_adaptive_price_impact(
    estimates: MarketMakerOLSEstimates,
    pricing_error_weight: float,
) -> float:
    """Calculate lambda_hat from equation (4.2). / 使用方程 (4.2) 计算 lambda_hat。"""

    estimates = _validate_estimates(estimates)
    theta = _finite_real(pricing_error_weight, "theta / 定价误差权重 theta")
    if theta <= 0.0:
        raise ValueError("theta must be strictly positive. / theta 必须严格大于零。")

    try:
        numerator = (
            theta * estimates.gamma_1_hat + estimates.xi_1_hat
        )
        denominator = theta + estimates.xi_1_hat ** 2
    except OverflowError as error:
        raise ValueError(
            "Adaptive-price inputs overflowed. / 自适应定价输入导致数值溢出。"
        ) from error
    if not isfinite(numerator) or not isfinite(denominator):
        raise ValueError(
            "Adaptive-price inputs overflowed. / 自适应定价输入导致数值溢出。"
        )
    # theta>0 makes the denominator strictly positive. / theta>0 保证分母严格为正。
    price_impact = numerator / denominator
    if not isfinite(price_impact):
        raise ValueError("lambda_hat is not finite. / lambda_hat 不是有限数。")
    return price_impact


def calculate_adaptive_price_quote(
    observed_order_flow_y: float,
    estimates: MarketMakerOLSEstimates,
    pricing_error_weight: float,
) -> AdaptivePriceQuote:
    """Insert current y into the already-estimated continuous price rule.

    把本期 y 代入已经估计好的连续定价规则。
    """

    current_y = _finite_real(
        observed_order_flow_y,
        "observed order flow y / 做市商观察的订单流 y",
    )
    estimates = _validate_estimates(estimates)
    price_impact = calculate_adaptive_price_impact(
        estimates,
        pricing_error_weight,
    )
    continuous_price = (
        estimates.gamma_0_hat + price_impact * current_y
    )
    if not isfinite(continuous_price):
        raise ValueError("Adaptive price is not finite. / 自适应价格不是有限数。")
    return AdaptivePriceQuote(
        observed_order_flow_y=current_y,
        price_impact_lambda_hat=price_impact,
        continuous_price_p_hat=continuous_price,
    )


def main() -> None:
    """Validate the formula by hand and through Steps 22-23. / 手算并连接步骤 22-23。"""

    # A small exact hand calculation / 一个可以完全手算的小例子:
    # theta=1, gamma_1=2, xi_1=3
    # lambda=(1*2+3)/(1+3^2)=5/10=0.5
    # p(y=4)=gamma_0+lambda*y=10+0.5*4=12
    hand_estimates = MarketMakerOLSEstimates(
        xi_0_hat=7.0,  # deliberately arbitrary and unused / 故意任意且不进入价格
        xi_1_hat=3.0,
        gamma_0_hat=10.0,
        gamma_1_hat=2.0,
        sample_size=4,
    )
    hand_quote = calculate_adaptive_price_quote(
        observed_order_flow_y=4.0,
        estimates=hand_estimates,
        pricing_error_weight=1.0,
    )
    assert abs(hand_quote.price_impact_lambda_hat - 0.5) < 1e-15
    assert abs(hand_quote.continuous_price_p_hat - 12.0) < 1e-15

    # xi_0 is absent from the printed paper formula. Changing only xi_0 must
    # not change lambda or price. / 原文价格公式没有 xi_0；只改变 xi_0 不能改变
    # lambda 或价格。
    changed_xi_0_estimates = MarketMakerOLSEstimates(
        xi_0_hat=-999.0,
        xi_1_hat=3.0,
        gamma_0_hat=10.0,
        gamma_1_hat=2.0,
        sample_size=4,
    )
    quote_after_changing_only_xi_0 = calculate_adaptive_price_quote(
        observed_order_flow_y=4.0,
        estimates=changed_xi_0_estimates,
        pricing_error_weight=1.0,
    )
    assert quote_after_changing_only_xi_0 == hand_quote

    # y=0 gives the paper's intercept gamma_0 exactly. / y=0 时价格正好等于 gamma_0。
    zero_flow_quote = calculate_adaptive_price_quote(
        observed_order_flow_y=0.0,
        estimates=hand_estimates,
        pricing_error_weight=1.0,
    )
    assert zero_flow_quote.continuous_price_p_hat == 10.0

    # If xi_1=0, lambda=gamma_1. / 当 xi_1=0 时，lambda=gamma_1。
    no_insensitive_slope_estimates = MarketMakerOLSEstimates(
        xi_0_hat=0.0,
        xi_1_hat=0.0,
        gamma_0_hat=1.0,
        gamma_1_hat=0.25,
        sample_size=4,
    )
    no_insensitive_slope_quote = calculate_adaptive_price_quote(
        observed_order_flow_y=2.0,
        estimates=no_insensitive_slope_estimates,
        pricing_error_weight=0.1,
    )
    assert no_insensitive_slope_quote.price_impact_lambda_hat == 0.25
    assert no_insensitive_slope_quote.continuous_price_p_hat == 1.5

    # Full toy pipeline: rolling history -> OLS -> current continuous price.
    # 完整玩具管线：滚动历史 -> OLS -> 本期连续价格。
    history = MarketMakerHistory(window_size=4)
    historical_rows = (
        MarketObservation(0.80, 0.98, 10.0, -2.0),
        MarketObservation(0.90, 1.00, 0.0, 0.0),
        MarketObservation(1.00, 1.02, -10.0, 2.0),
        MarketObservation(1.10, 1.04, -20.0, 4.0),
    )
    for row in historical_rows:
        history.append(row)
    assert history.is_full
    history_before_pricing = history.snapshot()

    rolling_estimates = fit_market_maker_regressions(
        history_before_pricing
    )
    paper_theta = 0.1
    current_order_flow_y = 10.0
    rolling_quote = calculate_adaptive_price_quote(
        observed_order_flow_y=current_order_flow_y,
        estimates=rolling_estimates,
        pricing_error_weight=paper_theta,
    )

    expected_numerator = 0.1 * 0.05 + 500.0
    expected_denominator = 0.1 + 500.0 ** 2
    expected_lambda = expected_numerator / expected_denominator
    expected_price = 0.90 + expected_lambda * current_order_flow_y
    assert abs(rolling_quote.price_impact_lambda_hat - expected_lambda) < 1e-15
    assert abs(rolling_quote.continuous_price_p_hat - expected_price) < 1e-15
    assert abs(expected_lambda - 0.00200001919999232) < 1e-15
    assert abs(expected_price - 0.9200001919999232) < 1e-15

    # Pricing reads history and current y but changes neither. / 定价读取历史和本期 y，
    # 但不会修改它们，也不会加入本期记录。
    assert history.snapshot() == history_before_pricing
    assert len(history) == 4
    assert rolling_quote.observed_order_flow_y == current_order_flow_y

    # Preserve unexpected coefficient signs; never abs or clip. / 保留异常符号，不取绝对值。
    negative_xi_estimates = MarketMakerOLSEstimates(
        xi_0_hat=0.0,
        xi_1_hat=-3.0,
        gamma_0_hat=0.0,
        gamma_1_hat=2.0,
        sample_size=4,
    )
    negative_xi_lambda = calculate_adaptive_price_impact(
        negative_xi_estimates,
        pricing_error_weight=1.0,
    )
    assert abs(negative_xi_lambda - (-0.1)) < 1e-15

    # The quote is an immutable receipt. / 定价结果是一张不可修改的流水单。
    try:
        setattr(hand_quote, "continuous_price_p_hat", 999.0)
    except FrozenInstanceError:
        quote_is_frozen = True
    else:
        quote_is_frozen = False
    assert quote_is_frozen

    # Explicitly reject invalid theta, order flow, and coefficient records.
    # 明确拒绝无效 theta、订单流与系数记录。
    for bad_theta in (0.0, -0.1, float("nan"), float("inf")):
        try:
            calculate_adaptive_price_impact(hand_estimates, bad_theta)
        except (TypeError, ValueError):
            bad_theta_was_rejected = True
        else:
            bad_theta_was_rejected = False
        assert bad_theta_was_rejected

    try:
        calculate_adaptive_price_quote(
            float("nan"),
            hand_estimates,
            pricing_error_weight=1.0,
        )
    except ValueError:
        bad_order_flow_was_rejected = True
    else:
        bad_order_flow_was_rejected = False
    assert bad_order_flow_was_rejected

    invalid_estimates = MarketMakerOLSEstimates(
        xi_0_hat=0.0,
        xi_1_hat=float("nan"),
        gamma_0_hat=1.0,
        gamma_1_hat=0.25,
        sample_size=4,
    )
    try:
        calculate_adaptive_price_impact(
            invalid_estimates,
            pricing_error_weight=1.0,
        )
    except ValueError:
        bad_estimates_were_rejected = True
    else:
        bad_estimates_were_rejected = False
    assert bad_estimates_were_rejected

    huge_estimates = MarketMakerOLSEstimates(
        xi_0_hat=0.0,
        xi_1_hat=1e308,
        gamma_0_hat=1.0,
        gamma_1_hat=0.25,
        sample_size=4,
    )
    try:
        calculate_adaptive_price_impact(
            huge_estimates,
            pricing_error_weight=1.0,
        )
    except ValueError:
        coefficient_overflow_was_rejected = True
    else:
        coefficient_overflow_was_rejected = False
    assert coefficient_overflow_was_rejected

    print("Step 24: Adaptive market-maker price / 步骤 24：自适应做市商定价")
    print("\nA. Exact hand calculation / A. 精确手算")
    print("  numerator / 分子: 1 x 2 + 3 = 5")
    print("  denominator / 分母: 1 + 3^2 = 10")
    print(f"  lambda_hat / lambda 估计值: {hand_quote.price_impact_lambda_hat:.6f}")
    print(f"  p_hat(y=4) / y=4 时的价格: {hand_quote.continuous_price_p_hat:.6f}")
    print("  xi_0_hat changed from 7 to -999; price did not change. / xi_0_hat 从 7 改为 -999，价格不变。")
    print("\nB. Steps 22 -> 23 -> 24 / B. 步骤 22 -> 23 -> 24")
    print(f"  xi_1_hat / xi_1 估计值: {rolling_estimates.xi_1_hat:.6f}")
    print(f"  gamma_0_hat / gamma_0 估计值: {rolling_estimates.gamma_0_hat:.6f}")
    print(f"  gamma_1_hat / gamma_1 估计值: {rolling_estimates.gamma_1_hat:.6f}")
    print(f"  current y / 本期 y: {current_order_flow_y:.6f}")
    print(f"  lambda_hat / lambda 估计值: {rolling_quote.price_impact_lambda_hat:.12f}")
    print(f"  continuous p_hat / 连续价格: {rolling_quote.continuous_price_p_hat:.12f}")
    print("The continuous price was not rounded or clipped. / 连续价格未被舍入或截断。")
    print(
        "No current-period z_t, profit, Q update, or history append occurred. / "
        "尚未产生本期 z_t、利润、Q 更新或历史追加。"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
