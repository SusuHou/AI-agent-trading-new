"""Step 7: calculate the theoretical market-maker price, equation (3.4).

步骤 7：计算论文公式 (3.4) 中的理论做市商价格。

Unlike Step 6, this function does not search through a short list of candidate
prices. It uses the paper's formula to calculate the minimizing price directly.

与 Step 6 不同，这个函数不会在几个候选价格中搜索，而是直接使用论文公式计算
使目标函数最小的价格。
"""

from math import isclose

from step_06_market_maker_objective import evaluate_market_maker_objective


def calculate_theoretical_price(
    total_order_flow: float,
    value_mean: float,
    expected_value_given_flow: float,
    investor_slope: float,
    pricing_error_weight: float,
) -> float:
    """Calculate p_t from paper equation (3.4).

    根据论文公式 (3.4) 计算 p_t。

    expected_value_given_flow means E[v_t | y_t]: the market maker's expected
    fundamental value after observing total order flow y_t. It is not necessarily
    the realized true value v_t.

    expected_value_given_flow 表示 E[v_t | y_t]：做市商看到总订单流 y_t 后，对基本
    价值的预期。它不一定等于这一期最终实现的真实价值 v_t。
    """

    if investor_slope < 0.0:
        raise ValueError("Investor slope xi must be non-negative. / 投资者斜率 xi 不能为负。")

    if pricing_error_weight <= 0.0:
        raise ValueError("Pricing-error weight theta must be positive. / 定价误差权重 theta 必须大于零。")

    denominator = investor_slope**2 + pricing_error_weight

    # First part: price response to observed total order flow y_t.
    # 第一部分：价格对已观察总订单流 y_t 的反应。
    order_flow_component = (
        investor_slope / denominator
    ) * total_order_flow

    # Second part: weight placed on the unconditional mean v_bar.
    # 第二部分：给予无条件平均价值 v_bar 的权重。
    value_mean_component = (
        investor_slope**2 / denominator
    ) * value_mean

    # Third part: weight placed on information inferred from y_t.
    # 第三部分：给予从 y_t 中推断出的基本价值信息的权重。
    inferred_value_component = (
        pricing_error_weight / denominator
    ) * expected_value_given_flow

    return (
        order_flow_component
        + value_mean_component
        + inferred_value_component
    )


def main() -> None:
    """Run direct, objective, and limiting-case validations.

    运行直接手算、目标函数以及极限情形验证。
    """

    # Small test-only numbers make equation (3.4) easy to calculate by hand.
    # 这里使用较小的测试数字，让公式 (3.4) 容易手算；它们不是论文基准参数。
    total_order_flow = 0.5
    value_mean = 1.0
    expected_value_given_flow = 1.2
    investor_slope = 2.0
    pricing_error_weight = 1.0

    theoretical_price = calculate_theoretical_price(
        total_order_flow,
        value_mean,
        expected_value_given_flow,
        investor_slope,
        pricing_error_weight,
    )

    # Hand calculation / 手算：
    # denominator = 2^2 + 1 = 5
    # p = (2/5)(0.5) + (4/5)(1.0) + (1/5)(1.2)
    #   = 0.20 + 0.80 + 0.24 = 1.24
    assert isclose(theoretical_price, 1.24, abs_tol=1e-12)

    print("Step 7: Theoretical price / 步骤 7：理论价格")
    print(f"Denominator xi^2 + theta / 分母: {investor_slope**2 + pricing_error_weight:.2f}")
    print("Hand calculation / 手算: 0.20 + 0.80 + 0.24")
    print(f"Theoretical price p_t / 理论价格: {theoretical_price:.2f}")

    # Verify—not search—that the formula's price has a lower Step 6 objective
    # than one nearby lower price and one nearby higher price.
    # 这里只是验证而不是搜索：检查公式价格的 Step 6 损失是否低于左右邻近价格。
    possible_fundamental_values = [1.0, 1.4]
    conditional_probabilities = [0.5, 0.5]
    lower_price = theoretical_price - 0.10
    higher_price = theoretical_price + 0.10

    lower_result = evaluate_market_maker_objective(
        total_order_flow,
        lower_price,
        value_mean,
        investor_slope,
        pricing_error_weight,
        possible_fundamental_values,
        conditional_probabilities,
    )
    theoretical_result = evaluate_market_maker_objective(
        total_order_flow,
        theoretical_price,
        value_mean,
        investor_slope,
        pricing_error_weight,
        possible_fundamental_values,
        conditional_probabilities,
    )
    higher_result = evaluate_market_maker_objective(
        total_order_flow,
        higher_price,
        value_mean,
        investor_slope,
        pricing_error_weight,
        possible_fundamental_values,
        conditional_probabilities,
    )

    print("\nObjective verification only / 仅验证目标函数:")
    print(f"  At lower price {lower_price:.2f} / 较低价格: {lower_result['objective']:.5f}")
    print(f"  At formula price {theoretical_price:.2f} / 公式价格: {theoretical_result['objective']:.5f}")
    print(f"  At higher price {higher_price:.2f} / 较高价格: {higher_result['objective']:.5f}")

    assert theoretical_result["objective"] < lower_result["objective"]
    assert theoretical_result["objective"] < higher_result["objective"]

    # Paper limit 1: when xi=0, price equals E[v_t | y_t].
    # 论文极限 1：当 xi=0 时，价格等于 E[v_t | y_t]。
    no_insensitive_investors_price = calculate_theoretical_price(
        total_order_flow,
        value_mean,
        expected_value_given_flow,
        investor_slope=0.0,
        pricing_error_weight=pricing_error_weight,
    )
    assert isclose(
        no_insensitive_investors_price,
        expected_value_given_flow,
        abs_tol=1e-12,
    )

    # Paper limit 2: for extremely large xi, price approaches v_bar + y_t/xi.
    # 论文极限 2：当 xi 非常大时，价格接近 v_bar + y_t/xi。
    very_large_investor_slope = 1_000_000.0
    large_slope_price = calculate_theoretical_price(
        total_order_flow,
        value_mean,
        expected_value_given_flow,
        investor_slope=very_large_investor_slope,
        pricing_error_weight=pricing_error_weight,
    )
    market_clearing_limit = (
        value_mean + total_order_flow / very_large_investor_slope
    )
    assert isclose(
        large_slope_price,
        market_clearing_limit,
        abs_tol=1e-12,
    )

    print("\nLimit xi = 0 / 极限 xi = 0:")
    print(f"  Price equals E[v_t | y_t] / 价格等于条件预期: {no_insensitive_investors_price:.2f}")
    print("Limit xi -> very large / 极限 xi 非常大:")
    print(f"  Formula price / 公式价格: {large_slope_price:.7f}")
    print(f"  Clearing limit v_bar + y_t/xi / 市场清算极限: {market_clearing_limit:.7f}")

    print("\nValidation passed / 验证通过")


if __name__ == "__main__":
    main()
