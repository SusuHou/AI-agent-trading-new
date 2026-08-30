"""Step 8: calculate the non-collusive Nash benchmark order.

步骤 8：计算非合谋纳什基准下的交易订单。

Run / 运行:
    py -3 steps/step_08_nash_benchmark.py

This step takes the Nash price-impact coefficient lambda^N as an input.  The
later fixed-point step will solve for the paper-calibrated lambda^N.  Here we
use a small test value so every result can be checked by hand.

本步骤把纳什价格冲击系数 lambda^N 当作输入。后面的不动点步骤才会求出论文校准
的 lambda^N。这里使用一个较小的测试值，让每个结果都可以手算核对。
"""

from math import isclose

from step_03_total_order_flow import calculate_total_order_flow
from step_05_speculator_profit import calculate_profit


def calculate_nash_intensity(
    number_of_speculators: int,
    price_impact: float,
) -> float:
    """Return chi^N = 1 / ((I + 1) * lambda^N).

    返回纳什交易强度 chi^N = 1 / ((I + 1) * lambda^N)。

    ``number_of_speculators`` is I, the total number of informed speculators.
    ``price_impact`` is lambda^N, the amount by which price responds to one
    additional unit of total order flow in the Nash benchmark.

    ``number_of_speculators`` 是知情投机者总数 I。``price_impact`` 是 lambda^N，
    表示纳什基准下总订单流增加一个单位时，价格变化多少。
    """

    if not isinstance(number_of_speculators, int) or number_of_speculators < 1:
        raise ValueError(
            "I must be a positive integer. / I 必须是正整数。"
        )

    if price_impact <= 0.0:
        raise ValueError(
            "Nash price impact lambda^N must be positive. / "
            "纳什价格冲击 lambda^N 必须大于零。"
        )

    return 1.0 / ((number_of_speculators + 1) * price_impact)


def calculate_nash_order(
    fundamental_value: float,
    value_mean: float,
    nash_intensity: float,
) -> float:
    """Return one trader's Nash order x^N(v) = chi^N * (v - v_bar).

    返回一位交易者的纳什订单 x^N(v) = chi^N * (v - v_bar)。

    A positive order means buying, a negative order means short-selling, and
    zero means no trade. / 正数表示买入，负数表示卖空，零表示不交易。
    """

    if nash_intensity <= 0.0:
        raise ValueError(
            "Nash intensity chi^N must be positive. / "
            "纳什交易强度 chi^N 必须大于零。"
        )

    value_signal = fundamental_value - value_mean
    return nash_intensity * value_signal


def calculate_nash_price(
    total_order_flow: float,
    value_mean: float,
    price_impact: float,
) -> float:
    """Return the benchmark price p^N(y) = v_bar + lambda^N * y.

    返回纳什基准价格 p^N(y) = v_bar + lambda^N * y。
    """

    if price_impact <= 0.0:
        raise ValueError(
            "Nash price impact lambda^N must be positive. / "
            "纳什价格冲击 lambda^N 必须大于零。"
        )

    return value_mean + price_impact * total_order_flow


def expected_profit_against_nash_traders(
    own_order: float,
    fundamental_value: float,
    value_mean: float,
    number_of_speculators: int,
    nash_intensity: float,
    price_impact: float,
) -> float:
    """Calculate expected one-period profit when the other traders use Nash.

    当其他交易者使用纳什策略时，计算自己的预期单期利润。

    The Nash benchmark price is p^N(y) = v_bar + lambda^N y.  Expected noise
    order is zero, so this hand-check uses the expected total order flow.

    纳什基准价格是 p^N(y) = v_bar + lambda^N y。噪声订单的期望为零，因此这个
    手算检查使用预期总订单流。
    """

    value_signal = fundamental_value - value_mean
    one_other_order = nash_intensity * value_signal
    all_other_orders = (number_of_speculators - 1) * one_other_order
    expected_noise_order = 0.0
    expected_total_order_flow = (
        own_order + all_other_orders + expected_noise_order
    )
    expected_price = calculate_nash_price(
        expected_total_order_flow,
        value_mean,
        price_impact,
    )

    # Reuse the profit function already validated in Step 5.
    # 重用第 5 步已经验证过的利润函数。
    return calculate_profit(fundamental_value, expected_price, own_order)


def main() -> None:
    """Run transparent hand checks of the Nash formulas. / 运行可手算的纳什公式检查。"""

    # These are test-only numbers, not the paper's calibrated Nash solution.
    # 这些只是测试数字，不是论文校准后的纳什解。
    number_of_speculators = 2
    price_impact = 0.5
    value_mean = 1.0

    nash_intensity = calculate_nash_intensity(
        number_of_speculators,
        price_impact,
    )

    # Hand calculation / 手算:
    # chi^N = 1 / ((2 + 1) * 0.5) = 2 / 3.
    expected_intensity = 2.0 / 3.0
    assert isclose(nash_intensity, expected_intensity, abs_tol=1e-12)

    high_value = 1.3
    mean_value = 1.0
    low_value = 0.7

    buy_order = calculate_nash_order(
        high_value,
        value_mean,
        nash_intensity,
    )
    zero_order = calculate_nash_order(
        mean_value,
        value_mean,
        nash_intensity,
    )
    short_order = calculate_nash_order(
        low_value,
        value_mean,
        nash_intensity,
    )

    # x^N(1.3) = (2/3) * (1.3 - 1.0) = 0.2.
    # x^N(0.7) = (2/3) * (0.7 - 1.0) = -0.2.
    assert isclose(buy_order, 0.2, abs_tol=1e-12)
    assert isclose(zero_order, 0.0, abs_tol=1e-12)
    assert isclose(short_order, -0.2, abs_tol=1e-12)

    # At v=1.3, both Nash traders buy 0.2.  Because E[u_t]=0, expected total
    # order flow is 0.2 + 0.2 + 0 = 0.4 and expected price is 1 + 0.5*0.4=1.2.
    # 当 v=1.3 时，两位纳什交易者各买入 0.2。由于 E[u_t]=0，预期总订单流是
    # 0.2 + 0.2 + 0 = 0.4，预期价格是 1 + 0.5*0.4 = 1.2。
    expected_noise_order = 0.0
    expected_total_order_flow = calculate_total_order_flow(
        buy_order,
        buy_order,
        expected_noise_order,
    )
    expected_nash_price = calculate_nash_price(
        expected_total_order_flow,
        value_mean,
        price_impact,
    )
    assert isclose(expected_total_order_flow, 0.4, abs_tol=1e-12)
    assert isclose(expected_nash_price, 1.2, abs_tol=1e-12)

    # A realized noise order changes both total flow and price.  This reuses
    # the total-order-flow function already validated in Step 3.
    # 实际噪声订单会改变总订单流和价格。这里重用第 3 步已经验证的总订单流函数。
    realized_noise_order = 0.1
    realized_total_order_flow = calculate_total_order_flow(
        buy_order,
        buy_order,
        realized_noise_order,
    )
    realized_nash_price = calculate_nash_price(
        realized_total_order_flow,
        value_mean,
        price_impact,
    )
    assert isclose(realized_total_order_flow, 0.5, abs_tol=1e-12)
    assert isclose(realized_nash_price, 1.25, abs_tol=1e-12)

    # The Nash first-order-condition identity is:
    # 第一阶条件恒等式是：
    # (I + 1) * lambda^N * chi^N = 1.
    first_order_condition_residual = (
        (number_of_speculators + 1) * price_impact * nash_intensity - 1.0
    )
    assert isclose(first_order_condition_residual, 0.0, abs_tol=1e-12)

    # Check that the Nash order gives higher expected profit than a nearby
    # smaller or larger order, holding the other trader's order fixed.
    # 固定另一位交易者的订单，检查纳什订单的预期利润是否高于附近较小或较大的订单。
    smaller_order = buy_order - 0.1
    larger_order = buy_order + 0.1

    smaller_order_profit = expected_profit_against_nash_traders(
        smaller_order,
        high_value,
        value_mean,
        number_of_speculators,
        nash_intensity,
        price_impact,
    )
    nash_order_profit = expected_profit_against_nash_traders(
        buy_order,
        high_value,
        value_mean,
        number_of_speculators,
        nash_intensity,
        price_impact,
    )
    larger_order_profit = expected_profit_against_nash_traders(
        larger_order,
        high_value,
        value_mean,
        number_of_speculators,
        nash_intensity,
        price_impact,
    )

    assert nash_order_profit > smaller_order_profit
    assert nash_order_profit > larger_order_profit

    print("Step 8: Nash benchmark / 步骤 8：纳什基准")
    print("Test-only inputs / 仅用于测试的输入:")
    print(f"  Number of informed speculators I / 知情投机者数量: {number_of_speculators}")
    print(f"  Nash price impact lambda^N / 纳什价格冲击: {price_impact:.2f}")
    print(f"  Mean fundamental value v_bar / 基本价值均值: {value_mean:.2f}")
    print(f"Nash intensity chi^N / 纳什交易强度: {nash_intensity:.6f}")
    print(f"When v = {high_value:.1f}, x^N = {buy_order:.2f} (buy / 买入)")
    print(f"When v = {mean_value:.1f}, x^N = {zero_order:.2f} (no trade / 不交易)")
    print(f"When v = {low_value:.1f}, x^N = {short_order:.2f} (short / 卖空)")
    print(
        "Expected total flow at v = 1.3, using E[u_t] = 0 / "
        f"v=1.3 时的预期总订单流: {expected_total_order_flow:.2f}"
    )
    print(
        "Expected Nash price / 预期纳什价格: "
        f"{expected_nash_price:.2f}"
    )
    print(
        f"With realized noise u_t = {realized_noise_order:.1f}, "
        f"y_t = {realized_total_order_flow:.2f} and p_t = {realized_nash_price:.2f}. / "
        f"当实际噪声 u_t = {realized_noise_order:.1f} 时，"
        f"y_t = {realized_total_order_flow:.2f}，p_t = {realized_nash_price:.2f}。"
    )

    print("\nNearby-order profit check / 附近订单利润检查:")
    print(f"  Smaller order {smaller_order:.2f}: {smaller_order_profit:.5f}")
    print(f"  Nash order {buy_order:.2f}: {nash_order_profit:.5f}")
    print(f"  Larger order {larger_order:.2f}: {larger_order_profit:.5f}")
    print(
        "First-order-condition residual / 第一阶条件残差: "
        f"{first_order_condition_residual:.2e}"
    )
    print(
        "The calibrated lambda^N is not solved yet; Step 10 will solve the "
        "coupled fixed point. / 尚未求出校准后的 lambda^N；第 10 步将求解联立不动点。"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
