"""Step 9: calculate the perfect-cartel benchmark order.

步骤 9：计算完全卡特尔基准下的交易订单。

Run / 运行:
    py -3 steps/step_09_cartel_benchmark.py

This step takes the cartel price-impact coefficient lambda^M as an input.  The
later fixed-point step will solve for the paper-calibrated lambda^M.  Here we
use a small test value so every result can be checked by hand.

本步骤把卡特尔价格冲击系数 lambda^M 当作输入。后面的不动点步骤才会求出论文校准
的 lambda^M。这里使用一个较小的测试值，让每个结果都可以手算核对。
"""

from math import isclose

from step_03_total_order_flow import calculate_total_order_flow
from step_05_speculator_profit import calculate_profit


def calculate_cartel_intensity(
    number_of_speculators: int,
    price_impact: float,
) -> float:
    """Return chi^M = 1 / (2 * I * lambda^M).

    返回卡特尔交易强度 chi^M = 1 / (2 * I * lambda^M)。

    ``number_of_speculators`` is I, the total number of informed speculators.
    ``price_impact`` is lambda^M, the amount by which price responds to one
    additional unit of total order flow in the perfect-cartel benchmark.

    ``number_of_speculators`` 是知情投机者总数 I。``price_impact`` 是 lambda^M，
    表示完全卡特尔基准下总订单流增加一个单位时，价格变化多少。
    """

    if not isinstance(number_of_speculators, int) or number_of_speculators < 1:
        raise ValueError(
            "I must be a positive integer. / I 必须是正整数。"
        )

    if price_impact <= 0.0:
        raise ValueError(
            "Cartel price impact lambda^M must be positive. / "
            "卡特尔价格冲击 lambda^M 必须大于零。"
        )

    return 1.0 / (2.0 * number_of_speculators * price_impact)


def calculate_cartel_order(
    fundamental_value: float,
    value_mean: float,
    cartel_intensity: float,
) -> float:
    """Return one member's cartel order x^M(v) = chi^M * (v - v_bar).

    返回一位成员的卡特尔订单 x^M(v) = chi^M * (v - v_bar)。

    A positive order means buying, a negative order means short-selling, and
    zero means no trade. / 正数表示买入，负数表示卖空，零表示不交易。
    """

    if cartel_intensity <= 0.0:
        raise ValueError(
            "Cartel intensity chi^M must be positive. / "
            "卡特尔交易强度 chi^M 必须大于零。"
        )

    value_signal = fundamental_value - value_mean
    return cartel_intensity * value_signal


def calculate_cartel_price(
    total_order_flow: float,
    value_mean: float,
    price_impact: float,
) -> float:
    """Return the benchmark price p^M(y) = v_bar + lambda^M * y.

    返回卡特尔基准价格 p^M(y) = v_bar + lambda^M * y。
    """

    if price_impact <= 0.0:
        raise ValueError(
            "Cartel price impact lambda^M must be positive. / "
            "卡特尔价格冲击 lambda^M 必须大于零。"
        )

    return value_mean + price_impact * total_order_flow


def expected_joint_cartel_profit(
    symmetric_order: float,
    fundamental_value: float,
    value_mean: float,
    number_of_speculators: int,
    price_impact: float,
) -> float:
    """Return joint expected profit when all cartel members use one order.

    当所有卡特尔成员使用同一个订单时，返回共同预期利润。

    Expected noise order is zero.  Each member earns (v-p)x, and the cartel
    adds the profits of all I members. / 噪声订单的期望为零。每位成员获得
    (v-p)x，卡特尔把全部 I 位成员的利润相加。
    """

    if not isinstance(number_of_speculators, int) or number_of_speculators < 1:
        raise ValueError(
            "I must be a positive integer. / I 必须是正整数。"
        )

    expected_noise_order = 0.0
    expected_total_order_flow = (
        number_of_speculators * symmetric_order + expected_noise_order
    )
    expected_price = calculate_cartel_price(
        expected_total_order_flow,
        value_mean,
        price_impact,
    )

    # Reuse the profit function already validated in Step 5.
    # 重用第 5 步已经验证过的利润函数。
    one_member_profit = calculate_profit(
        fundamental_value,
        expected_price,
        symmetric_order,
    )
    return number_of_speculators * one_member_profit


def main() -> None:
    """Run transparent hand checks of the cartel formulas. / 运行可手算的卡特尔公式检查。"""

    # These are test-only numbers, not the paper's calibrated cartel solution.
    # 这些只是测试数字，不是论文校准后的卡特尔解。
    number_of_speculators = 2
    price_impact = 0.5
    value_mean = 1.0

    cartel_intensity = calculate_cartel_intensity(
        number_of_speculators,
        price_impact,
    )

    # Hand calculation / 手算:
    # chi^M = 1 / (2 * 2 * 0.5) = 0.5.
    assert isclose(cartel_intensity, 0.5, abs_tol=1e-12)

    high_value = 1.3
    mean_value = 1.0
    low_value = 0.7

    buy_order = calculate_cartel_order(
        high_value,
        value_mean,
        cartel_intensity,
    )
    zero_order = calculate_cartel_order(
        mean_value,
        value_mean,
        cartel_intensity,
    )
    short_order = calculate_cartel_order(
        low_value,
        value_mean,
        cartel_intensity,
    )

    # x^M(1.3) = 0.5 * (1.3 - 1.0) = 0.15.
    # x^M(0.7) = 0.5 * (0.7 - 1.0) = -0.15.
    assert isclose(buy_order, 0.15, abs_tol=1e-12)
    assert isclose(zero_order, 0.0, abs_tol=1e-12)
    assert isclose(short_order, -0.15, abs_tol=1e-12)

    # Both members use x^M=0.15.  Since E[u_t]=0, expected total flow is
    # 0.15 + 0.15 + 0 = 0.30 and expected price is 1 + 0.5*0.30 = 1.15.
    # 两位成员都使用 x^M=0.15。由于 E[u_t]=0，预期总订单流是
    # 0.15 + 0.15 + 0 = 0.30，预期价格是 1 + 0.5*0.30 = 1.15。
    expected_noise_order = 0.0
    expected_total_order_flow = calculate_total_order_flow(
        buy_order,
        buy_order,
        expected_noise_order,
    )
    expected_cartel_price = calculate_cartel_price(
        expected_total_order_flow,
        value_mean,
        price_impact,
    )
    assert isclose(expected_total_order_flow, 0.3, abs_tol=1e-12)
    assert isclose(expected_cartel_price, 1.15, abs_tol=1e-12)

    # A realized noise order still changes total flow and price.  This reuses
    # the two-trader order-flow function already validated in Step 3.
    # 实际噪声订单仍然会改变总订单流和价格。这里重用第 3 步已验证的双交易者函数。
    realized_noise_order = 0.1
    realized_total_order_flow = calculate_total_order_flow(
        buy_order,
        buy_order,
        realized_noise_order,
    )
    realized_cartel_price = calculate_cartel_price(
        realized_total_order_flow,
        value_mean,
        price_impact,
    )
    assert isclose(realized_total_order_flow, 0.4, abs_tol=1e-12)
    assert isclose(realized_cartel_price, 1.2, abs_tol=1e-12)
    realized_profit_per_member = calculate_profit(
        high_value,
        realized_cartel_price,
        buy_order,
    )
    assert isclose(realized_profit_per_member, 0.015, abs_tol=1e-12)

    # The cartel first-order-condition identity is:
    # 卡特尔一阶条件恒等式是：
    # 2 * I * lambda^M * chi^M = 1.
    first_order_condition_residual = (
        2.0 * number_of_speculators * price_impact * cartel_intensity - 1.0
    )
    assert isclose(first_order_condition_residual, 0.0, abs_tol=1e-12)

    # Vary every member's order together because the cartel chooses all of
    # their orders jointly. / 同时改变每位成员的订单，因为卡特尔联合选择全部订单。
    smaller_order = buy_order - 0.05
    larger_order = buy_order + 0.05

    smaller_order_joint_profit = expected_joint_cartel_profit(
        smaller_order,
        high_value,
        value_mean,
        number_of_speculators,
        price_impact,
    )
    cartel_order_joint_profit = expected_joint_cartel_profit(
        buy_order,
        high_value,
        value_mean,
        number_of_speculators,
        price_impact,
    )
    larger_order_joint_profit = expected_joint_cartel_profit(
        larger_order,
        high_value,
        value_mean,
        number_of_speculators,
        price_impact,
    )

    assert cartel_order_joint_profit > smaller_order_joint_profit
    assert cartel_order_joint_profit > larger_order_joint_profit
    cartel_order_profit_per_member = (
        cartel_order_joint_profit / number_of_speculators
    )

    print("Step 9: Perfect-cartel benchmark / 步骤 9：完全卡特尔基准")
    print("Test-only inputs / 仅用于测试的输入:")
    print(f"  Number of informed speculators I / 知情投机者数量: {number_of_speculators}")
    print(f"  Cartel price impact lambda^M / 卡特尔价格冲击: {price_impact:.2f}")
    print(f"  Mean fundamental value v_bar / 基本价值均值: {value_mean:.2f}")
    print(f"Cartel intensity chi^M / 卡特尔交易强度: {cartel_intensity:.6f}")
    print(f"When v = {high_value:.1f}, x^M = {buy_order:.2f} (buy / 买入)")
    print(f"When v = {mean_value:.1f}, x^M = {zero_order:.2f} (no trade / 不交易)")
    print(f"When v = {low_value:.1f}, x^M = {short_order:.2f} (short / 卖空)")
    print(
        "Expected total flow at v = 1.3, using E[u_t] = 0 / "
        f"v=1.3 时的预期总订单流: {expected_total_order_flow:.2f}"
    )
    print(
        "Expected cartel price / 预期卡特尔价格: "
        f"{expected_cartel_price:.2f}"
    )
    print(
        f"With realized noise u_t = {realized_noise_order:.1f}, "
        f"y_t = {realized_total_order_flow:.2f}, p_t = {realized_cartel_price:.2f}, "
        f"and profit per member = {realized_profit_per_member:.5f}. / "
        f"当实际噪声 u_t = {realized_noise_order:.1f} 时，"
        f"y_t = {realized_total_order_flow:.2f}，p_t = {realized_cartel_price:.2f}，"
        f"每位成员利润 = {realized_profit_per_member:.5f}。"
    )
    print(
        "Information-insensitive investors are not removed: their slope xi "
        "will help determine lambda^M in Step 10. / 信息不敏感投资者没有被删除："
        "其斜率 xi 将在第 10 步参与决定 lambda^M。"
    )

    print("\nNearby symmetric-order joint-profit check / 附近对称订单共同利润检查:")
    print(f"  Smaller order {smaller_order:.2f}: {smaller_order_joint_profit:.5f}")
    print(f"  Cartel order {buy_order:.2f}: {cartel_order_joint_profit:.5f}")
    print(f"  Larger order {larger_order:.2f}: {larger_order_joint_profit:.5f}")
    print(
        "  Profit per member at cartel order / 卡特尔订单下每位成员利润: "
        f"{cartel_order_profit_per_member:.5f}"
    )
    print(
        "First-order-condition residual / 第一阶条件残差: "
        f"{first_order_condition_residual:.2e}"
    )
    print(
        "The calibrated lambda^M is not solved yet; Step 10 will solve the "
        "coupled fixed point. / 尚未求出校准后的 lambda^M；第 10 步将求解联立不动点。"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
