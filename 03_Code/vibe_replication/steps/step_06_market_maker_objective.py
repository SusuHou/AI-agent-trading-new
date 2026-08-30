"""Step 6: evaluate the market maker's objective in paper equation (3.3).

步骤 6：计算论文公式 (3.3) 中的做市商目标函数。

This step only gives each candidate price a loss score. It does not yet derive
or automatically choose the theoretical optimal price from equation (3.4).

这一步只给每个候选价格计算一个“损失分数”。它还不会推导或自动选择公式
(3.4) 中的理论最优价格。
"""

from math import isclose

from step_04_information_insensitive_investors import (
    calculate_insensitive_order,
)


def calculate_expected_squared_pricing_error(
    candidate_price: float,
    possible_fundamental_values: list[float],
    conditional_probabilities: list[float],
) -> float:
    """Calculate E[(p_t - v_t)^2 | y_t].

    计算 E[(p_t - v_t)^2 | y_t]，也就是做市商认为价格误差平方的平均值。

    The probabilities represent the market maker's beliefs after observing y_t.
    这些概率代表做市商观察到 y_t 后，对不同基本价值的判断。
    """

    if len(possible_fundamental_values) == 0:
        raise ValueError("At least one fundamental value is required. / 至少需要一个基本价值。")

    if len(possible_fundamental_values) != len(conditional_probabilities):
        raise ValueError("Values and probabilities must have the same length. / 价值与概率数量必须相同。")

    if any(probability < 0.0 for probability in conditional_probabilities):
        raise ValueError("Probabilities cannot be negative. / 概率不能是负数。")

    if not isclose(sum(conditional_probabilities), 1.0, abs_tol=1e-12):
        raise ValueError("Probabilities must add up to 1. / 所有概率之和必须等于 1。")

    expected_error = 0.0

    for fundamental_value, probability in zip(
        possible_fundamental_values,
        conditional_probabilities,
    ):
        squared_error = (candidate_price - fundamental_value) ** 2
        expected_error += probability * squared_error

    return expected_error


def evaluate_market_maker_objective(
    total_order_flow: float,
    candidate_price: float,
    value_mean: float,
    investor_slope: float,
    pricing_error_weight: float,
    possible_fundamental_values: list[float],
    conditional_probabilities: list[float],
) -> dict[str, float]:
    """Evaluate E[(y_t + z_t)^2 + theta(p_t - v_t)^2 | y_t].

    计算 E[(y_t + z_t)^2 + theta(p_t - v_t)^2 | y_t]。

    A smaller objective value means a better candidate price for the market maker.
    目标函数越小，说明这个候选价格对做市商越好。
    """

    if investor_slope < 0.0:
        raise ValueError("Investor slope xi must be non-negative. / 投资者斜率 xi 不能为负。")

    if pricing_error_weight <= 0.0:
        raise ValueError("Pricing-error weight theta must be positive. / 定价误差权重 theta 必须大于零。")

    # z_t = -xi * (p_t - v_bar)
    # 信息不敏感投资者看到候选价格后提交订单 z_t。
    insensitive_order = calculate_insensitive_order(
        candidate_price,
        value_mean,
        investor_slope,
    )
    if isclose(insensitive_order, 0.0, abs_tol=1e-12):
        insensitive_order = 0.0

    # The market maker must absorb -(y_t + z_t), so its quadratic inventory
    # cost is (y_t + z_t)^2.
    # 做市商必须接下 -(y_t + z_t)，所以库存成本是 (y_t + z_t)^2。
    net_order = total_order_flow + insensitive_order
    if isclose(net_order, 0.0, abs_tol=1e-12):
        net_order = 0.0
    inventory_cost = net_order**2

    expected_pricing_error = calculate_expected_squared_pricing_error(
        candidate_price,
        possible_fundamental_values,
        conditional_probabilities,
    )
    weighted_pricing_error = pricing_error_weight * expected_pricing_error
    objective = inventory_cost + weighted_pricing_error

    # A dictionary lets us inspect every part instead of seeing only one total.
    # 返回字典，这样我们不仅能看到总分，还能检查每个组成部分。
    return {
        "insensitive_order": insensitive_order,
        "net_order": net_order,
        "inventory_cost": inventory_cost,
        "expected_pricing_error": expected_pricing_error,
        "weighted_pricing_error": weighted_pricing_error,
        "objective": objective,
    }


def main() -> None:
    """Compare three candidate prices using a hand-checkable example.

    用一个可以手算的例子比较三个候选价格。
    """

    # Fixed example / 固定例子
    total_order_flow = 10.0
    value_mean = 1.0
    investor_slope = 500.0
    pricing_error_weight = 0.1

    # Toy conditional belief after observing y_t=10:
    # v_t may be 0.8 or 1.2, each with probability 50%.
    # 观察 y_t=10 后的玩具判断：v_t 可能是 0.8 或 1.2，各有 50% 概率。
    possible_fundamental_values = [0.8, 1.2]
    conditional_probabilities = [0.5, 0.5]
    candidate_prices = [1.00, 1.02, 1.04]

    results: dict[float, dict[str, float]] = {}

    print("Step 6: Market-maker objective / 步骤 6：做市商目标函数")
    print(f"Observed total order flow y_t / 已观察总订单流: {total_order_flow:.2f}")

    for candidate_price in candidate_prices:
        result = evaluate_market_maker_objective(
            total_order_flow,
            candidate_price,
            value_mean,
            investor_slope,
            pricing_error_weight,
            possible_fundamental_values,
            conditional_probabilities,
        )
        results[candidate_price] = result

        print(f"\nCandidate price p_t / 候选价格: {candidate_price:.2f}")
        print(f"  Insensitive order z_t / 信息不敏感订单: {result['insensitive_order']:.2f}")
        print(f"  Net order y_t + z_t / 净订单: {result['net_order']:.2f}")
        print(f"  Inventory cost / 库存成本: {result['inventory_cost']:.5f}")
        print(f"  Expected squared pricing error / 预期价格误差平方: {result['expected_pricing_error']:.5f}")
        print(f"  Weighted pricing error / 加权价格误差: {result['weighted_pricing_error']:.5f}")
        print(f"  Total objective / 目标函数总值: {result['objective']:.5f}")

    # Hand-calculated expected results / 手算得到的期望结果
    assert isclose(results[1.00]["objective"], 100.00400, abs_tol=1e-10)
    assert isclose(results[1.02]["objective"], 0.00404, abs_tol=1e-10)
    assert isclose(results[1.04]["objective"], 100.00416, abs_tol=1e-10)

    # Among these three candidates, 1.02 has the smallest loss.
    # 在这三个候选价格中，1.02 的损失最小。
    best_candidate = min(
        candidate_prices,
        key=lambda price: results[price]["objective"],
    )
    assert isclose(best_candidate, 1.02, abs_tol=1e-12)

    print(f"\nBest of these three candidates / 三个候选中最好的是: {best_candidate:.2f}")
    print("Validation passed / 验证通过")
    print(
        "This compares only three prices; Step 7 derives the theoretical rule. / "
        "这里只比较了三个价格；Step 7 才会推导理论定价规则。"
    )


if __name__ == "__main__":
    main()
