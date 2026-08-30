"""Integration Checkpoint A: connect the basic market formulas.

整合检查点 A：把已经分别验证过的基础市场公式连接起来。

This file does not introduce a new paper equation. It only proves that Steps 3–5
can work together with one fixed, hand-checkable example.

这个文件不加入新的论文公式。它只用一个固定、可以手算的例子，证明步骤 3–5
能够连接并一起运行。
"""

from step_03_total_order_flow import calculate_total_order_flow
from step_04_information_insensitive_investors import calculate_insensitive_order
from step_05_speculator_profit import calculate_profit


def main() -> None:
    """Run one fixed market example and validate every result.

    运行一个固定的市场例子，并验证每一个结果。
    """

    # Fixed inputs / 固定输入
    fundamental_value = 1.20
    market_price = 1.01
    informed_order_1 = 2.0
    informed_order_2 = -1.0
    noise_order = 0.5
    value_mean = 1.0
    investor_slope = 500.0

    # Step 3: y_t = x_1,t + x_2,t + u_t
    # 步骤 3：知情交易者订单加上噪声订单，得到总订单流。
    total_order_flow = calculate_total_order_flow(
        informed_order_1,
        informed_order_2,
        noise_order,
    )

    # Step 4: z_t = -xi * (p_t - v_bar)
    # 步骤 4：信息不敏感投资者看到价格后提交订单。
    insensitive_order = calculate_insensitive_order(
        market_price,
        value_mean,
        investor_slope,
    )

    # y_t + z_t is the net order the market maker cares about.
    # y_t + z_t 是做市商关心的市场净订单。
    combined_order = total_order_flow + insensitive_order

    # Step 5: pi_i,t = (v_t - p_t) * x_i,t
    # 步骤 5：分别计算两位知情交易者的利润。
    profit_1 = calculate_profit(
        fundamental_value,
        market_price,
        informed_order_1,
    )
    profit_2 = calculate_profit(
        fundamental_value,
        market_price,
        informed_order_2,
    )

    print("Integration Checkpoint A / 整合检查点 A")
    print(f"Total informed + noise order y_t / 总订单流: {total_order_flow:.2f}")
    print(f"Insensitive-investor order z_t / 信息不敏感投资者订单: {insensitive_order:.2f}")
    print(f"Combined order y_t + z_t / 市场净订单: {combined_order:.2f}")
    print(f"Trader 1 profit / 交易者 1 利润: {profit_1:.2f}")
    print(f"Trader 2 profit / 交易者 2 利润: {profit_2:.2f}")

    # These expected values can all be calculated by hand.
    # 下面的期望值全部都可以手算。
    assert abs(total_order_flow - 1.5) < 1e-12
    assert abs(insensitive_order - (-5.0)) < 1e-12
    assert abs(combined_order - (-3.5)) < 1e-12
    assert abs(profit_1 - 0.38) < 1e-12
    assert abs(profit_2 - (-0.19)) < 1e-12

    print("Validation passed / 验证通过")
    print(
        "Price is fixed for now; the market-maker price rule comes later. / "
        "目前价格是手动固定的；做市商定价规则将在后面的步骤加入。"
    )


if __name__ == "__main__":
    main()
