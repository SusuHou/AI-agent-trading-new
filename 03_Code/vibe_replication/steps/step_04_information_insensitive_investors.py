"""Step 4: calculate information-insensitive investors' aggregate order.
第四步：计算信息不敏感投资者的聚合订单。

Run / 运行:
    py -3 steps/step_04_information_insensitive_investors.py
"""


def calculate_insensitive_order(market_price, value_mean, investor_slope):
    """Return z_t = -xi * (p_t - v_bar). / 返回信息不敏感投资者订单。"""
    return -investor_slope * (market_price - value_mean)


def main():
    """Run the three standalone direction checks. / 运行三个独立方向检查。"""
    value_mean = 1.0
    investor_slope = 500.0

    order_when_high = calculate_insensitive_order(1.01, value_mean, investor_slope)
    order_when_low = calculate_insensitive_order(0.99, value_mean, investor_slope)
    order_at_mean = calculate_insensitive_order(1.00, value_mean, investor_slope)

    print(f"Price 1.01 -> z_t = {order_when_high:.1f} (sell / 卖出)")
    print(f"Price 0.99 -> z_t = {order_when_low:.1f} (buy / 买入)")
    print(f"Price 1.00 -> z_t = {order_at_mean + 0.0:.1f} (no order / 不交易)")

    assert abs(order_when_high - (-5.0)) < 0.000001
    assert abs(order_when_low - 5.0) < 0.000001
    assert order_at_mean == 0.0
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
