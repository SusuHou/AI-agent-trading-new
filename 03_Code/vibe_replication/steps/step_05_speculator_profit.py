"""Step 5: calculate one informed AI speculator's trading profit.
第五步：计算一个知情 AI 投机者的交易利润。

Run / 运行:
    py -3 steps/step_05_speculator_profit.py
"""


def calculate_profit(fundamental_value, market_price, order):
    """Return pi_i,t = (v_t - p_t) * x_i,t. / 返回投机者利润。"""
    return (fundamental_value - market_price) * order


def main():
    """Run the four standalone profit checks. / 运行四个独立利润检查。"""
    buy_underpriced = calculate_profit(1.20, 1.00, 2.0)
    buy_overpriced = calculate_profit(0.80, 1.00, 2.0)
    short_overpriced = calculate_profit(0.80, 1.00, -2.0)
    short_underpriced = calculate_profit(1.20, 1.00, -2.0)

    print(f"Buy underpriced / 买入低估资产: {buy_underpriced:.2f}")
    print(f"Buy overpriced / 买入高估资产: {buy_overpriced:.2f}")
    print(f"Short overpriced / 卖空高估资产: {short_overpriced:.2f}")
    print(f"Short underpriced / 卖空低估资产: {short_underpriced:.2f}")

    assert abs(buy_underpriced - 0.40) < 0.000001
    assert abs(buy_overpriced - (-0.40)) < 0.000001
    assert abs(short_overpriced - 0.40) < 0.000001
    assert abs(short_underpriced - (-0.40)) < 0.000001
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
