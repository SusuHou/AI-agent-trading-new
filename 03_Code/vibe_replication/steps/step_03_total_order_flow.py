"""Step 3: calculate total informed-plus-noise order flow.
第三步：计算知情订单与噪声订单组成的总订单流。

Run / 运行:
    py -3 steps/step_03_total_order_flow.py
"""


def calculate_total_order_flow(informed_order_1, informed_order_2, noise_order):
    """Return y_t = x_1,t + x_2,t + u_t. / 返回总订单流。"""
    return informed_order_1 + informed_order_2 + noise_order


def main():
    """Run the standalone hand-checkable example. / 运行可手算的独立例子。"""
    informed_order_1 = 2.0
    informed_order_2 = -1.0
    noise_order = 0.5

    total_order_flow = calculate_total_order_flow(
        informed_order_1,
        informed_order_2,
        noise_order,
    )

    print(f"AI trader 1 order x_1,t / AI 1 订单: {informed_order_1}")
    print(f"AI trader 2 order x_2,t / AI 2 订单: {informed_order_2}")
    print(f"Noise-trader order u_t / 噪声订单: {noise_order}")
    print(f"Total order flow y_t / 总订单流: {total_order_flow}")

    assert total_order_flow == 1.5
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
