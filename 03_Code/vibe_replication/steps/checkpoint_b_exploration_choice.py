"""Checkpoint B: connect Step 18 and Step 19 for one market period.

整合检查点 B：把第 18 步和第 19 步连接成一个市场期。

Run / 运行:
    py -3 -X utf8 steps/checkpoint_b_exploration_choice.py

This checkpoint performs only four operations / 本检查点只做四件事:

    current value index
    -> calculate epsilon from this value's past-visit counter
    -> two traders independently choose actions using the same epsilon
    -> increase the current value's counter once

    当前价值编号
    -> 根据该价值的过去访问次数计算 epsilon
    -> 两位交易者使用同一个 epsilon，各自独立选择动作
    -> 当前价值的计数只增加一次

It does not calculate orders, prices, profits, or update Q-values yet.
这里暂时不计算订单、价格、利润，也不更新 Q 值。
"""

import random

from step_18_epsilon_greedy_action import (
    choose_action_index_epsilon_greedy,
)
from step_19_value_specific_epsilon import (
    exploration_probability_for_value,
    initialize_value_visit_counts,
    record_value_visit,
)


def main() -> None:
    """Run one transparent integration example. / 运行一个透明的整合示例。"""

    number_of_values = 10       # Ten possible v values / 十个可能的 v
    number_of_actions = 15      # Fifteen possible actions / 十五个可能动作
    beta = 5e-7                 # Paper baseline / 论文基准值

    # Python indexes start at zero. Index 3 means the FOURTH value in V.
    # Python 编号从零开始，所以 index 3 代表 V 中的第四个价值。
    current_value_index = 3

    # This ONE shared list contains one counter for each of the ten values.
    # 这个全市场共享的列表，为十个价值分别保存一个出现次数。
    value_visit_counts = initialize_value_visit_counts(number_of_values)

    # Testing shortcut: pretend value index 3 appeared 1,000,000 times before.
    # 测试捷径：假设价值编号 3 过去已经出现过 1,000,000 次。
    value_visit_counts[current_value_index] = 1_000_000
    counter_before_period = value_visit_counts[current_value_index]

    # Step 19 supplies epsilon. It reads the counter but does not change it.
    # 第 19 步提供 epsilon：读取计数，但暂时不修改计数。
    epsilon = exploration_probability_for_value(
        current_value_index,
        value_visit_counts,
        beta,
    )
    assert abs(epsilon - 0.6065306597126334) < 1e-15
    assert value_visit_counts[current_value_index] == counter_before_period

    # Each row has 15 Q-values: one estimated long-term value for each action.
    # 每行有 15 个 Q 值：每个动作对应一个长期价值估计。
    trader_1_q_row = [0.0] * number_of_actions
    trader_2_q_row = [0.0] * number_of_actions
    trader_1_q_row[4] = 10.0   # Trader 1's current best action / AI 1 当前最佳动作
    trader_2_q_row[9] = 10.0   # Trader 2's current best action / AI 2 当前最佳动作

    # Step 18 consumes the SAME epsilon, but each trader has an independent RNG.
    # 第 18 步使用同一个 epsilon，但两位交易者各自独立抽签。
    trader_1_decision = choose_action_index_epsilon_greedy(
        trader_1_q_row,
        epsilon,
        random.Random(1),
    )
    trader_2_decision = choose_action_index_epsilon_greedy(
        trader_2_q_row,
        epsilon,
        random.Random(2),
    )

    # With these fixed seeds, one explores and the other exploits. This proves
    # that sharing epsilon does not mean sharing the random result.
    # 使用这两个固定种子，一位探索、一位利用。这证明共享 epsilon 不等于共享抽签结果。
    assert trader_1_decision.mode == "exploration"
    assert trader_2_decision.mode == "exploitation"
    assert value_visit_counts[current_value_index] == counter_before_period

    # Only after BOTH decisions, record this system visit ONCE.
    # 只有两位都选完动作后，系统才把本期访问记录一次。
    record_value_visit(current_value_index, value_visit_counts)
    counter_after_period = value_visit_counts[current_value_index]
    assert counter_after_period == counter_before_period + 1
    assert sum(value_visit_counts) == counter_after_period

    print("Checkpoint B: Steps 18 + 19 / 整合检查点 B：第 18 + 19 步")
    print(
        "Current value index / 当前价值编号: "
        f"{current_value_index} (the fourth value / 第四个价值)"
    )
    print(
        "Counter before this period / 本期之前的出现次数: "
        f"{counter_before_period:,}"
    )
    print(f"Shared epsilon / 两位共享的 epsilon: {epsilon:.9f}")
    print(f"Trader 1 decision / AI 1 的决定: {trader_1_decision}")
    print(f"Trader 2 decision / AI 2 的决定: {trader_2_decision}")
    print(
        "Counter after both decisions / 两位都决定之后的出现次数: "
        f"{counter_after_period:,}"
    )
    print(
        "The counter rose by ONE, not by two. / "
        "计数只增加了一次，不是两次。"
    )
    print("No Q-value was updated. / 没有更新 Q 值。")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
