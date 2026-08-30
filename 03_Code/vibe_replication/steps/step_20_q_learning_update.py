"""Step 20: update exactly one Q-table cell with equation (2.4).

步骤 20：使用方程 (2.4)，准确更新 Q 表中的一个格子。

Run / 运行:
    py -3 -X utf8 steps/step_20_q_learning_update.py

Paper rule / 论文规则:

    Q_new(s_t, x_t)
        = (1-alpha) * Q_old(s_t, x_t)
          + alpha * [profit_t + rho * max_x' Q_old(s_(t+1), x')]

Beginner meaning / 初学者含义:

    new Q = mostly the old estimate
            + a small share of the new learning target

    新 Q = 大部分保留旧估计
           + 小部分吸收新的学习目标

The paper uses the REALIZED current profit and the REALIZED next state. Only
the cell belonging to the visited current state and chosen action changes.

论文使用已经实现的当期利润和实际到达的下一状态。只有“本期访问状态 + 已选动作”
对应的一个格子发生改变。

Appendix implementation detail / 附录实现细节:

For the reported experiments, Online Appendix 4.3 says the authors accelerate
learning by averaging the continuation value over the possible values of
v_(t+1), instead of using only the one realized v_(t+1). Because all ten value
grid points are equally likely, the continuation value used later will be:

    (1/n_v) * sum over v' in V of max_x' Q((p_t, v_t, v'), x')

论文报告的实验还采用了一个加速方法：不只使用本次抽到的 v_(t+1)，而是对下一期
所有可能基本价值对应的延续价值取平均。十个价值点等概率，因此稍后完整复现将使用
上面的平均值。本文件同时验证“基础方程版本”和“论文实验加速版本”。

This isolated step does not choose an action or generate the profit. Those
inputs will be connected to the market in later steps. / 本步骤不选择动作，也不
生成利润；它们将在后续步骤中与市场连接。
"""

from collections.abc import Sequence
from math import isfinite

import numpy as np

from step_17_q_value_meaning import state_value_from_q_row


def calculate_q_value_from_continuation(
    old_q_value: float,
    realized_profit: float,
    continuation_value: float,
    learning_rate: float,
    discount_factor: float,
) -> float:
    """Blend old knowledge with one supplied continuation value.

    使用一个已经算好的延续价值，把旧知识与新信息混合起来。
    """

    if not all(
        isfinite(number)
        for number in (old_q_value, realized_profit, continuation_value)
    ):
        raise ValueError(
            "Q, profit, and continuation must be finite. / "
            "Q 值、利润和延续价值必须是有限数。"
        )
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("alpha must lie in (0,1]. / alpha 必须位于 (0,1]。")
    if not 0.0 < discount_factor < 1.0:
        raise ValueError("rho must lie in (0,1). / rho 必须位于 (0,1)。")

    learning_target = (
        realized_profit + discount_factor * continuation_value
    )
    new_q_value = (
        (1.0 - learning_rate) * old_q_value
        + learning_rate * learning_target
    )
    if not isfinite(new_q_value):
        raise ValueError("The updated Q-value is not finite. / 更新后的 Q 值不是有限数。")
    return new_q_value


def calculate_updated_q_value(
    old_q_value: float,
    realized_profit: float,
    next_state_q_values: Sequence[float],
    learning_rate: float,
    discount_factor: float,
) -> float:
    """Calculate one new Q-value without changing a Q-table.

    只计算一个新的 Q 值，但暂时不修改 Q 表。
    """

    # Step 17 defines V(s_next)=max_x' Q(s_next,x').
    # 第 17 步定义 V(s_next)=max_x' Q(s_next,x')。
    best_next_q_value = state_value_from_q_row(next_state_q_values)
    return calculate_q_value_from_continuation(
        old_q_value,
        realized_profit,
        best_next_q_value,
        learning_rate,
        discount_factor,
    )


def expected_continuation_over_next_values(
    possible_next_state_q_rows: Sequence[Sequence[float]],
) -> float:
    """Average max Q across equally likely next-value states.

    对所有等概率的下一期价值状态：每行先取最大 Q，再对这些最大值求平均。

    The real paper grid supplies ten rows here. A smaller toy grid is allowed
    for a hand-check. / 正式论文网格会传入十行；手算测试可以使用较小的玩具网格。
    """

    q_rows = np.asarray(possible_next_state_q_rows, dtype=float)
    if q_rows.ndim != 2 or q_rows.shape[0] == 0 or q_rows.shape[1] == 0:
        raise ValueError(
            "Possible next Q-values must form a non-empty 2-D table. / "
            "可能的下一状态 Q 值必须构成非空二维表。"
        )
    if not np.isfinite(q_rows).all():
        raise ValueError("Possible next Q-values must be finite. / 下一状态 Q 值必须有限。")

    best_q_for_each_next_value = np.max(q_rows, axis=1)
    return float(np.mean(best_q_for_each_next_value))


def _validate_q_table_for_update(q_table: np.ndarray) -> None:
    """Check the shared requirements for either update version. / 检查两种更新的共同要求。"""

    if not isinstance(q_table, np.ndarray) or q_table.ndim != 2:
        raise TypeError("q_table must be a 2-D NumPy array. / Q 表必须是二维 NumPy 数组。")
    if q_table.shape[0] == 0 or q_table.shape[1] == 0:
        raise ValueError("q_table cannot be empty. / Q 表不能为空。")
    if not np.issubdtype(q_table.dtype, np.floating):
        raise TypeError("q_table must use floating-point numbers. / Q 表必须使用浮点数。")
    if not np.isfinite(q_table).all():
        raise ValueError("Every Q-value must be finite. / 每个 Q 值必须是有限数。")


def _validate_index(index: int, size: int, label: str) -> None:
    """Reject invalid indexes, including NumPy's negative shortcut. / 拒绝无效编号。"""

    if not isinstance(index, int):
        raise TypeError(f"{label} must be an integer. / {label} 必须是整数。")
    if not 0 <= index < size:
        raise IndexError(f"{label} is outside Q. / {label} 超出 Q 表。")


def update_one_q_table_cell(
    q_table: np.ndarray,
    current_state_id: int,
    chosen_action_index: int,
    realized_profit: float,
    next_state_id: int,
    learning_rate: float,
    discount_factor: float,
) -> float:
    """Mutate only Q[current_state_id, chosen_action_index].

    只修改 Q[current_state_id, chosen_action_index] 这一个格子。

    The function returns the new scalar so the caller can record or display
    it. / 函数同时返回新的标量，方便外部记录或显示。
    """

    _validate_q_table_for_update(q_table)
    _validate_index(current_state_id, q_table.shape[0], "Current state ID / 当前状态编号")
    _validate_index(next_state_id, q_table.shape[0], "Next state ID / 下一状态编号")
    _validate_index(chosen_action_index, q_table.shape[1], "Action index / 动作编号")

    old_q_value = float(q_table[current_state_id, chosen_action_index])

    # Copy the old next-state row BEFORE writing. This matters if the current
    # and next state happen to be identical. / 写入前先复制旧的下一状态 Q 行；当
    # 当前状态与下一状态相同时，这一点尤其重要。
    old_next_state_q_values = q_table[next_state_id, :].copy()
    new_q_value = calculate_updated_q_value(
        old_q_value,
        realized_profit,
        old_next_state_q_values,
        learning_rate,
        discount_factor,
    )

    # This is the one and only write in the Q-update function.
    # 这是整个 Q 更新函数中唯一一次写入。
    q_table[current_state_id, chosen_action_index] = new_q_value
    return new_q_value


def update_one_q_table_cell_expected_over_next_values(
    q_table: np.ndarray,
    current_state_id: int,
    chosen_action_index: int,
    realized_profit: float,
    possible_next_state_ids: Sequence[int],
    learning_rate: float,
    discount_factor: float,
) -> float:
    """Apply the appendix acceleration and change only the visited cell.

    使用附录的加速方法，并且仍然只修改被访问的一个格子。

    Later, ``possible_next_state_ids`` will contain ten states that share the
    same ``(p_t, v_t)`` and differ only in ``v_(t+1)``. / 稍后这里会传入十个状态：
    它们拥有相同的 ``(p_t, v_t)``，只在 ``v_(t+1)`` 上不同。
    """

    _validate_q_table_for_update(q_table)
    _validate_index(current_state_id, q_table.shape[0], "Current state ID / 当前状态编号")
    _validate_index(chosen_action_index, q_table.shape[1], "Action index / 动作编号")

    next_state_ids = list(possible_next_state_ids)
    if not next_state_ids:
        raise ValueError("Possible next states cannot be empty. / 可能的下一状态不能为空。")
    for state_id in next_state_ids:
        _validate_index(state_id, q_table.shape[0], "Possible next state ID / 可能的下一状态编号")
    if len(set(next_state_ids)) != len(next_state_ids):
        raise ValueError("Possible next states must be unique. / 可能的下一状态不能重复。")

    # Copy all old rows before the one write. / 在唯一一次写入之前复制所有旧行。
    old_possible_next_q_rows = q_table[next_state_ids, :].copy()
    expected_continuation = expected_continuation_over_next_values(
        old_possible_next_q_rows
    )
    old_q_value = float(q_table[current_state_id, chosen_action_index])
    new_q_value = calculate_q_value_from_continuation(
        old_q_value,
        realized_profit,
        expected_continuation,
        learning_rate,
        discount_factor,
    )

    q_table[current_state_id, chosen_action_index] = new_q_value
    return new_q_value


def main() -> None:
    """Validate equation (2.4) with a hand calculation. / 用手算验证方程 (2.4)。"""

    # Row = state, column = action. / 行代表状态，列代表动作。
    toy_q_table = np.array(
        [
            [10.0, 5.0, 0.0],
            [12.0, 20.0, 8.0],
        ],
        dtype=float,
    )
    q_table_before_update = toy_q_table.copy()

    current_state_id = 0
    chosen_action_index = 0
    next_state_id = 1
    realized_profit = 2.0
    learning_rate = 0.01       # paper baseline alpha / 论文基准 alpha
    discount_factor = 0.95     # paper baseline rho / 论文基准 rho

    old_q_value = float(
        toy_q_table[current_state_id, chosen_action_index]
    )
    best_next_q_value = float(np.max(toy_q_table[next_state_id, :]))
    learning_target = (
        realized_profit + discount_factor * best_next_q_value
    )
    old_knowledge_part = (1.0 - learning_rate) * old_q_value
    new_information_part = learning_rate * learning_target

    updated_q_value = update_one_q_table_cell(
        toy_q_table,
        current_state_id,
        chosen_action_index,
        realized_profit,
        next_state_id,
        learning_rate,
        discount_factor,
    )

    # Hand calculation / 手算:
    # target = 2 + 0.95*20 = 21
    # new Q  = 0.99*10 + 0.01*21 = 9.90 + 0.21 = 10.11
    assert abs(best_next_q_value - 20.0) < 1e-12
    assert abs(learning_target - 21.0) < 1e-12
    assert abs(old_knowledge_part - 9.90) < 1e-12
    assert abs(new_information_part - 0.21) < 1e-12
    assert abs(updated_q_value - 10.11) < 1e-12

    # Prove that exactly one cell changed. / 证明只有一个格子改变。
    changed_cells = np.argwhere(
        toy_q_table != q_table_before_update
    )
    assert changed_cells.tolist() == [[0, 0]]
    expected_q_table = q_table_before_update.copy()
    expected_q_table[0, 0] = 10.11
    np.testing.assert_allclose(
        toy_q_table,
        expected_q_table,
        rtol=0.0,
        atol=1e-12,
    )

    # Appendix acceleration: a tiny world with two possible next values.
    # 附录加速版本：使用只有两个下一期价值的玩具世界进行手算。
    accelerated_q_table = np.array(
        [
            [10.0, 5.0, 0.0],   # current state / 当前状态
            [12.0, 20.0, 8.0],  # possible next value A / 可能的下一价值 A
            [3.0, 7.0, 6.0],    # possible next value B / 可能的下一价值 B
        ],
        dtype=float,
    )
    accelerated_before = accelerated_q_table.copy()
    possible_next_state_ids = (1, 2)
    possible_best_q_values = np.max(
        accelerated_q_table[list(possible_next_state_ids), :],
        axis=1,
    )
    expected_continuation = expected_continuation_over_next_values(
        accelerated_q_table[list(possible_next_state_ids), :]
    )
    accelerated_target = (
        realized_profit + discount_factor * expected_continuation
    )
    accelerated_new_q = update_one_q_table_cell_expected_over_next_values(
        accelerated_q_table,
        current_state_id,
        chosen_action_index,
        realized_profit,
        possible_next_state_ids,
        learning_rate,
        discount_factor,
    )

    # max rows = [20, 7]; average continuation = 13.5
    # target = 2 + 0.95*13.5 = 14.825
    # new Q = 0.99*10 + 0.01*14.825 = 10.04825
    np.testing.assert_allclose(
        possible_best_q_values,
        np.array([20.0, 7.0]),
        rtol=0.0,
        atol=1e-12,
    )
    assert abs(expected_continuation - 13.5) < 1e-12
    assert abs(accelerated_target - 14.825) < 1e-12
    assert abs(accelerated_new_q - 10.04825) < 1e-12
    accelerated_changed_cells = np.argwhere(
        accelerated_q_table != accelerated_before
    )
    assert accelerated_changed_cells.tolist() == [[0, 0]]

    # The real grid has ten equally likely next values. / 正式网格有十个等概率下一价值。
    ten_next_q_rows = np.array(
        [
            [float(index), float(index + 10), -1.0]
            for index in range(10)
        ],
        dtype=float,
    )
    # Row maxima are 10,11,...,19, whose mean is 14.5.
    # 每行最大值为 10,11,...,19，其平均值为 14.5。
    assert abs(
        expected_continuation_over_next_values(ten_next_q_rows) - 14.5
    ) < 1e-12

    # Self-transition check: continuation must come from the OLD row.
    # 自我转移检查：延续价值必须来自更新前的旧 Q 行。
    self_transition_table = np.array([[10.0, 5.0, 0.0]], dtype=float)
    self_transition_new_q = update_one_q_table_cell(
        self_transition_table,
        current_state_id=0,
        chosen_action_index=0,
        realized_profit=2.0,
        next_state_id=0,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
    )
    # old max=10; target=2+0.95*10=11.5; new=0.99*10+0.01*11.5=10.015
    assert abs(self_transition_new_q - 10.015) < 1e-12

    print("Step 20: One Q-learning update / 步骤 20：一次 Q-learning 更新")
    print("\nA. Direct equation (2.4) / A. 方程 (2.4) 的直接版本")
    print(f"Q-table before / 更新前 Q 表:\n{q_table_before_update}")
    print("\nVisited cell / 被访问格子: Q[state 0, action 0]")
    print(f"Old Q / 旧 Q 值: {old_q_value:.2f}")
    print(f"Realized profit / 已实现利润: {realized_profit:.2f}")
    print(f"Next-state Q row / 下一状态 Q 行: {q_table_before_update[next_state_id]}")
    print(f"Best next Q / 下一状态最大 Q: {best_next_q_value:.2f}")
    print(
        "Learning target = profit + rho x best next Q / 学习目标: "
        f"{realized_profit:.2f} + {discount_factor:.2f} x "
        f"{best_next_q_value:.2f} = {learning_target:.2f}"
    )
    print(
        "Old-knowledge part = (1-alpha) x old Q / 旧知识部分: "
        f"{old_knowledge_part:.2f}"
    )
    print(
        "New-information part = alpha x target / 新信息部分: "
        f"{new_information_part:.2f}"
    )
    print(f"New Q / 新 Q 值: {updated_q_value:.2f}")
    print(f"\nQ-table after / 更新后 Q 表:\n{toy_q_table}")
    print(f"Changed cell indexes / 改变的格子编号: {changed_cells.tolist()}")
    print("Exactly one cell changed. / 只有一个格子发生改变。")

    print(
        "\nB. Appendix expected-v acceleration / "
        "B. 附录对下一价值取期望的加速版本"
    )
    print(
        "Best Q for each possible next value / "
        f"每个可能下一价值的最大 Q: {possible_best_q_values}"
    )
    print(
        "Expected continuation / 平均延续价值: "
        f"(20 + 7) / 2 = {expected_continuation:.3f}"
    )
    print(
        "Learning target / 学习目标: "
        f"2 + 0.95 x 13.5 = {accelerated_target:.3f}"
    )
    print(f"New Q / 新 Q 值: {accelerated_new_q:.5f}")
    print(
        "Changed cell indexes / 改变的格子编号: "
        f"{accelerated_changed_cells.tolist()}"
    )
    print(
        "The full paper grid will average ten next-value rows. / "
        "正式论文网格将对十个下一价值状态的 Q 行取平均。"
    )
    print(
        "We will use this appendix version for the reported-paper replication. / "
        "复现论文报告结果时，我们将使用这个附录版本。"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
