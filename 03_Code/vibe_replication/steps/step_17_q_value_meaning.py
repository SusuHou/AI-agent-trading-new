"""Step 17: explain and validate what one Q-table cell means.

步骤 17：解释并验证 Q 表中一个格子的含义。

Run / 运行:
    py -3 steps/step_17_q_value_meaning.py

Paper equations / 论文方程:

    V_i(s) = max_x {E[pi_i|s,x] + rho E[V_i(s')|s,x]}             (2.1)

    Q_i(s, x) = E[pi_i | s,x] + rho E[V_i(s') | s,x]              (2.2)

    V_i(s) = max_x Q_i(s,x)                                       (definition)

    Q_i(s, x) = E[pi_i + rho max_x' Q_i(s',x') | s,x]             (2.3)

Beginner meaning / 初学者含义:
``Q(s,x)`` is the expected discounted profit from deliberately taking action
``x`` in state ``s`` now, then behaving optimally from the next state onward.
It is not a probability, price, order quantity, or one-period realized profit.

``Q(s,x)`` 表示：现在处于状态 ``s`` 时，先明确选择动作 ``x``，然后从下一状态
开始采用最优行为，由此得到的预期贴现利润。它不是概率、价格、订单量，也不只是
本期已经实现的利润。

In equation (2.3), ``x'`` is this same agent's possible NEXT action, not an
opponent action.  Step 16 created an estimated initial table ``Q_hat_0``; it
did not reveal the unknown true Q-function.

在方程 (2.3) 中，``x'`` 是同一智能体“下一期”的可能动作，不是对手动作。第 16 步
建立的是初始估计表 ``Q_hat_0``，并不代表我们已经知道真实 Q 函数。

This step reads Q-values only.  It does not choose an action and does not
change any table cell.

本步骤只读取 Q 值；不会选择动作，也不会修改任何格子。
"""

from collections.abc import Sequence
from math import isclose, isfinite
from pathlib import Path
import sys

import numpy as np


# Allow direct execution and later imports. / 同时支持直接运行与后续导入复用。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from src.step01_value_grid import build_value_grid, discrete_value_std
from step_10_fixed_point_solver import solve_benchmark_fixed_point
from step_12_action_grid import (
    build_action_multiplier_grid,
    calculate_orders_for_value,
)
from step_14_state_representation import (
    build_paper_price_grids,
    encode_state_index,
    number_of_states,
)
from step_16_initial_q_table import build_initial_q_table


def expected_q_from_components(
    expected_current_profit: float,
    discount_factor: float,
    expected_next_state_value: float,
) -> float:
    """Evaluate equation (2.2) after its expectations are known.

    在两个期望已经给定后，计算方程 (2.2)。

    Q(s,x) = expected current profit + rho * expected V(s')

    ``expected_next_state_value`` must mean E[V(s')]=E[max Q(s',x')].
    When several next states are possible, take the maximum INSIDE each next
    state first, and only then average across states. / 若有多个可能的下一状态，
    必须先在每个状态内部取最大 Q，再对不同状态求平均。
    """

    if not all(
        isfinite(number)
        for number in (expected_current_profit, expected_next_state_value)
    ):
        raise ValueError(
            "Expected values must be finite. / 期望值必须是有限数。"
        )
    if not 0.0 < discount_factor < 1.0:
        raise ValueError("rho must lie in (0,1). / rho 必须位于 (0,1) 内。")

    return (
        expected_current_profit
        + discount_factor * expected_next_state_value
    )


def state_value_from_q_row(q_values_for_one_state: Sequence[float]) -> float:
    """Return V(s)=max_x Q(s,x), without selecting an action index.

    返回 V(s)=max_x Q(s,x)，但不选择动作编号。

    Returning only the maximum VALUE deliberately postpones action selection
    and Q-value tie-breaking to Step 18.

    此函数只返回最大的“数值”，动作选择和 Q 值并列时如何处理留到第 18 步。
    """

    q_row = np.asarray(q_values_for_one_state, dtype=float)
    if q_row.ndim != 1 or q_row.size == 0:
        raise ValueError("A Q-row must be one non-empty row. / Q 行必须是一维且非空。")
    if not np.isfinite(q_row).all():
        raise ValueError("Q-values must be finite. / Q 值必须是有限数。")
    return float(np.max(q_row))


def expected_q_table_shape(
    number_of_prices: int,
    number_of_values: int,
    number_of_actions: int,
) -> tuple[int, int]:
    """Return the stored matrix shape (|S|, |X|). / 返回 Q 表的存储形状。"""

    if not isinstance(number_of_actions, int) or number_of_actions < 1:
        raise ValueError("n_x must be a positive integer. / n_x 必须是正整数。")
    return (
        number_of_states(number_of_prices, number_of_values),
        number_of_actions,
    )


def validate_q_table_dimensions(
    q_table: np.ndarray,
    number_of_prices: int,
    number_of_values: int,
    number_of_actions: int,
) -> None:
    """Require the exact paper-calibrated table layout. / 检查 Q 表布局。"""

    if not isinstance(q_table, np.ndarray):
        raise TypeError("q_table must be a NumPy array. / q_table 必须是 NumPy 数组。")

    required_shape = expected_q_table_shape(
        number_of_prices,
        number_of_values,
        number_of_actions,
    )
    if q_table.shape != required_shape:
        raise ValueError(
            f"Q-table shape must be {required_shape}, got {q_table.shape}. / "
            f"Q 表形状应为 {required_shape}，实际为 {q_table.shape}。"
        )
    if not np.isfinite(q_table).all():
        raise ValueError("Q-table must be finite. / Q 表必须只包含有限数。")


def q_value_at(
    q_table: np.ndarray,
    state_indexes: tuple[int, int, int],
    action_index: int,
    number_of_prices: int,
    number_of_values: int,
) -> float:
    """Read one scalar Q(s,x) using economic state indexes.

    使用经济状态的三个编号读取一个 Q(s,x) 标量。
    """

    if not isinstance(q_table, np.ndarray) or q_table.ndim != 2:
        raise ValueError("Q-table must be a 2-D array. / Q 表必须是二维数组。")
    if not isinstance(action_index, int):
        raise TypeError("Action index must be an integer. / 动作编号必须是整数。")
    if not 0 <= action_index < q_table.shape[1]:
        raise IndexError("Action index is outside X. / 动作编号超出 X。")

    state_id = encode_state_index(
        state_indexes,
        number_of_prices,
        number_of_values,
    )
    if state_id >= q_table.shape[0]:
        raise IndexError("State ID is outside the Q-table. / 状态编号超出 Q 表。")
    return float(q_table[state_id, action_index])


def q_table_as_tensor(
    q_table: np.ndarray,
    number_of_prices: int,
    number_of_values: int,
    number_of_actions: int,
) -> np.ndarray:
    """Show the same data as (P, previous V, current V, X).

    把同一组数据显示成（P、上期 V、本期 V、X）四个维度。

    This is a different VIEW of the same numbers, not another Q-table.
    / 这只是同一组数字的另一种查看方式，不是第二张 Q 表。
    """

    validate_q_table_dimensions(
        q_table,
        number_of_prices,
        number_of_values,
        number_of_actions,
    )
    return q_table.reshape(
        number_of_prices,
        number_of_values,
        number_of_values,
        number_of_actions,
    )


def main() -> None:
    """Connect the equations to the Step 16 table. / 把方程连接到第 16 步的表。"""

    # Hand-check equations (2.1)-(2.3) with two possible next states.
    # 使用两个可能的下一状态手算检查方程 (2.1)-(2.3)。
    toy_good_next_q_row = (8.0, 5.0)
    toy_bad_next_q_row = (2.0, 0.0)
    toy_good_next_value = state_value_from_q_row(toy_good_next_q_row)
    toy_bad_next_value = state_value_from_q_row(toy_bad_next_q_row)
    toy_expected_next_value = (
        0.5 * toy_good_next_value + 0.5 * toy_bad_next_value
    )
    # The toy aggressive action reaches good/bad with 50/50 probability;
    # the toy safe action reaches good with certainty. / 玩具例子中，激进动作以
    # 50/50 概率到达好/差状态，保守动作确定到达好状态。
    toy_aggressive_q = expected_q_from_components(
        expected_current_profit=4.0,
        discount_factor=0.5,
        expected_next_state_value=toy_expected_next_value,
    )
    toy_safe_q = expected_q_from_components(
        expected_current_profit=2.0,
        discount_factor=0.5,
        expected_next_state_value=toy_good_next_value,
    )
    toy_current_state_value = state_value_from_q_row(
        (toy_aggressive_q, toy_safe_q)
    )
    assert isclose(toy_good_next_value, 8.0, abs_tol=1e-12)
    assert isclose(toy_bad_next_value, 2.0, abs_tol=1e-12)
    assert isclose(toy_expected_next_value, 5.0, abs_tol=1e-12)
    assert isclose(toy_aggressive_q, 6.5, abs_tol=1e-12)
    assert isclose(toy_safe_q, 6.0, abs_tol=1e-12)
    assert isclose(toy_current_state_value, 6.5, abs_tol=1e-12)

    # Rebuild the actual low-noise initial table by reusing validated steps.
    # 重用已验证步骤，重新建立实际低噪声初始 Q 表。
    parameters = PaperParameters()
    value_grid = build_value_grid(
        parameters.value_mean,
        parameters.value_std,
        parameters.num_value_points,
    )
    discrete_fundamental_std = discrete_value_std(
        value_grid,
        parameters.value_mean,
    )
    nash_solution = solve_benchmark_fixed_point(
        "nash",
        parameters.num_speculators,
        parameters.noise_std,
        discrete_fundamental_std,
        parameters.investor_slope,
        parameters.pricing_error_weight,
    )
    cartel_solution = solve_benchmark_fixed_point(
        "cartel",
        parameters.num_speculators,
        parameters.noise_std,
        discrete_fundamental_std,
        parameters.investor_slope,
        parameters.pricing_error_weight,
    )
    action_multipliers = build_action_multiplier_grid(
        nash_solution["intensity"],
        cartel_solution["intensity"],
        parameters.grid_widening,
        parameters.num_action_points,
    )
    price_grids = build_paper_price_grids(
        parameters,
        value_grid,
        parameters.noise_std,
    )
    q_table = build_initial_q_table(
        price_grids,
        value_grid,
        action_multipliers,
        parameters.value_mean,
        parameters.num_speculators,
        nash_solution["price_impact"],
        parameters.discount_factor,
    )

    q_table_before_reading = q_table.copy()
    validate_q_table_dimensions(
        q_table,
        parameters.num_price_points,
        parameters.num_value_points,
        parameters.num_action_points,
    )
    q_tensor = q_table_as_tensor(
        q_table,
        parameters.num_price_points,
        parameters.num_value_points,
        parameters.num_action_points,
    )

    # Reuse Step 15's example: state tuple (20,1,0), flat ID 2010.
    # 重用第 15 步的示例：状态组合 (20,1,0)，单一编号 2010。
    example_state_indexes = (20, 1, 0)
    example_action_index = 7
    example_state_id = encode_state_index(
        example_state_indexes,
        parameters.num_price_points,
        parameters.num_value_points,
    )
    assert example_state_id == 2_010

    value_from_safe_lookup = q_value_at(
        q_table,
        example_state_indexes,
        example_action_index,
        parameters.num_price_points,
        parameters.num_value_points,
    )
    value_from_matrix = float(q_table[example_state_id, example_action_index])
    value_from_tensor = float(
        q_tensor[
            example_state_indexes[0],
            example_state_indexes[1],
            example_state_indexes[2],
            example_action_index,
        ]
    )
    assert isclose(value_from_safe_lookup, value_from_matrix, abs_tol=1e-12)
    assert isclose(value_from_safe_lookup, value_from_tensor, abs_tol=1e-12)

    example_q_row = q_table[example_state_id, :]
    example_state_value = state_value_from_q_row(example_q_row)
    example_current_value = float(value_grid[example_state_indexes[2]])
    example_orders = calculate_orders_for_value(
        example_current_value,
        parameters.value_mean,
        action_multipliers,
    )
    example_order = example_orders[example_action_index]

    # First and last legal cells work. Invalid action indexes fail instead of
    # using NumPy's negative-index shortcut. / 首尾合法格子可读取；无效动作编号必须
    # 报错，不能使用 NumPy 的负数编号捷径。
    q_value_at(
        q_table,
        (0, 0, 0),
        0,
        parameters.num_price_points,
        parameters.num_value_points,
    )
    q_value_at(
        q_table,
        (30, 9, 9),
        14,
        parameters.num_price_points,
        parameters.num_value_points,
    )
    for invalid_action_index in (-1, 15):
        try:
            q_value_at(
                q_table,
                example_state_indexes,
                invalid_action_index,
                parameters.num_price_points,
                parameters.num_value_points,
            )
        except IndexError:
            invalid_action_was_rejected = True
        else:
            invalid_action_was_rejected = False
        assert invalid_action_was_rejected

    try:
        validate_q_table_dimensions(
            q_table[:, :-1],
            parameters.num_price_points,
            parameters.num_value_points,
            parameters.num_action_points,
        )
    except ValueError:
        wrong_shape_was_rejected = True
    else:
        wrong_shape_was_rejected = False
    assert wrong_shape_was_rejected

    # Prove that every operation in this step was read-only.
    # 证明本步骤的所有操作都只读取，没有修改 Q 表。
    assert np.array_equal(q_table, q_table_before_reading)

    print("Step 17: Meaning of Q(s,x) / 步骤 17：Q(s,x) 的含义")
    print("\nHand Bellman example / Bellman 手算例子:")
    print(
        "  Good next-state Q row -> V(good) / 好状态 Q 行与价值: "
        f"{toy_good_next_q_row} -> {toy_good_next_value:.2f}"
    )
    print(
        "  Bad next-state Q row -> V(bad) / 差状态 Q 行与价值: "
        f"{toy_bad_next_q_row} -> {toy_bad_next_value:.2f}"
    )
    print(
        "  E[V(s')] with 50/50 outcomes / 下一状态价值的期望: "
        f"{toy_expected_next_value:.2f}"
    )
    print("  Aggressive Q = 4 + 0.5 x 5 / 激进动作 Q 值: 6.50")
    print("  Safe action reaches good state for sure. / 保守动作确定到达好状态。")
    print("  Safe Q = 2 + 0.5 x 8 / 保守动作 Q 值: 6.00")
    print("  V(s) = max(6.5, 6.0) / 当前状态价值: 6.50")

    print("\nQ-table dimensions / Q 表维度:")
    print(
        "  Economic view / 经济含义视图: "
        "31 previous prices x 10 previous values x "
        "10 current values x 15 actions"
    )
    print(f"  Stored matrix / 实际存储矩阵: {q_table.shape}")
    print(f"  Expanded view / 展开视图: {q_tensor.shape}")
    print(f"  Cells per trader / 每位交易者的格子数: {q_table.size}")
    print("  Row = one state; column = one action. / 行=状态；列=动作。")

    print("\nOne concrete lookup / 一个具体格子:")
    print(
        f"  State indexes {example_state_indexes} -> state ID {example_state_id} / "
        f"状态组合 {example_state_indexes} -> 状态编号 {example_state_id}"
    )
    print(f"  Action index / 动作编号: {example_action_index}")
    print(f"  Raw order x / 实际订单 x: {example_order:.9f}")
    print(f"  Q[2010,7] / 该格子的初始 Q 值: {value_from_safe_lookup:.9f}")
    print(
        "  Initial estimated V(s)=max Q row / 初始状态价值估计: "
        f"{example_state_value:.9f}"
    )

    print(
        "\nThe 2-D and 4-D lookups returned the same scalar. / "
        "二维与四维查看方式返回了同一个数。"
    )
    print(
        "Q is expected discounted profit, not a probability or current-period "
        "profit alone. / Q 是预期贴现利润，不是概率，也不只是当期利润。"
    )
    print(
        "Step 16's table is Q_hat_0, an initial estimate rather than the true "
        "Q-function. / 第 16 步的表是初始估计 Q_hat_0，不是真实 Q 函数。"
    )
    print(
        "No action was selected; no Q-value was updated. / "
        "没有选择动作，也没有更新任何 Q 值。"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
