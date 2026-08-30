"""Step 16: initialize one informed speculator's Q-table.

步骤 16：初始化一位知情投机者的 Q 表。

Run / 运行:
    py -3 steps/step_16_initial_q_table.py

Paper rule / 论文规则:

    Q_i,0(s, x) = 1 / ((1-rho) n_x)
                    * sum over x_-i in X of
                      [v - (v_bar + lambda_N(x + (I-1)x_-i))] x

The table has one row for each state and one column for each action:

    3,100 states x 15 actions = 46,500 cells.

Q 表的每一行对应一个状态，每一列对应一个动作：

    3,100 个状态 x 15 个动作 = 46,500 个格子。

This is only an informed INITIAL GUESS.  No action is selected and no
Q-learning update occurs in this step.

这只是一个有经济依据的“初始猜测”。本步骤不选择动作，也不进行 Q-learning 更新。
"""

from collections.abc import Sequence
from math import fsum, isclose, isfinite
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
    decode_state_index,
    encode_state_index,
    number_of_price_points,
    number_of_states,
)


def calculate_initial_q_value(
    current_value: float,
    value_mean: float,
    own_order: float,
    other_order_choices: Sequence[float],
    number_of_speculators: int,
    nash_price_impact: float,
    discount_factor: float,
) -> float:
    """Calculate one cell of the paper's initial Q-table.

    计算论文初始 Q 表中的一个格子。

    For each possible opponent action, we calculate the current trader's
    one-period profit.  We average those profits because the opponent is
    assumed to choose uniformly, then divide by ``1-rho`` to convert the
    one-period average into the paper's discounted starting value.  The
    arguments ``own_order`` and ``other_order_choices`` must be RAW orders
    from the current value's row X(v), not the Step 12 multipliers.

    对每个可能的对手动作，先计算当前交易者的单期利润。因为初始化时假设对手等概率
    选择动作，所以先取平均；再除以 ``1-rho``，得到论文规定的贴现初始价值。
    ``own_order`` 与 ``other_order_choices`` 必须是当前价值对应的实际订单 X(v)，
    不能把第 12 步的乘数直接当成订单。

    In the baseline I=2, ``x_-i`` is simply the other trader's order.  The
    ``(I-1)`` term is retained exactly as printed in the paper.

    在基准设定 I=2 时，``x_-i`` 就是另一位交易者的订单。代码仍严格保留论文中的
    ``(I-1)`` 项。
    """

    if not all(
        isfinite(number)
        for number in (current_value, value_mean, own_order)
    ):
        raise ValueError(
            "Values and own order must be finite. / 价值和自己的订单必须为有限数。"
        )
    if not isinstance(number_of_speculators, int) or number_of_speculators < 1:
        raise ValueError("I must be a positive integer. / I 必须是正整数。")
    if nash_price_impact <= 0.0 or not isfinite(nash_price_impact):
        raise ValueError("lambda^N must be positive. / lambda^N 必须大于零。")
    if not 0.0 < discount_factor < 1.0:
        raise ValueError("rho must lie in (0,1). / rho 必须位于 (0,1) 内。")

    opponent_orders = [float(order) for order in other_order_choices]
    if not opponent_orders:
        raise ValueError("X cannot be empty. / 动作集合 X 不能为空。")
    if not all(isfinite(order) for order in opponent_orders):
        raise ValueError("Orders in X must be finite. / X 中的订单必须为有限数。")

    one_period_profits: list[float] = []
    for other_order in opponent_orders:
        total_informed_order = (
            own_order
            + (number_of_speculators - 1) * other_order
        )
        benchmark_price = (
            value_mean + nash_price_impact * total_informed_order
        )
        one_period_profit = (
            current_value - benchmark_price
        ) * own_order
        one_period_profits.append(one_period_profit)

    average_one_period_profit = (
        fsum(one_period_profits) / len(one_period_profits)
    )
    return average_one_period_profit / (1.0 - discount_factor)


def build_initial_q_table(
    price_grids: Sequence[Sequence[float]],
    value_grid: Sequence[float],
    action_multipliers: Sequence[float],
    value_mean: float,
    number_of_speculators: int,
    nash_price_impact: float,
    discount_factor: float,
) -> np.ndarray:
    """Return a Q-table with shape (number of states, number of actions).

    返回形状为（状态数量，动作数量）的 Q 表。

    The formula depends on the CURRENT value and action, but not on the
    previous price or previous value.  We therefore calculate the small
    ``current-value x action`` block once and copy it into all matching states.

    初始化公式依赖“本期价值”和“本期动作”，但不依赖上期价格或上期价值。因此先计算
    一个较小的“本期价值 x 动作”区块，再复制到所有拥有相同本期价值的状态中。
    """

    if len(price_grids) < 1:
        raise ValueError("P(v) cannot be empty. / 价格网格 P(v) 不能为空。")
    if len(value_grid) < 1:
        raise ValueError("V cannot be empty. / 价值网格 V 不能为空。")
    if len(action_multipliers) < 1:
        raise ValueError("X cannot be empty. / 动作网格 X 不能为空。")
    if not all(isfinite(float(value)) for value in value_grid):
        raise ValueError("V must contain finite values. / V 必须包含有限数值。")
    if not all(isfinite(float(item)) for item in action_multipliers):
        raise ValueError("Action multipliers must be finite. / 动作乘数必须为有限数。")

    number_of_prices = number_of_price_points(price_grids)
    number_of_values = len(value_grid)
    number_of_actions = len(action_multipliers)
    total_states = number_of_states(number_of_prices, number_of_values)

    # First calculate only the genuinely different Q-values.
    # 首先只计算真正不同的 Q 值。
    q_by_current_value_and_action = np.empty(
        (number_of_values, number_of_actions),
        dtype=float,
    )

    for current_value_index, current_value in enumerate(value_grid):
        raw_orders = calculate_orders_for_value(
            float(current_value),
            value_mean,
            list(action_multipliers),
        )
        # Keep Step 12's stable action-index order.  When v < v_bar, raw
        # orders are numerically reversed; sorting them would silently change
        # what each action column means. / 保留第 12 步稳定的动作编号。v < v_bar
        # 时实际订单的数值顺序会反向；重新排序会悄悄改变每一列动作的含义。
        for action_index, own_order in enumerate(raw_orders):
            q_by_current_value_and_action[
                current_value_index,
                action_index,
            ] = calculate_initial_q_value(
                float(current_value),
                value_mean,
                own_order,
                raw_orders,
                number_of_speculators,
                nash_price_impact,
                discount_factor,
            )

    # Expand the small block to all states.  Step 14 tells us which current-v
    # index belongs to each flat state ID. / 把小区块扩展到全部状态。第 14 步告诉
    # 我们每个单一状态编号对应哪个本期价值编号。
    q_table = np.empty((total_states, number_of_actions), dtype=float)
    for state_id in range(total_states):
        _, _, current_value_index = decode_state_index(
            state_id,
            number_of_prices,
            number_of_values,
        )
        q_table[state_id, :] = q_by_current_value_and_action[
            current_value_index,
            :,
        ]

    return q_table


def main() -> None:
    """Build the paper tables and run transparent checks. / 建表并透明验证。"""

    # A tiny example that can be calculated without any earlier step.
    # 一个不依赖前面步骤、可以完全手算的小例子。
    toy_q_value = calculate_initial_q_value(
        current_value=2.0,
        value_mean=1.0,
        own_order=1.0,
        other_order_choices=(1.0, 3.0),
        number_of_speculators=2,
        nash_price_impact=0.25,
        discount_factor=0.5,
    )
    # Opponent 1 gives profit 0.5; opponent 3 gives profit 0.0.
    # Average 0.25 divided by (1-rho)=0.5 gives Q=0.5.
    # 对手为 1 时利润 0.5，对手为 3 时利润 0；平均 0.25，再除以 0.5 得 0.5。
    assert isclose(toy_q_value, 0.5, abs_tol=1e-12)

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

    print("Step 16: Initial Q-table / 步骤 16：初始 Q 表")
    print("Tiny hand-check Q / 小型手算 Q 值: 0.500000 (passed / 通过)")
    print(
        "Discount multiplier 1/(1-rho) / 贴现乘数: "
        f"{1.0 / (1.0 - parameters.discount_factor):.1f}"
    )
    print(
        "One table per informed trader / 每位知情交易者一张表: "
        f"{parameters.num_price_points * parameters.num_value_points**2} "
        f"states x {parameters.num_action_points} actions"
    )

    noise_environments = (
        ("LOW NOISE / 低噪声", parameters.noise_std),
        ("HIGH NOISE / 高噪声", 100.0),
    )

    for environment_name, noise_std in noise_environments:
        nash_solution = solve_benchmark_fixed_point(
            "nash",
            parameters.num_speculators,
            noise_std,
            discrete_fundamental_std,
            parameters.investor_slope,
            parameters.pricing_error_weight,
        )
        cartel_solution = solve_benchmark_fixed_point(
            "cartel",
            parameters.num_speculators,
            noise_std,
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
            noise_std,
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

        expected_shape = (3_100, 15)
        assert q_table.shape == expected_shape
        assert q_table.size == 46_500
        assert np.isfinite(q_table).all()

        # The same current value must give identical initial Q rows even when
        # previous price and previous value differ. / 本期价值相同时，即使上期价格和
        # 上期价值不同，初始 Q 行也必须相同。
        current_value_index = 7
        first_history_state = encode_state_index(
            (0, 0, current_value_index),
            parameters.num_price_points,
            parameters.num_value_points,
        )
        different_history_state = encode_state_index(
            (
                parameters.num_price_points - 1,
                parameters.num_value_points - 1,
                current_value_index,
            ),
            parameters.num_price_points,
            parameters.num_value_points,
        )
        np.testing.assert_allclose(
            q_table[first_history_state],
            q_table[different_history_state],
            atol=1e-12,
            rtol=0.0,
        )

        # Symmetric value signals should have the same initial Q values at the
        # same action indexes. / 对称的价值信号在相同动作编号下应有相同初始 Q 值。
        low_value_state = encode_state_index(
            (0, 0, 2),
            parameters.num_price_points,
            parameters.num_value_points,
        )
        high_value_state = encode_state_index(
            (0, 0, 7),
            parameters.num_price_points,
            parameters.num_value_points,
        )
        np.testing.assert_allclose(
            q_table[low_value_state],
            q_table[high_value_state],
            atol=1e-10,
            rtol=0.0,
        )

        # Reuse Step 15's seed-42 example state to make the output concrete.
        # 重用第 15 步 seed=42 的示例状态，让输出更加具体。
        example_state_indexes = (20, 1, 0)
        example_state_id = encode_state_index(
            example_state_indexes,
            parameters.num_price_points,
            parameters.num_value_points,
        )
        assert example_state_id == 2_010
        example_value = float(value_grid[example_state_indexes[2]])
        example_orders = calculate_orders_for_value(
            example_value,
            parameters.value_mean,
            action_multipliers,
        )

        print(f"\n{environment_name}: sigma_u = {noise_std:.1f}")
        print(
            "  Nash lambda used by initialization / 初始化使用的纳什 lambda: "
            f"{nash_solution['price_impact']:.12f}"
        )
        print(f"  Q-table shape / Q 表形状: {q_table.shape}")
        print(f"  Number of cells / 格子总数: {q_table.size}")
        print(
            "  Q-value range / Q 值范围: "
            f"[{float(q_table.min()):.9f}, {float(q_table.max()):.9f}]"
        )
        print(
            "  Example state from Step 15 / 第 15 步示例状态: "
            f"indexes {example_state_indexes}, ID {example_state_id}"
        )
        print(
            "  Current fundamental in that state / 该状态的本期价值: "
            f"v = {example_value:.9f}"
        )
        print("  Three example action columns / 三个示例动作列:")
        for action_index in (0, 7, 14):
            print(
                f"    action {action_index:2d}: "
                f"x = {example_orders[action_index]:.9f}, "
                f"initial Q = {q_table[example_state_id, action_index]:.9f}"
            )

    print(
        "\nPrevious price and previous value do not enter the printed initial-Q "
        "formula, so matching current values repeat the same 15 Q numbers. / "
        "论文的初始 Q 公式不包含上期价格和上期价值，因此本期价值相同的状态会重复"
        "同一组 15 个 Q 值。"
    )
    print(
        "Noise is set to its mean zero here, but sigma_u and the insensitive "
        "investors still affect lambda^N through Step 10. / 此处噪声订单取均值零，"
        "但 sigma_u 与信息不敏感投资者仍通过第 10 步影响 lambda^N。"
    )
    print("No Q-learning update occurs yet. / 目前尚未进行 Q-learning 更新。")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
