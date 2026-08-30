"""Step 15: draw the simulation's initial market state uniformly.

步骤 15：从所有可能状态中均匀抽取模拟的初始市场状态。

Run / 运行:
    py -3 steps/step_15_initial_state.py

Paper rule / 论文规则:
    s_0 = (p_{-1}, v_{-1}, v_0) ~ Uniform(P x V x V)

Before period 0, no transaction has happened inside the simulation.  The
program nevertheless needs a previous price and previous value because they
are part of the agents' state.  The paper initializes this artificial starting
memory by selecting uniformly from the complete state space.

在第 0 期之前，模拟中还没有发生交易。但是智能体的状态需要“上一期价格”和
“上一期价值”，因此程序必须先建立一份人工的起始记忆。论文的做法是从完整状态
空间中均匀抽取一个状态。

This step draws the state only.  It does NOT create or update a Q-table.
本步骤只抽取初始状态，不会建立或更新 Q 表。
"""

from collections.abc import Sequence
from pathlib import Path
import random
import sys


# Allow direct execution and later imports. / 同时支持直接运行与后续导入复用。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from src.step01_value_grid import build_value_grid
from step_14_state_representation import (
    build_paper_price_grids,
    decode_state_index,
    encode_state_index,
    number_of_price_points,
    number_of_states,
)


def draw_initial_state_indexes(
    price_grids: Sequence[Sequence[float]],
    value_grid: Sequence[float],
    random_generator: random.Random,
) -> tuple[int, int, int]:
    """Draw (previous-price, previous-value, current-value) indexes.

    抽取（上期价格、上期价值、本期价值）三个编号。

    Think of this as rolling three fair dice: the first has one face per price
    in P, and the other two each have one face per value in V.  Because the
    three indexes are drawn independently, every complete state has the same
    probability.  Receiving the actual grids also prevents this function from
    accidentally using sizes that disagree with the selected environment.

    可以把它想象成掷三颗公平骰子：第一颗的面数等于 P 中的价格点数，另外两颗的
    面数等于 V 中的价值点数。三个编号独立抽取，因此每个完整状态的概率相同。
    函数直接接收实际网格，也可以防止网格大小与当前环境不一致。
    """

    if len(value_grid) < 1:
        raise ValueError("value_grid cannot be empty. / 价值网格不能为空。")
    if len(price_grids) != len(value_grid):
        raise ValueError(
            "There must be one price row per value. / 每个价值必须对应一行价格。"
        )
    if not isinstance(random_generator, random.Random):
        raise TypeError(
            "random_generator must be random.Random. / "
            "random_generator 必须是 random.Random。"
        )

    # Preserve RNG order: price, previous value, current value. / 保持随机抽样
    # 顺序：价格、上期价值、本期价值。
    previous_price_index = random_generator.randrange(
        number_of_price_points(price_grids)
    )
    previous_value_index = random_generator.randrange(len(value_grid))
    current_value_index = random_generator.randrange(len(value_grid))

    return (
        previous_price_index,
        previous_value_index,
        current_value_index,
    )


def main() -> None:
    """Draw one paper-calibrated state and validate it. / 抽取并验证一个状态。"""

    parameters = PaperParameters()
    random_seed = 42

    value_grid = build_value_grid(
        parameters.value_mean,
        parameters.value_std,
        parameters.num_value_points,
    )
    price_grids = build_paper_price_grids(
        parameters,
        value_grid,
        parameters.noise_std,
    )

    random_generator = random.Random(random_seed)
    initial_state_indexes = draw_initial_state_indexes(
        price_grids,
        value_grid,
        random_generator,
    )

    previous_price_index, previous_value_index, current_value_index = (
        initial_state_indexes
    )
    initial_state_values = (
        price_grids[previous_value_index][previous_price_index],
        value_grid[previous_value_index],
        value_grid[current_value_index],
    )
    initial_state_id = encode_state_index(
        initial_state_indexes,
        parameters.num_price_points,
        parameters.num_value_points,
    )

    total_states = number_of_states(
        parameters.num_price_points,
        parameters.num_value_points,
    )
    probability_of_each_state = 1.0 / total_states

    # Validation 1: the same seed reproduces the same first state.
    # 验证 1：相同随机种子会再次产生完全相同的第一个状态。
    repeated_state = draw_initial_state_indexes(
        price_grids,
        value_grid,
        random.Random(random_seed),
    )
    assert repeated_state == initial_state_indexes

    # Validation 2: every draw stays inside P x V x V.
    # 验证 2：每次抽取都必须位于 P x V x V 的合法范围内。
    validation_generator = random.Random(20260828)
    for _ in range(10_000):
        price_index, previous_index, current_index = draw_initial_state_indexes(
            price_grids,
            value_grid,
            validation_generator,
        )
        assert 0 <= price_index < parameters.num_price_points
        assert 0 <= previous_index < parameters.num_value_points
        assert 0 <= current_index < parameters.num_value_points

    # Validation 3: the one-number ID can be decoded without losing anything.
    # 验证 3：单一状态编号可以无损还原成原来的三个编号。
    assert decode_state_index(
        initial_state_id,
        parameters.num_price_points,
        parameters.num_value_points,
    ) == initial_state_indexes
    assert total_states == 3_100

    print("Step 15: Initial state / 步骤 15：初始状态")
    print(f"Random seed / 随机种子: {random_seed}")
    print(
        "Price-grid environment sigma_u / 价格网格对应的噪声环境: "
        f"{parameters.noise_std}"
    )
    print(
        "Paper rule / 论文规则: "
        "s_0 = (p_(-1), v_(-1), v_0) uniformly from P x V x V"
    )
    print(
        "Grid shape / 网格形状: "
        f"{parameters.num_price_points} x "
        f"{parameters.num_value_points} x "
        f"{parameters.num_value_points}"
    )
    print(f"Number of possible states / 可能状态总数: {total_states}")
    print(
        "Probability of this exact state / 任一特定状态的概率: "
        f"1/{total_states} = {probability_of_each_state:.8f} "
        f"= {100.0 * probability_of_each_state:.6f}%"
    )
    print("\nDrawn starting memory / 抽到的起始记忆:")
    print(
        "  Previous price / 上期价格: "
        f"P[{previous_price_index}] = {initial_state_values[0]:.9f}"
    )
    print(
        "  Previous value / 上期基本价值: "
        f"V[{previous_value_index}] = {initial_state_values[1]:.9f}"
    )
    print(
        "  Current value / 本期基本价值: "
        f"V[{current_value_index}] = {initial_state_values[2]:.9f}"
    )
    print(f"  State indexes / 状态编号组合: {initial_state_indexes}")
    print(f"  Flat state ID / 单一状态编号: {initial_state_id}")
    print(
        "\nSeed 42 is a reproducibility choice, not a paper parameter. / "
        "随机种子 42 是为了可复现，并不是论文参数。"
    )
    print(
        "This step selects existing P and V entries; it draws no continuous "
        "value or noise order. / 本步骤只选择已有的 P、V 网格点；不抽取连续价值或噪声订单。"
    )
    print("No Q-table has been created yet. / 目前尚未创建 Q 表。")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
