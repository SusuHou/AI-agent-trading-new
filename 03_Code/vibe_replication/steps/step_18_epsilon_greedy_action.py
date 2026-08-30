"""Step 18: choose one action with the paper's epsilon-greedy rule.

步骤 18：使用论文的 epsilon-greedy 规则选择一个动作。

Run / 运行:
    py -3 steps/step_18_epsilon_greedy_action.py

Paper rule, equation (2.6) / 论文规则，方程 (2.6):

    with probability 1-epsilon: choose argmax_x Q(s,x)   (exploitation)
    with probability epsilon:   choose uniformly from X  (exploration)

    以 1-epsilon 的概率：选择 Q(s,x) 最大的动作（利用）
    以 epsilon 的概率：  从 X 中等概率随机选择（探索）

Explicit replication choice / 明确的复现选择:
The paper and online appendix do not say what to do when several actions have
exactly the same maximum Q-value.  We choose uniformly among the EXACT tied
maximizers.  This avoids systematically favoring a low action index.  It is an
implementation choice, not a rule attributed to the paper.

论文和在线附录没有说明多个动作的最大 Q 值完全相同时怎么办。我们的选择是：在
“精确并列”的最大动作中等概率随机选择，避免系统性偏向较小动作编号。这是实现选择，
不是论文明确规定的规则。

This step receives epsilon as an input.  Step 19 will calculate the paper's
state-dependent epsilon schedule.  This step does not update Q.

本步骤把 epsilon 当作输入。第 19 步才会计算论文的状态依赖探索率；本步骤不更新 Q。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
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


from step_12_action_grid import calculate_orders_for_value


@dataclass(frozen=True)
class ActionDecision:
    """The chosen Q-column and why it was chosen. / 被选中的 Q 列及选择模式。"""

    action_index: int
    mode: str  # "exploration" or "exploitation" / “探索”或“利用”


def maximizing_action_indexes(
    q_values_for_one_state: Sequence[float],
) -> tuple[int, ...]:
    """Return every action index with the exact maximum Q-value.

    返回所有“精确等于”最大 Q 值的动作编号。

    We deliberately do not use a tolerance: values that are merely close are
    not ties under mathematical argmax. / 这里故意不使用近似容差；数值接近不等于
    数学意义上的并列最大值。
    """

    q_values = [float(q_value) for q_value in q_values_for_one_state]
    if not q_values:
        raise ValueError("The Q-row cannot be empty. / Q 行不能为空。")
    if not all(isfinite(q_value) for q_value in q_values):
        raise ValueError("Every Q-value must be finite. / 每个 Q 值必须是有限数。")

    largest_q_value = max(q_values)
    return tuple(
        action_index
        for action_index, q_value in enumerate(q_values)
        if q_value == largest_q_value
    )


def choose_action_index_epsilon_greedy(
    q_values_for_one_state: Sequence[float],
    exploration_probability: float,
    random_generator: random.Random,
    action_random_generator: random.Random | None = None,
) -> ActionDecision:
    """Choose one action INDEX using equation (2.6).

    使用方程 (2.6) 选择一个动作“编号”。

    The returned index is a Q-table column.  After selection, Step 12 converts
    that stable index into the raw order x belonging to the current value v.

    返回值是 Q 表的列编号。选择完成后，第 12 步的公式再把稳定的动作编号转换为
    当前价值 v 对应的实际订单 x。
    """

    best_indexes = maximizing_action_indexes(q_values_for_one_state)

    if (
        not isfinite(exploration_probability)
        or not 0.0 <= exploration_probability <= 1.0
    ):
        raise ValueError("epsilon must lie in [0,1]. / epsilon 必须位于 [0,1]。")
    if not isinstance(random_generator, random.Random):
        raise TypeError(
            "random_generator must be random.Random. / "
            "random_generator 必须是 random.Random。"
        )
    if action_random_generator is None:
        # Backward-compatible educational mode: one stream supplies both draws.
        # / 向后兼容的教学模式：同一条随机流同时提供两类抽签。
        action_generator = random_generator
    elif isinstance(action_random_generator, random.Random):
        # Step 26 uses a separate stream so an action/tie draw cannot shift the
        # future explore-versus-exploit draws. / 第 26 步使用独立动作流，避免动作
        # 或并列抽签改变未来的探索/利用抽签位置。
        action_generator = action_random_generator
    else:
        raise TypeError(
            "action_random_generator must be random.Random or None. / "
            "action_random_generator 必须是 random.Random 或 None。"
        )

    # One random number decides exploration versus exploitation.
    # 一个 [0,1) 随机数决定探索还是利用。
    mode_draw = random_generator.random()
    if mode_draw < exploration_probability:
        # randrange makes every action INDEX equally likely. / 每个动作编号等概率。
        action_index = action_generator.randrange(
            len(q_values_for_one_state)
        )
        return ActionDecision(action_index, "exploration")

    # A unique maximum requires no additional random draw.  Only a true tie
    # invokes our explicit uniform tie rule. / 最大值唯一时不再抽随机数；只有真正
    # 并列时才使用我们明确规定的等概率并列规则。
    if len(best_indexes) == 1:
        action_index = best_indexes[0]
    else:
        action_index = action_generator.choice(best_indexes)
    return ActionDecision(action_index, "exploitation")


def main() -> None:
    """Force and validate both branches. / 强制触发并验证两个分支。"""

    toy_q_row = [1.0, 9.0, 2.0]
    toy_q_before = toy_q_row.copy()

    # epsilon=0 forces exploitation; the unique maximum is action 1.
    # epsilon=0 强制利用；唯一最大值位于动作 1。
    exploitation_decision = choose_action_index_epsilon_greedy(
        toy_q_row,
        exploration_probability=0.0,
        random_generator=random.Random(42),
    )
    assert exploitation_decision == ActionDecision(1, "exploitation")

    # epsilon=1 forces exploration. It may randomly pick the greedy action too;
    # the MODE is still exploration. / epsilon=1 强制探索。即使随机碰巧选中最佳动作，
    # 它的模式仍然是探索。
    exploration_decision = choose_action_index_epsilon_greedy(
        toy_q_row,
        exploration_probability=1.0,
        random_generator=random.Random(42),
    )
    assert exploration_decision == ActionDecision(0, "exploration")

    # Exact ties use our explicit uniform-random rule. / 精确并列使用明确的等概率规则。
    tied_q_row = [9.0, 9.0, 1.0]
    tie_generator = random.Random(20260828)
    tie_counts = [0, 0, 0]
    for _ in range(1_000):
        decision = choose_action_index_epsilon_greedy(
            tied_q_row,
            exploration_probability=0.0,
            random_generator=tie_generator,
        )
        assert decision.mode == "exploitation"
        assert decision.action_index in (0, 1)
        tie_counts[decision.action_index] += 1
    assert tie_counts[0] > 0 and tie_counts[1] > 0
    assert tie_counts[2] == 0

    # Values that are only close are not ties. / 仅仅接近的数值不算并列。
    near_tie_indexes = maximizing_action_indexes(
        [9.0, 9.0 - 1e-12, 1.0]
    )
    assert near_tie_indexes == (0,)

    # Exploration is uniform over indexes.  A fixed-seed smoke test must visit
    # all three indexes. / 探索对编号等概率；固定种子抽样必须覆盖三个编号。
    exploration_generator = random.Random(7)
    exploration_counts = [0, 0, 0]
    for _ in range(6_000):
        decision = choose_action_index_epsilon_greedy(
            toy_q_row,
            exploration_probability=1.0,
            random_generator=exploration_generator,
        )
        assert decision.mode == "exploration"
        exploration_counts[decision.action_index] += 1
    assert all(count > 0 for count in exploration_counts)

    # The same seed reproduces the complete action-and-mode sequence.
    # 相同种子会复现完整的动作与模式序列。
    first_generator = random.Random(123)
    second_generator = random.Random(123)
    first_sequence = [
        choose_action_index_epsilon_greedy(toy_q_row, 0.25, first_generator)
        for _ in range(100)
    ]
    second_sequence = [
        choose_action_index_epsilon_greedy(toy_q_row, 0.25, second_generator)
        for _ in range(100)
    ]
    assert first_sequence == second_sequence

    # Invalid probabilities and non-finite Q-values fail clearly.
    # 无效概率与非有限 Q 值必须明确报错。
    for invalid_epsilon in (-0.01, 1.01):
        try:
            choose_action_index_epsilon_greedy(
                toy_q_row,
                invalid_epsilon,
                random.Random(1),
            )
        except ValueError:
            invalid_epsilon_was_rejected = True
        else:
            invalid_epsilon_was_rejected = False
        assert invalid_epsilon_was_rejected

    try:
        choose_action_index_epsilon_greedy(
            [1.0, float("nan")],
            0.5,
            random.Random(1),
        )
    except ValueError:
        nonfinite_q_was_rejected = True
    else:
        nonfinite_q_was_rejected = False
    assert nonfinite_q_was_rejected

    # The function reads Q but never changes it. / 函数只读取 Q，不会修改 Q。
    assert toy_q_row == toy_q_before

    # Show how an action index becomes the current value's raw order.
    # 展示动作编号如何转换为当前价值对应的实际订单。
    toy_current_value = 1.2
    toy_value_mean = 1.0
    toy_action_multipliers = [1.0, 2.0, 3.0]
    toy_raw_orders = calculate_orders_for_value(
        toy_current_value,
        toy_value_mean,
        toy_action_multipliers,
    )
    exploitation_order = toy_raw_orders[
        exploitation_decision.action_index
    ]
    exploration_order = toy_raw_orders[
        exploration_decision.action_index
    ]

    print("Step 18: Epsilon-greedy action / 步骤 18：Epsilon-greedy 动作选择")
    print(f"Toy Q-row / 玩具 Q 行: {toy_q_row}")
    formatted_toy_orders = ", ".join(
        f"{order:.6f}" for order in toy_raw_orders
    )
    print(
        "Raw orders at v=1.2 / v=1.2 时的实际订单: "
        f"[{formatted_toy_orders}]"
    )

    print("\nForced exploitation / 强制利用 (epsilon=0):")
    print(f"  Decision / 决定: {exploitation_decision}")
    print(f"  Raw order x / 实际订单: {exploitation_order:.6f}")

    print("\nForced exploration / 强制探索 (epsilon=1, seed=42):")
    print(f"  Decision / 决定: {exploration_decision}")
    print(f"  Raw order x / 实际订单: {exploration_order:.6f}")

    print("\nExact-tie test / 精确并列测试:")
    print(f"  Tied Q-row / 并列 Q 行: {tied_q_row}")
    print(f"  Counts over 1,000 exploitations / 1,000 次利用计数: {tie_counts}")
    print(
        "  Only tied indexes 0 and 1 were selected. / "
        "只有并列的编号 0 与 1 被选中。"
    )

    print("\nUniform-exploration smoke test / 均匀探索抽样检查:")
    print(f"  Counts over 6,000 explorations / 6,000 次探索计数: {exploration_counts}")

    print(
        "\nReplication choice / 复现选择: exact argmax ties are broken "
        "uniformly at random. The paper does not specify this. / 最大 Q 精确并列时"
        "等概率随机选择；论文没有规定此细节。"
    )
    print(
        "Step 19 will calculate epsilon_t(v); this step only consumes epsilon. / "
        "第 19 步将计算 epsilon_t(v)；本步骤只使用给定的 epsilon。"
    )
    print("No Q-value was updated. / 没有更新任何 Q 值。")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
