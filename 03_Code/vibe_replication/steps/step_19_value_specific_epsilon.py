"""Step 19: count visits to each value and calculate epsilon_t(v).

步骤 19：分别统计每个价值点的访问次数，并计算 epsilon_t(v)。

Run / 运行:
    py -3 steps/step_19_value_specific_epsilon.py

Paper rule, equation (4.3) / 论文规则，方程 (4.3):

    epsilon_t(v) = exp[-beta * t(v)]

``t(v)`` is the number of times THE SYSTEM visited value v in earlier
periods.  There is one shared list of ten value counters per simulation
session—not one list per trader.

``t(v)`` 表示整个系统在更早时期访问价值 v 的次数。每个模拟 session 共享一组
十个价值计数器，而不是每位交易者各有一组。

Correct one-period timing / 正确的单期顺序:

    epsilon = exploration_probability_for_value(v_index, counts, beta)
    trader 1 chooses using this epsilon and its own RNG
    trader 2 chooses using this SAME epsilon and its own RNG
    record_value_visit(v_index, counts)

First calculate epsilon from PAST visits.  Increase the selected counter once
only AFTER both traders choose.  Do not increase it separately for each trader.

先根据“过去访问次数”计算 epsilon。等两个交易者都选完动作后，才把当前价值的
计数增加一次。不要为每位交易者分别增加一次。
"""

from math import exp, isclose, isfinite


def initialize_value_visit_counts(number_of_values: int) -> list[int]:
    """Create one shared zero counter for every value. / 为每个价值建立共享的零计数器。"""

    if not isinstance(number_of_values, int) or number_of_values <= 0:
        raise ValueError("Number of values must be positive. / 价值数量必须是正整数。")
    return [0] * number_of_values


def calculate_exploration_probability(
    visit_count: int,
    exploration_decay: float,
) -> float:
    """Return epsilon=exp(-beta*n) for one value point.

    返回一个价值点的 epsilon=exp(-beta*n)。
    """

    if not isinstance(visit_count, int) or visit_count < 0:
        raise ValueError("Visit count must be a non-negative integer. / 访问次数必须是非负整数。")
    if not isfinite(exploration_decay) or exploration_decay <= 0.0:
        raise ValueError("beta must be positive and finite. / beta 必须是有限正数。")

    return exp(-exploration_decay * visit_count)


def exploration_probability_for_value(
    current_value_index: int,
    value_visit_counts: list[int],
    exploration_decay: float,
) -> float:
    """Calculate epsilon using this value's number of PAST visits.

    使用当前价值在过去的访问次数计算本期 epsilon。

    This function does not change the counter.  Therefore both informed
    traders can use the same epsilon before the visit is recorded. / 这个函数
    不修改计数器，因此两位知情交易者可以在记录本期访问前使用同一个 epsilon。
    """

    if not value_visit_counts:
        raise ValueError("The visit-counter list cannot be empty. / 访问计数器列表不能为空。")
    if not all(
        isinstance(count, int) and count >= 0
        for count in value_visit_counts
    ):
        raise ValueError("All visit counts must be non-negative integers. / 所有访问次数必须是非负整数。")
    if not isinstance(current_value_index, int):
        raise TypeError("Value index must be an integer. / 价值编号必须是整数。")
    if not 0 <= current_value_index < len(value_visit_counts):
        raise IndexError("Value index is outside V. / 价值编号超出 V。")

    return calculate_exploration_probability(
        value_visit_counts[current_value_index],
        exploration_decay,
    )


def record_value_visit(
    current_value_index: int,
    value_visit_counts: list[int],
) -> None:
    """Increase the system counter once after both traders act.

    两位交易者都选完动作后，把系统计数器增加一次。
    """

    # Reuse the validation above with a harmless beta value. / 用一个有效 beta
    # 复用上面的列表与编号检查；返回的 epsilon 在这里不需要。
    exploration_probability_for_value(
        current_value_index,
        value_visit_counts,
        exploration_decay=1.0,
    )
    value_visit_counts[current_value_index] += 1


def main() -> None:
    """Validate value-specific decay and exact timing. / 验证分价值衰减与时序。"""

    number_of_values = 10
    exploration_decay = 5e-7  # paper baseline beta / 论文基准 beta

    # Direct formula checks without running millions of periods.
    # 直接检查公式，不需要真的循环数百万期。
    epsilon_at_zero = calculate_exploration_probability(
        0,
        exploration_decay,
    )
    epsilon_after_one_visit = calculate_exploration_probability(
        1,
        exploration_decay,
    )
    epsilon_after_one_million = calculate_exploration_probability(
        1_000_000,
        exploration_decay,
    )
    epsilon_after_ten_million = calculate_exploration_probability(
        10_000_000,
        exploration_decay,
    )

    assert isclose(epsilon_at_zero, 1.0, abs_tol=0.0)
    assert isclose(
        epsilon_after_one_visit,
        exp(-5e-7),
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert isclose(
        epsilon_after_one_million,
        exp(-0.5),
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert isclose(
        epsilon_after_ten_million,
        exp(-5.0),
        rel_tol=0.0,
        abs_tol=1e-15,
    )

    # One shared system vector has ten separate value entries, all initially zero.
    # 全市场共享一个计数向量，其中十个价值位置最初都为零。
    visit_counts = initialize_value_visit_counts(number_of_values)
    assert visit_counts == [0] * number_of_values
    assert all(
        exploration_probability_for_value(index, visit_counts, exploration_decay)
        == 1.0
        for index in range(number_of_values)
    )
    selected_value_index = 3

    counts_before_first_period = visit_counts.copy()
    first_period_epsilon = exploration_probability_for_value(
        selected_value_index,
        visit_counts,
        exploration_decay,
    )
    # At this point both traders would independently choose an action using
    # first_period_epsilon. / 此时两个交易者会使用 first_period_epsilon 各自选动作。
    trader_1_epsilon = first_period_epsilon
    trader_2_epsilon = first_period_epsilon
    assert visit_counts == counts_before_first_period
    assert trader_1_epsilon == trader_2_epsilon == 1.0
    record_value_visit(selected_value_index, visit_counts)
    assert first_period_epsilon == 1.0
    assert counts_before_first_period == [0] * number_of_values
    assert visit_counts[selected_value_index] == 1
    assert sum(visit_counts) == 1

    # The second visit uses the previous count 1, then changes it to 2.
    # 第二次访问先使用旧计数 1，再把它更新为 2。
    second_period_epsilon = exploration_probability_for_value(
        selected_value_index,
        visit_counts,
        exploration_decay,
    )
    assert visit_counts[selected_value_index] == 1
    record_value_visit(selected_value_index, visit_counts)
    assert isclose(
        second_period_epsilon,
        epsilon_after_one_visit,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert visit_counts[selected_value_index] == 2
    assert sum(visit_counts) == 2

    # Visiting a different value changes only that counter. / 访问另一个价值点时，
    # 只改变该价值点自己的计数器。
    other_value_index = 7
    third_period_epsilon = exploration_probability_for_value(
        other_value_index,
        visit_counts,
        exploration_decay,
    )
    record_value_visit(other_value_index, visit_counts)
    assert third_period_epsilon == 1.0
    assert visit_counts[selected_value_index] == 2
    assert visit_counts[other_value_index] == 1
    assert sum(visit_counts) == 3
    assert all(
        count == 0
        for index, count in enumerate(visit_counts)
        if index not in (selected_value_index, other_value_index)
    )

    # State specificity: one heavily visited value can have low epsilon while
    # an unvisited value still has epsilon=1. / 一个价值点可以因访问很多次而具有较低
    # epsilon，同时未访问价值点仍保持 epsilon=1。
    comparison_counts = [0] * number_of_values
    comparison_counts[selected_value_index] = 1_000_000
    visited_value_epsilon = calculate_exploration_probability(
        comparison_counts[selected_value_index],
        exploration_decay,
    )
    unvisited_value_epsilon = calculate_exploration_probability(
        comparison_counts[0],
        exploration_decay,
    )
    assert isclose(visited_value_epsilon, exp(-0.5), abs_tol=1e-15)
    assert unvisited_value_epsilon == 1.0

    # Invalid indexes fail instead of changing a wrong counter.
    # 无效编号必须报错，不能修改错误的计数器。
    counts_before_invalid_call = visit_counts.copy()
    for invalid_index in (-1, number_of_values):
        try:
            exploration_probability_for_value(
                invalid_index,
                visit_counts,
                exploration_decay,
            )
        except IndexError:
            invalid_index_was_rejected = True
        else:
            invalid_index_was_rejected = False
        assert invalid_index_was_rejected
    assert visit_counts == counts_before_invalid_call

    print("Step 19: Value-specific epsilon / 步骤 19：分价值探索率")
    print(f"Paper beta / 论文 beta: {exploration_decay:.1e}")
    print("Number of shared counters / 全市场共享计数器数量: 10")

    print("\nFormula checkpoints / 公式检查点:")
    print(f"  t(v) =          0 -> epsilon = {epsilon_at_zero:.9f}")
    print(f"  t(v) =          1 -> epsilon = {epsilon_after_one_visit:.9f}")
    print(f"  t(v) =  1,000,000 -> epsilon = {epsilon_after_one_million:.9f}")
    print(f"  t(v) = 10,000,000 -> epsilon = {epsilon_after_ten_million:.9f}")

    print("\nThree-period counter example / 三期计数示例:")
    print("  Visited value indexes / 被访问的价值编号: 3, 3, 7")
    print(f"  Final counters / 最终计数器: {visit_counts}")
    print(
        "  First visit to index 3 used epsilon / 第一次访问编号 3 使用: "
        f"{first_period_epsilon:.9f}"
    )
    print(
        "  Second visit to index 3 used epsilon / 第二次访问编号 3 使用: "
        f"{second_period_epsilon:.9f}"
    )
    print(
        "  First visit to index 7 used epsilon / 第一次访问编号 7 使用: "
        f"{third_period_epsilon:.9f}"
    )

    print("\nState-specific comparison / 分价值比较:")
    print(
        "  Value 3 after 1,000,000 visits / 价值 3 访问一百万次后: "
        f"epsilon = {visited_value_epsilon:.9f}"
    )
    print(
        "  Unvisited value 0 / 尚未访问的价值 0: "
        f"epsilon = {unvisited_value_epsilon:.9f}"
    )

    print(
        "\nBoth traders use the same period epsilon, but draw their modes "
        "independently later. / 两位交易者使用同一个本期 epsilon，但之后独立抽取"
        "探索或利用模式。"
    )
    print(
        "Only after both choices, the counter increases once for the period. / "
        "两个交易者都选完后，计数器才为本期增加一次。"
    )
    print("No action was chosen and no Q-value was updated. / 没有选择动作，也没有更新 Q 值。")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
