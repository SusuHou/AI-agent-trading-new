"""Step 21: create two independent informed Q-learning traders.

步骤 21：建立两位相互独立的知情 Q-learning 交易者。

Run / 运行:
    py -3 -X utf8 steps/step_21_two_independent_q_traders.py

Paper baseline / 论文基准:
    I = 2 informed AI speculators / 两位知情 AI 投机者

Private to each trader / 每位交易者各自拥有:
    - its own Q-table / 自己的 Q 表
    - its own random draws / 自己的随机抽签
    - its own chosen action and realized profit / 自己的动作与实际利润
    - its own Q update / 自己的 Q 更新

Shared by the market system / 整个市场系统共享:
    - the current state and fundamental value / 当前状态与基本价值
    - one epsilon calculated from one system visit-counter vector
      / 根据一组系统访问计数器算出的同一个 epsilon

The paper requires independent exploration decisions but does not specify a
seed-assignment scheme. We therefore inject explicit seeds. Step 26 now derives
stable named seeds and, for formal sessions, gives each trader one stream for
the mode draw and another for its random action or exact-tie draw.

论文要求两位交易者独立决定是否探索，但没有说明如何分配随机种子。因此这里明确
传入种子。第 26 步现在会稳定地产生命名种子；正式 session 中，每位交易者分别拥有
模式抽签流，以及随机动作/精确并列抽签流。

This step uses a small 3-state x 15-action table so every change is visible.
The final paper table supplied by Step 16 has 3,100 x 15 cells per trader.

本步骤使用 3 状态 x 15 动作的小表，方便看清每次变化。正式复现时，第 16 步会为
每位交易者提供 3,100 x 15 的 Q 表。

Code-design rule / 代码设计规则:
    - InformedQTrader is a normal mutable class because it learns and changes.
      / InformedQTrader 使用普通可变 class，因为它需要学习和改变。
    - QUpdateRecord is a frozen dataclass because a past event must not change.
      / QUpdateRecord 使用冻结 dataclass，因为已经发生的记录不应再改变。
"""

from dataclasses import FrozenInstanceError, dataclass
from math import isfinite
import random
from typing import Sequence

import numpy as np

from step_18_epsilon_greedy_action import (
    ActionDecision,
    choose_action_index_epsilon_greedy,
)
from step_19_value_specific_epsilon import (
    exploration_probability_for_value,
    initialize_value_visit_counts,
    record_value_visit,
)
from step_20_q_learning_update import (
    update_one_q_table_cell_expected_over_next_values,
)


class InformedQTrader:
    """One trader's private learning memory and random stream.

    一位交易者自己的学习记忆与随机数流。
    """

    def __init__(
        self,
        name: str,
        q_table: np.ndarray,
        random_seed: int,
        action_random_seed: int | None = None,
    ) -> None:
        """Validate inputs and create private mutable memory.

        检查输入，并建立该交易者自己的可变学习记忆。
        """

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Trader name cannot be empty. / 交易者名称不能为空。")
        if not isinstance(q_table, np.ndarray) or q_table.ndim != 2:
            raise TypeError("Initial Q-table must be a 2-D array. / 初始 Q 表必须是二维数组。")
        if q_table.shape[0] == 0 or q_table.shape[1] == 0:
            raise ValueError("Initial Q-table cannot be empty. / 初始 Q 表不能为空。")
        if not np.isfinite(q_table).all():
            raise ValueError("Initial Q-values must be finite. / 初始 Q 值必须是有限数。")
        if not isinstance(random_seed, int):
            raise TypeError("Random seed must be an integer. / 随机种子必须是整数。")
        if action_random_seed is not None and not isinstance(
            action_random_seed,
            int,
        ):
            raise TypeError("Action random seed must be an integer or None. / 动作随机种子必须是整数或 None。")
        if action_random_seed == random_seed:
            raise ValueError("Explicit mode and action seeds must differ. / 明确提供的模式与动作种子必须不同。")

        # These explicit assignments are the work that @dataclass previously
        # generated invisibly. / 这些明确赋值，就是 @dataclass 以前在背后自动
        # 生成的工作。
        self.name = name
        self.random_seed = random_seed
        self.mode_random_seed = random_seed
        self.action_random_seed = action_random_seed

        # copy=True is essential: otherwise two agents could silently share
        # and overwrite the same memory. / 必须复制，否则两位 AI 可能悄悄共享并
        # 覆盖同一块记忆。
        self.q_table = np.array(q_table, dtype=float, copy=True)
        # ``random_generator`` remains an alias for older steps. It is now the
        # mode stream that decides explore versus exploit. / 为兼容旧步骤，
        # random_generator 仍然保留；它现在代表探索/利用的模式随机流。
        self.random_generator = random.Random(random_seed)
        self.mode_random_generator = self.random_generator
        if action_random_seed is None:
            self.action_random_generator = self.mode_random_generator
        else:
            self.action_random_generator = random.Random(action_random_seed)

    def choose_action(
        self,
        current_state_id: int,
        epsilon: float,
    ) -> ActionDecision:
        """Read this trader's Q-row and independently choose. / 读取自己的 Q 行并独立选择。"""

        if not isinstance(current_state_id, int):
            raise TypeError("State ID must be an integer. / 状态编号必须是整数。")
        if not 0 <= current_state_id < self.q_table.shape[0]:
            raise IndexError("State ID is outside this Q-table. / 状态编号超出该 Q 表。")

        return choose_action_index_epsilon_greedy(
            self.q_table[current_state_id, :],
            epsilon,
            self.mode_random_generator,
            self.action_random_generator,
        )

    def learn_from_period(
        self,
        current_state_id: int,
        chosen_action_index: int,
        realized_profit: float,
        possible_next_state_ids: Sequence[int],
        learning_rate: float,
        discount_factor: float,
    ) -> float:
        """Update only this trader's visited cell. / 只更新该交易者被访问的格子。"""

        return update_one_q_table_cell_expected_over_next_values(
            self.q_table,
            current_state_id,
            chosen_action_index,
            realized_profit,
            possible_next_state_ids,
            learning_rate,
            discount_factor,
        )


@dataclass(frozen=True)
class QUpdateRecord:
    """An immutable receipt for one Q-table change.

    一次 Q 表变化的不可修改“流水单”。

    The mutable Q-table remains inside the trader. This record stores only the
    small facts needed to inspect that one update; it does not copy the full
    Q-table. / 可变 Q 表仍留在交易者内部。本记录只保存检查一次更新所需的少量
    信息，不复制整张 Q 表。
    """

    period_number: int
    trader_name: str
    state_id: int
    action_index: int
    decision_mode: str
    epsilon: float
    realized_profit: float
    old_q_value: float
    new_q_value: float


def build_two_informed_traders(
    initial_q_table: np.ndarray,
    random_seeds: tuple[int, int],
    action_random_seeds: tuple[int, int] | None = None,
) -> tuple[InformedQTrader, InformedQTrader]:
    """Create two private copies from the same initial values.

    根据同一组初始数值，为两位交易者分别复制独立 Q 表。
    """

    if not isinstance(random_seeds, tuple) or len(random_seeds) != 2:
        raise ValueError("Exactly two seeds are required. / 必须提供两个随机种子。")
    if random_seeds[0] == random_seeds[1]:
        raise ValueError("Use different seeds for independent draws. / 独立抽签应使用不同种子。")
    if action_random_seeds is not None:
        if not isinstance(action_random_seeds, tuple) or len(action_random_seeds) != 2:
            raise ValueError("Exactly two action seeds are required. / 必须提供两个动作种子。")
        all_explicit_seeds = (*random_seeds, *action_random_seeds)
        if not all(isinstance(seed, int) for seed in all_explicit_seeds):
            raise TypeError("All random seeds must be integers. / 所有随机种子都必须是整数。")
        if len(set(all_explicit_seeds)) != len(all_explicit_seeds):
            raise ValueError("Explicit mode and action seeds must all differ. / 明确提供的模式与动作种子必须全部不同。")

    trader_1 = InformedQTrader(
        "Trader 1 / 交易者 1",
        initial_q_table,
        random_seeds[0],
        None if action_random_seeds is None else action_random_seeds[0],
    )
    trader_2 = InformedQTrader(
        "Trader 2 / 交易者 2",
        initial_q_table,
        random_seeds[1],
        None if action_random_seeds is None else action_random_seeds[1],
    )
    return trader_1, trader_2


def choose_actions_for_one_shared_period(
    traders: tuple[InformedQTrader, InformedQTrader],
    current_state_id: int,
    current_value_index: int,
    shared_value_visit_counts: list[int],
    exploration_decay: float,
) -> tuple[float, tuple[ActionDecision, ActionDecision]]:
    """Give both traders one epsilon, then count one system visit.

    给两位交易者同一个 epsilon；两位都选择后，系统访问次数只加一。
    """

    if len(traders) != 2:
        raise ValueError("This paper baseline requires two traders. / 论文基准要求两位交易者。")
    if traders[0] is traders[1]:
        raise ValueError(
            "The two entries must be different trader objects. / "
            "两个位置必须是不同的交易者对象。"
        )
    if not isfinite(exploration_decay) or exploration_decay <= 0.0:
        raise ValueError("beta must be positive and finite. / beta 必须是有限正数。")

    epsilon = exploration_probability_for_value(
        current_value_index,
        shared_value_visit_counts,
        exploration_decay,
    )

    # Each method call uses that trader's private random generator. Python
    # evaluates the calls one after another, but the second trader receives no
    # information about the first decision, so the economic choices are
    # simultaneous. / 每次调用都使用该交易者自己的随机数生成器。Python 虽然
    # 依次执行两行，但第二位看不到第一位的决定，因此经济意义上仍是同时选择。
    decisions = (
        traders[0].choose_action(current_state_id, epsilon),
        traders[1].choose_action(current_state_id, epsilon),
    )

    # One market period contains one current value, regardless of two agents.
    # 一个市场期只有一个本期价值，不因有两位 AI 而计数两次。
    record_value_visit(current_value_index, shared_value_visit_counts)
    return epsilon, decisions


def main() -> None:
    """Prove private learning and shared system counting. / 验证私有学习与共享计数。"""

    period_number = 1
    number_of_actions = 15
    initial_q_table = np.zeros((3, number_of_actions), dtype=float)

    # Both traders begin with the same NUMBERS, as symmetric paper agents.
    # 两位对称交易者开始时拥有相同的数值。
    initial_q_table[0, 4] = 10.0
    initial_q_table[1, 4] = 20.0
    initial_q_table[2, 4] = 7.0

    traders = build_two_informed_traders(
        initial_q_table,
        random_seeds=(1, 2),
    )
    trader_1, trader_2 = traders

    # Same initial numbers, but separate memory and separate RNG objects.
    # 初始数值相同，但内存与随机数生成器互相独立。
    np.testing.assert_array_equal(trader_1.q_table, trader_2.q_table)
    assert not np.shares_memory(trader_1.q_table, initial_q_table)
    assert not np.shares_memory(trader_2.q_table, initial_q_table)
    assert not np.shares_memory(trader_1.q_table, trader_2.q_table)
    assert trader_1.random_generator is not trader_2.random_generator

    # Integration guard: accidentally passing the same object twice must fail
    # before either action is drawn. / 整合保护：若误把同一对象传入两次，必须在
    # 抽取任何动作之前明确报错。
    try:
        choose_actions_for_one_shared_period(
            (trader_1, trader_1),
            current_state_id=0,
            current_value_index=0,
            shared_value_visit_counts=initialize_value_visit_counts(10),
            exploration_decay=5e-7,
        )
    except ValueError:
        duplicate_trader_was_rejected = True
    else:
        duplicate_trader_was_rejected = False
    assert duplicate_trader_was_rejected

    shared_visit_counts = initialize_value_visit_counts(10)
    current_value_index = 3  # fourth value in V / V 中的第四个价值
    shared_visit_counts[current_value_index] = 1_000_000
    counter_before = shared_visit_counts[current_value_index]

    epsilon, decisions = choose_actions_for_one_shared_period(
        traders,
        current_state_id=0,
        current_value_index=current_value_index,
        shared_value_visit_counts=shared_visit_counts,
        exploration_decay=5e-7,
    )
    trader_1_decision, trader_2_decision = decisions

    assert abs(epsilon - 0.6065306597126334) < 1e-15
    assert trader_1_decision == ActionDecision(13, "exploration")
    assert trader_2_decision == ActionDecision(4, "exploitation")
    assert shared_visit_counts[current_value_index] == counter_before + 1
    assert sum(shared_visit_counts) == counter_before + 1

    trader_1_before = trader_1.q_table.copy()
    trader_2_before = trader_2.q_table.copy()
    possible_next_state_ids = (1, 2)
    learning_rate = 0.01
    discount_factor = 0.95

    # Profits are manually supplied here. The complete market will generate
    # them in Step 25. / 此处手工提供利润；第 25 步的完整市场会生成利润。
    trader_1_profit = 2.0
    trader_2_profit = -1.0

    # The mutable agents do not store an ever-growing history themselves.
    # The outside simulation owns the record list. / 可变 agent 自己不保存不断
    # 膨胀的历史；外部 simulation 负责保存记录列表。
    learning_history: list[QUpdateRecord] = []

    trader_1_old_q = float(
        trader_1.q_table[0, trader_1_decision.action_index]
    )

    trader_1_new_q = trader_1.learn_from_period(
        current_state_id=0,
        chosen_action_index=trader_1_decision.action_index,
        realized_profit=trader_1_profit,
        possible_next_state_ids=possible_next_state_ids,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
    )
    learning_history.append(
        QUpdateRecord(
            period_number=period_number,
            trader_name=trader_1.name,
            state_id=0,
            action_index=trader_1_decision.action_index,
            decision_mode=trader_1_decision.mode,
            epsilon=epsilon,
            realized_profit=trader_1_profit,
            old_q_value=trader_1_old_q,
            new_q_value=trader_1_new_q,
        )
    )

    # Updating Trader 1 must not touch Trader 2. / 更新交易者 1 不能影响交易者 2。
    np.testing.assert_array_equal(trader_2.q_table, trader_2_before)
    trader_1_after_own_update = trader_1.q_table.copy()

    trader_2_old_q = float(
        trader_2.q_table[0, trader_2_decision.action_index]
    )
    trader_2_new_q = trader_2.learn_from_period(
        current_state_id=0,
        chosen_action_index=trader_2_decision.action_index,
        realized_profit=trader_2_profit,
        possible_next_state_ids=possible_next_state_ids,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
    )
    learning_history.append(
        QUpdateRecord(
            period_number=period_number,
            trader_name=trader_2.name,
            state_id=0,
            action_index=trader_2_decision.action_index,
            decision_mode=trader_2_decision.mode,
            epsilon=epsilon,
            realized_profit=trader_2_profit,
            old_q_value=trader_2_old_q,
            new_q_value=trader_2_new_q,
        )
    )

    # Updating Trader 2 must not touch Trader 1. / 更新交易者 2 不能影响交易者 1。
    np.testing.assert_array_equal(
        trader_1.q_table,
        trader_1_after_own_update,
    )

    trader_1_changed = np.argwhere(trader_1.q_table != trader_1_before)
    trader_2_changed = np.argwhere(trader_2.q_table != trader_2_before)
    assert trader_1_changed.tolist() == [[0, 13]]
    assert trader_2_changed.tolist() == [[0, 4]]

    # Their common continuation is mean(max row 1, max row 2)=(20+7)/2=13.5.
    # 共同延续价值为 (20+7)/2=13.5。
    assert abs(trader_1_new_q - 0.14825) < 1e-12
    assert abs(trader_2_new_q - 10.01825) < 1e-12

    # The two mutable Q-tables changed, while both receipts preserved their
    # before-and-after values. / 两张可变 Q 表已经更新，但两张流水单永久保留了
    # 更新前后的数值。
    assert len(learning_history) == 2
    assert learning_history[0].old_q_value == 0.0
    assert learning_history[0].new_q_value == trader_1_new_q
    assert learning_history[1].old_q_value == 10.0
    assert learning_history[1].new_q_value == trader_2_new_q

    # frozen=True must reject an attempt to rewrite history. / frozen=True
    # 必须拒绝篡改已经保存的历史。
    try:
        setattr(learning_history[0], "new_q_value", 999.0)
    except FrozenInstanceError:
        record_is_frozen = True
    else:
        record_is_frozen = False
    assert record_is_frozen

    print("Step 21: Two independent Q-traders / 步骤 21：两位独立 Q 交易者")
    print(f"Shared epsilon / 共享 epsilon: {epsilon:.9f}")
    print(
        "Shared value counter / 共享价值计数: "
        f"{counter_before:,} -> {shared_visit_counts[current_value_index]:,}"
    )
    print(f"Trader 1 decision / 交易者 1 决定: {trader_1_decision}")
    print(f"Trader 2 decision / 交易者 2 决定: {trader_2_decision}")
    print(
        "Action index 13 means the 14th action-grid column, not an order of "
        "13 units. / 动作编号 13 表示动作网格第 14 列，不是交易 13 个单位。"
    )
    print(
        "Independent draws may still choose the same action by chance. / "
        "独立抽签仍可能碰巧选中相同动作。"
    )
    print("\nPrivate Q updates / 各自独立的 Q 更新:")
    print(
        "  Trader 1 changed / 交易者 1 改变: "
        f"{trader_1_changed.tolist()}, new Q = {trader_1_new_q:.5f}"
    )
    print(
        "  Trader 2 changed / 交易者 2 改变: "
        f"{trader_2_changed.tolist()}, new Q = {trader_2_new_q:.5f}"
    )
    print("\nImmutable update records / 不可修改的更新记录:")
    for record in learning_history:
        print(
            f"  {record.trader_name}: "
            f"Q[{record.state_id}, {record.action_index}] "
            f"{record.old_q_value:.5f} -> {record.new_q_value:.5f}, "
            f"profit = {record.realized_profit:.2f}"
        )
    print(
        "Each past record rejects edits; the outside list can still append new "
        "records. / 每条旧记录拒绝修改；外部列表仍可继续追加新记录。"
    )
    print(
        "Updating one table never changed the other table. / "
        "更新一张 Q 表从未改变另一张 Q 表。"
    )
    print(
        "The counter increased once, not once per trader. / "
        "共享计数只增加一次，不是每位交易者各增加一次。"
    )
    print("No market price or profit was generated here. / 此处尚未生成市场价格或利润。")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
