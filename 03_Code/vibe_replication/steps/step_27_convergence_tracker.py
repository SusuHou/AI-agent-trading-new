"""Step 27: track convergence of the informed traders' greedy policies.

步骤 27：跟踪知情交易者的贪心策略是否收敛。

Run / 运行:
    py -3 -X utf8 steps/step_27_convergence_tracker.py

Paper rule / 论文规则:
    All informed AI speculators' optimal strategies must remain unchanged for
    1,000,000 consecutive periods in a session. / 一个 session 中，所有知情
    AI 投机者的最优策略必须连续 1,000,000 期保持不变。

What we compare / 我们比较什么:
    A policy is the best-action rule for every state, not the action randomly
    realized this period. Q-values may move without changing that rule. / 策略
    是每个状态下的最优动作规则，不是本期随机实现的动作；Q 值可以继续变化，
    只要最优动作规则不变，就不会重置计数器。

Explicit replication choices / 明确的复现选择:
    1. For an exact tie, store the complete set of maximizing actions. Randomly
       choosing a different member of the same set is not a policy change.
       / 精确并列时保存完整最优动作集合；在同一集合内随机选到另一个动作，
       不算策略变化。
    2. Save policy(Q_0) before period 0. After every completed Q update, an
       unchanged joint policy adds one to the streak; any change resets it to
       zero. The changed period is not counted as the first stable period.
       / 第 0 期前保存 policy(Q_0)。每期 Q 更新完成后比较：全部不变就加一，
       任一 agent 改变就归零；发生变化的当期不算第一期稳定期。

Efficiency / 效率:
    The complete policy is built once. One market period can update only the
    current state's Q-row for each trader, so later comparisons inspect only
    those rows. With two traders and the baseline grids this is 2 x 15 values,
    instead of 2 x 3,100 x 15 values per period. / 完整策略只初始化一次。以后
    每期每位 trader 只有当前状态的一行 Q 值可能变化，因此只需检查这两行。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from numbers import Integral
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from step_21_two_independent_q_traders import InformedQTrader
from step_25_one_market_period import build_paper_inputs
from step_26_reproducible_random_streams import (
    RandomizedMarketSession,
    build_randomized_paper_session,
)


PAPER_UNCHANGED_PERIODS = 1_000_000
MAX_MASK_ACTIONS = 64
TRACKER_STATE_SCHEMA_VERSION = "step27-training-tracker-state-v1"


def exact_maximizer_mask(q_row: np.ndarray) -> int:
    """Encode the row's complete exact-argmax set as bits in one integer.

    把这一行所有“精确并列最大”的动作编号编码成一个整数的二进制位。

    Example / 例子:
        q_row = [5, 5, 1] -> actions {0, 1} -> binary 011 -> integer 3

    A scalar loop is intentional: it creates no temporary Boolean array in the
    hot loop. / 这里有意使用标量循环，避免高频循环产生临时布尔数组。
    """

    if not isinstance(q_row, np.ndarray) or q_row.ndim != 1:
        raise TypeError("Q-row must be a one-dimensional NumPy array. / Q 行必须是一维 NumPy 数组。")
    number_of_actions = q_row.shape[0]
    if number_of_actions == 0:
        raise ValueError("Q-row cannot be empty. / Q 行不能为空。")
    if number_of_actions > MAX_MASK_ACTIONS:
        raise ValueError("This bit mask supports at most 64 actions. / 此位掩码最多支持 64 个动作。")

    best_value = float("-inf")
    mask = 0
    for action_index in range(number_of_actions):
        value = float(q_row[action_index])
        if not isfinite(value):
            raise ValueError("Every Q-value must be finite. / 每个 Q 值都必须是有限数。")
        if value > best_value:
            best_value = value
            mask = 1 << action_index
        elif value == best_value:
            mask |= 1 << action_index
    return mask


def _exact_maximizer_mask_at(
    q_table: np.ndarray,
    state_index: int,
) -> int:
    """Hot-loop form: scan one table row once without creating a row view.

    高频版本：不建立 Q 行切片，只直接扫描二维 Q 表中的一行一次。
    """

    best_value = float("-inf")
    mask = 0
    for action_index in range(q_table.shape[1]):
        value = float(q_table[state_index, action_index])
        if not isfinite(value):
            raise ValueError("Every Q-value must be finite. / 每个 Q 值都必须是有限数。")
        if value > best_value:
            best_value = value
            mask = 1 << action_index
        elif value == best_value:
            mask |= 1 << action_index
    return mask


def action_indexes_from_mask(mask: int, number_of_actions: int) -> tuple[int, ...]:
    """Decode a policy bit mask for readable tests and reporting.

    把策略位掩码还原为动作编号，供测试与输出阅读。
    """

    if isinstance(mask, bool) or not isinstance(mask, Integral) or mask <= 0:
        raise ValueError("mask must be a positive integer. / mask 必须是正整数。")
    if (
        isinstance(number_of_actions, bool)
        or not isinstance(number_of_actions, int)
        or not 1 <= number_of_actions <= MAX_MASK_ACTIONS
    ):
        raise ValueError("number_of_actions must lie in [1, 64]. / 动作数量必须位于 [1, 64]。")
    python_mask = int(mask)
    if python_mask >= 1 << number_of_actions:
        raise ValueError("mask refers to an unavailable action. / mask 指向了不存在的动作。")
    return tuple(
        action_index
        for action_index in range(number_of_actions)
        if python_mask & (1 << action_index)
    )


@dataclass(frozen=True)
class ConvergenceReceipt:
    """Immutable facts recorded once when the threshold is reached.

    达到阈值时只生成一次、之后不可修改的收敛记录。
    """

    required_unchanged_periods: int
    convergence_period_index: int
    training_periods_completed: int
    unchanged_run_first_comparison_period_index: int
    policy_change_events: int


@dataclass(frozen=True)
class PolicyConvergenceTrackerState:
    """Lossless immutable state for a not-yet-converged tracker.

    尚未收敛 tracker 的无损、不可修改状态。

    The policy masks are stored as bytes rather than a mutable NumPy array.
    / 策略 masks 保存为 bytes，而不是可被外部修改的 NumPy 数组。
    """

    schema_version: str
    required_unchanged_periods: int
    number_of_agents: int
    number_of_states: int
    number_of_actions: int
    policy_mask_shape: tuple[int, int]
    policy_mask_dtype: str
    policy_mask_bytes: bytes
    policy_mask_sha256: str
    full_policy_build_count: int
    full_policy_rows_scanned: int
    current_rows_checked: int
    periods_observed: int
    unchanged_periods: int
    policy_change_events: int
    policy_entries_changed: int
    last_policy_change_period_index: int | None


class PolicyConvergenceTracker:
    """Mutable online counter for all agents' joint greedy policy.

    对所有 agent 的联合贪心策略进行在线计数的可变 tracker。

    The tracker is mutable because its streak changes every period. The final
    receipt is frozen because a historical convergence event must not change.
    / tracker 必须可变，因为稳定期数每期都会变化；最终 receipt 使用 frozen，
    因为已经发生的收敛事件不应被修改。
    """

    def __init__(
        self,
        traders: Sequence[InformedQTrader],
        required_unchanged_periods: int = PAPER_UNCHANGED_PERIODS,
    ) -> None:
        if (
            isinstance(required_unchanged_periods, bool)
            or not isinstance(required_unchanged_periods, int)
            or required_unchanged_periods < 1
        ):
            raise ValueError("The convergence threshold must be a positive integer. / 收敛阈值必须是正整数。")
        if not isinstance(traders, Sequence) or len(traders) == 0:
            raise ValueError("At least one informed trader is required. / 至少需要一位知情交易者。")

        trader_references: list[InformedQTrader] = []
        q_table_references: list[np.ndarray] = []
        expected_shape: tuple[int, int] | None = None
        for trader in traders:
            if not isinstance(trader, InformedQTrader):
                raise TypeError("Every item must be an InformedQTrader. / 每个元素都必须是 InformedQTrader。")
            q_table = trader.q_table
            if not isinstance(q_table, np.ndarray) or q_table.ndim != 2:
                raise TypeError("Every Q-table must be two-dimensional. / 每张 Q 表都必须是二维数组。")
            if q_table.shape[0] == 0 or q_table.shape[1] == 0:
                raise ValueError("Q-tables cannot be empty. / Q 表不能为空。")
            if q_table.shape[1] > MAX_MASK_ACTIONS:
                raise ValueError("The tracker supports at most 64 actions. / tracker 最多支持 64 个动作。")
            if expected_shape is None:
                expected_shape = q_table.shape
            elif q_table.shape != expected_shape:
                raise ValueError("All traders must use the same Q-table shape. / 所有 trader 必须使用相同的 Q 表形状。")
            if not np.isfinite(q_table).all():
                raise ValueError("Every initial Q-value must be finite. / 所有初始 Q 值都必须是有限数。")
            if any(trader is existing for existing in trader_references):
                raise ValueError("Each trader object must be distinct. / 每个 trader 对象必须相互独立。")
            if any(
                np.shares_memory(q_table, existing_q_table)
                for existing_q_table in q_table_references
            ):
                raise ValueError("Traders cannot share Q-table memory. / 不同 trader 不能共享 Q 表内存。")
            trader_references.append(trader)
            q_table_references.append(q_table)

        if expected_shape is None:
            raise RuntimeError("The validated Q-table shape is missing. / 已验证的 Q 表形状丢失。")

        self.required_unchanged_periods = required_unchanged_periods
        self.number_of_agents = len(trader_references)
        self.number_of_states = expected_shape[0]
        self.number_of_actions = expected_shape[1]
        self._traders = tuple(trader_references)
        self._q_tables = tuple(q_table_references)

        # uint64 stores up to 64 action-membership bits in one fixed-size cell.
        # / 一个 uint64 格子最多保存 64 个动作的成员标记。
        self._policy_masks = np.empty(
            (self.number_of_agents, self.number_of_states),
            dtype=np.uint64,
        )
        self.full_policy_build_count = 0
        self.full_policy_rows_scanned = 0
        self.current_rows_checked = 0
        self._build_complete_policy_once()

        self.periods_observed = 0
        self.unchanged_periods = 0
        self.policy_change_events = 0
        self.policy_entries_changed = 0
        self.last_policy_change_period_index: int | None = None
        self.convergence_receipt: ConvergenceReceipt | None = None
        self.converged_policy_masks: np.ndarray | None = None
        self._attached_session: RandomizedMarketSession | None = None

    @classmethod
    def from_traders(
        cls,
        traders: Sequence[InformedQTrader],
        required_unchanged_periods: int = PAPER_UNCHANGED_PERIODS,
    ) -> "PolicyConvergenceTracker":
        """Readable factory used when attaching the tracker to a session.

        建立 tracker 的可读工厂函数。
        """

        return cls(traders, required_unchanged_periods)

    def export_training_state(self) -> PolicyConvergenceTrackerState:
        """Export every counter needed to continue convergence timing exactly.

        导出精确继续计算收敛时点所需的全部计数器。
        """

        if self.converged:
            raise RuntimeError(
                "Use the convergence receipt after convergence; this state is "
                "training-only. / 收敛后应使用 convergence receipt；本状态只用于训练期。"
            )
        mask_bytes = self._policy_masks.tobytes(order="C")
        return PolicyConvergenceTrackerState(
            schema_version=TRACKER_STATE_SCHEMA_VERSION,
            required_unchanged_periods=self.required_unchanged_periods,
            number_of_agents=self.number_of_agents,
            number_of_states=self.number_of_states,
            number_of_actions=self.number_of_actions,
            policy_mask_shape=tuple(int(size) for size in self._policy_masks.shape),
            policy_mask_dtype=self._policy_masks.dtype.str,
            policy_mask_bytes=mask_bytes,
            policy_mask_sha256=sha256(mask_bytes).hexdigest(),
            full_policy_build_count=self.full_policy_build_count,
            full_policy_rows_scanned=self.full_policy_rows_scanned,
            current_rows_checked=self.current_rows_checked,
            periods_observed=self.periods_observed,
            unchanged_periods=self.unchanged_periods,
            policy_change_events=self.policy_change_events,
            policy_entries_changed=self.policy_entries_changed,
            last_policy_change_period_index=self.last_policy_change_period_index,
        )

    @classmethod
    def from_training_state(
        cls,
        traders: Sequence[InformedQTrader],
        state: PolicyConvergenceTrackerState,
    ) -> "PolicyConvergenceTracker":
        """Rebuild a detached tracker and validate masks against live Q-values.

        重建一个尚未连接 session 的 tracker，并用实时 Q 值核对保存的 masks。
        """

        if not isinstance(state, PolicyConvergenceTrackerState):
            raise TypeError("state must be PolicyConvergenceTrackerState. / tracker state 类型错误。")
        if state.schema_version != TRACKER_STATE_SCHEMA_VERSION:
            raise ValueError("Tracker-state schema is unsupported. / tracker 状态格式不支持。")
        integer_fields = (
            state.required_unchanged_periods,
            state.number_of_agents,
            state.number_of_states,
            state.number_of_actions,
            state.full_policy_build_count,
            state.full_policy_rows_scanned,
            state.current_rows_checked,
            state.periods_observed,
            state.unchanged_periods,
            state.policy_change_events,
            state.policy_entries_changed,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise TypeError("Tracker counters must be integers. / tracker 计数器必须是整数。")
        if state.required_unchanged_periods < 1:
            raise ValueError("Saved convergence threshold must be positive. / 保存的收敛阈值必须为正。")
        expected_shape = (state.number_of_agents, state.number_of_states)
        if state.policy_mask_shape != expected_shape:
            raise ValueError("Saved policy-mask shape is inconsistent. / 保存的策略 mask 形状不一致。")
        try:
            dtype = np.dtype(state.policy_mask_dtype)
        except TypeError as error:
            raise ValueError("Saved policy-mask dtype is invalid. / 保存的策略 mask dtype 无效。") from error
        if dtype != np.dtype(np.uint64):
            raise TypeError("Policy masks must use uint64. / 策略 masks 必须使用 uint64。")
        expected_bytes = state.number_of_agents * state.number_of_states * dtype.itemsize
        if len(state.policy_mask_bytes) != expected_bytes:
            raise ValueError("Saved policy-mask byte length is wrong. / 保存的策略 mask 字节长度错误。")
        if sha256(state.policy_mask_bytes).hexdigest() != state.policy_mask_sha256:
            raise ValueError("Saved policy-mask checksum failed. / 保存的策略 mask 校验失败。")
        saved_masks = np.frombuffer(
            state.policy_mask_bytes,
            dtype=dtype,
        ).reshape(expected_shape, order="C")

        restored = cls(
            traders,
            required_unchanged_periods=state.required_unchanged_periods,
        )
        if (
            restored.number_of_agents != state.number_of_agents
            or restored.number_of_states != state.number_of_states
            or restored.number_of_actions != state.number_of_actions
        ):
            raise ValueError("Saved tracker dimensions differ from the Q-tables. / 保存的 tracker 维度与 Q 表不同。")
        if not np.array_equal(restored._policy_masks, saved_masks):
            raise ValueError(
                "Saved policy masks do not match the restored Q-table argmax sets. / "
                "保存的策略 masks 与恢复 Q 表的最优动作集合不同。"
            )

        if (
            state.full_policy_build_count != 1
            or state.full_policy_rows_scanned
            != state.number_of_agents * state.number_of_states
            or state.current_rows_checked
            != state.number_of_agents * state.periods_observed
        ):
            raise ValueError("Saved tracker scan accounting is inconsistent. / 保存的 tracker 扫描计数不一致。")
        if not 0 <= state.unchanged_periods < state.required_unchanged_periods:
            raise ValueError("Saved unchanged-policy streak is invalid. / 保存的策略不变 streak 无效。")
        if not 0 <= state.policy_change_events <= state.periods_observed:
            raise ValueError("Saved policy-change count is invalid. / 保存的策略变化次数无效。")
        if not 0 <= state.policy_entries_changed <= state.number_of_agents * state.policy_change_events:
            raise ValueError("Saved changed-entry count is invalid. / 保存的变化条目数无效。")
        if state.policy_change_events == 0:
            if state.policy_entries_changed != 0 or state.last_policy_change_period_index is not None:
                raise ValueError("Zero policy changes require empty change history. / 零次策略变化要求空变化历史。")
            if state.unchanged_periods != state.periods_observed:
                raise ValueError("Unchanged streak disagrees with zero-change history. / streak 与零变化历史不一致。")
        else:
            last = state.last_policy_change_period_index
            if isinstance(last, bool) or not isinstance(last, int) or not 0 <= last < state.periods_observed:
                raise ValueError("Last policy-change period is invalid. / 最后策略变化时期无效。")
            if state.policy_entries_changed < state.policy_change_events:
                raise ValueError("Every change event must change at least one policy entry. / 每次变化事件至少改变一个策略条目。")
            if state.unchanged_periods != state.periods_observed - last - 1:
                raise ValueError("Unchanged streak disagrees with the last change. / streak 与最后变化时期不一致。")

        restored.current_rows_checked = state.current_rows_checked
        restored.periods_observed = state.periods_observed
        restored.unchanged_periods = state.unchanged_periods
        restored.policy_change_events = state.policy_change_events
        restored.policy_entries_changed = state.policy_entries_changed
        restored.last_policy_change_period_index = state.last_policy_change_period_index
        return restored

    @property
    def converged(self) -> bool:
        """Whether the unchanged-policy threshold has been reached. / 是否已达到策略不变阈值。"""

        return self.convergence_receipt is not None

    def attach_to_session(self, session: RandomizedMarketSession) -> None:
        """Validate ownership before period 0, then attach safely.

        在第 0 期前先核对 tracker 与 session 是否属于同一组 traders，再安全连接。

        Use this method instead of calling the Step-26 hook directly. A wrong
        binding is rejected before any market state changes. / 请使用本方法，
        不要直接调用第 26 步接口；错误绑定会在市场状态改变前被拒绝。
        """

        if not isinstance(session, RandomizedMarketSession):
            raise TypeError("session must be a RandomizedMarketSession. / session 类型错误。")
        if self._attached_session is not None:
            raise RuntimeError("This tracker is already attached. / 此 tracker 已经连接。")
        if session.period_number != 0:
            raise RuntimeError("Attach the tracker before period 0. / 必须在第 0 期前连接 tracker。")
        if len(session.traders) != self.number_of_agents:
            raise ValueError("The tracker and session have different agent counts. / tracker 与 session 的 agent 数量不同。")
        for agent_index in range(self.number_of_agents):
            if session.traders[agent_index] is not self._traders[agent_index]:
                raise ValueError("The tracker belongs to different traders. / tracker 属于另一组 traders。")
            if session.traders[agent_index].q_table is not self._q_tables[agent_index]:
                raise ValueError("The tracker belongs to different Q-tables. / tracker 属于另一组 Q 表。")
        session.attach_after_q_update_observer(self.observe_after_q_update)
        self._attached_session = session

    def _build_complete_policy_once(self) -> None:
        """Read every state once before period 0. / 第 0 期前完整读取每个状态一次。"""

        if self.full_policy_build_count != 0:
            raise RuntimeError("The complete policy may be built only once. / 完整策略只能建立一次。")
        for agent_index, q_table in enumerate(self._q_tables):
            for state_index in range(self.number_of_states):
                self._policy_masks[agent_index, state_index] = np.uint64(
                    _exact_maximizer_mask_at(q_table, state_index)
                )
                self.full_policy_rows_scanned += 1
        self.full_policy_build_count = 1

    def observe_after_q_update(
        self,
        period_index: int,
        updated_state_index: int,
        traders: Sequence[InformedQTrader],
    ) -> None:
        """Compare only the Q-rows that could have changed this period.

        只比较本期可能发生变化的 Q 行。

        This signature matches Step 26's observer hook. It deliberately returns
        None and stores no period-by-period history. / 此参数顺序与第 26 步的
        observer 接口一致；它有意返回 None，也不保存逐期历史。
        """

        # Convergence is a latched event. Step 28 will switch the session into
        # its measurement phase beginning next period. / 收敛一旦命中就锁定；
        # 第 28 步会从下一期切换到测量阶段。
        if self.converged:
            return None
        if (
            isinstance(period_index, bool)
            or not isinstance(period_index, int)
            or period_index != self.periods_observed
        ):
            raise ValueError("Periods must be observed once in zero-based order. / 必须从 0 开始按顺序观察每一期，且每期一次。")
        if (
            isinstance(updated_state_index, bool)
            or not isinstance(updated_state_index, int)
            or not 0 <= updated_state_index < self.number_of_states
        ):
            raise IndexError("The updated state index is outside the Q-table. / 更新状态编号超出 Q 表范围。")
        if self._attached_session is None:
            if not isinstance(traders, Sequence) or len(traders) != self.number_of_agents:
                raise ValueError("The observer received the wrong number of traders. / observer 收到了错误数量的 trader。")
            for agent_index in range(self.number_of_agents):
                if traders[agent_index] is not self._traders[agent_index]:
                    raise ValueError("The observer must receive the original trader objects. / observer 必须收到原来的 trader 对象。")
                if traders[agent_index].q_table is not self._q_tables[agent_index]:
                    raise ValueError("A trader's Q-table object was unexpectedly replaced. / trader 的 Q 表对象被意外替换。")

        joint_policy_changed = False
        entries_changed_this_period = 0
        for agent_index in range(self.number_of_agents):
            new_mask = _exact_maximizer_mask_at(
                self._q_tables[agent_index],
                updated_state_index,
            )
            old_mask = int(
                self._policy_masks[agent_index, updated_state_index]
            )
            self.current_rows_checked += 1
            if new_mask != old_mask:
                self._policy_masks[agent_index, updated_state_index] = np.uint64(
                    new_mask
                )
                joint_policy_changed = True
                entries_changed_this_period += 1

        if joint_policy_changed:
            self.unchanged_periods = 0
            self.policy_change_events += 1
            self.policy_entries_changed += entries_changed_this_period
            self.last_policy_change_period_index = period_index
        else:
            self.unchanged_periods += 1

        self.periods_observed += 1
        if self.unchanged_periods == self.required_unchanged_periods:
            self.convergence_receipt = ConvergenceReceipt(
                required_unchanged_periods=self.required_unchanged_periods,
                convergence_period_index=period_index,
                training_periods_completed=period_index + 1,
                unchanged_run_first_comparison_period_index=(
                    period_index - self.required_unchanged_periods + 1
                ),
                policy_change_events=self.policy_change_events,
            )
            frozen_snapshot = self._policy_masks.copy()
            frozen_snapshot.flags.writeable = False
            self.converged_policy_masks = frozen_snapshot
        return None


def _assert_sessions_have_same_mutable_state(
    first: object,
    second: object,
) -> None:
    """Test helper: prove that observing did not alter the market path.

    测试辅助函数：证明只进行观察并没有改变市场路径。
    """

    assert first.period_number == second.period_number
    assert first.previous_price == second.previous_price
    assert first.previous_value == second.previous_value
    assert first.current_value == second.current_value
    assert first.shared_value_visit_counts == second.shared_value_visit_counts
    assert first.all_random_states() == second.all_random_states()
    assert first.market_maker.snapshot() == second.market_maker.snapshot()
    assert all(
        np.array_equal(left.q_table, right.q_table)
        for left, right in zip(first.traders, second.traders, strict=True)
    )


def main() -> None:
    """Run small hand-readable tests plus one real-session integration test.

    运行容易手工阅读的小测试，以及一个真实 session 整合测试。
    """

    # 1. The bit mask represents every exact maximizer. / 1. 位掩码表示全部精确最大动作。
    assert action_indexes_from_mask(
        exact_maximizer_mask(np.array([1.0, 5.0, 2.0])),
        3,
    ) == (1,)
    assert action_indexes_from_mask(
        exact_maximizer_mask(np.array([5.0, 5.0, 1.0])),
        3,
    ) == (0, 1)
    assert action_indexes_from_mask(
        exact_maximizer_mask(np.array([2.0, 2.0, 2.0])),
        3,
    ) == (0, 1, 2)
    assert action_indexes_from_mask(
        exact_maximizer_mask(np.array([5.0, 5.0 - 1e-12, 1.0])),
        3,
    ) == (0,)

    # 2. Known K=3 sequence: unchanged, change, then three unchanged periods.
    # / 2. 已知 K=3 序列：不变、变化，然后连续三期不变。
    toy_initial_q = np.array(
        [
            [3.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    toy_traders = (
        InformedQTrader("toy trader 1", toy_initial_q, 1),
        InformedQTrader("toy trader 2", toy_initial_q, 2),
    )
    tracker = PolicyConvergenceTracker.from_traders(
        toy_traders,
        required_unchanged_periods=3,
    )
    policy_storage_identity = id(tracker._policy_masks)
    toy_streaks: list[int] = []

    toy_traders[0].q_table[0, :] = [30.0, 2.0, 1.0]  # values move; best action stays 0 / 数值变但最优动作仍是 0
    assert tracker.observe_after_q_update(0, 0, toy_traders) is None
    toy_streaks.append(tracker.unchanged_periods)

    toy_traders[0].q_table[1, :] = [0.0, 5.0, 0.0]  # best action changes 0 -> 1 / 最优动作改变
    tracker.observe_after_q_update(1, 1, toy_traders)
    toy_streaks.append(tracker.unchanged_periods)

    tracker.observe_after_q_update(2, 2, toy_traders)
    toy_streaks.append(tracker.unchanged_periods)
    toy_traders[1].q_table[0, :] = [40.0, 3.0, 2.0]  # Q moves, policy does not / Q 变化但策略不变
    tracker.observe_after_q_update(3, 0, toy_traders)
    toy_streaks.append(tracker.unchanged_periods)
    tracker.observe_after_q_update(4, 2, toy_traders)
    toy_streaks.append(tracker.unchanged_periods)

    assert toy_streaks == [1, 0, 1, 2, 3]
    assert tracker.converged
    assert tracker.convergence_receipt == ConvergenceReceipt(
        required_unchanged_periods=3,
        convergence_period_index=4,
        training_periods_completed=5,
        unchanged_run_first_comparison_period_index=2,
        policy_change_events=1,
    )
    assert tracker.full_policy_build_count == 1
    assert tracker.full_policy_rows_scanned == 2 * 3
    assert tracker.current_rows_checked == 2 * 5
    assert id(tracker._policy_masks) == policy_storage_identity
    assert tracker.converged_policy_masks is not None
    assert not tracker.converged_policy_masks.flags.writeable
    assert action_indexes_from_mask(
        tracker.converged_policy_masks[0, 1],
        tracker.number_of_actions,
    ) == (1,)

    # Once hit, convergence is historical and stays latched. Step 28 will own
    # the phase switch. / 一旦命中，收敛就成为历史事实并保持锁定；阶段切换属于第 28 步。
    latched_receipt = tracker.convergence_receipt
    latched_policy = tracker.converged_policy_masks.copy()
    rows_checked_at_convergence = tracker.current_rows_checked
    toy_traders[0].q_table[0, :] = [0.0, 10.0, 0.0]
    tracker.observe_after_q_update(5, 0, toy_traders)
    assert tracker.convergence_receipt is latched_receipt
    assert np.array_equal(tracker.converged_policy_masks, latched_policy)
    assert tracker.current_rows_checked == rows_checked_at_convergence

    # 3. An unchanged exact-tie set is stable; shrinking or expanding it is a
    # policy change. / 3. 同一个精确并列集合是稳定的；集合缩小或扩大都算变化。
    tied_q = np.array([[5.0, 5.0, 1.0]], dtype=float)
    tied_traders = (
        InformedQTrader("tie trader 1", tied_q, 3),
        InformedQTrader("tie trader 2", tied_q, 4),
    )
    tie_tracker = PolicyConvergenceTracker.from_traders(
        tied_traders,
        required_unchanged_periods=10,
    )
    tied_traders[0].q_table[0, :] = [6.0, 6.0, 0.0]
    tied_traders[1].q_table[0, :] = [7.0, 7.0, 0.0]
    tie_tracker.observe_after_q_update(0, 0, tied_traders)
    assert tie_tracker.unchanged_periods == 1
    tied_traders[0].q_table[0, :] = [8.0, 7.0, 0.0]  # set {0,1} -> {0}
    tie_tracker.observe_after_q_update(1, 0, tied_traders)
    assert tie_tracker.unchanged_periods == 0
    assert tie_tracker.policy_change_events == 1
    tied_traders[0].q_table[0, :] = [9.0, 9.0, 0.0]  # expand agent 1 / agent 1 扩大集合
    tied_traders[1].q_table[0, :] = [6.0, 8.0, 0.0]  # shrink agent 2 / agent 2 缩小集合
    tie_tracker.observe_after_q_update(2, 0, tied_traders)
    assert tie_tracker.unchanged_periods == 0
    assert tie_tracker.policy_change_events == 2  # reset once, not twice / 本期只归零一次
    assert tie_tracker.policy_entries_changed == 3

    # 4. Boundary K=1: one unchanged transition converges; a changed one does
    # not. / 4. 边界 K=1：一次不变就收敛；第一次发生变化则不会收敛。
    boundary_q = np.array([[2.0, 1.0]], dtype=float)
    unchanged_traders = (
        InformedQTrader("boundary 1", boundary_q, 5),
        InformedQTrader("boundary 2", boundary_q, 6),
    )
    unchanged_tracker = PolicyConvergenceTracker.from_traders(
        unchanged_traders,
        required_unchanged_periods=1,
    )
    unchanged_tracker.observe_after_q_update(0, 0, unchanged_traders)
    assert unchanged_tracker.converged

    changed_traders = (
        InformedQTrader("changed 1", boundary_q, 7),
        InformedQTrader("changed 2", boundary_q, 8),
    )
    changed_tracker = PolicyConvergenceTracker.from_traders(
        changed_traders,
        required_unchanged_periods=1,
    )
    changed_traders[1].q_table[0, :] = [1.0, 3.0]
    changed_tracker.observe_after_q_update(0, 0, changed_traders)
    assert not changed_tracker.converged
    assert changed_tracker.unchanged_periods == 0
    changed_tracker.observe_after_q_update(1, 0, changed_traders)
    assert changed_tracker.converged
    assert changed_tracker.convergence_receipt is not None
    assert changed_tracker.convergence_receipt.convergence_period_index == 1

    # A change exactly where K=2 would otherwise be reached must reset to zero.
    # / 恰好在本可达到 K=2 的时期发生变化，仍必须归零。
    threshold_traders = (
        InformedQTrader("threshold 1", boundary_q, 9),
        InformedQTrader("threshold 2", boundary_q, 10),
    )
    threshold_tracker = PolicyConvergenceTracker.from_traders(
        threshold_traders,
        required_unchanged_periods=2,
    )
    threshold_tracker.observe_after_q_update(0, 0, threshold_traders)
    assert threshold_tracker.unchanged_periods == 1
    threshold_traders[0].q_table[0, :] = [1.0, 4.0]
    threshold_tracker.observe_after_q_update(1, 0, threshold_traders)
    assert threshold_tracker.unchanged_periods == 0
    assert not threshold_tracker.converged

    # Duplicate agents would not be independent and are rejected. The tracker
    # otherwise remains generic in I for later comparative experiments.
    # / 重复引用同一 agent 不具备独立性，因此拒绝；tracker 本身仍支持之后改变 I。
    duplicate_trader = InformedQTrader("duplicate", boundary_q, 11)
    try:
        PolicyConvergenceTracker.from_traders(
            (duplicate_trader, duplicate_trader),
            required_unchanged_periods=2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate traders should fail. / 重复 trader 应被拒绝。")

    # 5. Real Step-26 session: observing must not consume randomness or mutate
    # the economic path. / 5. 真实第 26 步 session：观察不能消耗随机数或改变经济路径。
    parameters = PaperParameters()
    (
        value_grid,
        price_grid,
        action_multipliers,
        initial_q_table,
        prehistory,
    ) = build_paper_inputs(parameters)
    session_arguments = dict(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        initial_q_table=initial_q_table,
        prehistory=prehistory,
        experiment_seed=20260828,
        experiment_cell_key="baseline_low_noise|A3=nash",
        session_index=0,
    )
    control_session = build_randomized_paper_session(**session_arguments)
    observed_session = build_randomized_paper_session(**session_arguments)
    paper_tracker = PolicyConvergenceTracker.from_traders(
        observed_session.traders,
        required_unchanged_periods=PAPER_UNCHANGED_PERIODS,
    )

    # A tracker for observed_session cannot be attached to control_session.
    # Rejection happens before period 0 and changes nothing. / 属于 observed_session
    # 的 tracker 不能接到 control_session；错误会在第 0 期前被拒绝且不改变状态。
    control_random_states_before = control_session.all_random_states()
    try:
        paper_tracker.attach_to_session(control_session)
    except ValueError:
        pass
    else:
        raise AssertionError("A foreign-session binding should fail. / 绑定错误 session 应失败。")
    assert control_session.period_number == 0
    assert control_session.after_q_update_observer is None
    assert control_session.all_random_states() == control_random_states_before

    paper_tracker.attach_to_session(observed_session)
    paper_policy_storage_identity = id(paper_tracker._policy_masks)
    integration_periods = 12
    for _ in range(integration_periods):
        control_session.run_next_random_period_without_trace()
        observed_session.run_next_random_period_without_trace()

    _assert_sessions_have_same_mutable_state(
        control_session,
        observed_session,
    )
    assert paper_tracker.periods_observed == integration_periods
    assert paper_tracker.full_policy_build_count == 1
    assert paper_tracker.full_policy_rows_scanned == 2 * 3_100
    assert paper_tracker.current_rows_checked == 2 * integration_periods
    assert id(paper_tracker._policy_masks) == paper_policy_storage_identity
    assert not paper_tracker.converged

    # The Step-26 hook intentionally rejects late attachment. / 第 26 步接口会故意拒绝中途连接。
    late_session = build_randomized_paper_session(**session_arguments)
    late_session.run_next_random_period_without_trace()
    late_tracker = PolicyConvergenceTracker.from_traders(
        late_session.traders,
        required_unchanged_periods=3,
    )
    try:
        late_tracker.attach_to_session(late_session)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Attaching after period 0 should fail. / 第 0 期后连接应失败。")

    full_values_per_period = (
        paper_tracker.number_of_agents
        * paper_tracker.number_of_states
        * paper_tracker.number_of_actions
    )
    checked_values_per_period = (
        paper_tracker.number_of_agents
        * paper_tracker.number_of_actions
    )

    print("Step 27: Policy convergence tracker / 步骤 27：策略收敛跟踪器")
    print(f"Paper threshold / 论文阈值: {PAPER_UNCHANGED_PERIODS:,} unchanged periods / 个不变时期")
    print("Compared object / 比较对象: full greedy policy / 全状态贪心策略")
    print("Tie-set handling / 并列集合处理: explicit replication choice / 明确复现选择")
    print(f"Toy streak after each period / 玩具例子每期稳定计数: {toy_streaks}")
    print(f"Toy convergence period index / 玩具例子收敛时期编号: {tracker.convergence_receipt.convergence_period_index}")
    print(f"Toy training periods completed / 玩具例子已完成训练期数: {tracker.convergence_receipt.training_periods_completed}")
    print(
        "Q changed without resetting at toy t=0 and t=3. / "
        "玩具例子的 t=0 与 t=3 中，Q 值变化但计数没有归零。"
    )
    print(
        f"Rows checked in each real period / 每个真实时期检查的 Q 行数: "
        f"{paper_tracker.number_of_agents}"
    )
    print(
        f"Q-values inspected: {checked_values_per_period:,} instead of "
        f"{full_values_per_period:,} per period / 每期检查 {checked_values_per_period:,} "
        f"而非 {full_values_per_period:,} 个 Q 值"
    )
    print(f"Real integration periods / 真实整合测试期数: {integration_periods}")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
