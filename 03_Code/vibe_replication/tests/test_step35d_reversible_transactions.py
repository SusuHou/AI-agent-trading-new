"""Focused tests for Step-35D reversible short-path transactions.

第 35D 步“可逆短路径事务”的针对性自动测试。

These tests validate only the reusable rollback foundation. They do not run or
claim the paper's 10,000 IRF paths. / 这些测试只验证可重复使用的回滚底层；它们
不会运行或声称已经完成论文的一万条 IRF 路径。
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = PROJECT_ROOT / "steps"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from step_22_market_maker_rolling_history import (  # noqa: E402
    MarketMakerHistory,
    MarketObservation,
)
from step_24b_fast_rolling_ols import RollingMarketMakerOLS  # noqa: E402
from steps.step_35a_converged_market_checkpoint import (  # noqa: E402
    restore_detached_frozen_branch,
)
from steps.step_35b_paired_irf_path import _build_demo_checkpoint  # noqa: E402


def _row(index: int) -> MarketObservation:
    """Create one varying, finite row for hand-sized OLS tests. / 建立一条有限且有变化的测试行。"""

    return MarketObservation(
        fundamental_value_v=0.8 + 0.07 * index,
        market_price_p=0.95 + 0.013 * index,
        insensitive_order_z=2.0 - 0.4 * index,
        informed_and_noise_order_y=-1.5 + 0.6 * index,
    )


def _session_snapshot(session: object) -> tuple[object, ...]:
    """Capture exact causal state for before/after comparisons.

    保存精确因果状态，用于回滚前后逐项比较。
    """

    return (
        session.period_number,
        session.previous_price,
        session.previous_value,
        session.current_value,
        session.execution_mode,
        session.frozen_draw_source_mode,
        tuple(session.shared_value_visit_counts),
        session.all_random_states(),
        session.market_maker.export_state(),
        session.frozen_policy_action_indexes_snapshot().tobytes(order="C"),
        tuple(
            (
                trader.q_table.tobytes(order="C"),
                bool(trader.q_table.flags.writeable),
            )
            for trader in session.traders
        ),
    )


class TestHistoryRollbackHelper(unittest.TestCase):
    """Validate the private O(k) deque reversal. / 验证私有 O(k) deque 反向恢复。"""

    def test_two_full_window_appends_are_reversed_exactly(self) -> None:
        """[a,b,c] -> [c,x,y] -> [a,b,c]. / 手算两次追加再回滚。"""

        history = MarketMakerHistory(window_size=3)
        original = (_row(0), _row(1), _row(2))
        for row in original:
            self.assertIsNone(history.append(row))

        appended = (_row(3), _row(4))
        evicted = tuple(history.append(row) for row in appended)
        self.assertEqual(evicted, original[:2])
        self.assertEqual(history.snapshot(), (original[2], *appended))

        history._rollback_full_window_appends(
            appended_rows=appended,
            evicted_rows=evicted,  # type: ignore[arg-type]
        )
        self.assertEqual(history.snapshot(), original)

    def test_wrong_suffix_is_rejected_before_any_mutation(self) -> None:
        """A stale log cannot partially alter the deque. / 过期日志不能只改一半 deque。"""

        history = MarketMakerHistory(window_size=3)
        original = (_row(0), _row(1), _row(2))
        for row in original:
            history.append(row)
        actual_new = _row(3)
        evicted = history.append(actual_new)
        before_rejection = history.snapshot()

        with self.assertRaisesRegex(RuntimeError, "suffix"):
            history._rollback_full_window_appends(
                appended_rows=(_row(99),),
                evicted_rows=(evicted,),  # type: ignore[arg-type]
            )
        self.assertEqual(history.snapshot(), before_rejection)


class TestRollingMakerTransaction(unittest.TestCase):
    """Validate bounded append logging and exact OLS rollback. / 验证追加上限、日志与 OLS 精确回滚。"""

    @staticmethod
    def _full_maker() -> RollingMarketMakerOLS:
        """Build a five-row maker whose next path crosses a rebuild boundary.

        建立五行做市商，使下一条短路径跨过一次重同步边界。
        """

        maker = RollingMarketMakerOLS(
            window_size=5,
            resynchronize_every=4,
        )
        for index in range(5):
            maker.append_completed_observation(_row(index))
        return maker

    def test_rollback_restores_rows_statistics_phase_and_counters(self) -> None:
        """Exact exported state survives three appends and a resynchronization.

        三次追加并跨过重同步后，导出的全部状态仍能精确恢复。
        """

        maker = self._full_maker()
        exact_start = maker.export_state()
        token = maker.begin_reversible_append_transaction(max_appends=3)
        for index in range(5, 8):
            maker.append_completed_observation(_row(index))

        self.assertEqual(maker.reversible_append_count(token), 3)
        self.assertNotEqual(maker.export_state(), exact_start)
        self.assertEqual(maker.rollback_reversible_append_transaction(token), 3)
        self.assertEqual(maker.export_state(), exact_start)

        # A token is single-use; accepting it twice could undo a later path.
        # token 只能使用一次；若能重复使用，可能误撤销后来的路径。
        with self.assertRaisesRegex(RuntimeError, "stale or foreign"):
            maker.rollback_reversible_append_transaction(token)

    def test_append_limit_rejects_atomically_then_allows_rollback(self) -> None:
        """The third append cannot enter a transaction capped at two.

        上限为两次时，第三次追加不能进入事务。
        """

        maker = self._full_maker()
        exact_start = maker.export_state()
        token = maker.begin_reversible_append_transaction(max_appends=2)
        maker.append_completed_observation(_row(5))
        maker.append_completed_observation(_row(6))
        state_at_limit = maker.export_state()

        with self.assertRaisesRegex(RuntimeError, "append limit"):
            maker.append_completed_observation(_row(7))
        self.assertEqual(maker.export_state(), state_at_limit)
        self.assertEqual(maker.rollback_reversible_append_transaction(token), 2)
        self.assertEqual(maker.export_state(), exact_start)

    def test_begin_requires_a_full_window_and_one_live_transaction(self) -> None:
        """Invalid begins fail without silently replacing the first token.

        非法 begin 不会悄悄替换第一张 token。
        """

        partial = RollingMarketMakerOLS(window_size=3)
        partial.append_completed_observation(_row(0))
        with self.assertRaisesRegex(RuntimeError, "full T_m"):
            partial.begin_reversible_append_transaction(max_appends=1)

        maker = self._full_maker()
        token = maker.begin_reversible_append_transaction(max_appends=1)
        with self.assertRaisesRegex(RuntimeError, "already active"):
            maker.begin_reversible_append_transaction(max_appends=1)
        self.assertEqual(maker.rollback_reversible_append_transaction(token), 0)

    def test_short_transaction_does_not_snapshot_the_full_history(self) -> None:
        """Begin, one ordinary append, and rollback avoid an O(T_m) copy.

        begin、一次普通追加和 rollback 都不会执行 O(T_m) 全历史复制。
        """

        maker = RollingMarketMakerOLS(
            window_size=5,
            resynchronize_every=100,
        )
        for index in range(5):
            maker.append_completed_observation(_row(index))
        exact_start = maker.export_state()

        # If the implementation tries to call the public full snapshot inside
        # this short transaction, the test fails immediately. / 若短事务内部试图
        # 调用公开的完整 snapshot，本测试会立即失败。
        with patch.object(
            maker,
            "snapshot",
            side_effect=AssertionError("full snapshot used"),
        ):
            token = maker.begin_reversible_append_transaction(max_appends=1)
            maker.append_completed_observation(_row(5))
            self.assertEqual(
                maker.rollback_reversible_append_transaction(token),
                1,
            )
        self.assertEqual(maker.export_state(), exact_start)


class TestSessionTransaction(unittest.TestCase):
    """Validate one reusable detached frozen branch. / 验证一个可重复使用的脱离式固定分支。"""

    @classmethod
    def setUpClass(cls) -> None:
        """Create one trusted origin checkpoint for all focused tests.

        为所有针对性测试建立一个可信起点 checkpoint。
        """

        cls.checkpoint = _build_demo_checkpoint()

    def setUp(self) -> None:
        """Each test owns a new detached branch. / 每个测试使用一个全新脱离分支。"""

        self.branch = restore_detached_frozen_branch(self.checkpoint)

    def _run_three_periods(self) -> tuple[object, ...]:
        """Run a deterministic supplied path of length three. / 运行一条确定的三期外部抽样路径。"""

        return tuple(
            self.branch.run_next_frozen_policy_period_with_supplied_draws(
                noise_order_u=noise,
                next_value_index=next_index,
            )
            for noise, next_index in (
                (0.02, 1),
                (-0.03, 7),
                (0.01, 4),
            )
        )

    def test_three_period_path_rolls_back_to_exact_origin_and_replays(self) -> None:
        """Market state, maker, RNGs, policy, Q, and counters all return exactly.

        市场状态、做市商、随机流、策略、Q 表和计数器都精确回到起点。
        """

        exact_start = _session_snapshot(self.branch)
        first_token = self.branch.begin_reversible_frozen_supplied_path(
            max_periods=3,
        )
        first_observations = self._run_three_periods()
        self.assertEqual(
            self.branch.rollback_reversible_frozen_supplied_path(first_token),
            3,
        )
        self.assertEqual(_session_snapshot(self.branch), exact_start)

        second_token = self.branch.begin_reversible_frozen_supplied_path(
            max_periods=3,
        )
        second_observations = self._run_three_periods()
        self.assertEqual(first_observations, second_observations)
        self.assertEqual(
            self.branch.rollback_reversible_frozen_supplied_path(second_token),
            3,
        )
        self.assertEqual(_session_snapshot(self.branch), exact_start)

    def test_period_limit_rejects_fourth_period_before_mutation(self) -> None:
        """A three-period transaction cannot accidentally execute t=4.

        三期事务不能意外执行第 t=4 期。
        """

        exact_start = _session_snapshot(self.branch)
        token = self.branch.begin_reversible_frozen_supplied_path(max_periods=3)
        self._run_three_periods()
        state_at_limit = _session_snapshot(self.branch)

        with self.assertRaisesRegex(RuntimeError, "period limit"):
            self.branch.run_next_frozen_policy_period_with_supplied_draws(
                noise_order_u=0.0,
                next_value_index=0,
            )
        self.assertEqual(_session_snapshot(self.branch), state_at_limit)
        self.assertEqual(
            self.branch.rollback_reversible_frozen_supplied_path(token),
            3,
        )
        self.assertEqual(_session_snapshot(self.branch), exact_start)

    def test_active_path_rejects_internal_rng_and_measurement_finish(self) -> None:
        """Only supplied draws may advance the active disposable path.

        活动的一次性路径只能由外部抽样推进。
        """

        exact_start = _session_snapshot(self.branch)
        token = self.branch.begin_reversible_frozen_supplied_path(max_periods=2)

        with self.assertRaisesRegex(RuntimeError, "internal draws"):
            self.branch.run_next_frozen_policy_period()
        with self.assertRaisesRegex(RuntimeError, "Rollback"):
            self.branch.finish_frozen_greedy_measurement()
        self.assertEqual(
            self.branch.rollback_reversible_frozen_supplied_path(token),
            0,
        )
        self.assertEqual(_session_snapshot(self.branch), exact_start)

    def test_foreign_and_reused_tokens_are_rejected(self) -> None:
        """Tokens cannot cross session boundaries or roll back twice.

        token 不能跨 session 使用，也不能重复回滚。
        """

        other = restore_detached_frozen_branch(self.checkpoint)
        token = self.branch.begin_reversible_frozen_supplied_path(max_periods=1)
        other_token = other.begin_reversible_frozen_supplied_path(max_periods=1)

        with self.assertRaisesRegex(RuntimeError, "stale or foreign"):
            self.branch.rollback_reversible_frozen_supplied_path(other_token)
        self.assertEqual(
            self.branch.rollback_reversible_frozen_supplied_path(token),
            0,
        )
        with self.assertRaisesRegex(RuntimeError, "stale or foreign"):
            self.branch.rollback_reversible_frozen_supplied_path(token)
        self.assertEqual(
            other.rollback_reversible_frozen_supplied_path(other_token),
            0,
        )

    def test_rollback_recovers_after_post_append_failure(self) -> None:
        """Even an exception after one committed maker row remains reversible.

        即使一条做市商记录已提交后才报错，事务仍可精确回滚。
        """

        exact_start = _session_snapshot(self.branch)
        token = self.branch.begin_reversible_frozen_supplied_path(max_periods=2)
        original_runner = self.branch._run_period_with_draw_suppliers

        def advance_then_fail(*args: object, **kwargs: object) -> object:
            """Inject failure after the normal period has advanced. / 在正常推进后注入报错。"""

            original_runner(*args, **kwargs)
            raise RuntimeError("injected after append / 追加后注入错误")

        with patch.object(
            self.branch,
            "_run_period_with_draw_suppliers",
            side_effect=advance_then_fail,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected after append"):
                self.branch.run_next_frozen_policy_period_with_supplied_draws(
                    noise_order_u=0.02,
                    next_value_index=1,
                )

        self.assertEqual(
            self.branch.rollback_reversible_frozen_supplied_path(token),
            1,
        )
        self.assertEqual(_session_snapshot(self.branch), exact_start)


if __name__ == "__main__":
    unittest.main()
