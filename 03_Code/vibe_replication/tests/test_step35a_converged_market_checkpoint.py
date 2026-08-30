"""Independent tests for Step 35A. / Step 35A 的独立自动测试。"""

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import random
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = PROJECT_ROOT / "steps"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from step_22_market_maker_rolling_history import MarketObservation
from step_24b_fast_rolling_ols import RollingMarketMakerOLS
from step_25_one_market_period import build_paper_inputs
from step_26_reproducible_random_streams import build_randomized_paper_session
from step_28_session_phases import SessionPhase, SessionPhaseController
import steps.step_35a_converged_market_checkpoint as step35a
from steps.step_35a_converged_market_checkpoint import (
    ImmutableArraySnapshot,
    capture_at_convergence_boundary,
    restore_detached_frozen_branch,
    restore_two_independent_branches,
)


def _assert_numpy_global_state_equal(
    testcase: unittest.TestCase,
    first: tuple[object, ...],
    second: tuple[object, ...],
) -> None:
    """Compare NumPy's legacy global state without ambiguous array truth.

    比较 NumPy 全局状态，避免数组布尔值含糊报错。
    """

    testcase.assertEqual(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    testcase.assertEqual(first[2:], second[2:])


def _contains_numpy_array(value: object) -> bool:
    """Recursively detect a mutable NumPy array in a checkpoint.

    递归检查 checkpoint 是否偷偷包含可变 NumPy 数组。
    """

    if isinstance(value, np.ndarray):
        return True
    if is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_numpy_array(getattr(value, field.name))
            for field in fields(value)
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_numpy_array(item) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_numpy_array(key) or _contains_numpy_array(item)
            for key, item in value.items()
        )
    return False


class TestConvergedMarketCheckpoint(unittest.TestCase):
    """Attack capture, restore, parity, and ownership. / 攻击保存、恢复、一致性与所有权。"""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the expensive paper inputs once. / 昂贵的论文输入只建立一次。"""

        cls.parameters = PaperParameters()
        (
            cls.value_grid,
            cls.price_grid,
            cls.actions,
            cls.initial_q,
            cls.prehistory,
        ) = build_paper_inputs(cls.parameters)
        cls.session_counter = 0

    def _fresh_controller(self) -> SessionPhaseController:
        """Return a controller one training period before capture.

        返回一个只需运行一期训练即可保存的 controller。
        """

        type(self).session_counter += 1
        stable_q = np.zeros_like(self.initial_q)
        stable_q[:, 0] = 1_000_000_000.0
        session = build_randomized_paper_session(
            parameters=self.parameters,
            value_grid=self.value_grid,
            price_grid=self.price_grid,
            action_multipliers=self.actions,
            initial_q_table=stable_q,
            prehistory=self.prehistory,
            experiment_seed=35_000_001,
            experiment_cell_key="step35a_unit_test",
            session_index=type(self).session_counter,
        )
        return SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=1,
            measurement_periods_required=3,
        )

    def _boundary(self) -> SessionPhaseController:
        """Advance to the exact post-convergence/pre-measurement point.

        推进到“已收敛、第一条测量尚未发生”的精确时点。
        """

        controller = self._fresh_controller()
        self.assertIsNone(controller.run_next_period())
        self.assertIs(controller.phase, SessionPhase.MEASUREMENT)
        self.assertEqual(controller.measurement_periods_completed, 0)
        return controller

    def test_capture_is_read_only_and_checkpoint_has_no_numpy_arrays(self) -> None:
        """Capture changes neither local nor process-global state.

        保存不得改变 session，也不得改变进程的全局随机状态。
        """

        controller = self._boundary()
        session = controller.session
        local_before = (
            session.period_number,
            session.previous_price,
            session.previous_value,
            session.current_value,
            tuple(session.shared_value_visit_counts),
            session.all_random_states(),
            session.market_maker.export_state(),
            tuple(trader.q_table.tobytes() for trader in session.traders),
        )
        python_global_before = random.getstate()
        numpy_global_before = np.random.get_state()
        checkpoint = capture_at_convergence_boundary(controller)
        local_after = (
            session.period_number,
            session.previous_price,
            session.previous_value,
            session.current_value,
            tuple(session.shared_value_visit_counts),
            session.all_random_states(),
            session.market_maker.export_state(),
            tuple(trader.q_table.tobytes() for trader in session.traders),
        )

        self.assertEqual(local_before, local_after)
        self.assertEqual(random.getstate(), python_global_before)
        _assert_numpy_global_state_equal(
            self,
            np.random.get_state(),
            numpy_global_before,
        )
        self.assertFalse(_contains_numpy_array(checkpoint))
        self.assertEqual(len(checkpoint.payload.market_maker_state.rows), 10_000)
        self.assertEqual(len(checkpoint.payload.all_seven_rng_states), 7)
        self.assertEqual(len(checkpoint.payload.implementation_tree_sha256), 64)
        self.assertTrue(checkpoint.payload.platform_system)
        self.assertTrue(checkpoint.payload.platform_machine)
        self.assertIsInstance(checkpoint.payload.price_grid, tuple)
        self.assertEqual(len(checkpoint.payload.price_grid), 10)
        self.assertTrue(
            all(
                isinstance(row, tuple) and len(row) == 31
                for row in checkpoint.payload.price_grid
            )
        )

    def test_restore_is_exact_and_detached(self) -> None:
        """Every saved causal state is restored without old callbacks.

        所有已保存因果状态都精确恢复，并且不带旧 callback。
        """

        checkpoint = capture_at_convergence_boundary(self._boundary())
        branch = restore_detached_frozen_branch(checkpoint)
        payload = checkpoint.payload

        self.assertEqual(branch.parameters, payload.parameters)
        self.assertEqual(branch.value_grid, payload.value_grid)
        self.assertEqual(branch.price_grid, payload.price_grid)
        self.assertEqual(branch.action_multipliers, payload.action_multipliers)
        self.assertEqual(branch.period_number, payload.origin_global_period)
        self.assertEqual(branch.previous_price, payload.previous_price)
        self.assertEqual(branch.previous_value, payload.previous_value)
        self.assertEqual(branch.current_value, payload.current_value)
        self.assertEqual(
            tuple(branch.shared_value_visit_counts),
            payload.shared_value_visit_counts,
        )
        self.assertEqual(branch.all_random_states(), payload.all_seven_rng_states)
        self.assertEqual(
            branch.market_maker.export_state(),
            payload.market_maker_state,
        )
        self.assertEqual(branch.execution_mode, "measurement")
        self.assertIsNone(branch.after_q_update_observer)
        self.assertTrue(all(not trader.q_table.flags.writeable for trader in branch.traders))

    def test_original_controller_and_restored_branch_have_same_next_period(self) -> None:
        """The clone reproduces the source's next future period exactly.

        恢复分支必须逐字段复现源 session 的下一期。
        """

        controller = self._boundary()
        checkpoint = capture_at_convergence_boundary(controller)
        branch = restore_detached_frozen_branch(checkpoint)
        source_observation = controller.run_next_period()
        branch_observation = branch.run_next_frozen_policy_period()
        self.assertEqual(source_observation, branch_observation)
        self.assertEqual(
            controller.session.market_maker.export_state(),
            branch.market_maker.export_state(),
        )
        self.assertEqual(controller.session.all_random_states(), branch.all_random_states())

    def test_integrated_parity_crosses_market_maker_resynchronization(self) -> None:
        """Source and restore remain exact when the real session rebuilds OLS.

        真实 session 跨过 OLS 重同步时，源与恢复分支仍须完全一致。
        """

        small_parameters = PaperParameters(market_maker_window=20)
        values, prices, actions, initial_q, prehistory = build_paper_inputs(
            small_parameters
        )
        stable_q = np.zeros_like(initial_q)
        stable_q[:, 0] = 1_000_000_000.0
        session = build_randomized_paper_session(
            parameters=small_parameters,
            value_grid=values,
            price_grid=prices,
            action_multipliers=actions,
            initial_q_table=stable_q,
            prehistory=prehistory,
            experiment_seed=35_000_002,
            experiment_cell_key="step35a_integrated_resynchronization",
            session_index=0,
        )
        controller = SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=1,
            measurement_periods_required=20,
        )
        self.assertIsNone(controller.run_next_period())
        checkpoint = capture_at_convergence_boundary(controller)
        branch = restore_detached_frozen_branch(checkpoint)
        initial_resynchronizations = (
            checkpoint.payload.market_maker_state.resynchronization_count
        )
        self.assertEqual(
            checkpoint.payload.market_maker_state.updates_since_resynchronization,
            1,
        )

        for _ in range(19):
            self.assertEqual(
                controller.run_next_period(),
                branch.run_next_frozen_policy_period(),
            )
            self.assertEqual(
                controller.session.market_maker.export_state(),
                branch.market_maker.export_state(),
            )
        self.assertEqual(
            branch.market_maker.resynchronization_count,
            initial_resynchronizations + 1,
        )

    def test_two_branches_are_equal_but_do_not_share_mutable_state(self) -> None:
        """Moving A leaves B still parked at the checkpoint.

        推进 A 时，B 必须仍停留在 checkpoint。
        """

        checkpoint = capture_at_convergence_boundary(self._boundary())
        first, second = restore_two_independent_branches(checkpoint)
        second_before = (
            second.period_number,
            second.all_random_states(),
            second.market_maker.export_state(),
        )
        first_observation = first.run_next_frozen_policy_period()
        self.assertEqual(
            second_before,
            (
                second.period_number,
                second.all_random_states(),
                second.market_maker.export_state(),
            ),
        )
        second_observation = second.run_next_frozen_policy_period()
        self.assertEqual(first_observation, second_observation)
        self.assertIsNot(first.market_maker, second.market_maker)
        self.assertIsNot(
            first.shared_value_visit_counts,
            second.shared_value_visit_counts,
        )
        for left, right in zip(first.traders, second.traders, strict=True):
            self.assertIsNot(left, right)
            self.assertFalse(np.shares_memory(left.q_table, right.q_table))

    def test_gaussian_cache_is_preserved(self) -> None:
        """An odd Gaussian draw leaves one cached normal variate to save.

        奇数次正态抽样会缓存第二个正态数，这个隐藏状态也必须保存。
        """

        controller = self._boundary()
        checkpoint = capture_at_convergence_boundary(controller)
        noise_state = checkpoint.payload.all_seven_rng_states[2]
        self.assertIsNotNone(noise_state[2])
        branch = restore_detached_frozen_branch(checkpoint)
        self.assertEqual(
            branch.streams.noise_generator.getstate(),
            noise_state,
        )
        self.assertEqual(
            controller.run_next_period(),
            branch.run_next_frozen_policy_period(),
        )

    def test_capture_rejects_wrong_time_on_both_sides(self) -> None:
        """Too early and one row too late are both rejected.

        太早和晚了一条测量记录都必须拒绝。
        """

        training_controller = self._fresh_controller()
        with self.assertRaises(RuntimeError):
            capture_at_convergence_boundary(training_controller)

        measurement_controller = self._boundary()
        self.assertIsNotNone(measurement_controller.run_next_period())
        self.assertEqual(measurement_controller.measurement_periods_completed, 1)
        with self.assertRaises(RuntimeError):
            capture_at_convergence_boundary(measurement_controller)

    def test_stale_outer_digest_rejects_tampering(self) -> None:
        """Changing one field without a matching checksum cannot restore.

        修改一个字段但不更新总校验码时，恢复必须失败。
        """

        checkpoint = capture_at_convergence_boundary(self._boundary())
        tampered_payload = replace(
            checkpoint.payload,
            origin_global_period=checkpoint.payload.origin_global_period + 1,
        )
        tampered = replace(checkpoint, payload=tampered_payload)
        with self.assertRaises(ValueError):
            restore_detached_frozen_branch(tampered)

    def test_recomputed_digest_cannot_make_a_flat_grid_valid(self) -> None:
        """Even a matching checksum cannot revive the old global-grid schema.

        即使重新计算校验码，旧的一维全局网格也不能成为有效 checkpoint。
        """

        checkpoint = capture_at_convergence_boundary(self._boundary())
        malformed_payload = replace(
            checkpoint.payload,
            price_grid=checkpoint.payload.price_grid[0],  # type: ignore[arg-type]
        )
        malformed_checkpoint = replace(
            checkpoint,
            payload=malformed_payload,
            checkpoint_sha256=step35a._payload_digest(malformed_payload),
        )
        with self.assertRaises((TypeError, ValueError)):
            restore_detached_frozen_branch(malformed_checkpoint)

    def test_array_snapshot_rejects_corrupt_inner_bytes(self) -> None:
        """The small per-array checksum catches damaged Q bytes.

        每个数组自己的校验码能够发现损坏的 Q bytes。
        """

        array = np.array([[1.0, 2.0]], dtype=np.float64)
        snapshot = ImmutableArraySnapshot.capture(array)
        broken = replace(
            snapshot,
            c_order_bytes=snapshot.c_order_bytes[:-1] + b"x",
        )
        with self.assertRaises(ValueError):
            broken.restore(writeable=True)


class TestRollingMarketMakerExactState(unittest.TestCase):
    """Protect the exact OLS accumulator and rebuild phase. / 保护精确 OLS 累加器与重建阶段。"""

    @staticmethod
    def _row(index: int) -> MarketObservation:
        """Create varied, identifiable OLS data. / 建立有变化、可识别的 OLS 数据。"""

        price = 0.98 + 0.011 * index
        order_flow = -2.0 + 0.7 * index
        return MarketObservation(
            fundamental_value_v=0.9 + 0.08 * order_flow + 0.003 * index,
            market_price_p=price,
            insensitive_order_z=500.0 - 500.0 * price + 0.02 * index,
            informed_and_noise_order_y=order_flow,
        )

    def test_export_restore_crosses_same_resynchronization_boundary(self) -> None:
        """Both copies rebuild on the same later append.

        两份做市商必须在同一次后续追加时触发重同步。
        """

        original = RollingMarketMakerOLS(
            window_size=4,
            resynchronize_every=3,
        )
        for index in range(4):
            original.append_completed_observation(self._row(index))
        state = original.export_state()
        self.assertEqual(state.updates_since_resynchronization, 1)
        restored = RollingMarketMakerOLS.from_state(state)
        self.assertEqual(restored.export_state(), state)

        for index in (4, 5):
            original.append_completed_observation(self._row(index))
            restored.append_completed_observation(self._row(index))
            self.assertEqual(original.export_state(), restored.export_state())
        self.assertEqual(original.resynchronization_count, 2)

    def test_invalid_saved_rebuild_phase_is_rejected(self) -> None:
        """A counter outside [0, interval) is impossible.

        不在 [0, 重同步间隔) 内的倒计数是不可能状态。
        """

        maker = RollingMarketMakerOLS(window_size=2, resynchronize_every=2)
        maker.append_completed_observation(self._row(0))
        maker.append_completed_observation(self._row(1))
        invalid = replace(
            maker.export_state(),
            updates_since_resynchronization=2,
        )
        with self.assertRaises(ValueError):
            RollingMarketMakerOLS.from_state(invalid)

    def test_saved_statistics_must_agree_with_saved_rows(self) -> None:
        """Correct sizes and finite values cannot hide a wrong regression.

        即使样本量正确且数值有限，也不能偷偷换成错误回归。
        """

        maker = RollingMarketMakerOLS(window_size=3, resynchronize_every=10)
        for index in range(3):
            maker.append_completed_observation(self._row(index))
        state = maker.export_state()
        inconsistent_demand = replace(
            state.demand_statistics,
            mean_x=state.demand_statistics.mean_x + 1.0,
        )
        inconsistent_state = replace(
            state,
            demand_statistics=inconsistent_demand,
        )
        with self.assertRaises(ValueError):
            RollingMarketMakerOLS.from_state(inconsistent_state)

    def test_saved_counters_must_describe_a_possible_history(self) -> None:
        """Finite in-range counters must still be causally possible.

        即使计数有限且各自未越界，它们的组合也必须在因果上可能发生。
        """

        maker = RollingMarketMakerOLS(window_size=3, resynchronize_every=10)
        for index in range(3):
            maker.append_completed_observation(self._row(index))
        state = maker.export_state()
        too_many_since_rebuild = replace(
            state,
            updates_since_resynchronization=4,
        )
        missing_recorded_rebuild = replace(
            state,
            successful_append_count=100,
            updates_since_resynchronization=0,
            resynchronization_count=0,
        )
        with self.assertRaises(ValueError):
            RollingMarketMakerOLS.from_state(too_many_since_rebuild)
        with self.assertRaises(ValueError):
            RollingMarketMakerOLS.from_state(missing_recorded_rebuild)

    def test_rows_alone_are_not_the_checkpoint_contract(self) -> None:
        """The exported state includes the hidden accumulators and counters.

        导出的状态明确包含隐藏累加器和计数器，而不只是历史行。
        """

        maker = RollingMarketMakerOLS(window_size=3, resynchronize_every=10)
        for index in range(3):
            maker.append_completed_observation(self._row(index))
        state = maker.export_state()
        self.assertEqual(state.demand_statistics.sample_size, 3)
        self.assertEqual(state.value_statistics.sample_size, 3)
        self.assertEqual(state.successful_append_count, 3)
        self.assertEqual(state.updates_since_resynchronization, 3)


if __name__ == "__main__":
    unittest.main()
