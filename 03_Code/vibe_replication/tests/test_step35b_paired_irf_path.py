"""Independent tests for Step 35B. / Step 35B 的独立自动测试。"""

from dataclasses import FrozenInstanceError, replace
from math import isclose
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = PROJECT_ROOT / "steps"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from step_25_one_market_period import build_paper_inputs
from step_26_reproducible_random_streams import build_randomized_paper_session
from step_28_session_phases import SessionPhase, SessionPhaseController
from steps.step_34_mechanism_classifier import (
    PAPER_PATHS_PER_SESSION,
    PAPER_RESPONSE_PERIOD,
    PAPER_SHOCK_PERIOD,
    PAPER_TARGET_PRICE_DEVIATION,
    calibrate_uniform_noise_shock,
)
from steps.step_35a_converged_market_checkpoint import (
    capture_at_convergence_boundary,
    restore_detached_frozen_branch,
    restore_two_independent_branches,
)
import steps.step_35b_paired_irf_path as step35b
from steps.step_35b_paired_irf_path import (
    CONTROL_FIRST,
    IRF_LOCAL_PERIODS,
    TREATMENT_FIRST,
    _build_demo_checkpoint,
    audit_all_planned_paired_path_seed_uniqueness,
    build_paired_path_draw_schedule,
    derive_paired_path_seed_manifest,
    run_and_audit_one_paired_irf_path,
    run_one_paired_irf_path,
)


def _branch_snapshot(branch: object) -> tuple[object, ...]:
    """Capture all causal state needed for atomic-rejection checks.

    保存原子拒绝测试所需的全部因果状态。
    """

    return (
        branch.period_number,
        branch.previous_price,
        branch.previous_value,
        branch.current_value,
        branch.execution_mode,
        branch.frozen_draw_source_mode,
        tuple(branch.shared_value_visit_counts),
        branch.all_random_states(),
        branch.market_maker.export_state(),
        tuple(
            (
                trader.q_table.tobytes(),
                bool(trader.q_table.flags.writeable),
            )
            for trader in branch.traders
        ),
    )


def _build_state_dependent_trigger_checkpoint() -> object:
    """Build a frozen policy whose action changes with lagged-price index.

    建立一个会随滞后价格编号改变动作的固定策略。

    This is an engineering wiring fixture, not a calibrated paper result. / 这只是
    工程接线夹具，不是经过论文校准的结果。
    """

    parameters = PaperParameters(investor_slope=5.0)
    value_grid, price_grid, actions, initial_q, prehistory = build_paper_inputs(
        parameters
    )
    state_dependent_q = np.zeros_like(initial_q)
    states_per_price = len(value_grid) ** 2
    for state_id in range(state_dependent_q.shape[0]):
        price_index = state_id // states_per_price
        action_index = price_index % len(actions)
        state_dependent_q[state_id, action_index] = 1_000_000_000.0

    session = build_randomized_paper_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=actions,
        initial_q_table=state_dependent_q,
        prehistory=prehistory,
        experiment_seed=352_001,
        experiment_cell_key="step35b_state_dependent_fixture",
        session_index=0,
    )
    controller = SessionPhaseController.create_for_fresh_session(
        session,
        convergence_periods_required=5,
        measurement_periods_required=4,
    )
    while controller.phase is SessionPhase.TRAINING:
        self_check = controller.run_next_period()
        if self_check is not None:
            raise RuntimeError("Fixture emitted measurement too early. / 夹具过早产生测量记录。")
    return capture_at_convergence_boundary(controller)


class TestPairedIRFPath(unittest.TestCase):
    """Validate one complete t=1,...,4 causal pair. / 验证一条完整 t=1,...,4 因果配对路径。"""

    @classmethod
    def setUpClass(cls) -> None:
        """Build one immutable checkpoint and audited receipt once.

        一次性建立不可修改 checkpoint 与已审计凭证。
        """

        cls.checkpoint = _build_demo_checkpoint()
        cls.irf_seed = 20_260_829
        cls.calibration = calibrate_uniform_noise_shock(2.0, 0.5, 0.5)
        cls.receipt = run_and_audit_one_paired_irf_path(
            cls.checkpoint,
            irf_experiment_seed=cls.irf_seed,
            path_index=0,
            shock_calibration=cls.calibration,
        )

    def test_carried_t0_and_transactions_map_to_global_time(self) -> None:
        """The completed t0 outcome feeds local t1; new trades map to g..g+3.

        已完成的 t0 结果进入局部 t1；新交易映射到 g..g+3。
        """

        receipt = self.receipt
        origin = receipt.checkpoint_global_origin_period
        self.assertEqual(receipt.irf_local_state_origin, 0)
        self.assertEqual(receipt.carried_local_t0_global_period, origin - 1)
        self.assertEqual(
            receipt.carried_local_t0_price,
            self.checkpoint.payload.previous_price,
        )
        self.assertEqual(
            receipt.carried_local_t0_value,
            self.checkpoint.payload.previous_value,
        )
        self.assertEqual(
            receipt.local_t1_current_value,
            self.checkpoint.payload.current_value,
        )
        self.assertTrue(receipt.t0_outcome_carried_not_reexecuted)
        self.assertEqual(receipt.local_transaction_periods, (1, 2, 3, 4))
        self.assertEqual(
            tuple(period.expected_global_period for period in receipt.periods),
            tuple(origin + local_period - 1 for local_period in IRF_LOCAL_PERIODS),
        )
        self.assertEqual(receipt.periods[2].local_period, PAPER_SHOCK_PERIOD)
        self.assertEqual(receipt.periods[3].local_period, PAPER_RESPONSE_PERIOD)
        first = receipt.periods[0].control_observation
        self.assertEqual(
            first.fundamental_value_v,
            self.checkpoint.payload.current_value,
        )
        self.assertEqual(
            first.current_state_indexes[1:],
            (
                self.checkpoint.payload.value_grid.index(
                    self.checkpoint.payload.previous_value
                ),
                self.checkpoint.payload.value_grid.index(
                    self.checkpoint.payload.current_value
                ),
            ),
        )

    def test_control_and_treatment_are_exactly_equal_before_shock(self) -> None:
        """The complete t=1 and t=2 observations match bit for bit.

        t=1 与 t=2 的完整观测逐字段完全相同。
        """

        for period in self.receipt.periods[:2]:
            self.assertEqual(
                period.control_observation,
                period.treatment_observation,
            )
            self.assertFalse(period.shock_applied)
            self.assertEqual(period.signed_treatment_shock_u, 0.0)
        self.assertTrue(self.receipt.pre_shock_exact_parity_verified)

    def test_t3_shock_is_adverse_additive_and_applied_exactly_once(self) -> None:
        """Treatment u3 equals ordinary u3 plus one signed shock.

        实验组 u3 等于普通 u3 加一次带符号冲击。
        """

        shock_records = [
            period for period in self.receipt.periods if period.shock_applied
        ]
        self.assertEqual(len(shock_records), 1)
        record = shock_records[0]
        self.assertEqual(record.local_period, PAPER_SHOCK_PERIOD)
        self.assertIsNotNone(record.applied_noise_shock)
        self.assertEqual(
            record.control_noise_used_for_pricing,
            record.ordinary_noise_order_u,
        )
        self.assertEqual(
            record.treatment_noise_used_for_pricing,
            record.ordinary_noise_order_u + record.signed_treatment_shock_u,
        )
        expected_direction = (
            1.0
            if record.common_current_fundamental_value
            > self.checkpoint.payload.parameters.value_mean
            else -1.0
        )
        self.assertEqual(
            record.signed_treatment_shock_u,
            expected_direction * self.calibration.absolute_noise_shock,
        )
        self.assertTrue(self.receipt.shock_applied_exactly_once_verified)

    def test_t3_pre_noise_state_actions_and_prior_ols_are_identical(self) -> None:
        """Divergence begins only after the t3 noise enters order flow.

        差异只能在 t3 噪声进入订单流之后开始。
        """

        control = self.receipt.periods[2].control_observation
        treatment = self.receipt.periods[2].treatment_observation
        self.assertEqual(control.current_state_indexes, treatment.current_state_indexes)
        self.assertEqual(control.action_indexes, treatment.action_indexes)
        self.assertEqual(control.raw_orders_x, treatment.raw_orders_x)
        self.assertEqual(control.xi_0_hat, treatment.xi_0_hat)
        self.assertEqual(control.xi_1_hat, treatment.xi_1_hat)
        self.assertEqual(control.gamma_0_hat, treatment.gamma_0_hat)
        self.assertEqual(control.gamma_1_hat, treatment.gamma_1_hat)
        self.assertEqual(
            control.price_impact_lambda_hat,
            treatment.price_impact_lambda_hat,
        )
        self.assertNotEqual(control.noise_order_u, treatment.noise_order_u)
        self.assertTrue(
            isclose(
                self.receipt.paired_t3_treatment_minus_control_oriented_price_increment,
                self.receipt.expected_paired_t3_increment_from_lambda,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )

    def test_common_draws_continue_after_endogenous_divergence(self) -> None:
        """At t4 ordinary u and next value remain common.

        即使内生状态已分化，t4 的普通 u 与下一价值仍保持相同。
        """

        response = self.receipt.periods[3]
        self.assertEqual(response.local_period, PAPER_RESPONSE_PERIOD)
        self.assertEqual(
            response.control_observation.noise_order_u,
            response.ordinary_noise_order_u,
        )
        self.assertEqual(
            response.treatment_observation.noise_order_u,
            response.ordinary_noise_order_u,
        )
        self.assertEqual(
            response.control_observation.next_value_index,
            response.treatment_observation.next_value_index,
        )
        self.assertTrue(self.receipt.common_draws_verified)

    def test_branch_internal_rng_q_and_visit_counts_remain_frozen(self) -> None:
        """External schedules do not secretly resume learning or session RNGs.

        外部抽样不会偷偷恢复学习或推进 session 内部随机流。
        """

        self.assertTrue(self.receipt.branch_internal_rngs_unchanged_verified)
        self.assertTrue(self.receipt.frozen_learning_state_verified)
        self.assertEqual(
            self.receipt.control_final_state.frozen_draw_source_mode,
            "supplied",
        )
        self.assertEqual(
            self.receipt.treatment_final_state.frozen_draw_source_mode,
            "supplied",
        )

    def test_named_maker_rows_do_not_cross_between_branches(self) -> None:
        """The shocked t3 flow enters treatment history only.

        带冲击的 t3 订单流只进入实验组历史。
        """

        record = self.receipt.periods[2]
        control = record.control_observation
        treatment = record.treatment_observation
        self.assertTrue(
            isclose(
                treatment.total_order_flow_y - control.total_order_flow_y,
                record.signed_treatment_shock_u,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(record.completed_rows_match_named_branches)
        self.assertTrue(
            self.receipt.independent_market_maker_histories_verified
        )
        self.assertTrue(
            self.receipt.each_branch_t4_ols_uses_own_t3_history_verified
        )
        self.assertTrue(self.receipt.periods[3].prior_ols_matches_named_branch_history)

    def test_branch_execution_order_is_economically_irrelevant(self) -> None:
        """The runtime actually replayed and compared both orders.

        运行时确实用两种先后顺序重放并比较。
        """

        self.assertTrue(self.receipt.branch_order_invariance_verified)
        self.assertEqual(self.receipt.execution_replays_for_order_audit, 2)
        self.assertEqual(self.receipt.branch_execution_order, CONTROL_FIRST)

    def test_receipt_cannot_be_mistaken_for_formal_classification(self) -> None:
        """One path is not 10,000 paths and cannot classify a session.

        一条路径不是一万条路径，也不能分类一个 session。
        """

        self.assertEqual(self.receipt.paths_economically_represented, 1)
        self.assertEqual(
            self.receipt.protocol_disclosure.paper_requires_paths_per_session,
            PAPER_PATHS_PER_SESSION,
        )
        self.assertFalse(self.receipt.full_paper_path_count_verified)
        self.assertFalse(self.receipt.classification_ready)
        self.assertTrue(self.receipt.calibration_arithmetic_verified)
        self.assertTrue(self.receipt.paper_1_2_percent_target_used)
        self.assertFalse(
            self.receipt.protocol_disclosure.scalable_formal_runner_verified
        )
        self.assertFalse(self.receipt.calibration_aggregate_provenance_verified)
        disclosure = self.receipt.protocol_disclosure
        self.assertTrue(disclosure.paper_states_all_sessions_converged_at_local_t0)
        self.assertFalse(disclosure.paper_specifies_t0_transaction_or_fork_semantics)
        self.assertTrue(disclosure.replication_carries_completed_t0_outcome)
        self.assertTrue(disclosure.replication_executes_no_new_t0_transaction)
        self.assertTrue(disclosure.replication_carries_common_checkpoint_v1_across_paths)
        self.assertFalse(disclosure.t0_and_initial_v1_sensitivity_resolved)
        self.assertTrue(disclosure.control_branch_is_not_paper_long_run_baseline)
        self.assertFalse(disclosure.long_run_order_baseline_computed)
        self.assertFalse(disclosure.paper_specifies_q_update_or_freeze_rule_inside_irf)

    def test_receipt_is_frozen(self) -> None:
        """Historical path evidence cannot be edited. / 历史路径证据不能被修改。"""

        with self.assertRaises(FrozenInstanceError):
            self.receipt.classification_ready = True  # type: ignore[misc]


class TestPairedPathRandomness(unittest.TestCase):
    """Audit path identities and one-draw scheduling. / 审计路径身份与每期一次抽样。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoint = _build_demo_checkpoint()
        cls.irf_seed = 20_260_829

    def test_same_path_replays_and_different_path_changes(self) -> None:
        """Path identity is independent of scheduling order.

        路径身份不受任务调度先后影响。
        """

        first = build_paired_path_draw_schedule(
            self.checkpoint,
            irf_experiment_seed=self.irf_seed,
            path_index=2,
        )
        replay = build_paired_path_draw_schedule(
            self.checkpoint,
            irf_experiment_seed=self.irf_seed,
            path_index=2,
        )
        different = build_paired_path_draw_schedule(
            self.checkpoint,
            irf_experiment_seed=self.irf_seed,
            path_index=3,
        )
        self.assertEqual(first, replay)
        self.assertNotEqual(first.seed_manifest, different.seed_manifest)
        self.assertNotEqual(
            first.ordinary_noise_orders_u,
            different.ordinary_noise_orders_u,
        )

        out_of_order = {
            index: build_paired_path_draw_schedule(
                self.checkpoint,
                irf_experiment_seed=self.irf_seed,
                path_index=index,
            )
            for index in (2, 0, 1)
        }
        in_order = {
            index: build_paired_path_draw_schedule(
                self.checkpoint,
                irf_experiment_seed=self.irf_seed,
                path_index=index,
            )
            for index in (0, 1, 2)
        }
        self.assertEqual(out_of_order, in_order)

    def test_schedule_matches_independent_one_draw_per_period_oracles(self) -> None:
        """Catch hidden, missing, or double schedule draws.

        发现隐藏、遗漏或重复的路径抽样。
        """

        schedule = build_paired_path_draw_schedule(
            self.checkpoint,
            irf_experiment_seed=self.irf_seed,
            path_index=0,
        )
        noise_oracle = random.Random(
            schedule.seed_manifest.ordinary_noise_seed
        )
        value_oracle = random.Random(schedule.seed_manifest.next_value_seed)
        expected_noise = tuple(
            noise_oracle.gauss(0.0, schedule.noise_standard_deviation)
            for _ in IRF_LOCAL_PERIODS
        )
        expected_values = tuple(
            value_oracle.randrange(schedule.number_of_value_points)
            for _ in IRF_LOCAL_PERIODS
        )
        self.assertEqual(schedule.ordinary_noise_orders_u, expected_noise)
        self.assertEqual(schedule.next_value_indexes, expected_values)
        self.assertEqual(
            schedule.ordinary_noise_final_rng_state,
            noise_oracle.getstate(),
        )
        self.assertEqual(
            schedule.next_value_final_rng_state,
            value_oracle.getstate(),
        )
        self.assertEqual(schedule.draws_per_stream, 4)

    def test_all_planned_path_child_seeds_are_unique_without_claiming_execution(self) -> None:
        """Audit 20,000 identities, not 10,000 simulated paths.

        核对 20,000 个身份，但不冒充已经模拟 10,000 条路径。
        """

        audit = audit_all_planned_paired_path_seed_uniqueness(
            self.checkpoint,
            irf_experiment_seed=self.irf_seed,
        )
        self.assertEqual(audit.planned_path_count, 10_000)
        self.assertEqual(audit.planned_child_stream_count, 20_000)
        self.assertEqual(audit.unique_child_seed_count, 20_000)
        self.assertTrue(audit.all_planned_child_seeds_are_unique)
        self.assertEqual(audit.child_seed_bits, 256)
        self.assertFalse(audit.formal_cross_session_uniqueness_verified)
        self.assertFalse(audit.paths_were_executed)

    def test_checkpoint_digest_is_provenance_not_seed_entropy(self) -> None:
        """Stable economic session identity, not source-file bytes, seeds paths.

        路径种子使用稳定的经济 session 身份，不使用源代码文件字节。
        """

        manifest = derive_paired_path_seed_manifest(
            self.checkpoint,
            irf_experiment_seed=self.irf_seed,
            path_index=0,
        )
        self.assertEqual(manifest.child_seed_bits, 256)
        self.assertEqual(
            manifest.source_session_seed,
            self.checkpoint.payload.seed_manifest.session_seed,
        )
        self.assertTrue(manifest.checkpoint_digest_is_provenance_not_seed_entropy)
        self.assertFalse(manifest.formal_cross_session_seed_uniqueness_verified)

    def test_invalid_seed_and_path_identity_are_rejected(self) -> None:
        """Booleans, negatives, and index 10,000 are invalid.

        布尔值、负数与编号 10,000 都无效。
        """

        for invalid_seed in (True, -1, 2**64):
            with self.subTest(irf_experiment_seed=invalid_seed):
                with self.assertRaises((TypeError, ValueError)):
                    derive_paired_path_seed_manifest(
                        self.checkpoint,
                        irf_experiment_seed=invalid_seed,
                        path_index=0,
                    )
        for invalid_path in (True, -1, 10_000, 1.5):
            with self.subTest(path_index=invalid_path):
                with self.assertRaises((TypeError, ValueError)):
                    derive_paired_path_seed_manifest(
                        self.checkpoint,
                        irf_experiment_seed=self.irf_seed,
                        path_index=invalid_path,
                    )


class TestStateDependentTriggerWiring(unittest.TestCase):
    """Prove the shocked p3 is really used by the frozen t4 policy.

    证明受到冲击的 p3 确实进入固定策略的 t4 决策。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoint = _build_state_dependent_trigger_checkpoint()
        # A deliberately large 12.5% debug target makes the t3 price cross a
        # price-grid boundary. It is NOT the paper's 1.2% calibration. / 故意
        # 使用较大的 12.5% 调试目标，让 t3 价格跨越网格边界；这不是论文的 1.2%。
        cls.debug_calibration = calibrate_uniform_noise_shock(
            2.0,
            0.5,
            0.5,
            target_normalized_price_deviation=0.125,
        )
        cls.receipt = run_one_paired_irf_path(
            cls.checkpoint,
            irf_experiment_seed=20_260_829,
            path_index=0,
            shock_calibration=cls.debug_calibration,
        )

    def test_t3_price_index_becomes_t4_state_and_changes_action(self) -> None:
        """The causal chain is p3 -> lagged-price state -> frozen action.

        因果链为：p3 → 滞后价格状态 → 固定策略动作。
        """

        shock = self.receipt.periods[2]
        response = self.receipt.periods[3]
        self.assertEqual(
            shock.control_observation.current_state_indexes,
            shock.treatment_observation.current_state_indexes,
        )
        self.assertEqual(
            shock.control_observation.action_indexes,
            shock.treatment_observation.action_indexes,
        )
        self.assertNotEqual(
            shock.control_observation.next_state_indexes[0],
            shock.treatment_observation.next_state_indexes[0],
        )
        self.assertEqual(
            response.control_observation.current_state_indexes,
            shock.control_observation.next_state_indexes,
        )
        self.assertEqual(
            response.treatment_observation.current_state_indexes,
            shock.treatment_observation.next_state_indexes,
        )
        self.assertNotEqual(
            response.control_observation.action_indexes,
            response.treatment_observation.action_indexes,
        )

        frozen_policy = self.checkpoint.payload.frozen_policy_action_indexes.restore(
            writeable=False
        )
        control_state_id = response.control_observation.current_state_id
        treatment_state_id = response.treatment_observation.current_state_id
        self.assertEqual(
            response.control_observation.action_indexes,
            tuple(int(frozen_policy[i, control_state_id]) for i in range(2)),
        )
        self.assertEqual(
            response.treatment_observation.action_indexes,
            tuple(int(frozen_policy[i, treatment_state_id]) for i in range(2)),
        )
        self.assertEqual(
            response.control_observation.noise_order_u,
            response.treatment_observation.noise_order_u,
        )
        self.assertFalse(self.receipt.paper_1_2_percent_target_used)
        self.assertFalse(self.receipt.classification_ready)
        self.assertTrue(self.receipt.frozen_learning_state_verified)


class TestSuppliedFrozenDrawAPI(unittest.TestCase):
    """Protect the small Step-26 production bridge. / 保护第 26 步的小型正式接口。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoint = _build_demo_checkpoint()

    def test_supplied_draws_consume_no_internal_rng_and_lock_mode(self) -> None:
        """Successful supplied execution leaves all seven states parked.

        成功的外部抽样执行不会推进七条内部随机状态。
        """

        branch = restore_detached_frozen_branch(self.checkpoint)
        random_states_before = branch.all_random_states()
        observation = branch.run_next_frozen_policy_period_with_supplied_draws(
            noise_order_u=0.125,
            next_value_index=2,
        )
        self.assertEqual(observation.noise_order_u, 0.125)
        self.assertEqual(observation.next_value_index, 2)
        self.assertEqual(branch.all_random_states(), random_states_before)
        self.assertEqual(branch.frozen_draw_source_mode, "supplied")
        before_rejected_switch = _branch_snapshot(branch)
        with self.assertRaises(RuntimeError):
            branch.run_next_frozen_policy_period()
        self.assertEqual(_branch_snapshot(branch), before_rejected_switch)

    def test_internal_then_supplied_switch_is_rejected(self) -> None:
        """The opposite draw-source switch is also forbidden.

        相反方向的抽样来源切换同样被禁止。
        """

        branch = restore_detached_frozen_branch(self.checkpoint)
        branch.run_next_frozen_policy_period()
        self.assertEqual(branch.frozen_draw_source_mode, "internal")
        before = _branch_snapshot(branch)
        with self.assertRaises(RuntimeError):
            branch.run_next_frozen_policy_period_with_supplied_draws(
                noise_order_u=0.0,
                next_value_index=0,
            )
        self.assertEqual(_branch_snapshot(branch), before)

    def test_invalid_supplied_inputs_fail_atomically_then_valid_path_matches(self) -> None:
        """Bad concrete draws never partly price or append a row.

        错误外部抽样绝不会只完成部分定价或追加历史。
        """

        invalid_calls = (
            {"noise_order_u": True, "next_value_index": 0},
            {"noise_order_u": float("nan"), "next_value_index": 0},
            {"noise_order_u": float("inf"), "next_value_index": 0},
            {"noise_order_u": 0.0, "next_value_index": True},
            {"noise_order_u": 0.0, "next_value_index": -1},
            {
                "noise_order_u": 0.0,
                "next_value_index": len(self.checkpoint.payload.value_grid),
            },
        )
        for arguments in invalid_calls:
            branch = restore_detached_frozen_branch(self.checkpoint)
            untouched = restore_detached_frozen_branch(self.checkpoint)
            before = _branch_snapshot(branch)
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    branch.run_next_frozen_policy_period_with_supplied_draws(
                        **arguments
                    )
                self.assertEqual(_branch_snapshot(branch), before)
                valid_branch = branch.run_next_frozen_policy_period_with_supplied_draws(
                    noise_order_u=0.1,
                    next_value_index=3,
                )
                valid_untouched = untouched.run_next_frozen_policy_period_with_supplied_draws(
                    noise_order_u=0.1,
                    next_value_index=3,
                )
                self.assertEqual(valid_branch, valid_untouched)

    def test_controlled_source_rejects_supplied_draws_without_mutation(self) -> None:
        """A live Step-28 source cannot bypass its controller.

        仍由第 28 步管理的源 session 不能绕过 controller。
        """

        # The demo checkpoint proves capture, while its helper does not expose
        # the source controller. Build a fresh source through Step35B's helper
        # logic by restoring equal branches, then use a minimal controller-owned
        # source from Step35A's public test/demo helper. / demo helper 不返回源
        # controller，因此这里使用第 35A 步的小型边界 helper。
        from steps.step_35a_converged_market_checkpoint import _build_demo_boundary

        controller = _build_demo_boundary()
        self.assertIs(controller.phase, SessionPhase.MEASUREMENT)
        source = controller.session
        before = _branch_snapshot(source)
        with self.assertRaises(RuntimeError):
            source.run_next_frozen_policy_period_with_supplied_draws(
                noise_order_u=0.0,
                next_value_index=0,
            )
        self.assertEqual(_branch_snapshot(source), before)

    def test_supplied_next_value_has_no_lookahead(self) -> None:
        """Changing only v-next changes next state, not today's market.

        只改变下一价值会改变下一状态，但不会反向改变今天的市场。
        """

        first, second = restore_two_independent_branches(self.checkpoint)
        low_next = first.run_next_frozen_policy_period_with_supplied_draws(
            noise_order_u=0.25,
            next_value_index=0,
        )
        high_next = second.run_next_frozen_policy_period_with_supplied_draws(
            noise_order_u=0.25,
            next_value_index=len(self.checkpoint.payload.value_grid) - 1,
        )
        current_outcome_low = (
            low_next.current_state_indexes,
            low_next.action_indexes,
            low_next.raw_orders_x,
            low_next.noise_order_u,
            low_next.total_order_flow_y,
            low_next.continuous_price_p,
            low_next.insensitive_order_z,
            low_next.profits,
        )
        current_outcome_high = (
            high_next.current_state_indexes,
            high_next.action_indexes,
            high_next.raw_orders_x,
            high_next.noise_order_u,
            high_next.total_order_flow_y,
            high_next.continuous_price_p,
            high_next.insensitive_order_z,
            high_next.profits,
        )
        self.assertEqual(current_outcome_low, current_outcome_high)
        self.assertNotEqual(low_next.next_value_index, high_next.next_value_index)


class TestPairedRunnerValidation(unittest.TestCase):
    """Reject bad path requests before restoring two makers. / 在恢复两个做市商前拒绝错误请求。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoint = _build_demo_checkpoint()
        cls.calibration = calibrate_uniform_noise_shock(2.0, 0.5, 0.5)

    def test_invalid_requests_do_not_call_expensive_restore(self) -> None:
        """Cheap validation precedes two full checkpoint restores.

        便宜的输入检查必须先于两次完整 checkpoint 恢复。
        """

        invalid_cases = (
            {"irf_experiment_seed": True, "path_index": 0, "shock_calibration": self.calibration},
            {"irf_experiment_seed": 1, "path_index": 10_000, "shock_calibration": self.calibration},
            {"irf_experiment_seed": 1, "path_index": 0, "shock_calibration": 0.1},
            {
                "irf_experiment_seed": 1,
                "path_index": 0,
                "shock_calibration": self.calibration,
                "branch_execution_order": "misspelled",
            },
        )
        for arguments in invalid_cases:
            with self.subTest(arguments=arguments):
                with patch.object(
                    step35b,
                    "restore_two_independent_branches",
                ) as restore_mock:
                    with self.assertRaises((TypeError, ValueError)):
                        run_one_paired_irf_path(
                            self.checkpoint,
                            **arguments,
                        )
                    restore_mock.assert_not_called()

    def test_tampered_calibration_receipts_fail_before_restore(self) -> None:
        """Frozen copies are recomputed; forged arithmetic/provenance is rejected.

        冻结副本也会被重新计算；伪造的算术或数据来源声明会被拒绝。
        """

        tampered_receipts = (
            replace(self.calibration, absolute_noise_shock=True),
            replace(self.calibration, absolute_noise_shock=999.0),
            replace(self.calibration, target_normalized_price_deviation=0.99),
            replace(self.calibration, implied_oriented_price_increment=123.0),
            replace(self.calibration, implied_target_error=123.0),
            replace(self.calibration, aggregate_provenance_verified=True),
            replace(
                self.calibration,
                underlying_price_impact_positivity_verified_from_raw_paths=True,
            ),
            replace(self.calibration, protocol_adds_shock_to_ordinary_noise=False),
            replace(self.calibration, formula="forged"),
        )
        for tampered in tampered_receipts:
            with self.subTest(tampered=tampered):
                with patch.object(
                    step35b,
                    "restore_two_independent_branches",
                ) as restore_mock:
                    with self.assertRaises((TypeError, ValueError)):
                        run_one_paired_irf_path(
                            self.checkpoint,
                            irf_experiment_seed=1,
                            path_index=0,
                            shock_calibration=tampered,
                        )
                    restore_mock.assert_not_called()

    def test_nonpaper_target_is_allowed_but_flagged_as_sensitivity(self) -> None:
        """A 2% wiring/sensitivity run cannot masquerade as the paper's 1.2% run.

        2% 接线或敏感性运行不能冒充论文的 1.2% 运行。
        """

        sensitivity = calibrate_uniform_noise_shock(
            2.0,
            0.5,
            0.5,
            target_normalized_price_deviation=0.02,
        )
        receipt = run_one_paired_irf_path(
            self.checkpoint,
            irf_experiment_seed=1,
            path_index=0,
            shock_calibration=sensitivity,
        )
        self.assertNotEqual(
            sensitivity.target_normalized_price_deviation,
            PAPER_TARGET_PRICE_DEVIATION,
        )
        self.assertFalse(receipt.paper_1_2_percent_target_used)
        self.assertFalse(receipt.classification_ready)
        self.assertFalse(receipt.calibration_aggregate_provenance_verified)

    def test_package_qualified_public_objects_have_one_class_identity(self) -> None:
        """Canonical package imports accept Step34 and Step35A public objects.

        统一 package 导入可以正确接收第 34、35A 步的公共对象。
        """

        self.assertIs(type(self.calibration), step35b.UniformShockCalibration)
        self.assertIs(type(self.checkpoint), step35b.ConvergedMarketCheckpoint)
        receipt = run_one_paired_irf_path(
            self.checkpoint,
            irf_experiment_seed=1,
            path_index=0,
            shock_calibration=self.calibration,
        )
        self.assertTrue(receipt.calibration_arithmetic_verified)

    def test_explicit_branch_orders_produce_identical_named_records(self) -> None:
        """An independent direct comparison supports the audit wrapper.

        独立直接比较支持顺序审计 wrapper 的结论。
        """

        control_first = run_one_paired_irf_path(
            self.checkpoint,
            irf_experiment_seed=1,
            path_index=1,
            shock_calibration=self.calibration,
            branch_execution_order=CONTROL_FIRST,
        )
        treatment_first = run_one_paired_irf_path(
            self.checkpoint,
            irf_experiment_seed=1,
            path_index=1,
            shock_calibration=self.calibration,
            branch_execution_order=TREATMENT_FIRST,
        )
        self.assertEqual(control_first.draw_schedule, treatment_first.draw_schedule)
        self.assertEqual(control_first.periods, treatment_first.periods)
        self.assertEqual(
            control_first.control_final_state,
            treatment_first.control_final_state,
        )
        self.assertEqual(
            control_first.treatment_final_state,
            treatment_first.treatment_final_state,
        )


if __name__ == "__main__":
    unittest.main()
