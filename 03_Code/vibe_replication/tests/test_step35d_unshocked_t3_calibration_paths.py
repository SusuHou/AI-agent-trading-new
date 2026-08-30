"""Independent tests for Step 35D. / 第 35D 步独立自动测试。"""

from dataclasses import FrozenInstanceError, replace
from math import fsum
from pathlib import Path
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
from step_24_adaptive_market_maker_price import calculate_adaptive_price_impact
from step_26_reproducible_random_streams import build_randomized_paper_session
from step_28_session_phases import SessionPhase, SessionPhaseController
from steps.step_35a_converged_market_checkpoint import (
    restore_detached_frozen_branch,
)
from steps.step_35b_paired_irf_path import (
    build_paired_path_draw_schedule,
    build_paired_path_draw_schedule_from_verified_context,
    prepare_verified_paired_path_schedule_context,
)
import steps.step_35b_paired_irf_path as step35b
from steps.step_35c_irf_long_run_baseline import (
    OnlineIRFLongRunBaselineScorer,
)
import steps.step_35d_unshocked_t3_calibration_paths as step35d
import steps.step_36e_complete_measurement_runner as step36e
from steps.step_35d_unshocked_t3_calibration_paths import (
    CALIBRATION_PERIOD_COUNT,
    OnlineUnshockedT3Moments,
    ReusableUnshockedT3Workspace,
    run_unshocked_t3_calibration_paths,
    validate_unshocked_t3_session_calibration_receipt,
)


class TestStep35DUnshockedCalibration(unittest.TestCase):
    """Exercise real checkpoint, path, rollback, and receipt wiring.

    使用真实 checkpoint、路径、回滚与凭证接线进行测试。
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Build one short live Step-35C source shared by read-only tests.

        建立一个短小但真实的第 35C 步来源，供只读测试共享。
        """

        parameters = PaperParameters()
        value_grid, price_grid, actions, initial_q, prehistory = build_paper_inputs(
            parameters
        )
        stable_q = np.zeros_like(initial_q)
        stable_q[:, 0] = 1_000_000_000.0
        session = build_randomized_paper_session(
            parameters=parameters,
            value_grid=value_grid,
            price_grid=price_grid,
            action_multipliers=actions,
            initial_q_table=stable_q,
            prehistory=prehistory,
            experiment_seed=20_260_829,
            experiment_cell_key="step35d_tests",
            session_index=7,
        )
        scorer = OnlineIRFLongRunBaselineScorer(session)
        controller = SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=2,
            measurement_periods_required=30,
            measurement_sink=scorer.observe,
        )
        while controller.phase is SessionPhase.TRAINING:
            if controller.training_periods_completed >= 2:
                raise AssertionError("The stable fixture did not converge.")
            controller.run_next_period()
        checkpoint = scorer.capture_and_bind_convergence_checkpoint(controller)
        controller.run_until_complete()
        scorer.finalize(controller)
        cls.parameters = parameters
        cls.source_session = session
        cls.controller = controller
        cls.scorer = scorer
        cls.checkpoint = checkpoint
        cls.irf_seed = 20_260_835

    def build_workspace(self) -> ReusableUnshockedT3Workspace:
        """Create one reusable branch from the shared immutable source.

        从共享不可变来源建立一个可复用分支。
        """

        return ReusableUnshockedT3Workspace(
            self.checkpoint,
            baseline_scorer=self.scorer,
            irf_experiment_seed=self.irf_seed,
        )

    def test_preverified_schedule_is_exactly_the_step35b_schedule(self) -> None:
        """The fast schedule factory must not define new randomness.

        快速抽样器绝不能偷偷定义另一套随机数。
        """

        context = prepare_verified_paired_path_schedule_context(
            self.checkpoint,
            irf_experiment_seed=self.irf_seed,
        )
        for path_index in (0, 1, 37, 9_999):
            with self.subTest(path_index=path_index):
                reference = build_paired_path_draw_schedule(
                    self.checkpoint,
                    irf_experiment_seed=self.irf_seed,
                    path_index=path_index,
                )
                fast = build_paired_path_draw_schedule_from_verified_context(
                    context,
                    path_index=path_index,
                )
                self.assertEqual(fast, reference)

    def test_live_baseline_requires_the_exact_bound_checkpoint_object(self) -> None:
        """Equal-looking copied metadata is not live stream provenance.

        看起来相同的复制数据不等于实时数据流来源证明。
        """

        self.assertIs(
            self.scorer.verified_live_result_for_step35d(self.checkpoint),
            self.scorer.finalize(self.controller),
        )
        copied_wrapper = replace(self.checkpoint)
        with self.assertRaises(ValueError):
            self.scorer.verified_live_result_for_step35d(copied_wrapper)

    def test_hot_reusable_path_exactly_matches_a_fresh_restore(self) -> None:
        """Compare all three observations, not only the final mean.

        比较三期全部 observation，而不只比较最终均值。
        """

        path_index = 37
        schedule = build_paired_path_draw_schedule(
            self.checkpoint,
            irf_experiment_seed=self.irf_seed,
            path_index=path_index,
        )
        fresh = restore_detached_frozen_branch(self.checkpoint)
        expected_rows = []
        pre_t3_estimates = None
        expected_pre_t3_lambda = None
        for offset in range(CALIBRATION_PERIOD_COUNT):
            if offset == CALIBRATION_PERIOD_COUNT - 1:
                pre_t3_estimates = fresh.market_maker.estimates()
                expected_pre_t3_lambda = calculate_adaptive_price_impact(
                    pre_t3_estimates,
                    self.parameters.pricing_error_weight,
                )
            expected_rows.append(
                fresh.run_next_frozen_policy_period_with_supplied_draws(
                noise_order_u=schedule.ordinary_noise_orders_u[offset],
                next_value_index=schedule.next_value_indexes[offset],
            )
            )
        expected = tuple(expected_rows)
        self.assertIsNotNone(pre_t3_estimates)
        self.assertEqual(expected[-1].price_impact_lambda_hat, expected_pre_t3_lambda)
        post_t3_lambda = calculate_adaptive_price_impact(
            fresh.market_maker.estimates(),
            self.parameters.pricing_error_weight,
        )
        self.assertNotEqual(expected_pre_t3_lambda, post_t3_lambda)
        workspace = self.build_workspace()
        actual = workspace.run_path(path_index)
        self.assertEqual(actual.observations, expected)
        self.assertFalse(actual.shock_applied)
        self.assertFalse(actual.treatment_branch_run)
        self.assertFalse(actual.t4_executed)
        workspace.verify_exact_checkpoint_reset()
        repeated = workspace.run_path(path_index)
        self.assertEqual(repeated, actual)
        workspace.close_and_verify()

    def test_path_after_a_different_path_matches_a_standalone_run(self) -> None:
        """Rollback isolation is path-independent, not only replay-stable.

        回滚隔离不只保证重复同一路径，也不受前一条不同路径影响。
        """

        after_other_workspace = self.build_workspace()
        after_other_workspace.run_path(91)
        after_other = after_other_workspace.run_path(7)
        after_other_workspace.close_and_verify()

        standalone_workspace = self.build_workspace()
        standalone = standalone_workspace.run_path(7)
        standalone_workspace.close_and_verify()
        self.assertEqual(after_other, standalone)

    def test_online_reducer_matches_independent_batch_arithmetic(self) -> None:
        """Hand off five paths one by one and compare with math.fsum.

        逐条交给汇总器五条路径，再与独立 math.fsum 比较。
        """

        workspace = self.build_workspace()
        results = [workspace.run_path(index) for index in range(5)]
        moments = OnlineUnshockedT3Moments(
            schedule_context=workspace.schedule_context,
            value_mean=self.parameters.value_mean,
            pricing_error_weight=self.parameters.pricing_error_weight,
        )
        for result in results:
            moments.add(result)
        mean_lambda, mean_price, minimum_lambda, bad_count, _ = (
            moments.summarized_values()
        )
        self.assertEqual(
            mean_lambda,
            fsum(result.t3_price_impact_lambda_hat for result in results)
            / len(results),
        )
        self.assertEqual(
            mean_price,
            fsum(result.t3_unshocked_oriented_price for result in results)
            / len(results),
        )
        self.assertEqual(
            minimum_lambda,
            min(result.t3_price_impact_lambda_hat for result in results),
        )
        self.assertEqual(
            bad_count,
            sum(result.t3_price_impact_lambda_hat <= 0.0 for result in results),
        )
        # The reducer owns only scalar/tuple/hash state, never a result list.
        # / 汇总器只保存标量、tuple 与哈希，不保存路径列表。
        self.assertFalse(any(isinstance(value, list) for value in vars(moments).values()))
        workspace.close_and_verify()

    def test_reducer_rejects_forged_manifest_and_ols_inputs_atomically(self) -> None:
        """Discarded paths remain bound to their parent seeds and OLS slopes.

        已丢弃路径仍必须绑定父种子身份与 OLS 斜率。
        """

        workspace = self.build_workspace()
        result = workspace.run_path(0)
        bad_manifest = replace(
            result.seed_manifest,
            irf_experiment_seed=result.seed_manifest.irf_experiment_seed + 1,
        )
        forged_manifest_result = replace(result, seed_manifest=bad_manifest)
        t3 = result.observations[-1]
        forged_t3 = replace(t3, xi_1_hat=999.0, gamma_1_hat=-999.0)
        forged_ols_result = replace(
            result,
            observations=(*result.observations[:-1], forged_t3),
        )
        for forged in (forged_manifest_result, forged_ols_result):
            moments = OnlineUnshockedT3Moments(
                schedule_context=workspace.schedule_context,
                value_mean=self.parameters.value_mean,
                pricing_error_weight=self.parameters.pricing_error_weight,
            )
            before = moments.audit_state()
            with self.assertRaises(ValueError):
                moments.add(forged)
            self.assertEqual(moments.audit_state(), before)
        workspace.close_and_verify()

    def test_cross_path_seed_collision_is_rejected_before_execution(self) -> None:
        """Ten thousand repeated draws cannot masquerade as 10,000 paths.

        一万次重复同一抽样不能冒充一万条独立路径。
        """

        def colliding_seed(*args, **kwargs):
            stream_label = args[-1]
            return 1 if stream_label == b"ordinary_noise" else 2

        with patch.object(
            step35b,
            "_derive_paired_child_seed",
            side_effect=colliding_seed,
        ):
            with self.assertRaisesRegex(RuntimeError, "collided"):
                self.build_workspace()

    def test_out_of_order_path_is_rejected_atomically(self) -> None:
        """Planned identities cannot be counted as executed canonical paths.

        计划中的路径身份不能冒充已执行的标准路径。
        """

        workspace = self.build_workspace()
        result_one = workspace.run_path(1)
        moments = OnlineUnshockedT3Moments(
            schedule_context=workspace.schedule_context,
            value_mean=self.parameters.value_mean,
            pricing_error_weight=self.parameters.pricing_error_weight,
        )
        before = moments.audit_state()
        with self.assertRaises(ValueError):
            moments.add(result_one)
        self.assertEqual(moments.audit_state(), before)
        workspace.close_and_verify()

    def test_interrupted_path_is_rolled_back_but_workspace_is_poisoned(self) -> None:
        """A failure cannot contaminate or silently resume the next path.

        一次失败不能污染下一条路径，也不能悄悄继续运行。
        """

        workspace = self.build_workspace()
        original = workspace._branch.run_next_frozen_policy_period_with_supplied_draws
        calls = 0

        def fail_on_second_period(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected failure")
            return original(**kwargs)

        with patch.object(
            workspace._branch,
            "run_next_frozen_policy_period_with_supplied_draws",
            side_effect=fail_on_second_period,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                workspace.run_path(0)
        self.assertTrue(workspace.is_poisoned)
        workspace.verify_exact_checkpoint_reset()
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            workspace.run_path(0)

    def test_debug_receipt_is_truthful_and_tamper_evident(self) -> None:
        """A short run is useful but must not claim paper completion.

        短调试有用，但不得声称已完成论文规模。
        """

        receipt = run_unshocked_t3_calibration_paths(
            self.checkpoint,
            baseline_scorer=self.scorer,
            irf_experiment_seed=self.irf_seed,
            path_count=25,
        )
        validate_unshocked_t3_session_calibration_receipt(receipt)
        self.assertEqual(receipt.paths_executed, 25)
        self.assertEqual(receipt.successful_transaction_rollbacks, 25)
        self.assertEqual(receipt.full_checkpoint_reset_audits, 2)
        self.assertEqual(receipt.raw_path_results_retained, 0)
        self.assertFalse(receipt.paper_paths_per_session_count_matched_for_calibration)
        self.assertFalse(receipt.ready_for_formal_paper_cell_aggregation)
        self.assertFalse(receipt.shock_applied)
        self.assertEqual(receipt.treatment_paths_executed, 0)
        self.assertFalse(receipt.t4_response_aggregated)
        self.assertFalse(receipt.classification_ready)
        with self.assertRaises(FrozenInstanceError):
            receipt.paths_executed = 10_000  # type: ignore[misc]
        with self.assertRaises(ValueError):
            validate_unshocked_t3_session_calibration_receipt(
                replace(receipt, paths_executed=10_000)
            )
        for changed_fields in (
            {"executed_path_fields_sha256": "not-a-sha256"},
            {"irf_experiment_seed": -1},
        ):
            forged = replace(receipt, **changed_fields, receipt_payload_sha256="")
            forged = replace(
                forged,
                receipt_payload_sha256=step35d._receipt_payload_digest(forged),
            )
            with self.assertRaises(ValueError):
                validate_unshocked_t3_session_calibration_receipt(forged)

    def test_receipt_checksum_survives_explicit_dealiasing(self) -> None:
        """Equal scientific values must hash equally after wire reconstruction.

        wire 重建拆开共享内存后，相等科研数值仍必须得到相同 checksum。
        """

        receipt = run_unshocked_t3_calibration_paths(
            self.checkpoint,
            baseline_scorer=self.scorer,
            irf_experiment_seed=self.irf_seed,
            path_count=3,
        )
        # The live receipt happens to reuse the same seed-manifest object in
        # two fields. The wire format reconstructs two equal, independent
        # objects. That invisible memory-layout change must not affect the
        # scientific digest. / 实时 receipt 的两个字段碰巧共用同一个 seed
        # manifest；wire 读取后变成两个相等但独立的对象。这种看不见的内存
        # 变化不能影响科研摘要。
        self.assertIs(
            receipt.source_seed_manifest,
            receipt.long_run_baseline_receipt.session_seed_manifest,
        )
        rebuilt = step36e._wire_decode(
            step36e._wire_encode(receipt),
            {
                type(receipt).__name__: type(receipt),
            },
        )
        self.assertEqual(rebuilt, receipt)
        self.assertIsNot(
            rebuilt.source_seed_manifest,
            rebuilt.long_run_baseline_receipt.session_seed_manifest,
        )
        self.assertEqual(
            step35d._receipt_payload_digest(rebuilt),
            receipt.receipt_payload_sha256,
        )
        validate_unshocked_t3_session_calibration_receipt(rebuilt)

    def test_aggregate_replays_and_changes_with_experiment_seed(self) -> None:
        """Same identity replays exactly; a new IRF seed changes the paths.

        同一身份精确重放；不同 IRF 种子会改变路径。
        """

        first = run_unshocked_t3_calibration_paths(
            self.checkpoint,
            baseline_scorer=self.scorer,
            irf_experiment_seed=self.irf_seed,
            path_count=20,
        )
        replay = run_unshocked_t3_calibration_paths(
            self.checkpoint,
            baseline_scorer=self.scorer,
            irf_experiment_seed=self.irf_seed,
            path_count=20,
        )
        changed = run_unshocked_t3_calibration_paths(
            self.checkpoint,
            baseline_scorer=self.scorer,
            irf_experiment_seed=self.irf_seed + 1,
            path_count=20,
        )
        self.assertEqual(first, replay)
        self.assertNotEqual(
            first.executed_path_fields_sha256,
            changed.executed_path_fields_sha256,
        )

    def test_exactly_10000_executions_claim_count_only_not_full_irf(self) -> None:
        """Exercise all canonical indexes without exaggerating Step 35D.

        实际执行全部标准编号，但仍不夸大第 35D 步的完成范围。
        """

        receipt = run_unshocked_t3_calibration_paths(
            self.checkpoint,
            baseline_scorer=self.scorer,
            irf_experiment_seed=self.irf_seed,
            path_count=10_000,
        )
        self.assertEqual(receipt.paths_executed, 10_000)
        self.assertTrue(receipt.paper_paths_per_session_count_matched_for_calibration)
        # The fixture uses only 30 baseline rows and a two-period convergence
        # threshold, so this remains a scale test—not a paper result. / fixture
        # 只有 30 条基准记录与两期收敛阈值，因此仍只是规模测试。
        self.assertFalse(receipt.paper_measurement_and_convergence_scale_verified)
        self.assertFalse(receipt.ready_for_formal_paper_cell_aggregation)
        self.assertFalse(receipt.full_paper_irf_protocol_verified)
        self.assertFalse(receipt.paper_figure_ready)

    def test_invalid_path_counts_and_shock_shortcuts_are_absent(self) -> None:
        """Step 35D cannot call a shock or classifier by accident.

        第 35D 步不能意外调用冲击或分类器。
        """

        for invalid_count in (0, 10_001, True, 1.5):
            with self.subTest(invalid_count=invalid_count):
                with self.assertRaises((TypeError, ValueError)):
                    run_unshocked_t3_calibration_paths(
                        self.checkpoint,
                        baseline_scorer=self.scorer,
                        irf_experiment_seed=self.irf_seed,
                        path_count=invalid_count,  # type: ignore[arg-type]
                    )
        self.assertFalse(hasattr(step35d, "add_adverse_shock_to_noise"))
        self.assertFalse(hasattr(step35d, "calibrate_uniform_noise_shock"))
        self.assertFalse(hasattr(step35d, "classify_normalized_order_responses"))


if __name__ == "__main__":
    unittest.main()
