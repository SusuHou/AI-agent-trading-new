"""Independent tests for Step 35E. / 第 35E 步独立自动测试。"""

from dataclasses import FrozenInstanceError, replace
from math import fsum
from pathlib import Path
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
from step_25_one_market_period import build_paper_inputs
from step_26_reproducible_random_streams import build_randomized_paper_session
from step_28_session_phases import SessionPhase, SessionPhaseController
from steps.step_35c_irf_long_run_baseline import (
    OnlineIRFLongRunBaselineScorer,
)
from steps.step_35d_unshocked_t3_calibration_paths import (
    run_unshocked_t3_calibration_paths,
)
import steps.step_35e_cell_shock_calibration as step35e
from steps.step_35e_cell_shock_calibration import (
    UndefinedCellShockCalibrationError,
    calculate_cell_shock_calibration_arithmetic,
    calibrate_experiment_cell_uniform_shock,
    validate_cell_shock_calibration_arithmetic,
    validate_experiment_cell_shock_calibration_receipt,
)


class TestStep35EHandArithmetic(unittest.TestCase):
    """Check the equations without running a market. / 不运行市场，直接核对方程。"""

    def test_exact_level_hand_calculation_and_shortcut_sensitivity(self) -> None:
        """The exact rule hits the level target; the shortcut need not.

        exact 规则命中价格水平目标；shortcut 不一定命中。
        """

        result = calculate_cell_shock_calibration_arithmetic(
            mean_long_run_oriented_price=3.0,
            mean_unshocked_t3_oriented_price=2.8,
            mean_actual_t3_price_impact_lambda=1.0,
            minimum_actual_t3_price_impact_lambda=0.5,
        )
        self.assertAlmostEqual(result.exact_level_absolute_noise_shock, 0.236)
        self.assertAlmostEqual(
            result.increment_shortcut_absolute_noise_shock,
            0.036,
        )
        self.assertAlmostEqual(
            result.exact_level_achieved_normalized_level_deviation,
            0.012,
        )
        self.assertAlmostEqual(
            result.increment_shortcut_achieved_normalized_level_deviation,
            (2.8 + 0.036 - 3.0) / 3.0,
        )
        self.assertFalse(result.formulas_coincide)
        validate_cell_shock_calibration_arithmetic(result)

    def test_formulas_coincide_when_unshocked_t3_equals_long_run(self) -> None:
        """Show the special assumption hidden inside the shortcut.

        展示 shortcut 隐含的特殊假设：无冲击 t=3 均值等于长期均值。
        """

        result = calculate_cell_shock_calibration_arithmetic(2.0, 2.0, 0.5, 0.2)
        self.assertAlmostEqual(result.exact_level_absolute_noise_shock, 0.048)
        self.assertAlmostEqual(result.increment_shortcut_absolute_noise_shock, 0.048)
        self.assertTrue(result.formulas_coincide)

    def test_invalid_domains_are_rejected_without_absolute_value_fixes(self) -> None:
        """Never hide a bad denominator or negative required shock with abs().

        绝不使用 abs() 掩盖错误分母或负的所需冲击。
        """

        invalid_cases = (
            (0.0, -0.1, 1.0, 0.5, 0.012),
            (1.0, 0.5, 0.0, 0.0, 0.012),
            (1.0, 0.5, 1.0, 0.0, 0.012),
            (1.0, 0.5, 1.0, 2.0, 0.012),
            (1.0, 0.5, 1.0, 0.5, 0.0),
            # P0 already exceeds 1.012 * P, so a positive shock cannot hit it.
            # / P0 已超过目标水平，正冲击无法命中目标。
            (1.0, 1.1, 1.0, 0.5, 0.012),
            (float("nan"), 0.5, 1.0, 0.5, 0.012),
            (1.0, 0.5, float("inf"), 0.5, 0.012),
        )
        for values in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError, ArithmeticError)):
                    calculate_cell_shock_calibration_arithmetic(*values)

    def test_arithmetic_is_frozen_and_changed_copy_is_rejected(self) -> None:
        """A dataclass copy cannot change one answer silently. / dataclass 副本不能偷偷改答案。"""

        result = calculate_cell_shock_calibration_arithmetic(3.0, 2.8, 1.0, 0.5)
        with self.assertRaises(FrozenInstanceError):
            result.exact_level_absolute_noise_shock = 99.0  # type: ignore[misc]
        with self.assertRaises(ValueError):
            validate_cell_shock_calibration_arithmetic(
                replace(result, exact_level_absolute_noise_shock=99.0)
            )


class TestStep35ERealReceiptPooling(unittest.TestCase):
    """Pool short but genuine Step-35C/35D receipts. / 汇总短小但真实的35C/35D凭证。"""

    EXPERIMENT_SEED = 20_260_829
    IRF_SEED = 20_260_835
    # This stable key gives the two-session debug fixture a positive exact-level
    # numerator.  A different short sample can legitimately start above the
    # target and must be rejected by the production function. / 这个固定标签让
    # 两个 session 的调试样本具有正的 exact-level 分子；其他很短的样本可能
    # 合法地从目标之上开始，生产函数必须拒绝那种情况。
    CELL_KEY = "step35c_demo_only"

    @classmethod
    def _build_source_receipt(
        cls,
        *,
        session_index: int,
        cell_key: str,
        measurement_periods: int,
        path_count: int,
    ):
        """Build one real short receipt with stable actions. / 用稳定动作建立一份真实短凭证。"""

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
            experiment_seed=cls.EXPERIMENT_SEED,
            experiment_cell_key=cell_key,
            session_index=session_index,
        )
        scorer = OnlineIRFLongRunBaselineScorer(session)
        controller = SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=2,
            measurement_periods_required=measurement_periods,
            measurement_sink=scorer.observe,
        )
        while controller.phase is SessionPhase.TRAINING:
            if controller.training_periods_completed >= 2:
                raise AssertionError("Stable debug session did not converge.")
            controller.run_next_period()
        checkpoint = scorer.capture_and_bind_convergence_checkpoint(controller)
        controller.run_until_complete()
        scorer.finalize(controller)
        return run_unshocked_t3_calibration_paths(
            checkpoint,
            baseline_scorer=scorer,
            irf_experiment_seed=cls.IRF_SEED,
            path_count=path_count,
        )

    @classmethod
    def setUpClass(cls) -> None:
        """Use unequal counts so a wrong unweighted mean is detectable.

        故意使用不同样本数，使错误的未加权均值可以被发现。
        """

        cls.receipt_0 = cls._build_source_receipt(
            session_index=0,
            cell_key=cls.CELL_KEY,
            measurement_periods=200,
            path_count=200,
        )
        cls.receipt_1 = cls._build_source_receipt(
            session_index=1,
            cell_key=cls.CELL_KEY,
            measurement_periods=400,
            path_count=400,
        )
        cls.other_cell_receipt_1 = cls._build_source_receipt(
            session_index=1,
            cell_key="different_step35e_cell",
            measurement_periods=400,
            path_count=400,
        )

    def test_count_weighted_pooling_and_actual_t3_lambda(self) -> None:
        """Pool discarded rows by their counts and use Step-35D lambda_3.

        按底层行数加权，并使用第 35D 步真正的 lambda_3。
        """

        result = calibrate_experiment_cell_uniform_shock(
            (self.receipt_0, self.receipt_1),
            expected_session_count=2,
        )
        receipts = (self.receipt_0, self.receipt_1)
        expected_long_run = fsum(
            receipt.long_run_mean_oriented_price
            * receipt.long_run_baseline_receipt.measurement_periods_scored
            for receipt in receipts
        ) / 600
        expected_t3_price = fsum(
            receipt.mean_unshocked_t3_oriented_price * receipt.paths_executed
            for receipt in receipts
        ) / 600
        expected_t3_lambda = fsum(
            receipt.mean_t3_price_impact_lambda * receipt.paths_executed
            for receipt in receipts
        ) / 600
        self.assertEqual(result.total_long_run_measurement_rows, 600)
        self.assertEqual(result.total_unshocked_t3_paths, 600)
        self.assertEqual(result.pooled_long_run_mean_oriented_price, expected_long_run)
        self.assertEqual(result.pooled_unshocked_t3_mean_oriented_price, expected_t3_price)
        self.assertEqual(
            result.pooled_actual_t3_mean_price_impact_lambda,
            expected_t3_lambda,
        )
        # Long-run lambda is diagnostic only and is never used as the divisor.
        # / 长期 lambda 只是诊断量，绝不作为这里的分母。
        long_run_diagnostic_lambda = fsum(
            receipt.long_run_baseline_receipt.mean_price_impact_lambda
            * receipt.long_run_baseline_receipt.measurement_periods_scored
            for receipt in receipts
        ) / 600
        self.assertEqual(
            result.arithmetic.mean_actual_t3_price_impact_lambda,
            expected_t3_lambda,
        )
        self.assertNotEqual(
            expected_t3_lambda,
            long_run_diagnostic_lambda,
            "This fixture must detect accidental use of Step-35C long-run lambda.",
        )
        self.assertEqual(
            result.parameter_snapshot,
            self.receipt_0.long_run_baseline_receipt.parameter_snapshot,
        )
        self.assertTrue(np.isfinite(long_run_diagnostic_lambda))
        validate_experiment_cell_shock_calibration_receipt(result)

    def test_order_invariance_no_input_mutation_and_one_common_magnitude(self) -> None:
        """Caller order changes neither the receipt nor its list. / 调用顺序不改变结果或输入列表。"""

        forward = calibrate_experiment_cell_uniform_shock(
            (self.receipt_0, self.receipt_1),
            expected_session_count=2,
        )
        caller_list = [self.receipt_1, self.receipt_0]
        before = tuple(caller_list)
        reverse = calibrate_experiment_cell_uniform_shock(
            caller_list,
            expected_session_count=2,
        )
        self.assertEqual(tuple(caller_list), before)
        self.assertEqual(reverse, forward)
        self.assertEqual(reverse.ordered_session_indexes, (0, 1))
        self.assertEqual(
            reverse.selected_absolute_noise_shock,
            reverse.arithmetic.exact_level_absolute_noise_shock,
        )
        self.assertTrue(reverse.one_common_magnitude_selected_for_entire_cell)
        self.assertTrue(reverse.uniform_cell_shock_calibrated)
        self.assertFalse(reverse.shock_applied)
        self.assertEqual(reverse.treatment_paths_executed, 0)
        self.assertFalse(reverse.t4_response_aggregated)

    def test_debug_receipt_never_claims_formal_paper_scale(self) -> None:
        """Two sessions can test code, not reproduce 1,000 sessions. / 两个 session 只能测代码。"""

        result = calibrate_experiment_cell_uniform_shock(
            (self.receipt_0, self.receipt_1),
            expected_session_count=2,
        )
        self.assertFalse(result.paper_1000_sessions_verified)
        self.assertFalse(result.paper_10000_paths_per_session_verified)
        self.assertFalse(result.paper_scale_source_receipts_verified)
        self.assertFalse(result.formal_cross_session_seed_namespace_audit_verified)
        self.assertFalse(result.ready_for_formal_step35f)
        self.assertFalse(result.paper_target_observed_on_shocked_paths)
        self.assertFalse(result.classification_ready)
        self.assertFalse(result.paper_figure_ready)
        self.assertEqual(result.unique_session_seed_count, 2)
        self.assertEqual(result.unique_base_stream_seed_count, 14)

    def test_duplicate_missing_and_mixed_cell_receipts_are_rejected(self) -> None:
        """Never silently drop or combine the wrong sessions. / 绝不悄悄遗漏或混合错误 session。"""

        bad_inputs = (
            ((self.receipt_0,), 2),
            ((self.receipt_0, self.receipt_0), 2),
            ((self.receipt_0, self.other_cell_receipt_1), 2),
        )
        for receipts, expected in bad_inputs:
            with self.subTest(receipts=len(receipts)):
                with self.assertRaises((TypeError, ValueError)):
                    calibrate_experiment_cell_uniform_shock(
                        receipts,
                        expected_session_count=expected,
                    )

    def test_tampered_source_or_cell_receipt_is_rejected(self) -> None:
        """Both input and output receipts remain tamper-evident. / 输入与输出凭证都可发现篡改。"""

        forged_source = replace(
            self.receipt_0,
            mean_t3_price_impact_lambda=(
                self.receipt_0.mean_t3_price_impact_lambda + 1.0
            ),
        )
        with self.assertRaises(ValueError):
            calibrate_experiment_cell_uniform_shock(
                (forged_source, self.receipt_1),
                expected_session_count=2,
            )
        result = calibrate_experiment_cell_uniform_shock(
            (self.receipt_0, self.receipt_1),
            expected_session_count=2,
        )
        with self.assertRaises(FrozenInstanceError):
            result.selected_absolute_noise_shock = 99.0  # type: ignore[misc]
        with self.assertRaises(ValueError):
            validate_experiment_cell_shock_calibration_receipt(
                replace(result, selected_absolute_noise_shock=99.0)
            )

    def test_invalid_expected_session_counts_and_targets(self) -> None:
        """Counts and targets are explicit, never guessed. / session 数与目标必须明确。"""

        for invalid_count in (0, 1_001, True, 1.5):
            with self.subTest(invalid_count=invalid_count):
                with self.assertRaises((TypeError, ValueError)):
                    calibrate_experiment_cell_uniform_shock(
                        (self.receipt_0, self.receipt_1),
                        expected_session_count=invalid_count,  # type: ignore[arg-type]
                    )
        with self.assertRaises(ValueError):
            calibrate_experiment_cell_uniform_shock(
                (self.receipt_0, self.receipt_1),
                expected_session_count=2,
                target_normalized_price_level_deviation=0.02,
            )


if __name__ == "__main__":
    unittest.main()
