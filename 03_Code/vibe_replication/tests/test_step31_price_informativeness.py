"""Independent tests for Step 31 price informativeness. / Step 31 价格信息效率独立测试。"""

from dataclasses import FrozenInstanceError, replace
from math import isclose
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
from src.step01_value_grid import discrete_value_std
from step_25_one_market_period import build_paper_inputs
from step_26_reproducible_random_streams import build_randomized_paper_session
from step_28_session_phases import SessionPhaseController
from step_30_trading_intensity import OnlineTradingIntensityScorer
from step_31_price_informativeness import (
    PRICE_INFORMATIVENESS_FORMULA,
    PriceInformativenessCalculation,
    UndefinedPriceInformativenessError,
    build_price_informativeness_receipt,
    calculate_price_informativeness,
)


class TestPriceInformativenessFormula(unittest.TestCase):
    """Check equation IA.4.5 independently of the market. / 独立于市场检查 IA.4.5。"""

    def test_hand_example_equals_twenty_five(self) -> None:
        """The printed four-number example has an exact answer. / 打印的四数例有精确答案。"""

        result = calculate_price_informativeness(2, 1.25, 1.0, 0.5)
        self.assertEqual(result.number_of_agents, 2)
        self.assertAlmostEqual(result.aggregate_informed_slope, 2.5)
        self.assertAlmostEqual(result.standard_deviation_ratio, 2.0)
        self.assertAlmostEqual(result.informed_flow_variance, 6.25)
        self.assertAlmostEqual(result.noise_order_variance, 0.25)
        self.assertAlmostEqual(result.price_informativeness, 25.0)
        self.assertAlmostEqual(
            result.price_informativeness,
            (2 * 1.25) ** 2 * (1.0 / 0.5) ** 2,
        )

        # Direct variance check using V=(0,2) and the two Step-30 hand lines:
        # total orders are -1 and 4, with population variance 6.25.
        # / 使用 V=(0,2) 直接核对：总订单为 -1 与 4，总体方差为 6.25。
        total_orders = np.asarray((-1.0, 4.0))
        direct_signal_variance = float(
            np.mean((total_orders - np.mean(total_orders)) ** 2)
        )
        self.assertAlmostEqual(direct_signal_variance, 6.25)
        self.assertAlmostEqual(
            direct_signal_variance / (0.5**2),
            result.price_informativeness,
        )

    def test_squared_scaling_and_zero_intensity(self) -> None:
        """The square in the paper has observable consequences. / 论文中的平方有明确后果。"""

        baseline = calculate_price_informativeness(2, 1.0, 0.8, 0.5)
        twice_agents = calculate_price_informativeness(4, 1.0, 0.8, 0.5)
        twice_noise = calculate_price_informativeness(2, 1.0, 0.8, 1.0)
        negative_policy = calculate_price_informativeness(2, -1.0, 0.8, 0.5)
        zero_policy = calculate_price_informativeness(2, 0.0, 0.8, 0.5)
        thousand_times_noise = calculate_price_informativeness(
            2,
            1.0,
            0.8,
            500.0,
        )
        three_agents = calculate_price_informativeness(3, 1.0, 0.8, 0.5)

        self.assertAlmostEqual(
            twice_agents.price_informativeness,
            4.0 * baseline.price_informativeness,
        )
        self.assertAlmostEqual(
            twice_noise.price_informativeness,
            baseline.price_informativeness / 4.0,
        )
        self.assertAlmostEqual(
            negative_policy.price_informativeness,
            baseline.price_informativeness,
        )
        self.assertEqual(zero_policy.price_informativeness, 0.0)
        self.assertAlmostEqual(
            thousand_times_noise.price_informativeness,
            baseline.price_informativeness / 1_000_000.0,
        )
        self.assertEqual(three_agents.number_of_agents, 3)

    def test_invalid_inputs_and_overflow_are_rejected(self) -> None:
        """Undefined inputs never fabricate an informativeness score. / 无效输入不伪造指标。"""

        for invalid_agents in (0, -1, True, 1.5):
            with self.subTest(number_of_agents=invalid_agents):
                with self.assertRaises(ValueError):
                    calculate_price_informativeness(
                        invalid_agents,  # type: ignore[arg-type]
                        1.0,
                        0.8,
                        0.5,
                    )
        for invalid_intensity in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                calculate_price_informativeness(2, invalid_intensity, 0.8, 0.5)
        for invalid_value_std in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(UndefinedPriceInformativenessError):
                calculate_price_informativeness(2, 1.0, invalid_value_std, 0.5)
        for invalid_noise_std in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(UndefinedPriceInformativenessError):
                calculate_price_informativeness(2, 1.0, 0.8, invalid_noise_std)
        with self.assertRaises(OverflowError):
            calculate_price_informativeness(2, 1e308, 1.0, 0.5)
        with self.assertRaises(OverflowError):
            calculate_price_informativeness(1, 1.0, 1e-200, 1e-100)
        # The true ratio is 1, but both raw variances overflow and therefore
        # cannot be preserved in the auditable result. / 真比率虽为 1，但两个
        # 原始方差都会溢出，无法保存为自洽的审计结果。
        with self.assertRaises(OverflowError):
            calculate_price_informativeness(1, 1e200, 1.0, 1e200)

    def test_pure_result_is_frozen(self) -> None:
        """A reported calculation cannot be silently edited. / 已报告计算不能被悄悄修改。"""

        result = calculate_price_informativeness(2, 1.25, 1.0, 0.5)
        self.assertIsInstance(result, PriceInformativenessCalculation)
        with self.assertRaises(FrozenInstanceError):
            result.price_informativeness = 0.0


class TestStep31SessionIntegration(unittest.TestCase):
    """Connect one completed Step-30 result to Step 31. / 把完整 Step-30 结果接入 Step 31。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.parameters = PaperParameters()
        (
            cls.value_grid,
            cls.price_grid,
            cls.action_multipliers,
            initial_q_table,
            cls.prehistory,
        ) = build_paper_inputs(cls.parameters)
        cls.stable_q_table = np.zeros_like(initial_q_table, dtype=float)
        cls.stable_q_table[:, 0] = 1_000_000_000.0

    def run_completed_session(self, label: str):
        """Return a completed session, Step-30 scorer, and controller. / 返回完整 session、Step-30 scorer 与 controller。"""

        session = build_randomized_paper_session(
            parameters=self.parameters,
            value_grid=self.value_grid,
            price_grid=self.price_grid,
            action_multipliers=self.action_multipliers,
            initial_q_table=self.stable_q_table,
            prehistory=self.prehistory,
            experiment_seed=20260828,
            experiment_cell_key=f"step31_low_noise|{label}",
            session_index=0,
        )
        intensity_scorer = OnlineTradingIntensityScorer(session)
        controller = SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=2,
            measurement_periods_required=100,
            measurement_sink=intensity_scorer.observe,
        )
        controller.run_until_complete(maximum_training_periods=2)
        return session, intensity_scorer, controller

    def test_completed_step30_builds_auditable_step31_receipt(self) -> None:
        """Use the grid std and configured noise from the same session. / 使用同一 session 的网格标准差与设定噪声。"""

        session, intensity_scorer, controller = self.run_completed_session(
            "integration"
        )
        intensity = intensity_scorer.finalize(controller)
        period_before = session.period_number
        random_states_before = session.all_random_states()
        receipt = build_price_informativeness_receipt(
            intensity_scorer,
            controller,
        )
        repeated_receipt = build_price_informativeness_receipt(
            intensity_scorer,
            controller,
        )
        sigma_v_hat = discrete_value_std(
            np.asarray(self.value_grid, dtype=float),
            self.parameters.value_mean,
        )
        expected = (
            (self.parameters.num_speculators * intensity.average_trading_intensity) ** 2
            * (sigma_v_hat / self.parameters.noise_std) ** 2
        )

        self.assertAlmostEqual(receipt.discrete_value_std, sigma_v_hat)
        self.assertAlmostEqual(receipt.discrete_value_std, 0.937970, places=5)
        realized_sample_std = (
            intensity.centered_value_sum_squares
            / intensity.measurement_periods_scored
        ) ** 0.5
        self.assertFalse(
            isclose(
                receipt.discrete_value_std,
                realized_sample_std,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        self.assertFalse(
            isclose(
                receipt.discrete_value_std,
                receipt.continuous_value_std_parameter,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        )
        self.assertEqual(receipt.noise_std, self.parameters.noise_std)
        self.assertAlmostEqual(receipt.price_informativeness, expected)
        self.assertAlmostEqual(
            receipt.price_informativeness,
            receipt.informed_flow_variance / receipt.noise_order_variance,
        )
        self.assertEqual(
            receipt.measurement_periods_scored,
            intensity.measurement_periods_scored,
        )
        self.assertEqual(receipt.session_seed_manifest, session.streams.manifest)
        self.assertEqual(receipt.slope_by_agent, intensity.slope_by_agent)
        self.assertEqual(receipt.value_grid, tuple(float(v) for v in self.value_grid))
        self.assertEqual(receipt.formula, PRICE_INFORMATIVENESS_FORMULA)
        self.assertTrue(receipt.uses_discrete_value_grid_std)
        self.assertTrue(receipt.uses_configured_noise_std)
        self.assertEqual(repeated_receipt, receipt)
        self.assertEqual(session.period_number, period_before)
        self.assertEqual(session.all_random_states(), random_states_before)
        with self.assertRaises(FrozenInstanceError):
            receipt.price_informativeness = 0.0

    def test_incompatible_step30_provenance_is_rejected(self) -> None:
        """A Step-30 scorer cannot use another session's controller. / Step-30 scorer 不能使用另一 session 的 controller。"""

        _, intensity_scorer, source_controller = self.run_completed_session(
            "source"
        )
        _, _, other_controller = self.run_completed_session("other")
        with self.assertRaises(RuntimeError):
            build_price_informativeness_receipt(
                intensity_scorer,
                other_controller,
            )
        # The correct pair remains usable after the rejected crossed pair.
        # / 拒绝接错线后，正确配对仍然可用。
        receipt = build_price_informativeness_receipt(
            intensity_scorer,
            source_controller,
        )
        self.assertGreaterEqual(receipt.price_informativeness, 0.0)

    def test_incomplete_session_and_wrong_types_are_rejected(self) -> None:
        """Step 31 cannot run before its source session completes. / 来源 session 完成前不能运行 Step 31。"""

        _, completed_scorer, completed_controller = self.run_completed_session(
            "complete_source"
        )
        fresh_session = build_randomized_paper_session(
            parameters=self.parameters,
            value_grid=self.value_grid,
            price_grid=self.price_grid,
            action_multipliers=self.action_multipliers,
            initial_q_table=self.stable_q_table,
            prehistory=self.prehistory,
            experiment_seed=20260828,
            experiment_cell_key="step31_low_noise|fresh",
            session_index=0,
        )
        fresh_scorer = OnlineTradingIntensityScorer(fresh_session)
        fresh_controller = SessionPhaseController.create_for_fresh_session(
            fresh_session,
            convergence_periods_required=2,
            measurement_periods_required=100,
            measurement_sink=fresh_scorer.observe,
        )
        random_states_before = fresh_session.all_random_states()
        with self.assertRaises(RuntimeError):
            build_price_informativeness_receipt(
                fresh_scorer,
                fresh_controller,
            )
        self.assertEqual(fresh_session.period_number, 0)
        self.assertEqual(fresh_session.all_random_states(), random_states_before)
        with self.assertRaises(TypeError):
            build_price_informativeness_receipt(
                "not a scorer",  # type: ignore[arg-type]
                completed_controller,
            )
        with self.assertRaises(TypeError):
            build_price_informativeness_receipt(
                completed_scorer,
                object(),  # type: ignore[arg-type]
            )

    def test_shifted_grid_and_obsolete_step30_version_are_rejected(self) -> None:
        """Grid meaning and estimator version remain explicit. / 明确保护网格含义与估计器版本。"""

        shifted_session, shifted_scorer, shifted_controller = (
            self.run_completed_session("shifted_grid")
        )
        shifted_session.value_grid = tuple(
            float(value) + 0.25
            for value in shifted_session.value_grid
        )
        with self.assertRaises(RuntimeError):
            build_price_informativeness_receipt(
                shifted_scorer,
                shifted_controller,
            )

        _, old_scorer, old_controller = self.run_completed_session(
            "old_estimator"
        )
        current_receipt = old_scorer.finalize(old_controller)
        old_scorer._final_receipt = replace(  # deliberate corruption test / 故意损坏测试
            current_receipt,
            estimator_version="obsolete-step30-version",
        )
        with self.assertRaises(RuntimeError):
            build_price_informativeness_receipt(
                old_scorer,
                old_controller,
            )

    def test_post_measurement_parameter_and_symmetric_grid_rebinding_fail(self) -> None:
        """Step 31 uses the pre-training context, not rebound live fields. / Step 31 使用训练前环境，不使用事后重绑字段。"""

        noise_session, noise_scorer, noise_controller = (
            self.run_completed_session("rebound_noise")
        )
        noise_scorer.finalize(noise_controller)
        noise_session.parameters = replace(
            noise_session.parameters,
            noise_std=100.0,
        )
        with self.assertRaises(RuntimeError):
            build_price_informativeness_receipt(
                noise_scorer,
                noise_controller,
            )

        grid_session, grid_scorer, grid_controller = (
            self.run_completed_session("rescaled_grid")
        )
        grid_scorer.finalize(grid_controller)
        value_mean = grid_session.parameters.value_mean
        grid_session.value_grid = tuple(
            value_mean + 2.0 * (float(value) - value_mean)
            for value in grid_session.value_grid
        )
        # This corrupted grid keeps the same mean, so a mean-only check would
        # miss it. / 损坏后的网格均值不变，因此只检查均值无法发现。
        self.assertAlmostEqual(
            sum(grid_session.value_grid) / len(grid_session.value_grid),
            value_mean,
        )
        with self.assertRaises(RuntimeError):
            build_price_informativeness_receipt(
                grid_scorer,
                grid_controller,
            )


if __name__ == "__main__":
    unittest.main()
