"""Independent tests for Step 33 mispricing. / Step 33 错误定价独立测试。"""

from array import array
from dataclasses import FrozenInstanceError, replace
from math import fma, fsum, isclose, isfinite, nextafter
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
from step_23_market_maker_ols import MarketMakerOLSEstimates
from step_24_adaptive_market_maker_price import calculate_adaptive_price_impact
from step_25_one_market_period import build_paper_inputs
from step_26_reproducible_random_streams import (
    FrozenPolicyPeriodObservation,
    build_randomized_paper_session,
)
from step_28_session_phases import SessionPhaseController
from step_29_matched_path_collusion_profitability import (
    MatchedPathCollusionScorer,
    build_matched_path_benchmarks,
)
from step_30_trading_intensity import (
    OnlineTradingIntensityScorer,
    build_measurement_sink_fanout,
)
from step_32_market_liquidity import OnlineMarketLiquidityScorer
from step_33_mispricing import (
    DEFINITION_3_4_ABSOLUTE_FORMULA,
    PAPER_MISPRICING_FORMULA,
    PAPER_PRINTED_AGGREGATION,
    REPLICATION_AGGREGATION,
    DeferredOnlineMispricingScorer,
    MispricingPairSummary,
    PeriodMispricingCalculation,
    UndefinedMispricingError,
    calculate_period_mispricing,
    summarize_mispricing_pairs,
)


class TestPeriodMispricingFormula(unittest.TestCase):
    """Check IA.4.7 independently of a market session. / 独立检查 IA.4.7。"""

    def test_hand_example(self) -> None:
        """A calculator gives loading .75 and mispricing 1.5. / 计算器得到 .75 与 1.5。"""

        result = calculate_period_mispricing(
            fundamental_value_v=3.0,
            value_mean=1.0,
            price_impact_lambda_hat=0.5,
            number_of_agents=2,
            average_trading_intensity=0.25,
        )
        self.assertIsInstance(result, PeriodMispricingCalculation)
        self.assertEqual(result.aggregate_informed_slope, 0.5)
        self.assertEqual(result.loading_factor, 0.75)
        self.assertEqual(result.absolute_loading_factor, 0.75)
        self.assertEqual(result.absolute_value_deviation, 2.0)
        self.assertEqual(result.paper_signed_expression, 1.5)
        self.assertEqual(result.definition_3_4_absolute_error, 1.5)
        self.assertTrue(result.paper_nonnegative_domain_satisfied)
        self.assertTrue(result.paper_formula_matches_definition)

    def test_zero_negative_lambda_and_zero_aggregate_slope(self) -> None:
        """Check easy boundaries without relying on the session. / 不依赖 session 检查简单边界。"""

        exact_zero = calculate_period_mispricing(2.0, 1.0, 0.5, 2, 1.0)
        self.assertEqual(exact_zero.loading_factor, 0.0)
        self.assertEqual(exact_zero.paper_signed_expression, 0.0)
        self.assertTrue(exact_zero.paper_nonnegative_domain_satisfied)

        negative_lambda = calculate_period_mispricing(
            2.0,
            1.0,
            -0.5,
            2,
            0.5,
        )
        self.assertEqual(negative_lambda.loading_factor, 1.5)
        self.assertEqual(negative_lambda.paper_signed_expression, 1.5)

        zero_slope = calculate_period_mispricing(4.0, 1.0, 999.0, 2, 0.0)
        self.assertEqual(zero_slope.loading_factor, 1.0)
        self.assertEqual(zero_slope.paper_signed_expression, 3.0)

    def test_negative_loading_exposes_the_paper_ambiguity(self) -> None:
        """Never silently add an absolute value to the paper. / 绝不偷偷替原文补绝对值。"""

        result = calculate_period_mispricing(2.0, 1.0, 0.75, 2, 1.0)
        self.assertEqual(result.loading_factor, -0.5)
        self.assertEqual(result.paper_signed_expression, -0.5)
        self.assertEqual(result.definition_3_4_absolute_error, 0.5)
        self.assertFalse(result.paper_nonnegative_domain_satisfied)
        self.assertFalse(result.paper_formula_matches_definition)

    def test_fma_preserves_a_near_zero_loading(self) -> None:
        """One rounding avoids a false exact zero. / 只舍入一次，避免虚假的精确零。"""

        self.assertEqual(1.0 - 0.002 * 500.0, 0.0)
        result = calculate_period_mispricing(2.0, 1.0, 0.002, 2, 250.0)
        self.assertEqual(result.loading_factor, fma(-0.002, 500.0, 1.0))
        self.assertNotEqual(result.loading_factor, 0.0)
        self.assertGreater(result.absolute_loading_factor, 0.0)
        self.assertGreater(result.definition_3_4_absolute_error, 0.0)

    def test_negative_loading_at_mean_value_still_gives_zero(self) -> None:
        """At v=v_bar both formulas are zero despite a negative loading. / v=v_bar 时即使系数为负，两式仍为零。"""

        result = calculate_period_mispricing(1.0, 1.0, 2.0, 2, 0.5)
        self.assertEqual(result.loading_factor, -1.0)
        self.assertFalse(result.paper_nonnegative_domain_satisfied)
        self.assertTrue(result.paper_formula_matches_definition)
        self.assertEqual(result.paper_signed_expression, 0.0)
        self.assertEqual(result.definition_3_4_absolute_error, 0.0)

    def test_invalid_inputs_overflow_and_frozen_result(self) -> None:
        """Bad inputs fail explicitly and the result is immutable. / 错误输入明确失败，结果不可修改。"""

        for bad_i in (0, -1, True, 2.5, "2"):
            with self.subTest(number_of_agents=bad_i):
                with self.assertRaises(ValueError):
                    calculate_period_mispricing(2.0, 1.0, 0.5, bad_i, 0.25)  # type: ignore[arg-type]
        for bad_number in (True, "1", None):
            with self.subTest(number=bad_number):
                with self.assertRaises(TypeError):
                    calculate_period_mispricing(bad_number, 1.0, 0.5, 2, 0.25)  # type: ignore[arg-type]
        for nonfinite in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                calculate_period_mispricing(2.0, 1.0, nonfinite, 2, 0.25)
        with self.assertRaises(OverflowError):
            calculate_period_mispricing(1e308, -1e308, 0.5, 2, 0.25)
        with self.assertRaises(OverflowError):
            calculate_period_mispricing(2.0, 1.0, 1e308, 2, 1e308)

        result = calculate_period_mispricing(3.0, 1.0, 0.5, 2, 0.25)
        with self.assertRaises(FrozenInstanceError):
            result.paper_signed_expression = 999.0


class TestDeferredPairReplay(unittest.TestCase):
    """Check why Step 33 retains two float64 values per period. / 检查为何每期保留两个 float64。"""

    def test_three_row_hand_aggregation(self) -> None:
        """Terms 0, 1.2, 1.2 give sum 2.4 and mean .8. / 三项求和 2.4、均值 .8。"""

        result = summarize_mispricing_pairs(
            (0.2, 0.4, -0.2),
            (0.0, 2.0, 1.0),
            2,
            0.5,
        )
        self.assertIsInstance(result, MispricingPairSummary)
        self.assertEqual(result.observations, 3)
        self.assertAlmostEqual(result.paper_signed_expression_sum, 2.4)
        self.assertAlmostEqual(result.definition_3_4_absolute_sum, 2.4)
        self.assertAlmostEqual(result.reported_mispricing_sum, 2.4)
        self.assertAlmostEqual(result.reported_average_mispricing, 0.8)
        self.assertTrue(result.paper_nonnegative_domain_satisfied)
        self.assertTrue(result.paper_formula_matches_definition_on_observed_path)
        self.assertEqual(result.negative_loading_period_count, 0)
        self.assertEqual(result.formula_disagreement_period_count, 0)

    def test_same_signed_sum_can_hide_different_absolute_errors(self) -> None:
        """A signed running sum cannot reconstruct the definition later. / 带符号累计和不能事后恢复绝对误差。"""

        # I*chi=1. Path A loadings are (+1,-1); path B loadings are (0,0).
        # Both printed sums equal zero, but absolute sums are 2 and 0. / 两条
        # 路径的原文字面和都是 0，但绝对误差和分别为 2 与 0。
        path_a = summarize_mispricing_pairs((0.0, 2.0), (1.0, 1.0), 2, 0.5)
        path_b = summarize_mispricing_pairs((1.0, 1.0), (1.0, 1.0), 2, 0.5)
        self.assertEqual(path_a.paper_signed_expression_sum, 0.0)
        self.assertEqual(path_b.paper_signed_expression_sum, 0.0)
        self.assertEqual(path_a.definition_3_4_absolute_sum, 2.0)
        self.assertEqual(path_b.definition_3_4_absolute_sum, 0.0)
        self.assertIsNone(path_a.reported_average_mispricing)
        self.assertFalse(path_a.paper_nonnegative_domain_satisfied)
        self.assertFalse(path_a.paper_formula_matches_definition_on_observed_path)
        self.assertEqual(path_a.formula_disagreement_period_count, 1)
        self.assertTrue(path_b.paper_nonnegative_domain_satisfied)

    def test_negative_loading_with_zero_deviation_remains_reportable(self) -> None:
        """The coefficient diagnostic and observed formula equality are distinct. / 系数诊断与本路径公式一致性不同。"""

        result = summarize_mispricing_pairs((2.0,), (0.0,), 2, 0.5)
        self.assertFalse(result.paper_nonnegative_domain_satisfied)
        self.assertTrue(result.paper_formula_matches_definition_on_observed_path)
        self.assertEqual(result.negative_loading_period_count, 1)
        self.assertEqual(result.formula_disagreement_period_count, 0)
        self.assertEqual(result.reported_mispricing_sum, 0.0)
        self.assertEqual(result.reported_average_mispricing, 0.0)
        self.assertEqual(result.first_negative_pair_index, 0)
        self.assertIsNone(result.first_formula_disagreement_pair_index)

    def test_full_paper_length_matches_independent_fsum(self) -> None:
        """100,000 compact pairs match an independent batch formula. / 十万紧凑数对匹配独立批量公式。"""

        lambda_pattern = (0.1, 0.2, -0.1, 0.4)
        deviation_pattern = (0.5, 1.5, 2.0, 0.25)
        lambdas = array("d", lambda_pattern * 25_000)
        deviations = array("d", deviation_pattern * 25_000)
        result = summarize_mispricing_pairs(lambdas, deviations, 2, 0.5)
        expected_terms = tuple(
            (1.0 - lambda_hat) * deviation
            for lambda_hat, deviation in zip(
                lambda_pattern,
                deviation_pattern,
                strict=True,
            )
        )
        expected_sum = fsum(expected_terms) * 25_000
        self.assertEqual(result.observations, 100_000)
        self.assertAlmostEqual(
            result.paper_signed_expression_sum,
            expected_sum,
            delta=1e-9,
        )
        self.assertAlmostEqual(
            result.reported_average_mispricing,
            expected_sum / 100_000,
            delta=1e-14,
        )
        self.assertEqual(lambdas.itemsize, 8)
        self.assertEqual(deviations.itemsize, 8)
        self.assertEqual(
            (len(lambdas) + len(deviations)) * 8,
            1_600_000,
        )

    def test_invalid_pair_inputs_fail(self) -> None:
        """Empty, unequal, negative, and nonfinite pairs are rejected. / 拒绝空、错长、负数与非有限输入。"""

        with self.assertRaises(UndefinedMispricingError):
            summarize_mispricing_pairs((), (), 2, 0.5)
        with self.assertRaises(ValueError):
            summarize_mispricing_pairs((0.1,), (1.0, 2.0), 2, 0.5)
        with self.assertRaises(ValueError):
            summarize_mispricing_pairs((0.1,), (-1.0,), 2, 0.5)
        with self.assertRaises(ValueError):
            summarize_mispricing_pairs((float("nan"),), (1.0,), 2, 0.5)


class TestStep33SessionIntegration(unittest.TestCase):
    """Connect Step 33 to the exact Step-28/30 path. / 把 Step 33 接入 Step-28/30 路径。"""

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
        cls.benchmarks = build_matched_path_benchmarks(
            cls.parameters,
            cls.value_grid,
        )

    def build_session(self, label: str, *, session_index: int = 0):
        """Build one fresh session with a stable greedy policy. / 建立一个贪心策略稳定的新 session。"""

        return build_randomized_paper_session(
            parameters=self.parameters,
            value_grid=self.value_grid,
            price_grid=self.price_grid,
            action_multipliers=self.action_multipliers,
            initial_q_table=self.stable_q_table,
            prehistory=self.prehistory,
            experiment_seed=20260828,
            experiment_cell_key=f"step33_low_noise|{label}",
            session_index=session_index,
        )

    def manual_observation(
        self,
        session,
        period_number: int,
        *,
        xi_1_hat: float,
        gamma_1_hat: float,
        lambda_hat: float,
    ) -> FrozenPolicyPeriodObservation:
        """Build a complete row for atomic-rejection tests. / 为原子拒绝测试建立完整记录。"""

        value = float(session.value_grid[0])
        return FrozenPolicyPeriodObservation(
            period_number=period_number,
            current_state_indexes=(0, 0, 0),
            current_state_id=0,
            current_value_index=0,
            fundamental_value_v=value,
            action_indexes=(0, 0),
            raw_orders_x=(0.0, 0.0),
            noise_order_u=0.0,
            total_order_flow_y=0.0,
            xi_0_hat=0.0,
            xi_1_hat=xi_1_hat,
            gamma_0_hat=1.0,
            gamma_1_hat=gamma_1_hat,
            price_impact_lambda_hat=lambda_hat,
            continuous_price_p=1.0,
            insensitive_order_z=0.0,
            profits=(0.0, 0.0),
            next_value_index=0,
            next_state_indexes=(0, 0, 0),
            next_price_was_clipped=False,
        )

    def test_steps_29_30_32_33_share_one_completed_path(self) -> None:
        """Four sibling metrics consume identical rows and provenance. / 四个平级指标消费同一路径与来源。"""

        session = self.build_session("fanout")
        profitability = MatchedPathCollusionScorer(session, self.benchmarks)
        intensity = OnlineTradingIntensityScorer(session)
        unused_intensity = OnlineTradingIntensityScorer(session)
        liquidity = OnlineMarketLiquidityScorer(session)
        mispricing = DeferredOnlineMispricingScorer(session)
        audit_lambdas: list[float] = []
        audit_deviations: list[float] = []

        def audit_sink(index: int, observation: FrozenPolicyPeriodObservation) -> None:
            self.assertEqual(index, len(audit_lambdas))
            audit_lambdas.append(observation.price_impact_lambda_hat)
            audit_deviations.append(
                abs(observation.fundamental_value_v - self.parameters.value_mean)
            )

        combined_sink = build_measurement_sink_fanout(
            profitability.observe,
            intensity.observe,
            liquidity.observe,
            mispricing.observe,
            audit_sink,
        )
        controller = SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=2,
            measurement_periods_required=100,
            measurement_sink=combined_sink,
        )
        with self.assertRaises(RuntimeError):
            mispricing.finalize(intensity, controller)
        controller.run_until_complete(maximum_training_periods=2)
        profit_receipt = profitability.finalize(controller)
        intensity_receipt = intensity.finalize(controller)
        liquidity_receipt = liquidity.finalize(controller)
        receipt = mispricing.finalize(intensity, controller)
        self.assertIs(mispricing.finalize(intensity, controller), receipt)

        independent = summarize_mispricing_pairs(
            audit_lambdas,
            audit_deviations,
            self.parameters.num_speculators,
            intensity_receipt.average_trading_intensity,
        )
        self.assertAlmostEqual(
            receipt.paper_signed_expression_sum,
            independent.paper_signed_expression_sum,
        )
        self.assertAlmostEqual(
            receipt.definition_3_4_absolute_sum,
            independent.definition_3_4_absolute_sum,
        )
        self.assertEqual(
            receipt.reported_average_mispricing,
            independent.reported_average_mispricing,
        )
        self.assertEqual(receipt.measurement_periods_scored, 100)
        self.assertEqual(receipt.paper_formula, PAPER_MISPRICING_FORMULA)
        self.assertEqual(
            receipt.definition_3_4_absolute_formula,
            DEFINITION_3_4_ABSOLUTE_FORMULA,
        )
        self.assertEqual(receipt.paper_printed_aggregation, PAPER_PRINTED_AGGREGATION)
        self.assertEqual(receipt.replication_aggregation, REPLICATION_AGGREGATION)
        self.assertFalse(receipt.paper_printed_first_factor_has_absolute_value)
        self.assertTrue(receipt.paper_prose_calls_aggregation_average)
        self.assertFalse(receipt.paper_printed_one_over_t)
        self.assertTrue(receipt.uses_full_measurement_window_chi)
        self.assertTrue(receipt.uses_period_specific_prior_history_lambda)
        self.assertTrue(
            receipt.uses_conditional_expected_mispricing_not_realized_price_error
        )
        self.assertTrue(receipt.uses_fused_multiply_add)
        self.assertEqual(receipt.compact_float64_values_buffered_before_finalize, 200)
        self.assertEqual(receipt.compact_buffer_bytes_before_finalize, 1_600)
        self.assertTrue(receipt.compact_storage_is_linear_in_measurement_periods)
        self.assertFalse(receipt.full_observation_rows_stored)
        self.assertTrue(receipt.compact_buffers_cleared_after_finalize)
        self.assertEqual(mispricing.buffered_rows, 0)
        self.assertEqual(mispricing.buffered_bytes, 0)
        self.assertEqual(
            receipt.first_global_period_index,
            profit_receipt.first_global_period_index,
        )
        self.assertEqual(
            receipt.last_global_period_index,
            liquidity_receipt.last_global_period_index,
        )
        self.assertEqual(
            receipt.session_seed_manifest,
            intensity_receipt.session_seed_manifest,
        )
        self.assertFalse(
            any(
                name in mispricing.__dict__
                for name in ("rows", "history", "observations", "prices")
            )
        )
        with self.assertRaises(FrozenInstanceError):
            receipt.reported_average_mispricing = 0.0
        with self.assertRaises(RuntimeError):
            mispricing.finalize(unused_intensity, controller)

    def test_receipt_flags_negative_loading_with_positive_deviation(self) -> None:
        """The ambiguity decision fields survive final receipt construction. / 歧义决策字段会进入最终 receipt。"""

        session = self.build_session("negative_receipt")
        intensity = OnlineTradingIntensityScorer(session)
        mispricing = DeferredOnlineMispricingScorer(session)
        controller = SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=1,
            measurement_periods_required=4,
            measurement_sink=build_measurement_sink_fanout(
                intensity.observe,
                mispricing.observe,
            ),
        )
        controller.run_until_complete(maximum_training_periods=1)
        intensity_receipt = intensity.finalize(controller)
        aggregate_slope = (
            self.parameters.num_speculators
            * intensity_receipt.average_trading_intensity
        )
        self.assertNotEqual(aggregate_slope, 0.0)
        self.assertTrue(all(value > 0.0 for value in mispricing._absolute_value_deviations))

        # Test-only branch injection: pure tests already validate the formula;
        # here we isolate receipt mapping by making the first loading -1 and
        # every later loading +1. / 仅用于测试的分支注入：纯函数测试已验证公式；
        # 这里把首个系数设为 -1、其余设为 +1，只检查 receipt 字段映射。
        negative_lambda = 2.0 / aggregate_slope
        mispricing._price_impact_lambdas = array(
            "d",
            (negative_lambda, 0.0, 0.0, 0.0),
        )
        receipt = mispricing.finalize(intensity, controller)
        self.assertFalse(receipt.paper_nonnegative_domain_satisfied)
        self.assertFalse(
            receipt.paper_formula_matches_definition_on_observed_path
        )
        self.assertTrue(receipt.requires_explicit_research_decision)
        self.assertEqual(receipt.negative_loading_period_count, 1)
        self.assertEqual(receipt.formula_disagreement_period_count, 1)
        self.assertEqual(
            receipt.first_negative_global_period_index,
            receipt.first_global_period_index,
        )
        self.assertEqual(
            receipt.first_formula_disagreement_global_period_index,
            receipt.first_global_period_index,
        )
        self.assertIsNone(receipt.reported_mispricing_sum)
        self.assertIsNone(receipt.reported_average_mispricing)
        self.assertNotEqual(
            receipt.paper_signed_expression_sum,
            receipt.definition_3_4_absolute_sum,
        )

    def test_recorded_lambda_value_and_sequence_are_checked_atomically(self) -> None:
        """A forged row leaves both compact buffers unchanged. / 伪造记录不改变两个紧凑缓冲区。"""

        session = self.build_session("bad_rows")
        scorer = DeferredOnlineMispricingScorer(session)
        valid_lambda = calculate_adaptive_price_impact(
            MarketMakerOLSEstimates(
                xi_0_hat=0.0,
                xi_1_hat=0.0,
                gamma_0_hat=1.0,
                gamma_1_hat=0.2,
                sample_size=2,
            ),
            self.parameters.pricing_error_weight,
        )
        valid = self.manual_observation(
            session,
            100,
            xi_1_hat=0.0,
            gamma_1_hat=0.2,
            lambda_hat=valid_lambda,
        )
        scorer.observe(0, valid)
        before = (
            scorer.rows_scored,
            scorer.last_global_period_index,
            scorer.buffered_rows,
            scorer.buffered_bytes,
            tuple(scorer._price_impact_lambdas),
            tuple(scorer._absolute_value_deviations),
        )
        with self.assertRaises(ValueError):
            scorer.observe(0, replace(valid, period_number=101))
        with self.assertRaises(ValueError):
            scorer.observe(1, replace(valid, period_number=102))
        with self.assertRaises(ValueError):
            scorer.observe(
                1,
                replace(
                    valid,
                    period_number=101,
                    fundamental_value_v=999.0,
                ),
            )
        with self.assertRaises(ValueError):
            scorer.observe(
                1,
                replace(
                    valid,
                    period_number=101,
                    price_impact_lambda_hat=nextafter(valid_lambda, 1.0),
                ),
            )
        after = (
            scorer.rows_scored,
            scorer.last_global_period_index,
            scorer.buffered_rows,
            scorer.buffered_bytes,
            tuple(scorer._price_impact_lambdas),
            tuple(scorer._absolute_value_deviations),
        )
        self.assertEqual(after, before)

    def test_incomplete_cross_session_and_rebound_context_are_rejected(self) -> None:
        """A receipt cannot mix sessions or changed economics. / receipt 不能混用 session 或变更后的经济环境。"""

        first_session = self.build_session("first")
        first_intensity = OnlineTradingIntensityScorer(first_session)
        first_mispricing = DeferredOnlineMispricingScorer(first_session)
        first_sink = build_measurement_sink_fanout(
            first_intensity.observe,
            first_mispricing.observe,
        )
        first_controller = SessionPhaseController.create_for_fresh_session(
            first_session,
            convergence_periods_required=1,
            measurement_periods_required=3,
            measurement_sink=first_sink,
        )
        with self.assertRaises(RuntimeError):
            first_mispricing.finalize(first_intensity, first_controller)

        second_session = self.build_session("second")
        second_intensity = OnlineTradingIntensityScorer(second_session)
        second_mispricing = DeferredOnlineMispricingScorer(second_session)
        with self.assertRaises(ValueError):
            build_measurement_sink_fanout(
                first_mispricing.observe,
                second_mispricing.observe,
            )
        second_controller = SessionPhaseController.create_for_fresh_session(
            second_session,
            convergence_periods_required=1,
            measurement_periods_required=3,
            measurement_sink=build_measurement_sink_fanout(
                second_intensity.observe,
                second_mispricing.observe,
            ),
        )
        with self.assertRaises(RuntimeError):
            first_mispricing.finalize(second_intensity, second_controller)

        first_controller.run_until_complete(maximum_training_periods=1)
        first_session.parameters = replace(self.parameters, investor_slope=499.0)
        with self.assertRaises(RuntimeError):
            first_mispricing.finalize(first_intensity, first_controller)


if __name__ == "__main__":
    unittest.main()
