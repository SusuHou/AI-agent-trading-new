"""Independent tests for Step 32 market liquidity. / Step 32 市场流动性独立测试。"""

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
from step_32_market_liquidity import (
    MARKET_LIQUIDITY_FORMULA,
    PAPER_PRINTED_AGGREGATION,
    REPLICATION_AGGREGATION,
    MarketLiquidityAggregate,
    OnlineMarketLiquidityAccumulator,
    OnlineMarketLiquidityScorer,
    PeriodMarketLiquidityCalculation,
    UndefinedMarketLiquidityError,
    calculate_period_market_liquidity,
)


class TestPeriodMarketLiquidityFormula(unittest.TestCase):
    """Check IA.4.6 independently of the market session. / 独立检查 IA.4.6。"""

    def test_hand_example_and_inventory_derivative(self) -> None:
        """xi=2 and lambda=.25 give sensitivity .5 and L=2. / 手算得到 .5 与 2。"""

        result = calculate_period_market_liquidity(2.0, 0.25)
        self.assertEqual(result.investor_slope_xi, 2.0)
        self.assertEqual(result.price_impact_lambda_hat, 0.25)
        self.assertEqual(result.rounded_xi_times_lambda_hat, 0.5)
        self.assertEqual(result.signed_inventory_sensitivity, 0.5)
        self.assertEqual(result.absolute_inventory_sensitivity, 0.5)
        self.assertEqual(result.market_liquidity, 2.0)
        self.assertFalse(result.is_exactly_singular)
        self.assertFalse(result.reciprocal_overflowed)

        # Independent finite-difference check of m=-(z+y). / 独立有限差分检查 m=-(z+y)。
        def inventory(noise_order: float) -> float:
            total_flow = 3.0 + noise_order
            price = 1.0 + 0.25 * total_flow
            insensitive_order = -2.0 * (price - 1.0)
            return -(insensitive_order + total_flow)

        step = 1e-6
        derivative_magnitude = abs(
            (inventory(step) - inventory(-step)) / (2.0 * step)
        )
        self.assertAlmostEqual(derivative_magnitude, 0.5, places=9)
        self.assertAlmostEqual(1.0 / derivative_magnitude, 2.0, places=8)

    def test_absolute_value_zero_xi_and_negative_lambda(self) -> None:
        """The sign can change; xi=0 always gives one. / 符号可变；xi=0 恒为 1。"""

        self.assertEqual(
            calculate_period_market_liquidity(2.0, 0.75).market_liquidity,
            2.0,
        )
        self.assertEqual(
            calculate_period_market_liquidity(2.0, -0.5).market_liquidity,
            0.5,
        )
        self.assertEqual(
            calculate_period_market_liquidity(0.0, -100.0).market_liquidity,
            1.0,
        )
        self.assertEqual(
            calculate_period_market_liquidity(500.0, 0.0).market_liquidity,
            1.0,
        )

    def test_fma_preserves_finite_near_singular_values(self) -> None:
        """Never turn a finite huge value into zero or clip it. / 不把有限大数误成零或截断。"""

        # Ordinary arithmetic rounds this to zero on binary64.
        # / 普通 binary64 运算会把它舍入为零。
        self.assertEqual(1.0 - 500.0 * 0.002, 0.0)
        result = calculate_period_market_liquidity(500.0, 0.002)
        self.assertEqual(
            result.signed_inventory_sensitivity,
            fma(-500.0, 0.002, 1.0),
        )
        self.assertNotEqual(result.signed_inventory_sensitivity, 0.0)
        self.assertTrue(isfinite(result.market_liquidity))
        self.assertGreater(result.market_liquidity, 1e16)

        near = calculate_period_market_liquidity(
            2.0,
            0.5000000000000001,
        )
        self.assertFalse(near.is_exactly_singular)
        self.assertAlmostEqual(
            near.market_liquidity,
            4.503599627370496e15,
            delta=1.0,
        )

    def test_exact_singularity_is_explicit_infinity(self) -> None:
        """A true zero denominator is tagged, not epsilon-adjusted. / 真零分母被标记，不加 epsilon。"""

        result = calculate_period_market_liquidity(2.0, 0.5)
        self.assertEqual(result.signed_inventory_sensitivity, 0.0)
        self.assertEqual(result.absolute_inventory_sensitivity, 0.0)
        self.assertEqual(result.market_liquidity, float("inf"))
        self.assertTrue(result.is_exactly_singular)
        self.assertFalse(result.reciprocal_overflowed)

    def test_invalid_inputs_and_overflow_are_rejected(self) -> None:
        """Bad structural inputs never fabricate a finite metric. / 错误输入不能伪造有限指标。"""

        for bad_xi in (True, "2", None):
            with self.subTest(xi=bad_xi):
                with self.assertRaises(TypeError):
                    calculate_period_market_liquidity(bad_xi, 0.2)  # type: ignore[arg-type]
        for bad_xi in (-1.0,):
            with self.assertRaises(ValueError):
                calculate_period_market_liquidity(bad_xi, 0.2)
        for nonfinite_xi in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                calculate_period_market_liquidity(nonfinite_xi, 0.2)
        for bad_lambda in (True, "0.2", None):
            with self.assertRaises(TypeError):
                calculate_period_market_liquidity(2.0, bad_lambda)  # type: ignore[arg-type]
        for nonfinite_lambda in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                calculate_period_market_liquidity(2.0, nonfinite_lambda)
        with self.assertRaises(OverflowError):
            calculate_period_market_liquidity(1e308, 1e308)

    def test_period_result_is_frozen(self) -> None:
        """A reported period cannot be silently edited. / 已报告时期不能被悄悄修改。"""

        result = calculate_period_market_liquidity(2.0, 0.25)
        self.assertIsInstance(result, PeriodMarketLiquidityCalculation)
        with self.assertRaises(FrozenInstanceError):
            result.market_liquidity = 999.0


class TestOnlineMarketLiquidityAccumulator(unittest.TestCase):
    """Check nonlinear aggregation and constant memory. / 检查非线性汇总与固定内存。"""

    def test_period_first_aggregation_hand_example(self) -> None:
        """Average L_t, never insert average lambda into the inverse. / 先算各期 L 再平均。"""

        lambdas = (0.0, 0.25, -0.25)
        accumulator = OnlineMarketLiquidityAccumulator()
        for lambda_hat in lambdas:
            accumulator.add(
                calculate_period_market_liquidity(2.0, lambda_hat)
            )
        summary = accumulator.summarize()
        self.assertIsInstance(summary, MarketLiquidityAggregate)
        self.assertEqual(summary.observations, 3)
        self.assertAlmostEqual(summary.literal_liquidity_sum, 11.0 / 3.0)
        self.assertAlmostEqual(summary.average_market_liquidity, 11.0 / 9.0)
        self.assertAlmostEqual(summary.minimum_period_liquidity, 2.0 / 3.0)
        self.assertEqual(summary.maximum_period_liquidity, 2.0)
        self.assertEqual(summary.minimum_absolute_inventory_sensitivity, 0.5)

        wrong_order_of_operations = calculate_period_market_liquidity(
            2.0,
            fsum(lambdas) / len(lambdas),
        ).market_liquidity
        self.assertNotAlmostEqual(
            summary.average_market_liquidity,
            wrong_order_of_operations,
        )

    def test_exact_infinity_is_counted_without_poisoning_lifecycle(self) -> None:
        """An infinite period gives an infinite extended-real mean. / 无穷时期给出无穷扩展实数平均。"""

        accumulator = OnlineMarketLiquidityAccumulator()
        accumulator.add(calculate_period_market_liquidity(2.0, 0.0))
        accumulator.add(calculate_period_market_liquidity(2.0, 0.5))
        accumulator.add(calculate_period_market_liquidity(2.0, -0.5))
        summary = accumulator.summarize()
        self.assertEqual(summary.observations, 3)
        self.assertEqual(summary.infinite_period_count, 1)
        self.assertEqual(summary.exact_singular_period_count, 1)
        self.assertEqual(summary.reciprocal_overflow_period_count, 0)
        self.assertEqual(summary.literal_liquidity_sum, float("inf"))
        self.assertEqual(summary.average_market_liquidity, float("inf"))
        self.assertEqual(summary.minimum_absolute_inventory_sensitivity, 0.0)
        self.assertEqual(summary.maximum_period_liquidity, float("inf"))

    def test_full_paper_length_matches_fsum_in_constant_memory(self) -> None:
        """100,000 online values equal an independent batch sum. / 十万条在线值匹配批量求和。"""

        pattern = tuple(
            calculate_period_market_liquidity(2.0, lambda_hat)
            for lambda_hat in (0.0, 0.25, -0.25, 0.75)
        )
        accumulator = OnlineMarketLiquidityAccumulator()
        for index in range(100_000):
            accumulator.add(pattern[index % len(pattern)])
        summary = accumulator.summarize()
        batch_values = tuple(item.market_liquidity for item in pattern) * 25_000
        expected_sum = fsum(batch_values)
        self.assertEqual(summary.observations, 100_000)
        self.assertAlmostEqual(
            summary.literal_liquidity_sum,
            expected_sum,
            delta=1e-9,
        )
        self.assertAlmostEqual(
            summary.average_market_liquidity,
            expected_sum / 100_000,
            delta=1e-14,
        )
        self.assertFalse(
            any(
                name in accumulator.__dict__
                for name in ("rows", "history", "observations", "values")
            )
        )

    def test_bad_add_is_atomic_and_empty_summary_fails(self) -> None:
        """A rejected calculation leaves every moment unchanged. / 被拒绝的结果不改变任何统计量。"""

        accumulator = OnlineMarketLiquidityAccumulator()
        with self.assertRaises(UndefinedMarketLiquidityError):
            accumulator.summarize()
        valid = calculate_period_market_liquidity(2.0, 0.25)
        accumulator.add(valid)
        before = dict(accumulator.__dict__)
        forged = replace(valid, market_liquidity=float("inf"))
        with self.assertRaises(ValueError):
            accumulator.add(forged)
        forged_finite = replace(valid, market_liquidity=999.0)
        with self.assertRaises(ValueError):
            accumulator.add(forged_finite)
        self.assertEqual(accumulator.__dict__, before)


class TestStep32SessionIntegration(unittest.TestCase):
    """Connect Step 32 to the exact Step-28 measurement path. / 把 Step 32 接入 Step-28 路径。"""

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

    def build_session(
        self,
        label: str,
        *,
        parameters: PaperParameters | None = None,
        session_index: int = 0,
    ):
        """Build one fresh deterministic-identity session. / 建立一个新鲜且身份确定的 session。"""

        if parameters is None:
            parameters = self.parameters
            value_grid = self.value_grid
            price_grid = self.price_grid
            action_multipliers = self.action_multipliers
            q_table = self.stable_q_table
            prehistory = self.prehistory
        else:
            (
                value_grid,
                price_grid,
                action_multipliers,
                initial_q_table,
                prehistory,
            ) = build_paper_inputs(parameters)
            q_table = np.zeros_like(initial_q_table, dtype=float)
            q_table[:, 0] = 1_000_000_000.0
        return build_randomized_paper_session(
            parameters=parameters,
            value_grid=value_grid,
            price_grid=price_grid,
            action_multipliers=action_multipliers,
            initial_q_table=q_table,
            prehistory=prehistory,
            experiment_seed=20260828,
            experiment_cell_key=f"step32_low_noise|{label}",
            session_index=session_index,
        )

    @staticmethod
    def manual_observation(
        period_number: int,
        *,
        xi_1_hat: float,
        gamma_1_hat: float,
        lambda_hat: float,
    ) -> FrozenPolicyPeriodObservation:
        """Build a complete row for scorer rejection tests. / 为 scorer 拒绝测试建立完整记录。"""

        return FrozenPolicyPeriodObservation(
            period_number=period_number,
            current_state_indexes=(0, 0, 0),
            current_state_id=0,
            current_value_index=0,
            fundamental_value_v=1.0,
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

    def test_step29_step30_step32_share_one_completed_path(self) -> None:
        """All sibling metrics consume identical rows and provenance. / 三个平级指标消费同一路径。"""

        session = self.build_session("fanout")
        profitability = MatchedPathCollusionScorer(session, self.benchmarks)
        intensity = OnlineTradingIntensityScorer(session)
        liquidity = OnlineMarketLiquidityScorer(session)
        audit_lambdas: list[float] = []

        def audit_sink(
            index: int,
            observation: FrozenPolicyPeriodObservation,
        ) -> None:
            self.assertEqual(index, len(audit_lambdas))
            estimates = MarketMakerOLSEstimates(
                xi_0_hat=observation.xi_0_hat,
                xi_1_hat=observation.xi_1_hat,
                gamma_0_hat=observation.gamma_0_hat,
                gamma_1_hat=observation.gamma_1_hat,
                sample_size=self.parameters.market_maker_window,
            )
            independently_recomputed_lambda = calculate_adaptive_price_impact(
                estimates,
                self.parameters.pricing_error_weight,
            )
            self.assertAlmostEqual(
                observation.price_impact_lambda_hat,
                independently_recomputed_lambda,
                places=15,
            )
            audit_lambdas.append(observation.price_impact_lambda_hat)

        combined_sink = build_measurement_sink_fanout(
            profitability.observe,
            intensity.observe,
            liquidity.observe,
            audit_sink,
        )
        controller = SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=2,
            measurement_periods_required=100,
            measurement_sink=combined_sink,
        )
        with self.assertRaises(RuntimeError):
            liquidity.finalize(controller)
        controller.run_until_complete(maximum_training_periods=2)
        profit_receipt = profitability.finalize(controller)
        intensity_receipt = intensity.finalize(controller)
        receipt = liquidity.finalize(controller)
        self.assertIs(liquidity.finalize(controller), receipt)

        batch_liquidity = tuple(
            calculate_period_market_liquidity(
                self.parameters.investor_slope,
                lambda_hat,
            ).market_liquidity
            for lambda_hat in audit_lambdas
        )
        self.assertTrue(all(isfinite(value) for value in batch_liquidity))
        expected_sum = fsum(batch_liquidity)
        self.assertAlmostEqual(receipt.literal_liquidity_sum, expected_sum)
        self.assertAlmostEqual(
            receipt.average_market_liquidity,
            expected_sum / 100,
        )
        self.assertEqual(receipt.measurement_periods_scored, 100)
        self.assertEqual(receipt.investor_slope_xi, 500.0)
        self.assertEqual(receipt.formula, MARKET_LIQUIDITY_FORMULA)
        self.assertEqual(
            receipt.paper_printed_aggregation,
            PAPER_PRINTED_AGGREGATION,
        )
        self.assertEqual(
            receipt.replication_aggregation,
            REPLICATION_AGGREGATION,
        )
        self.assertFalse(receipt.paper_printed_one_over_t)
        self.assertTrue(receipt.paper_prose_calls_aggregation_average)
        self.assertTrue(receipt.uses_configured_structural_xi)
        self.assertTrue(receipt.uses_period_specific_prior_history_lambda)
        self.assertTrue(receipt.uses_fused_multiply_add)
        self.assertEqual(receipt.infinite_period_count, 0)
        self.assertEqual(receipt.first_infinite_global_period_index, None)
        self.assertEqual(
            receipt.first_global_period_index,
            profit_receipt.first_global_period_index,
        )
        self.assertEqual(
            receipt.last_global_period_index,
            intensity_receipt.last_global_period_index,
        )
        self.assertEqual(
            receipt.session_seed_manifest,
            intensity_receipt.session_seed_manifest,
        )
        self.assertFalse(
            any(
                name in liquidity.__dict__
                for name in ("rows", "history", "observations", "lambdas")
            )
        )
        with self.assertRaises(FrozenInstanceError):
            receipt.average_market_liquidity = 0.0

    def test_recorded_lambda_and_period_sequence_are_checked_atomically(self) -> None:
        """Forged or skipped rows never change the scorer. / 伪造或跳号记录不改变 scorer。"""

        session = self.build_session("bad_rows")
        scorer = OnlineMarketLiquidityScorer(session)
        # theta=.1, xi1=0, gamma1=.2. Compute the binary-float result through
        # Step 24 itself rather than typing the rounded decimal 0.2. / 通过
        # Step 24 本身得到二进制浮点结果，不手写已舍入的十进制 0.2。
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
            100,
            xi_1_hat=0.0,
            gamma_1_hat=0.2,
            lambda_hat=valid_lambda,
        )
        scorer.observe(0, valid)
        before = (
            scorer.rows_scored,
            scorer.last_global_period_index,
            dict(scorer._accumulator.__dict__),
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
                    price_impact_lambda_hat=0.3,
                ),
            )
        # Even a one-ULP change is rejected because liquidity can be extremely
        # sensitive near xi*lambda=1. / 即使只改一个 ULP 也拒绝，因为奇点附近
        # 流动性对 lambda 极其敏感。
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
            dict(scorer._accumulator.__dict__),
        )
        self.assertEqual(after, before)

    def test_exact_singular_period_does_not_fail_the_sink(self) -> None:
        """A deterministic economic singularity is recorded, not retried forever. / 确定性奇点被记录而非无限重跑。"""

        parameters = replace(self.parameters, investor_slope=2.0)
        session = self.build_session("singular", parameters=parameters)
        scorer = OnlineMarketLiquidityScorer(session)
        # theta=.1, xi1=0, gamma1=.5 -> recorded lambda=.5.
        singular = self.manual_observation(
            10,
            xi_1_hat=0.0,
            gamma_1_hat=0.5,
            lambda_hat=0.5,
        )
        scorer.observe(0, singular)
        summary = scorer._accumulator.summarize()
        self.assertEqual(scorer.rows_scored, 1)
        self.assertEqual(summary.infinite_period_count, 1)
        self.assertEqual(summary.average_market_liquidity, float("inf"))
        self.assertEqual(scorer.first_infinite_global_period_index, 10)

    def test_cross_session_wiring_and_parameter_rebinding_are_rejected(self) -> None:
        """Receipts cannot mix sessions or changed economics. / 结果不能混用 session 或事后参数。"""

        first_session = self.build_session("first")
        first_scorer = OnlineMarketLiquidityScorer(first_session)
        first_controller = SessionPhaseController.create_for_fresh_session(
            first_session,
            convergence_periods_required=1,
            measurement_periods_required=2,
            measurement_sink=first_scorer.observe,
        )
        with self.assertRaises(RuntimeError):
            first_scorer.finalize(first_controller)

        second_session = self.build_session("second")
        second_scorer = OnlineMarketLiquidityScorer(second_session)
        with self.assertRaises(ValueError):
            build_measurement_sink_fanout(
                first_scorer.observe,
                second_scorer.observe,
            )
        second_controller = SessionPhaseController.create_for_fresh_session(
            second_session,
            convergence_periods_required=1,
            measurement_periods_required=2,
            measurement_sink=second_scorer.observe,
        )
        with self.assertRaises(RuntimeError):
            first_scorer.finalize(second_controller)

        first_controller.run_until_complete(maximum_training_periods=1)
        first_session.parameters = replace(
            self.parameters,
            investor_slope=499.0,
        )
        with self.assertRaises(RuntimeError):
            first_scorer.finalize(first_controller)


if __name__ == "__main__":
    unittest.main()
