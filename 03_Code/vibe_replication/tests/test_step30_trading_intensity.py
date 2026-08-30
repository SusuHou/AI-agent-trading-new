"""Independent tests for Step 30 trading intensity. / Step 30 交易强度独立测试。"""

from dataclasses import FrozenInstanceError
from math import fsum, isclose
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
from step_26_reproducible_random_streams import (
    FrozenPolicyPeriodObservation,
    build_randomized_paper_session,
)
from step_28_session_phases import (
    SessionPhase,
    SessionPhaseController,
)
from step_29_matched_path_collusion_profitability import (
    MatchedPathCollusionScorer,
    build_matched_path_benchmarks,
)
from step_30_trading_intensity import (
    OnlineTradingIntensityScorer,
    OnlineTradingPolicyMoments,
    UndefinedTradingIntensityError,
    build_measurement_sink_fanout,
    fit_trading_policy_batch_ols,
)


class TestOnlineTradingPolicyMoments(unittest.TestCase):
    """Check the economic estimator independently of session machinery.

    不依赖 session 管理器，独立检查经济估计器。
    """

    def test_unrestricted_intercept_hand_example(self) -> None:
        """Recover two hand-known intercepts and slopes. / 恢复两组手算截距与斜率。"""

        moments = OnlineTradingPolicyMoments(2)
        for value, orders in (
            (0.0, (3.0, -4.0)),
            (1.0, (5.0, -3.5)),
            (2.0, (7.0, -3.0)),
        ):
            moments.add(value, orders)
        fit = moments.fit(value_mean_parameter=1.0)
        np.testing.assert_allclose(fit.intercept_by_agent, (3.0, -4.0), atol=1e-12)
        np.testing.assert_allclose(fit.slope_by_agent, (2.0, 0.5), atol=1e-12)
        self.assertAlmostEqual(fit.average_trading_intensity, 1.25)

    def test_theoretical_shape_is_diagnostic_not_constraint(self) -> None:
        """A theoretical line gives zero restriction residual. / 理论直线给出零约束残差。"""

        value_mean = 1.0
        values = (0.5, 1.0, 1.5)
        slopes = (2.0, 0.5)
        moments = OnlineTradingPolicyMoments(2)
        for value in values:
            moments.add(
                value,
                tuple(slope * (value - value_mean) for slope in slopes),
            )
        fit = moments.fit(value_mean)
        np.testing.assert_allclose(fit.slope_by_agent, slopes, atol=1e-12)
        np.testing.assert_allclose(fit.intercept_by_agent, (-2.0, -0.5), atol=1e-12)
        np.testing.assert_allclose(
            fit.theory_restriction_residual_by_agent,
            (0.0, 0.0),
            atol=1e-12,
        )

        # The hand example above has nonzero residuals, proving that the
        # estimator did not impose the theory. / 上一手算例残差非零，证明未强制理论约束。
        unrestricted = fit_trading_policy_batch_ols(
            (0.0, 1.0, 2.0),
            ((3.0, -4.0), (5.0, -3.5), (7.0, -3.0)),
            value_mean_parameter=1.0,
        )
        np.testing.assert_allclose(
            unrestricted.theory_restriction_residual_by_agent,
            (5.0, -3.5),
            atol=1e-12,
        )

    def test_streaming_matches_independent_numpy_ols(self) -> None:
        """Nonlinear rows match an independent batch oracle. / 非线性记录与独立批量 oracle 一致。"""

        values = (-2.0, -0.3, 0.4, 1.1, 3.0, 4.2)
        rows = tuple(
            (
                1.2 + 0.7 * value + 0.08 * value**2,
                -0.4 + 1.8 * value - 0.03 * value**3,
                2.0 - 0.2 * value + (0.1 if value > 1 else -0.1),
            )
            for value in values
        )
        online = OnlineTradingPolicyMoments(3)
        for value, orders in zip(values, rows, strict=True):
            online.add(value, orders)
        online_fit = online.fit(value_mean_parameter=1.0)
        batch_fit = fit_trading_policy_batch_ols(values, rows, 1.0)
        np.testing.assert_allclose(
            online_fit.intercept_by_agent,
            batch_fit.intercept_by_agent,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            online_fit.slope_by_agent,
            batch_fit.slope_by_agent,
            rtol=1e-13,
            atol=1e-13,
        )

        # Reversing row order should not materially change the estimate.
        # / 反转记录顺序不应实质改变估计。
        reversed_online = OnlineTradingPolicyMoments(3)
        for value, orders in reversed(tuple(zip(values, rows, strict=True))):
            reversed_online.add(value, orders)
        reversed_fit = reversed_online.fit(1.0)
        np.testing.assert_allclose(
            reversed_fit.slope_by_agent,
            online_fit.slope_by_agent,
            rtol=1e-13,
            atol=1e-13,
        )

    def test_constant_orders_give_valid_zero_slopes(self) -> None:
        """Zero sensitivity is a valid estimate, not an error. / 零敏感度是有效估计。"""

        moments = OnlineTradingPolicyMoments(2)
        for value in (0.0, 1.0, 2.0):
            moments.add(value, (5.0, -2.0))
        fit = moments.fit(1.0)
        np.testing.assert_allclose(fit.slope_by_agent, (0.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(fit.intercept_by_agent, (5.0, -2.0), atol=1e-12)

    def test_full_paper_measurement_length_matches_batch_ols(self) -> None:
        """All 100,000 streamed rows match batch OLS. / 十万条流式记录匹配批量 OLS。"""

        value_grid = np.linspace(-0.6, 2.6, 10)
        values = np.tile(value_grid, 10_000)
        row_numbers = np.arange(values.size)
        orders = np.column_stack(
            (
                np.where(values >= 1.0, 2.3, 0.8) * (values - 1.0),
                -0.4 + 1.1 * values + 0.02 * ((row_numbers % 7) - 3),
            )
        )
        online = OnlineTradingPolicyMoments(2)
        for value, order_row in zip(values, orders, strict=True):
            online.add(float(value), order_row)
        online_fit = online.fit(value_mean_parameter=1.0)
        batch_fit = fit_trading_policy_batch_ols(values, orders, 1.0)

        self.assertEqual(online_fit.observations, 100_000)
        np.testing.assert_allclose(
            online_fit.intercept_by_agent,
            batch_fit.intercept_by_agent,
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            online_fit.slope_by_agent,
            batch_fit.slope_by_agent,
            rtol=1e-11,
            atol=1e-11,
        )

    def test_degenerate_values_and_bad_rows_fail_without_mutation(self) -> None:
        """Invalid data never fabricate a coefficient. / 无效数据不伪造系数。"""

        moments = OnlineTradingPolicyMoments(2)
        moments.add(1.0, (2.0, 3.0))
        before = (
            moments.count,
            moments.mean_value,
            tuple(moments.mean_order_by_agent),
            tuple(moments.centered_value_order_sum_by_agent),
        )
        with self.assertRaises(ValueError):
            moments.add(2.0, (1.0,))
        with self.assertRaises(ValueError):
            moments.add(float("nan"), (1.0, 2.0))
        with self.assertRaises(ValueError):
            moments.add(2.0, (1.0, float("inf")))
        after = (
            moments.count,
            moments.mean_value,
            tuple(moments.mean_order_by_agent),
            tuple(moments.centered_value_order_sum_by_agent),
        )
        self.assertEqual(after, before)
        moments.add(1.0, (4.0, 5.0))
        with self.assertRaises(UndefinedTradingIntensityError):
            moments.fit(1.0)


class TestStep30SessionIntegration(unittest.TestCase):
    """Check Step 30 on the exact Step-28 measurement path.

    在精确 Step-28 测量路径上检查 Step 30。
    """

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

    def build_session(self, label: str, session_index: int = 0):
        """Build one fresh deterministic-identity test session. / 建立一个新鲜测试 session。"""

        return build_randomized_paper_session(
            parameters=self.parameters,
            value_grid=self.value_grid,
            price_grid=self.price_grid,
            action_multipliers=self.action_multipliers,
            initial_q_table=self.stable_q_table,
            prehistory=self.prehistory,
            experiment_seed=20260828,
            experiment_cell_key=f"step30_low_noise|{label}",
            session_index=session_index,
        )

    def make_observation(
        self,
        period_number: int,
        value_index: int,
        raw_orders: tuple[float, ...] | None = None,
        action_indexes: tuple[int, ...] = (0, 0),
    ) -> FrozenPolicyPeriodObservation:
        """Build one complete row for rejection tests only. / 仅供拒绝测试建立完整记录。"""

        value = self.value_grid[value_index]
        if raw_orders is None:
            available_orders = tuple(
                multiplier * (value - self.parameters.value_mean)
                for multiplier in self.action_multipliers
            )
            raw_orders = tuple(
                available_orders[action_index]
                for action_index in action_indexes
            )
        return FrozenPolicyPeriodObservation(
            period_number=period_number,
            current_state_indexes=(0, 0, value_index),
            current_state_id=0,
            current_value_index=value_index,
            fundamental_value_v=value,
            action_indexes=action_indexes,
            raw_orders_x=raw_orders,
            noise_order_u=0.0,
            total_order_flow_y=fsum(raw_orders),
            xi_0_hat=500.0,
            xi_1_hat=500.0,
            gamma_0_hat=1.0,
            gamma_1_hat=0.0,
            price_impact_lambda_hat=0.0,
            continuous_price_p=1.0,
            insensitive_order_z=0.0,
            profits=(0.0, 0.0),
            next_value_index=0,
            next_state_indexes=(0, value_index, 0),
            next_price_was_clipped=False,
        )

    def test_bad_observations_are_rejected_atomically(self) -> None:
        """Indexes, periods, values, and raw orders are checked. / 检查编号、时期、价值与订单。"""

        session = self.build_session("bad_rows")
        scorer = OnlineTradingIntensityScorer(session)
        scorer.observe(0, self.make_observation(100, 0))
        with self.assertRaises(ValueError):
            scorer.observe(0, self.make_observation(101, 1))
        with self.assertRaises(ValueError):
            scorer.observe(1, self.make_observation(102, 1))
        with self.assertRaises(ValueError):
            scorer.observe(1, self.make_observation(101, 1, (1.0,)))

        # Correctly shaped but forged orders must also fail before moments move.
        # / 形状正确但伪造的订单也必须在统计量改变前失败。
        with self.assertRaises(ValueError):
            scorer.observe(1, self.make_observation(101, 1, (999.0, 999.0)))
        with self.assertRaises(ValueError):
            scorer.observe(
                1,
                self.make_observation(
                    101,
                    1,
                    (0.0, 0.0),
                    action_indexes=(0, 999),
                ),
            )

        wrong_value = self.make_observation(101, 1)
        wrong_value = FrozenPolicyPeriodObservation(
            **{
                **wrong_value.__dict__,
                "fundamental_value_v": wrong_value.fundamental_value_v + 0.25,
            }
        )
        with self.assertRaises(ValueError):
            scorer.observe(1, wrong_value)
        self.assertEqual(scorer.rows_scored, 1)
        self.assertEqual(scorer._moments.count, 1)
        self.assertEqual(scorer.last_global_period_index, 100)

    def test_step29_step30_fanout_matches_batch_ols(self) -> None:
        """Both metrics consume one identical completed path. / 两个指标消费同一完整路径。"""

        session = self.build_session("fanout")
        profitability = MatchedPathCollusionScorer(session, self.benchmarks)
        intensity = OnlineTradingIntensityScorer(session)
        audit_values: list[float] = []
        audit_orders: list[tuple[float, ...]] = []

        def audit_sink(index: int, observation: FrozenPolicyPeriodObservation) -> None:
            self.assertEqual(index, len(audit_values))
            audit_values.append(observation.fundamental_value_v)
            audit_orders.append(tuple(observation.raw_orders_x))

        combined_sink = build_measurement_sink_fanout(
            profitability.observe,
            intensity.observe,
            audit_sink,
        )
        controller = SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=2,
            measurement_periods_required=100,
            measurement_sink=combined_sink,
        )
        with self.assertRaises(RuntimeError):
            intensity.finalize(controller)
        controller.run_until_complete(maximum_training_periods=2)
        profit_receipt = profitability.finalize(controller)
        intensity_receipt = intensity.finalize(controller)
        self.assertIs(intensity.finalize(controller), intensity_receipt)

        batch = fit_trading_policy_batch_ols(
            audit_values,
            audit_orders,
            self.parameters.value_mean,
        )
        np.testing.assert_allclose(
            intensity_receipt.intercept_by_agent,
            batch.intercept_by_agent,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            intensity_receipt.slope_by_agent,
            batch.slope_by_agent,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(profit_receipt.measurement_periods_scored, 100)
        self.assertEqual(intensity_receipt.measurement_periods_scored, 100)
        self.assertEqual(
            profit_receipt.first_global_period_index,
            intensity_receipt.first_global_period_index,
        )
        self.assertEqual(
            profit_receipt.last_global_period_index,
            intensity_receipt.last_global_period_index,
        )
        self.assertEqual(
            profit_receipt.session_seed_manifest,
            intensity_receipt.session_seed_manifest,
        )
        self.assertEqual(intensity_receipt.parameter_snapshot, session.parameters)
        self.assertEqual(
            intensity_receipt.value_grid_snapshot,
            tuple(float(value) for value in session.value_grid),
        )
        self.assertAlmostEqual(
            intensity_receipt.discrete_value_std_snapshot,
            0.937970,
            places=5,
        )
        self.assertAlmostEqual(
            intensity_receipt.average_trading_intensity,
            self.action_multipliers[0],
            delta=1e-10,
        )
        self.assertFalse(
            any(
                name in intensity.__dict__
                for name in ("rows", "history", "observations")
            )
        )
        with self.assertRaises(FrozenInstanceError):
            intensity_receipt.average_trading_intensity = 0.0
        with self.assertRaises(RuntimeError):
            intensity.observe(
                100,
                self.make_observation(102, 0, (1.0, 2.0)),
            )

        other_session = self.build_session("wrong_controller")
        other_controller = SessionPhaseController.create_for_fresh_session(
            other_session,
            convergence_periods_required=1,
            measurement_periods_required=2,
            measurement_sink=lambda index, observation: None,
        )
        with self.assertRaises(RuntimeError):
            intensity.finalize(other_controller)

    def test_fanout_failure_invalidates_every_partial_metric(self) -> None:
        """A later sink failure makes the whole seeded run unusable. / 后续 sink 失败使整个种子运行无效。"""

        session = self.build_session("failing_sink")
        profitability = MatchedPathCollusionScorer(session, self.benchmarks)
        intensity = OnlineTradingIntensityScorer(session)

        def fail_deliberately(
            index: int,
            observation: FrozenPolicyPeriodObservation,
        ) -> None:
            raise RuntimeError("deliberate test failure")

        combined_sink = build_measurement_sink_fanout(
            profitability.observe,
            intensity.observe,
            fail_deliberately,
        )
        controller = SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=1,
            measurement_periods_required=2,
            measurement_sink=combined_sink,
        )
        with self.assertRaises(RuntimeError):
            controller.run_until_complete(maximum_training_periods=1)
        self.assertIs(controller.phase, SessionPhase.FAILED)
        self.assertEqual(profitability.rows_scored, 1)
        self.assertEqual(intensity.rows_scored, 1)
        with self.assertRaises(RuntimeError):
            profitability.finalize(controller)
        with self.assertRaises(RuntimeError):
            intensity.finalize(controller)

    def test_fanout_validates_registration_and_order(self) -> None:
        """Fan-out is small, deterministic, and rejects duplicate sinks. / fan-out 小型、确定且拒绝重复 sink。"""

        with self.assertRaises(ValueError):
            build_measurement_sink_fanout()
        with self.assertRaises(TypeError):
            build_measurement_sink_fanout(lambda i, o: None, 3)  # type: ignore[arg-type]
        calls: list[str] = []

        def first(index: int, observation: FrozenPolicyPeriodObservation) -> None:
            calls.append("first")

        def second(index: int, observation: FrozenPolicyPeriodObservation) -> None:
            calls.append("second")

        with self.assertRaises(ValueError):
            build_measurement_sink_fanout(first, first)

        # Python creates a new bound-method object on each ``.observe`` access.
        # The fan-out must still recognize one scorer registered twice.
        # / Python 每次读取 ``.observe`` 都会产生新绑定方法对象，但仍须识别重复 scorer。
        one_session = self.build_session("duplicate_bound_sink")
        one_scorer = OnlineTradingIntensityScorer(one_session)
        with self.assertRaises(ValueError):
            build_measurement_sink_fanout(
                one_scorer.observe,
                one_scorer.observe,
            )

        # Reject crossed wires before an expensive measurement run begins.
        # / 在昂贵的测量运行开始前拒绝跨 session 接错线。
        another_session = self.build_session("crossed_session_sink")
        another_scorer = OnlineTradingIntensityScorer(another_session)
        with self.assertRaises(ValueError):
            build_measurement_sink_fanout(
                one_scorer.observe,
                another_scorer.observe,
            )

        # Even a correctly built one-session fan-out cannot be attached to a
        # different controller session. / 正确建立的单 session fan-out 也不能
        # 连接到另一 session 的 controller。
        one_session_fanout = build_measurement_sink_fanout(one_scorer.observe)
        with self.assertRaises(ValueError):
            SessionPhaseController.create_for_fresh_session(
                another_session,
                convergence_periods_required=1,
                measurement_periods_required=2,
                measurement_sink=one_session_fanout,
            )
        self.assertEqual(another_session.period_number, 0)
        self.assertIsNone(another_session.after_q_update_observer)

        fanout = build_measurement_sink_fanout(first, second)
        fanout(0, self.make_observation(0, 0, (1.0, 2.0)))
        self.assertEqual(calls, ["first", "second"])


if __name__ == "__main__":
    unittest.main()
