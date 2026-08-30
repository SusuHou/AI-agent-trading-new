"""Independent tests for Step 35C. / 第 35C 步的独立自动测试。"""

from dataclasses import FrozenInstanceError, replace
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
from step_05_speculator_profit import calculate_profit
from step_25_one_market_period import build_paper_inputs
from step_26_reproducible_random_streams import (
    FrozenPolicyPeriodObservation,
    build_randomized_paper_session,
)
from step_28_session_phases import SessionPhase, SessionPhaseController
from step_30_trading_intensity import build_measurement_sink_fanout
from steps.step_34_mechanism_classifier import orient_order, orient_price
from steps.step_35c_irf_long_run_baseline import (
    OnlineIRFLongRunBaselineScorer,
    OnlineIRFLongRunMoments,
    validate_irf_long_run_baseline_receipt,
)


class TestOnlineIRFLongRunMoments(unittest.TestCase):
    """Check the pure constant-memory arithmetic. / 检查纯固定内存计算。"""

    def test_each_row_is_oriented_before_averaging(self) -> None:
        """Opposite raw signs cancel, while economically oriented orders do not.

        相反的原始符号会抵消，但经济方向调整后的订单不会抵消。
        """

        moments = OnlineIRFLongRunMoments(2)
        moments.add(
            fundamental_value_v=2.0,
            continuous_price_p=1.5,
            raw_orders_x=(4.0, 6.0),
            profits=(2.0, 3.0),
            price_impact_lambda_hat=0.2,
            value_mean=1.0,
        )
        moments.add(
            fundamental_value_v=0.0,
            continuous_price_p=0.5,
            raw_orders_x=(-4.0, -6.0),
            profits=(2.0, 3.0),
            price_impact_lambda_hat=0.4,
            value_mean=1.0,
        )
        result = moments.summarize()
        self.assertEqual(result.mean_centered_raw_price, 0.0)
        self.assertEqual(result.mean_raw_order_by_agent, (0.0, 0.0))
        self.assertEqual(result.mean_oriented_price, 0.5)
        self.assertEqual(result.mean_oriented_order_by_agent, (4.0, 6.0))
        self.assertEqual(result.mean_profit_by_agent, (2.0, 3.0))
        self.assertTrue(result.oriented_before_averaging)
        self.assertEqual(result.value_above_mean_count, 1)
        self.assertEqual(result.value_below_mean_count, 1)

    def test_compensated_sum_keeps_a_small_order_between_large_orders(self) -> None:
        """The sequence 1e16, 1, -1e16 retains the exact small residual.

        序列 1e16、1、-1e16 能保留中间的小数值 1。
        """

        moments = OnlineIRFLongRunMoments(1)
        for order in (1e16, 1.0, -1e16):
            moments.add(
                fundamental_value_v=2.0,
                continuous_price_p=1.5,
                raw_orders_x=(order,),
                profits=(0.0,),
                price_impact_lambda_hat=0.2,
                value_mean=1.0,
            )
        result = moments.summarize()
        self.assertEqual(result.mean_raw_order_by_agent, (1.0 / 3.0,))
        self.assertEqual(result.mean_oriented_order_by_agent, (1.0 / 3.0,))

    def test_invalid_rows_are_rejected_atomically(self) -> None:
        """Bad values, shapes, and overflow never partly alter moments.

        错误数值、形状与溢出绝不会只修改一部分统计量。
        """

        moments = OnlineIRFLongRunMoments(2)
        before = moments.audit_state()
        invalid_rows = (
            {"fundamental_value_v": float("nan"), "continuous_price_p": 1.0, "raw_orders_x": (1.0, 1.0), "profits": (1.0, 1.0), "price_impact_lambda_hat": 0.2, "value_mean": 1.0},
            {"fundamental_value_v": 2.0, "continuous_price_p": 1.0, "raw_orders_x": (1.0,), "profits": (1.0, 1.0), "price_impact_lambda_hat": 0.2, "value_mean": 1.0},
            {"fundamental_value_v": 2.0, "continuous_price_p": 1.0, "raw_orders_x": (1.0, 1.0), "profits": (1.0, float("inf")), "price_impact_lambda_hat": 0.2, "value_mean": 1.0},
            {"fundamental_value_v": 2.0, "continuous_price_p": 1e308, "raw_orders_x": (1.0, 1.0), "profits": (1.0, 1.0), "price_impact_lambda_hat": 1e308, "value_mean": -1e308},
        )
        for row in invalid_rows:
            with self.subTest(row=row):
                with self.assertRaises((TypeError, ValueError, OverflowError)):
                    moments.add(**row)
                self.assertEqual(moments.audit_state(), before)

    def test_nonpositive_lambda_is_recorded_not_repaired(self) -> None:
        """No abs() or epsilon silently turns a bad lambda positive.

        不使用 abs 或 epsilon 偷偷把错误 lambda 变成正数。
        """

        moments = OnlineIRFLongRunMoments(1)
        for price_impact in (0.0, -0.25):
            moments.add(
                fundamental_value_v=2.0,
                continuous_price_p=1.5,
                raw_orders_x=(1.0,),
                profits=(1.0,),
                price_impact_lambda_hat=price_impact,
                value_mean=1.0,
            )
        result = moments.summarize()
        self.assertEqual(result.nonpositive_price_impact_count, 2)
        self.assertEqual(result.minimum_price_impact_lambda, -0.25)
        self.assertEqual(result.mean_price_impact_lambda, -0.125)


class TestIRFLongRunBaselineIntegration(unittest.TestCase):
    """Connect Step 35C to the real Step-28 measurement lifecycle.

    把第 35C 步连接到真实的第 28 步测量生命周期。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.parameters = PaperParameters()
        (
            cls.value_grid,
            cls.price_grid,
            cls.action_multipliers,
            initial_q,
            cls.prehistory,
        ) = build_paper_inputs(cls.parameters)
        cls.stable_q = np.zeros_like(initial_q)
        cls.stable_q[:, 0] = 1_000_000_000.0

    def build_session(self, label: str, session_index: int = 0):
        """Build one fresh deterministic-identity session. / 建立一个新 session。"""

        return build_randomized_paper_session(
            parameters=self.parameters,
            value_grid=self.value_grid,
            price_grid=self.price_grid,
            action_multipliers=self.action_multipliers,
            initial_q_table=self.stable_q,
            prehistory=self.prehistory,
            experiment_seed=20_260_829,
            experiment_cell_key=f"step35c|{label}",
            session_index=session_index,
        )

    def make_observation(
        self,
        *,
        period_number: int,
        value_index: int,
        action_indexes: tuple[int, int] = (0, 0),
    ) -> FrozenPolicyPeriodObservation:
        """Create one internally coherent row for rejection tests.

        为拒绝测试建立一条内部一致的记录。
        """

        value = self.value_grid[value_index]
        available_orders = tuple(
            multiplier * (value - self.parameters.value_mean)
            for multiplier in self.action_multipliers
        )
        orders = tuple(available_orders[index] for index in action_indexes)
        total_flow = fsum(orders)
        gamma_0 = 1.0
        price_impact = 0.002
        price = gamma_0 + price_impact * total_flow
        profits = tuple(
            calculate_profit(value, price, order) for order in orders
        )
        return FrozenPolicyPeriodObservation(
            period_number=period_number,
            current_state_indexes=(0, 0, value_index),
            current_state_id=0,
            current_value_index=value_index,
            fundamental_value_v=value,
            action_indexes=action_indexes,
            raw_orders_x=orders,
            noise_order_u=0.0,
            total_order_flow_y=total_flow,
            xi_0_hat=500.0,
            xi_1_hat=500.0,
            gamma_0_hat=gamma_0,
            gamma_1_hat=0.0,
            price_impact_lambda_hat=price_impact,
            continuous_price_p=price,
            insensitive_order_z=-500.0 * (price - 1.0),
            profits=profits,
            next_value_index=0,
            next_state_indexes=(0, value_index, 0),
            next_price_was_clipped=False,
        )

    def run_baseline(self, label: str, measurement_periods: int = 100):
        """Run one short, real session and return receipt plus raw audit rows.

        运行一个短小真实 session，并返回凭证与原始核对行。
        """

        session = self.build_session(label)
        scorer = OnlineIRFLongRunBaselineScorer(session)
        raw_rows: list[FrozenPolicyPeriodObservation] = []

        def collect_raw_row(
            measurement_index: int,
            observation: FrozenPolicyPeriodObservation,
        ) -> None:
            self.assertEqual(measurement_index, len(raw_rows))
            raw_rows.append(observation)

        combined_sink = build_measurement_sink_fanout(
            scorer.observe,
            collect_raw_row,
        )
        controller = SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=2,
            measurement_periods_required=measurement_periods,
            measurement_sink=combined_sink,
        )
        while controller.phase is SessionPhase.TRAINING:
            if controller.training_periods_completed >= 2:
                raise AssertionError("The test fixture did not converge.")
            controller.run_next_period()
        scorer.capture_and_bind_convergence_checkpoint(controller)
        controller.run_until_complete()
        return session, controller, scorer, scorer.finalize(controller), raw_rows

    def test_real_measurement_matches_independent_batch_means(self) -> None:
        """Online means equal independent orient-then-batch calculations.

        在线均值等于独立的“先调方向、再批量平均”计算。
        """

        _, controller, scorer, receipt, rows = self.run_baseline("batch")
        count = len(rows)
        expected_prices = [
            orient_price(
                row.continuous_price_p,
                row.fundamental_value_v,
                self.parameters.value_mean,
            )
            for row in rows
        ]
        expected_orders = [
            [
                orient_order(
                    row.raw_orders_x[agent],
                    row.fundamental_value_v,
                    self.parameters.value_mean,
                )
                for row in rows
            ]
            for agent in range(2)
        ]
        expected_profits = [
            [row.profits[agent] for row in rows] for agent in range(2)
        ]
        expected_lambdas = [row.price_impact_lambda_hat for row in rows]
        self.assertEqual(receipt.measurement_periods_scored, count)
        self.assertTrue(
            isclose(
                receipt.mean_oriented_price,
                fsum(expected_prices) / count,
                rel_tol=1e-15,
                abs_tol=1e-15,
            )
        )
        np.testing.assert_allclose(
            receipt.mean_oriented_order_by_agent,
            tuple(fsum(values) / count for values in expected_orders),
            rtol=1e-15,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            receipt.mean_profit_by_agent,
            tuple(fsum(values) / count for values in expected_profits),
            rtol=1e-15,
            atol=1e-15,
        )
        self.assertTrue(
            isclose(
                receipt.mean_price_impact_lambda,
                fsum(expected_lambdas) / count,
                rel_tol=1e-15,
                abs_tol=1e-15,
            )
        )
        self.assertEqual(
            receipt.minimum_price_impact_lambda,
            min(expected_lambdas),
        )
        self.assertIsNotNone(scorer._source_checkpoint)
        self.assertEqual(
            receipt.source_checkpoint_sha256,
            scorer._source_checkpoint.checkpoint_sha256,  # type: ignore[union-attr]
        )
        self.assertIs(scorer.finalize(controller), receipt)

    def test_receipt_keeps_claim_boundaries_explicit(self) -> None:
        """A debug baseline is not an IRF, shock, or classification result.

        调试基准不是 IRF、冲击或分类结果。
        """

        _, _, _, receipt, _ = self.run_baseline("boundaries", 20)
        self.assertTrue(receipt.oriented_before_averaging)
        self.assertTrue(receipt.paper_defines_expectations_but_not_estimator)
        self.assertTrue(
            receipt.same_session_post_convergence_window_is_replication_interpretation
        )
        self.assertTrue(receipt.per_session_denominator_is_replication_interpretation)
        self.assertFalse(receipt.unshocked_control_used_as_denominator)
        self.assertTrue(receipt.per_session_denominator_not_pooled_across_sessions)
        self.assertTrue(receipt.measurement_sink_delivery_verified)
        self.assertTrue(receipt.exact_convergence_checkpoint_provenance_verified)
        self.assertTrue(receipt.same_session_scored_sample_provenance_verified)
        self.assertFalse(receipt.complete_raw_observation_digest_included)
        self.assertFalse(
            receipt.paper_post_convergence_measurement_length_100000_verified
        )
        self.assertFalse(receipt.paper_links_that_window_to_irf_denominator)
        self.assertFalse(receipt.paper_convergence_threshold_1000000_verified)
        self.assertFalse(receipt.paper_scale_thresholds_and_provenance_verified)
        self.assertTrue(receipt.ready_for_price_normalization)
        self.assertTrue(receipt.ready_for_order_normalization)
        self.assertTrue(receipt.ready_for_profit_normalization)
        self.assertTrue(receipt.ready_for_all_figure3_normalizations)
        self.assertTrue(receipt.long_run_lambda_is_diagnostic_not_shock_calibration_input)
        self.assertFalse(receipt.ready_for_shock_calibration)
        self.assertEqual(receipt.irf_paths_aggregated, 0)
        self.assertFalse(receipt.shock_applied)
        self.assertFalse(receipt.classification_ready)
        self.assertTrue(
            receipt.checksum_detects_stale_replacement_not_authentication
        )
        self.assertFalse(receipt.standalone_receipt_authenticates_streamed_means)
        self.assertTrue(receipt.step35d_must_use_live_scorer_and_matching_checkpoint)
        with self.assertRaises(FrozenInstanceError):
            receipt.classification_ready = True  # type: ignore[misc]

    def test_same_session_identity_reproduces_digest_and_means(self) -> None:
        """A replay gives identical provenance; a different cell changes it.

        重放给出相同来源；不同实验单元会改变它。
        """

        first = self.run_baseline("replay", 30)[3]
        replay = self.run_baseline("replay", 30)[3]
        different = self.run_baseline("different", 30)[3]
        self.assertEqual(first, replay)
        self.assertNotEqual(
            first.session_seed_manifest.session_seed,
            different.session_seed_manifest.session_seed,
        )
        self.assertNotEqual(
            first.scored_fields_sha256,
            different.scored_fields_sha256,
        )

    def test_detached_scorer_cannot_claim_noop_controller_rows(self) -> None:
        """The former coherent-foreign-row provenance attack is rejected.

        过去可用“内部一致的外来记录”伪造来源；现在该攻击会被拒绝。
        """

        session = self.build_session("detached")
        scorer = OnlineIRFLongRunBaselineScorer(session)

        def noop_sink(
            measurement_index: int,
            observation: FrozenPolicyPeriodObservation,
        ) -> None:
            del measurement_index, observation

        noop_sink._measurement_session = session  # type: ignore[attr-defined]
        controller = SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=2,
            measurement_periods_required=5,
            measurement_sink=noop_sink,
        )
        while controller.phase is SessionPhase.TRAINING:
            controller.run_next_period()
        with self.assertRaises(RuntimeError):
            scorer.capture_and_bind_convergence_checkpoint(controller)
        controller.run_until_complete()
        foreign_row = self.make_observation(
            period_number=controller.final_receipt.measurement_first_period_index,  # type: ignore[union-attr]
            value_index=2,
        )
        with self.assertRaises(RuntimeError):
            scorer.observe(0, foreign_row)
        with self.assertRaises(RuntimeError):
            scorer.finalize(controller)

    def test_plain_callable_cannot_forge_official_fanout_membership(self) -> None:
        """Look-alike public attributes do not make a no-op an official fan-out.

        即使添加相似公开属性，no-op 普通函数也不会变成正式 fan-out。
        """

        session = self.build_session("fake-fanout")
        scorer = OnlineIRFLongRunBaselineScorer(session)

        def fake_fanout(
            measurement_index: int,
            observation: FrozenPolicyPeriodObservation,
        ) -> None:
            del measurement_index, observation

        fake_fanout._measurement_session = session  # type: ignore[attr-defined]
        fake_fanout._measurement_sink_keys = (  # type: ignore[attr-defined]
            ("bound_method", id(scorer), id(type(scorer).observe)),
        )
        controller = SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=2,
            measurement_periods_required=5,
            measurement_sink=fake_fanout,
        )
        while controller.phase is SessionPhase.TRAINING:
            controller.run_next_period()
        with self.assertRaises(RuntimeError):
            scorer.capture_and_bind_convergence_checkpoint(controller)

    def test_plain_function_cannot_spoof_bound_method_identity(self) -> None:
        """Fake ``__self__``/``__func__`` fields do not make a MethodType.

        伪造 ``__self__``/``__func__`` 字段不会把普通函数变成 MethodType。
        """

        for wiring in ("direct", "official-fanout"):
            with self.subTest(wiring=wiring):
                session = self.build_session(f"method-spoof-{wiring}")
                scorer = OnlineIRFLongRunBaselineScorer(session)

                def malicious_proxy(
                    measurement_index: int,
                    observation: FrozenPolicyPeriodObservation,
                ) -> None:
                    del measurement_index, observation

                malicious_proxy.__self__ = scorer  # type: ignore[attr-defined]
                malicious_proxy.__func__ = type(scorer).observe  # type: ignore[attr-defined]
                registered_sink = (
                    malicious_proxy
                    if wiring == "direct"
                    else build_measurement_sink_fanout(malicious_proxy)
                )
                controller = SessionPhaseController.create_for_fresh_session(
                    session,
                    convergence_periods_required=2,
                    measurement_periods_required=5,
                    measurement_sink=registered_sink,
                )
                while controller.phase is SessionPhase.TRAINING:
                    controller.run_next_period()
                with self.assertRaises(RuntimeError):
                    scorer.capture_and_bind_convergence_checkpoint(controller)

    def test_receipt_validator_rejects_stale_dataclass_replacements(self) -> None:
        """A replacement retaining the old checksum is detected, not authenticated.

        保留旧校验码的 replacement 会被发现；这不等于密码学认证。
        """

        receipt = self.run_baseline("receipt-integrity", 20)[3]
        validate_irf_long_run_baseline_receipt(receipt)
        stale_replacements = (
            replace(receipt, mean_oriented_price=999.0),
            replace(receipt, classification_ready=True),
            replace(receipt, ready_for_shock_calibration=True),
            replace(receipt, source_checkpoint_sha256="0" * 64),
        )
        for stale in stale_replacements:
            with self.subTest(stale=stale):
                with self.assertRaises(ValueError):
                    validate_irf_long_run_baseline_receipt(stale)

    def test_registered_sink_is_immutable_for_controller_lifetime(self) -> None:
        """The sink cannot be swapped during measurement and restored later.

        测量途中不能替换 sink，并在结束前偷偷换回来。
        """

        _, controller, scorer, receipt, _ = self.run_baseline(
            "sink-rebound",
            20,
        )
        validate_irf_long_run_baseline_receipt(receipt)
        with self.assertRaises(AttributeError):
            controller.measurement_sink = lambda index, row: None
        self.assertIs(scorer.finalize(controller), receipt)

    def test_manual_rows_outside_live_controller_delivery_are_atomic(self) -> None:
        """Even coherent rows cannot be inserted outside the live sink call.

        即使记录内部一致，也不能在 controller 实时 sink 调用之外插入。
        """

        session = self.build_session("forged")
        scorer = OnlineIRFLongRunBaselineScorer(session)
        controller = SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=2,
            measurement_periods_required=5,
            measurement_sink=scorer.observe,
        )
        while controller.phase is SessionPhase.TRAINING:
            controller.run_next_period()
        scorer.capture_and_bind_convergence_checkpoint(controller)
        first_period = session.period_number
        valid = self.make_observation(period_number=first_period, value_index=2)
        before = (
            scorer.rows_scored,
            scorer.last_global_period_index,
            scorer._moments.audit_state(),
            scorer._digest.hexdigest(),
        )
        invalid_rows = (
            valid,
            replace(valid, continuous_price_p=999.0),
            replace(valid, profits=(999.0, 999.0)),
            replace(valid, raw_orders_x=(999.0, 999.0)),
        )
        for row in invalid_rows:
            with self.subTest(row=row):
                with self.assertRaises(RuntimeError):
                    scorer.observe(0, row)
                after = (
                    scorer.rows_scored,
                    scorer.last_global_period_index,
                    scorer._moments.audit_state(),
                    scorer._digest.hexdigest(),
                )
                self.assertEqual(after, before)

        # The same scorer accepts the controller's actual next delivery. /
        # 同一个 scorer 会接受 controller 真正发出的下一条记录。
        controller.run_next_period()
        self.assertEqual(scorer.rows_scored, 1)
        self.assertEqual(controller.measurement_periods_completed, 1)

    def test_finalize_rejects_incomplete_crossed_or_rebound_session(self) -> None:
        """Only the completed, unchanged, bound session can issue a receipt.

        只有已完成、未改变且正确绑定的 session 可以签发凭证。
        """

        session = self.build_session("incomplete")
        scorer = OnlineIRFLongRunBaselineScorer(session)
        controller = SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=2,
            measurement_periods_required=5,
            measurement_sink=scorer.observe,
        )
        with self.assertRaises(RuntimeError):
            scorer.finalize(controller)

        other_session = self.build_session("other")
        other_controller = SessionPhaseController.create_for_fresh_session(
            other_session,
            convergence_periods_required=2,
            measurement_periods_required=5,
        )
        with self.assertRaises(ValueError):
            scorer.finalize(other_controller)

        while controller.phase is SessionPhase.TRAINING:
            controller.run_next_period()
        scorer.capture_and_bind_convergence_checkpoint(controller)
        controller.run_until_complete()
        session.parameters = PaperParameters(noise_std=100.0)
        with self.assertRaises(RuntimeError):
            scorer.finalize(controller)


if __name__ == "__main__":
    unittest.main()
