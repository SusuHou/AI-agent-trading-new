"""Independent tests for Step 34. / Step 34 的独立自动测试。"""

from dataclasses import FrozenInstanceError
from math import inf, isclose, nextafter
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = PROJECT_ROOT / "steps"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from step_34_mechanism_classifier import (
    CLASSIFIER_VERSION,
    PAPER_CLASSIFIER_AGENTS,
    PAPER_HIGH_RESPONSE_THRESHOLD,
    PAPER_LOW_RESPONSE_THRESHOLD,
    PAPER_PATHS_PER_SESSION,
    PAPER_RESPONSE_PERIOD,
    PAPER_SHOCK_PERIOD,
    PAPER_TARGET_PRICE_DEVIATION,
    REPLICATION_LOW_RESPONSE_RULE,
    AppliedNoiseShock,
    CollusionMechanism,
    MechanismClassification,
    NormalizedOrderResponse,
    UndefinedOrderResponseError,
    UndefinedShockCalibrationError,
    UndefinedShockDirectionError,
    UniformShockCalibration,
    add_adverse_shock_to_noise,
    calculate_normalized_order_response,
    calibrate_uniform_noise_shock,
    classify_normalized_order_responses,
    orient_order,
    orient_price,
    orientation_sign,
)


class TestPaperOrientation(unittest.TestCase):
    """Check the Figure-3 sign transformation. / 检查图 3 的符号转换。"""

    def test_orientation_sign_above_below_and_equal(self) -> None:
        """The sign is +1, -1, or 0. / 符号分别是 +1、-1 或 0。"""

        self.assertEqual(orientation_sign(2.0, 1.0), 1)
        self.assertEqual(orientation_sign(0.0, 1.0), -1)
        self.assertEqual(orientation_sign(1.0, 1.0), 0)

    def test_oriented_prices_and_orders_do_not_use_absolute_values(self) -> None:
        """Orientation changes direction; it is not a blanket abs(). / 方向转换不是一律取绝对值。"""

        self.assertTrue(isclose(orient_price(1.2, 2.0, 1.0), 0.2))
        self.assertTrue(isclose(orient_price(0.8, 0.0, 1.0), 0.2))
        self.assertTrue(isclose(orient_price(1.2, 0.0, 1.0), -0.2))
        self.assertEqual(orient_order(3.0, 2.0, 1.0), 3.0)
        self.assertEqual(orient_order(-3.0, 0.0, 1.0), 3.0)
        self.assertEqual(orient_order(-3.0, 2.0, 1.0), -3.0)

    def test_orient_before_averaging_prevents_cancellation(self) -> None:
        """Buy-high and sell-low raw orders cancel, oriented orders do not. / 高值买入与低值卖出的原订单相消，方向调整后不会。"""

        raw_orders = (4.0, -4.0)
        values = (2.0, 0.0)
        self.assertEqual(sum(raw_orders) / 2, 0.0)
        oriented = tuple(
            orient_order(order, value, 1.0)
            for order, value in zip(raw_orders, values, strict=True)
        )
        self.assertEqual(oriented, (4.0, 4.0))
        self.assertEqual(sum(oriented) / 2, 4.0)

    def test_equal_value_orients_to_zero_but_cannot_direct_a_shock(self) -> None:
        """sign(0)=0 for a transform, while an adverse shock is undefined. / 转换时 sign(0)=0，但逆向冲击没有方向。"""

        self.assertEqual(orient_price(9.0, 1.0, 1.0), 0.0)
        self.assertEqual(orient_order(9.0, 1.0, 1.0), 0.0)
        with self.assertRaises(UndefinedShockDirectionError):
            add_adverse_shock_to_noise(0.0, 1.0, 1.0, 0.1)

    def test_invalid_orientation_inputs_are_rejected(self) -> None:
        """Booleans and nonfinite numbers are not silently accepted. / 布尔值与非有限数不会被悄悄接受。"""

        with self.assertRaises(TypeError):
            orientation_sign(True, 1.0)
        with self.assertRaises(ValueError):
            orient_price(inf, 2.0, 1.0)
        with self.assertRaises(ValueError):
            orient_order(float("nan"), 2.0, 1.0)


class TestShockCalibration(unittest.TestCase):
    """Check the disclosed one-common-magnitude calibration. / 检查公开说明的统一幅度校准。"""

    def test_hand_calculation_hits_one_point_two_percent(self) -> None:
        """0.012*2/0.5 gives 0.048 and the target exactly up to rounding. / 手算得到 0.048 与 1.2%。"""

        result = calibrate_uniform_noise_shock(2.0, 0.5, 0.5)
        self.assertIsInstance(result, UniformShockCalibration)
        self.assertTrue(isclose(result.absolute_noise_shock, 0.048, rel_tol=0.0, abs_tol=1e-15))
        self.assertTrue(isclose(result.implied_oriented_price_increment, 0.024, rel_tol=0.0, abs_tol=1e-15))
        self.assertTrue(isclose(result.implied_shocked_mean_oriented_price, 2.024, rel_tol=0.0, abs_tol=1e-15))
        self.assertTrue(isclose(result.implied_normalized_price_deviation, 0.012, rel_tol=0.0, abs_tol=1e-15))
        self.assertEqual(result.target_normalized_price_deviation, PAPER_TARGET_PRICE_DEVIATION)
        self.assertTrue(result.protocol_uses_one_common_magnitude_across_sessions_and_paths)
        self.assertTrue(result.common_magnitude_is_replication_interpretation)
        self.assertFalse(result.protocol_requires_cell_aggregates_from_same_calibration_sample)
        self.assertTrue(result.protocol_requires_aggregates_from_same_experiment_cell)
        self.assertTrue(result.protocol_allows_distinct_long_run_and_t3_samples)
        self.assertTrue(result.aggregate_inputs_are_caller_supplied)
        self.assertFalse(result.aggregate_provenance_verified)
        self.assertFalse(result.underlying_price_impact_positivity_verified_from_raw_paths)
        self.assertTrue(result.protocol_adds_shock_to_ordinary_noise)
        self.assertTrue(result.shock_addition_is_replication_interpretation)
        self.assertTrue(result.numerical_calibration_rule_is_replication_choice)

    def test_one_common_shock_hits_target_with_heterogeneous_lambdas_on_average(self) -> None:
        """A common magnitude uses mean lambda, not path-specific shocks. / 统一幅度使用平均 lambda，不逐路径改变幅度。"""

        lambdas = (0.25, 0.50, 0.75)
        mean_lambda = sum(lambdas) / len(lambdas)
        result = calibrate_uniform_noise_shock(2.0, mean_lambda, min(lambdas))
        path_increments = tuple(
            price_impact * result.absolute_noise_shock
            for price_impact in lambdas
        )
        achieved_average = (sum(path_increments) / len(path_increments)) / 2.0
        self.assertTrue(isclose(achieved_average, 0.012, rel_tol=0.0, abs_tol=1e-15))
        self.assertEqual(len(set(result.absolute_noise_shock for _ in lambdas)), 1)

    def test_one_cell_magnitude_is_reused_across_sessions(self) -> None:
        """The replication protocol calibrates once, then reuses one magnitude. / 复现协议只校准一次，再跨 session 复用同一幅度。"""

        session_lambdas = ((0.25, 0.50), (0.75, 1.00))
        all_lambdas = tuple(
            value
            for session in session_lambdas
            for value in session
        )
        result = calibrate_uniform_noise_shock(
            2.0,
            sum(all_lambdas) / len(all_lambdas),
            min(all_lambdas),
        )
        used_by_session = tuple(
            tuple(result.absolute_noise_shock for _ in session)
            for session in session_lambdas
        )
        self.assertEqual(len({value for session in used_by_session for value in session}), 1)

    def test_shock_sign_follows_value_and_is_added_to_ordinary_noise(self) -> None:
        """The same magnitude adds above the mean and subtracts below it. / 相同幅度在均值上方相加、下方相减。"""

        above = add_adverse_shock_to_noise(0.10, 2.0, 1.0, 0.048)
        below = add_adverse_shock_to_noise(0.10, 0.0, 1.0, 0.048)
        self.assertIsInstance(above, AppliedNoiseShock)
        self.assertEqual(above.orientation, 1)
        self.assertEqual(below.orientation, -1)
        self.assertEqual(above.signed_adverse_shock, 0.048)
        self.assertEqual(below.signed_adverse_shock, -0.048)
        self.assertTrue(isclose(above.noise_order_used_for_pricing, 0.148))
        self.assertTrue(isclose(below.noise_order_used_for_pricing, 0.052))
        self.assertEqual(above.shock_period, 3)

    def test_invalid_calibration_domains_are_rejected(self) -> None:
        """No abs(), epsilon, or clipping repairs an invalid domain. / 不用 abs、epsilon 或截断修补无效定义域。"""

        for mean_price, mean_lambda, minimum_lambda, target in (
            (0.0, 0.5, 0.5, 0.012),
            (-1.0, 0.5, 0.5, 0.012),
            (2.0, 0.0, 0.5, 0.012),
            (2.0, -0.5, 0.5, 0.012),
            (2.0, 0.5, 0.0, 0.012),
            (2.0, 0.5, -0.1, 0.012),
            (2.0, 0.5, 0.5, 0.0),
            (2.0, 0.5, 0.5, -0.012),
        ):
            with self.subTest(
                mean_price=mean_price,
                mean_lambda=mean_lambda,
                minimum_lambda=minimum_lambda,
                target=target,
            ):
                with self.assertRaises(UndefinedShockCalibrationError):
                    calibrate_uniform_noise_shock(
                        mean_price,
                        mean_lambda,
                        minimum_lambda,
                        target,
                    )

        with self.assertRaises(ValueError):
            calibrate_uniform_noise_shock(float("nan"), 0.5, 0.5)
        with self.assertRaises(TypeError):
            calibrate_uniform_noise_shock(True, 0.5, 0.5)
        with self.assertRaises(ValueError):
            calibrate_uniform_noise_shock(2.0, 0.5, 0.6)
        with self.assertRaises(ValueError):
            add_adverse_shock_to_noise(0.0, 2.0, 1.0, 0.0)

    def test_calibration_overflow_is_rejected(self) -> None:
        """A finite input combination may still overflow its product. / 有限输入的乘积仍可能溢出。"""

        with self.assertRaises(OverflowError):
            calibrate_uniform_noise_shock(
                sys.float_info.max,
                1.0,
                1.0,
                target_normalized_price_deviation=2.0,
            )
        with self.assertRaises(OverflowError):
            add_adverse_shock_to_noise(
                sys.float_info.max,
                2.0,
                1.0,
                sys.float_info.max,
            )

    def test_calibration_receipts_are_frozen(self) -> None:
        """Audited inputs cannot be changed after calculation. / 计算后的审计输入不能被修改。"""

        result = calibrate_uniform_noise_shock(2.0, 0.5, 0.5)
        applied = add_adverse_shock_to_noise(0.1, 2.0, 1.0, 0.048)
        with self.assertRaises(FrozenInstanceError):
            result.absolute_noise_shock = 99.0
        with self.assertRaises(FrozenInstanceError):
            applied.signed_adverse_shock = -99.0


class TestNormalizedOrderResponse(unittest.TestCase):
    """Check the long-run-mean response formula. / 检查长期均值标准化公式。"""

    def test_positive_and_negative_hand_responses(self) -> None:
        """100 to 100.06 is 0.0006; 100 to 99.996 is -0.00004. / 两个可手算反应。"""

        positive = calculate_normalized_order_response(1, 100.0, 100.06)
        negative = calculate_normalized_order_response(2, 100.0, 99.996)
        self.assertIsInstance(positive, NormalizedOrderResponse)
        self.assertTrue(isclose(positive.normalized_response, 6e-4, rel_tol=0.0, abs_tol=1e-15))
        self.assertTrue(isclose(negative.normalized_response, -4e-5, rel_tol=0.0, abs_tol=1e-15))
        self.assertEqual(positive.response_period, PAPER_RESPONSE_PERIOD)

    def test_nonpositive_or_invalid_long_run_mean_is_rejected(self) -> None:
        """The denominator must have the paper's positive orientation. / 分母必须具有论文预期的正方向。"""

        with self.assertRaises(UndefinedOrderResponseError):
            calculate_normalized_order_response(1, 0.0, 1.0)
        with self.assertRaises(UndefinedOrderResponseError):
            calculate_normalized_order_response(1, -1.0, 1.0)
        with self.assertRaises(ValueError):
            calculate_normalized_order_response(1, 1.0, inf)
        with self.assertRaises(ValueError):
            calculate_normalized_order_response(0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            calculate_normalized_order_response(3, 1.0, 1.0)

    def test_response_receipt_is_frozen(self) -> None:
        """The calculation receipt is immutable. / 计算凭证不可修改。"""

        result = calculate_normalized_order_response(1, 1.0, 1.1)
        with self.assertRaises(FrozenInstanceError):
            result.normalized_response = 0.0


class TestMechanismClassifier(unittest.TestCase):
    """Attack the strict threshold rules at their boundaries. / 在边界处检验严格阈值规则。"""

    def test_paper_constants_are_exactly_related(self) -> None:
        """The audited high threshold is ten times the low one. / 核对后的高阈值是低阈值十倍。"""

        self.assertEqual(PAPER_LOW_RESPONSE_THRESHOLD, 5e-5)
        self.assertEqual(PAPER_HIGH_RESPONSE_THRESHOLD, 5e-4)
        self.assertEqual(
            PAPER_HIGH_RESPONSE_THRESHOLD,
            10 * PAPER_LOW_RESPONSE_THRESHOLD,
        )
        self.assertEqual(PAPER_CLASSIFIER_AGENTS, 2)
        self.assertEqual(PAPER_PATHS_PER_SESSION, 10_000)

    def test_both_agents_above_high_threshold_are_price_trigger(self) -> None:
        """One floating step above is enough because comparison is strict. / 高一个浮点步即可通过严格比较。"""

        just_above = nextafter(PAPER_HIGH_RESPONSE_THRESHOLD, inf)
        result = classify_normalized_order_responses((just_above, 0.001))
        self.assertEqual(result.mechanism, CollusionMechanism.PRICE_TRIGGER)
        self.assertEqual(result.price_trigger_pass_by_agent, (True, True))
        self.assertEqual(result.over_pruning_pass_by_agent, (False, False))

    def test_both_agents_inside_low_threshold_are_over_pruning(self) -> None:
        """Small positive and negative reactions both pass abs(response)<low. / 小正反应与小负反应都通过绝对值低阈值。"""

        just_inside = nextafter(PAPER_LOW_RESPONSE_THRESHOLD, 0.0)
        result = classify_normalized_order_responses((just_inside, -just_inside))
        self.assertEqual(result.mechanism, CollusionMechanism.OVER_PRUNING)
        self.assertEqual(result.price_trigger_pass_by_agent, (False, False))
        self.assertEqual(result.over_pruning_pass_by_agent, (True, True))

    def test_exact_high_threshold_is_unclassified(self) -> None:
        """Equality does not satisfy ``>``. / 相等不满足 ``>``。"""

        result = classify_normalized_order_responses(
            (PAPER_HIGH_RESPONSE_THRESHOLD, PAPER_HIGH_RESPONSE_THRESHOLD)
        )
        self.assertEqual(result.mechanism, CollusionMechanism.UNCLASSIFIED)
        self.assertEqual(result.price_trigger_pass_by_agent, (False, False))

    def test_exact_low_threshold_is_unclassified(self) -> None:
        """Equality does not satisfy ``<``. / 相等不满足 ``<``。"""

        result = classify_normalized_order_responses(
            (PAPER_LOW_RESPONSE_THRESHOLD, -PAPER_LOW_RESPONSE_THRESHOLD)
        )
        self.assertEqual(result.mechanism, CollusionMechanism.UNCLASSIFIED)
        self.assertEqual(result.over_pruning_pass_by_agent, (False, False))

    def test_mixed_agents_and_large_negative_responses_are_unclassified(self) -> None:
        """Never average agents or treat aggressive negative reaction as punishment. / 不平均 agent，也不把强烈负反应当成惩罚。"""

        cases = (
            (0.001, 0.0),
            (0.001, -0.001),
            (-0.001, -0.001),
            (0.0001, 0.0001),
        )
        for responses in cases:
            with self.subTest(responses=responses):
                result = classify_normalized_order_responses(responses)
                self.assertEqual(
                    result.mechanism,
                    CollusionMechanism.UNCLASSIFIED,
                )

    def test_no_tolerance_rounding_or_isclose_enters_classification(self) -> None:
        """Adjacent floats fall on their literal sides of the threshold. / 相邻浮点数严格落在各自阈值一侧。"""

        below_high = nextafter(PAPER_HIGH_RESPONSE_THRESHOLD, 0.0)
        above_high = nextafter(PAPER_HIGH_RESPONSE_THRESHOLD, inf)
        below_low = nextafter(PAPER_LOW_RESPONSE_THRESHOLD, 0.0)
        above_low = nextafter(PAPER_LOW_RESPONSE_THRESHOLD, inf)
        self.assertEqual(
            classify_normalized_order_responses((below_high, below_high)).mechanism,
            CollusionMechanism.UNCLASSIFIED,
        )
        self.assertEqual(
            classify_normalized_order_responses((above_high, above_high)).mechanism,
            CollusionMechanism.PRICE_TRIGGER,
        )
        self.assertEqual(
            classify_normalized_order_responses((below_low, -below_low)).mechanism,
            CollusionMechanism.OVER_PRUNING,
        )
        self.assertEqual(
            classify_normalized_order_responses((above_low, -above_low)).mechanism,
            CollusionMechanism.UNCLASSIFIED,
        )

    def test_wrong_agent_count_nonfinite_and_boolean_responses_are_rejected(self) -> None:
        """The paper classifier has exactly two finite numeric responses. / 论文分类器恰好需要两个有限数值反应。"""

        for responses in ((), (0.0,), (0.0, 0.0, 0.0)):
            with self.subTest(responses=responses):
                with self.assertRaises(ValueError):
                    classify_normalized_order_responses(responses)
        with self.assertRaises(TypeError):
            classify_normalized_order_responses("0,0")
        with self.assertRaises(TypeError):
            classify_normalized_order_responses((True, 0.0))
        with self.assertRaises(ValueError):
            classify_normalized_order_responses((float("nan"), 0.0))

    def test_invalid_threshold_contracts_are_rejected(self) -> None:
        """Low must be positive and strictly below high. / 低阈值必须为正且严格低于高阈值。"""

        invalid_calls = (
            {"low_response_threshold": 0.0},
            {"high_response_threshold": PAPER_LOW_RESPONSE_THRESHOLD},
        )
        for keywords in invalid_calls:
            with self.subTest(keywords=keywords):
                with self.assertRaises(ValueError):
                    classify_normalized_order_responses((0.0, 0.0), **keywords)

    def test_custom_sensitivity_thresholds_are_disclosed(self) -> None:
        """Alternative thresholds work only with an explicit receipt flag. / 替代阈值只有在凭证明确标记后才可使用。"""

        result = classify_normalized_order_responses(
            (0.03, 0.04),
            low_response_threshold=0.01,
            high_response_threshold=0.02,
        )
        self.assertEqual(result.mechanism, CollusionMechanism.PRICE_TRIGGER)
        self.assertFalse(result.paper_thresholds_used)
        self.assertIn("< 0.01", result.low_rule_used)

    def test_receipt_records_scope_ambiguity_and_is_frozen(self) -> None:
        """The result says what this function did and did not do. / 结果明确说明本函数做了什么、没做什么。"""

        result = classify_normalized_order_responses((0.0, 0.0))
        self.assertIsInstance(result, MechanismClassification)
        self.assertEqual(result.mechanism, CollusionMechanism.OVER_PRUNING)
        self.assertEqual(result.paper_required_shock_period, PAPER_SHOCK_PERIOD)
        self.assertEqual(result.paper_required_response_period, PAPER_RESPONSE_PERIOD)
        self.assertEqual(result.paper_required_paths_per_session, PAPER_PATHS_PER_SESSION)
        self.assertTrue(result.both_agents_required)
        self.assertTrue(result.strict_inequalities_used)
        self.assertTrue(result.unclassified_label_is_replication_completion_rule)
        self.assertTrue(result.exact_threshold_behavior_follows_strict_inequalities)
        self.assertTrue(result.input_statistics_are_caller_supplied)
        self.assertFalse(result.input_horizon_verified)
        self.assertFalse(result.input_path_count_verified)
        self.assertFalse(result.same_session_and_checkpoint_provenance_verified)
        self.assertFalse(result.irf_paths_generated_by_this_function)
        self.assertTrue(result.low_rule_parentheses_are_replication_interpretation)
        self.assertEqual(result.low_rule_used, REPLICATION_LOW_RESPONSE_RULE)
        self.assertEqual(result.classifier_version, CLASSIFIER_VERSION)
        with self.assertRaises(FrozenInstanceError):
            result.mechanism = CollusionMechanism.PRICE_TRIGGER

    def test_classification_is_deterministic_and_does_not_mutate_input(self) -> None:
        """A pure classifier returns the same receipt and leaves a list unchanged. / 纯分类器结果相同且不修改输入列表。"""

        responses = [0.001, 0.001]
        before = responses.copy()
        first = classify_normalized_order_responses(responses)
        second = classify_normalized_order_responses(responses)
        self.assertEqual(first, second)
        self.assertEqual(responses, before)


if __name__ == "__main__":
    unittest.main()
