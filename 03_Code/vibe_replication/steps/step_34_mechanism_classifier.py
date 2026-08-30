"""Step 34: calibrate the paper's shock and classify its collusion mechanism.

第 34 步：校准论文的噪声冲击，并判断合谋机制。

Run the hand-checkable demonstration / 运行可以手算的演示:
    py -3 -X utf8 steps/step_34_mechanism_classifier.py

Run the separate automated tests / 运行独立自动测试:
    py -3 -X utf8 -m unittest discover -s tests \\
        -p "test_step34_mechanism_classifier.py" -v

Paper definitions / 论文定义:

    p_tilde_t = (p_t - v_bar) * sign(v_t - v_bar)
    x_tilde_i,t = x_i,t * sign(v_t - v_bar)

    response_i = (
        mean(x_tilde_i,4 after the shock) - E[x_tilde_i]
    ) / E[x_tilde_i]

At t=3 the adverse noise shock has the same sign as ``v_3-v_bar``. As a
disclosed replication interpretation, we add one common-magnitude shock to
ordinary noise and target an average normalized oriented-price increase of
1.2%. At t=4 this replication completes the two strict paper conditions with
three labels / 在 t=3，逆向噪声冲击与 ``v_3-v_bar`` 同号。作为公开记录的
复现解释，我们把一个统一幅度冲击加到普通噪声上，并以平均标准化方向调整价格提高
1.2% 为目标。在 t=4，本复现把原文两个严格条件补全为三个标签：

    price trigger / 价格触发:
        BOTH responses > 5e-4

    over-pruning / 过度剪枝:
        BOTH abs(responses) < 5e-5

    otherwise, as our completion rule / 其他情况（本复现的补全规则）:
        unclassified / 未分类

Both inequalities are strict. Exact equality is therefore unclassified. / 两个
不等式都是严格不等式，所以恰好等于阈值时属于未分类。

Scope boundary / 本步骤的边界:
    This file implements pure arithmetic only. It does NOT pretend to generate
    the paper's 10,000 impulse-response paths. The current market session still
    needs a safe checkpoint-and-clone interface before a shocked branch and an
    unshocked branch can be run without changing one another. Step 35 will own
    that runner. / 本文件只实现纯计算，不假装已经生成论文的 10,000 条冲击反应
    路径。现有市场 session 还需要安全的快照与克隆接口，才能保证有冲击组和无冲击组
    不会互相改变；完整运行器由第 35 步负责。

Documented paper ambiguity / 已记录的原文歧义:
    Appendix Section 4.5 prints the low-response condition with missing
    parentheses. Read literally, it is dimensionally inconsistent. Following
    the correctly printed Figure-3 normalization, this replication uses
    ``abs((x_tilde-E[x_tilde])/E[x_tilde]) < 5e-5``. / 附录第 4.5 节的
    低反应条件漏印了括号，若逐字读取会量纲不一致。本复现依据图 3 正确打印的标准化
    公式，采用 ``abs((x_tilde-E[x_tilde])/E[x_tilde]) < 5e-5``。
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Integral, Real
from collections.abc import Sequence


PAPER_TARGET_PRICE_DEVIATION = 0.012
PAPER_LOW_RESPONSE_THRESHOLD = 5e-5
PAPER_HIGH_RESPONSE_THRESHOLD = 5e-4
PAPER_SHOCK_PERIOD = 3
PAPER_RESPONSE_PERIOD = 4
PAPER_PATHS_PER_SESSION = 10_000
PAPER_CLASSIFIER_AGENTS = 2

PAPER_ORIENTED_PRICE_FORMULA = (
    "p_tilde_t = (p_t - v_bar) * sign(v_t - v_bar)"
)
PAPER_ORIENTED_ORDER_FORMULA = (
    "x_tilde_i_t = x_i_t * sign(v_t - v_bar)"
)
PAPER_NORMALIZED_ORDER_RESPONSE_FORMULA = (
    "r_i = (mean_shocked_x_tilde_i_4 - long_run_mean_x_tilde_i) "
    "/ long_run_mean_x_tilde_i"
)
REPLICATION_LOW_RESPONSE_RULE = (
    "abs((x_tilde_i_4 - E[x_tilde_i]) / E[x_tilde_i]) < 5e-5"
)
REPLICATION_SHOCK_CALIBRATION_FORMULA = (
    "absolute_u_shock = target * cell_mean[p_tilde] "
    "/ cell_mean[lambda_hat_3]"
)
CLASSIFIER_VERSION = "appendix-4.5-v1-pure-contract"


class UndefinedShockDirectionError(ArithmeticError):
    """The paper gives no adverse direction when ``v_t=v_bar``. / 当 v_t=v_bar 时原文没有定义逆向冲击方向。"""


class UndefinedShockCalibrationError(ArithmeticError):
    """The requested normalized price shock cannot be calibrated. / 无法校准所要求的标准化价格冲击。"""


class UndefinedOrderResponseError(ArithmeticError):
    """The normalized oriented-order response is undefined. / 标准化方向调整订单反应未定义。"""


class CollusionMechanism(str, Enum):
    """The three mutually exclusive Step-34 labels. / 第 34 步三个互斥标签。"""

    PRICE_TRIGGER = "price_trigger"
    OVER_PRUNING = "over_pruning"
    UNCLASSIFIED = "unclassified"


def _finite_real(number: float, label: str) -> float:
    """Return one finite non-Boolean real. / 返回一个有限、非布尔实数。"""

    if isinstance(number, bool) or not isinstance(number, Real):
        raise TypeError(f"{label} must be a real number. / {label} 必须是实数。")
    converted = float(number)
    if not isfinite(converted):
        raise ValueError(f"{label} must be finite. / {label} 必须是有限数。")
    return converted


def _positive_integer(number: int, label: str) -> int:
    """Return one positive non-Boolean integer. / 返回一个正的、非布尔整数。"""

    if (
        isinstance(number, bool)
        or not isinstance(number, Integral)
        or int(number) < 1
    ):
        raise ValueError(f"{label} must be a positive integer. / {label} 必须是正整数。")
    return int(number)


def orientation_sign(fundamental_value: float, value_mean: float) -> int:
    """Return ``sign(v_t-v_bar)`` without subtracting huge floats.

    不先相减，直接比较并返回 ``sign(v_t-v_bar)``，可避免极大数相减溢出。
    """

    value = _finite_real(fundamental_value, "fundamental_value v_t")
    mean = _finite_real(value_mean, "value_mean v_bar")
    if value > mean:
        return 1
    if value < mean:
        return -1
    return 0


def orient_price(
    price: float,
    fundamental_value: float,
    value_mean: float,
) -> float:
    """Calculate ``(p_t-v_bar)*sign(v_t-v_bar)``. / 计算论文的方向调整价格。"""

    observed_price = _finite_real(price, "price p_t")
    mean = _finite_real(value_mean, "value_mean v_bar")
    direction = orientation_sign(fundamental_value, mean)
    price_gap = observed_price - mean
    if not isfinite(price_gap):
        raise OverflowError("p_t - v_bar overflowed. / p_t-v_bar 发生溢出。")
    oriented = price_gap * direction
    return 0.0 if oriented == 0.0 else oriented


def orient_order(
    order: float,
    fundamental_value: float,
    value_mean: float,
) -> float:
    """Calculate ``x_i,t*sign(v_t-v_bar)``. / 计算论文的方向调整订单。"""

    observed_order = _finite_real(order, "order x_i_t")
    direction = orientation_sign(fundamental_value, value_mean)
    oriented = observed_order * direction
    return 0.0 if oriented == 0.0 else oriented


@dataclass(frozen=True)
class UniformShockCalibration:
    """An immutable receipt for one common-magnitude shock. / 一个统一幅度冲击的不可修改计算凭证。"""

    experiment_cell_mean_oriented_price: float
    experiment_cell_mean_price_impact_lambda: float
    experiment_cell_minimum_price_impact_lambda: float
    target_normalized_price_deviation: float
    absolute_noise_shock: float
    implied_oriented_price_increment: float
    implied_shocked_mean_oriented_price: float
    implied_normalized_price_deviation: float
    implied_target_error: float
    paper_shock_period: int
    paper_response_period: int
    paper_required_paths_per_session: int
    protocol_uses_one_common_magnitude_across_sessions_and_paths: bool
    common_magnitude_is_replication_interpretation: bool
    protocol_requires_cell_aggregates_from_same_calibration_sample: bool
    protocol_requires_aggregates_from_same_experiment_cell: bool
    protocol_allows_distinct_long_run_and_t3_samples: bool
    aggregate_inputs_are_caller_supplied: bool
    aggregate_provenance_verified: bool
    underlying_price_impact_positivity_verified_from_raw_paths: bool
    protocol_adds_shock_to_ordinary_noise: bool
    shock_addition_is_replication_interpretation: bool
    numerical_calibration_rule_is_replication_choice: bool
    formula: str


def calibrate_uniform_noise_shock(
    experiment_cell_mean_oriented_price: float,
    experiment_cell_mean_price_impact_lambda: float,
    experiment_cell_minimum_price_impact_lambda: float,
    target_normalized_price_deviation: float = PAPER_TARGET_PRICE_DEVIATION,
) -> UniformShockCalibration:
    """Calibrate one shock magnitude from the average price response.

    根据平均价格反应校准一个统一冲击幅度。

    Hand example / 手算例子:
        E[p_tilde] = 2, E[lambda] = min(lambda) = 0.5, target = 0.012
        |u_shock| = 0.012 * 2 / 0.5 = 0.048
        price increment = 0.5 * 0.048 = 0.024
        normalized increment = 0.024 / 2 = 0.012

    The appendix states the 1.2% target but not its numerical calibration
    routine. Using one common magnitude and the mean ``lambda`` is therefore a
    disclosed replication choice. This low-level function requires a positive
    caller-supplied minimum ``lambda`` but cannot verify its raw paths. Step 35
    must prove that the two means and minimum cover one entire experiment cell
    (all sessions and paths) before reusing one magnitude everywhere. / 附录规定
    1.2% 目标，却未说明数值校准算法。因此“统一幅度 + 平均 lambda”是公开记录的
    复现选择。这个低层函数要求调用者提供正的最小 lambda，却不能验证原始路径；
    第 35 步必须证明两个均值和最小值覆盖整个实验单元的全部 session 与路径，然后
    才能在各处复用同一个幅度。
    """

    mean_price = _finite_real(
        experiment_cell_mean_oriented_price,
        "experiment_cell_mean_oriented_price E[p_tilde]",
    )
    mean_lambda = _finite_real(
        experiment_cell_mean_price_impact_lambda,
        "experiment_cell_mean_price_impact_lambda E[lambda_hat_3]",
    )
    minimum_lambda = _finite_real(
        experiment_cell_minimum_price_impact_lambda,
        "experiment_cell_minimum_price_impact_lambda min[lambda_hat_3]",
    )
    target = _finite_real(
        target_normalized_price_deviation,
        "target_normalized_price_deviation",
    )
    if mean_price <= 0.0:
        raise UndefinedShockCalibrationError(
            "E[p_tilde] must be positive. / E[p_tilde] 必须为正。"
        )
    if mean_lambda <= 0.0:
        raise UndefinedShockCalibrationError(
            "E[lambda_hat_3] must be positive. / E[lambda_hat_3] 必须为正。"
        )
    if minimum_lambda <= 0.0:
        raise UndefinedShockCalibrationError(
            "Every supplied calibration-path lambda must be positive. "
            "/ 所有校准路径的 lambda 都必须为正。"
        )
    if minimum_lambda > mean_lambda:
        raise ValueError(
            "The supplied minimum lambda cannot exceed its mean. "
            "/ 提供的最小 lambda 不能大于其均值。"
        )
    if target <= 0.0:
        raise UndefinedShockCalibrationError(
            "The target deviation must be positive. / 目标偏差必须为正。"
        )

    numerator = target * mean_price
    if not isfinite(numerator):
        raise OverflowError("The calibration numerator overflowed. / 校准分子溢出。")
    magnitude = numerator / mean_lambda
    if not isfinite(magnitude) or magnitude <= 0.0:
        raise UndefinedShockCalibrationError(
            "The shock magnitude is not a positive finite number. / 冲击幅度不是有限正数。"
        )
    price_increment = mean_lambda * magnitude
    shocked_mean = mean_price + price_increment
    achieved = price_increment / mean_price
    error = achieved - target
    if not all(
        isfinite(value)
        for value in (price_increment, shocked_mean, achieved, error)
    ):
        raise OverflowError("The calibrated price response overflowed. / 校准后的价格反应溢出。")

    return UniformShockCalibration(
        experiment_cell_mean_oriented_price=mean_price,
        experiment_cell_mean_price_impact_lambda=mean_lambda,
        experiment_cell_minimum_price_impact_lambda=minimum_lambda,
        target_normalized_price_deviation=target,
        absolute_noise_shock=magnitude,
        implied_oriented_price_increment=price_increment,
        implied_shocked_mean_oriented_price=shocked_mean,
        implied_normalized_price_deviation=achieved,
        implied_target_error=0.0 if error == 0.0 else error,
        paper_shock_period=PAPER_SHOCK_PERIOD,
        paper_response_period=PAPER_RESPONSE_PERIOD,
        paper_required_paths_per_session=PAPER_PATHS_PER_SESSION,
        protocol_uses_one_common_magnitude_across_sessions_and_paths=True,
        common_magnitude_is_replication_interpretation=True,
        # Step 35C supplies the long-run price sample; Step 35D supplies the
        # separate local-t=3 lambda sample. They must belong to one experiment
        # cell, but they are not literally the same observations. / 第 35C 步
        # 提供长期价格样本，第 35D 步提供另一份局部 t=3 lambda 样本；二者必须
        # 属于同一实验单元，但不是同一批 observation。
        protocol_requires_cell_aggregates_from_same_calibration_sample=False,
        protocol_requires_aggregates_from_same_experiment_cell=True,
        protocol_allows_distinct_long_run_and_t3_samples=True,
        aggregate_inputs_are_caller_supplied=True,
        aggregate_provenance_verified=False,
        underlying_price_impact_positivity_verified_from_raw_paths=False,
        protocol_adds_shock_to_ordinary_noise=True,
        shock_addition_is_replication_interpretation=True,
        numerical_calibration_rule_is_replication_choice=True,
        formula=REPLICATION_SHOCK_CALIBRATION_FORMULA,
    )


def validate_uniform_shock_calibration_receipt(
    calibration: UniformShockCalibration,
) -> UniformShockCalibration:
    """Recompute and authenticate one base Step-34 calibration receipt.

    重新计算并验证一份基础的第 34 步冲击校准凭证。

    ``frozen=True`` prevents ordinary assignment after construction, but a
    caller can still create a changed copy with ``dataclasses.replace``. This
    validator therefore rebuilds the receipt from its four primitive numeric
    inputs and requires every derived number, protocol flag, and formula to
    match. It deliberately authenticates arithmetic only: the base receipt
    cannot prove that its caller-supplied aggregates came from the paper's full
    experiment cell. / ``frozen=True`` 只能阻止构造后的普通赋值，调用者仍可用
    ``dataclasses.replace`` 生成被改动的副本。因此本函数从四个原始数值重新计算，
    并要求所有派生数值、协议标志和公式完全一致。它只验证算术，不会假装基础凭证
    已证明调用者提供的总体数据来自论文完整实验单元。
    """

    if not isinstance(calibration, UniformShockCalibration):
        raise TypeError(
            "calibration must be UniformShockCalibration. / "
            "calibration 类型必须是 UniformShockCalibration。"
        )
    try:
        expected = calibrate_uniform_noise_shock(
            calibration.experiment_cell_mean_oriented_price,
            calibration.experiment_cell_mean_price_impact_lambda,
            calibration.experiment_cell_minimum_price_impact_lambda,
            calibration.target_normalized_price_deviation,
        )
    except (TypeError, ValueError, ArithmeticError, OverflowError) as error:
        raise ValueError(
            "The calibration receipt contains invalid primitive inputs. / "
            "冲击校准凭证包含无效的原始输入。"
        ) from error
    if calibration != expected:
        raise ValueError(
            "The calibration receipt is internally inconsistent or claims "
            "unverified provenance. / 冲击校准凭证内部不一致，或声称了未经验证的数据来源。"
        )
    return calibration


@dataclass(frozen=True)
class AppliedNoiseShock:
    """One signed shock added to one ordinary noise order. / 加到一笔普通噪声订单上的一次带符号冲击。"""

    fundamental_value: float
    value_mean: float
    orientation: int
    ordinary_noise_order: float
    absolute_shock_magnitude: float
    signed_adverse_shock: float
    noise_order_used_for_pricing: float
    shock_period: int


def add_adverse_shock_to_noise(
    ordinary_noise_order: float,
    fundamental_value: float,
    value_mean: float,
    absolute_shock_magnitude: float,
) -> AppliedNoiseShock:
    """Apply our disclosed choice to add the calibrated shock to ordinary ``u_3``.

    按照公开说明的复现选择，把校准后的逆向冲击加到普通 ``u_3`` 上；不是替换普通噪声。
    """

    ordinary = _finite_real(ordinary_noise_order, "ordinary_noise_order")
    value = _finite_real(fundamental_value, "fundamental_value v_3")
    mean = _finite_real(value_mean, "value_mean v_bar")
    magnitude = _finite_real(
        absolute_shock_magnitude,
        "absolute_shock_magnitude",
    )
    if magnitude <= 0.0:
        raise ValueError("Shock magnitude must be positive. / 冲击幅度必须为正。")
    direction = orientation_sign(value, mean)
    if direction == 0:
        raise UndefinedShockDirectionError(
            "v_3=v_bar gives no adverse shock direction. / v_3=v_bar 时没有逆向冲击方向。"
        )
    signed_shock = direction * magnitude
    used_noise = ordinary + signed_shock
    if not isfinite(used_noise):
        raise OverflowError("ordinary u_3 + shock overflowed. / 普通 u_3 与冲击之和溢出。")

    return AppliedNoiseShock(
        fundamental_value=value,
        value_mean=mean,
        orientation=direction,
        ordinary_noise_order=ordinary,
        absolute_shock_magnitude=magnitude,
        signed_adverse_shock=signed_shock,
        noise_order_used_for_pricing=used_noise,
        shock_period=PAPER_SHOCK_PERIOD,
    )


@dataclass(frozen=True)
class NormalizedOrderResponse:
    """One agent's inspectable t=4 response calculation. / 一位 agent 可逐项检查的 t=4 反应计算。"""

    agent_number: int
    long_run_mean_oriented_order: float
    shocked_t4_mean_oriented_order: float
    oriented_order_change: float
    normalized_response: float
    response_period: int
    formula: str


def calculate_normalized_order_response(
    agent_number: int,
    long_run_mean_oriented_order: float,
    shocked_t4_mean_oriented_order: float,
) -> NormalizedOrderResponse:
    """Calculate one paper-normalized order response. / 计算一位 agent 的论文标准化订单反应。"""

    agent = _positive_integer(agent_number, "agent_number")
    if agent not in (1, 2):
        raise ValueError(
            "Appendix Section 4.5 supports agent numbers 1 and 2 only. "
            "/ 附录第 4.5 节只支持 agent 编号 1 和 2。"
        )
    baseline = _finite_real(
        long_run_mean_oriented_order,
        "long_run_mean_oriented_order E[x_tilde_i]",
    )
    shocked = _finite_real(
        shocked_t4_mean_oriented_order,
        "shocked_t4_mean_oriented_order",
    )
    if baseline <= 0.0:
        raise UndefinedOrderResponseError(
            "E[x_tilde_i] must be positive. / E[x_tilde_i] 必须为正。"
        )
    change = shocked - baseline
    response = change / baseline
    if not isfinite(change) or not isfinite(response):
        raise OverflowError("The normalized order response overflowed. / 标准化订单反应溢出。")

    return NormalizedOrderResponse(
        agent_number=agent,
        long_run_mean_oriented_order=baseline,
        shocked_t4_mean_oriented_order=shocked,
        oriented_order_change=0.0 if change == 0.0 else change,
        normalized_response=0.0 if response == 0.0 else response,
        response_period=PAPER_RESPONSE_PERIOD,
        formula=PAPER_NORMALIZED_ORDER_RESPONSE_FORMULA,
    )


def _exact_two_responses(responses: Sequence[float]) -> tuple[float, float]:
    """Copy and validate the paper's two agent responses. / 复制并验证论文中的两个 agent 反应。"""

    if isinstance(responses, (str, bytes)) or not isinstance(responses, Sequence):
        raise TypeError("responses must be a sequence. / responses 必须是序列。")
    if len(responses) != PAPER_CLASSIFIER_AGENTS:
        raise ValueError(
            "Appendix Section 4.5 requires exactly two agent responses. "
            "/ 附录第 4.5 节要求恰好两个 agent 反应。"
        )
    return (
        _finite_real(responses[0], "responses[0]"),
        _finite_real(responses[1], "responses[1]"),
    )


@dataclass(frozen=True)
class MechanismClassification:
    """An immutable, auditable Step-34 classification. / 一份不可修改、可审计的第 34 步分类。"""

    mechanism: CollusionMechanism
    normalized_order_responses: tuple[float, float]
    price_trigger_pass_by_agent: tuple[bool, bool]
    over_pruning_pass_by_agent: tuple[bool, bool]
    low_response_threshold: float
    high_response_threshold: float
    paper_thresholds_used: bool
    both_agents_required: bool
    strict_inequalities_used: bool
    unclassified_label_is_replication_completion_rule: bool
    exact_threshold_behavior_follows_strict_inequalities: bool
    paper_required_shock_period: int
    paper_required_response_period: int
    paper_required_paths_per_session: int
    input_statistics_are_caller_supplied: bool
    input_horizon_verified: bool
    input_path_count_verified: bool
    same_session_and_checkpoint_provenance_verified: bool
    irf_paths_generated_by_this_function: bool
    low_rule_parentheses_are_replication_interpretation: bool
    low_rule_used: str
    classifier_version: str


def classify_normalized_order_responses(
    responses: Sequence[float],
    *,
    low_response_threshold: float = PAPER_LOW_RESPONSE_THRESHOLD,
    high_response_threshold: float = PAPER_HIGH_RESPONSE_THRESHOLD,
) -> MechanismClassification:
    """Apply the two strict, two-agent Appendix-4.5 rules.

    应用附录第 4.5 节的两个严格、双 agent 分类规则。

    ``responses`` must already be session-level responses after orienting every
    raw path, averaging the t=4 oriented orders, and normalizing by long-run
    means. This function does not classify individual paths and vote. / ``responses``
    必须是 session 层面的反应：先逐条路径调整方向，再平均 t=4 订单，最后除以长期
    均值。本函数不会逐路径分类后投票。

    This arithmetic primitive cannot prove where two naked floats came from.
    It records the paper's required t=3/t=4/10,000-path context but marks all
    such provenance unverified. Step 35 must accept only a trusted aggregate
    receipt before using this primitive for results. / 这个算术原语无法证明两个裸
    浮点数的来源。它记录论文要求的 t=3、t=4 与 10,000 条路径背景，但把这些来源
    全部标成“尚未验证”。第 35 步必须只接收可信的聚合 receipt，再调用本原语生成结果。
    """

    pair = _exact_two_responses(responses)
    low = _finite_real(low_response_threshold, "low_response_threshold")
    high = _finite_real(high_response_threshold, "high_response_threshold")
    if low <= 0.0 or high <= low:
        raise ValueError(
            "Thresholds must satisfy 0 < low < high. / 阈值必须满足 0 < low < high。"
        )

    trigger_flags = (pair[0] > high, pair[1] > high)
    pruning_flags = (abs(pair[0]) < low, abs(pair[1]) < low)
    if all(trigger_flags):
        mechanism = CollusionMechanism.PRICE_TRIGGER
    elif all(pruning_flags):
        mechanism = CollusionMechanism.OVER_PRUNING
    else:
        mechanism = CollusionMechanism.UNCLASSIFIED

    return MechanismClassification(
        mechanism=mechanism,
        normalized_order_responses=pair,
        price_trigger_pass_by_agent=trigger_flags,
        over_pruning_pass_by_agent=pruning_flags,
        low_response_threshold=low,
        high_response_threshold=high,
        paper_thresholds_used=(
            low == PAPER_LOW_RESPONSE_THRESHOLD
            and high == PAPER_HIGH_RESPONSE_THRESHOLD
        ),
        both_agents_required=True,
        strict_inequalities_used=True,
        unclassified_label_is_replication_completion_rule=True,
        exact_threshold_behavior_follows_strict_inequalities=True,
        paper_required_shock_period=PAPER_SHOCK_PERIOD,
        paper_required_response_period=PAPER_RESPONSE_PERIOD,
        paper_required_paths_per_session=PAPER_PATHS_PER_SESSION,
        input_statistics_are_caller_supplied=True,
        input_horizon_verified=False,
        input_path_count_verified=False,
        same_session_and_checkpoint_provenance_verified=False,
        irf_paths_generated_by_this_function=False,
        low_rule_parentheses_are_replication_interpretation=True,
        low_rule_used=(
            REPLICATION_LOW_RESPONSE_RULE
            if low == PAPER_LOW_RESPONSE_THRESHOLD
            else (
                "abs((x_tilde_i_4 - E[x_tilde_i]) / E[x_tilde_i]) "
                f"< {low!r}"
            )
        ),
        classifier_version=CLASSIFIER_VERSION,
    )


def main() -> None:
    """Run a calculator-style demonstration. / 运行计算器式演示。"""

    calibration = calibrate_uniform_noise_shock(
        experiment_cell_mean_oriented_price=2.0,
        experiment_cell_mean_price_impact_lambda=0.5,
        experiment_cell_minimum_price_impact_lambda=0.5,
    )
    assert abs(calibration.absolute_noise_shock - 0.048) < 1e-15
    assert abs(calibration.implied_normalized_price_deviation - 0.012) < 1e-15

    high_value_shock = add_adverse_shock_to_noise(
        ordinary_noise_order=0.10,
        fundamental_value=2.0,
        value_mean=1.0,
        absolute_shock_magnitude=calibration.absolute_noise_shock,
    )
    low_value_shock = add_adverse_shock_to_noise(
        ordinary_noise_order=0.10,
        fundamental_value=0.0,
        value_mean=1.0,
        absolute_shock_magnitude=calibration.absolute_noise_shock,
    )
    assert abs(high_value_shock.noise_order_used_for_pricing - 0.148) < 1e-15
    assert abs(low_value_shock.noise_order_used_for_pricing - 0.052) < 1e-15

    agent_1 = calculate_normalized_order_response(1, 100.0, 100.06)
    agent_2 = calculate_normalized_order_response(2, 200.0, 200.12)
    price_trigger = classify_normalized_order_responses(
        (agent_1.normalized_response, agent_2.normalized_response)
    )
    over_pruning = classify_normalized_order_responses((4e-5, -4e-5))
    mixed = classify_normalized_order_responses((6e-4, 0.0))
    assert price_trigger.mechanism is CollusionMechanism.PRICE_TRIGGER
    assert over_pruning.mechanism is CollusionMechanism.OVER_PRUNING
    assert mixed.mechanism is CollusionMechanism.UNCLASSIFIED

    print("Step 34: Mechanism classifier / 第 34 步：机制分类器")
    print("Shock calibration hand example / 冲击校准手算例子:")
    print("  E[p_tilde] / 长期方向调整价格均值: 2.000")
    print("  E[lambda_3] / t=3 平均价格冲击: 0.500")
    print(f"  Common |u_shock| / 统一冲击幅度: {calibration.absolute_noise_shock:.3f}")
    print(f"  Implied price deviation / 公式隐含价格偏差: {calibration.implied_normalized_price_deviation:.3%}")
    print("Our provisional protocol adds rather than substitutes / 当前暂定协议采用相加而非替换:")
    print(f"  v>v_bar: 0.100 + 0.048 = {high_value_shock.noise_order_used_for_pricing:.3f}")
    print(f"  v<v_bar: 0.100 - 0.048 = {low_value_shock.noise_order_used_for_pricing:.3f}")
    print("Classification examples / 分类例子:")
    print(f"  (0.0006, 0.0006) -> {price_trigger.mechanism.value}")
    print(f"  (0.00004, -0.00004) -> {over_pruning.mechanism.value}")
    print(f"  (0.0006, 0.0) -> {mixed.mechanism.value}")
    print("The actual 10,000-path provenance is not verified here. / 本步骤不验证真实的 10,000 条路径来源。")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
