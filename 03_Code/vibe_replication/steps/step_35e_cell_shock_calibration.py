"""Step 35E: pool one experiment cell and calibrate one common t=3 shock.

第 35E 步：汇总一个实验单元，并校准一个统一的 t=3 冲击。

Run the small demonstration / 运行小型演示:
    py -3 -X utf8 steps/step_35e_cell_shock_calibration.py

Paper target / 论文目标:
    The appendix chooses one adverse noise-shock magnitude for every session in
    the same experiment cell.  The shocked oriented-price LEVEL at local t=3
    should be 1.2% above its long-run mean. / 附录为同一实验单元中的所有
    session 选择一个统一的逆向噪声冲击幅度，并要求局部 t=3 的受冲击方向
    调整价格“水平”比长期均值高 1.2%。

Primary finite-sample formula / 主要有限样本公式:
    m_level = ((1 + delta) * mean_long_run_price
               - mean_unshocked_t3_price) / mean_actual_t3_lambda

The old increment shortcut is retained only as a sensitivity / 旧增量公式仅作敏感性:
    m_increment = delta * mean_long_run_price / mean_actual_t3_lambda

The two magnitudes coincide only when the unshocked t=3 price mean equals the
long-run price mean.  The paper states the 1.2% target but does not disclose
its numerical calibration algorithm, weighting convention, or software
protocol.  Our exact-level rule and raw-observation weighting are therefore
explicit replication choices. / 只有当无冲击 t=3 价格均值恰好等于长期价格
均值时，两种幅度才相同。原文给出了 1.2% 目标，却没有公开数值算法、加权
方式或软件协议；因此 exact-level 规则与按底层 observation 数量加权都是
本复现明确披露的选择。

Strict boundary / 明确边界:
    This step computes one immutable positive magnitude.  It applies no shock,
    runs no treatment path, does not execute t=4, and issues no mechanism
    label.  Those actions belong to Step 35F. / 本步骤只计算一个不可修改的
    正冲击幅度；不施加冲击、不运行实验路径、不执行 t=4，也不分类机制。
    这些属于第 35F 步。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from math import fsum, isclose, isfinite
from numbers import Integral, Real
from pathlib import Path
import pickle
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from step_26_reproducible_random_streams import (
    SessionSeedManifest,
    build_session_seed_manifest,
)
from step_28_session_phases import SessionPhase
from steps.step_34_mechanism_classifier import PAPER_TARGET_PRICE_DEVIATION
from steps.step_35d_unshocked_t3_calibration_paths import (
    UnshockedT3SessionCalibrationReceipt,
    run_unshocked_t3_calibration_paths,
    validate_unshocked_t3_session_calibration_receipt,
)


PAPER_SESSIONS_PER_EXPERIMENT_CELL = 1_000
CELL_SHOCK_CALIBRATION_VERSION = "step35e-cell-shock-calibration-v1"
CELL_SHOCK_RECEIPT_DOMAIN = b"vibe-replication.step35e.cell-shock-receipt.v1\0"
ORDERED_SESSION_RECEIPTS_DOMAIN = b"vibe-replication.step35e.ordered-session-receipts.v1\0"
EXACT_LEVEL_RULE = "exact_finite_sample_t3_level"
INCREMENT_SHORTCUT_RULE = "treatment_control_increment_sensitivity"
RAW_OBSERVATION_WEIGHTING = (
    "weight long-run means by measurement rows and t3 means by executed paths"
)
EXACT_LEVEL_FORMULA = (
    "m_level = ((1 + target) * mean_long_run_p_tilde "
    "- mean_unshocked_p_tilde_3) / mean_actual_lambda_3"
)
INCREMENT_SHORTCUT_FORMULA = (
    "m_increment = target * mean_long_run_p_tilde / mean_actual_lambda_3"
)


class UndefinedCellShockCalibrationError(ArithmeticError):
    """The requested positive adverse cell shock does not exist. / 所要求的正逆向冲击不存在。"""


def _finite_real(number: float, label: str) -> float:
    """Return one finite non-Boolean float. / 返回一个有限、非布尔浮点数。"""

    if isinstance(number, bool) or not isinstance(number, Real):
        raise TypeError(f"{label} must be a real number. / {label} 必须是实数。")
    converted = float(number)
    if not isfinite(converted):
        raise ValueError(f"{label} must be finite. / {label} 必须是有限数。")
    return converted


def _positive_integer(number: int, label: str) -> int:
    """Return one positive non-Boolean integer. / 返回一个正的、非布尔整数。"""

    if isinstance(number, bool) or not isinstance(number, Integral):
        raise TypeError(f"{label} must be an integer. / {label} 必须是整数。")
    checked = int(number)
    if checked < 1:
        raise ValueError(f"{label} must be positive. / {label} 必须为正。")
    return checked


def _weighted_mean(
    mean_and_count: Sequence[tuple[float, int]],
    label: str,
) -> tuple[float, int]:
    """Pool means as if their discarded underlying rows were concatenated.

    像把已经丢弃的底层 observation 重新拼接一样，对均值进行加权。
    """

    if not mean_and_count:
        raise ValueError(f"{label} has no observations. / {label} 没有 observation。")
    checked: list[tuple[float, int]] = []
    for mean, count in mean_and_count:
        checked.append(
            (
                _finite_real(mean, f"{label} mean"),
                _positive_integer(count, f"{label} count"),
            )
        )
    total_count = sum(count for _, count in checked)
    numerator = fsum(mean * count for mean, count in checked)
    pooled = numerator / total_count
    if not isfinite(pooled):
        raise OverflowError(f"{label} pooled mean overflowed. / {label} 汇总均值溢出。")
    return pooled, total_count


@dataclass(frozen=True)
class CellShockCalibrationArithmetic:
    """Hand-checkable comparison of the exact-level and shortcut formulas.

    exact-level 与 shortcut 两种公式的可手算、不可修改比较。
    """

    target_normalized_price_level_deviation: float
    mean_long_run_oriented_price: float
    mean_unshocked_t3_oriented_price: float
    mean_actual_t3_price_impact_lambda: float
    minimum_actual_t3_price_impact_lambda: float
    exact_level_required_oriented_price_increment: float
    exact_level_required_normalized_increment: float
    exact_level_absolute_noise_shock: float
    exact_level_implied_shocked_t3_oriented_price: float
    exact_level_achieved_normalized_level_deviation: float
    exact_level_target_error: float
    increment_shortcut_absolute_noise_shock: float
    increment_shortcut_implied_shocked_t3_oriented_price: float
    increment_shortcut_achieved_normalized_level_deviation: float
    increment_shortcut_achieved_normalized_treatment_control_increment: float
    increment_shortcut_level_target_error: float
    absolute_magnitude_difference: float
    formulas_coincide: bool
    primary_rule: str
    exact_level_formula: str
    increment_shortcut_formula: str
    paper_specifies_numerical_calibration_algorithm: bool
    exact_level_rule_is_replication_interpretation: bool
    increment_shortcut_retained_as_sensitivity: bool


def calculate_cell_shock_calibration_arithmetic(
    mean_long_run_oriented_price: float,
    mean_unshocked_t3_oriented_price: float,
    mean_actual_t3_price_impact_lambda: float,
    minimum_actual_t3_price_impact_lambda: float,
    target_normalized_price_level_deviation: float = PAPER_TARGET_PRICE_DEVIATION,
) -> CellShockCalibrationArithmetic:
    """Calculate both Step-35E magnitudes from four pooled moments.

    使用四个汇总统计量计算第 35E 步的两种冲击幅度。

    Beginner hand example / 初学者手算例子:
        long-run price P = 3
        unshocked t=3 price P0 = 2.8
        actual t=3 lambda L = min(L) = 1
        target delta = 0.012

        exact:    m = ((1.012 * 3) - 2.8) / 1 = 0.236
        shortcut: m = (0.012 * 3) / 1 = 0.036

        Only 0.236 makes (2.8 + 1 * m - 3) / 3 equal 0.012. /
        只有 0.236 能使 (2.8 + 1*m - 3)/3 等于 0.012。
    """

    long_run = _finite_real(
        mean_long_run_oriented_price,
        "mean_long_run_oriented_price",
    )
    unshocked_t3 = _finite_real(
        mean_unshocked_t3_oriented_price,
        "mean_unshocked_t3_oriented_price",
    )
    mean_lambda = _finite_real(
        mean_actual_t3_price_impact_lambda,
        "mean_actual_t3_price_impact_lambda",
    )
    minimum_lambda = _finite_real(
        minimum_actual_t3_price_impact_lambda,
        "minimum_actual_t3_price_impact_lambda",
    )
    target = _finite_real(
        target_normalized_price_level_deviation,
        "target_normalized_price_level_deviation",
    )
    if long_run <= 0.0:
        raise UndefinedCellShockCalibrationError(
            "The long-run oriented-price denominator must be positive. "
            "/ 长期方向调整价格分母必须为正。"
        )
    if mean_lambda <= 0.0 or minimum_lambda <= 0.0:
        raise UndefinedCellShockCalibrationError(
            "Every actual t=3 lambda and its mean must be positive. "
            "/ 每条路径真正的 t=3 lambda 及其均值都必须为正。"
        )
    if minimum_lambda > mean_lambda and not isclose(
        minimum_lambda,
        mean_lambda,
        rel_tol=1e-15,
        abs_tol=0.0,
    ):
        raise ValueError(
            "The minimum t=3 lambda cannot exceed its mean. "
            "/ t=3 lambda 最小值不能大于均值。"
        )
    if target <= 0.0:
        raise UndefinedCellShockCalibrationError(
            "The normalized level target must be positive. / 标准化价格水平目标必须为正。"
        )

    target_level = (1.0 + target) * long_run
    required_increment = target_level - unshocked_t3
    if not isfinite(target_level) or not isfinite(required_increment):
        raise OverflowError("The exact-level numerator overflowed. / exact-level 分子溢出。")
    if required_increment <= 0.0:
        raise UndefinedCellShockCalibrationError(
            "The unshocked t=3 level is already at or above the target, so a "
            "positive adverse shock cannot hit it. / 无冲击 t=3 水平已经达到或超过目标，"
            "因此无法用正逆向冲击命中该目标。"
        )

    exact_magnitude = required_increment / mean_lambda
    shortcut_magnitude = target * long_run / mean_lambda
    exact_shocked_level = unshocked_t3 + mean_lambda * exact_magnitude
    shortcut_shocked_level = unshocked_t3 + mean_lambda * shortcut_magnitude
    exact_level_deviation = (exact_shocked_level - long_run) / long_run
    shortcut_level_deviation = (shortcut_shocked_level - long_run) / long_run
    exact_required_increment_normalized = required_increment / long_run
    shortcut_increment_normalized = (
        mean_lambda * shortcut_magnitude / long_run
    )
    exact_error = exact_level_deviation - target
    shortcut_error = shortcut_level_deviation - target
    difference = exact_magnitude - shortcut_magnitude
    formulas_coincide = isclose(
        exact_magnitude,
        shortcut_magnitude,
        rel_tol=1e-15,
        abs_tol=0.0,
    )
    values = (
        exact_magnitude,
        shortcut_magnitude,
        exact_shocked_level,
        shortcut_shocked_level,
        exact_level_deviation,
        shortcut_level_deviation,
        exact_required_increment_normalized,
        shortcut_increment_normalized,
        exact_error,
        shortcut_error,
        difference,
    )
    if any(not isfinite(value) for value in values):
        raise OverflowError("Shock-calibration arithmetic overflowed. / 冲击校准计算溢出。")
    if exact_magnitude <= 0.0 or shortcut_magnitude <= 0.0:
        raise UndefinedCellShockCalibrationError(
            "Both shock magnitudes must be positive. / 两种冲击幅度都必须为正。"
        )

    return CellShockCalibrationArithmetic(
        target_normalized_price_level_deviation=target,
        mean_long_run_oriented_price=long_run,
        mean_unshocked_t3_oriented_price=unshocked_t3,
        mean_actual_t3_price_impact_lambda=mean_lambda,
        minimum_actual_t3_price_impact_lambda=minimum_lambda,
        exact_level_required_oriented_price_increment=required_increment,
        exact_level_required_normalized_increment=(
            exact_required_increment_normalized
        ),
        exact_level_absolute_noise_shock=exact_magnitude,
        exact_level_implied_shocked_t3_oriented_price=exact_shocked_level,
        exact_level_achieved_normalized_level_deviation=exact_level_deviation,
        exact_level_target_error=0.0 if exact_error == 0.0 else exact_error,
        increment_shortcut_absolute_noise_shock=shortcut_magnitude,
        increment_shortcut_implied_shocked_t3_oriented_price=(
            shortcut_shocked_level
        ),
        increment_shortcut_achieved_normalized_level_deviation=(
            shortcut_level_deviation
        ),
        increment_shortcut_achieved_normalized_treatment_control_increment=(
            shortcut_increment_normalized
        ),
        increment_shortcut_level_target_error=(
            0.0 if shortcut_error == 0.0 else shortcut_error
        ),
        absolute_magnitude_difference=(
            0.0 if formulas_coincide else difference
        ),
        formulas_coincide=formulas_coincide,
        primary_rule=EXACT_LEVEL_RULE,
        exact_level_formula=EXACT_LEVEL_FORMULA,
        increment_shortcut_formula=INCREMENT_SHORTCUT_FORMULA,
        paper_specifies_numerical_calibration_algorithm=False,
        exact_level_rule_is_replication_interpretation=True,
        increment_shortcut_retained_as_sensitivity=True,
    )


def validate_cell_shock_calibration_arithmetic(
    arithmetic: CellShockCalibrationArithmetic,
) -> None:
    """Recompute every derived arithmetic field. / 重新计算全部派生字段。"""

    if not isinstance(arithmetic, CellShockCalibrationArithmetic):
        raise TypeError("arithmetic has the wrong type. / arithmetic 类型错误。")
    expected = calculate_cell_shock_calibration_arithmetic(
        arithmetic.mean_long_run_oriented_price,
        arithmetic.mean_unshocked_t3_oriented_price,
        arithmetic.mean_actual_t3_price_impact_lambda,
        arithmetic.minimum_actual_t3_price_impact_lambda,
        arithmetic.target_normalized_price_level_deviation,
    )
    if arithmetic != expected:
        raise ValueError("Shock arithmetic was changed. / 冲击计算凭证已被修改。")


@dataclass(frozen=True)
class ExperimentCellShockCalibrationReceipt:
    """Immutable Step-35E receipt for one experiment cell.

    一个实验单元的不可修改第 35E 步凭证。
    """

    protocol_version: str
    primary_rule: str
    pooling_rule: str
    experiment_seed: int
    experiment_cell_key: str
    experiment_cell_seed: int
    irf_experiment_seed: int
    session_seed_derivation_version: str
    path_seed_derivation_version: str
    implementation_tree_sha256: str
    parameter_snapshot: PaperParameters
    value_grid_snapshot: tuple[float, ...]
    sessions_expected: int
    sessions_received: int
    ordered_session_indexes: tuple[int, ...]
    total_long_run_measurement_rows: int
    total_unshocked_t3_paths: int
    number_of_agents: int
    pooled_long_run_mean_oriented_price: float
    pooled_long_run_mean_oriented_order_by_agent: tuple[float, ...]
    pooled_long_run_mean_profit_by_agent: tuple[float, ...]
    pooled_unshocked_t3_mean_oriented_price: float
    pooled_actual_t3_mean_price_impact_lambda: float
    minimum_actual_t3_price_impact_lambda: float
    nonpositive_actual_t3_lambda_count: int
    arithmetic: CellShockCalibrationArithmetic
    selected_absolute_noise_shock: float
    increment_shortcut_sensitivity_shock: float
    ordered_source_receipts_sha256: str
    unique_session_index_count: int
    unique_session_seed_count: int
    unique_base_stream_seed_count: int
    planned_base_stream_seed_count: int
    unique_checkpoint_digest_count: int
    unique_step35d_receipt_digest_count: int
    unique_baseline_receipt_digest_count: int
    same_experiment_cell_verified: bool
    all_source_receipts_validated: bool
    canonical_session_index_coverage_verified: bool
    distinct_session_identities_verified: bool
    cross_session_base_stream_seed_uniqueness_verified: bool
    cross_session_path_seed_namespace_inputs_unique: bool
    all_twenty_million_path_child_seed_outputs_explicitly_compared: bool
    raw_observation_weighting_used: bool
    all_actual_t3_lambdas_positive: bool
    one_common_magnitude_selected_for_entire_cell: bool
    uniform_cell_shock_calibrated: bool
    paper_1000_sessions_verified: bool
    paper_10000_paths_per_session_verified: bool
    paper_scale_source_receipts_verified: bool
    formal_cross_session_seed_namespace_audit_verified: bool
    ready_for_formal_step35f: bool
    exact_level_target_satisfied_in_calibration_arithmetic: bool
    paper_target_observed_on_shocked_paths: bool
    shock_applied: bool
    treatment_paths_executed: int
    t4_response_aggregated: bool
    classification_ready: bool
    mechanism_label_ready: bool
    paper_figure_ready: bool
    paper_specifies_numerical_calibration_algorithm: bool
    exact_level_rule_is_replication_interpretation: bool
    increment_shortcut_retained_as_sensitivity: bool
    checksum_detects_stale_replacement_not_authentication: bool
    standalone_receipt_authenticates_discarded_source_rows: bool
    receipt_payload_sha256: str


def _cell_receipt_payload_digest(
    receipt: ExperimentCellShockCalibrationReceipt,
) -> str:
    """Checksum every receipt field except this checksum. / 校验除自身以外的全部字段。"""

    unsigned = replace(receipt, receipt_payload_sha256="")
    return sha256(
        CELL_SHOCK_RECEIPT_DOMAIN + pickle.dumps(unsigned, protocol=5)
    ).hexdigest()


def _is_sha256_text(value: object) -> bool:
    """Return whether a value is lowercase SHA-256 text. / 判断是否为小写 SHA-256 文本。"""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_experiment_cell_shock_calibration_receipt(
    receipt: ExperimentCellShockCalibrationReceipt,
) -> None:
    """Reject a changed or logically exaggerated Step-35E receipt.

    拒绝被修改或夸大完成程度的第 35E 步凭证。
    """

    if not isinstance(receipt, ExperimentCellShockCalibrationReceipt):
        raise TypeError("receipt has the wrong type. / receipt 类型错误。")
    if receipt.protocol_version != CELL_SHOCK_CALIBRATION_VERSION:
        raise ValueError("Receipt version is unsupported. / receipt 版本不支持。")
    if not _is_sha256_text(receipt.receipt_payload_sha256):
        raise ValueError("Receipt checksum format is invalid. / receipt 校验码格式错误。")
    if _cell_receipt_payload_digest(receipt) != receipt.receipt_payload_sha256:
        raise ValueError("Receipt checksum failed. / receipt 校验失败。")
    if not _is_sha256_text(receipt.ordered_source_receipts_sha256):
        raise ValueError("Source-receipt digest is invalid. / 来源 receipt 摘要无效。")
    validate_cell_shock_calibration_arithmetic(receipt.arithmetic)

    expected_count = _positive_integer(receipt.sessions_expected, "sessions_expected")
    received_count = _positive_integer(receipt.sessions_received, "sessions_received")
    if expected_count > PAPER_SESSIONS_PER_EXPERIMENT_CELL:
        raise ValueError("sessions_expected cannot exceed 1,000. / sessions_expected 不能超过 1000。")
    canonical_indexes = tuple(range(expected_count))
    canonical = (
        received_count == expected_count
        and receipt.ordered_session_indexes == canonical_indexes
    )
    distinct = (
        receipt.unique_session_index_count == received_count
        and receipt.unique_session_seed_count == received_count
        and receipt.unique_checkpoint_digest_count == received_count
        and receipt.unique_step35d_receipt_digest_count == received_count
        and receipt.unique_baseline_receipt_digest_count == received_count
    )
    base_seed_unique = (
        receipt.unique_base_stream_seed_count
        == receipt.planned_base_stream_seed_count
        == 7 * received_count
    )
    positive_lambdas = (
        receipt.pooled_actual_t3_mean_price_impact_lambda > 0.0
        and receipt.minimum_actual_t3_price_impact_lambda > 0.0
        and receipt.nonpositive_actual_t3_lambda_count == 0
    )
    paper_sessions = canonical and expected_count == PAPER_SESSIONS_PER_EXPERIMENT_CELL
    formal_ready = (
        paper_sessions
        and receipt.paper_10000_paths_per_session_verified
        and receipt.paper_scale_source_receipts_verified
        and distinct
        and base_seed_unique
        and receipt.cross_session_path_seed_namespace_inputs_unique
        and positive_lambdas
    )
    exact_target = isclose(
        receipt.arithmetic.exact_level_achieved_normalized_level_deviation,
        receipt.arithmetic.target_normalized_price_level_deviation,
        rel_tol=0.0,
        abs_tol=8.0 * sys.float_info.epsilon,
    )
    if (
        receipt.primary_rule != EXACT_LEVEL_RULE
        or receipt.pooling_rule != RAW_OBSERVATION_WEIGHTING
        or receipt.selected_absolute_noise_shock
        != receipt.arithmetic.exact_level_absolute_noise_shock
        or receipt.increment_shortcut_sensitivity_shock
        != receipt.arithmetic.increment_shortcut_absolute_noise_shock
        or receipt.pooled_long_run_mean_oriented_price
        != receipt.arithmetic.mean_long_run_oriented_price
        or receipt.pooled_unshocked_t3_mean_oriented_price
        != receipt.arithmetic.mean_unshocked_t3_oriented_price
        or receipt.pooled_actual_t3_mean_price_impact_lambda
        != receipt.arithmetic.mean_actual_t3_price_impact_lambda
        or receipt.minimum_actual_t3_price_impact_lambda
        != receipt.arithmetic.minimum_actual_t3_price_impact_lambda
        or len(receipt.pooled_long_run_mean_oriented_order_by_agent)
        != receipt.number_of_agents
        or len(receipt.pooled_long_run_mean_profit_by_agent)
        != receipt.number_of_agents
        or receipt.total_long_run_measurement_rows < received_count
        or receipt.total_unshocked_t3_paths < received_count
        or receipt.nonpositive_actual_t3_lambda_count < 0
    ):
        raise ValueError("Receipt arithmetic or counts are inconsistent. / receipt 计算或计数不一致。")

    logical_claims = (
        receipt.same_experiment_cell_verified,
        receipt.all_source_receipts_validated,
        receipt.canonical_session_index_coverage_verified == canonical,
        receipt.distinct_session_identities_verified == distinct,
        receipt.cross_session_base_stream_seed_uniqueness_verified
        == base_seed_unique,
        receipt.cross_session_path_seed_namespace_inputs_unique,
        not receipt.all_twenty_million_path_child_seed_outputs_explicitly_compared,
        receipt.raw_observation_weighting_used,
        receipt.all_actual_t3_lambdas_positive == positive_lambdas,
        receipt.one_common_magnitude_selected_for_entire_cell,
        receipt.uniform_cell_shock_calibrated,
        receipt.paper_1000_sessions_verified == paper_sessions,
        receipt.formal_cross_session_seed_namespace_audit_verified
        == (paper_sessions and distinct and base_seed_unique),
        receipt.ready_for_formal_step35f == formal_ready,
        receipt.exact_level_target_satisfied_in_calibration_arithmetic
        == exact_target,
        not receipt.paper_target_observed_on_shocked_paths,
        not receipt.shock_applied,
        receipt.treatment_paths_executed == 0,
        not receipt.t4_response_aggregated,
        not receipt.classification_ready,
        not receipt.mechanism_label_ready,
        not receipt.paper_figure_ready,
        not receipt.paper_specifies_numerical_calibration_algorithm,
        receipt.exact_level_rule_is_replication_interpretation,
        receipt.increment_shortcut_retained_as_sensitivity,
        receipt.checksum_detects_stale_replacement_not_authentication,
        not receipt.standalone_receipt_authenticates_discarded_source_rows,
    )
    if not all(logical_claims):
        raise ValueError("Receipt claims are inconsistent. / receipt 声明不一致。")


def _ordered_receipts_digest(
    receipts: Sequence[UnshockedT3SessionCalibrationReceipt],
) -> str:
    """Bind the ordered Step-35D receipt checksums to one digest. / 把有序的35D校验码绑定成一个摘要。"""

    digest = sha256(ORDERED_SESSION_RECEIPTS_DOMAIN)
    for receipt in receipts:
        manifest = receipt.source_seed_manifest
        digest.update(manifest.session_index.to_bytes(8, "big"))
        digest.update(bytes.fromhex(receipt.receipt_payload_sha256))
    return digest.hexdigest()


def calibrate_experiment_cell_uniform_shock(
    session_receipts: Sequence[UnshockedT3SessionCalibrationReceipt],
    *,
    expected_session_count: int = PAPER_SESSIONS_PER_EXPERIMENT_CELL,
    target_normalized_price_level_deviation: float = PAPER_TARGET_PRICE_DEVIATION,
) -> ExperimentCellShockCalibrationReceipt:
    """Pool verified Step-35D receipts and select one exact-level magnitude.

    汇总已核验的第 35D 步凭证，并选择一个 exact-level 统一幅度。

    ``expected_session_count`` may be smaller than 1,000 for a code test, but
    the returned receipt then states that it is not paper-scale. / 为了代码
    测试，``expected_session_count`` 可以小于 1000；返回凭证会明确说明它不是
    论文规模结果。
    """

    expected = _positive_integer(expected_session_count, "expected_session_count")
    if expected > PAPER_SESSIONS_PER_EXPERIMENT_CELL:
        raise ValueError("expected_session_count cannot exceed 1,000. / 预期 session 数不能超过 1000。")
    if isinstance(session_receipts, (str, bytes)) or not isinstance(
        session_receipts,
        Sequence,
    ):
        raise TypeError("session_receipts must be a sequence. / session_receipts 必须是序列。")
    supplied = tuple(session_receipts)
    if len(supplied) != expected:
        raise ValueError(
            f"Expected exactly {expected} session receipts, got {len(supplied)}. "
            f"/ 必须正好提供 {expected} 个 session receipt，实际为 {len(supplied)}。"
        )
    for receipt in supplied:
        validate_unshocked_t3_session_calibration_receipt(receipt)
        if not receipt.ready_for_cell_aggregation:
            raise ValueError(
                "A Step-35D receipt is not ready for cell aggregation. "
                "/ 某个第 35D 步 receipt 尚不能进行实验单元汇总。"
            )
    receipts = tuple(
        sorted(supplied, key=lambda item: item.source_seed_manifest.session_index)
    )
    indexes = tuple(receipt.source_seed_manifest.session_index for receipt in receipts)
    if indexes != tuple(range(expected)):
        raise ValueError(
            "Session indexes must cover exactly 0..expected-1 with no gaps. "
            "/ session 编号必须无遗漏地覆盖 0..expected-1。"
        )

    reference = receipts[0]
    reference_baseline = reference.long_run_baseline_receipt
    reference_manifest = reference.source_seed_manifest
    target = _finite_real(
        target_normalized_price_level_deviation,
        "target_normalized_price_level_deviation",
    )
    same_cell_fields = (
        reference.implementation_tree_sha256,
        reference_baseline.parameter_snapshot,
        reference_baseline.value_grid_snapshot,
        reference_manifest.seed_derivation_version,
        reference_manifest.experiment_seed,
        reference_manifest.experiment_cell_key,
        reference_manifest.experiment_cell_seed,
        reference_manifest.rng_engine,
        reference_manifest.python_version,
        reference.irf_experiment_seed,
        reference.path_seed_derivation_version,
        reference.target_normalized_price_deviation,
        reference_baseline.number_of_agents,
    )
    for receipt in receipts:
        baseline = receipt.long_run_baseline_receipt
        manifest = receipt.source_seed_manifest
        candidate_fields = (
            receipt.implementation_tree_sha256,
            baseline.parameter_snapshot,
            baseline.value_grid_snapshot,
            manifest.seed_derivation_version,
            manifest.experiment_seed,
            manifest.experiment_cell_key,
            manifest.experiment_cell_seed,
            manifest.rng_engine,
            manifest.python_version,
            receipt.irf_experiment_seed,
            receipt.path_seed_derivation_version,
            receipt.target_normalized_price_deviation,
            baseline.number_of_agents,
        )
        if candidate_fields != same_cell_fields:
            raise ValueError(
                "All receipts must belong to the same experiment cell and build. "
                "/ 所有 receipt 必须来自同一实验单元与同一代码 build。"
            )
        rebuilt_manifest = build_session_seed_manifest(
            manifest.experiment_seed,
            manifest.experiment_cell_key,
            manifest.session_index,
        )
        if rebuilt_manifest != manifest:
            raise ValueError(
                "A source session seed manifest cannot be reproduced. "
                "/ 某个来源 session 的种子清单无法重建。"
            )
    if target != reference.target_normalized_price_deviation:
        raise ValueError(
            "The requested target differs from the Step-35D receipts. "
            "/ 请求的目标与第 35D 步 receipt 不一致。"
        )

    session_seeds = tuple(receipt.source_seed_manifest.session_seed for receipt in receipts)
    base_stream_seeds = tuple(
        seed
        for receipt in receipts
        for seed in receipt.source_seed_manifest.child_seeds()
    )
    checkpoint_digests = tuple(receipt.checkpoint_sha256 for receipt in receipts)
    step35d_digests = tuple(receipt.receipt_payload_sha256 for receipt in receipts)
    baseline_digests = tuple(receipt.baseline_receipt_payload_sha256 for receipt in receipts)
    identity_collections = (
        ("session indexes", indexes),
        ("session seeds", session_seeds),
        ("base stream seeds", base_stream_seeds),
        ("checkpoint digests", checkpoint_digests),
        ("Step-35D receipt digests", step35d_digests),
        ("baseline receipt digests", baseline_digests),
    )
    for label, values in identity_collections:
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {label} detected. / 检测到重复的 {label}。")

    long_run_price, total_measurement_rows = _weighted_mean(
        tuple(
            (
                receipt.long_run_mean_oriented_price,
                receipt.long_run_baseline_receipt.measurement_periods_scored,
            )
            for receipt in receipts
        ),
        "long-run oriented price",
    )
    mean_t3_price, total_t3_paths = _weighted_mean(
        tuple(
            (receipt.mean_unshocked_t3_oriented_price, receipt.paths_executed)
            for receipt in receipts
        ),
        "unshocked t3 oriented price",
    )
    mean_t3_lambda, lambda_path_count = _weighted_mean(
        tuple(
            (receipt.mean_t3_price_impact_lambda, receipt.paths_executed)
            for receipt in receipts
        ),
        "actual t3 lambda",
    )
    if lambda_path_count != total_t3_paths:
        raise RuntimeError("The two t=3 pooled samples differ. / 两个 t=3 汇总样本不一致。")
    minimum_lambda = min(
        receipt.minimum_t3_price_impact_lambda for receipt in receipts
    )
    nonpositive_count = sum(
        receipt.nonpositive_t3_lambda_count for receipt in receipts
    )
    agent_count = reference_baseline.number_of_agents
    pooled_orders = tuple(
        _weighted_mean(
            tuple(
                (
                    receipt.long_run_baseline_receipt.mean_oriented_order_by_agent[agent],
                    receipt.long_run_baseline_receipt.measurement_periods_scored,
                )
                for receipt in receipts
            ),
            f"agent {agent + 1} long-run oriented order",
        )[0]
        for agent in range(agent_count)
    )
    pooled_profits = tuple(
        _weighted_mean(
            tuple(
                (
                    receipt.long_run_baseline_receipt.mean_profit_by_agent[agent],
                    receipt.long_run_baseline_receipt.measurement_periods_scored,
                )
                for receipt in receipts
            ),
            f"agent {agent + 1} long-run profit",
        )[0]
        for agent in range(agent_count)
    )
    arithmetic = calculate_cell_shock_calibration_arithmetic(
        long_run_price,
        mean_t3_price,
        mean_t3_lambda,
        minimum_lambda,
        target,
    )

    paper_sessions = expected == PAPER_SESSIONS_PER_EXPERIMENT_CELL
    paper_paths = all(
        receipt.paper_paths_per_session_count_matched_for_calibration
        for receipt in receipts
    )
    paper_sources = all(
        receipt.ready_for_formal_paper_cell_aggregation for receipt in receipts
    )
    positive_lambdas = nonpositive_count == 0 and minimum_lambda > 0.0
    distinct_sessions = True  # Duplicates were rejected above. / 重复项已在上方拒绝。
    base_seed_unique = len(set(base_stream_seeds)) == len(base_stream_seeds)
    path_namespace_inputs_unique = len(set(session_seeds)) == len(session_seeds)
    formal_namespace_audit = paper_sessions and distinct_sessions and base_seed_unique
    formal_ready = (
        paper_sessions
        and paper_paths
        and paper_sources
        and positive_lambdas
        and formal_namespace_audit
        and path_namespace_inputs_unique
    )
    exact_target = isclose(
        arithmetic.exact_level_achieved_normalized_level_deviation,
        target,
        rel_tol=0.0,
        abs_tol=8.0 * sys.float_info.epsilon,
    )

    receipt = ExperimentCellShockCalibrationReceipt(
        protocol_version=CELL_SHOCK_CALIBRATION_VERSION,
        primary_rule=EXACT_LEVEL_RULE,
        pooling_rule=RAW_OBSERVATION_WEIGHTING,
        experiment_seed=reference_manifest.experiment_seed,
        experiment_cell_key=reference_manifest.experiment_cell_key,
        experiment_cell_seed=reference_manifest.experiment_cell_seed,
        irf_experiment_seed=reference.irf_experiment_seed,
        session_seed_derivation_version=reference_manifest.seed_derivation_version,
        path_seed_derivation_version=reference.path_seed_derivation_version,
        implementation_tree_sha256=reference.implementation_tree_sha256,
        parameter_snapshot=reference_baseline.parameter_snapshot,
        value_grid_snapshot=reference_baseline.value_grid_snapshot,
        sessions_expected=expected,
        sessions_received=len(receipts),
        ordered_session_indexes=indexes,
        total_long_run_measurement_rows=total_measurement_rows,
        total_unshocked_t3_paths=total_t3_paths,
        number_of_agents=agent_count,
        pooled_long_run_mean_oriented_price=long_run_price,
        pooled_long_run_mean_oriented_order_by_agent=pooled_orders,
        pooled_long_run_mean_profit_by_agent=pooled_profits,
        pooled_unshocked_t3_mean_oriented_price=mean_t3_price,
        pooled_actual_t3_mean_price_impact_lambda=mean_t3_lambda,
        minimum_actual_t3_price_impact_lambda=minimum_lambda,
        nonpositive_actual_t3_lambda_count=nonpositive_count,
        arithmetic=arithmetic,
        selected_absolute_noise_shock=arithmetic.exact_level_absolute_noise_shock,
        increment_shortcut_sensitivity_shock=(
            arithmetic.increment_shortcut_absolute_noise_shock
        ),
        ordered_source_receipts_sha256=_ordered_receipts_digest(receipts),
        unique_session_index_count=len(set(indexes)),
        unique_session_seed_count=len(set(session_seeds)),
        unique_base_stream_seed_count=len(set(base_stream_seeds)),
        planned_base_stream_seed_count=len(base_stream_seeds),
        unique_checkpoint_digest_count=len(set(checkpoint_digests)),
        unique_step35d_receipt_digest_count=len(set(step35d_digests)),
        unique_baseline_receipt_digest_count=len(set(baseline_digests)),
        same_experiment_cell_verified=True,
        all_source_receipts_validated=True,
        canonical_session_index_coverage_verified=True,
        distinct_session_identities_verified=distinct_sessions,
        cross_session_base_stream_seed_uniqueness_verified=base_seed_unique,
        cross_session_path_seed_namespace_inputs_unique=path_namespace_inputs_unique,
        all_twenty_million_path_child_seed_outputs_explicitly_compared=False,
        raw_observation_weighting_used=True,
        all_actual_t3_lambdas_positive=positive_lambdas,
        one_common_magnitude_selected_for_entire_cell=True,
        uniform_cell_shock_calibrated=True,
        paper_1000_sessions_verified=paper_sessions,
        paper_10000_paths_per_session_verified=paper_paths,
        paper_scale_source_receipts_verified=paper_sources,
        formal_cross_session_seed_namespace_audit_verified=formal_namespace_audit,
        ready_for_formal_step35f=formal_ready,
        exact_level_target_satisfied_in_calibration_arithmetic=exact_target,
        paper_target_observed_on_shocked_paths=False,
        shock_applied=False,
        treatment_paths_executed=0,
        t4_response_aggregated=False,
        classification_ready=False,
        mechanism_label_ready=False,
        paper_figure_ready=False,
        paper_specifies_numerical_calibration_algorithm=False,
        exact_level_rule_is_replication_interpretation=True,
        increment_shortcut_retained_as_sensitivity=True,
        checksum_detects_stale_replacement_not_authentication=True,
        standalone_receipt_authenticates_discarded_source_rows=False,
        receipt_payload_sha256="",
    )
    receipt = replace(
        receipt,
        receipt_payload_sha256=_cell_receipt_payload_digest(receipt),
    )
    validate_experiment_cell_shock_calibration_receipt(receipt)
    return receipt


def main() -> None:
    """Run one short one-session wiring demo, not a paper experiment.

    运行一个短小的单 session 接线演示；这不是论文实验。
    """

    from steps.step_35c_irf_long_run_baseline import _build_demo_controller

    controller, scorer = _build_demo_controller()
    while controller.phase is SessionPhase.TRAINING:
        if controller.training_periods_completed >= 5:
            raise TimeoutError("Debug convergence was not reached. / 调试收敛尚未达到。")
        controller.run_next_period()
    checkpoint = scorer.capture_and_bind_convergence_checkpoint(controller)
    controller.run_until_complete()
    scorer.finalize(controller)
    session_receipt = run_unshocked_t3_calibration_paths(
        checkpoint,
        baseline_scorer=scorer,
        irf_experiment_seed=20_260_835,
        path_count=100,
    )
    cell_receipt = calibrate_experiment_cell_uniform_shock(
        (session_receipt,),
        expected_session_count=1,
    )
    arithmetic = cell_receipt.arithmetic
    print("Step 35E: cell shock calibration / 第 35E 步：实验单元冲击校准")
    print(f"Debug sessions / 调试 session 数: {cell_receipt.sessions_received}")
    print(f"E[p_tilde] / 长期价格均值: {arithmetic.mean_long_run_oriented_price:.9f}")
    print(f"E[p_tilde_3^0] / 无冲击 t=3 价格均值: {arithmetic.mean_unshocked_t3_oriented_price:.9f}")
    print(f"E[actual lambda_3] / 真正 t=3 lambda 均值: {arithmetic.mean_actual_t3_price_impact_lambda:.12f}")
    print(f"Exact-level common shock / exact-level 统一冲击: {arithmetic.exact_level_absolute_noise_shock:.9f}")
    print(f"Increment shortcut sensitivity / 增量 shortcut 敏感性: {arithmetic.increment_shortcut_absolute_noise_shock:.9f}")
    print(f"Exact achieved level deviation / exact 实现的价格水平偏差: {arithmetic.exact_level_achieved_normalized_level_deviation:.9%}")
    print(f"Shortcut achieved level deviation / shortcut 实现的价格水平偏差: {arithmetic.increment_shortcut_achieved_normalized_level_deviation:.9%}")
    print("Shock applied / 已施加冲击: False")
    print("t=4 executed / 已执行 t=4: False")
    print("Paper 1,000-session scale / 论文一千个 session 规模: False (debug run / 调试运行)")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
