"""Step 35B: run one auditable paired control-treatment IRF path.

步骤 35B：运行一条可审计的配对“对照组—实验组”脉冲响应路径。

Run / 运行:
    py -3 -X utf8 steps/step_35b_paired_irf_path.py

Paper timing / 论文时间轴:
    - the completed local-t=0 price/value outcome is carried by the convergence
      checkpoint; Step 35B does not execute it a second time;
      / 已完成的局部 t=0 价格与价值结果保存在收敛 checkpoint 中；第 35B 步
      不会把这一期重新执行一次；
    - transactions run at local t=1,2,3,4;
      / 真正交易发生在局部 t=1、2、3、4；
    - the adverse noise shock enters treatment only at local t=3;
      / 逆向噪声冲击只在局部 t=3 进入实验组；
    - local t=4 exposes the endogenous response.
      / 局部 t=4 显示内生反应。

Common-random-number replication choice / 共同随机数复现选择:
    For one path index, one external driver draws each ordinary u_t and each
    next-value index exactly once, then supplies that same pair to control and
    treatment. Treatment adds one deterministic signed shock to ordinary u_3.
    The paper does not specify a control branch, common random numbers, or seed
    derivation, so all three are explicitly recorded as replication choices.
    / 对一个路径编号，外部 driver 每期只抽一次普通 u_t 和下一价值编号，再把同一对
    数提供给对照组与实验组；实验组只在普通 u_3 上增加一次确定的带符号冲击。论文
    没有规定对照分支、共同随机数或路径种子算法，因此三者都明确记为复现选择。

Scope boundary / 本步边界:
    This is a correctness reference for ONE paired path, not the paper's
    10,000-path/session runner and not a mechanism classification. The present
    full checkpoint restore is intentionally safe but too expensive to repeat
    at formal scale. / 本步只验证“一条配对路径”正确，不声称已完成每个 session
    10,000 条路径，也不进行机制分类。当前完整 checkpoint 恢复重视安全，但不能
    直接按正式规模重复使用。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import isclose, isfinite
from pathlib import Path
import random
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from step_14_state_representation import build_state_indexes
from step_22_market_maker_rolling_history import MarketObservation
from step_25_one_market_period import build_paper_inputs
from step_26_reproducible_random_streams import (
    FrozenPolicyPeriodObservation,
    RandomizedMarketSession,
    build_randomized_paper_session,
)
# Steps 25-28 still form one legacy bare-import cluster. Keep that internal
# cluster together, while the public Step34/35A/35B boundary below uses one
# canonical package identity. / 第 25-28 步仍是一个旧式裸导入簇；内部暂时保持
# 一致，而公共的第 34/35A/35B 边界统一使用 package 身份。
from step_28_session_phases import SessionPhase, SessionPhaseController
from steps.step_34_mechanism_classifier import (
    PAPER_PATHS_PER_SESSION,
    PAPER_RESPONSE_PERIOD,
    PAPER_SHOCK_PERIOD,
    PAPER_TARGET_PRICE_DEVIATION,
    AppliedNoiseShock,
    UniformShockCalibration,
    add_adverse_shock_to_noise,
    calibrate_uniform_noise_shock,
    orient_order,
    orient_price,
    validate_uniform_shock_calibration_receipt,
)
from steps.step_35a_converged_market_checkpoint import (
    ConvergedMarketCheckpoint,
    capture_at_convergence_boundary,
    restore_two_independent_branches,
    verify_converged_market_checkpoint,
)


PAIRED_IRF_PROTOCOL_VERSION = "step35b-one-paired-path-v1"
PAIRED_PATH_SEED_VERSION = "sha256-256bit-paired-irf-path-v2"
PAIRED_PATH_SEED_DOMAIN = b"vibe-replication.step35b.paired-path.v2\0"
IRF_LOCAL_STATE_ORIGIN = 0
IRF_FIRST_TRANSACTION_PERIOD = 1
IRF_LAST_TRANSACTION_PERIOD = PAPER_RESPONSE_PERIOD
IRF_LOCAL_PERIODS = tuple(
    range(IRF_FIRST_TRANSACTION_PERIOD, IRF_LAST_TRANSACTION_PERIOD + 1)
)
IRF_PERIOD_COUNT = len(IRF_LOCAL_PERIODS)
MAX_UINT64_PLUS_ONE = 2**64
CONTROL_FIRST = "control_then_treatment"
TREATMENT_FIRST = "treatment_then_control"
BRANCH_EXECUTION_ORDERS = (CONTROL_FIRST, TREATMENT_FIRST)


def _uint64(number: int, label: str) -> int:
    """Validate one non-Boolean unsigned 64-bit integer. / 检查 64 位无符号整数。"""

    if isinstance(number, bool) or not isinstance(number, int):
        raise TypeError(f"{label} must be an integer. / {label} 必须是整数。")
    if not 0 <= number < MAX_UINT64_PLUS_ONE:
        raise ValueError(f"{label} must lie in [0, 2^64). / {label} 必须位于 [0, 2^64)。")
    return number


def _path_index(path_index: int) -> int:
    """Validate one of the paper's planned 10,000 path identities.

    检查论文计划中的 10,000 个路径编号之一。
    """

    checked = _uint64(path_index, "path_index / 路径编号")
    if checked >= PAPER_PATHS_PER_SESSION:
        raise ValueError(
            "path_index must lie in [0, 10000). / path_index 必须位于 [0, 10000)。"
        )
    return checked


def _validated_calibration(
    calibration: UniformShockCalibration,
) -> UniformShockCalibration:
    """Validate the Step-34 receipt before any expensive branch restore.

    在昂贵的分支恢复之前，检查第 34 步的校准凭证。
    """

    calibration = validate_uniform_shock_calibration_receipt(calibration)
    if (
        calibration.paper_shock_period != PAPER_SHOCK_PERIOD
        or calibration.paper_response_period != PAPER_RESPONSE_PERIOD
        or calibration.paper_required_paths_per_session
        != PAPER_PATHS_PER_SESSION
        or not calibration.protocol_adds_shock_to_ordinary_noise
    ):
        raise ValueError(
            "Shock-calibration protocol is incompatible with Step 35B. / "
            "冲击校准协议与第 35B 步不兼容。"
        )
    return calibration


@dataclass(frozen=True)
class PairedPathSeedManifest:
    """Two path-specific external stream seeds. / 两条路径专用外部随机流种子。"""

    derivation_version: str
    irf_experiment_seed: int
    checkpoint_sha256: str
    source_session_index: int
    source_session_seed: int
    path_index: int
    ordinary_noise_seed: int
    next_value_seed: int
    rng_engine: str
    child_seed_bits: int
    checkpoint_digest_is_provenance_not_seed_entropy: bool
    formal_cross_session_seed_uniqueness_verified: bool
    path_seed_derivation_is_replication_choice: bool
    paper_specifies_path_seed_derivation: bool


@dataclass(frozen=True)
class VerifiedPairedPathScheduleContext:
    """Checkpoint facts verified once before a many-path run.

    在多路径运行开始前只核对一次的 checkpoint 信息。

    Rechecking a 10,000-row market-maker checkpoint for every one of 10,000
    paths would repeat the same expensive work.  Step 35D therefore creates
    this small immutable context once, keeps it private to one runner, and then
    derives exactly the same Step-35B schedules from it. / 如果每条路径都重新
    核对含 10,000 行历史的 checkpoint，就会重复同一项昂贵工作。第 35D 步
    因此只建立一次这个小型不可变 context，并继续生成与第 35B 步完全相同的
    路径抽样。
    """

    derivation_version: str
    irf_experiment_seed: int
    checkpoint_sha256: str
    source_session_index: int
    source_session_seed: int
    noise_standard_deviation: float
    number_of_value_points: int


def prepare_verified_paired_path_schedule_context(
    checkpoint: ConvergedMarketCheckpoint,
    *,
    irf_experiment_seed: int,
) -> VerifiedPairedPathScheduleContext:
    """Verify immutable inputs once for a reusable schedule factory.

    为可重复使用的路径抽样器核对一次不可变输入。
    """

    checked_seed = _uint64(
        irf_experiment_seed,
        "irf_experiment_seed / IRF 实验种子",
    )
    verify_converged_market_checkpoint(checkpoint)
    checkpoint_digest = checkpoint.checkpoint_sha256
    try:
        checkpoint_bytes = bytes.fromhex(checkpoint_digest)
    except ValueError as error:
        raise ValueError(
            "Checkpoint digest is not hexadecimal. / checkpoint 哈希不是十六进制。"
        ) from error
    if len(checkpoint_bytes) != 32:
        raise ValueError(
            "Checkpoint digest must be SHA-256. / checkpoint 哈希必须是 SHA-256。"
        )
    source_session_seed = _uint64(
        checkpoint.payload.seed_manifest.session_seed,
        "source_session_seed / 来源 session 种子",
    )
    noise_standard_deviation = float(checkpoint.payload.parameters.noise_std)
    number_of_value_points = len(checkpoint.payload.value_grid)
    if not isfinite(noise_standard_deviation) or noise_standard_deviation < 0.0:
        raise ValueError("noise_std must be finite and nonnegative. / noise_std 必须有限且非负。")
    if number_of_value_points < 1:
        raise ValueError("The value grid must not be empty. / 价值网格不能为空。")
    return VerifiedPairedPathScheduleContext(
        derivation_version=PAIRED_PATH_SEED_VERSION,
        irf_experiment_seed=checked_seed,
        checkpoint_sha256=checkpoint_digest,
        source_session_index=checkpoint.payload.seed_manifest.session_index,
        source_session_seed=source_session_seed,
        noise_standard_deviation=noise_standard_deviation,
        number_of_value_points=number_of_value_points,
    )


def _derive_paired_child_seed(
    irf_experiment_seed: int,
    source_session_seed: int,
    path_index: int,
    stream_label: bytes,
) -> int:
    """Pure named SHA-256 derivation after outer inputs are validated.

    外层输入检查完成后，使用纯命名 SHA-256 派生子种子。
    """

    payload = (
        PAIRED_PATH_SEED_DOMAIN
        + irf_experiment_seed.to_bytes(8, "big")
        + source_session_seed.to_bytes(8, "big")
        + path_index.to_bytes(8, "big")
        + b"\0"
        + stream_label
    )
    # Python's random.Random accepts arbitrarily large integer seeds. Keeping
    # all 256 SHA bits avoids an unnecessary 64-bit birthday-collision risk.
    # / Python 的 random.Random 可接受任意大整数种子；保留 SHA 的全部 256 位，
    # 避免不必要的 64 位生日碰撞风险。
    return int.from_bytes(sha256(payload).digest(), "big")


def derive_paired_path_seed_manifest_from_verified_context(
    context: VerifiedPairedPathScheduleContext,
    *,
    path_index: int,
) -> PairedPathSeedManifest:
    """Derive one manifest after the outer checkpoint was verified once.

    在外层 checkpoint 已核对一次后，派生一条路径的种子说明。
    """

    if not isinstance(context, VerifiedPairedPathScheduleContext):
        raise TypeError("context has the wrong type. / context 类型错误。")
    if context.derivation_version != PAIRED_PATH_SEED_VERSION:
        raise ValueError("Schedule-context version is unsupported. / 抽样 context 版本不支持。")
    checked_path_index = _path_index(path_index)
    noise_seed = _derive_paired_child_seed(
        context.irf_experiment_seed,
        context.source_session_seed,
        checked_path_index,
        b"ordinary_noise",
    )
    value_seed = _derive_paired_child_seed(
        context.irf_experiment_seed,
        context.source_session_seed,
        checked_path_index,
        b"next_value",
    )
    if noise_seed == value_seed:
        raise RuntimeError("Paired path child seeds collided. / 配对路径子种子发生碰撞。")
    return PairedPathSeedManifest(
        derivation_version=context.derivation_version,
        irf_experiment_seed=context.irf_experiment_seed,
        checkpoint_sha256=context.checkpoint_sha256,
        source_session_index=context.source_session_index,
        source_session_seed=context.source_session_seed,
        path_index=checked_path_index,
        ordinary_noise_seed=noise_seed,
        next_value_seed=value_seed,
        rng_engine="Python random.Random (MT19937)",
        child_seed_bits=256,
        checkpoint_digest_is_provenance_not_seed_entropy=True,
        formal_cross_session_seed_uniqueness_verified=False,
        path_seed_derivation_is_replication_choice=True,
        paper_specifies_path_seed_derivation=False,
    )


def derive_paired_path_seed_manifest(
    checkpoint: ConvergedMarketCheckpoint,
    *,
    irf_experiment_seed: int,
    path_index: int,
) -> PairedPathSeedManifest:
    """Derive two stable seeds without consuming any live RNG.

    不消耗任何实时随机流，稳定地产生两条路径种子。
    """

    context = prepare_verified_paired_path_schedule_context(
        checkpoint,
        irf_experiment_seed=irf_experiment_seed,
    )
    return derive_paired_path_seed_manifest_from_verified_context(
        context,
        path_index=path_index,
    )


@dataclass(frozen=True)
class PlannedPairedPathSeedAudit:
    """Uniqueness audit for identities only, not executed paths.

    只核对路径种子身份是否唯一；不代表已经执行这些路径。
    """

    derivation_version: str
    irf_experiment_seed: int
    checkpoint_sha256: str
    planned_path_count: int
    planned_child_stream_count: int
    unique_child_seed_count: int
    all_planned_child_seeds_are_unique: bool
    child_seed_bits: int
    formal_cross_session_uniqueness_verified: bool
    paths_were_executed: bool


def audit_all_planned_paired_path_seed_uniqueness(
    checkpoint: ConvergedMarketCheckpoint,
    *,
    irf_experiment_seed: int,
) -> PlannedPairedPathSeedAudit:
    """Prove 10,000 path labels give 20,000 distinct child seeds.

    证明 10,000 个路径标签产生 20,000 个互不重复的子种子。
    """

    checked_seed = _uint64(
        irf_experiment_seed,
        "irf_experiment_seed / IRF 实验种子",
    )
    verify_converged_market_checkpoint(checkpoint)
    digest = checkpoint.checkpoint_sha256
    source_session_seed = _uint64(
        checkpoint.payload.seed_manifest.session_seed,
        "source_session_seed / 来源 session 种子",
    )
    child_seeds: set[int] = set()
    for planned_path_index in range(PAPER_PATHS_PER_SESSION):
        child_seeds.add(
            _derive_paired_child_seed(
                checked_seed,
                source_session_seed,
                planned_path_index,
                b"ordinary_noise",
            )
        )
        child_seeds.add(
            _derive_paired_child_seed(
                checked_seed,
                source_session_seed,
                planned_path_index,
                b"next_value",
            )
        )
    planned_children = 2 * PAPER_PATHS_PER_SESSION
    unique_children = len(child_seeds)
    if unique_children != planned_children:
        raise RuntimeError("Planned IRF child seeds collided. / 计划 IRF 子种子发生碰撞。")
    return PlannedPairedPathSeedAudit(
        derivation_version=PAIRED_PATH_SEED_VERSION,
        irf_experiment_seed=checked_seed,
        checkpoint_sha256=digest,
        planned_path_count=PAPER_PATHS_PER_SESSION,
        planned_child_stream_count=planned_children,
        unique_child_seed_count=unique_children,
        all_planned_child_seeds_are_unique=True,
        child_seed_bits=256,
        formal_cross_session_uniqueness_verified=False,
        paths_were_executed=False,
    )


@dataclass(frozen=True)
class PairedPathDrawSchedule:
    """Four shared ordinary draws for one t=1,...,4 path.

    一条 t=1,...,4 路径共用的四组普通抽样。
    """

    seed_manifest: PairedPathSeedManifest
    local_periods: tuple[int, ...]
    ordinary_noise_orders_u: tuple[float, ...]
    next_value_indexes: tuple[int, ...]
    noise_standard_deviation: float
    number_of_value_points: int
    draws_per_stream: int
    ordinary_noise_final_rng_state: object
    next_value_final_rng_state: object
    control_and_treatment_share_each_draw: bool
    common_random_numbers_are_replication_choice: bool
    paper_specifies_common_random_numbers: bool


def build_paired_path_draw_schedule(
    checkpoint: ConvergedMarketCheckpoint,
    *,
    irf_experiment_seed: int,
    path_index: int,
) -> PairedPathDrawSchedule:
    """Draw one ordinary u and one next value per local transaction.

    每个局部交易时期只抽一次普通噪声和一次下一价值。
    """

    context = prepare_verified_paired_path_schedule_context(
        checkpoint,
        irf_experiment_seed=irf_experiment_seed,
    )
    return build_paired_path_draw_schedule_from_verified_context(
        context,
        path_index=path_index,
    )


def build_paired_path_draw_schedule_from_verified_context(
    context: VerifiedPairedPathScheduleContext,
    *,
    path_index: int,
) -> PairedPathDrawSchedule:
    """Build the exact Step-35B schedule without rechecking the checkpoint.

    不重复核对 checkpoint，生成与第 35B 步完全相同的抽样日程。

    Callers must first create ``context`` with
    :func:`prepare_verified_paired_path_schedule_context`. / 调用者必须先用
    ``prepare_verified_paired_path_schedule_context`` 建立 context。
    """

    manifest = derive_paired_path_seed_manifest_from_verified_context(
        context,
        path_index=path_index,
    )
    noise_standard_deviation = context.noise_standard_deviation
    number_of_values = context.number_of_value_points
    noise_generator = random.Random(manifest.ordinary_noise_seed)
    value_generator = random.Random(manifest.next_value_seed)
    ordinary_noise = tuple(
        noise_generator.gauss(0.0, noise_standard_deviation)
        for _ in IRF_LOCAL_PERIODS
    )
    next_values = tuple(
        value_generator.randrange(number_of_values)
        for _ in IRF_LOCAL_PERIODS
    )
    if not all(isfinite(number) for number in ordinary_noise):
        raise RuntimeError("A scheduled noise draw is not finite. / 路径噪声抽样不是有限数。")
    return PairedPathDrawSchedule(
        seed_manifest=manifest,
        local_periods=IRF_LOCAL_PERIODS,
        ordinary_noise_orders_u=ordinary_noise,
        next_value_indexes=next_values,
        noise_standard_deviation=noise_standard_deviation,
        number_of_value_points=number_of_values,
        draws_per_stream=IRF_PERIOD_COUNT,
        ordinary_noise_final_rng_state=noise_generator.getstate(),
        next_value_final_rng_state=value_generator.getstate(),
        control_and_treatment_share_each_draw=True,
        common_random_numbers_are_replication_choice=True,
        paper_specifies_common_random_numbers=False,
    )


@dataclass(frozen=True)
class PairedIRFPeriod:
    """One named control-treatment comparison at one local period.

    一个局部时期中带名称的对照组—实验组比较。
    """

    local_period: int
    expected_global_period: int
    common_current_fundamental_value: float
    ordinary_noise_order_u: float
    common_next_value_index: int
    control_noise_used_for_pricing: float
    treatment_noise_used_for_pricing: float
    signed_treatment_shock_u: float
    shock_applied: bool
    applied_noise_shock: AppliedNoiseShock | None
    control_observation: FrozenPolicyPeriodObservation
    treatment_observation: FrozenPolicyPeriodObservation
    control_oriented_price: float
    treatment_oriented_price: float
    paired_oriented_price_increment: float
    control_oriented_orders: tuple[float, float]
    treatment_oriented_orders: tuple[float, float]
    common_draws_verified: bool
    completed_rows_match_named_branches: bool
    prior_ols_matches_named_branch_history: bool


@dataclass(frozen=True)
class PairedIRFProtocolDisclosure:
    """Separate paper facts from software choices. / 区分原文事实与软件选择。"""

    paper_states_all_sessions_converged_at_local_t0: bool = True
    paper_specifies_t0_transaction_or_fork_semantics: bool = False
    paper_shock_occurs_at_local_t3: bool = True
    paper_response_is_measured_at_local_t4: bool = True
    paper_target_normalized_price_deviation: float = PAPER_TARGET_PRICE_DEVIATION
    paper_requires_paths_per_session: int = PAPER_PATHS_PER_SESSION
    paper_specifies_unshocked_control_branch: bool = False
    paper_specifies_common_random_numbers: bool = False
    paper_specifies_add_versus_replace_noise: bool = False
    paper_specifies_irf_start_state_carry_rule: bool = False
    paper_specifies_irf_path_seed_derivation: bool = False
    paper_specifies_rolling_ols_update_rule_inside_irf: bool = False
    paper_specifies_q_update_or_freeze_rule_inside_irf: bool = False
    replication_uses_unshocked_control_branch: bool = True
    replication_uses_common_random_numbers: bool = True
    replication_adds_shock_to_ordinary_noise: bool = True
    replication_carries_completed_t0_outcome: bool = True
    replication_executes_no_new_t0_transaction: bool = True
    replication_carries_checkpoint_current_state: bool = True
    replication_carries_common_checkpoint_v1_across_paths: bool = True
    replication_keeps_frozen_learned_policy: bool = True
    replication_keeps_rolling_ols_adaptive: bool = True
    shocked_t3_row_enters_treatment_maker_history_before_t4: bool = True
    control_branch_is_not_paper_long_run_baseline: bool = True
    long_run_order_baseline_computed: bool = False
    t0_and_initial_v1_sensitivity_resolved: bool = False
    single_paired_path_correctness_reference_only: bool = True
    scalable_formal_runner_verified: bool = False


@dataclass(frozen=True)
class PairedIRFFinalBranchSummary:
    """Small final state summary; do not duplicate the 10,000 maker rows.

    小型最终状态总结；不重复保存做市商的一万行历史。
    """

    final_period_number: int
    final_previous_price: float
    final_previous_value: float
    final_current_value: float
    market_maker_successful_append_count: int
    newest_market_maker_row: MarketObservation
    frozen_draw_source_mode: str | None


@dataclass(frozen=True)
class PairedIRFPathReceipt:
    """One immutable audit receipt, explicitly not a 10,000-path result.

    一份不可修改的审计凭证；明确不是 10,000 条路径结果。
    """

    protocol_version: str
    checkpoint_sha256: str
    source_session_index: int
    checkpoint_global_origin_period: int
    irf_local_state_origin: int
    carried_local_t0_global_period: int
    carried_local_t0_price: float
    carried_local_t0_value: float
    local_t1_current_value: float
    t0_outcome_carried_not_reexecuted: bool
    local_transaction_periods: tuple[int, ...]
    shock_local_period: int
    response_local_period: int
    seed_manifest: PairedPathSeedManifest
    draw_schedule: PairedPathDrawSchedule
    shock_calibration: UniformShockCalibration
    branch_execution_order: str
    periods: tuple[PairedIRFPeriod, ...]
    control_final_state: PairedIRFFinalBranchSummary
    treatment_final_state: PairedIRFFinalBranchSummary
    paired_t3_treatment_minus_control_oriented_price_increment: float
    expected_paired_t3_increment_from_lambda: float
    t3_price_identity_error: float
    control_t4_oriented_orders: tuple[float, float]
    treatment_t4_oriented_orders: tuple[float, float]
    paths_economically_represented: int
    execution_replays_for_order_audit: int
    full_paper_path_count_verified: bool
    classification_ready: bool
    pre_shock_exact_parity_verified: bool
    shock_applied_exactly_once_verified: bool
    common_draws_verified: bool
    branch_internal_rngs_unchanged_verified: bool
    frozen_learning_state_verified: bool
    independent_market_maker_histories_verified: bool
    each_branch_t4_ols_uses_own_t3_history_verified: bool
    branch_order_invariance_verified: bool
    calibration_arithmetic_verified: bool
    paper_1_2_percent_target_used: bool
    calibration_aggregate_provenance_verified: bool
    protocol_disclosure: PairedIRFProtocolDisclosure


def _completed_row(observation: FrozenPolicyPeriodObservation) -> MarketObservation:
    """Reconstruct the exact maker row from one public observation.

    从公开观测重建做市商应保存的精确历史行。
    """

    return MarketObservation(
        fundamental_value_v=observation.fundamental_value_v,
        market_price_p=observation.continuous_price_p,
        insensitive_order_z=observation.insensitive_order_z,
        informed_and_noise_order_y=observation.total_order_flow_y,
    )


def _final_summary(branch: RandomizedMarketSession) -> PairedIRFFinalBranchSummary:
    """Return only the final facts needed for audits. / 只返回审计所需最终事实。"""

    return PairedIRFFinalBranchSummary(
        final_period_number=branch.period_number,
        final_previous_price=branch.previous_price,
        final_previous_value=branch.previous_value,
        final_current_value=branch.current_value,
        market_maker_successful_append_count=(
            branch.market_maker.successful_append_count
        ),
        newest_market_maker_row=branch.market_maker.snapshot()[-1],
        frozen_draw_source_mode=branch.frozen_draw_source_mode,
    )


def _validate_execution_order(branch_execution_order: str) -> str:
    """Reject misspelled or unsupported branch order. / 拒绝错误的分支运行顺序。"""

    if branch_execution_order not in BRANCH_EXECUTION_ORDERS:
        raise ValueError(
            f"branch_execution_order must be one of {BRANCH_EXECUTION_ORDERS}. / "
            "branch_execution_order 不受支持。"
        )
    return branch_execution_order


def run_one_paired_irf_path(
    checkpoint: ConvergedMarketCheckpoint,
    *,
    irf_experiment_seed: int,
    path_index: int,
    shock_calibration: UniformShockCalibration,
    branch_execution_order: str = CONTROL_FIRST,
) -> PairedIRFPathReceipt:
    """Run one disposable control-treatment pair through local t=4.

    运行一组一次性的对照—实验分支，直到局部 t=4。

    If either disposable branch fails unexpectedly, this function raises and
    returns neither branch. The safe recovery is to restore the checkpoint
    again, not to guess a partial rollback. / 若任一一次性分支意外失败，本函数
    直接报错且不返回任何分支；安全恢复方式是重新恢复 checkpoint，而不是猜测如何
    回滚半条路径。
    """

    checked_seed = _uint64(
        irf_experiment_seed,
        "irf_experiment_seed / IRF 实验种子",
    )
    checked_path = _path_index(path_index)
    calibration = _validated_calibration(shock_calibration)
    execution_order = _validate_execution_order(branch_execution_order)
    verify_converged_market_checkpoint(checkpoint)
    schedule = build_paired_path_draw_schedule(
        checkpoint,
        irf_experiment_seed=checked_seed,
        path_index=checked_path,
    )
    control, treatment = restore_two_independent_branches(checkpoint)
    if control.market_maker is treatment.market_maker:
        raise RuntimeError("Control and treatment share a market maker. / 对照组与实验组共享做市商。")

    control_rng_start = control.all_random_states()
    treatment_rng_start = treatment.all_random_states()
    if control_rng_start != treatment_rng_start:
        raise RuntimeError("Paired branches do not begin with equal RNG states. / 配对分支初始随机状态不同。")
    control_q_start = tuple(trader.q_table.tobytes() for trader in control.traders)
    treatment_q_start = tuple(trader.q_table.tobytes() for trader in treatment.traders)
    control_policy_start = control.frozen_policy_action_indexes_snapshot()
    treatment_policy_start = treatment.frozen_policy_action_indexes_snapshot()
    control_counts_start = tuple(control.shared_value_visit_counts)
    treatment_counts_start = tuple(treatment.shared_value_visit_counts)
    control_appends_start = control.market_maker.successful_append_count
    treatment_appends_start = treatment.market_maker.successful_append_count

    period_records: list[PairedIRFPeriod] = []
    shock_application_count = 0
    pre_shock_parity = True
    common_draws = True
    t3_actual_increment: float | None = None
    t3_expected_increment: float | None = None
    t3_identity_error: float | None = None
    expected_t1_state_indexes = build_state_indexes(
        checkpoint.payload.previous_price,
        checkpoint.payload.previous_value,
        checkpoint.payload.current_value,
        checkpoint.payload.price_grid,
        checkpoint.payload.value_grid,
    )

    for schedule_index, local_period in enumerate(IRF_LOCAL_PERIODS):
        ordinary_noise = schedule.ordinary_noise_orders_u[schedule_index]
        next_value_index = schedule.next_value_indexes[schedule_index]
        expected_global_period = (
            checkpoint.payload.origin_global_period + local_period - 1
        )
        if control.current_value != treatment.current_value:
            raise RuntimeError("Paired current values diverged. / 配对分支当前价值发生分化。")
        current_value = control.current_value
        control_prior_ols = control.market_maker.estimates()
        treatment_prior_ols = treatment.market_maker.estimates()
        applied_shock: AppliedNoiseShock | None = None
        treatment_noise = ordinary_noise
        if local_period == PAPER_SHOCK_PERIOD:
            applied_shock = add_adverse_shock_to_noise(
                ordinary_noise,
                current_value,
                checkpoint.payload.parameters.value_mean,
                calibration.absolute_noise_shock,
            )
            treatment_noise = applied_shock.noise_order_used_for_pricing
            shock_application_count += 1

        # Both effective inputs are prepared before either disposable branch
        # advances. / 在任一分支推进前，先准备好两组完整输入。
        if execution_order == CONTROL_FIRST:
            control_observation = (
                control.run_next_frozen_policy_period_with_supplied_draws(
                    noise_order_u=ordinary_noise,
                    next_value_index=next_value_index,
                )
            )
            treatment_observation = (
                treatment.run_next_frozen_policy_period_with_supplied_draws(
                    noise_order_u=treatment_noise,
                    next_value_index=next_value_index,
                )
            )
        else:
            treatment_observation = (
                treatment.run_next_frozen_policy_period_with_supplied_draws(
                    noise_order_u=treatment_noise,
                    next_value_index=next_value_index,
                )
            )
            control_observation = (
                control.run_next_frozen_policy_period_with_supplied_draws(
                    noise_order_u=ordinary_noise,
                    next_value_index=next_value_index,
                )
            )

        if (
            control_observation.period_number != expected_global_period
            or treatment_observation.period_number != expected_global_period
        ):
            raise RuntimeError("Local/global IRF time mapping failed. / IRF 局部与全局时期映射错误。")
        if (
            control_observation.noise_order_u != ordinary_noise
            or treatment_observation.noise_order_u != treatment_noise
            or control_observation.next_value_index != next_value_index
            or treatment_observation.next_value_index != next_value_index
        ):
            raise RuntimeError("A paired supplied draw was not used exactly. / 配对外部抽样未被精确使用。")
        if local_period == IRF_FIRST_TRANSACTION_PERIOD and (
            control_observation.current_state_indexes != expected_t1_state_indexes
            or treatment_observation.current_state_indexes != expected_t1_state_indexes
            or control_observation.fundamental_value_v
            != checkpoint.payload.current_value
            or treatment_observation.fundamental_value_v
            != checkpoint.payload.current_value
        ):
            raise RuntimeError(
                "The carried t=0 outcome does not feed the t=1 state. / "
                "保存的 t=0 结果没有正确进入 t=1 状态。"
            )
        if local_period < PAPER_SHOCK_PERIOD:
            if control_observation != treatment_observation:
                pre_shock_parity = False
                raise RuntimeError("Branches diverged before the shock. / 两个分支在冲击前发生分化。")

        if local_period == PAPER_SHOCK_PERIOD:
            pre_noise_control = (
                control_observation.current_state_indexes,
                control_observation.current_state_id,
                control_observation.current_value_index,
                control_observation.fundamental_value_v,
                control_observation.action_indexes,
                control_observation.raw_orders_x,
                control_observation.xi_0_hat,
                control_observation.xi_1_hat,
                control_observation.gamma_0_hat,
                control_observation.gamma_1_hat,
                control_observation.price_impact_lambda_hat,
            )
            pre_noise_treatment = (
                treatment_observation.current_state_indexes,
                treatment_observation.current_state_id,
                treatment_observation.current_value_index,
                treatment_observation.fundamental_value_v,
                treatment_observation.action_indexes,
                treatment_observation.raw_orders_x,
                treatment_observation.xi_0_hat,
                treatment_observation.xi_1_hat,
                treatment_observation.gamma_0_hat,
                treatment_observation.gamma_1_hat,
                treatment_observation.price_impact_lambda_hat,
            )
            if pre_noise_control != pre_noise_treatment:
                raise RuntimeError("Shock-period branches differed before noise. / 冲击期分支在噪声到达前已经不同。")
            if applied_shock is None:
                raise RuntimeError("The treatment shock is missing. / 实验组冲击丢失。")
            shared_lambda = control_observation.price_impact_lambda_hat
            if shared_lambda <= 0.0:
                raise RuntimeError("Shock-period lambda must be positive. / 冲击期 lambda 必须为正。")
            t3_actual_increment = orient_price(
                treatment_observation.continuous_price_p,
                current_value,
                checkpoint.payload.parameters.value_mean,
            ) - orient_price(
                control_observation.continuous_price_p,
                current_value,
                checkpoint.payload.parameters.value_mean,
            )
            t3_expected_increment = (
                shared_lambda * calibration.absolute_noise_shock
            )
            t3_identity_error = t3_actual_increment - t3_expected_increment
            if not isclose(
                t3_actual_increment,
                t3_expected_increment,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError("The t=3 price-shock identity failed. / t=3 价格冲击恒等式失败。")

        control_row = _completed_row(control_observation)
        treatment_row = _completed_row(treatment_observation)
        rows_match = (
            control.market_maker.snapshot()[-1] == control_row
            and treatment.market_maker.snapshot()[-1] == treatment_row
        )
        if not rows_match:
            raise RuntimeError("A completed row entered the wrong maker history. / 完整记录进入了错误做市商历史。")
        control_prior_tuple = (
            control_prior_ols.xi_0_hat,
            control_prior_ols.xi_1_hat,
            control_prior_ols.gamma_0_hat,
            control_prior_ols.gamma_1_hat,
        )
        treatment_prior_tuple = (
            treatment_prior_ols.xi_0_hat,
            treatment_prior_ols.xi_1_hat,
            treatment_prior_ols.gamma_0_hat,
            treatment_prior_ols.gamma_1_hat,
        )
        control_observed_prior = (
            control_observation.xi_0_hat,
            control_observation.xi_1_hat,
            control_observation.gamma_0_hat,
            control_observation.gamma_1_hat,
        )
        treatment_observed_prior = (
            treatment_observation.xi_0_hat,
            treatment_observation.xi_1_hat,
            treatment_observation.gamma_0_hat,
            treatment_observation.gamma_1_hat,
        )
        prior_ols_matches = (
            control_prior_tuple == control_observed_prior
            and treatment_prior_tuple == treatment_observed_prior
        )
        if not prior_ols_matches:
            raise RuntimeError(
                "A branch did not price from its own prior OLS history. / "
                "某个分支没有使用自己的先前 OLS 历史定价。"
            )
        control_oriented_orders = tuple(
            orient_order(
                order,
                current_value,
                checkpoint.payload.parameters.value_mean,
            )
            for order in control_observation.raw_orders_x
        )
        treatment_oriented_orders = tuple(
            orient_order(
                order,
                current_value,
                checkpoint.payload.parameters.value_mean,
            )
            for order in treatment_observation.raw_orders_x
        )
        control_oriented_price = orient_price(
            control_observation.continuous_price_p,
            current_value,
            checkpoint.payload.parameters.value_mean,
        )
        treatment_oriented_price = orient_price(
            treatment_observation.continuous_price_p,
            current_value,
            checkpoint.payload.parameters.value_mean,
        )
        period_records.append(
            PairedIRFPeriod(
                local_period=local_period,
                expected_global_period=expected_global_period,
                common_current_fundamental_value=current_value,
                ordinary_noise_order_u=ordinary_noise,
                common_next_value_index=next_value_index,
                control_noise_used_for_pricing=ordinary_noise,
                treatment_noise_used_for_pricing=treatment_noise,
                signed_treatment_shock_u=(
                    0.0
                    if applied_shock is None
                    else applied_shock.signed_adverse_shock
                ),
                shock_applied=applied_shock is not None,
                applied_noise_shock=applied_shock,
                control_observation=control_observation,
                treatment_observation=treatment_observation,
                control_oriented_price=control_oriented_price,
                treatment_oriented_price=treatment_oriented_price,
                paired_oriented_price_increment=(
                    treatment_oriented_price - control_oriented_price
                ),
                control_oriented_orders=control_oriented_orders,  # type: ignore[arg-type]
                treatment_oriented_orders=treatment_oriented_orders,  # type: ignore[arg-type]
                common_draws_verified=True,
                completed_rows_match_named_branches=True,
                prior_ols_matches_named_branch_history=True,
            )
        )

    if shock_application_count != 1:
        raise RuntimeError("Treatment shock was not applied exactly once. / 实验组冲击没有恰好应用一次。")
    if (
        control.all_random_states() != control_rng_start
        or treatment.all_random_states() != treatment_rng_start
    ):
        raise RuntimeError("A branch-internal RNG moved. / 分支内部随机流发生推进。")
    frozen_learning = (
        tuple(trader.q_table.tobytes() for trader in control.traders)
        == control_q_start
        and tuple(trader.q_table.tobytes() for trader in treatment.traders)
        == treatment_q_start
        and tuple(control.shared_value_visit_counts) == control_counts_start
        and tuple(treatment.shared_value_visit_counts) == treatment_counts_start
        and all(not trader.q_table.flags.writeable for trader in control.traders)
        and all(not trader.q_table.flags.writeable for trader in treatment.traders)
        and np.array_equal(
            control.frozen_policy_action_indexes_snapshot(),
            control_policy_start,
        )
        and np.array_equal(
            treatment.frozen_policy_action_indexes_snapshot(),
            treatment_policy_start,
        )
    )
    if not frozen_learning:
        raise RuntimeError("Frozen learning state changed. / 冻结学习状态发生改变。")
    if (
        control.market_maker.successful_append_count
        != control_appends_start + IRF_PERIOD_COUNT
        or treatment.market_maker.successful_append_count
        != treatment_appends_start + IRF_PERIOD_COUNT
    ):
        raise RuntimeError("A maker did not append once per IRF period. / 做市商没有每个 IRF 时期追加一次。")
    if t3_actual_increment is None or t3_expected_increment is None or t3_identity_error is None:
        raise RuntimeError("The t=3 price audit is missing. / t=3 价格审计丢失。")
    periods = tuple(period_records)
    response_record = next(
        period for period in periods if period.local_period == PAPER_RESPONSE_PERIOD
    )
    paper_target_used = isclose(
        calibration.target_normalized_price_deviation,
        PAPER_TARGET_PRICE_DEVIATION,
        rel_tol=0.0,
        abs_tol=0.0,
    )
    return PairedIRFPathReceipt(
        protocol_version=PAIRED_IRF_PROTOCOL_VERSION,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        source_session_index=checkpoint.payload.seed_manifest.session_index,
        checkpoint_global_origin_period=checkpoint.payload.origin_global_period,
        irf_local_state_origin=IRF_LOCAL_STATE_ORIGIN,
        carried_local_t0_global_period=(
            checkpoint.payload.origin_global_period - 1
        ),
        carried_local_t0_price=checkpoint.payload.previous_price,
        carried_local_t0_value=checkpoint.payload.previous_value,
        local_t1_current_value=checkpoint.payload.current_value,
        t0_outcome_carried_not_reexecuted=True,
        local_transaction_periods=IRF_LOCAL_PERIODS,
        shock_local_period=PAPER_SHOCK_PERIOD,
        response_local_period=PAPER_RESPONSE_PERIOD,
        seed_manifest=schedule.seed_manifest,
        draw_schedule=schedule,
        shock_calibration=calibration,
        branch_execution_order=execution_order,
        periods=periods,
        control_final_state=_final_summary(control),
        treatment_final_state=_final_summary(treatment),
        paired_t3_treatment_minus_control_oriented_price_increment=(
            t3_actual_increment
        ),
        expected_paired_t3_increment_from_lambda=t3_expected_increment,
        t3_price_identity_error=(0.0 if t3_identity_error == 0.0 else t3_identity_error),
        control_t4_oriented_orders=response_record.control_oriented_orders,
        treatment_t4_oriented_orders=response_record.treatment_oriented_orders,
        paths_economically_represented=1,
        execution_replays_for_order_audit=1,
        full_paper_path_count_verified=False,
        classification_ready=False,
        pre_shock_exact_parity_verified=pre_shock_parity,
        shock_applied_exactly_once_verified=True,
        common_draws_verified=common_draws,
        branch_internal_rngs_unchanged_verified=True,
        frozen_learning_state_verified=True,
        independent_market_maker_histories_verified=True,
        each_branch_t4_ols_uses_own_t3_history_verified=(
            response_record.prior_ols_matches_named_branch_history
        ),
        branch_order_invariance_verified=False,
        calibration_arithmetic_verified=True,
        paper_1_2_percent_target_used=paper_target_used,
        # A base Step-34 receipt has no raw cell/sample digest, even if someone
        # forges a Boolean on a copied dataclass. / 基础第 34 步凭证没有原始实验
        # 单元或样本哈希，因此绝不把可伪造的布尔值当作来源证明。
        calibration_aggregate_provenance_verified=False,
        protocol_disclosure=PairedIRFProtocolDisclosure(),
    )


def run_and_audit_one_paired_irf_path(
    checkpoint: ConvergedMarketCheckpoint,
    *,
    irf_experiment_seed: int,
    path_index: int,
    shock_calibration: UniformShockCalibration,
) -> PairedIRFPathReceipt:
    """Run both branch orders and return one order-invariant receipt.

    用两种分支先后顺序分别运行，并返回一份顺序不变的凭证。
    """

    control_first = run_one_paired_irf_path(
        checkpoint,
        irf_experiment_seed=irf_experiment_seed,
        path_index=path_index,
        shock_calibration=shock_calibration,
        branch_execution_order=CONTROL_FIRST,
    )
    treatment_first = run_one_paired_irf_path(
        checkpoint,
        irf_experiment_seed=irf_experiment_seed,
        path_index=path_index,
        shock_calibration=shock_calibration,
        branch_execution_order=TREATMENT_FIRST,
    )
    if (
        control_first.draw_schedule != treatment_first.draw_schedule
        or control_first.periods != treatment_first.periods
        or control_first.control_final_state
        != treatment_first.control_final_state
        or control_first.treatment_final_state
        != treatment_first.treatment_final_state
    ):
        raise RuntimeError("Branch execution order changed the economic path. / 分支运行顺序改变了经济路径。")
    return replace(
        control_first,
        execution_replays_for_order_audit=2,
        branch_order_invariance_verified=True,
    )


def _build_demo_checkpoint() -> ConvergedMarketCheckpoint:
    """Build a nonzero-origin checkpoint for a visible wiring demo.

    建立全局起点非零的 checkpoint，用于可见的接线演示。
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
        experiment_cell_key="step35b_demo_only",
        session_index=0,
    )
    controller = SessionPhaseController.create_for_fresh_session(
        session,
        convergence_periods_required=5,
        measurement_periods_required=4,
    )
    while controller.phase is SessionPhase.TRAINING:
        result = controller.run_next_period()
        if result is not None:
            raise RuntimeError("Demo emitted measurement too early. / 演示过早产生测量记录。")
    if controller.measurement_periods_completed != 0:
        raise RuntimeError("Demo passed the capture boundary. / 演示越过了保存边界。")
    return capture_at_convergence_boundary(controller)


def main() -> None:
    """Run one hand-auditable paired-path wiring demonstration.

    运行一条可以逐项核对的配对路径接线演示。
    """

    checkpoint = _build_demo_checkpoint()
    # This familiar Step-34 hand calibration is only a wiring demo. Its
    # aggregate provenance remains explicitly unverified. / 使用第 34 步熟悉的
    # 手算校准只为演示接线；其总体数据来源仍明确标为未验证。
    calibration = calibrate_uniform_noise_shock(2.0, 0.5, 0.5)
    receipt = run_and_audit_one_paired_irf_path(
        checkpoint,
        irf_experiment_seed=20_260_829,
        path_index=0,
        shock_calibration=calibration,
    )

    shock_record = receipt.periods[PAPER_SHOCK_PERIOD - 1]
    response_record = receipt.periods[PAPER_RESPONSE_PERIOD - 1]
    print("Step 35B: one paired IRF path / 步骤 35B：一条配对 IRF 路径")
    print(f"Checkpoint global origin / checkpoint 全局起点: {receipt.checkpoint_global_origin_period}")
    print(f"Local transactions / 局部交易时期: {receipt.local_transaction_periods}")
    print(f"Global observations / 对应全局时期: {tuple(row.expected_global_period for row in receipt.periods)}")
    print(f"Path index / 路径编号: {receipt.seed_manifest.path_index}")
    print(f"Ordinary u_3 / 普通 u_3: {shock_record.ordinary_noise_order_u:.6f}")
    print(f"Signed treatment shock / 实验组带符号冲击: {shock_record.signed_treatment_shock_u:.6f}")
    print(f"Control noise used / 对照组实际噪声: {shock_record.control_noise_used_for_pricing:.6f}")
    print(f"Treatment noise used / 实验组实际噪声: {shock_record.treatment_noise_used_for_pricing:.6f}")
    print(
        "t=3 paired treatment-control oriented price increment / "
        "t=3 实验组减对照组的方向调整价格增量: "
        f"{receipt.paired_t3_treatment_minus_control_oriented_price_increment:.9f}"
    )
    print(f"t=4 control oriented orders / t=4 对照组方向调整订单: {response_record.control_oriented_orders}")
    print(f"t=4 treatment oriented orders / t=4 实验组方向调整订单: {response_record.treatment_oriented_orders}")
    print("Common ordinary draws verified / 相同普通抽样验证通过")
    print("Shock applied to treatment exactly once / 实验组冲击恰好应用一次")
    print("Branch execution order does not matter / 分支运行先后不影响结果")
    print("One path only; classification is not ready / 仅一条路径；尚不能进行机制分类")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
