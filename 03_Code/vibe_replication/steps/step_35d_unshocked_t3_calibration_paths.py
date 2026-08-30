"""Step 35D: aggregate unshocked local-t=3 calibration paths.

步骤 35D：汇总无冲击的局部 t=3 校准路径。

Run / 运行:
    py -3 -X utf8 steps/step_35d_unshocked_t3_calibration_paths.py

What this step does / 本步骤做什么:
    For one already-trained session, reuse its exact Step-35A convergence
    checkpoint and run independent unshocked continuations through local
    t=1,2,3.  It collects the actual prior-history ``lambda_hat_3`` used to
    price t=3 and the unshocked oriented price ``p_tilde_3``. / 对一个已经训练
    完成的 session，复用第 35A 步的精确收敛 checkpoint，独立运行无冲击的
    t=1、2、3 续接路径，并收集真正用于 t=3 定价的旧历史 ``lambda_hat_3``
    以及无冲击方向调整价格 ``p_tilde_3``。

Paper boundary / 原文边界:
    The paper averages 10,000 stochastic paths per trained session.  It does
    not disclose a numerical shock-calibration algorithm.  Therefore this step
    only gathers the two finite-sample moments that Step 35E will pool across
    the experiment cell. / 原文对每个已训练 session 平均 10,000 条随机路径，
    但没有公开数值冲击校准算法。因此本步骤只收集两个有限样本统计量；第 35E
    步才会跨实验单元汇总它们。

Strict non-goals / 明确不做:
    No shock is applied, no treatment branch is run, t=4 is not executed, and
    no mechanism is classified. / 不施加冲击、不运行实验分支、不执行 t=4、
    不分类合谋机制。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import isclose, isfinite
from numbers import Integral, Real
from pathlib import Path
import struct
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from step_26_reproducible_random_streams import (
    FrozenPolicyPeriodObservation,
    RandomizedMarketSession,
    SessionSeedManifest,
)
from step_24_adaptive_market_maker_price import calculate_adaptive_price_impact
from step_28_session_phases import SessionPhase
from steps.step_34_mechanism_classifier import (
    PAPER_PATHS_PER_SESSION,
    PAPER_SHOCK_PERIOD,
    PAPER_TARGET_PRICE_DEVIATION,
    orient_price,
)
from steps.step_35a_converged_market_checkpoint import (
    ConvergedMarketCheckpoint,
    restore_detached_frozen_branch,
    verify_converged_market_checkpoint,
)
from steps.step_35b_paired_irf_path import (
    PAIRED_PATH_SEED_VERSION,
    PairedPathSeedManifest,
    VerifiedPairedPathScheduleContext,
    audit_all_planned_paired_path_seed_uniqueness,
    build_paired_path_draw_schedule_from_verified_context,
    derive_paired_path_seed_manifest_from_verified_context,
    prepare_verified_paired_path_schedule_context,
)
from steps.step_35c_irf_long_run_baseline import (
    IRFLongRunBaselineReceipt,
    OnlineIRFLongRunBaselineScorer,
    validate_irf_long_run_baseline_receipt,
)


UNSHOCKED_T3_PROTOCOL_VERSION = "step35d-unshocked-t3-calibration-v2"
UNSHOCKED_T3_DIGEST_DOMAIN = b"vibe-replication.step35d.unshocked-t3-path.v1\0"
UNSHOCKED_T3_RECEIPT_DOMAIN = b"vibe-replication.step35d.session-receipt.v2\0"
CALIBRATION_LOCAL_PERIODS = (1, 2, PAPER_SHOCK_PERIOD)
CALIBRATION_PERIOD_COUNT = len(CALIBRATION_LOCAL_PERIODS)


def _finite_real(number: float, label: str) -> float:
    """Return one finite non-Boolean float. / 返回一个有限、非布尔浮点数。"""

    if isinstance(number, bool) or not isinstance(number, Real):
        raise TypeError(f"{label} must be a real number. / {label} 必须是实数。")
    converted = float(number)
    if not isfinite(converted):
        raise ValueError(f"{label} must be finite. / {label} 必须是有限数。")
    return converted


def _checked_path_count(path_count: int) -> int:
    """Validate a debug or paper per-session path count. / 检查调试或论文路径数。"""

    if isinstance(path_count, bool) or not isinstance(path_count, Integral):
        raise TypeError("path_count must be an integer. / path_count 必须是整数。")
    checked = int(path_count)
    if not 1 <= checked <= PAPER_PATHS_PER_SESSION:
        raise ValueError("path_count must lie in [1, 10000]. / path_count 必须位于 [1, 10000]。")
    return checked


def _next_neumaier_sum(
    running_sum: float,
    compensation: float,
    value: float,
) -> tuple[float, float]:
    """Add one value accurately using two stored floats. / 用两个浮点数准确累加。"""

    candidate_sum = running_sum + value
    if abs(running_sum) >= abs(value):
        candidate_compensation = compensation + running_sum - candidate_sum + value
    else:
        candidate_compensation = compensation + value - candidate_sum + running_sum
    if not isfinite(candidate_sum) or not isfinite(candidate_compensation):
        raise OverflowError("Calibration sum overflowed. / 校准求和发生溢出。")
    return candidate_sum, candidate_compensation


def _completed_path_payload(
    *,
    checkpoint_sha256: str,
    path_index: int,
    seed_manifest: PairedPathSeedManifest,
    ordinary_noise_orders: tuple[float, ...],
    next_value_indexes: tuple[int, ...],
    observations: tuple[FrozenPolicyPeriodObservation, ...],
) -> bytes:
    """Encode every executed field used by this calibration pass.

    确定性编码本校准实际执行并使用的全部字段。
    """

    observation_fields = tuple(
        (
            observation.period_number,
            observation.current_state_indexes,
            observation.current_value_index,
            observation.fundamental_value_v.hex(),
            observation.action_indexes,
            tuple(float(order).hex() for order in observation.raw_orders_x),
            observation.noise_order_u.hex(),
            observation.total_order_flow_y.hex(),
            observation.xi_0_hat.hex(),
            observation.xi_1_hat.hex(),
            observation.gamma_0_hat.hex(),
            observation.gamma_1_hat.hex(),
            observation.price_impact_lambda_hat.hex(),
            observation.continuous_price_p.hex(),
            observation.next_value_index,
            observation.next_state_indexes,
        )
        for observation in observations
    )
    fields = (
        checkpoint_sha256,
        path_index,
        seed_manifest,
        tuple(number.hex() for number in ordinary_noise_orders),
        next_value_indexes,
        observation_fields,
    )
    return repr(fields).encode("ascii") + b"\0"


@dataclass(frozen=True)
class UnshockedT3CalibrationPathResult:
    """One compact result returned only until the online reducer consumes it.

    一条紧凑路径结果；在线汇总器读取后不会长期保存它。
    """

    protocol_version: str
    checkpoint_sha256: str
    path_index: int
    seed_manifest: PairedPathSeedManifest
    executed_local_periods: tuple[int, ...]
    ordinary_noise_orders_u: tuple[float, ...]
    next_value_indexes: tuple[int, ...]
    observations: tuple[FrozenPolicyPeriodObservation, ...]
    t3_price_impact_lambda_hat: float
    t3_unshocked_oriented_price: float
    t3_fundamental_value: float
    t3_continuous_price: float
    t3_total_order_flow: float
    path_fields_sha256: str
    ordinary_noise_only: bool
    shock_applied: bool
    treatment_branch_run: bool
    t4_executed: bool
    rollback_periods_verified: int


class ReusableUnshockedT3Workspace:
    """Restore once, then run and exactly roll back one branch per path.

    只恢复一次；之后每条路径运行完都把同一分支精确回滚。

    This object is intentionally single-threaded.  Future Step 36 parallelism
    should assign separate sessions to separate processes. / 本对象故意只供单
    线程使用；未来第 36 步应把不同 session 分配给不同进程。
    """

    def __init__(
        self,
        checkpoint: ConvergedMarketCheckpoint,
        *,
        baseline_scorer: OnlineIRFLongRunBaselineScorer,
        irf_experiment_seed: int,
    ) -> None:
        if not isinstance(baseline_scorer, OnlineIRFLongRunBaselineScorer):
            raise TypeError("baseline_scorer has the wrong type. / baseline_scorer 类型错误。")
        verify_converged_market_checkpoint(checkpoint)
        baseline = baseline_scorer.verified_live_result_for_step35d(checkpoint)
        validate_irf_long_run_baseline_receipt(baseline)
        self.checkpoint = checkpoint
        self.baseline_receipt = baseline
        self.schedule_context: VerifiedPairedPathScheduleContext = (
            prepare_verified_paired_path_schedule_context(
                checkpoint,
                irf_experiment_seed=irf_experiment_seed,
            )
        )
        self.seed_audit = audit_all_planned_paired_path_seed_uniqueness(
            checkpoint,
            irf_experiment_seed=irf_experiment_seed,
        )
        if (
            not self.seed_audit.all_planned_child_seeds_are_unique
            or self.seed_audit.planned_path_count != PAPER_PATHS_PER_SESSION
            or self.seed_audit.paths_were_executed
        ):
            raise RuntimeError("The planned path-seed audit is inconsistent. / 计划路径种子核对不一致。")
        self._branch: RandomizedMarketSession = restore_detached_frozen_branch(
            checkpoint
        )
        self._initial_rng_states = self._branch.all_random_states()
        self._initial_q_digests = tuple(
            sha256(trader.q_table.tobytes(order="C")).hexdigest()
            for trader in self._branch.traders
        )
        self._initial_policy_digest = sha256(
            self._branch.frozen_policy_action_indexes_snapshot().tobytes(order="C")
        ).hexdigest()
        self._initial_visit_counts = tuple(self._branch.shared_value_visit_counts)
        self.paths_completed = 0
        self.rollbacks_completed = 0
        self._poisoned = False
        self._closed = False
        self.verify_exact_checkpoint_reset()

    @property
    def is_poisoned(self) -> bool:
        """Report whether an interrupted path made reuse unsafe. / 报告工作区是否不再安全。"""

        return self._poisoned

    def _cheap_reset_check(self) -> None:
        """Check O(1) reset facts after every path. / 每条路径后检查 O(1) 回滚事实。"""

        payload = self.checkpoint.payload
        if (
            self._branch.period_number != payload.origin_global_period
            or self._branch.previous_price != payload.previous_price
            or self._branch.previous_value != payload.previous_value
            or self._branch.current_value != payload.current_value
            or self._branch.frozen_draw_source_mode is not None
            or self._branch.market_maker.successful_append_count
            != payload.market_maker_state.successful_append_count
            or self._branch.market_maker.resynchronization_count
            != payload.market_maker_state.resynchronization_count
        ):
            raise RuntimeError("A reusable branch did not return to t=0. / 可复用分支没有回到 t=0。")

    def verify_exact_checkpoint_reset(self) -> None:
        """Perform the heavier full-state audit at safe batch boundaries.

        在安全批次边界执行较重的完整状态核对。
        """

        self._cheap_reset_check()
        payload = self.checkpoint.payload
        if self._branch.market_maker.export_state() != payload.market_maker_state:
            raise RuntimeError("Market-maker rollback is not exact. / 做市商回滚并不精确。")
        if self._branch.all_random_states() != self._initial_rng_states:
            raise RuntimeError("Supplied paths changed an internal RNG. / 外部抽样路径改变了内部 RNG。")
        q_digests = tuple(
            sha256(trader.q_table.tobytes(order="C")).hexdigest()
            for trader in self._branch.traders
        )
        policy_digest = sha256(
            self._branch.frozen_policy_action_indexes_snapshot().tobytes(order="C")
        ).hexdigest()
        if (
            q_digests != self._initial_q_digests
            or policy_digest != self._initial_policy_digest
            or tuple(self._branch.shared_value_visit_counts)
            != self._initial_visit_counts
            or any(trader.q_table.flags.writeable for trader in self._branch.traders)
        ):
            raise RuntimeError("Frozen learning state changed. / 固定学习状态发生改变。")

    def run_path(self, path_index: int) -> UnshockedT3CalibrationPathResult:
        """Run t=1..3 with ordinary draws, then roll back before returning.

        使用普通抽样运行 t=1..3，并在返回结果前完成回滚。
        """

        if self._closed:
            raise RuntimeError("The workspace is closed. / 工作区已经关闭。")
        if self._poisoned:
            raise RuntimeError("The workspace is poisoned and must be rebuilt. / 工作区已失效，必须重建。")
        schedule = build_paired_path_draw_schedule_from_verified_context(
            self.schedule_context,
            path_index=path_index,
        )
        token = self._branch.begin_reversible_frozen_supplied_path(
            max_periods=CALIBRATION_PERIOD_COUNT,
        )
        observations: list[FrozenPolicyPeriodObservation] = []
        try:
            for schedule_offset, local_period in enumerate(
                CALIBRATION_LOCAL_PERIODS
            ):
                prior_ols = self._branch.market_maker.estimates()
                observation = (
                    self._branch.run_next_frozen_policy_period_with_supplied_draws(
                        noise_order_u=schedule.ordinary_noise_orders_u[
                            schedule_offset
                        ],
                        next_value_index=schedule.next_value_indexes[
                            schedule_offset
                        ],
                    )
                )
                expected_global_period = (
                    self.checkpoint.payload.origin_global_period
                    + local_period
                    - 1
                )
                if observation.period_number != expected_global_period:
                    raise RuntimeError("Local/global time mapping failed. / 局部与全局时期映射错误。")
                if (
                    observation.xi_0_hat != prior_ols.xi_0_hat
                    or observation.xi_1_hat != prior_ols.xi_1_hat
                    or observation.gamma_0_hat != prior_ols.gamma_0_hat
                    or observation.gamma_1_hat != prior_ols.gamma_1_hat
                ):
                    raise RuntimeError(
                        "A period did not use its own pre-append OLS estimates. / "
                        "某一期没有使用追加本期记录之前的 OLS 估计。"
                    )
                expected_lambda = calculate_adaptive_price_impact(
                    prior_ols,
                    self.checkpoint.payload.parameters.pricing_error_weight,
                )
                if observation.price_impact_lambda_hat != expected_lambda:
                    raise RuntimeError(
                        "Recorded lambda differs from the pre-append OLS price rule. / "
                        "记录的 lambda 与追加前 OLS 定价公式不一致。"
                    )
                observations.append(observation)
            t3 = observations[-1]
            expected_price = (
                t3.gamma_0_hat
                + t3.price_impact_lambda_hat * t3.total_order_flow_y
            )
            if not isfinite(expected_price) or not isclose(
                t3.continuous_price_p,
                expected_price,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError("t=3 price does not use its recorded prior-history lambda. / t=3 价格未使用记录的旧历史 lambda。")
            executed_observations = tuple(observations)
            executed_noise = tuple(
                schedule.ordinary_noise_orders_u[:CALIBRATION_PERIOD_COUNT]
            )
            executed_values = tuple(
                schedule.next_value_indexes[:CALIBRATION_PERIOD_COUNT]
            )
            payload = _completed_path_payload(
                checkpoint_sha256=self.checkpoint.checkpoint_sha256,
                path_index=path_index,
                seed_manifest=schedule.seed_manifest,
                ordinary_noise_orders=executed_noise,
                next_value_indexes=executed_values,
                observations=executed_observations,
            )
            result_without_rollback = UnshockedT3CalibrationPathResult(
                protocol_version=UNSHOCKED_T3_PROTOCOL_VERSION,
                checkpoint_sha256=self.checkpoint.checkpoint_sha256,
                path_index=path_index,
                seed_manifest=schedule.seed_manifest,
                executed_local_periods=CALIBRATION_LOCAL_PERIODS,
                ordinary_noise_orders_u=executed_noise,
                next_value_indexes=executed_values,
                observations=executed_observations,
                t3_price_impact_lambda_hat=t3.price_impact_lambda_hat,
                t3_unshocked_oriented_price=orient_price(
                    t3.continuous_price_p,
                    t3.fundamental_value_v,
                    self.checkpoint.payload.parameters.value_mean,
                ),
                t3_fundamental_value=t3.fundamental_value_v,
                t3_continuous_price=t3.continuous_price_p,
                t3_total_order_flow=t3.total_order_flow_y,
                path_fields_sha256=sha256(
                    UNSHOCKED_T3_DIGEST_DOMAIN + payload
                ).hexdigest(),
                ordinary_noise_only=True,
                shock_applied=False,
                treatment_branch_run=False,
                t4_executed=False,
                rollback_periods_verified=0,
            )
        except BaseException:
            # Try to leave no partial path behind, but never trust this workspace
            # again after an unexpected failure. / 尝试清除半条路径，但意外失败后
            # 绝不继续信任并复用这个工作区。
            self._poisoned = True
            try:
                self._branch.rollback_reversible_frozen_supplied_path(token)
            except BaseException:
                pass
            raise

        try:
            rolled_back_periods = (
                self._branch.rollback_reversible_frozen_supplied_path(token)
            )
            if rolled_back_periods != CALIBRATION_PERIOD_COUNT:
                raise RuntimeError("Rollback period count is wrong. / 回滚时期数错误。")
            self._cheap_reset_check()
        except BaseException:
            self._poisoned = True
            raise
        self.paths_completed += 1
        self.rollbacks_completed += 1
        return replace(
            result_without_rollback,
            rollback_periods_verified=rolled_back_periods,
        )

    def close_and_verify(self) -> None:
        """Prove exact reset once more, then prevent further reuse.

        再次证明精确回滚，之后禁止继续复用。
        """

        if self._poisoned:
            raise RuntimeError("A poisoned workspace cannot finalize. / 已失效工作区不能完成。")
        if not self._closed:
            self.verify_exact_checkpoint_reset()
            self._closed = True


class OnlineUnshockedT3Moments:
    """Constant-memory reducer with exact ascending path coverage.

    固定内存汇总器，并要求路径编号严格从零递增。
    """

    def __init__(
        self,
        *,
        schedule_context: VerifiedPairedPathScheduleContext,
        value_mean: float,
        pricing_error_weight: float,
    ) -> None:
        if not isinstance(schedule_context, VerifiedPairedPathScheduleContext):
            raise TypeError("schedule_context has the wrong type. / schedule_context 类型错误。")
        self.schedule_context = schedule_context
        self.checkpoint_sha256 = schedule_context.checkpoint_sha256
        self.source_session_index = int(schedule_context.source_session_index)
        self.value_mean = _finite_real(value_mean, "value_mean")
        self.pricing_error_weight = _finite_real(
            pricing_error_weight,
            "pricing_error_weight",
        )
        if self.pricing_error_weight <= 0.0:
            raise ValueError("pricing_error_weight must be positive. / pricing_error_weight 必须为正。")
        self.count = 0
        self._lambda_sum = (0.0, 0.0)
        self._oriented_price_sum = (0.0, 0.0)
        self.minimum_lambda = float("inf")
        self.nonpositive_lambda_count = 0
        self._digest = sha256(UNSHOCKED_T3_DIGEST_DOMAIN)

    def audit_state(self) -> tuple[object, ...]:
        """Expose a fixed-size immutable state for tests. / 返回固定大小状态供测试。"""

        return (
            self.checkpoint_sha256,
            self.source_session_index,
            self.schedule_context,
            self.value_mean,
            self.pricing_error_weight,
            self.count,
            self._lambda_sum,
            self._oriented_price_sum,
            self.minimum_lambda,
            self.nonpositive_lambda_count,
            self._digest.hexdigest(),
        )

    def add(self, result: UnshockedT3CalibrationPathResult) -> None:
        """Validate one completed-and-rolled-back path, then commit it.

        检查一条已完成且已回滚的路径，再一次性加入汇总。
        """

        if not isinstance(result, UnshockedT3CalibrationPathResult):
            raise TypeError("result has the wrong type. / result 类型错误。")
        if (
            result.protocol_version != UNSHOCKED_T3_PROTOCOL_VERSION
            or result.checkpoint_sha256 != self.checkpoint_sha256
            or result.seed_manifest.checkpoint_sha256 != self.checkpoint_sha256
            or result.seed_manifest.source_session_index != self.source_session_index
            or result.path_index != self.count
            or result.seed_manifest.path_index != self.count
        ):
            raise ValueError("Path identity or canonical order is wrong. / 路径身份或标准顺序错误。")
        expected_manifest = derive_paired_path_seed_manifest_from_verified_context(
            self.schedule_context,
            path_index=result.path_index,
        )
        if result.seed_manifest != expected_manifest:
            raise ValueError("Path seed manifest does not rederive exactly. / 路径种子说明无法精确重新派生。")
        if (
            result.executed_local_periods != CALIBRATION_LOCAL_PERIODS
            or len(result.observations) != CALIBRATION_PERIOD_COUNT
            or len(result.ordinary_noise_orders_u) != CALIBRATION_PERIOD_COUNT
            or len(result.next_value_indexes) != CALIBRATION_PERIOD_COUNT
            or result.rollback_periods_verified != CALIBRATION_PERIOD_COUNT
            or not result.ordinary_noise_only
            or result.shock_applied
            or result.treatment_branch_run
            or result.t4_executed
        ):
            raise ValueError("Path protocol claims are inconsistent. / 路径协议声明不一致。")
        t3 = result.observations[-1]
        expected_oriented_price = orient_price(
            t3.continuous_price_p,
            t3.fundamental_value_v,
            self.value_mean,
        )
        try:
            expected_lambda = (
                self.pricing_error_weight * t3.gamma_1_hat + t3.xi_1_hat
            ) / (
                self.pricing_error_weight + t3.xi_1_hat ** 2
            )
        except (OverflowError, ZeroDivisionError) as error:
            raise ValueError(
                "Recorded OLS inputs cannot produce a finite lambda. / "
                "记录的 OLS 输入无法产生有限 lambda。"
            ) from error
        if (
            not isfinite(expected_lambda)
            or t3.price_impact_lambda_hat != expected_lambda
        ):
            raise ValueError("t=3 lambda disagrees with its recorded OLS inputs. / t=3 lambda 与记录的 OLS 输入不一致。")
        if (
            result.t3_fundamental_value != t3.fundamental_value_v
            or result.t3_continuous_price != t3.continuous_price_p
            or result.t3_total_order_flow != t3.total_order_flow_y
            or result.t3_price_impact_lambda_hat
            != t3.price_impact_lambda_hat
            or result.t3_unshocked_oriented_price != expected_oriented_price
            or any(
                observation.noise_order_u != noise_order
                or observation.next_value_index != next_value_index
                for observation, noise_order, next_value_index in zip(
                    result.observations,
                    result.ordinary_noise_orders_u,
                    result.next_value_indexes,
                    strict=True,
                )
            )
        ):
            raise ValueError("Reported path fields differ from its observations. / 路径报告字段与 observation 不一致。")
        expected_path_digest = sha256(
            UNSHOCKED_T3_DIGEST_DOMAIN
            + _completed_path_payload(
                checkpoint_sha256=result.checkpoint_sha256,
                path_index=result.path_index,
                seed_manifest=result.seed_manifest,
                ordinary_noise_orders=result.ordinary_noise_orders_u,
                next_value_indexes=result.next_value_indexes,
                observations=result.observations,
            )
        ).hexdigest()
        if result.path_fields_sha256 != expected_path_digest:
            raise ValueError("Path-fields digest failed. / 路径字段摘要校验失败。")
        price_impact = _finite_real(
            result.t3_price_impact_lambda_hat,
            "t3_price_impact_lambda_hat",
        )
        oriented_price = _finite_real(
            result.t3_unshocked_oriented_price,
            "t3_unshocked_oriented_price",
        )
        candidate_lambda_sum = _next_neumaier_sum(
            *self._lambda_sum,
            price_impact,
        )
        candidate_price_sum = _next_neumaier_sum(
            *self._oriented_price_sum,
            oriented_price,
        )
        candidate_digest = self._digest.copy()
        candidate_digest.update(
            repr(
                (
                    result.path_index,
                    result.path_fields_sha256,
                    price_impact.hex(),
                    oriented_price.hex(),
                )
            ).encode("ascii")
            + b"\0"
        )
        self._lambda_sum = candidate_lambda_sum
        self._oriented_price_sum = candidate_price_sum
        self.minimum_lambda = min(self.minimum_lambda, price_impact)
        if price_impact <= 0.0:
            self.nonpositive_lambda_count += 1
        self._digest = candidate_digest
        self.count += 1

    def summarized_values(self) -> tuple[float, float, float, int, str]:
        """Return means, minimum, bad-lambda count, and digest. / 返回均值、最小值、异常数和摘要。"""

        if self.count < 1:
            raise RuntimeError("No paths were aggregated. / 尚未汇总任何路径。")
        lambda_total = self._lambda_sum[0] + self._lambda_sum[1]
        price_total = self._oriented_price_sum[0] + self._oriented_price_sum[1]
        mean_lambda = lambda_total / self.count
        mean_price = price_total / self.count
        if not isfinite(mean_lambda) or not isfinite(mean_price):
            raise OverflowError("A calibration mean is not finite. / 校准均值不是有限数。")
        return (
            mean_lambda,
            mean_price,
            self.minimum_lambda,
            self.nonpositive_lambda_count,
            self._digest.hexdigest(),
        )


@dataclass(frozen=True)
class UnshockedT3SessionCalibrationReceipt:
    """One session's immutable Step-35D calibration moments.

    一个 session 的不可修改第 35D 步校准统计凭证。
    """

    protocol_version: str
    checkpoint_sha256: str
    implementation_tree_sha256: str
    source_seed_manifest: SessionSeedManifest
    irf_experiment_seed: int
    path_seed_derivation_version: str
    paths_requested: int
    paths_executed: int
    first_path_index: int
    last_path_index: int
    executed_local_periods: tuple[int, ...]
    mean_t3_price_impact_lambda: float
    minimum_t3_price_impact_lambda: float
    nonpositive_t3_lambda_count: int
    mean_unshocked_t3_oriented_price: float
    executed_path_fields_sha256: str
    long_run_baseline_receipt: IRFLongRunBaselineReceipt
    long_run_mean_oriented_price: float
    baseline_receipt_payload_sha256: str
    baseline_scored_fields_sha256: str
    source_session_full_restores: int
    successful_transaction_rollbacks: int
    full_checkpoint_reset_audits: int
    maximum_live_market_branches: int
    raw_path_results_retained: int
    planned_within_session_child_seed_uniqueness_verified: bool
    planned_child_stream_count: int
    canonical_zero_based_path_coverage_verified: bool
    constant_memory_online_aggregation_verified: bool
    exact_checkpoint_reset_after_batch_verified: bool
    live_step35c_baseline_provenance_verified: bool
    same_session_checkpoint_and_baseline_verified: bool
    different_samples_from_same_experiment_cell: bool
    paper_paths_per_session_count_matched_for_calibration: bool
    paper_measurement_and_convergence_scale_verified: bool
    all_t3_lambdas_positive: bool
    ready_for_cell_aggregation: bool
    ready_for_formal_paper_cell_aggregation: bool
    paper_1000_sessions_verified: bool
    formal_cross_session_seed_uniqueness_verified: bool
    uniform_cell_shock_calibrated: bool
    ready_for_uniform_shock_calibration: bool
    shock_applied: bool
    treatment_paths_executed: int
    t4_response_aggregated: bool
    classification_ready: bool
    mechanism_label_ready: bool
    paper_1_2_percent_achieved: bool
    full_paper_irf_protocol_verified: bool
    paper_figure_ready: bool
    paper_specifies_numerical_calibration_algorithm: bool
    replication_collects_control_t3_level_for_finite_sample_audit: bool
    target_normalized_price_deviation: float
    checksum_detects_stale_replacement_not_authentication: bool
    standalone_receipt_authenticates_executed_paths: bool
    receipt_payload_sha256: str


def _receipt_payload_digest(
    receipt: UnshockedT3SessionCalibrationReceipt,
) -> str:
    """Hash explicit values, not invisible Python object-sharing patterns.

    对明确数值取 hash，而不是对 Python 看不见的对象共享方式取 hash。

    Two equal nested dataclasses may share one memory reference before saving
    but become two equal objects after loading.  That is not an economic
    difference, so it must not change a scientific checksum. / 两个相等的嵌套
    dataclass 在保存前可能共用一个内存引用，读取后则成为两个数值相等的对象；
    这不是经济差异，因此不能改变科研 checksum。
    """

    digest = sha256(UNSHOCKED_T3_RECEIPT_DOMAIN)
    unsigned = asdict(replace(receipt, receipt_payload_sha256=""))
    _update_canonical_receipt_digest(digest, unsigned)
    return digest.hexdigest()


def _update_canonical_receipt_digest(digest: object, value: object) -> None:
    """Add one value with explicit type and length tags. / 用类型与长度标签加入一个值。"""

    def add(tag: bytes, payload: bytes = b"") -> None:
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    if value is None:
        add(b"N")
    elif isinstance(value, bool):
        add(b"B", b"1" if value else b"0")
    elif isinstance(value, int):
        add(b"I", str(value).encode("ascii"))
    elif isinstance(value, float):
        add(b"F", struct.pack(">d", value))
    elif isinstance(value, str):
        add(b"S", value.encode("utf-8"))
    elif isinstance(value, bytes):
        add(b"Y", value)
    elif isinstance(value, tuple):
        add(b"T", len(value).to_bytes(8, "big"))
        for item in value:
            _update_canonical_receipt_digest(digest, item)
    elif isinstance(value, list):
        add(b"L", len(value).to_bytes(8, "big"))
        for item in value:
            _update_canonical_receipt_digest(digest, item)
    elif isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(
                "Receipt dictionary keys must be strings. / receipt 字典键必须是字符串。"
            )
        add(b"D", len(value).to_bytes(8, "big"))
        for key in sorted(value):
            _update_canonical_receipt_digest(digest, key)
            _update_canonical_receipt_digest(digest, value[key])
    else:
        raise TypeError(
            f"Unsupported receipt digest type {type(value).__name__}. / "
            "receipt 摘要遇到不支持的类型。"
        )


def _is_sha256_text(value: object) -> bool:
    """Return whether a value is lowercase SHA-256 text. / 判断是否为小写 SHA-256 文本。"""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_unshocked_t3_session_calibration_receipt(
    receipt: UnshockedT3SessionCalibrationReceipt,
) -> None:
    """Reject a changed or logically exaggerated Step-35D receipt.

    拒绝被修改或夸大完成程度的第 35D 步凭证。
    """

    if not isinstance(receipt, UnshockedT3SessionCalibrationReceipt):
        raise TypeError("receipt has the wrong type. / receipt 类型错误。")
    if receipt.protocol_version != UNSHOCKED_T3_PROTOCOL_VERSION:
        raise ValueError("Receipt version is unsupported. / receipt 版本不支持。")
    if not _is_sha256_text(receipt.receipt_payload_sha256):
        raise ValueError("Receipt checksum format is invalid. / receipt 校验码格式无效。")
    if _receipt_payload_digest(receipt) != receipt.receipt_payload_sha256:
        raise ValueError("Receipt checksum failed. / receipt 校验失败。")
    if not _is_sha256_text(receipt.executed_path_fields_sha256):
        raise ValueError("Executed-path digest is invalid. / 已执行路径摘要无效。")
    if (
        isinstance(receipt.irf_experiment_seed, bool)
        or not isinstance(receipt.irf_experiment_seed, int)
        or not 0 <= receipt.irf_experiment_seed < 2**64
    ):
        raise ValueError("irf_experiment_seed is outside uint64. / irf_experiment_seed 超出 uint64。")
    validate_irf_long_run_baseline_receipt(receipt.long_run_baseline_receipt)
    count = _checked_path_count(receipt.paths_executed)
    canonical = (
        receipt.paths_requested == count
        and receipt.first_path_index == 0
        and receipt.last_path_index == count - 1
    )
    paper_path_count = canonical and count == PAPER_PATHS_PER_SESSION
    positive_lambdas = (
        receipt.mean_t3_price_impact_lambda > 0.0
        and receipt.minimum_t3_price_impact_lambda > 0.0
        and receipt.nonpositive_t3_lambda_count == 0
    )
    paper_scale = (
        receipt.long_run_baseline_receipt.paper_scale_thresholds_and_provenance_verified
    )
    if any(
        not isfinite(float(value))
        for value in (
            receipt.mean_t3_price_impact_lambda,
            receipt.minimum_t3_price_impact_lambda,
            receipt.mean_unshocked_t3_oriented_price,
            receipt.long_run_mean_oriented_price,
            receipt.target_normalized_price_deviation,
        )
    ):
        raise ValueError("Receipt contains a nonfinite value. / receipt 含非有限数。")
    if (
        not 0 <= receipt.nonpositive_t3_lambda_count <= count
        or receipt.executed_local_periods != CALIBRATION_LOCAL_PERIODS
        or receipt.path_seed_derivation_version != PAIRED_PATH_SEED_VERSION
        or receipt.checkpoint_sha256
        != receipt.long_run_baseline_receipt.source_checkpoint_sha256
        or receipt.implementation_tree_sha256
        != receipt.long_run_baseline_receipt.source_implementation_tree_sha256
        or receipt.source_seed_manifest
        != receipt.long_run_baseline_receipt.session_seed_manifest
        or receipt.long_run_mean_oriented_price
        != receipt.long_run_baseline_receipt.mean_oriented_price
        or receipt.baseline_receipt_payload_sha256
        != receipt.long_run_baseline_receipt.receipt_payload_sha256
        or receipt.baseline_scored_fields_sha256
        != receipt.long_run_baseline_receipt.scored_fields_sha256
    ):
        raise ValueError("Receipt provenance or counts are inconsistent. / receipt 来源或计数不一致。")
    expected_formal_ready = paper_path_count and paper_scale and positive_lambdas
    logical_claims = (
        receipt.source_session_full_restores == 1,
        receipt.successful_transaction_rollbacks == count,
        receipt.full_checkpoint_reset_audits == 2,
        receipt.maximum_live_market_branches == 1,
        receipt.raw_path_results_retained == 0,
        receipt.planned_within_session_child_seed_uniqueness_verified,
        receipt.planned_child_stream_count == 2 * PAPER_PATHS_PER_SESSION,
        receipt.canonical_zero_based_path_coverage_verified == canonical,
        receipt.constant_memory_online_aggregation_verified,
        receipt.exact_checkpoint_reset_after_batch_verified,
        receipt.live_step35c_baseline_provenance_verified,
        receipt.same_session_checkpoint_and_baseline_verified,
        receipt.different_samples_from_same_experiment_cell,
        receipt.paper_paths_per_session_count_matched_for_calibration
        == paper_path_count,
        receipt.paper_measurement_and_convergence_scale_verified == paper_scale,
        receipt.all_t3_lambdas_positive == positive_lambdas,
        receipt.ready_for_cell_aggregation == (canonical and positive_lambdas),
        receipt.ready_for_formal_paper_cell_aggregation == expected_formal_ready,
        not receipt.paper_1000_sessions_verified,
        not receipt.formal_cross_session_seed_uniqueness_verified,
        not receipt.uniform_cell_shock_calibrated,
        not receipt.ready_for_uniform_shock_calibration,
        not receipt.shock_applied,
        receipt.treatment_paths_executed == 0,
        not receipt.t4_response_aggregated,
        not receipt.classification_ready,
        not receipt.mechanism_label_ready,
        not receipt.paper_1_2_percent_achieved,
        not receipt.full_paper_irf_protocol_verified,
        not receipt.paper_figure_ready,
        not receipt.paper_specifies_numerical_calibration_algorithm,
        receipt.replication_collects_control_t3_level_for_finite_sample_audit,
        receipt.target_normalized_price_deviation
        == PAPER_TARGET_PRICE_DEVIATION,
        receipt.checksum_detects_stale_replacement_not_authentication,
        not receipt.standalone_receipt_authenticates_executed_paths,
    )
    if not all(logical_claims):
        raise ValueError("Receipt claims are inconsistent. / receipt 声明不一致。")


def run_unshocked_t3_calibration_paths(
    checkpoint: ConvergedMarketCheckpoint,
    *,
    baseline_scorer: OnlineIRFLongRunBaselineScorer,
    irf_experiment_seed: int,
    path_count: int = PAPER_PATHS_PER_SESSION,
) -> UnshockedT3SessionCalibrationReceipt:
    """Run one session's canonical calibration pass in constant memory.

    使用固定内存，运行一个 session 的标准校准路径。
    """

    checked_count = _checked_path_count(path_count)
    workspace = ReusableUnshockedT3Workspace(
        checkpoint,
        baseline_scorer=baseline_scorer,
        irf_experiment_seed=irf_experiment_seed,
    )
    moments = OnlineUnshockedT3Moments(
        schedule_context=workspace.schedule_context,
        value_mean=checkpoint.payload.parameters.value_mean,
        pricing_error_weight=(
            checkpoint.payload.parameters.pricing_error_weight
        ),
    )
    for path_index in range(checked_count):
        # The result exists only for this one iteration. / result 只在本次循环存在。
        moments.add(workspace.run_path(path_index))
    workspace.close_and_verify()
    mean_lambda, mean_price, minimum_lambda, nonpositive_count, path_digest = (
        moments.summarized_values()
    )
    baseline = workspace.baseline_receipt
    canonical = moments.count == checked_count
    paper_path_count = canonical and checked_count == PAPER_PATHS_PER_SESSION
    positive_lambdas = (
        mean_lambda > 0.0
        and minimum_lambda > 0.0
        and nonpositive_count == 0
    )
    paper_scale = baseline.paper_scale_thresholds_and_provenance_verified
    receipt = UnshockedT3SessionCalibrationReceipt(
        protocol_version=UNSHOCKED_T3_PROTOCOL_VERSION,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        implementation_tree_sha256=checkpoint.payload.implementation_tree_sha256,
        source_seed_manifest=checkpoint.payload.seed_manifest,
        irf_experiment_seed=workspace.schedule_context.irf_experiment_seed,
        path_seed_derivation_version=workspace.schedule_context.derivation_version,
        paths_requested=checked_count,
        paths_executed=moments.count,
        first_path_index=0,
        last_path_index=moments.count - 1,
        executed_local_periods=CALIBRATION_LOCAL_PERIODS,
        mean_t3_price_impact_lambda=mean_lambda,
        minimum_t3_price_impact_lambda=minimum_lambda,
        nonpositive_t3_lambda_count=nonpositive_count,
        mean_unshocked_t3_oriented_price=mean_price,
        executed_path_fields_sha256=path_digest,
        long_run_baseline_receipt=baseline,
        long_run_mean_oriented_price=baseline.mean_oriented_price,
        baseline_receipt_payload_sha256=baseline.receipt_payload_sha256,
        baseline_scored_fields_sha256=baseline.scored_fields_sha256,
        source_session_full_restores=1,
        successful_transaction_rollbacks=workspace.rollbacks_completed,
        full_checkpoint_reset_audits=2,
        maximum_live_market_branches=1,
        raw_path_results_retained=0,
        planned_within_session_child_seed_uniqueness_verified=(
            workspace.seed_audit.all_planned_child_seeds_are_unique
        ),
        planned_child_stream_count=(
            workspace.seed_audit.planned_child_stream_count
        ),
        canonical_zero_based_path_coverage_verified=canonical,
        constant_memory_online_aggregation_verified=True,
        exact_checkpoint_reset_after_batch_verified=True,
        live_step35c_baseline_provenance_verified=True,
        same_session_checkpoint_and_baseline_verified=True,
        different_samples_from_same_experiment_cell=True,
        paper_paths_per_session_count_matched_for_calibration=paper_path_count,
        paper_measurement_and_convergence_scale_verified=paper_scale,
        all_t3_lambdas_positive=positive_lambdas,
        ready_for_cell_aggregation=(canonical and positive_lambdas),
        ready_for_formal_paper_cell_aggregation=(
            paper_path_count and paper_scale and positive_lambdas
        ),
        paper_1000_sessions_verified=False,
        formal_cross_session_seed_uniqueness_verified=False,
        uniform_cell_shock_calibrated=False,
        ready_for_uniform_shock_calibration=False,
        shock_applied=False,
        treatment_paths_executed=0,
        t4_response_aggregated=False,
        classification_ready=False,
        mechanism_label_ready=False,
        paper_1_2_percent_achieved=False,
        full_paper_irf_protocol_verified=False,
        paper_figure_ready=False,
        paper_specifies_numerical_calibration_algorithm=False,
        replication_collects_control_t3_level_for_finite_sample_audit=True,
        target_normalized_price_deviation=PAPER_TARGET_PRICE_DEVIATION,
        checksum_detects_stale_replacement_not_authentication=True,
        standalone_receipt_authenticates_executed_paths=False,
        receipt_payload_sha256="",
    )
    receipt = replace(receipt, receipt_payload_sha256=_receipt_payload_digest(receipt))
    validate_unshocked_t3_session_calibration_receipt(receipt)
    return receipt


def main() -> None:
    """Run a short visible debug pass; this is not a paper-scale claim.

    运行一个短小可见的调试版本；这不是论文规模结果。
    """

    # Reuse Step 35C's public workflow, but deliberately shorten both samples.
    # / 复用第 35C 步工作流，但故意缩短两种样本。
    from steps.step_35c_irf_long_run_baseline import _build_demo_controller

    controller, scorer = _build_demo_controller()
    while controller.phase is SessionPhase.TRAINING:
        if controller.training_periods_completed >= 5:
            raise TimeoutError("Debug convergence was not reached. / 调试收敛尚未达到。")
        controller.run_next_period()
    checkpoint = scorer.capture_and_bind_convergence_checkpoint(controller)
    controller.run_until_complete()
    scorer.finalize(controller)
    receipt = run_unshocked_t3_calibration_paths(
        checkpoint,
        baseline_scorer=scorer,
        irf_experiment_seed=20_260_835,
        path_count=100,
    )
    print("Step 35D: unshocked t=3 calibration / 第 35D 步：无冲击 t=3 校准")
    print(f"Debug paths executed / 调试路径数: {receipt.paths_executed}")
    print(f"E[lambda_hat_3] / t=3 lambda 均值: {receipt.mean_t3_price_impact_lambda:.12f}")
    print(f"E[p_tilde_3^0] / 无冲击 t=3 方向价格均值: {receipt.mean_unshocked_t3_oriented_price:.9f}")
    print(f"Transaction rollbacks / 事务回滚次数: {receipt.successful_transaction_rollbacks}")
    print("Shock applied / 已施加冲击: False")
    print("t=4 executed / 已执行 t=4: False")
    print("Classification ready / 可以分类: False")
    print("Paper 10,000-path count matched / 符合论文一万条路径: False (debug run / 调试运行)")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
