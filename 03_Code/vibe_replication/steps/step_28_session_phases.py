"""Step 28: separate Q-learning from post-convergence measurement.

步骤 28：严格分开 Q-learning 训练期与收敛后的测量期。

Run / 运行:
    py -3 -X utf8 steps/step_28_session_phases.py

Paper-supported protocol / 原文支持的协议:
    - measure outcomes for T = 100,000 periods after convergence;
      / 收敛后测量 T = 100,000 期；
    - use the learned limit/greedy policies based on Q at T_c;
      / 使用由 T_c 时 Q 表决定的极限/贪心策略；
    - random fundamental values and noise orders continue;
      / 基本价值与噪声订单继续随机产生；
    - the market maker keeps rolling and re-estimating every period.
      / 做市商继续每期滚动更新与重新估计。

Explicit replication choices / 明确复现选择:
    1. The prose says "100,000 periods after convergence," while displayed
       summation bounds appear inclusive at both ends. We measure exactly the
       next 100,000 periods: T_c+1, ..., T_c+100,000.
       / 文字说“收敛后十万期”，但公式边界看起来两端都包含。我们明确采用
       T_c+1 到 T_c+100,000，恰好十万条新记录。
    2. Freeze Q-tables and visit counters, disable exploration, and use the
       frozen greedy policy. The paper defines Q-tilde=Q_(T_c) and limit
       actions but does not literally state the software switch.
       / 冻结 Q 表与访问计数、关闭探索、使用固定贪心策略。论文定义了
       Q-tilde=Q_(T_c) 与极限动作，但没有逐字给出软件切换指令。
    3. If several actions are exactly tied at convergence, choose the lowest
       action index once and keep it fixed. The paper gives no tie rule.
       / 收敛时若精确并列，固定选择最小动作编号；论文没有规定并列规则。

Scope boundary / 本步边界:
    This step emits raw post-convergence observations but calculates no paper
    outcome metric. Matched-path profits and Delta-C begin in Step 29.
    / 本步骤只输出原始测量观测，不计算论文结果指标；相同路径基准利润和
    Delta-C 从第 29 步开始。
"""

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass
from enum import Enum
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
from step_14_state_representation import (
    build_state_indexes,
    continuous_price_to_index,
    encode_state_index,
    number_of_states,
)
from step_25_one_market_period import build_paper_inputs
from step_26_reproducible_random_streams import (
    FrozenPolicyPeriodObservation,
    RandomizedMarketSession,
    build_randomized_paper_session,
)
from step_27_convergence_tracker import (
    ConvergenceReceipt,
    PAPER_UNCHANGED_PERIODS,
    PolicyConvergenceTracker,
)


PAPER_MEASUREMENT_PERIODS = 100_000
EXACT_TIE_RULE = "lowest_action_index_once_at_convergence"


class SessionPhase(str, Enum):
    """Three success stages plus one terminal failure state.

    三个正常阶段，加上一个终止失败状态；Enum 可以防止拼写错误。
    """

    TRAINING = "training"
    MEASUREMENT = "measurement"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class SessionPhaseReceipt:
    """One immutable summary created after the last measurement period.

    最后一条测量完成后生成的一份不可修改总结。
    """

    convergence_receipt: ConvergenceReceipt
    measurement_periods_required: int
    measurement_first_period_index: int
    measurement_last_period_index: int
    measurement_periods_completed: int
    total_session_periods_completed: int
    q_learning_disabled: bool
    exploration_disabled: bool
    market_maker_remained_adaptive: bool
    exact_tie_rule: str


MeasurementSink = Callable[[int, FrozenPolicyPeriodObservation], None]


def _positive_integer(number: int, label: str) -> int:
    """Validate a non-Boolean positive integer. / 检查非布尔正整数。"""

    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise ValueError(f"{label} must be a positive integer. / {label} 必须是正整数。")
    return number


def _validate_measurement_sink_for_session(
    session: RandomizedMarketSession,
    measurement_sink: MeasurementSink | None,
) -> None:
    """Validate an optional sink without advancing the session.

    在不推进 session 的情况下检查可选 measurement sink。
    """

    if measurement_sink is not None and not callable(measurement_sink):
        raise TypeError("measurement_sink must be callable or None. / measurement_sink 必须可调用或为 None。")
    if measurement_sink is None:
        return
    sink_owner = getattr(measurement_sink, "__self__", None)
    sink_session = getattr(sink_owner, "_session", None)
    if sink_session is None:
        sink_session = getattr(measurement_sink, "_measurement_session", None)
    if sink_session is not None:
        if not isinstance(sink_session, RandomizedMarketSession):
            raise TypeError("The sink's session declaration is invalid. / sink 的 session 声明无效。")
        if sink_session is not session:
            raise ValueError("measurement_sink belongs to another session. / measurement_sink 属于另一个 session。")


class SessionPhaseController:
    """Own the TRAINING -> MEASUREMENT -> COMPLETE transition.

    负责 TRAINING → MEASUREMENT → COMPLETE 的唯一阶段切换。

    The controller stores counters and one final receipt, not 100,000 rows.
    A later step may attach an online ``measurement_sink`` to aggregate each
    temporary observation in constant memory. / controller 只保存计数器和一份
    最终总结，不保存十万行。后续步骤可连接在线 measurement_sink，以固定内存
    逐条汇总临时观测。
    """

    def __init__(
        self,
        *,
        session: RandomizedMarketSession,
        tracker: PolicyConvergenceTracker,
        controller_token: object,
        measurement_periods_required: int,
        measurement_sink: MeasurementSink | None,
    ) -> None:
        """Internal constructor; use ``create_for_fresh_session``.

        内部构造函数；请使用 create_for_fresh_session。
        """

        self.session = session
        self.tracker = tracker
        self._controller_token = controller_token
        self.measurement_periods_required = measurement_periods_required
        # The registered sink is causal provenance. It cannot be swapped after
        # the controller starts. / 注册的 sink 属于因果来源；controller 启动后
        # 不能再被替换。
        self.__measurement_sink = measurement_sink
        self.phase = SessionPhase.TRAINING
        self.measurement_periods_completed = 0
        self.measurement_first_period_index: int | None = None
        self.final_receipt: SessionPhaseReceipt | None = None
        self.failure_period_index: int | None = None
        self._counts_at_measurement_start: tuple[int, ...] | None = None
        self._trader_rng_at_measurement_start: tuple[object, ...] | None = None
        self._maker_appends_at_measurement_start: int | None = None

    @property
    def measurement_sink(self) -> MeasurementSink | None:
        """Read the immutable lifetime measurement sink.

        读取 controller 生命周期内不可替换的测量 sink。
        """

        return self.__measurement_sink

    @classmethod
    def create_for_fresh_session(
        cls,
        session: RandomizedMarketSession,
        *,
        convergence_periods_required: int = PAPER_UNCHANGED_PERIODS,
        measurement_periods_required: int = PAPER_MEASUREMENT_PERIODS,
        measurement_sink: MeasurementSink | None = None,
    ) -> "SessionPhaseController":
        """Validate everything before attaching the convergence observer.

        在连接收敛 observer 之前完成全部检查，避免错误输入改变 session。
        """

        convergence_periods_required = _positive_integer(
            convergence_periods_required,
            "convergence_periods_required",
        )
        measurement_periods_required = _positive_integer(
            measurement_periods_required,
            "measurement_periods_required",
        )
        if not isinstance(session, RandomizedMarketSession):
            raise TypeError("session must be a RandomizedMarketSession. / session 类型错误。")
        _validate_measurement_sink_for_session(session, measurement_sink)
        if session.period_number != 0:
            raise RuntimeError("The phase controller requires a fresh period-0 session. / 阶段控制器要求尚未运行的第 0 期 session。")
        if session.execution_mode != "training":
            raise RuntimeError("The fresh session must be in training mode. / 新 session 必须处于训练模式。")
        if session.after_q_update_observer is not None:
            raise RuntimeError("The fresh session already has an observer. / 新 session 已经连接了 observer。")

        tracker = PolicyConvergenceTracker.from_traders(
            session.traders,
            required_unchanged_periods=convergence_periods_required,
        )
        tracker.attach_to_session(session)
        controller_token = object()
        session.claim_phase_controller(controller_token)
        return cls(
            session=session,
            tracker=tracker,
            controller_token=controller_token,
            measurement_periods_required=measurement_periods_required,
            measurement_sink=measurement_sink,
        )

    @classmethod
    def create_for_restored_training_session(
        cls,
        session: RandomizedMarketSession,
        tracker: PolicyConvergenceTracker,
        *,
        measurement_periods_required: int,
        measurement_sink: MeasurementSink | None = None,
    ) -> "SessionPhaseController":
        """Reconnect new runtime objects before installing a saved period.

        在安装保存时期之前，把新建的运行对象重新连接起来。

        Tokens and bound callbacks are deliberately rebuilt, never loaded from
        disk. / token 与绑定 callback 会重新建立，绝不从磁盘读取。
        """

        measurement_periods_required = _positive_integer(
            measurement_periods_required,
            "measurement_periods_required",
        )
        if not isinstance(session, RandomizedMarketSession):
            raise TypeError("session must be a RandomizedMarketSession. / session 类型错误。")
        if not isinstance(tracker, PolicyConvergenceTracker):
            raise TypeError("tracker must be PolicyConvergenceTracker. / tracker 类型错误。")
        _validate_measurement_sink_for_session(session, measurement_sink)
        if session.period_number != 0 or session.execution_mode != "training":
            raise RuntimeError("Restored wiring must begin on a temporary period-0 session. / 恢复连接必须从临时第 0 期 session 开始。")
        if session.after_q_update_observer is not None:
            raise RuntimeError("The temporary session already has an observer. / 临时 session 已有 observer。")
        if tracker.converged or tracker.periods_observed < 1:
            raise RuntimeError("A restored tracker must be mid-training and non-empty. / 恢复 tracker 必须处于非空、未收敛训练中。")

        tracker.attach_to_session(session)
        controller_token = object()
        session.claim_phase_controller(controller_token)
        return cls(
            session=session,
            tracker=tracker,
            controller_token=controller_token,
            measurement_periods_required=measurement_periods_required,
            measurement_sink=measurement_sink,
        )

    def install_restored_training_position(
        self,
        *,
        period_number: int,
        previous_price: float,
        previous_value: float,
        current_value: float,
        all_seven_rng_states: tuple[object, ...],
    ) -> None:
        """Finish a one-time restore and verify controller/tracker alignment.

        完成一次性恢复，并核对 controller 与 tracker 的时期对齐。
        """

        if self.phase is not SessionPhase.TRAINING or self.session.period_number != 0:
            raise RuntimeError("The restored position can be installed only once. / 恢复位置只能安装一次。")
        if period_number != self.tracker.periods_observed:
            raise ValueError("Session period differs from restored tracker history. / session 时期与恢复 tracker 历史不同。")
        self.session.install_restored_training_position(
            period_number=period_number,
            previous_price=previous_price,
            previous_value=previous_value,
            current_value=current_value,
            all_seven_rng_states=all_seven_rng_states,
            controller_token=self._controller_token,
        )
        if self.training_periods_completed != self.session.period_number:
            raise RuntimeError("Restored training counters are misaligned. / 恢复后的训练计数没有对齐。")

    @property
    def training_periods_completed(self) -> int:
        """Number of completed Q-learning periods. / 已完成的 Q-learning 时期数。"""

        return self.tracker.periods_observed

    def _begin_measurement_after_convergence(self) -> None:
        """Perform the boundary switch only after the training call returns.

        只有收敛训练时期完整返回之后，才执行阶段切换。
        """

        if not self.tracker.converged:
            raise RuntimeError("Cannot measure before convergence. / 收敛前不能测量。")
        if self.tracker.convergence_receipt is None:
            raise RuntimeError("Convergence receipt is missing. / 收敛记录丢失。")
        if self.tracker.converged_policy_masks is None:
            raise RuntimeError("Converged policy snapshot is missing. / 收敛策略快照丢失。")
        expected_first_period = (
            self.tracker.convergence_receipt.convergence_period_index + 1
        )
        if self.session.period_number != expected_first_period:
            raise RuntimeError("Training and measurement boundary is inconsistent. / 训练与测量边界不一致。")

        self.session.begin_frozen_greedy_measurement(
            self.tracker.converged_policy_masks,
            controller_token=self._controller_token,
        )
        self.measurement_first_period_index = self.session.period_number
        self._counts_at_measurement_start = tuple(
            self.session.shared_value_visit_counts
        )
        self._trader_rng_at_measurement_start = tuple(
            state
            for trader in self.session.traders
            for state in (
                trader.mode_random_generator.getstate(),
                trader.action_random_generator.getstate(),
            )
        )
        self._maker_appends_at_measurement_start = (
            self.session.market_maker.successful_append_count
        )
        self.phase = SessionPhase.MEASUREMENT

    def _verify_frozen_measurement_invariants(self) -> None:
        """Prove the final receipt's three behavioral claims from live state.

        根据实时状态证明最终总结中的三项行为声明，而不是只写死布尔值。
        """

        if self._counts_at_measurement_start is None:
            raise RuntimeError("Frozen visit counts are missing. / 冻结访问计数丢失。")
        if self._trader_rng_at_measurement_start is None:
            raise RuntimeError("Frozen trader RNG states are missing. / 冻结 trader RNG 状态丢失。")
        if self._maker_appends_at_measurement_start is None:
            raise RuntimeError("Maker append baseline is missing. / 做市商追加基线丢失。")
        if tuple(self.session.shared_value_visit_counts) != self._counts_at_measurement_start:
            raise RuntimeError("Visit counters changed during measurement. / 测量期访问计数发生变化。")
        current_trader_rng = tuple(
            state
            for trader in self.session.traders
            for state in (
                trader.mode_random_generator.getstate(),
                trader.action_random_generator.getstate(),
            )
        )
        if current_trader_rng != self._trader_rng_at_measurement_start:
            raise RuntimeError("A trader RNG moved during measurement. / 测量期 trader 随机流发生推进。")
        if any(trader.q_table.flags.writeable for trader in self.session.traders):
            raise RuntimeError("A frozen Q-table became writable. / 冻结 Q 表重新变成可写。")
        expected_maker_appends = (
            self._maker_appends_at_measurement_start
            + self.measurement_periods_required
        )
        if (
            self.session.market_maker.successful_append_count
            != expected_maker_appends
        ):
            raise RuntimeError("The market maker did not append once per measurement period. / 做市商没有在每个测量期恰好追加一次。")

    def run_next_period(
        self,
    ) -> FrozenPolicyPeriodObservation | None:
        """Run exactly one period in the controller's current phase.

        根据 controller 当前阶段，准确运行一个时期。

        Returns None during training and one temporary raw observation during
        measurement. / 训练期返回 None；测量期返回一条临时原始观测。
        """

        if self.phase in (SessionPhase.COMPLETE, SessionPhase.FAILED):
            raise RuntimeError("This session is complete or failed. / 此 session 已完成或失败。")

        if self.phase is SessionPhase.TRAINING:
            if self.tracker.converged:
                self._begin_measurement_after_convergence()
                return self.run_next_period()
            self.session.run_next_training_period_for_controller(
                self._controller_token
            )
            if self.tracker.converged:
                self._begin_measurement_after_convergence()
            return None

        if self.measurement_first_period_index is None:
            raise RuntimeError("The first measurement index is missing. / 首个测量期编号丢失。")
        measurement_index = self.measurement_periods_completed
        expected_global_period = (
            self.measurement_first_period_index + measurement_index
        )
        if self.session.period_number != expected_global_period:
            raise RuntimeError("Measurement periods are not consecutive. / 测量时期不连续。")

        observation = self.session.run_next_measurement_period_for_controller(
            self._controller_token
        )
        if observation.period_number != expected_global_period:
            raise RuntimeError("Measurement observation has the wrong period. / 测量观测的时期编号错误。")

        # The sink must accept the row before the controller can claim success.
        # If it fails, the whole seeded session is marked failed and should be
        # restarted; no false completion receipt is created. / sink 必须先成功
        # 接收本行，controller 才能声称成功。若失败，整个种子 session 标记失败，
        # 应重新运行；不会生成虚假的完成总结。
        if self.measurement_sink is not None:
            try:
                self.measurement_sink(measurement_index, observation)
            except Exception:
                self.failure_period_index = observation.period_number
                self.phase = SessionPhase.FAILED
                raise

        self.measurement_periods_completed += 1
        if (
            self.measurement_periods_completed
            == self.measurement_periods_required
        ):
            try:
                self._verify_frozen_measurement_invariants()
                self.session.finish_frozen_greedy_measurement(
                    controller_token=self._controller_token
                )
                if self.tracker.convergence_receipt is None:
                    raise RuntimeError("Convergence receipt is missing. / 收敛记录丢失。")
                completed_receipt = SessionPhaseReceipt(
                    convergence_receipt=self.tracker.convergence_receipt,
                    measurement_periods_required=(
                        self.measurement_periods_required
                    ),
                    measurement_first_period_index=(
                        self.measurement_first_period_index
                    ),
                    measurement_last_period_index=observation.period_number,
                    measurement_periods_completed=(
                        self.measurement_periods_completed
                    ),
                    total_session_periods_completed=(
                        self.session.period_number
                    ),
                    q_learning_disabled=True,
                    exploration_disabled=True,
                    market_maker_remained_adaptive=True,
                    exact_tie_rule=EXACT_TIE_RULE,
                )
            except Exception:
                self.failure_period_index = observation.period_number
                self.phase = SessionPhase.FAILED
                self.final_receipt = None
                raise
            self.final_receipt = completed_receipt
            self.phase = SessionPhase.COMPLETE

        return observation

    def run_until_complete(
        self,
        *,
        maximum_training_periods: int | None = None,
    ) -> SessionPhaseReceipt:
        """Run one session, optionally imposing a test/debug training cap.

        运行一个完整 session；测试或调试时可设置最大训练期数。

        A cap is not a convergence substitute: reaching it raises TimeoutError.
        / 上限不能替代收敛；达到上限但尚未收敛时会报 TimeoutError。
        """

        if maximum_training_periods is not None:
            maximum_training_periods = _positive_integer(
                maximum_training_periods,
                "maximum_training_periods",
            )
        if (
            self.phase is not SessionPhase.COMPLETE
            and self.measurement_sink is None
        ):
            raise RuntimeError(
                "run_until_complete requires a measurement sink so formal "
                "observations are not discarded. / run_until_complete 要求连接 "
                "measurement sink，避免正式观测被丢弃。"
            )
        while self.phase is SessionPhase.TRAINING:
            if (
                maximum_training_periods is not None
                and self.training_periods_completed
                >= maximum_training_periods
            ):
                raise TimeoutError("The debug training cap was reached before convergence. / 达到调试训练上限但尚未收敛。")
            self.run_next_period()
        while self.phase is SessionPhase.MEASUREMENT:
            self.run_next_period()
        if self.phase is SessionPhase.FAILED:
            raise RuntimeError(
                "The measurement sink failed; restart this seeded session. / "
                "测量 sink 失败；请重新运行此种子 session。"
            )
        if self.final_receipt is None:
            raise RuntimeError("Final phase receipt is missing. / 最终阶段总结丢失。")
        return self.final_receipt


def _build_test_session(
    *,
    parameters: PaperParameters,
    value_grid: tuple[float, ...],
    price_grid: tuple[tuple[float, ...], ...],
    action_multipliers: tuple[float, ...],
    stable_q_table: np.ndarray,
    prehistory: object,
    session_index: int,
) -> RandomizedMarketSession:
    """Create a same-design test session with a distinct session seed.

    使用不同 session 种子建立同一设计的测试 session。
    """

    return build_randomized_paper_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        initial_q_table=stable_q_table,
        prehistory=prehistory,
        experiment_seed=20260828,
        experiment_cell_key="step28_low_noise|A3=nash",
        session_index=session_index,
    )


def _trader_random_states(
    session: RandomizedMarketSession,
) -> tuple[object, ...]:
    """Return only the four agent RNG states. / 只返回四条 agent 随机流状态。"""

    return tuple(
        state
        for trader in session.traders
        for state in (
            trader.mode_random_generator.getstate(),
            trader.action_random_generator.getstate(),
        )
    )


def main() -> None:
    """Validate a K=2, T=3 phase path and all important boundaries.

    使用 K=2、T=3 的小路径验证全部重要阶段边界。
    """

    parameters = PaperParameters()
    (
        value_grid,
        price_grid,
        action_multipliers,
        paper_initial_q_table,
        prehistory,
    ) = build_paper_inputs(parameters)

    # Make action 0 uniquely best in every state by a huge safe margin. Random
    # exploration may update another action, but cannot change the greedy rule
    # in this tiny phase test. / 让动作 0 在所有状态以巨大安全差距成为唯一最优。
    # 随机探索可以更新别的动作，但不会改变本测试中的贪心规则。
    stable_q_table = np.zeros_like(paper_initial_q_table, dtype=float)
    stable_q_table[:, 0] = 1_000_000_000.0

    session = _build_test_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        stable_q_table=stable_q_table,
        prehistory=prehistory,
        session_index=0,
    )

    # Invalid construction must fail before attaching an observer or consuming
    # randomness. / 无效构造必须在连接 observer 或消耗随机数之前失败。
    untouched_random_states = session.all_random_states()
    try:
        SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=2,
            measurement_periods_required=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("A zero measurement target should fail. / 测量目标为零应失败。")
    assert session.period_number == 0
    assert session.after_q_update_observer is None
    assert session.all_random_states() == untouched_random_states

    sink_calls: list[tuple[int, int]] = []

    def tiny_test_sink(
        measurement_index: int,
        observation: FrozenPolicyPeriodObservation,
    ) -> None:
        """Store only three indexes in this toy test. / 玩具测试仅保存三个编号。"""

        sink_calls.append((measurement_index, observation.period_number))

    controller = SessionPhaseController.create_for_fresh_session(
        session,
        convergence_periods_required=2,
        measurement_periods_required=3,
        measurement_sink=tiny_test_sink,
    )

    # Measurement cannot be called while the session is training. / 训练期不能直接调用测量。
    premature_random_states = session.all_random_states()
    try:
        session.run_next_frozen_policy_period()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Premature measurement should fail. / 提前测量应失败。")
    assert session.all_random_states() == premature_random_states
    assert session.period_number == 0
    try:
        session.run_next_random_period_without_trace()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Direct training should not bypass the controller. / 直接训练不能绕过 controller。")
    assert session.all_random_states() == premature_random_states
    assert session.period_number == 0

    # Training periods 0 and 1. The convergence period is training only.
    # / 训练时期为 0 和 1；达到收敛的时期仍只属于训练。
    assert controller.run_next_period() is None
    assert controller.phase is SessionPhase.TRAINING
    assert controller.tracker.unchanged_periods == 1
    assert sink_calls == []
    assert controller.run_next_period() is None
    assert controller.phase is SessionPhase.MEASUREMENT
    assert controller.tracker.convergence_receipt is not None
    assert controller.tracker.convergence_receipt.convergence_period_index == 1
    assert session.period_number == 2
    assert sink_calls == []

    # Save the exact boundary state. Nothing is reset at convergence. / 保存精确边界状态；收敛时不重置。
    boundary_state_values = (
        session.previous_price,
        session.previous_value,
        session.current_value,
    )
    boundary_state_indexes = build_state_indexes(
        *boundary_state_values,
        session.price_grid,
        session.value_grid,
    )
    q_at_convergence = tuple(
        trader.q_table.copy()
        for trader in session.traders
    )
    counts_at_convergence = tuple(session.shared_value_visit_counts)
    trader_rng_at_convergence = _trader_random_states(session)
    maker_appends_at_convergence = (
        session.market_maker.successful_append_count
    )
    maker_oldest_at_convergence = session.market_maker.snapshot()[0]

    # Independently advance cloned value/noise generators exactly three draws.
    # / 独立复制价值与噪声随机流，并各自准确推进三次。
    expected_value_generator = random.Random()
    expected_value_generator.setstate(
        session.streams.value_generator.getstate()
    )
    expected_noise_generator = random.Random()
    expected_noise_generator.setstate(
        session.streams.noise_generator.getstate()
    )
    expected_noise_orders = tuple(
        expected_noise_generator.gauss(0.0, parameters.noise_std)
        for _ in range(3)
    )
    expected_next_value_indexes = tuple(
        expected_value_generator.randrange(len(value_grid))
        for _ in range(3)
    )

    # A training call after transition is rejected before mutation. / 切换后再调用训练会在变动前被拒绝。
    transition_random_states = session.all_random_states()
    try:
        session.run_next_random_period_without_trace()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Training after transition should fail. / 切换后训练应失败。")
    assert session.all_random_states() == transition_random_states
    assert session.period_number == 2

    observations = (
        controller.run_next_period(),
        controller.run_next_period(),
        controller.run_next_period(),
    )
    assert all(
        isinstance(observation, FrozenPolicyPeriodObservation)
        for observation in observations
    )
    measurement_observations = tuple(
        observation
        for observation in observations
        if isinstance(observation, FrozenPolicyPeriodObservation)
    )
    assert len(measurement_observations) == 3
    first_observation = observations[0]
    if not isinstance(first_observation, FrozenPolicyPeriodObservation):
        raise RuntimeError("First observation is missing. / 第一条观测丢失。")

    assert first_observation.period_number == 2
    assert first_observation.current_state_indexes == boundary_state_indexes
    assert first_observation.fundamental_value_v == boundary_state_values[2]
    assert tuple(
        observation.noise_order_u
        for observation in measurement_observations
    ) == expected_noise_orders
    assert tuple(
        observation.next_value_index
        for observation in measurement_observations
    ) == expected_next_value_indexes
    assert all(
        right.current_state_indexes == left.next_state_indexes
        for left, right in zip(
            measurement_observations,
            measurement_observations[1:],
        )
    )
    for observation in measurement_observations:
        current_value_price_row = price_grid[observation.current_value_index]
        assert observation.next_state_indexes[0] == continuous_price_to_index(
            observation.continuous_price_p,
            current_value_price_row,
        )
        assert observation.next_price_was_clipped == (
            observation.continuous_price_p < current_value_price_row[0]
            or observation.continuous_price_p > current_value_price_row[-1]
        )
    assert sink_calls == [(0, 2), (1, 3), (2, 4)]
    assert controller.phase is SessionPhase.COMPLETE
    assert session.execution_mode == "complete"
    assert session.period_number == 5

    # Q, visits, and agent RNGs freeze; environment RNGs and maker continue.
    # / Q、访问计数与 agent RNG 冻结；环境 RNG 与做市商继续。
    assert all(
        np.array_equal(trader.q_table, frozen_q)
        for trader, frozen_q in zip(
            session.traders,
            q_at_convergence,
            strict=True,
        )
    )
    assert all(not trader.q_table.flags.writeable for trader in session.traders)
    assert tuple(session.shared_value_visit_counts) == counts_at_convergence
    assert _trader_random_states(session) == trader_rng_at_convergence
    assert (
        session.streams.value_generator.getstate()
        == expected_value_generator.getstate()
    )
    assert (
        session.streams.noise_generator.getstate()
        == expected_noise_generator.getstate()
    )
    assert (
        session.market_maker.successful_append_count
        == maker_appends_at_convergence + 3
    )
    assert session.market_maker.snapshot()[0] != maker_oldest_at_convergence
    assert not hasattr(controller, "measurement_history")

    expected_receipt = SessionPhaseReceipt(
        convergence_receipt=controller.tracker.convergence_receipt,
        measurement_periods_required=3,
        measurement_first_period_index=2,
        measurement_last_period_index=4,
        measurement_periods_completed=3,
        total_session_periods_completed=5,
        q_learning_disabled=True,
        exploration_disabled=True,
        market_maker_remained_adaptive=True,
        exact_tie_rule=EXACT_TIE_RULE,
    )
    assert controller.final_receipt == expected_receipt
    try:
        controller.final_receipt.measurement_periods_completed = 99  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("The final receipt must be frozen. / 最终总结必须不可修改。")

    # An extra call after COMPLETE must change absolutely nothing. / 完成后的额外调用不能改变任何内容。
    final_random_states = session.all_random_states()
    final_market_history = session.market_maker.snapshot()
    final_q_tables = tuple(trader.q_table.copy() for trader in session.traders)
    final_counts = tuple(session.shared_value_visit_counts)
    final_state = (
        session.period_number,
        session.previous_price,
        session.previous_value,
        session.current_value,
    )
    try:
        controller.run_next_period()
    except RuntimeError:
        pass
    else:
        raise AssertionError("A completed controller should reject extra work. / 已完成 controller 应拒绝额外运行。")
    assert session.all_random_states() == final_random_states
    assert session.market_maker.snapshot() == final_market_history
    assert tuple(session.shared_value_visit_counts) == final_counts
    assert (
        session.period_number,
        session.previous_price,
        session.previous_value,
        session.current_value,
    ) == final_state
    assert all(
        np.array_equal(trader.q_table, final_q)
        for trader, final_q in zip(
            session.traders,
            final_q_tables,
            strict=True,
        )
    )

    # K=1 boundary: convergence at 0, then first measurement at 1. / K=1 边界：在 0 收敛，测量从 1 开始。
    k1_session = _build_test_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        stable_q_table=stable_q_table,
        prehistory=prehistory,
        session_index=1,
    )
    k1_controller = SessionPhaseController.create_for_fresh_session(
        k1_session,
        convergence_periods_required=1,
        measurement_periods_required=1,
    )
    k1_random_before_missing_sink = k1_session.all_random_states()
    try:
        k1_controller.run_until_complete(maximum_training_periods=2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Bulk running without a sink should fail. / 没有 sink 的整段运行应失败。")
    assert k1_session.period_number == 0
    assert k1_session.all_random_states() == k1_random_before_missing_sink
    assert k1_controller.run_next_period() is None
    assert k1_controller.phase is SessionPhase.MEASUREMENT
    assert k1_session.period_number == 1
    k1_observation = k1_controller.run_next_period()
    assert isinstance(k1_observation, FrozenPolicyPeriodObservation)
    assert k1_observation.period_number == 1
    assert k1_controller.final_receipt is not None
    assert k1_controller.final_receipt.measurement_first_period_index == 1

    # A shape-valid but Q-inconsistent mask fails late in the full scan without
    # freezing or otherwise mutating the session. / 形状正确但与 Q 不一致的 mask
    # 即使在完整扫描末尾才发现，也不能冻结或改变 session。
    corrupt_session = _build_test_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        stable_q_table=stable_q_table,
        prehistory=prehistory,
        session_index=3,
    )
    corrupt_session.run_next_random_period_without_trace()
    corrupt_masks = np.ones(
        (
            len(corrupt_session.traders),
            number_of_states(
                corrupt_session.number_of_prices,
                len(corrupt_session.value_grid),
            ),
        ),
        dtype=np.uint64,
    )
    corrupt_masks[-1, -1] = 1 << 1
    corrupt_random_before = corrupt_session.all_random_states()
    corrupt_market_before = corrupt_session.market_maker.snapshot()
    corrupt_state_before = (
        corrupt_session.period_number,
        corrupt_session.previous_price,
        corrupt_session.previous_value,
        corrupt_session.current_value,
    )
    try:
        corrupt_session.begin_frozen_greedy_measurement(corrupt_masks)
    except ValueError:
        pass
    else:
        raise AssertionError("A stale policy mask should fail. / 过期策略 mask 应失败。")
    assert corrupt_session.execution_mode == "training"
    assert all(trader.q_table.flags.writeable for trader in corrupt_session.traders)
    assert corrupt_session.all_random_states() == corrupt_random_before
    assert corrupt_session.market_maker.snapshot() == corrupt_market_before
    assert (
        corrupt_session.period_number,
        corrupt_session.previous_price,
        corrupt_session.previous_value,
        corrupt_session.current_value,
    ) == corrupt_state_before

    # A failing sink invalidates the seeded run and never creates a success
    # receipt, even on the last planned row. / sink 失败会使本种子运行失效，
    # 即使发生在最后一条也绝不会生成成功总结。
    failed_sink_calls: list[tuple[int, int]] = []

    def deliberately_failing_sink(
        measurement_index: int,
        observation: FrozenPolicyPeriodObservation,
    ) -> None:
        failed_sink_calls.append((measurement_index, observation.period_number))
        raise RuntimeError("deliberate sink failure / 故意的 sink 失败")

    failed_session = _build_test_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        stable_q_table=stable_q_table,
        prehistory=prehistory,
        session_index=4,
    )
    failed_controller = SessionPhaseController.create_for_fresh_session(
        failed_session,
        convergence_periods_required=1,
        measurement_periods_required=1,
        measurement_sink=deliberately_failing_sink,
    )
    failed_controller.run_next_period()
    try:
        failed_controller.run_next_period()
    except RuntimeError:
        pass
    else:
        raise AssertionError("The deliberate sink failure should surface. / 故意 sink 失败应被抛出。")
    assert failed_sink_calls == [(0, 1)]
    assert failed_controller.phase is SessionPhase.FAILED
    assert failed_controller.measurement_periods_completed == 0
    assert failed_controller.final_receipt is None
    assert failed_controller.failure_period_index == 1
    failed_random_after_error = failed_session.all_random_states()
    try:
        failed_controller.run_next_period()
    except RuntimeError:
        pass
    else:
        raise AssertionError("A failed controller must stay terminal. / 失败 controller 必须保持终止。")
    assert failed_session.all_random_states() == failed_random_after_error

    # A final invariant failure after the sink accepts row T is also terminal;
    # it must never permit an accidental T+1 row. / sink 接受第 T 行后若最终
    # 不变量失败，也必须终止，绝不能允许意外的第 T+1 行。
    invariant_session = _build_test_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        stable_q_table=stable_q_table,
        prehistory=prehistory,
        session_index=7,
    )
    invariant_sink_calls: list[int] = []
    invariant_controller = SessionPhaseController.create_for_fresh_session(
        invariant_session,
        convergence_periods_required=1,
        measurement_periods_required=1,
        measurement_sink=(
            lambda index, observation: invariant_sink_calls.append(index)
        ),
    )
    invariant_controller.run_next_period()
    invariant_session.shared_value_visit_counts[0] += 1  # deliberate tampering / 故意篡改
    try:
        invariant_controller.run_next_period()
    except RuntimeError:
        pass
    else:
        raise AssertionError("The tampered final invariant should fail. / 被篡改的最终不变量应失败。")
    assert invariant_sink_calls == [0]
    assert invariant_controller.phase is SessionPhase.FAILED
    assert invariant_controller.measurement_periods_completed == 1
    assert invariant_controller.final_receipt is None
    assert invariant_session.period_number == 2
    invariant_random_after_failure = invariant_session.all_random_states()
    try:
        invariant_controller.run_next_period()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Finalization failure must stay terminal. / 收尾失败必须保持终止。")
    assert invariant_session.period_number == 2
    assert invariant_session.all_random_states() == invariant_random_after_failure

    # A debug training cap raises rather than pretending to be convergence.
    # / 调试训练上限只会报错，不能冒充收敛。
    capped_session = _build_test_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        stable_q_table=stable_q_table,
        prehistory=prehistory,
        session_index=5,
    )
    capped_sink_calls: list[int] = []
    capped_controller = SessionPhaseController.create_for_fresh_session(
        capped_session,
        convergence_periods_required=3,
        measurement_periods_required=1,
        measurement_sink=lambda index, observation: capped_sink_calls.append(index),
    )
    try:
        capped_controller.run_until_complete(maximum_training_periods=2)
    except TimeoutError:
        pass
    else:
        raise AssertionError("The debug cap should raise before convergence. / 调试上限应在收敛前报错。")
    assert capped_controller.phase is SessionPhase.TRAINING
    assert capped_controller.training_periods_completed == 2
    assert capped_session.period_number == 2
    assert capped_sink_calls == []

    # Exact-tie rule: choose the lower index once, then keep it fixed.
    # / 精确并列规则：一次性选择较小编号，之后固定。
    tie_session = _build_test_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        stable_q_table=stable_q_table,
        prehistory=prehistory,
        session_index=6,
    )
    tie_session.run_next_random_period_without_trace()
    tie_state_indexes = build_state_indexes(
        tie_session.previous_price,
        tie_session.previous_value,
        tie_session.current_value,
        tie_session.price_grid,
        tie_session.value_grid,
    )
    tie_state_id = encode_state_index(
        tie_state_indexes,
        tie_session.number_of_prices,
        len(tie_session.value_grid),
    )
    tie_masks = np.ones(
        (
            len(tie_session.traders),
            number_of_states(
                tie_session.number_of_prices,
                len(tie_session.value_grid),
            ),
        ),
        dtype=np.uint64,
    )
    tie_session.traders[0].q_table[tie_state_id, :] = 0.0
    tie_session.traders[0].q_table[tie_state_id, (3, 7)] = 1.0
    tie_session.traders[1].q_table[tie_state_id, :] = 0.0
    tie_session.traders[1].q_table[tie_state_id, (2, 8)] = 1.0
    tie_masks[0, tie_state_id] = (1 << 3) | (1 << 7)
    tie_masks[1, tie_state_id] = (1 << 2) | (1 << 8)
    tie_session.begin_frozen_greedy_measurement(tie_masks)
    tie_observation = tie_session.run_next_frozen_policy_period()
    assert tie_observation.action_indexes == (3, 2)

    print("Step 28: Session phases / 步骤 28：session 阶段")
    print(f"Paper measurement target / 论文测量目标: {PAPER_MEASUREMENT_PERIODS:,}")
    print("Toy training periods / 玩具训练时期: [0, 1]")
    print("Toy measurement periods / 玩具测量时期: [2, 3, 4]")
    print("Toy phase path / 玩具阶段路径: TRAINING -> MEASUREMENT -> COMPLETE")
    print("Convergence period counted as measurement / 收敛当期计入测量: no / 否 (explicit replication choice / 明确复现选择)")
    print("Q, exploration, visit counters / Q、探索、访问计数: frozen / 冻结 (explicit replication choice / 明确复现选择)")
    print("Value and noise / 价值与噪声: continued / 继续 (paper rule / 论文规则)")
    print("Rolling OLS / 滚动 OLS: continued / 继续 (paper-supported inference / 原文支持的推断)")
    print(f"Exact-tie rule / 精确并列规则: {EXACT_TIE_RULE} (replication choice / 复现选择)")
    print("Internal measurement history retained / 内部保存逐期测量历史: no / 否")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
