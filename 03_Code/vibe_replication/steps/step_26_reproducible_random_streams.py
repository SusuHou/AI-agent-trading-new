"""Step 26: reproducible, independent random streams and a short live session.

步骤 26：建立可复现、相互独立的随机流，并运行一个短期真实随机 session。

Run / 运行:
    py -3 -X utf8 steps/step_26_reproducible_random_streams.py

Paper rules / 论文规则:
    - v_(t+1) is drawn uniformly from the ten equal-probability V points.
      / 下一价值从十个等概率价值点中抽取。
    - u_t ~ N(0, sigma_u^2). / 噪声订单服从连续正态分布。
    - the two agents randomize independently. / 两位 agent 独立随机化。
    - simulation sessions are independent. / 不同 session 相互独立。

Replication choice A4 / 复现选择 A4:
The paper gives no numerical seeds or RNG-allocation rule. We derive stable,
named child seeds from one experiment seed and session number. Never use
Python's hash(), the clock, the process ID, or parallel completion order.

论文没有公布种子或随机流分配方法。我们从一个实验种子和 session 编号稳定地产生
命名子种子；不使用 Python hash()、系统时间、进程编号或并行完成顺序。

Seven owned streams / 七条各有归属的随机流:
    initial_state
    fundamental_value
    noise_order
    trader_1_mode
    trader_1_action_tie
    trader_2_mode
    trader_2_action_tie

Mode and action/tie are separate because an extra exploratory action or exact
Q tie must not shift that trader's future explore-versus-exploit draws. / 模式
抽签与动作/并列抽签分开，避免一次额外动作抽签改变未来的探索/利用随机序列。
"""

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from numbers import Real
from pathlib import Path
import platform
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
from step_03_total_order_flow import calculate_total_order_flow
from step_04_information_insensitive_investors import (
    calculate_insensitive_order,
)
from step_05_speculator_profit import calculate_profit
from step_12_action_grid import calculate_orders_for_value
from step_14_state_representation import (
    build_state_indexes,
    continuous_price_to_index,
    encode_state_index,
    fundamental_value_to_index,
    number_of_price_points,
    number_of_states,
    validate_price_grids_by_value,
)
from step_15_initial_state import draw_initial_state_indexes
from step_18_epsilon_greedy_action import ActionDecision
from step_19_value_specific_epsilon import initialize_value_visit_counts
from step_20_q_learning_update import (
    calculate_q_value_from_continuation,
    expected_continuation_over_next_values,
)
from step_21_two_independent_q_traders import (
    InformedQTrader,
    build_two_informed_traders,
    choose_actions_for_one_shared_period,
)
from step_22_market_maker_rolling_history import MarketObservation
from step_24_adaptive_market_maker_price import (
    calculate_adaptive_price_quote,
)
from step_24b_fast_rolling_ols import (
    RollingMarketMakerAppendTransactionToken,
    RollingMarketMakerOLS,
)
from step_24c_initial_market_maker_history import (
    SyntheticMarketMakerPrehistory,
    preload_rolling_market_maker,
)
from step_25_one_market_period import (
    build_paper_inputs,
    run_one_market_period,
)


SEED_DERIVATION_VERSION = "sha256-named-v1"
SEED_DOMAIN = b"vibe-replication.step26.seed.v1\0"
MAX_UINT64_PLUS_ONE = 2**64
STREAM_LABELS = (
    "initial_state",
    "fundamental_value",
    "noise_order",
    "trader_1_mode",
    "trader_1_action_tie",
    "trader_2_mode",
    "trader_2_action_tie",
)


def _uint64(number: int, label: str) -> int:
    """Return one non-boolean unsigned 64-bit integer. / 检查并返回 64 位无符号整数。"""

    if isinstance(number, bool) or not isinstance(number, int):
        raise TypeError(f"{label} must be an integer. / {label} 必须是整数。")
    if not 0 <= number < MAX_UINT64_PLUS_ONE:
        raise ValueError(f"{label} must lie in [0, 2^64). / {label} 必须位于 [0, 2^64)。")
    return number


def _derive_child_seed(parent_seed: int, label: str) -> int:
    """Derive one stable child seed without consuming any RNG.

    不消耗任何随机流，稳定地产生一个子种子。
    """

    checked_parent = _uint64(parent_seed, "parent seed / 父种子")
    if not isinstance(label, str) or not label:
        raise ValueError("Seed label cannot be empty. / 种子标签不能为空。")
    try:
        encoded_label = label.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("Seed label must be ASCII. / 种子标签必须使用 ASCII。") from error

    payload = (
        SEED_DOMAIN
        + checked_parent.to_bytes(8, "big")
        + b"\0"
        + encoded_label
    )
    # The first eight digest bytes form one stable unsigned 64-bit seed.
    # / 摘要前八个字节构成稳定的 64 位无符号种子。
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def derive_experiment_cell_seed(
    experiment_seed: int,
    experiment_cell_key: str,
) -> int:
    """Give one parameter/initializer cell its own reproducible identity.

    为一个参数/初始化实验单元建立独立、可复现的身份。
    """

    checked_experiment_seed = _uint64(
        experiment_seed,
        "experiment seed / 实验种子",
    )
    if not isinstance(experiment_cell_key, str) or not experiment_cell_key:
        raise ValueError("experiment_cell_key cannot be empty. / 实验单元标签不能为空。")
    return _derive_child_seed(
        checked_experiment_seed,
        f"experiment_cell:{experiment_cell_key}",
    )


def derive_session_seed(
    experiment_seed: int,
    experiment_cell_key: str,
    session_index: int,
) -> int:
    """Derive session j independently of parameter cells and execution order.

    产生不受参数单元或执行顺序影响的 session 种子。
    """

    experiment_cell_seed = derive_experiment_cell_seed(
        experiment_seed,
        experiment_cell_key,
    )
    checked_session_index = _uint64(
        session_index,
        "session index / session 编号",
    )
    return _derive_child_seed(
        experiment_cell_seed,
        f"session:{checked_session_index}",
    )


def derive_named_stream_seed(session_seed: int, stream_label: str) -> int:
    """Derive one of the seven approved stream seeds. / 产生七条指定随机流之一的种子。"""

    if stream_label not in STREAM_LABELS:
        raise ValueError(
            f"Unknown stream label {stream_label!r}. / 未知随机流标签 {stream_label!r}。"
        )
    return _derive_child_seed(session_seed, stream_label)


@dataclass(frozen=True)
class SessionSeedManifest:
    """Immutable record of every seed needed to replay one session.

    重放一个 session 所需全部种子的不可修改记录。
    """

    seed_derivation_version: str
    experiment_seed: int
    experiment_cell_key: str
    experiment_cell_seed: int
    session_index: int
    session_seed: int
    initial_state_seed: int
    fundamental_value_seed: int
    noise_order_seed: int
    trader_1_mode_seed: int
    trader_1_action_tie_seed: int
    trader_2_mode_seed: int
    trader_2_action_tie_seed: int
    rng_engine: str
    python_version: str

    def child_seeds(self) -> tuple[int, ...]:
        """Return all seven child seeds in documented order. / 按记录顺序返回七个子种子。"""

        return (
            self.initial_state_seed,
            self.fundamental_value_seed,
            self.noise_order_seed,
            self.trader_1_mode_seed,
            self.trader_1_action_tie_seed,
            self.trader_2_mode_seed,
            self.trader_2_action_tie_seed,
        )


def build_session_seed_manifest(
    experiment_seed: int,
    experiment_cell_key: str,
    session_index: int,
) -> SessionSeedManifest:
    """Create and validate one complete named-seed record. / 建立并检查完整命名种子记录。"""

    experiment_cell_seed = derive_experiment_cell_seed(
        experiment_seed,
        experiment_cell_key,
    )
    session_seed = derive_session_seed(
        experiment_seed,
        experiment_cell_key,
        session_index,
    )
    derived = {
        label: derive_named_stream_seed(session_seed, label)
        for label in STREAM_LABELS
    }
    if len(set(derived.values())) != len(derived):
        raise RuntimeError("Derived stream seeds unexpectedly collided. / 子种子意外重复。")

    return SessionSeedManifest(
        seed_derivation_version=SEED_DERIVATION_VERSION,
        experiment_seed=experiment_seed,
        experiment_cell_key=experiment_cell_key,
        experiment_cell_seed=experiment_cell_seed,
        session_index=session_index,
        session_seed=session_seed,
        initial_state_seed=derived["initial_state"],
        fundamental_value_seed=derived["fundamental_value"],
        noise_order_seed=derived["noise_order"],
        trader_1_mode_seed=derived["trader_1_mode"],
        trader_1_action_tie_seed=derived["trader_1_action_tie"],
        trader_2_mode_seed=derived["trader_2_mode"],
        trader_2_action_tie_seed=derived["trader_2_action_tie"],
        rng_engine="Python random.Random (MT19937)",
        python_version=platform.python_version(),
    )


class SessionRandomStreams:
    """Mutable environment RNGs owned by exactly one session.

    只属于一个 session 的可变环境随机流。

    Trader RNGs are created inside the two traders from the four trader seeds;
    this object owns only initial-state, value, and noise generators. / 四条交易者
    随机流由两个 trader 自己建立；本对象只持有初始状态、价值和噪声生成器。
    """

    def __init__(self, manifest: SessionSeedManifest) -> None:
        if not isinstance(manifest, SessionSeedManifest):
            raise TypeError("manifest must be SessionSeedManifest. / manifest 类型错误。")
        self.manifest = manifest
        self.initial_state_generator = random.Random(
            manifest.initial_state_seed
        )
        self.value_generator = random.Random(
            manifest.fundamental_value_seed
        )
        self.noise_generator = random.Random(
            manifest.noise_order_seed
        )

    def draw_initial_state(
        self,
        price_grid: Sequence[Sequence[float]],
        value_grid: Sequence[float],
    ) -> tuple[int, int, int]:
        """Use only the initial-state stream. / 只使用初始状态随机流。"""

        return draw_initial_state_indexes(
            price_grid,
            value_grid,
            self.initial_state_generator,
        )

    def draw_next_value_index(self, number_of_values: int) -> int:
        """Draw one equal-probability V index. / 等概率抽取一个价值编号。"""

        if (
            isinstance(number_of_values, bool)
            or not isinstance(number_of_values, int)
            or number_of_values < 2
        ):
            raise ValueError("number_of_values must be an integer >= 2. / 价值数量必须是至少为 2 的整数。")
        return self.value_generator.randrange(number_of_values)

    def draw_noise_order(self, noise_standard_deviation: float) -> float:
        """Draw continuous u_t ~ N(0, sigma_u^2). / 抽取连续正态噪声订单。"""

        if (
            isinstance(noise_standard_deviation, bool)
            or not isinstance(noise_standard_deviation, Real)
        ):
            raise TypeError("sigma_u must be a real number. / sigma_u 必须是实数。")
        sigma_u = float(noise_standard_deviation)
        if not isfinite(sigma_u) or sigma_u <= 0.0:
            raise ValueError("sigma_u must be positive and finite. / sigma_u 必须是有限正数。")
        return self.noise_generator.gauss(0.0, sigma_u)

    def build_traders(
        self,
        initial_q_table: np.ndarray,
    ) -> tuple[InformedQTrader, InformedQTrader]:
        """Create two traders with four separate private streams. / 用四条独立私有流建立两位 trader。"""

        return build_two_informed_traders(
            initial_q_table,
            random_seeds=(
                self.manifest.trader_1_mode_seed,
                self.manifest.trader_2_mode_seed,
            ),
            action_random_seeds=(
                self.manifest.trader_1_action_tie_seed,
                self.manifest.trader_2_action_tie_seed,
            ),
        )

    def environment_states(self) -> tuple[object, object, object]:
        """Return exact RNG states for testing/checkpoints. / 返回精确随机状态供测试或存档。"""

        return (
            self.initial_state_generator.getstate(),
            self.value_generator.getstate(),
            self.noise_generator.getstate(),
        )


@dataclass(frozen=True)
class RandomizedPeriodTrace:
    """Small immutable audit record; do not retain 100 million of these.

    小型不可修改审计记录；正式一亿期运行时不能保存一亿条。
    """

    period_number: int
    current_state_indexes: tuple[int, int, int]
    current_value_index: int
    epsilon: float
    decisions: tuple[ActionDecision, ActionDecision]
    raw_orders_x: tuple[float, float]
    noise_order_u: float
    total_order_flow_y: float
    continuous_price_p: float
    insensitive_order_z: float
    profits: tuple[float, float]
    next_value_index: int
    next_state_indexes: tuple[int, int, int]
    old_q_values: tuple[float, float]
    new_q_values: tuple[float, float]


@dataclass(frozen=True)
class ShortSessionTrace:
    """Compact trace used only for Step 26 reproducibility tests. / 仅供第 26 步复现测试的短轨迹。"""

    seed_manifest: SessionSeedManifest
    initial_state_indexes: tuple[int, int, int]
    periods: tuple[RandomizedPeriodTrace, ...]
    final_value_visit_counts: tuple[int, ...]


@dataclass(frozen=True)
class FrozenPolicyPeriodObservation:
    """One immutable post-convergence row; callers need not retain it.

    一条不可修改的收敛后观测；调用者不需要把所有时期保存在内存中。

    Step 28 freezes the informed agents but keeps random value/noise draws and
    the adaptive market maker running. This row exposes the raw ingredients
    needed by later metric steps without calculating those metrics here.
    / 第 28 步冻结知情 agents，但继续抽取价值与噪声，并继续更新自适应做市商。
    本记录只提供之后指标需要的原料，本步骤不提前计算指标。
    """

    period_number: int
    current_state_indexes: tuple[int, int, int]
    current_state_id: int
    current_value_index: int
    fundamental_value_v: float
    action_indexes: tuple[int, int]
    raw_orders_x: tuple[float, float]
    noise_order_u: float
    total_order_flow_y: float
    xi_0_hat: float
    xi_1_hat: float
    gamma_0_hat: float
    gamma_1_hat: float
    price_impact_lambda_hat: float
    continuous_price_p: float
    insensitive_order_z: float
    profits: tuple[float, float]
    next_value_index: int
    next_state_indexes: tuple[int, int, int]
    next_price_was_clipped: bool


@dataclass(frozen=True)
class FrozenSuppliedPathTransactionToken:
    """Opaque proof for one bounded, disposable frozen-policy path.

    一条有时期上限、用后即丢弃的固定策略路径之不透明凭证。

    This token is not a random seed and contains no paper parameter. It merely
    proves which live session/path transaction may be rolled back. / 此 token
    不是随机种子，也不包含论文参数；它只证明哪一个实时 session 短路径可被回滚。
    """

    _owner_marker: object
    _transaction_marker: object
    max_periods: int


@dataclass
class _ActiveFrozenSuppliedPath:
    """Private causal start state for a reusable short path. / 可重复短路径的私有因果起点。"""

    token: FrozenSuppliedPathTransactionToken
    maker_token: RollingMarketMakerAppendTransactionToken
    starting_period_number: int
    starting_previous_price: float
    starting_previous_value: float
    starting_current_value: float
    starting_draw_source_mode: str | None
    starting_random_states: tuple[object, ...]
    starting_value_visit_counts: tuple[int, ...]
    starting_frozen_policy: np.ndarray
    starting_q_tables: tuple[np.ndarray, np.ndarray]
    completed_periods: int = 0


class RandomizedMarketSession:
    """One validated mutable session with a lean repeated-period method.

    一个经过一次完整检查、随后使用精简逐期方法的可变 session。
    """

    def __init__(
        self,
        *,
        parameters: PaperParameters,
        value_grid: Sequence[float],
        price_grid: Sequence[Sequence[float]],
        action_multipliers: Sequence[float],
        traders: tuple[InformedQTrader, InformedQTrader],
        market_maker: RollingMarketMakerOLS,
        shared_value_visit_counts: list[int],
        streams: SessionRandomStreams,
        initial_state_indexes: tuple[int, int, int],
    ) -> None:
        """Validate full static memory once, never once per period. / 只在 session 开始时完整检查一次。"""

        if not isinstance(parameters, PaperParameters):
            raise TypeError("parameters must be PaperParameters. / parameters 类型错误。")
        if parameters.num_speculators != 2:
            raise ValueError("Step 26 currently implements exactly I=2. / 第 26 步目前只实现 I=2。")
        self.full_q_validation_count = 0
        values = tuple(float(value) for value in value_grid)
        prices = validate_price_grids_by_value(
            price_grid,
            parameters.num_value_points,
            parameters.num_price_points,
        )
        number_of_prices = number_of_price_points(prices)
        multipliers = tuple(float(value) for value in action_multipliers)
        if len(values) != parameters.num_value_points:
            raise ValueError("V size differs from n_v. / V 大小与 n_v 不一致。")
        if len(multipliers) != parameters.num_action_points:
            raise ValueError("X size differs from n_x. / X 大小与 n_x 不一致。")
        for grid, label in (
            (values, "V"),
            (multipliers, "X"),
        ):
            if not all(isfinite(value) for value in grid):
                raise ValueError(f"{label} must be finite. / {label} 必须有限。")
            if any(right <= left for left, right in zip(grid, grid[1:])):
                raise ValueError(f"{label} must be strictly increasing. / {label} 必须严格递增。")
        if not isinstance(streams, SessionRandomStreams):
            raise TypeError("streams must be SessionRandomStreams. / streams 类型错误。")
        if not isinstance(market_maker, RollingMarketMakerOLS):
            raise TypeError("market_maker type is wrong. / market_maker 类型错误。")
        if (
            market_maker.window_size != parameters.market_maker_window
            or not market_maker.is_full
        ):
            raise ValueError("Market-maker D_0 must be a full T_m window. / 做市商 D_0 必须是完整 T_m 窗口。")
        if len(traders) != 2 or traders[0] is traders[1]:
            raise ValueError("Exactly two different traders are required. / 必须有两个不同 trader。")
        if len(shared_value_visit_counts) != len(values):
            raise ValueError("Visit-counter size differs from V. / 访问计数器大小与 V 不一致。")
        if any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for count in shared_value_visit_counts
        ):
            raise ValueError("Visit counts must be non-negative integers. / 访问计数必须是非负整数。")

        expected_q_shape = (
            number_of_states(number_of_prices, len(values)),
            len(multipliers),
        )
        self._validate_full_q_tables_once(traders, expected_q_shape)

        all_trader_generators = (
            traders[0].mode_random_generator,
            traders[0].action_random_generator,
            traders[1].mode_random_generator,
            traders[1].action_random_generator,
        )
        if len({id(generator) for generator in all_trader_generators}) != 4:
            raise ValueError("The four trader RNG objects must be private. / 四条 trader 随机流必须相互独立。")
        expected_trader_seeds = (
            streams.manifest.trader_1_mode_seed,
            streams.manifest.trader_1_action_tie_seed,
            streams.manifest.trader_2_mode_seed,
            streams.manifest.trader_2_action_tie_seed,
        )
        actual_trader_seeds = (
            traders[0].mode_random_seed,
            traders[0].action_random_seed,
            traders[1].mode_random_seed,
            traders[1].action_random_seed,
        )
        if actual_trader_seeds != expected_trader_seeds:
            raise ValueError("Trader seeds differ from the session manifest. / trader 种子与 session 清单不一致。")
        if np.shares_memory(traders[0].q_table, traders[1].q_table):
            raise ValueError("Trader Q-tables cannot share memory. / 两张 Q 表不能共享内存。")

        # Encoding validates all three initial indexes. / 编码函数会检查三个初始编号。
        encode_state_index(initial_state_indexes, number_of_prices, len(values))

        self.parameters = parameters
        self.value_grid = values
        self.price_grid = prices
        self.number_of_prices = number_of_prices
        self.action_multipliers = multipliers
        self.traders = traders
        self.market_maker = market_maker
        self.shared_value_visit_counts = shared_value_visit_counts
        self.streams = streams
        self.initial_state_indexes = initial_state_indexes
        self.previous_price = prices[initial_state_indexes[1]][initial_state_indexes[0]]
        self.previous_value = values[initial_state_indexes[1]]
        self.current_value = values[initial_state_indexes[2]]
        self.period_number = 0
        self.execution_mode = "training"
        self._frozen_policy_action_indexes: np.ndarray | None = None
        self._frozen_draw_source_mode: str | None = None
        self._phase_controller_token: object | None = None
        # Step 35D may reuse one detached branch for many short supplied-draw
        # paths. The marker and active record keep that reuse explicit and
        # bounded. / 第 35D 步可重复使用一个脱离 controller 的分支来运行许多
        # 外部抽样短路径；此标记和活动记录让复用过程明确且有上限。
        self._frozen_path_transaction_owner_marker = object()
        self._active_frozen_supplied_path: (
            _ActiveFrozenSuppliedPath | None
        ) = None
        self.after_q_update_observer: (
            Callable[
                [int, int, tuple[InformedQTrader, InformedQTrader]],
                None,
            ]
            | None
        ) = None

        # Cache all raw orders once: rows are V indexes, columns are actions.
        # / 一次性缓存全部实际订单：行是价值编号，列是动作编号。
        self.orders_by_value_and_action = tuple(
            tuple(
                calculate_orders_for_value(
                    value,
                    parameters.value_mean,
                    list(multipliers),
                )
            )
            for value in values
        )

    def claim_phase_controller(self, controller_token: object) -> None:
        """Give one Step-28 controller exclusive period-execution rights.

        把逐期运行权交给一个第 28 步 controller，防止绕过阶段边界。
        """

        if controller_token is None:
            raise ValueError("Controller token cannot be None. / controller token 不能为 None。")
        if self.period_number != 0 or self.execution_mode != "training":
            raise RuntimeError("A controller may claim only a fresh session. / controller 只能接管新 session。")
        if self._phase_controller_token is not None:
            raise RuntimeError("This session already has a controller. / 此 session 已有 controller。")
        self._phase_controller_token = controller_token

    def _check_controller_token(self, controller_token: object | None) -> None:
        """Reject bypass calls before they consume RNG or change the market.

        在消耗随机数或改变市场之前拒绝绕过 controller 的调用。
        """

        if (
            self._phase_controller_token is not None
            and controller_token is not self._phase_controller_token
        ):
            raise RuntimeError(
                "This session must be run through its phase controller. / "
                "此 session 必须通过阶段 controller 运行。"
            )

    def begin_frozen_greedy_measurement(
        self,
        converged_policy_masks: np.ndarray,
        *,
        controller_token: object | None = None,
    ) -> None:
        """Atomically switch from learning to one fixed greedy policy.

        在两个时期之间，把学习模式原子地切换为固定贪心策略模式。

        Exact-tie convention / 精确并列约定:
            Pick the lowest action index once and use it throughout measurement.
            The paper gives no tie rule, so Step 28 records this as an explicit
            replication choice. / 在收敛时选并列集合中的最小动作编号，并在整个
            测量阶段固定使用。论文没有规定并列规则，因此这是明确复现选择。
        """

        self._check_controller_token(controller_token)
        if self.execution_mode != "training":
            raise RuntimeError("The session is not in training mode. / session 不在训练模式。")
        if self.period_number < 1:
            raise RuntimeError("Measurement can begin only after a completed training period. / 至少完成一个训练时期后才能开始测量。")
        if not isinstance(converged_policy_masks, np.ndarray):
            raise TypeError("Policy masks must be a NumPy array. / 策略 masks 必须是 NumPy 数组。")
        expected_shape = (
            len(self.traders),
            number_of_states(self.number_of_prices, len(self.value_grid)),
        )
        if converged_policy_masks.shape != expected_shape:
            raise ValueError("Policy-mask shape is wrong. / 策略 mask 形状错误。")
        if not np.issubdtype(converged_policy_masks.dtype, np.integer):
            raise TypeError("Policy masks must contain integers. / 策略 mask 必须包含整数。")

        number_of_actions = len(self.action_multipliers)
        action_indexes = np.empty(expected_shape, dtype=np.int16)
        for agent_index in range(expected_shape[0]):
            for state_index in range(expected_shape[1]):
                mask = int(converged_policy_masks[agent_index, state_index])
                if mask <= 0 or mask >= 1 << number_of_actions:
                    raise ValueError(
                        "A policy mask refers to no action or an unavailable "
                        "action. / 策略 mask 没有动作或指向不存在的动作。"
                    )

                # Recompute the exact argmax set from the live Q row. This
                # catches stale or corrupted snapshots before anything freezes.
                # / 从实时 Q 行重算精确最优集合，在冻结前发现过期或损坏的快照。
                best_q = float("-inf")
                live_mask = 0
                for action_index in range(number_of_actions):
                    q_value = float(
                        self.traders[agent_index].q_table[
                            state_index,
                            action_index,
                        ]
                    )
                    if not isfinite(q_value):
                        raise ValueError("A live Q-value is not finite. / 实时 Q 值不是有限数。")
                    if q_value > best_q:
                        best_q = q_value
                        live_mask = 1 << action_index
                    elif q_value == best_q:
                        live_mask |= 1 << action_index
                if mask != live_mask:
                    raise ValueError("A policy mask does not match the live Q argmax set. / 策略 mask 与实时 Q 最优集合不一致。")
                lowest_set_bit = mask & -mask
                action_indexes[agent_index, state_index] = (
                    lowest_set_bit.bit_length() - 1
                )

        # Commit only after the entire mapping is valid. / 只有完整映射全部有效后才正式切换。
        action_indexes.flags.writeable = False
        for trader in self.traders:
            trader.q_table.flags.writeable = False
        self._frozen_policy_action_indexes = action_indexes
        self.execution_mode = "measurement"

    def frozen_policy_action_indexes_snapshot(self) -> np.ndarray:
        """Return a defensive read-only copy of the frozen policy table.

        返回固定策略动作表的只读防御性副本。

        The original table remains private so outside code cannot change the
        actions used by a live session. Step 35A needs only a safe snapshot for
        checkpointing. / 原表仍保持私有，外部代码不能改变实时 session 使用的
        动作。第 35A 步只取得一份安全快照用于存档。
        """

        if (
            self.execution_mode != "measurement"
            or self._frozen_policy_action_indexes is None
        ):
            raise RuntimeError(
                "A frozen policy exists only during measurement. / "
                "固定策略只在测量阶段存在。"
            )
        snapshot = self._frozen_policy_action_indexes.copy()
        snapshot.flags.writeable = False
        return snapshot

    def begin_reversible_frozen_supplied_path(
        self,
        *,
        max_periods: int,
    ) -> FrozenSuppliedPathTransactionToken:
        """Begin one bounded supplied-draw path that will be rolled back.

        开始一条有长度上限、稍后会回滚的外部抽样路径。

        This is allowed only on a detached, already-frozen measurement branch.
        The Q-tables, policy, visit counters, and all seven internal RNGs remain
        fixed; only the short market state and rolling maker advance. / 仅允许在
        已脱离 controller、已冻结策略的测量分支上使用。Q 表、策略、访问计数和
        七条内部随机流保持不变；只有短期市场状态与滚动做市商向前推进。

        Starting does not copy the 10,000-row maker history. The maker opens its
        own O(1)-start reversible append transaction. / 开始时不会复制一万行做市商
        历史；做市商会开启自己的 O(1) 起步可逆追加事务。
        """

        if (
            isinstance(max_periods, bool)
            or not isinstance(max_periods, int)
            or max_periods <= 0
        ):
            raise ValueError(
                "max_periods must be a positive integer. / "
                "max_periods 必须是正整数。"
            )
        self._check_controller_token(None)
        if self.execution_mode != "measurement":
            raise RuntimeError(
                "A reversible path requires measurement mode. / "
                "可逆路径要求 session 处于测量模式。"
            )
        if self._frozen_policy_action_indexes is None:
            raise RuntimeError("The frozen policy is missing. / 固定策略丢失。")
        if self._frozen_draw_source_mode == "internal":
            raise RuntimeError(
                "An internally drawn continuation cannot become a supplied path. / "
                "内部抽样续跑不能改成外部抽样短路径。"
            )
        if self._active_frozen_supplied_path is not None:
            raise RuntimeError(
                "A reversible supplied path is already active. / "
                "已有可逆外部抽样路径正在运行。"
            )
        if any(trader.q_table.flags.writeable for trader in self.traders):
            raise RuntimeError(
                "Frozen Q-tables must remain read-only. / 固定 Q 表必须保持只读。"
            )
        if self._frozen_policy_action_indexes.flags.writeable:
            raise RuntimeError(
                "The frozen policy table must remain read-only. / "
                "固定策略表必须保持只读。"
            )

        maker_token = self.market_maker.begin_reversible_append_transaction(
            max_appends=max_periods,
        )
        token = FrozenSuppliedPathTransactionToken(
            _owner_marker=self._frozen_path_transaction_owner_marker,
            _transaction_marker=object(),
            max_periods=max_periods,
        )
        self._active_frozen_supplied_path = _ActiveFrozenSuppliedPath(
            token=token,
            maker_token=maker_token,
            starting_period_number=self.period_number,
            starting_previous_price=self.previous_price,
            starting_previous_value=self.previous_value,
            starting_current_value=self.current_value,
            starting_draw_source_mode=self._frozen_draw_source_mode,
            starting_random_states=self.all_random_states(),
            starting_value_visit_counts=tuple(self.shared_value_visit_counts),
            starting_frozen_policy=self._frozen_policy_action_indexes,
            starting_q_tables=(
                self.traders[0].q_table,
                self.traders[1].q_table,
            ),
        )
        return token

    def _validated_active_frozen_supplied_path(
        self,
        token: FrozenSuppliedPathTransactionToken,
    ) -> _ActiveFrozenSuppliedPath:
        """Return the matching path record or reject stale/foreign proof.

        返回相符的路径记录；拒绝过期或来自其他 session 的凭证。
        """

        if not isinstance(token, FrozenSuppliedPathTransactionToken):
            raise TypeError("token has the wrong type. / token 类型错误。")
        active = self._active_frozen_supplied_path
        if (
            active is None
            or token is not active.token
            or token._owner_marker
            is not self._frozen_path_transaction_owner_marker
            or token._transaction_marker is not active.token._transaction_marker
        ):
            raise RuntimeError(
                "The frozen-path token is stale or foreign. / "
                "固定路径 token 已过期或来自其他 session。"
            )
        return active

    def rollback_reversible_frozen_supplied_path(
        self,
        token: FrozenSuppliedPathTransactionToken,
    ) -> int:
        """Undo the live short path and return its completed-period count.

        撤销当前短路径，并返回已完成的时期数。

        The method first checks every state that frozen supplied execution must
        leave unchanged. It then rolls back the maker and restores the four
        small session scalars plus the draw-source marker. / 本方法先核对固定外部
        抽样本应保持不变的所有状态，再回滚做市商，并恢复四个小型 session 状态
        数值和抽样来源标记。
        """

        active = self._validated_active_frozen_supplied_path(token)
        if self.execution_mode != "measurement":
            raise RuntimeError(
                "The session left measurement mode during the path. / "
                "短路径期间 session 离开了测量模式。"
            )
        if self._frozen_policy_action_indexes is not active.starting_frozen_policy:
            raise RuntimeError(
                "The frozen policy object changed during the path. / "
                "短路径期间固定策略对象发生改变。"
            )
        if any(
            trader.q_table is not starting_q
            for trader, starting_q in zip(
                self.traders,
                active.starting_q_tables,
                strict=True,
            )
        ):
            raise RuntimeError(
                "A Q-table object changed during the path. / "
                "短路径期间 Q 表对象发生改变。"
            )
        if any(trader.q_table.flags.writeable for trader in self.traders):
            raise RuntimeError(
                "A frozen Q-table became writable. / 固定 Q 表变成了可写。"
            )
        if tuple(self.shared_value_visit_counts) != (
            active.starting_value_visit_counts
        ):
            raise RuntimeError(
                "Value visit counts changed during a frozen path. / "
                "固定路径期间价值访问计数发生改变。"
            )
        if self.all_random_states() != active.starting_random_states:
            raise RuntimeError(
                "An internal RNG changed during a supplied path. / "
                "外部抽样路径期间内部随机流发生改变。"
            )
        maker_appends = self.market_maker.reversible_append_count(
            active.maker_token
        )
        session_periods_advanced = (
            self.period_number - active.starting_period_number
        )
        if (
            maker_appends > active.token.max_periods
            or maker_appends - active.completed_periods not in (0, 1)
            or maker_appends - session_periods_advanced not in (0, 1)
        ):
            raise RuntimeError(
                "Session time differs from the reversible maker log. / "
                "session 时期与可逆做市商日志不一致。"
            )

        rolled_back = self.market_maker.rollback_reversible_append_transaction(
            active.maker_token
        )
        self.period_number = active.starting_period_number
        self.previous_price = active.starting_previous_price
        self.previous_value = active.starting_previous_value
        self.current_value = active.starting_current_value
        self._frozen_draw_source_mode = active.starting_draw_source_mode
        self._active_frozen_supplied_path = None
        return rolled_back

    @property
    def frozen_draw_source_mode(self) -> str | None:
        """Report ``internal``, ``supplied``, or None before the first frozen period.

        返回 ``internal``、``supplied``，或固定阶段第一期之前的 None。

        Once a frozen continuation chooses a draw source, mixing in the other
        source would put calendar time and RNG time out of alignment. / 固定续跑
        一旦选定抽样来源，就不能改用另一来源，否则市场时期与随机流时期会错位。
        """

        return self._frozen_draw_source_mode

    def finish_frozen_greedy_measurement(
        self,
        *,
        controller_token: object | None = None,
    ) -> None:
        """Close measurement between periods; no RNG or market state changes.

        在两个时期之间结束测量；不改变随机数或市场状态。
        """

        self._check_controller_token(controller_token)
        if self._active_frozen_supplied_path is not None:
            raise RuntimeError(
                "Rollback the reversible path before finishing measurement. / "
                "结束测量前必须先回滚可逆路径。"
            )
        if self.execution_mode != "measurement":
            raise RuntimeError("The session is not measuring. / session 不在测量阶段。")
        self.execution_mode = "complete"

    def _validate_full_q_tables_once(
        self,
        traders: tuple[InformedQTrader, InformedQTrader],
        expected_q_shape: tuple[int, int],
    ) -> None:
        """Perform and count the expensive full-table scan exactly once.

        只执行并记录一次昂贵的完整 Q 表扫描。
        """

        if self.full_q_validation_count != 0:
            raise RuntimeError("Full Q validation may run only once. / 完整 Q 检查只能运行一次。")
        for trader in traders:
            if not isinstance(trader, InformedQTrader):
                raise TypeError("Trader type is wrong. / trader 类型错误。")
            if trader.q_table.shape != expected_q_shape:
                raise ValueError("Q-table shape is wrong. / Q 表形状错误。")
            if not np.issubdtype(trader.q_table.dtype, np.floating):
                raise TypeError("Q-table must use floats. / Q 表必须使用浮点数。")
            if not np.isfinite(trader.q_table).all():
                raise ValueError("Every initial Q-value must be finite. / 所有初始 Q 值必须有限。")
            if not trader.q_table.flags.writeable:
                raise ValueError("Q-table must be writable. / Q 表必须可写。")
        self.full_q_validation_count += 1

    def attach_after_q_update_observer(
        self,
        observer: Callable[
            [int, int, tuple[InformedQTrader, InformedQTrader]],
            None,
        ],
    ) -> None:
        """Attach one no-receipt online observer before period 0.

        在第 0 期之前连接一个不生成流水单的在线观察器。

        Step 27 uses this hook for convergence tracking. / 第 27 步使用这个接口
        在线跟踪收敛。
        """

        if not callable(observer):
            raise TypeError("observer must be callable. / observer 必须可以调用。")
        if self.period_number != 0:
            raise RuntimeError("Attach the observer before period 0. / 必须在第 0 期前连接 observer。")
        if self.after_q_update_observer is not None:
            raise RuntimeError("An observer is already attached. / 已经连接了 observer。")
        self.after_q_update_observer = observer

    def _run_period_with_draw_suppliers(
        self,
        noise_supplier: Callable[[], float],
        next_value_index_supplier: Callable[[], int],
        *,
        collect_trace: bool,
        frozen_policy_measurement: bool = False,
        controller_token: object | None = None,
    ) -> RandomizedPeriodTrace | FrozenPolicyPeriodObservation | None:
        """Run the paper timing; suppliers are invoked only when their shocks arrive.

        按论文时序运行；只有冲击真正到达时才调用相应抽样函数。
        """

        self._check_controller_token(controller_token)
        required_mode = (
            "measurement" if frozen_policy_measurement else "training"
        )
        if self.execution_mode != required_mode:
            raise RuntimeError(
                f"This call requires {required_mode} mode. / "
                f"本调用要求 session 处于 {required_mode} 模式。"
            )
        if frozen_policy_measurement and collect_trace:
            raise ValueError("A training trace cannot represent measurement. / 训练轨迹不能表示测量时期。")

        # 1. Current state and independent agent choices. / 1. 当前状态与两位 agent 独立选择。
        current_state_indexes = build_state_indexes(
            self.previous_price,
            self.previous_value,
            self.current_value,
            self.price_grid,
            self.value_grid,
        )
        current_state_id = encode_state_index(
            current_state_indexes,
            self.number_of_prices,
            len(self.value_grid),
        )
        current_value_index = fundamental_value_to_index(
            self.current_value,
            self.value_grid,
        )
        decisions: tuple[ActionDecision, ActionDecision] | None
        if frozen_policy_measurement:
            if self._frozen_policy_action_indexes is None:
                raise RuntimeError("The frozen policy is missing. / 固定策略丢失。")
            epsilon = 0.0
            decisions = None
            action_indexes = (
                int(self._frozen_policy_action_indexes[0, current_state_id]),
                int(self._frozen_policy_action_indexes[1, current_state_id]),
            )
        else:
            epsilon, decisions = choose_actions_for_one_shared_period(
                self.traders,
                current_state_id,
                current_value_index,
                self.shared_value_visit_counts,
                self.parameters.exploration_decay,
            )
            action_indexes = (
                decisions[0].action_index,
                decisions[1].action_index,
            )
        available_orders = self.orders_by_value_and_action[
            current_value_index
        ]
        raw_orders = (
            available_orders[action_indexes[0]],
            available_orders[action_indexes[1]],
        )

        # 2. Noise arrives only after actions. / 2. 两位完成动作后，噪声才到达。
        supplied_noise = noise_supplier()
        if (
            isinstance(supplied_noise, bool)
            or not isinstance(supplied_noise, Real)
            or not isfinite(float(supplied_noise))
        ):
            raise ValueError("Noise supplier returned an invalid u_t. / 噪声抽样函数返回了无效 u_t。")
        noise_order = float(supplied_noise)
        total_order_flow = calculate_total_order_flow(
            raw_orders[0],
            raw_orders[1],
            noise_order,
        )

        # 3. Maker uses D_t only, then price causes z and profits. / 3. 做市商只用旧 D_t 定价，随后产生 z 与利润。
        prior_estimates = self.market_maker.estimates()
        quote = calculate_adaptive_price_quote(
            total_order_flow,
            prior_estimates,
            self.parameters.pricing_error_weight,
        )
        continuous_price = quote.continuous_price_p_hat
        insensitive_order = calculate_insensitive_order(
            continuous_price,
            self.parameters.value_mean,
            self.parameters.investor_slope,
        )
        profits = (
            calculate_profit(
                self.current_value,
                continuous_price,
                raw_orders[0],
            ),
            calculate_profit(
                self.current_value,
                continuous_price,
                raw_orders[1],
            ),
        )
        if not all(
            isfinite(number)
            for number in (
                total_order_flow,
                continuous_price,
                insensitive_order,
                *profits,
            )
        ):
            raise ValueError("The realized market outcome is not finite. / 实际市场结果不是有限数。")

        completed_row = MarketObservation(
            fundamental_value_v=self.current_value,
            market_price_p=continuous_price,
            insensitive_order_z=insensitive_order,
            informed_and_noise_order_y=total_order_flow,
        )

        # 4. Only now draw v_(t+1), then prepare both Q updates. / 4. 现在才抽取下一价值，再准备两位的 Q 更新。
        next_value_index = next_value_index_supplier()
        if (
            isinstance(next_value_index, bool)
            or not isinstance(next_value_index, int)
            or not 0 <= next_value_index < len(self.value_grid)
        ):
            raise ValueError("Next-value supplier returned an invalid index. / 下一价值抽样函数返回了无效编号。")
        next_value = self.value_grid[next_value_index]
        current_value_price_row = self.price_grid[current_value_index]
        next_price_index = continuous_price_to_index(
            continuous_price,
            current_value_price_row,
        )
        next_state_indexes = (
            next_price_index,
            current_value_index,
            next_value_index,
        )
        old_q_values: list[float] = []
        new_q_values: list[float] = []
        if not frozen_policy_measurement:
            if decisions is None:
                raise RuntimeError("Training decisions are missing. / 训练动作记录丢失。")
            possible_next_state_ids = tuple(
                encode_state_index(
                    (
                        next_price_index,
                        current_value_index,
                        possible_value_index,
                    ),
                    self.number_of_prices,
                    len(self.value_grid),
                )
                for possible_value_index in range(len(self.value_grid))
            )
            for trader, decision, profit in zip(
                self.traders,
                decisions,
                profits,
                strict=True,
            ):
                old_q = float(
                    trader.q_table[current_state_id, decision.action_index]
                )
                continuation = expected_continuation_over_next_values(
                    trader.q_table[list(possible_next_state_ids), :].copy()
                )
                new_q = calculate_q_value_from_continuation(
                    old_q,
                    profit,
                    continuation,
                    self.parameters.learning_rate,
                    self.parameters.discount_factor,
                )
                old_q_values.append(old_q)
                new_q_values.append(new_q)

            # 5. Commit only after both candidates exist. / 5. 两个候选新值都算好后才一起写入。
            for trader, decision, new_q in zip(
                self.traders,
                decisions,
                new_q_values,
                strict=True,
            ):
                trader.q_table[
                    current_state_id,
                    decision.action_index,
                ] = new_q
        evicted_row = self.market_maker.append_completed_observation(
            completed_row
        )
        if evicted_row is None:
            raise RuntimeError("A full rolling window must evict one row. / 完整滚动窗口必须淘汰一行。")
        if (
            not frozen_policy_measurement
            and self.after_q_update_observer is not None
        ):
            self.after_q_update_observer(
                self.period_number,
                current_state_id,
                self.traders,
            )

        result: RandomizedPeriodTrace | FrozenPolicyPeriodObservation | None
        result = None
        if frozen_policy_measurement:
            result = FrozenPolicyPeriodObservation(
                period_number=self.period_number,
                current_state_indexes=current_state_indexes,
                current_state_id=current_state_id,
                current_value_index=current_value_index,
                fundamental_value_v=self.current_value,
                action_indexes=action_indexes,
                raw_orders_x=raw_orders,
                noise_order_u=noise_order,
                total_order_flow_y=total_order_flow,
                xi_0_hat=prior_estimates.xi_0_hat,
                xi_1_hat=prior_estimates.xi_1_hat,
                gamma_0_hat=prior_estimates.gamma_0_hat,
                gamma_1_hat=prior_estimates.gamma_1_hat,
                price_impact_lambda_hat=quote.price_impact_lambda_hat,
                continuous_price_p=continuous_price,
                insensitive_order_z=insensitive_order,
                profits=profits,
                next_value_index=next_value_index,
                next_state_indexes=next_state_indexes,
                next_price_was_clipped=(
                    continuous_price < current_value_price_row[0]
                    or continuous_price > current_value_price_row[-1]
                ),
            )
        elif collect_trace:
            if decisions is None:
                raise RuntimeError("Training decisions are missing. / 训练动作记录丢失。")
            result = RandomizedPeriodTrace(
                period_number=self.period_number,
                current_state_indexes=current_state_indexes,
                current_value_index=current_value_index,
                epsilon=epsilon,
                decisions=decisions,
                raw_orders_x=raw_orders,
                noise_order_u=noise_order,
                total_order_flow_y=total_order_flow,
                continuous_price_p=continuous_price,
                insensitive_order_z=insensitive_order,
                profits=profits,
                next_value_index=next_value_index,
                next_state_indexes=next_state_indexes,
                old_q_values=(old_q_values[0], old_q_values[1]),
                new_q_values=(new_q_values[0], new_q_values[1]),
            )

        # Advance the session state for the next call. / 推进 session 状态，供下一次调用。
        self.previous_price = continuous_price
        self.previous_value = self.current_value
        self.current_value = next_value
        self.period_number += 1
        return result

    def run_next_random_period(self) -> RandomizedPeriodTrace:
        """Draw live u_t and v_(t+1) at their exact causal positions. / 在准确因果位置抽取 u_t 与下一价值。"""

        trace = self._run_period_with_draw_suppliers(
            lambda: self.streams.draw_noise_order(
                self.parameters.noise_std
            ),
            lambda: self.streams.draw_next_value_index(
                len(self.value_grid)
            ),
            collect_trace=True,
        )
        if not isinstance(trace, RandomizedPeriodTrace):
            raise RuntimeError("Trace collection unexpectedly failed. / 轨迹收集意外失败。")
        return trace

    def run_next_random_period_without_trace(self) -> None:
        """Hot-loop form: update the market without allocating a trace object.

        高频循环版本：更新市场，但不分配轨迹对象。

        Step 27 will attach online convergence tracking; formal experiments
        must aggregate statistics online rather than retain every period. / 第 27
        步会连接在线收敛跟踪；正式实验必须在线汇总，不能保存每一期。
        """

        self._run_next_training_period_without_trace(None)

    def run_next_training_period_for_controller(
        self,
        controller_token: object,
    ) -> None:
        """Controller-owned training call; direct bypass calls are rejected.

        controller 专用训练调用；绕过 controller 的直接调用会被拒绝。
        """

        self._run_next_training_period_without_trace(controller_token)

    def _run_next_training_period_without_trace(
        self,
        controller_token: object | None,
    ) -> None:
        """Shared no-trace implementation with optional ownership proof.

        带可选所有权证明的无轨迹训练实现。
        """

        result = self._run_period_with_draw_suppliers(
            lambda: self.streams.draw_noise_order(
                self.parameters.noise_std
            ),
            lambda: self.streams.draw_next_value_index(
                len(self.value_grid)
            ),
            collect_trace=False,
            controller_token=controller_token,
        )
        if result is not None:
            raise RuntimeError("No-trace mode allocated a trace. / 无轨迹模式错误地分配了轨迹。")

    def run_next_frozen_policy_period(
        self,
    ) -> FrozenPolicyPeriodObservation:
        """Run one post-convergence period without exploration or Q learning.

        运行一个收敛后时期：不探索、不更新访问计数，也不更新 Q 表；价值、噪声
        和滚动做市商继续运行。
        """

        return self._run_next_frozen_policy_period(None)

    def run_next_frozen_policy_period_with_supplied_draws(
        self,
        *,
        noise_order_u: float,
        next_value_index: int,
    ) -> FrozenPolicyPeriodObservation:
        """Run one detached frozen period with externally paired draws.

        使用外部配对抽样，运行一个脱离 controller 的固定策略时期。

        Step 35B uses this public bridge so control and treatment receive the
        same *ordinary* noise and next value. The treatment may receive one
        disclosed additive shock in the supplied ``noise_order_u``. This method
        does not consume any of the session's seven RNG streams. / 第 35B 步用
        这个公开入口，让对照组和实验组收到相同的普通噪声与下一价值；实验组可在
        传入的 ``noise_order_u`` 中加入一次明确记录的额外冲击。本方法不会推进
        session 自己的七条随机流。

        Only a detached branch may use it. A Step-28 controlled source session
        must continue through its controller. / 只有脱离 controller 的分支可以
        使用；仍由第 28 步 controller 管理的源 session 必须继续经过 controller。
        """

        self._check_controller_token(None)
        active_path = self._active_frozen_supplied_path
        if (
            active_path is not None
            and active_path.completed_periods >= active_path.token.max_periods
        ):
            raise RuntimeError(
                "The reversible frozen-path period limit was reached. / "
                "已达到可逆固定路径的时期上限。"
            )
        if self.execution_mode != "measurement":
            raise RuntimeError(
                "Supplied draws require frozen measurement mode. / "
                "外部抽样要求固定策略测量模式。"
            )
        if self._frozen_policy_action_indexes is None:
            raise RuntimeError("The frozen policy is missing. / 固定策略丢失。")
        if self._frozen_draw_source_mode == "internal":
            raise RuntimeError(
                "Cannot switch an internally drawn continuation to supplied draws. / "
                "不能把内部抽样续跑切换为外部抽样。"
            )
        if (
            isinstance(noise_order_u, bool)
            or not isinstance(noise_order_u, Real)
            or not isfinite(float(noise_order_u))
        ):
            raise ValueError(
                "noise_order_u must be a finite real number. / "
                "noise_order_u 必须是有限实数。"
            )
        if (
            isinstance(next_value_index, bool)
            or not isinstance(next_value_index, int)
            or not 0 <= next_value_index < len(self.value_grid)
        ):
            raise ValueError(
                "next_value_index is outside V. / next_value_index 超出 V。"
            )

        result = self._run_period_with_draw_suppliers(
            lambda: float(noise_order_u),
            lambda: next_value_index,
            collect_trace=False,
            frozen_policy_measurement=True,
        )
        if not isinstance(result, FrozenPolicyPeriodObservation):
            raise RuntimeError("Measurement observation is missing. / 测量观测丢失。")
        self._frozen_draw_source_mode = "supplied"
        if active_path is not None:
            active_path.completed_periods += 1
        return result

    def run_next_measurement_period_for_controller(
        self,
        controller_token: object,
    ) -> FrozenPolicyPeriodObservation:
        """Controller-owned frozen-policy measurement call.

        controller 专用固定策略测量调用。
        """

        return self._run_next_frozen_policy_period(controller_token)

    def _run_next_frozen_policy_period(
        self,
        controller_token: object | None,
    ) -> FrozenPolicyPeriodObservation:
        """Shared measurement implementation with optional ownership proof.

        带可选所有权证明的测量实现。
        """

        if self._active_frozen_supplied_path is not None:
            raise RuntimeError(
                "A reversible supplied path cannot consume internal draws. / "
                "可逆外部抽样路径不能消耗内部随机数。"
            )
        if self._frozen_draw_source_mode == "supplied":
            raise RuntimeError(
                "Cannot switch a supplied-draw continuation to internal draws. / "
                "不能把外部抽样续跑切换为内部抽样。"
            )
        result = self._run_period_with_draw_suppliers(
            lambda: self.streams.draw_noise_order(
                self.parameters.noise_std
            ),
            lambda: self.streams.draw_next_value_index(
                len(self.value_grid)
            ),
            collect_trace=False,
            frozen_policy_measurement=True,
            controller_token=controller_token,
        )
        if not isinstance(result, FrozenPolicyPeriodObservation):
            raise RuntimeError("Measurement observation is missing. / 测量观测丢失。")
        self._frozen_draw_source_mode = "internal"
        return result

    def run_period_with_supplied_draws_for_test(
        self,
        noise_order_u: float,
        next_value_index: int,
    ) -> RandomizedPeriodTrace:
        """Test-only bridge used to compare exactly with Step 25. / 仅测试使用，用于与第 25 步逐项对比。"""

        trace = self._run_period_with_draw_suppliers(
            lambda: noise_order_u,
            lambda: next_value_index,
            collect_trace=True,
        )
        if not isinstance(trace, RandomizedPeriodTrace):
            raise RuntimeError("Trace collection unexpectedly failed. / 轨迹收集意外失败。")
        return trace

    def all_random_states(self) -> tuple[object, ...]:
        """Return all seven states for same-seed verification. / 返回七条随机流状态供同种子核对。"""

        return (
            *self.streams.environment_states(),
            self.traders[0].mode_random_generator.getstate(),
            self.traders[0].action_random_generator.getstate(),
            self.traders[1].mode_random_generator.getstate(),
            self.traders[1].action_random_generator.getstate(),
        )

    def install_restored_training_position(
        self,
        *,
        period_number: int,
        previous_price: float,
        previous_value: float,
        current_value: float,
        all_seven_rng_states: tuple[object, ...],
        controller_token: object,
    ) -> None:
        """Install a verified between-period training position once.

        一次性安装经过核对的“两期之间”训练位置。

        Step 36C first rebuilds all objects at temporary period zero so new
        callbacks and ownership tokens can be connected safely. This method
        then moves that new object to the saved period without running a market
        transaction or consuming a random draw. / 第 36C 步先在临时第 0 期重建
        全部对象，以便安全连接新的 callback 与所有权 token；随后本方法在不运行
        市场交易、不抽随机数的情况下，把新对象移到保存时期。
        """

        self._check_controller_token(controller_token)
        if self.period_number != 0 or self.execution_mode != "training":
            raise RuntimeError("Restore may be installed only on a fresh training object. / 只能在新的训练对象上安装恢复位置。")
        if self._frozen_policy_action_indexes is not None or self._active_frozen_supplied_path is not None:
            raise RuntimeError("A training restore cannot contain frozen-path state. / 训练恢复不能含固定策略短路径状态。")
        if self.market_maker.has_active_append_transaction:
            raise RuntimeError(
                "A training restore cannot contain an active maker rollback "
                "transaction. / 训练恢复不能含活动的做市商回滚事务。"
            )
        if isinstance(period_number, bool) or not isinstance(period_number, int) or period_number < 1:
            raise ValueError("Saved period_number must be a positive integer. / 保存的 period_number 必须是正整数。")
        scalar_values = (previous_price, previous_value, current_value)
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in scalar_values):
            raise TypeError("Saved market scalars must be real. / 保存的市场标量必须是实数。")
        if not all(isfinite(float(value)) for value in scalar_values):
            raise ValueError("Saved market scalars must be finite. / 保存的市场标量必须有限。")
        if float(previous_value) not in self.value_grid or float(current_value) not in self.value_grid:
            raise ValueError("Saved values must belong to V. / 保存的价值必须属于 V。")
        if sum(self.shared_value_visit_counts) != period_number:
            raise ValueError("Visit counters do not sum to the saved period. / 访问计数之和与保存时期不同。")
        expected_appends = self.parameters.market_maker_window + period_number
        if self.market_maker.successful_append_count != expected_appends:
            raise ValueError("Market-maker append count disagrees with the saved period. / 做市商追加次数与保存时期不同。")
        latest_row = self.market_maker.snapshot()[-1]
        if (
            latest_row.fundamental_value_v != float(previous_value)
            or latest_row.market_price_p != float(previous_price)
        ):
            raise ValueError("Latest maker row disagrees with the saved market position. / 最新做市商记录与保存市场位置不同。")
        if not isinstance(all_seven_rng_states, tuple) or len(all_seven_rng_states) != 7:
            raise ValueError("Exactly seven RNG states are required. / 必须恰好提供七条随机状态。")

        generators = (
            self.streams.initial_state_generator,
            self.streams.value_generator,
            self.streams.noise_generator,
            self.traders[0].mode_random_generator,
            self.traders[0].action_random_generator,
            self.traders[1].mode_random_generator,
            self.traders[1].action_random_generator,
        )
        if len({id(generator) for generator in generators}) != 7:
            raise RuntimeError("The seven restored RNG objects must be independent. / 七个恢复随机对象必须相互独立。")

        # Validate every state on disposable generators before committing any
        # mutation. / 提交任何修改前，先在一次性随机对象上验证全部状态。
        for state in all_seven_rng_states:
            probe = random.Random()
            try:
                probe.setstate(deepcopy(state))
            except (TypeError, ValueError) as error:
                raise ValueError("A saved RNG state is invalid. / 某条保存随机状态无效。") from error

        for generator, state in zip(generators, all_seven_rng_states, strict=True):
            generator.setstate(deepcopy(state))
        self.previous_price = float(previous_price)
        self.previous_value = float(previous_value)
        self.current_value = float(current_value)
        self.period_number = period_number
        if self.all_random_states() != all_seven_rng_states:
            raise RuntimeError("Restored RNG states differ after installation. / 安装后随机状态不同。")


def build_randomized_paper_session(
    *,
    parameters: PaperParameters,
    value_grid: Sequence[float],
    price_grid: Sequence[Sequence[float]],
    action_multipliers: Sequence[float],
    initial_q_table: np.ndarray,
    prehistory: SyntheticMarketMakerPrehistory,
    experiment_seed: int,
    experiment_cell_key: str,
    session_index: int,
) -> RandomizedMarketSession:
    """Factory: create one fully validated session from its seed identity.

    工厂函数：根据实验种子和 session 编号建立完整、已验证的 session。
    """

    manifest = build_session_seed_manifest(
        experiment_seed,
        experiment_cell_key,
        session_index,
    )
    streams = SessionRandomStreams(manifest)
    initial_state_indexes = streams.draw_initial_state(
        price_grid,
        value_grid,
    )
    traders = streams.build_traders(initial_q_table)
    maker = preload_rolling_market_maker(prehistory)
    counts = initialize_value_visit_counts(len(value_grid))
    return RandomizedMarketSession(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        traders=traders,
        market_maker=maker,
        shared_value_visit_counts=counts,
        streams=streams,
        initial_state_indexes=initial_state_indexes,
    )


def run_short_session(
    *,
    parameters: PaperParameters,
    value_grid: Sequence[float],
    price_grid: Sequence[Sequence[float]],
    action_multipliers: Sequence[float],
    initial_q_table: np.ndarray,
    prehistory: SyntheticMarketMakerPrehistory,
    experiment_seed: int,
    experiment_cell_key: str,
    session_index: int,
    number_of_periods: int,
) -> tuple[RandomizedMarketSession, ShortSessionTrace]:
    """Run a tiny trace for validation, not a paper experiment. / 运行用于验证的短轨迹，不是正式论文实验。"""

    if (
        isinstance(number_of_periods, bool)
        or not isinstance(number_of_periods, int)
        or number_of_periods < 1
    ):
        raise ValueError("number_of_periods must be positive. / 时期数量必须为正整数。")
    session = build_randomized_paper_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        initial_q_table=initial_q_table,
        prehistory=prehistory,
        experiment_seed=experiment_seed,
        experiment_cell_key=experiment_cell_key,
        session_index=session_index,
    )
    periods = tuple(
        session.run_next_random_period()
        for _ in range(number_of_periods)
    )
    trace = ShortSessionTrace(
        seed_manifest=session.streams.manifest,
        initial_state_indexes=session.initial_state_indexes,
        periods=periods,
        final_value_visit_counts=tuple(
            session.shared_value_visit_counts
        ),
    )
    return session, trace


def main() -> None:
    """Run reproducibility, independence, distribution, and integration checks.

    运行可复现性、独立性、分布和整合检查。
    """

    parameters = PaperParameters()
    (
        value_grid,
        price_grid,
        action_multipliers,
        initial_q_table,
        prehistory,
    ) = build_paper_inputs(parameters)

    experiment_seed = 20_260_828
    experiment_cell_key = "baseline_low_noise|A3=nash"
    session_index = 0
    manifest = build_session_seed_manifest(
        experiment_seed,
        experiment_cell_key,
        session_index,
    )

    # Golden vector: if seed derivation changes accidentally, this fails loudly.
    # These constants come from the documented SHA-256 algorithm, not from a
    # hidden author seed. / 黄金向量：若种子推导算法被意外改变，测试会立即失败；
    # 这些常数来自我们公开的 SHA-256 算法，不是作者隐藏种子。
    expected_golden_seeds = (
        12_529_419_287_348_292_412,
        12_882_879_626_787_795_628,
        13_337_810_911_989_636_088,
        8_647_772_052_303_364_739,
        1_168_921_028_552_616_311,
        7_185_050_218_830_501_798,
        15_882_235_836_082_817_352,
        4_988_201_117_946_133_787,
        11_264_650_842_863_659_048,
    )
    assert (
        manifest.experiment_cell_seed,
        manifest.session_seed,
        *manifest.child_seeds(),
    ) == expected_golden_seeds
    forward_derivation = {
        label: derive_named_stream_seed(manifest.session_seed, label)
        for label in STREAM_LABELS
    }
    reverse_derivation = {
        label: derive_named_stream_seed(manifest.session_seed, label)
        for label in reversed(STREAM_LABELS)
    }
    assert forward_derivation == reverse_derivation

    # The current kernel is genuinely two-agent code. Reject I=3 before any
    # mutable state changes; later comparative-I work must generalize the core
    # explicitly. / 当前核心确实只支持两位；I=3 必须在任何状态改变前被拒绝，
    # 以后比较不同 I 时必须明确泛化核心。
    rejected_streams = SessionRandomStreams(manifest)
    rejected_initial_indexes = rejected_streams.draw_initial_state(
        price_grid,
        value_grid,
    )
    rejected_traders = rejected_streams.build_traders(initial_q_table)
    rejected_maker = preload_rolling_market_maker(prehistory)
    rejected_counts = initialize_value_visit_counts(len(value_grid))
    rejected_rng_before = (
        rejected_streams.environment_states(),
        tuple(
            (
                trader.mode_random_generator.getstate(),
                trader.action_random_generator.getstate(),
            )
            for trader in rejected_traders
        ),
    )
    rejected_q_before = tuple(
        trader.q_table.copy()
        for trader in rejected_traders
    )
    rejected_history_before = rejected_maker.snapshot()
    try:
        RandomizedMarketSession(
            parameters=PaperParameters(num_speculators=3),
            value_grid=value_grid,
            price_grid=price_grid,
            action_multipliers=action_multipliers,
            traders=rejected_traders,
            market_maker=rejected_maker,
            shared_value_visit_counts=rejected_counts,
            streams=rejected_streams,
            initial_state_indexes=rejected_initial_indexes,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("I=3 was incorrectly accepted. / I=3 被错误接受。")
    assert rejected_counts == [0] * len(value_grid)
    assert rejected_maker.snapshot() == rejected_history_before
    assert all(
        np.array_equal(trader.q_table, old_q)
        for trader, old_q in zip(
            rejected_traders,
            rejected_q_before,
            strict=True,
        )
    )
    assert (
        rejected_streams.environment_states(),
        tuple(
            (
                trader.mode_random_generator.getstate(),
                trader.action_random_generator.getstate(),
            )
            for trader in rejected_traders
        ),
    ) == rejected_rng_before

    # The paper uses 1,000 independent sessions. Audit the complete planned
    # seed set once; this does not run the sessions. / 论文每个实验使用 1,000 个
    # 独立 session；这里一次性核对全部计划种子，但不运行这些 session。
    planned_manifests = tuple(
        build_session_seed_manifest(
            experiment_seed,
            experiment_cell_key,
            planned_index,
        )
        for planned_index in range(1_000)
    )
    planned_session_seeds = tuple(
        planned.session_seed
        for planned in planned_manifests
    )
    planned_child_seeds = tuple(
        child_seed
        for planned in planned_manifests
        for child_seed in planned.child_seeds()
    )
    assert len(set(planned_session_seeds)) == 1_000
    assert len(set(planned_child_seeds)) == 7_000
    assert not set(planned_session_seeds).intersection(planned_child_seeds)

    # Same seed and session index must reproduce the full economic trace.
    # / 相同实验种子与 session 编号必须复现完整经济轨迹。
    same_session_a, trace_a = run_short_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        initial_q_table=initial_q_table,
        prehistory=prehistory,
        experiment_seed=experiment_seed,
        experiment_cell_key=experiment_cell_key,
        session_index=session_index,
        number_of_periods=8,
    )
    same_session_b, trace_b = run_short_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        initial_q_table=initial_q_table,
        prehistory=prehistory,
        experiment_seed=experiment_seed,
        experiment_cell_key=experiment_cell_key,
        session_index=session_index,
        number_of_periods=8,
    )
    assert trace_a == trace_b
    assert same_session_a.all_random_states() == same_session_b.all_random_states()
    assert all(
        np.array_equal(trader_a.q_table, trader_b.q_table)
        for trader_a, trader_b in zip(
            same_session_a.traders,
            same_session_b.traders,
            strict=True,
        )
    )
    assert same_session_a.market_maker.snapshot() == same_session_b.market_maker.snapshot()

    # A different session number gets different named streams and a different
    # multi-draw trace. Individual first draws may coincide by chance. / 不同
    # session 编号得到不同命名流和不同的多次抽样轨迹；单个首次抽样仍可能巧合相同。
    different_session, different_trace = run_short_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        initial_q_table=initial_q_table,
        prehistory=prehistory,
        experiment_seed=experiment_seed,
        experiment_cell_key=experiment_cell_key,
        session_index=1,
        number_of_periods=8,
    )
    assert manifest.child_seeds() != different_trace.seed_manifest.child_seeds()
    assert trace_a.periods != different_trace.periods

    other_cell_manifest = build_session_seed_manifest(
        experiment_seed,
        "baseline_high_noise|A3=nash",
        session_index,
    )
    assert other_cell_manifest.experiment_cell_seed != manifest.experiment_cell_seed
    assert other_cell_manifest.child_seeds() != manifest.child_seeds()

    # Cross-consumption: using one named stream cannot move another stream.
    # / 交叉消耗检查：使用一条命名流不能推进另一条流。
    global_random_state_before = random.getstate()
    isolation_a = SessionRandomStreams(manifest)
    isolation_b = SessionRandomStreams(manifest)
    for _ in range(1_000):
        isolation_a.draw_noise_order(parameters.noise_std)
    value_path_a = tuple(
        isolation_a.draw_next_value_index(len(value_grid))
        for _ in range(100)
    )
    value_path_b = tuple(
        isolation_b.draw_next_value_index(len(value_grid))
        for _ in range(100)
    )
    assert value_path_a == value_path_b

    isolation_c = SessionRandomStreams(manifest)
    isolation_d = SessionRandomStreams(manifest)
    for _ in range(1_000):
        isolation_c.draw_next_value_index(len(value_grid))
    noise_path_c = tuple(
        isolation_c.draw_noise_order(parameters.noise_std)
        for _ in range(100)
    )
    noise_path_d = tuple(
        isolation_d.draw_noise_order(parameters.noise_std)
        for _ in range(100)
    )
    assert noise_path_c == noise_path_d

    isolation_e = SessionRandomStreams(manifest)
    isolation_f = SessionRandomStreams(manifest)
    for _ in range(100):
        isolation_e.draw_noise_order(parameters.noise_std)
        isolation_e.draw_next_value_index(len(value_grid))
    assert isolation_e.draw_initial_state(
        price_grid,
        value_grid,
    ) == isolation_f.draw_initial_state(price_grid, value_grid)
    assert random.getstate() == global_random_state_before

    # Separate trader mode and action streams: extra action draws do not move
    # the future mode sequence. / trader 的模式流与动作流分开：额外动作抽签不会
    # 推进未来模式序列。
    toy_q = np.array([[1.0, 9.0, 2.0]], dtype=float)
    split_a = InformedQTrader(
        "split A",
        toy_q,
        manifest.trader_1_mode_seed,
        manifest.trader_1_action_tie_seed,
    )
    split_b = InformedQTrader(
        "split B",
        toy_q,
        manifest.trader_1_mode_seed,
        manifest.trader_1_action_tie_seed,
    )
    split_b_action_state_before = split_b.action_random_generator.getstate()
    for _ in range(100):
        split_a.choose_action(0, 1.0)  # mode + random action / 模式 + 随机动作
        split_b.choose_action(0, 0.0)  # mode only; unique best / 只抽模式
    assert split_a.mode_random_generator.getstate() == split_b.mode_random_generator.getstate()
    assert split_b.action_random_generator.getstate() == split_b_action_state_before
    assert split_a.action_random_generator.getstate() != split_b.action_random_generator.getstate()

    tied_q = np.array([[9.0, 9.0, 1.0]], dtype=float)
    tied_trader = InformedQTrader(
        "tied",
        tied_q,
        manifest.trader_1_mode_seed,
        manifest.trader_1_action_tie_seed,
    )
    tied_action_state_before = tied_trader.action_random_generator.getstate()
    tied_decision = tied_trader.choose_action(0, 0.0)
    assert tied_decision.mode == "exploitation"
    assert tied_decision.action_index in (0, 1)
    assert tied_trader.action_random_generator.getstate() != tied_action_state_before

    cross_streams = SessionRandomStreams(manifest)
    cross_traders = cross_streams.build_traders(toy_q)
    cross_environment_before = cross_streams.environment_states()
    trader_2_states_before = (
        cross_traders[1].mode_random_generator.getstate(),
        cross_traders[1].action_random_generator.getstate(),
    )
    cross_traders[0].choose_action(0, 1.0)
    assert cross_streams.environment_states() == cross_environment_before
    assert (
        cross_traders[1].mode_random_generator.getstate(),
        cross_traders[1].action_random_generator.getstate(),
    ) == trader_2_states_before

    # Distribution smoke tests use fresh streams and do not touch live sessions.
    # / 分布抽样检查使用全新随机流，不触碰正式 session。
    distribution_streams = SessionRandomStreams(manifest)
    value_draw_count = 100_000
    value_counts = [0] * len(value_grid)
    for _ in range(value_draw_count):
        value_counts[
            distribution_streams.draw_next_value_index(len(value_grid))
        ] += 1
    target_count = value_draw_count / len(value_grid)
    assert max(
        abs(count - target_count)
        for count in value_counts
    ) < 0.05 * target_count

    noise_sample = np.array(
        [
            distribution_streams.draw_noise_order(parameters.noise_std)
            for _ in range(100_000)
        ],
        dtype=float,
    )
    assert abs(float(np.mean(noise_sample))) < 0.02 * parameters.noise_std
    assert abs(
        float(np.std(noise_sample)) - parameters.noise_std
    ) < 0.02 * parameters.noise_std

    # One supplied-draw hot period must match the readable Step 25 oracle.
    # / 精简路径在固定抽样下必须与可读的第 25 步完全一致。
    parity_streams = SessionRandomStreams(manifest)
    parity_initial_indexes = parity_streams.draw_initial_state(
        price_grid,
        value_grid,
    )
    parity_traders_hot = parity_streams.build_traders(initial_q_table)
    parity_traders_oracle = build_two_informed_traders(
        initial_q_table,
        random_seeds=(
            manifest.trader_1_mode_seed,
            manifest.trader_2_mode_seed,
        ),
        action_random_seeds=(
            manifest.trader_1_action_tie_seed,
            manifest.trader_2_action_tie_seed,
        ),
    )
    parity_counts_hot = initialize_value_visit_counts(len(value_grid))
    parity_counts_oracle = initialize_value_visit_counts(len(value_grid))
    parity_maker_hot = preload_rolling_market_maker(prehistory)
    parity_maker_oracle = preload_rolling_market_maker(prehistory)
    parity_session = RandomizedMarketSession(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        traders=parity_traders_hot,
        market_maker=parity_maker_hot,
        shared_value_visit_counts=parity_counts_hot,
        streams=parity_streams,
        initial_state_indexes=parity_initial_indexes,
    )
    fixed_noise = 0.05
    fixed_next_index = 3
    hot_trace = parity_session.run_period_with_supplied_draws_for_test(
        fixed_noise,
        fixed_next_index,
    )
    oracle_receipt = run_one_market_period(
        period_number=0,
        previous_price_p=price_grid[parity_initial_indexes[1]][parity_initial_indexes[0]],
        previous_value_v=value_grid[parity_initial_indexes[1]],
        current_value_v=value_grid[parity_initial_indexes[2]],
        next_value_v=value_grid[fixed_next_index],
        noise_order_u=fixed_noise,
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        traders=parity_traders_oracle,
        shared_value_visit_counts=parity_counts_oracle,
        market_maker=parity_maker_oracle,
    )
    assert hot_trace.epsilon == oracle_receipt.epsilon
    assert hot_trace.decisions == tuple(
        result.action_decision
        for result in oracle_receipt.trader_results
    )
    assert hot_trace.raw_orders_x == tuple(
        result.raw_order_x
        for result in oracle_receipt.trader_results
    )
    assert hot_trace.total_order_flow_y == oracle_receipt.total_order_flow_y
    assert hot_trace.continuous_price_p == oracle_receipt.adaptive_price_quote.continuous_price_p_hat
    assert hot_trace.insensitive_order_z == oracle_receipt.information_insensitive_order_z
    assert hot_trace.profits == tuple(
        result.q_update.realized_profit
        for result in oracle_receipt.trader_results
    )
    assert hot_trace.next_state_indexes == oracle_receipt.realized_next_state_indexes
    assert parity_counts_hot == parity_counts_oracle
    assert all(
        np.array_equal(hot.q_table, oracle.q_table)
        for hot, oracle in zip(
            parity_traders_hot,
            parity_traders_oracle,
            strict=True,
        )
    )
    assert parity_maker_hot.snapshot() == parity_maker_oracle.snapshot()
    assert tuple(
        (
            trader.mode_random_generator.getstate(),
            trader.action_random_generator.getstate(),
        )
        for trader in parity_traders_hot
    ) == tuple(
        (
            trader.mode_random_generator.getstate(),
            trader.action_random_generator.getstate(),
        )
        for trader in parity_traders_oracle
    )

    # Receipt-free mode must produce the same mutable session state without
    # allocating one trace per period. / 无轨迹模式不能每期分配流水单，但最终可变
    # session 状态必须与记录轨迹的版本相同。
    no_trace_session = build_randomized_paper_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        initial_q_table=initial_q_table,
        prehistory=prehistory,
        experiment_seed=experiment_seed,
        experiment_cell_key=experiment_cell_key,
        session_index=session_index,
    )
    for _ in range(8):
        assert no_trace_session.run_next_random_period_without_trace() is None
    assert no_trace_session.period_number == 8
    assert no_trace_session.shared_value_visit_counts == same_session_a.shared_value_visit_counts
    assert no_trace_session.all_random_states() == same_session_a.all_random_states()
    assert no_trace_session.market_maker.snapshot() == same_session_a.market_maker.snapshot()
    assert all(
        np.array_equal(no_trace.q_table, traced.q_table)
        for no_trace, traced in zip(
            no_trace_session.traders,
            same_session_a.traders,
            strict=True,
        )
    )

    # Full Q validation happened only at session construction, not in 8 periods.
    # / 完整 Q 检查只在 session 建立时发生一次，不会在八期中重复。
    assert same_session_a.full_q_validation_count == 1
    assert same_session_b.full_q_validation_count == 1
    assert different_session.full_q_validation_count == 1
    assert no_trace_session.full_q_validation_count == 1

    print("Step 26: Reproducible random streams / 步骤 26：可复现随机流")
    print(f"Seed method / 种子方法: {manifest.seed_derivation_version}")
    print(f"Experiment seed / 实验种子: {manifest.experiment_seed}")
    print(f"Experiment cell / 实验单元: {manifest.experiment_cell_key}")
    print(f"Experiment-cell seed / 实验单元种子: {manifest.experiment_cell_seed}")
    print(f"Session index / session 编号: {manifest.session_index}")
    print(f"Derived session seed / 派生 session 种子: {manifest.session_seed}")
    print("Seven child seeds / 七个子种子:")
    for label, seed in zip(STREAM_LABELS, manifest.child_seeds(), strict=True):
        print(f"  {label}: {seed}")
    print(
        "Initial state / 初始状态: "
        f"{trace_a.initial_state_indexes}"
    )
    print("First three random periods / 前三个随机时期:")
    for period in trace_a.periods[:3]:
        print(
            f"  t={period.period_number}: v_index={period.current_value_index}, "
            f"actions=({period.decisions[0].action_index},"
            f"{period.decisions[1].action_index}), "
            f"u={period.noise_order_u:.6f}, "
            f"p={period.continuous_price_p:.6f}, "
            f"next_v_index={period.next_value_index}"
        )
    print(
        "Same seed replay / 同种子重放: identical 8-period market trace"
    )
    print(
        "Different session / 不同 session: different derived streams and trace"
    )
    print(
        "Planned seed audit / 计划种子核对: 1,000 sessions and 7,000 child "
        "streams are unique / 1,000 个 session 与 7,000 条子流均不重复"
    )
    print(
        "Cross-consumption / 交叉消耗: noise, value, initial-state, and trader "
        "streams stayed isolated / 各随机流互不推进"
    )
    print(
        "Step 25 parity / 与第 25 步一致: fixed-draw hot period matched exactly"
    )
    print(
        "Full-Q validation count / 完整 Q 检查次数: 1 per session, not per period"
    )
    print(
        "Receipt-free hot path / 无轨迹高频路径: identical state without per-period trace objects"
    )
    print(
        "A4 status / A4 状态: explicit replication choice; the paper provides no seeds"
        " / 明确复现选择；论文未提供种子"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
