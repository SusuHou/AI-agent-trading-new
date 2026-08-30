"""Step 25: run one complete market period in the paper's causal order.

步骤 25：按照论文的因果顺序运行一个完整市场时期。

Run / 运行:
    py -3 -X utf8 steps/step_25_one_market_period.py

One-period protocol / 单期流程:
    given s_t=(p_(t-1),v_(t-1),v_t) and past-only D_t
    / 给定状态 s_t 和只含过去记录的 D_t

    -> compute epsilon from past visits to v_t
    -> both informed Q-traders choose independently
    -> record one system visit to v_t
    -> receive noise order u_t (not observed by traders beforehand)
    -> y_t = x_1,t + x_2,t + u_t
    -> estimate the maker's OLS from old D_t and set continuous p_t
    -> z_t = -xi(p_t-v_bar), and profits realize
    -> append completed (v_t,p_t,z_t,y_t) to form D_(t+1)
    -> form s_(t+1) using the supplied v_(t+1)
    -> each trader updates exactly one Q-cell using E_v'[max Q]

    / 先根据过去访问次数算 epsilon；两位知情 AI 独立选择；系统为本期价值计数
    一次；随后噪声订单到达；计算 y；做市商只用旧历史估计并给出连续价格；价格
    出现后才计算 z 与利润；完整本期记录进入下一期历史；形成下一状态；最后每位
    交易者使用 E_v'[max Q] 只更新一个 Q 格子。

This step deliberately receives fixed u_t and v_(t+1), making the complete
trace repeatable and hand-checkable. Step 26 will supply them from independent
random streams. / 本步骤故意接收固定的 u_t 与 v_(t+1)，让完整流程可重复、可手算；
第 26 步才会由独立随机数流提供它们。

This readable diagnostic version validates the complete Q-tables before a
period. Do not use that full-table check inside a 100-million-period hot loop;
Step 26 will validate a session once and keep the repeated path lean. / 这个易读
诊断版本会在一期开始前检查整张 Q 表。不能在一亿期的高频循环中每期做全表检查；
第 26 步会让每个 session 只完整验证一次，并保持重复路径精简。

The continuous price is used for z, profits, and market-maker history. It is
mapped to P only for the next finite Q-state, following explicit decision A2.
/ 连续价格用于 z、利润和做市商历史；只有建立下一个有限 Q 状态时才按照明确决定
A2 映射到价格网格 P。
"""

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass
from math import exp, isclose, isfinite
from numbers import Real
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from src.step01_value_grid import build_value_grid, discrete_value_std
from step_03_total_order_flow import calculate_total_order_flow
from step_04_information_insensitive_investors import (
    calculate_insensitive_order,
)
from step_05_speculator_profit import calculate_profit
from step_10_fixed_point_solver import solve_benchmark_fixed_point
from step_12_action_grid import (
    build_action_multiplier_grid,
    calculate_orders_for_value,
)
from step_14_state_representation import (
    build_paper_price_grids,
    build_state_indexes,
    continuous_price_to_index,
    encode_state_index,
    fundamental_value_to_index,
    number_of_price_points,
    number_of_states,
    validate_price_grids_by_value,
)
from step_16_initial_q_table import build_initial_q_table
from step_18_epsilon_greedy_action import ActionDecision
from step_19_value_specific_epsilon import initialize_value_visit_counts
from step_20_q_learning_update import (
    calculate_q_value_from_continuation,
    expected_continuation_over_next_values,
)
from step_21_two_independent_q_traders import (
    InformedQTrader,
    QUpdateRecord,
    build_two_informed_traders,
    choose_actions_for_one_shared_period,
)
from step_22_market_maker_rolling_history import MarketObservation
from step_23_market_maker_ols import (
    MarketMakerOLSEstimates,
    fit_market_maker_regressions,
)
from step_24_adaptive_market_maker_price import (
    AdaptivePriceQuote,
    calculate_adaptive_price_quote,
)
from step_24b_fast_rolling_ols import RollingMarketMakerOLS
from step_24c_initial_market_maker_history import (
    SyntheticMarketMakerPrehistory,
    build_synthetic_market_maker_prehistory,
    preload_rolling_market_maker,
)


@dataclass(frozen=True)
class TraderPeriodResult:
    """One trader's immutable action, order, profit, and Q-update receipt.

    一位交易者本期动作、订单、利润和 Q 更新的不可修改记录。
    """

    trader_name: str
    action_decision: ActionDecision
    raw_order_x: float
    q_update: QUpdateRecord


@dataclass(frozen=True)
class MarketPeriodReceipt:
    """An immutable audit trail for one completed period.

    一个完整市场时期的不可修改审计流水单。
    """

    period_number: int
    previous_price_p: float
    previous_value_v: float
    current_value_v: float
    current_state_indexes: tuple[int, int, int]
    current_state_id: int
    current_value_visit_count_before: int
    current_value_visit_count_after: int
    epsilon: float
    trader_results: tuple[TraderPeriodResult, TraderPeriodResult]
    noise_order_u: float
    total_order_flow_y: float
    market_maker_estimates_from_prior_history: MarketMakerOLSEstimates
    adaptive_price_quote: AdaptivePriceQuote
    information_insensitive_order_z: float
    appended_history_row: MarketObservation
    evicted_history_row: MarketObservation
    history_size_before: int
    history_size_after: int
    next_value_v: float
    next_price_index: int
    next_price_was_clipped: bool
    realized_next_state_indexes: tuple[int, int, int]
    realized_next_state_id: int
    possible_next_state_ids_for_q_update: tuple[int, ...]


def _finite_real(number: float, label: str) -> float:
    """Return one finite float with a beginner-readable error. / 检查并返回有限浮点数。"""

    if isinstance(number, bool) or not isinstance(number, Real):
        raise TypeError(f"{label} must be a real number. / {label} 必须是实数。")
    converted = float(number)
    if not isfinite(converted):
        raise ValueError(f"{label} must be finite. / {label} 必须是有限数。")
    return converted


def _validated_grid(
    grid: Sequence[float],
    expected_size: int,
    label: str,
    *,
    require_increasing: bool,
) -> tuple[float, ...]:
    """Validate one supplied paper grid before the market can mutate.

    在市场发生任何修改之前，检查一个传入的论文网格。
    """

    if len(grid) != expected_size:
        raise ValueError(
            f"{label} must contain exactly {expected_size} points. / "
            f"{label} 必须恰好包含 {expected_size} 个点。"
        )
    converted = tuple(
        _finite_real(value, f"{label}[{index}]")
        for index, value in enumerate(grid)
    )
    if require_increasing and any(
        right <= left
        for left, right in zip(converted, converted[1:])
    ):
        raise ValueError(
            f"{label} must be strictly increasing with no duplicates. / "
            f"{label} 必须严格递增且不能重复。"
        )
    return converted


def run_one_market_period(
    *,
    period_number: int,
    previous_price_p: float,
    previous_value_v: float,
    current_value_v: float,
    next_value_v: float,
    noise_order_u: float,
    parameters: PaperParameters,
    value_grid: Sequence[float],
    price_grid: Sequence[Sequence[float]],
    action_multipliers: Sequence[float],
    traders: tuple[InformedQTrader, InformedQTrader],
    shared_value_visit_counts: list[int],
    market_maker: RollingMarketMakerOLS,
) -> MarketPeriodReceipt:
    """Mutate one market system through exactly one completed period.

    让一个市场系统恰好运行并完成一个时期。

    The fixed next value is accepted for deterministic testing, but it is not
    economically used when actions, current price, z, or profits are computed.
    / 为了确定性测试，函数接收固定下一价值；但计算动作、本期价格、z 或利润时，
    它不会进入任何经济计算。
    """

    if (
        isinstance(period_number, bool)
        or not isinstance(period_number, int)
        or period_number < 0
    ):
        raise ValueError("period_number must be a non-negative integer. / period_number 必须是非负整数。")
    if not isinstance(parameters, PaperParameters):
        raise TypeError("parameters must be PaperParameters. / parameters 类型错误。")
    if parameters.num_speculators != 2:
        raise ValueError("Step 25 currently implements the paper baseline I=2. / 第 25 步目前实现论文基准 I=2。")
    if not isinstance(traders, tuple) or len(traders) != 2:
        raise ValueError("Exactly two informed traders are required. / 必须恰好有两位知情交易者。")
    if traders[0] is traders[1]:
        raise ValueError("The traders must be different objects. / 两位交易者必须是不同对象。")
    if not isinstance(market_maker, RollingMarketMakerOLS):
        raise TypeError("market_maker has the wrong type. / market_maker 类型错误。")
    if market_maker.window_size != parameters.market_maker_window:
        raise ValueError(
            "The maker window must equal parameter T_m. / 做市商窗口必须等于参数 T_m。"
        )
    if not market_maker.is_full:
        raise ValueError("The market maker needs a full prior D_t. / 做市商需要完整的旧 D_t。")

    previous_price = _finite_real(previous_price_p, "previous price / 上期价格")
    previous_value = _finite_real(previous_value_v, "previous value / 上期价值")
    current_value = _finite_real(current_value_v, "current value / 本期价值")
    supplied_next_value = _finite_real(next_value_v, "next value / 下一期价值")
    fixed_noise_order = _finite_real(noise_order_u, "noise order / 噪声订单")
    values = _validated_grid(
        value_grid,
        parameters.num_value_points,
        "value grid V / 价值网格 V",
        require_increasing=True,
    )
    prices_by_value = validate_price_grids_by_value(
        price_grid,
        parameters.num_value_points,
        parameters.num_price_points,
    )
    number_of_prices = number_of_price_points(prices_by_value)
    multipliers = _validated_grid(
        action_multipliers,
        parameters.num_action_points,
        "action multiplier grid X / 动作乘数网格 X",
        require_increasing=True,
    )
    if len(shared_value_visit_counts) != len(values):
        raise ValueError("Visit counts must have one entry per value. / 每个价值必须对应一个访问计数。")
    if any(
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for count in shared_value_visit_counts
    ):
        raise ValueError("Every visit count must be a non-negative integer. / 每个访问计数必须是非负整数。")

    expected_state_count = number_of_states(number_of_prices, len(values))
    for trader in traders:
        if not isinstance(trader, InformedQTrader):
            raise TypeError("Every trader must be InformedQTrader. / 每位交易者都必须是 InformedQTrader。")
        if trader.q_table.shape != (expected_state_count, len(multipliers)):
            raise ValueError("Trader Q-table shape does not match P x V x V x X. / Q 表形状与 P x V x V x X 不匹配。")
        if not np.issubdtype(trader.q_table.dtype, np.floating):
            raise TypeError("Each Q-table must use floating-point values. / 每张 Q 表必须使用浮点数。")
        if not np.isfinite(trader.q_table).all():
            raise ValueError("Every Q-value must be finite before a period starts. / 每期开始前所有 Q 值都必须有限。")
        if not trader.q_table.flags.writeable:
            raise ValueError("Each Q-table must be writable for learning. / 每张 Q 表都必须可写才能学习。")
    if np.shares_memory(traders[0].q_table, traders[1].q_table):
        raise ValueError("The two traders cannot share Q-table memory. / 两位交易者不能共享 Q 表内存。")
    if traders[0].random_generator is traders[1].random_generator:
        raise ValueError("The two traders need private random generators. / 两位交易者必须拥有独立随机数生成器。")
    if traders[0].action_random_generator is traders[1].action_random_generator:
        raise ValueError("The two traders need private action random generators. / 两位交易者必须拥有独立动作随机数生成器。")
    if (
        traders[0].mode_random_generator is traders[1].action_random_generator
        or traders[1].mode_random_generator is traders[0].action_random_generator
    ):
        raise ValueError("Random streams cannot cross between traders. / 两位交易者之间不能交叉共享随机流。")

    # 0. Preflight every deterministic condition before RNGs, counters, Q, or
    # history can change. This makes rejected input safe. / 0. 在随机数、计数器、
    # Q 表或历史发生变化之前，先检查所有确定性条件；无效输入不会留下半次更新。
    current_state_indexes = build_state_indexes(
        previous_price,
        previous_value,
        current_value,
        prices_by_value,
        values,
    )
    current_state_id = encode_state_index(
        current_state_indexes,
        number_of_prices,
        len(values),
    )
    current_value_index = fundamental_value_to_index(current_value, values)
    realized_next_value_index = fundamental_value_to_index(
        supplied_next_value,
        values,
    )
    raw_order_choices = calculate_orders_for_value(
        current_value,
        parameters.value_mean,
        list(multipliers),
    )
    if not all(isfinite(order) for order in raw_order_choices):
        raise ValueError("Every possible informed order must be finite. / 每个可能的知情订单都必须有限。")

    # OLS is read-only here. If the old history is singular or invalid, fail
    # before either trader draws an action. / 此处 OLS 只读取旧历史；若旧历史无法
    # 回归，要在交易者抽动作之前报错。
    prior_estimates = market_maker.estimates()
    # A magnitude bound checks every possible action pair without looping over
    # all n_x^2 pairs each period. / 使用绝对值上界一次覆盖所有动作组合，避免每期
    # 都循环检查 n_x^2 个组合。
    largest_absolute_order = max(abs(order) for order in raw_order_choices)
    largest_absolute_y = (
        2.0 * largest_absolute_order + abs(fixed_noise_order)
    )
    if not isfinite(largest_absolute_y):
        raise ValueError("Possible order flow can overflow. / 可能的订单流会溢出。")
    boundary_quotes = (
        calculate_adaptive_price_quote(
            -largest_absolute_y,
            prior_estimates,
            parameters.pricing_error_weight,
        ),
        calculate_adaptive_price_quote(
            largest_absolute_y,
            prior_estimates,
            parameters.pricing_error_weight,
        ),
    )
    largest_absolute_price = max(
        abs(quote.continuous_price_p_hat)
        for quote in boundary_quotes
    )
    largest_absolute_z = abs(parameters.investor_slope) * (
        largest_absolute_price + abs(parameters.value_mean)
    )
    largest_absolute_profit = (
        abs(current_value) + largest_absolute_price
    ) * largest_absolute_order
    if not all(
        isfinite(number)
        for number in (largest_absolute_z, largest_absolute_profit)
    ):
        raise ValueError(
            "A possible action pair produces a non-finite outcome. / "
            "某个可能的动作组合产生了非有限结果。"
        )

    # 1. Build s_t. The informed traders know current v_t now. / 1. 建立 s_t；
    # 知情交易者此时已经知道本期 v_t。
    visit_count_before = shared_value_visit_counts[current_value_index]

    # 2. Both traders choose before seeing u_t. The shared count increases once
    # after both decisions. / 2. 两位交易者在看不到 u_t 时选择；两位都选择后共享
    # 计数只增加一次。
    epsilon, decisions = choose_actions_for_one_shared_period(
        traders,
        current_state_id,
        current_value_index,
        shared_value_visit_counts,
        parameters.exploration_decay,
    )
    visit_count_after = shared_value_visit_counts[current_value_index]
    if visit_count_after != visit_count_before + 1:
        raise RuntimeError("Current value must be counted exactly once. / 本期价值必须恰好计数一次。")
    trader_orders = (
        raw_order_choices[decisions[0].action_index],
        raw_order_choices[decisions[1].action_index],
    )

    # 3. Noise arrives only now; then y_t becomes observable. / 3. 噪声现在才
    # 到达，随后 y_t 才能被观察。
    total_order_flow = calculate_total_order_flow(
        trader_orders[0],
        trader_orders[1],
        fixed_noise_order,
    )

    # 4. Estimate only from prior D_t, then insert current y_t. No current z or
    # history row exists yet. / 4. 只用旧 D_t 估计，再代入本期 y_t；此时本期 z
    # 和历史记录都还不存在。
    history_size_before = len(market_maker)
    price_quote = calculate_adaptive_price_quote(
        total_order_flow,
        prior_estimates,
        parameters.pricing_error_weight,
    )
    current_price = price_quote.continuous_price_p_hat

    # 5. Price is observed; only now do z_t and profits exist. / 5. 看到价格后，
    # 才产生 z_t 和利润。
    insensitive_order = calculate_insensitive_order(
        current_price,
        parameters.value_mean,
        parameters.investor_slope,
    )
    profits = (
        calculate_profit(current_value, current_price, trader_orders[0]),
        calculate_profit(current_value, current_price, trader_orders[1]),
    )

    # 6. The completed current row enters D_(t+1); it did not price itself.
    # / 6. 完整本期记录进入 D_(t+1)；它没有参与决定自己的价格。
    completed_row = MarketObservation(
        fundamental_value_v=current_value,
        market_price_p=current_price,
        insensitive_order_z=insensitive_order,
        informed_and_noise_order_y=total_order_flow,
    )
    evicted_row = market_maker.append_completed_observation(completed_row)
    if evicted_row is None:
        raise RuntimeError("A full rolling window must evict one old row. / 完整滚动窗口必须淘汰一条旧记录。")
    history_size_after = len(market_maker)
    if history_size_after != history_size_before:
        raise RuntimeError("A full rolling window must keep size T_m. / 完整滚动窗口必须保持 T_m 行。")

    # 7. Only after current outcomes exist, form the next state. Continuous p_t
    # is discretized here only. / 7. 本期结果出现后才建立下一状态；连续 p_t 只在
    # 这里离散化。
    current_value_price_row = prices_by_value[current_value_index]
    next_price_index = continuous_price_to_index(
        current_price,
        current_value_price_row,
    )
    current_value_index_again = fundamental_value_to_index(current_value, values)
    realized_next_state_indexes = (
        next_price_index,
        current_value_index_again,
        realized_next_value_index,
    )
    realized_next_state_id = encode_state_index(
        realized_next_state_indexes,
        number_of_prices,
        len(values),
    )
    possible_next_state_ids = tuple(
        encode_state_index(
            (next_price_index, current_value_index_again, next_value_index),
            number_of_prices,
            len(values),
        )
        for next_value_index in range(len(values))
    )

    # 8. Each private Q-table changes in exactly one visited cell, using the
    # appendix E_v'[max Q] acceleration. / 8. 每张私有 Q 表只改变被访问的一个
    # 格子，并使用附录的 E_v'[max Q] 加速。
    prepared_q_updates: list[tuple[float, float]] = []
    for trader, decision, profit in zip(
        traders,
        decisions,
        profits,
        strict=True,
    ):
        old_q_value = float(
            trader.q_table[current_state_id, decision.action_index]
        )
        possible_next_q_rows = trader.q_table[
            list(possible_next_state_ids),
            :,
        ].copy()
        expected_continuation = expected_continuation_over_next_values(
            possible_next_q_rows
        )
        new_q_value = calculate_q_value_from_continuation(
            old_q_value,
            profit,
            expected_continuation,
            parameters.learning_rate,
            parameters.discount_factor,
        )
        prepared_q_updates.append((old_q_value, new_q_value))

    # Commit only after both proposed updates have been calculated safely. / 两位
    # 的新 Q 值都安全算好后，才一起写入，避免只更新其中一位的半成品状态。
    q_update_records: list[QUpdateRecord] = []
    for trader, decision, profit, prepared_update in zip(
        traders,
        decisions,
        profits,
        prepared_q_updates,
        strict=True,
    ):
        old_q_value, new_q_value = prepared_update
        trader.q_table[current_state_id, decision.action_index] = new_q_value
        q_update_records.append(
            QUpdateRecord(
                period_number=period_number,
                trader_name=trader.name,
                state_id=current_state_id,
                action_index=decision.action_index,
                decision_mode=decision.mode,
                epsilon=epsilon,
                realized_profit=profit,
                old_q_value=old_q_value,
                new_q_value=new_q_value,
            )
        )

    trader_results = tuple(
        TraderPeriodResult(
            trader_name=trader.name,
            action_decision=decision,
            raw_order_x=raw_order,
            q_update=q_update,
        )
        for trader, decision, raw_order, q_update in zip(
            traders,
            decisions,
            trader_orders,
            q_update_records,
            strict=True,
        )
    )
    return MarketPeriodReceipt(
        period_number=period_number,
        previous_price_p=previous_price,
        previous_value_v=previous_value,
        current_value_v=current_value,
        current_state_indexes=current_state_indexes,
        current_state_id=current_state_id,
        current_value_visit_count_before=visit_count_before,
        current_value_visit_count_after=visit_count_after,
        epsilon=epsilon,
        trader_results=trader_results,  # type: ignore[arg-type]
        noise_order_u=fixed_noise_order,
        total_order_flow_y=total_order_flow,
        market_maker_estimates_from_prior_history=prior_estimates,
        adaptive_price_quote=price_quote,
        information_insensitive_order_z=insensitive_order,
        appended_history_row=completed_row,
        evicted_history_row=evicted_row,
        history_size_before=history_size_before,
        history_size_after=history_size_after,
        next_value_v=supplied_next_value,
        next_price_index=next_price_index,
        next_price_was_clipped=(
            current_price < current_value_price_row[0]
            or current_price > current_value_price_row[-1]
        ),
        realized_next_state_indexes=realized_next_state_indexes,
        realized_next_state_id=realized_next_state_id,
        possible_next_state_ids_for_q_update=possible_next_state_ids,
    )


def build_paper_inputs(
    parameters: PaperParameters,
) -> tuple[
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    np.ndarray,
    SyntheticMarketMakerPrehistory,
]:
    """Build the already-validated paper grids for this integration test.

    为本次整合测试建立前面已经分别验证过的论文网格。
    """

    value_grid_array = build_value_grid(
        parameters.value_mean,
        parameters.value_std,
        parameters.num_value_points,
    )
    value_grid = tuple(float(value) for value in value_grid_array)
    sigma_v_hat = discrete_value_std(
        value_grid_array,
        parameters.value_mean,
    )
    nash_solution = solve_benchmark_fixed_point(
        "nash",
        parameters.num_speculators,
        parameters.noise_std,
        sigma_v_hat,
        parameters.investor_slope,
        parameters.pricing_error_weight,
    )
    cartel_solution = solve_benchmark_fixed_point(
        "cartel",
        parameters.num_speculators,
        parameters.noise_std,
        sigma_v_hat,
        parameters.investor_slope,
        parameters.pricing_error_weight,
    )
    action_multipliers = tuple(
        build_action_multiplier_grid(
            nash_solution["intensity"],
            cartel_solution["intensity"],
            parameters.grid_widening,
            parameters.num_action_points,
        )
    )
    price_grid = tuple(
        tuple(float(price) for price in row)
        for row in build_paper_price_grids(
            parameters,
            value_grid,
            parameters.noise_std,
        )
    )
    initial_q_table = build_initial_q_table(
        price_grid,
        value_grid,
        action_multipliers,
        parameters.value_mean,
        parameters.num_speculators,
        nash_solution["price_impact"],
        parameters.discount_factor,
    )
    prehistory = build_synthetic_market_maker_prehistory(
        parameters,
        benchmark_name="nash",
    )
    return (
        value_grid,
        price_grid,
        action_multipliers,
        initial_q_table,
        prehistory,
    )


def main() -> None:
    """Run, print, and independently recalculate one deterministic period.

    运行、打印并独立重算一个确定性时期。
    """

    parameters = PaperParameters()
    (
        value_grid,
        price_grid,
        action_multipliers,
        initial_q_table,
        prehistory,
    ) = build_paper_inputs(parameters)

    # Test-only inputs. Around 4.6 million past visits makes epsilon about 0.1.
    # Seeds 1 and 2 then both deterministically take exploitation in this trace.
    # / 这些是测试输入。约 460 万次过去访问使 epsilon 约为 0.1；种子 1 和 2
    # 在本次轨迹中都会确定性地进入 exploitation。
    previous_value = value_grid[6]
    previous_price = price_grid[6][parameters.num_price_points // 2]
    current_value_index = 7
    current_value = value_grid[current_value_index]
    realized_next_value = value_grid[3]
    alternate_next_value = value_grid[9]
    fixed_noise_order = 0.05
    past_visit_count = 4_605_170

    traders = build_two_informed_traders(
        initial_q_table,
        random_seeds=(1, 2),
    )
    alternate_traders = build_two_informed_traders(
        initial_q_table,
        random_seeds=(1, 2),
    )
    q_tables_before = tuple(trader.q_table.copy() for trader in traders)
    shared_counts = initialize_value_visit_counts(len(value_grid))
    alternate_shared_counts = initialize_value_visit_counts(len(value_grid))
    shared_counts[current_value_index] = past_visit_count
    alternate_shared_counts[current_value_index] = past_visit_count
    market_maker = preload_rolling_market_maker(prehistory)
    alternate_market_maker = preload_rolling_market_maker(prehistory)

    receipt = run_one_market_period(
        period_number=0,
        previous_price_p=previous_price,
        previous_value_v=previous_value,
        current_value_v=current_value,
        next_value_v=realized_next_value,
        noise_order_u=fixed_noise_order,
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        traders=traders,
        shared_value_visit_counts=shared_counts,
        market_maker=market_maker,
    )

    # Repeat from identical memories but change only realized v_(t+1). Current
    # actions, price, profits, history, and accelerated Q targets must not look
    # ahead to it. / 从相同记忆重跑，只改变实际 v_(t+1)。本期动作、价格、利润、
    # 历史及加速 Q 目标都不能偷看它。
    alternate_receipt = run_one_market_period(
        period_number=0,
        previous_price_p=previous_price,
        previous_value_v=previous_value,
        current_value_v=current_value,
        next_value_v=alternate_next_value,
        noise_order_u=fixed_noise_order,
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        traders=alternate_traders,
        shared_value_visit_counts=alternate_shared_counts,
        market_maker=alternate_market_maker,
    )

    expected_epsilon = exp(
        -parameters.exploration_decay * past_visit_count
    )
    assert isclose(receipt.epsilon, expected_epsilon, rel_tol=0.0, abs_tol=1e-15)
    assert all(
        result.action_decision == ActionDecision(14, "exploitation")
        for result in receipt.trader_results
    )
    expected_orders = calculate_orders_for_value(
        current_value,
        parameters.value_mean,
        list(action_multipliers),
    )
    for result in receipt.trader_results:
        assert isclose(result.raw_order_x, expected_orders[14], abs_tol=1e-12)
    expected_y = (
        receipt.trader_results[0].raw_order_x
        + receipt.trader_results[1].raw_order_x
        + fixed_noise_order
    )
    assert isclose(receipt.total_order_flow_y, expected_y, abs_tol=1e-12)
    expected_price = (
        receipt.market_maker_estimates_from_prior_history.gamma_0_hat
        + receipt.adaptive_price_quote.price_impact_lambda_hat * expected_y
    )
    assert isclose(
        receipt.adaptive_price_quote.continuous_price_p_hat,
        expected_price,
        abs_tol=1e-12,
    )
    expected_z = calculate_insensitive_order(
        expected_price,
        parameters.value_mean,
        parameters.investor_slope,
    )
    assert isclose(receipt.information_insensitive_order_z, expected_z, abs_tol=1e-12)

    # Independently recalculate both Q targets from the untouched table copies.
    # / 从未修改的旧 Q 表独立重算两位交易者的 Q 目标。
    for trader_index, result in enumerate(receipt.trader_results):
        before = q_tables_before[trader_index]
        possible_rows = before[
            list(receipt.possible_next_state_ids_for_q_update),
            :,
        ]
        expected_continuation = float(np.mean(np.max(possible_rows, axis=1)))
        old_q = float(
            before[
                receipt.current_state_id,
                result.action_decision.action_index,
            ]
        )
        expected_new_q = (
            (1.0 - parameters.learning_rate) * old_q
            + parameters.learning_rate
            * (
                result.q_update.realized_profit
                + parameters.discount_factor * expected_continuation
            )
        )
        assert isclose(result.q_update.old_q_value, old_q, abs_tol=1e-12)
        assert isclose(result.q_update.new_q_value, expected_new_q, abs_tol=1e-12)
        changed_cells = np.argwhere(traders[trader_index].q_table != before)
        assert changed_cells.tolist() == [
            [receipt.current_state_id, result.action_decision.action_index]
        ]

    assert receipt.current_value_visit_count_before == past_visit_count
    assert receipt.current_value_visit_count_after == past_visit_count + 1
    assert sum(shared_counts) == past_visit_count + 1
    assert receipt.history_size_before == parameters.market_maker_window
    assert receipt.history_size_after == parameters.market_maker_window
    assert receipt.evicted_history_row == prehistory.rows[0]
    assert market_maker.snapshot()[-1] == receipt.appended_history_row
    assert len(receipt.possible_next_state_ids_for_q_update) == len(value_grid)
    assert len(set(receipt.possible_next_state_ids_for_q_update)) == len(value_grid)

    # Fast rolling OLS remains correct after the completed row replaces one
    # prehistory row. / 完整本期记录替换一条前历史后，快速滚动 OLS 仍然正确。
    post_period_fast_estimates = market_maker.estimates()
    post_period_readable_estimates = fit_market_maker_regressions(
        market_maker.snapshot()
    )
    for fast_coefficient, readable_coefficient in zip(
        (
            post_period_fast_estimates.xi_0_hat,
            post_period_fast_estimates.xi_1_hat,
            post_period_fast_estimates.gamma_0_hat,
            post_period_fast_estimates.gamma_1_hat,
        ),
        (
            post_period_readable_estimates.xi_0_hat,
            post_period_readable_estimates.xi_1_hat,
            post_period_readable_estimates.gamma_0_hat,
            post_period_readable_estimates.gamma_1_hat,
        ),
        strict=True,
    ):
        assert isclose(
            fast_coefficient,
            readable_coefficient,
            rel_tol=1e-10,
            abs_tol=1e-9,
        )

    # Only the realized next-state receipt may differ when v_(t+1) changes.
    # Expected-v continuation makes even the Q updates identical. / 只改变实际
    # v_(t+1) 时，只有实际下一状态记录可以不同；下一价值期望加速使 Q 更新也相同。
    assert receipt.current_state_id == alternate_receipt.current_state_id
    assert receipt.epsilon == alternate_receipt.epsilon
    assert receipt.trader_results == alternate_receipt.trader_results
    assert receipt.total_order_flow_y == alternate_receipt.total_order_flow_y
    assert receipt.adaptive_price_quote == alternate_receipt.adaptive_price_quote
    assert receipt.information_insensitive_order_z == alternate_receipt.information_insensitive_order_z
    assert receipt.appended_history_row == alternate_receipt.appended_history_row
    assert receipt.possible_next_state_ids_for_q_update == alternate_receipt.possible_next_state_ids_for_q_update
    assert receipt.realized_next_state_id != alternate_receipt.realized_next_state_id

    try:
        setattr(receipt, "total_order_flow_y", 999.0)
    except FrozenInstanceError:
        receipt_is_frozen = True
    else:
        receipt_is_frozen = False
    assert receipt_is_frozen

    # Rejected input must leave every mutable object exactly as it was. This
    # is the software-engineering meaning of "validate first, mutate second."
    # / 无效输入不能留下任何半成品修改；这就是软件工程里的“先验证、后修改”。
    def assert_rejected_call_is_atomic(
        label: str,
        *,
        supplied_parameters: PaperParameters = parameters,
        supplied_values: Sequence[float] = value_grid,
        supplied_multipliers: Sequence[float] = action_multipliers,
        supplied_next_value: float = realized_next_value,
        poison_second_q_table: bool = False,
    ) -> None:
        rejected_traders = build_two_informed_traders(
            initial_q_table,
            random_seeds=(1, 2),
        )
        if poison_second_q_table:
            rejected_traders[1].q_table[0, 0] = float("nan")
        rejected_counts = initialize_value_visit_counts(len(value_grid))
        rejected_counts[current_value_index] = past_visit_count
        rejected_maker = preload_rolling_market_maker(prehistory)

        rng_before = tuple(
            (
                trader.mode_random_generator.getstate(),
                trader.action_random_generator.getstate(),
            )
            for trader in rejected_traders
        )
        counts_before = rejected_counts.copy()
        q_before = tuple(
            trader.q_table.copy()
            for trader in rejected_traders
        )
        history_before = rejected_maker.snapshot()

        try:
            run_one_market_period(
                period_number=0,
                previous_price_p=previous_price,
                previous_value_v=previous_value,
                current_value_v=current_value,
                next_value_v=supplied_next_value,
                noise_order_u=fixed_noise_order,
                parameters=supplied_parameters,
                value_grid=supplied_values,
                price_grid=price_grid,
                action_multipliers=supplied_multipliers,
                traders=rejected_traders,
                shared_value_visit_counts=rejected_counts,
                market_maker=rejected_maker,
            )
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(
                f"Invalid case was accepted: {label} / 无效情形被接受：{label}"
            )

        assert tuple(
            (
                trader.mode_random_generator.getstate(),
                trader.action_random_generator.getstate(),
            )
            for trader in rejected_traders
        ) == rng_before
        assert rejected_counts == counts_before
        assert all(
            np.array_equal(trader.q_table, old_q, equal_nan=True)
            for trader, old_q in zip(rejected_traders, q_before, strict=True)
        )
        assert rejected_maker.snapshot() == history_before

    invalid_multipliers = list(action_multipliers)
    invalid_multipliers[3] = float("nan")
    assert_rejected_call_is_atomic(
        "off-grid next value / 网格外下一价值",
        supplied_next_value=value_grid[-1] + 1.0,
    )
    assert_rejected_call_is_atomic(
        "non-finite action multiplier / 非有限动作乘数",
        supplied_multipliers=invalid_multipliers,
    )
    assert_rejected_call_is_atomic(
        "invalid second Q-table / 第二张 Q 表无效",
        poison_second_q_table=True,
    )
    assert_rejected_call_is_atomic(
        "T_m mismatch / T_m 不一致",
        supplied_parameters=PaperParameters(market_maker_window=9_999),
    )
    assert_rejected_call_is_atomic(
        "unsorted value grid / 价值网格未排序",
        supplied_values=tuple(reversed(value_grid)),
    )

    print("Step 25: One complete market period / 步骤 25：一个完整市场时期")
    print(f"1. Current state ID / 当前状态编号: {receipt.current_state_id}")
    print(
        "2. Epsilon from past visits / 根据过去访问计算 epsilon: "
        f"{receipt.epsilon:.9f}"
    )
    for trader_number, result in enumerate(receipt.trader_results, start=1):
        print(
            f"3.{trader_number} Trader {trader_number} / 交易者 {trader_number}: "
            f"action index {result.action_decision.action_index}, "
            f"{result.action_decision.mode}, x={result.raw_order_x:.9f}"
        )
    print(f"4. Fixed noise u_t / 固定噪声 u_t: {receipt.noise_order_u:.9f}")
    print(f"5. Total flow y_t / 总订单流 y_t: {receipt.total_order_flow_y:.9f}")
    print(
        "6. OLS lambda and continuous price / OLS lambda 与连续价格: "
        f"lambda={receipt.adaptive_price_quote.price_impact_lambda_hat:.12f}, "
        f"p={receipt.adaptive_price_quote.continuous_price_p_hat:.12f}"
    )
    print(
        "7. Insensitive order z_t / 信息不敏感订单 z_t: "
        f"{receipt.information_insensitive_order_z:.9f}"
    )
    for trader_number, result in enumerate(receipt.trader_results, start=1):
        print(
            f"8.{trader_number} Profit and Q / 利润与 Q {trader_number}: "
            f"profit={result.q_update.realized_profit:.9f}, "
            f"Q {result.q_update.old_q_value:.9f} -> "
            f"{result.q_update.new_q_value:.9f}"
        )
    print(
        "9. History / 历史: one old row evicted and current row appended; "
        f"size remains {receipt.history_size_after:,}"
    )
    print(
        "10. Realized next state / 实际下一状态: "
        f"{receipt.realized_next_state_indexes} -> ID {receipt.realized_next_state_id}"
    )
    print(
        "Changing only realized v_(t+1) changed no current outcome or accelerated "
        "Q update. / 只改变实际 v_(t+1) 不会改变本期结果或加速 Q 更新。"
    )
    print(
        "Rejected-input atomicity checks / 无效输入原子性检查: "
        "5 passed; RNG, counters, Q, and history stayed unchanged / "
        "5 项通过；随机数、计数器、Q 表和历史均未改变"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
