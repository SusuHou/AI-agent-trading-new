"""Step 24C: create the market maker's missing initial history D_0.

步骤 24C：建立论文没有说明来源的做市商初始历史 D_0。

Run / 运行:
    py -3 -X utf8 steps/step_24c_initial_market_maker_history.py

Paper fact / 原文事实:
    The paper requires a complete rolling dataset

        D_t = {(v_(t-tau), p_(t-tau), z_(t-tau), y_(t-tau))}_{tau=1}^{T_m}

    and uses T_m=10,000. At t=0 this requires 10,000 observations from
    negative-index periods, but neither the paper nor its online appendix says
    how those observations are generated. / 论文要求一个完整滚动数据集，且
    T_m=10,000；因此 t=0 时已经需要一万条“负时间”记录，但正文和在线附录都
    没有说明这些记录如何产生。

Replication assumption A3 / 复现假设 A3:
    Our baseline D_0 is a deterministic, balanced, Nash-consistent synthetic
    prehistory. Every fundamental-value grid point is crossed with every point
    of a symmetric Gaussian-quantile noise design. Informed orders and prices
    follow the solved Nash benchmark, and z follows the paper's insensitive-
    investor demand rule. / 我们的基准 D_0 是确定性、平衡且与 Nash 一致的合成
    前历史。每个基本价值网格点都与对称高斯分位数噪声设计中的每一点配对；知情
    订单和价格采用求解后的 Nash 基准，z 采用论文的信息不敏感投资者需求规则。

Why balance rather than make one random draw / 为什么采用平衡设计而非一次随机抽样:
    This makes the initial OLS auditable, gives value and noise their intended
    moments, and avoids making an already-unspecified initialization depend on
    one lucky seed. It is our engineering/research choice, not a recovered
    paper rule. The class also supports a cartel-consistent prehistory so later
    sensitivity analysis can test initialization dependence. An expanding-
    window start remains a separate pending sensitivity check. / 平衡设计便于核对，
    精确匹配价值和噪声的目标矩，并避免让原本就未说明的初始化再依赖一个幸运种子。
    这是我们的研究与工程选择，不是从论文恢复出的规则。代码也支持 cartel 初始化，
    以便未来做敏感性分析；扩展窗口初始化仍是另一项待完成的敏感性检验。

Important boundary / 重要边界:
    The market maker receives only synthetic rows. It is not handed the true
    coefficients; it must recover them through the same Step-24B rolling OLS
    used in the live simulation. / 做市商只收到合成记录，不会被直接告知真实系数；
    它必须通过正式模拟使用的同一个 Step-24B 滚动 OLS 自己估计系数。
"""

from collections import Counter
from dataclasses import dataclass, replace
from math import fsum, isclose, isfinite, sqrt
from numbers import Real
from pathlib import Path
from statistics import NormalDist
import sys


# Allow direct execution from the steps folder while reusing src. / 允许直接从
# steps 文件夹运行，同时复用 src 中经过验证的代码。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.parameters import PaperParameters
from src.step01_value_grid import build_value_grid, discrete_value_std
from step_04_information_insensitive_investors import (
    calculate_insensitive_order,
)
from step_08_nash_benchmark import (
    calculate_nash_order,
    calculate_nash_price,
)
from step_09_cartel_benchmark import (
    calculate_cartel_order,
    calculate_cartel_price,
)
from step_10_fixed_point_solver import solve_benchmark_fixed_point
from step_22_market_maker_rolling_history import MarketObservation
from step_23_market_maker_ols import (
    MarketMakerOLSEstimates,
    fit_market_maker_regressions,
)
from step_24_adaptive_market_maker_price import (
    calculate_adaptive_price_impact,
    calculate_adaptive_price_quote,
)
from step_24b_fast_rolling_ols import RollingMarketMakerOLS


@dataclass(frozen=True)
class SyntheticMarketMakerPrehistory:
    """An immutable receipt describing one complete synthetic D_0.

    描述一份完整合成 D_0 的不可修改记录。
    """

    benchmark_name: str
    rows: tuple[MarketObservation, ...]
    value_grid: tuple[float, ...]
    balanced_noise_levels: tuple[float, ...]
    discrete_fundamental_std: float
    benchmark_price_impact: float
    benchmark_trading_intensity: float
    benchmark_gamma: float

    @property
    def window_size(self) -> int:
        """Return the number of rows supplied to the maker. / 返回提供给做市商的行数。"""

        return len(self.rows)


def _positive_finite_real(number: float, label: str) -> float:
    """Return one positive finite float with a clear error. / 检查并返回正的有限浮点数。"""

    if isinstance(number, bool) or not isinstance(number, Real):
        raise TypeError(f"{label} must be a real number. / {label} 必须是实数。")
    converted = float(number)
    if not isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{label} must be positive and finite. / {label} 必须是正的有限数。")
    return converted


def build_balanced_gaussian_noise_levels(
    noise_std: float,
    number_of_levels: int,
) -> tuple[float, ...]:
    """Return symmetric Gaussian quantiles with exact target population std.

    返回对称高斯分位数，并把总体标准差精确缩放到目标值。

    Levels appear as adjacent (-u,+u) pairs. This ensures every completed pair
    has mean zero. A deterministic greedy ordering keeps each prefix's average
    squared noise close to sigma_u^2, preventing the rolling window from first
    removing systematically high- or low-variance blocks. / 噪声以相邻 (-u,+u)
    对出现，因此每个完整对的均值都是零；确定性的贪心排序让每个前缀的噪声平方
    均值尽量接近 sigma_u^2，避免滚动窗口先系统性删除高方差或低方差区块。
    """

    target_std = _positive_finite_real(noise_std, "noise_std / 噪声标准差")
    if (
        isinstance(number_of_levels, bool)
        or not isinstance(number_of_levels, int)
        or number_of_levels < 2
        or number_of_levels % 2 != 0
    ):
        raise ValueError(
            "number_of_levels must be an even integer of at least two. / "
            "number_of_levels 必须是至少为 2 的偶数。"
        )

    normal = NormalDist()
    half_size = number_of_levels // 2
    positive_magnitudes = []
    for zero_based_index in range(half_size, number_of_levels):
        probability = (
            2 * (zero_based_index + 1) - 1
        ) / (2 * number_of_levels)
        positive_magnitudes.append(normal.inv_cdf(probability))

    raw_population_std = sqrt(
        fsum(magnitude**2 for magnitude in positive_magnitudes) / half_size
    )
    scale = target_std / raw_population_std
    scaled_magnitudes = [
        magnitude * scale for magnitude in positive_magnitudes
    ]

    # At step k, select the remaining magnitude whose squared value brings the
    # cumulative sum closest to k*sigma_u^2. This balances the second moment of
    # every complete prefix, not merely the full set. / 第 k 步选择一个剩余幅度，
    # 使累计平方和最接近 k*sigma_u^2；因此每个完整前缀的二阶矩也得到平衡，而不只是
    # 最终全集平衡。
    ordered_magnitudes: list[float] = []
    remaining_magnitudes = list(scaled_magnitudes)
    cumulative_squared_magnitude = 0.0
    target_variance = target_std**2
    while remaining_magnitudes:
        next_prefix_size = len(ordered_magnitudes) + 1
        target_cumulative_square = next_prefix_size * target_variance
        selected_index = min(
            range(len(remaining_magnitudes)),
            key=lambda index: (
                abs(
                    cumulative_squared_magnitude
                    + remaining_magnitudes[index] ** 2
                    - target_cumulative_square
                ),
                index,
            ),
        )
        selected_magnitude = remaining_magnitudes.pop(selected_index)
        ordered_magnitudes.append(selected_magnitude)
        cumulative_squared_magnitude += selected_magnitude**2

    balanced_levels: list[float] = []
    for magnitude in ordered_magnitudes:
        balanced_levels.extend((-magnitude, magnitude))

    levels = tuple(balanced_levels)
    level_mean = fsum(levels) / number_of_levels
    level_std = sqrt(
        fsum((level - level_mean) ** 2 for level in levels)
        / number_of_levels
    )
    assert abs(level_mean) < 1e-15 * max(1.0, target_std)
    assert isclose(level_std, target_std, rel_tol=1e-14, abs_tol=1e-14)
    return levels


def _maximum_pair_prefix_variance_error(
    balanced_noise_levels: tuple[float, ...],
    target_noise_std: float,
) -> float:
    """Return the worst relative variance error across complete pair prefixes.

    返回所有完整噪声对前缀中最大的相对方差误差。

    This diagnostic concerns row order during early eviction, not the full-set
    variance (which is exact by construction). / 这个诊断检查早期淘汰时的记录顺序，
    而不是已经由构造保证精确的全集方差。
    """

    target_variance = target_noise_std**2
    cumulative_square = 0.0
    maximum_relative_error = 0.0
    completed_pairs = 0
    for pair_start in range(0, len(balanced_noise_levels), 2):
        negative_noise, positive_noise = balanced_noise_levels[
            pair_start:pair_start + 2
        ]
        assert isclose(
            negative_noise,
            -positive_noise,
            rel_tol=0.0,
            abs_tol=1e-15 * max(1.0, target_noise_std),
        )
        cumulative_square += negative_noise**2 + positive_noise**2
        completed_pairs += 1
        prefix_variance = cumulative_square / (2 * completed_pairs)
        relative_error = abs(prefix_variance / target_variance - 1.0)
        maximum_relative_error = max(
            maximum_relative_error,
            relative_error,
        )
    return maximum_relative_error


def build_synthetic_market_maker_prehistory(
    parameters: PaperParameters,
    *,
    benchmark_name: str = "nash",
) -> SyntheticMarketMakerPrehistory:
    """Build exactly T_m benchmark-consistent frozen rows.

    精确建立 T_m 条与指定基准一致的冻结记录。
    """

    if not isinstance(parameters, PaperParameters):
        raise TypeError("parameters must be PaperParameters. / parameters 必须是 PaperParameters。")
    if benchmark_name not in {"nash", "cartel"}:
        raise ValueError("benchmark_name must be 'nash' or 'cartel'. / benchmark_name 必须是 nash 或 cartel。")
    if parameters.market_maker_window % parameters.num_value_points != 0:
        raise ValueError(
            "A3 balanced design requires T_m to be divisible by n_v. / "
            "A3 平衡设计要求 T_m 能被 n_v 整除。"
        )

    noise_levels_per_value = (
        parameters.market_maker_window // parameters.num_value_points
    )
    if noise_levels_per_value % 2 != 0:
        raise ValueError(
            "A3 balanced design requires an even number of noise levels per value. / "
            "A3 平衡设计要求每个价值点对应偶数个噪声点。"
        )

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
    fixed_point = solve_benchmark_fixed_point(
        benchmark_name,
        parameters.num_speculators,
        parameters.noise_std,
        sigma_v_hat,
        parameters.investor_slope,
        parameters.pricing_error_weight,
    )
    price_impact = fixed_point["price_impact"]
    trading_intensity = fixed_point["intensity"]
    gamma = fixed_point["gamma"]
    noise_levels = build_balanced_gaussian_noise_levels(
        parameters.noise_std,
        noise_levels_per_value,
    )

    if benchmark_name == "nash":
        order_function = calculate_nash_order
        price_function = calculate_nash_price
    else:
        order_function = calculate_cartel_order
        price_function = calculate_cartel_price

    rows: list[MarketObservation] = []
    # Each 2*n_v block contains every value twice and one (-u,+u) noise pair.
    # / 每个 2*n_v 行的小块包含每个价值两次，以及一对 (-u,+u) 噪声。
    for pair_start in range(0, len(noise_levels), 2):
        negative_noise, positive_noise = noise_levels[pair_start:pair_start + 2]
        for fundamental_value in value_grid:
            one_trader_order = order_function(
                fundamental_value,
                parameters.value_mean,
                trading_intensity,
            )
            aggregate_informed_order = (
                parameters.num_speculators * one_trader_order
            )
            for noise_order in (negative_noise, positive_noise):
                order_flow_y = aggregate_informed_order + noise_order
                market_price_p = price_function(
                    order_flow_y,
                    parameters.value_mean,
                    price_impact,
                )
                insensitive_order_z = calculate_insensitive_order(
                    market_price_p,
                    parameters.value_mean,
                    parameters.investor_slope,
                )
                rows.append(
                    MarketObservation(
                        fundamental_value_v=fundamental_value,
                        market_price_p=market_price_p,
                        insensitive_order_z=insensitive_order_z,
                        informed_and_noise_order_y=order_flow_y,
                    )
                )

    frozen_rows = tuple(rows)
    if len(frozen_rows) != parameters.market_maker_window:
        raise RuntimeError("A3 did not create exactly T_m rows. / A3 未能精确建立 T_m 行。")
    return SyntheticMarketMakerPrehistory(
        benchmark_name=benchmark_name,
        rows=frozen_rows,
        value_grid=value_grid,
        balanced_noise_levels=noise_levels,
        discrete_fundamental_std=sigma_v_hat,
        benchmark_price_impact=price_impact,
        benchmark_trading_intensity=trading_intensity,
        benchmark_gamma=gamma,
    )


def preload_rolling_market_maker(
    prehistory: SyntheticMarketMakerPrehistory,
) -> RollingMarketMakerOLS:
    """Give a new rolling OLS maker only the rows, then let it estimate.

    只把记录交给新的滚动 OLS 做市商，然后让它自己估计。
    """

    if not isinstance(prehistory, SyntheticMarketMakerPrehistory):
        raise TypeError("prehistory has the wrong type. / prehistory 类型错误。")
    market_maker = RollingMarketMakerOLS(
        window_size=prehistory.window_size,
        resynchronize_every=prehistory.window_size,
    )
    for row in prehistory.rows:
        market_maker.append_completed_observation(row)
    if not market_maker.is_full:
        raise RuntimeError("Initial market-maker history is not full. / 做市商初始历史未装满。")
    return market_maker


def validate_prehistory(
    prehistory: SyntheticMarketMakerPrehistory,
    parameters: PaperParameters,
) -> MarketMakerOLSEstimates:
    """Validate balance, structural equations, and recovered coefficients.

    验证平衡性、结构方程以及做市商自己恢复出的系数。
    """

    value_counts = Counter(
        row.fundamental_value_v for row in prehistory.rows
    )
    expected_count_per_value = (
        parameters.market_maker_window // parameters.num_value_points
    )
    assert set(value_counts) == set(prehistory.value_grid)
    assert all(
        count == expected_count_per_value for count in value_counts.values()
    )

    recovered_noise_orders: list[float] = []
    maximum_price_error = 0.0
    maximum_demand_error = 0.0
    for row in prehistory.rows:
        aggregate_informed_order = (
            parameters.num_speculators
            * prehistory.benchmark_trading_intensity
            * (row.fundamental_value_v - parameters.value_mean)
        )
        recovered_noise_orders.append(
            row.informed_and_noise_order_y - aggregate_informed_order
        )
        expected_price = (
            parameters.value_mean
            + prehistory.benchmark_price_impact
            * row.informed_and_noise_order_y
        )
        expected_z = calculate_insensitive_order(
            row.market_price_p,
            parameters.value_mean,
            parameters.investor_slope,
        )
        maximum_price_error = max(
            maximum_price_error,
            abs(row.market_price_p - expected_price),
        )
        maximum_demand_error = max(
            maximum_demand_error,
            abs(row.insensitive_order_z - expected_z),
        )
    assert maximum_price_error < 1e-12
    assert maximum_demand_error < 1e-12

    # Reconstruct the declared balanced row order and verify every recovered
    # noise realization, not only its aggregate moments. / 重建声明的平衡行顺序，
    # 逐行验证每个恢复出的噪声，而不只是验证整体矩。
    expected_value_noise_pairs: list[tuple[float, float]] = []
    for pair_start in range(0, len(prehistory.balanced_noise_levels), 2):
        noise_pair = prehistory.balanced_noise_levels[
            pair_start:pair_start + 2
        ]
        for fundamental_value in prehistory.value_grid:
            for noise_order in noise_pair:
                expected_value_noise_pairs.append(
                    (fundamental_value, noise_order)
                )
    assert len(expected_value_noise_pairs) == prehistory.window_size
    for row, recovered_noise, expected_pair in zip(
        prehistory.rows,
        recovered_noise_orders,
        expected_value_noise_pairs,
        strict=True,
    ):
        expected_value, expected_noise = expected_pair
        assert row.fundamental_value_v == expected_value
        assert isclose(
            recovered_noise,
            expected_noise,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

    sample_size = len(recovered_noise_orders)
    noise_mean = fsum(recovered_noise_orders) / sample_size
    noise_std = sqrt(
        fsum((noise - noise_mean) ** 2 for noise in recovered_noise_orders)
        / sample_size
    )
    value_noise_covariance = fsum(
        (row.fundamental_value_v - parameters.value_mean)
        * (noise - noise_mean)
        for row, noise in zip(
            prehistory.rows,
            recovered_noise_orders,
            strict=True,
        )
    ) / sample_size
    assert abs(noise_mean) < 1e-11 * max(1.0, parameters.noise_std)
    assert isclose(
        noise_std,
        parameters.noise_std,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert abs(value_noise_covariance) < (
        1e-10 * parameters.value_std * parameters.noise_std
    )
    # Each complete eviction block must remain close to the target second
    # moment. The old tail/center alternation failed this test by a wide margin.
    # / 每个完整淘汰区块的二阶矩都必须接近目标；旧的尾部/中心交错顺序会明显失败。
    assert _maximum_pair_prefix_variance_error(
        prehistory.balanced_noise_levels,
        parameters.noise_std,
    ) < 0.02

    readable_estimates = fit_market_maker_regressions(prehistory.rows)
    market_maker = preload_rolling_market_maker(prehistory)
    rolling_estimates = market_maker.estimates()
    assert rolling_estimates == readable_estimates

    expected_coefficients = (
        parameters.investor_slope * parameters.value_mean,
        parameters.investor_slope,
        parameters.value_mean,
        prehistory.benchmark_gamma,
    )
    recovered_coefficients = (
        rolling_estimates.xi_0_hat,
        rolling_estimates.xi_1_hat,
        rolling_estimates.gamma_0_hat,
        rolling_estimates.gamma_1_hat,
    )
    for recovered, expected in zip(
        recovered_coefficients,
        expected_coefficients,
        strict=True,
    ):
        assert isclose(recovered, expected, rel_tol=1e-10, abs_tol=1e-10)

    recovered_price_impact = calculate_adaptive_price_impact(
        rolling_estimates,
        parameters.pricing_error_weight,
    )
    assert isclose(
        recovered_price_impact,
        prehistory.benchmark_price_impact,
        rel_tol=1e-10,
        abs_tol=1e-12,
    )

    # A fresh quote from the estimated coefficients must reproduce the same
    # continuous benchmark price. / 使用估计系数生成的新报价必须恢复同一个连续基准价格。
    probe_order_flow_y = 12.345
    recovered_quote = calculate_adaptive_price_quote(
        probe_order_flow_y,
        rolling_estimates,
        parameters.pricing_error_weight,
    )
    expected_probe_price = (
        parameters.value_mean
        + prehistory.benchmark_price_impact * probe_order_flow_y
    )
    assert isclose(
        recovered_quote.continuous_price_p_hat,
        expected_probe_price,
        rel_tol=1e-10,
        abs_tol=1e-12,
    )
    return rolling_estimates


def main() -> None:
    """Build and validate Nash baseline plus a cartel sensitivity variant.

    建立并验证 Nash 基准，以及一个 cartel 敏感性变体。
    """

    parameters = PaperParameters()
    nash_prehistory = build_synthetic_market_maker_prehistory(
        parameters,
        benchmark_name="nash",
    )
    nash_estimates = validate_prehistory(nash_prehistory, parameters)

    # After the first live period completes, exactly the recorded oldest
    # synthetic row must leave, and fast OLS must still match readable OLS.
    # / 第一条真实时期记录完成后，必须恰好淘汰已记录的最旧合成行；高效 OLS 仍须
    # 匹配可读 OLS。
    live_market_maker = preload_rolling_market_maker(nash_prehistory)
    oldest_synthetic_row = live_market_maker.snapshot()[0]
    live_value = nash_prehistory.value_grid[
        len(nash_prehistory.value_grid) // 2
    ]
    live_order_flow_y = (
        parameters.num_speculators
        * nash_prehistory.benchmark_trading_intensity
        * (live_value - parameters.value_mean)
    )
    live_price = (
        parameters.value_mean
        + nash_prehistory.benchmark_price_impact * live_order_flow_y
    )
    first_live_completed_row = MarketObservation(
        fundamental_value_v=live_value,
        market_price_p=live_price,
        insensitive_order_z=calculate_insensitive_order(
            live_price,
            parameters.value_mean,
            parameters.investor_slope,
        ),
        informed_and_noise_order_y=live_order_flow_y,
    )
    evicted_row = live_market_maker.append_completed_observation(
        first_live_completed_row
    )
    assert evicted_row == oldest_synthetic_row
    assert first_live_completed_row == live_market_maker.snapshot()[-1]
    post_eviction_fast = live_market_maker.estimates()
    post_eviction_readable = fit_market_maker_regressions(
        live_market_maker.snapshot()
    )
    for fast_coefficient, readable_coefficient in zip(
        (
            post_eviction_fast.xi_0_hat,
            post_eviction_fast.xi_1_hat,
            post_eviction_fast.gamma_0_hat,
            post_eviction_fast.gamma_1_hat,
        ),
        (
            post_eviction_readable.xi_0_hat,
            post_eviction_readable.xi_1_hat,
            post_eviction_readable.gamma_0_hat,
            post_eviction_readable.gamma_1_hat,
        ),
        strict=True,
    ):
        assert isclose(
            fast_coefficient,
            readable_coefficient,
            rel_tol=1e-10,
            abs_tol=1e-9,
        )

    # This confirms the alternative initializer is executable; it does not yet
    # claim that final experimental outcomes are insensitive. / 这里只确认替代
    # 初始化器可以正确运行，尚不能声称最终实验结果对初始化不敏感。
    cartel_prehistory = build_synthetic_market_maker_prehistory(
        parameters,
        benchmark_name="cartel",
    )
    cartel_estimates = validate_prehistory(cartel_prehistory, parameters)

    # The paper studies both low and high noise. The same A3 construction must
    # recover the high-noise Nash coefficients too. / 论文同时研究低噪声与高噪声；
    # 同一 A3 构造也必须恢复高噪声 Nash 系数。
    high_noise_parameters = replace(parameters, noise_std=100.0)
    high_noise_prehistory = build_synthetic_market_maker_prehistory(
        high_noise_parameters,
        benchmark_name="nash",
    )
    high_noise_estimates = validate_prehistory(
        high_noise_prehistory,
        high_noise_parameters,
    )

    # Roll through ten complete 2*n_v blocks at high-noise scale. Re-appending
    # the evicted prefix rotates the same multiset, so coefficients should stay
    # unchanged while add/remove arithmetic is stressed. / 在高噪声尺度下滚动十个
    # 完整的 2*n_v 区块。把被淘汰前缀依次加回会旋转同一个多重集合，因此既能压力
    # 测试加入/移除运算，又应保持系数不变。
    high_noise_market_maker = preload_rolling_market_maker(
        high_noise_prehistory
    )
    transition_row_count = 10 * 2 * high_noise_parameters.num_value_points
    original_high_noise_rows = high_noise_market_maker.snapshot()
    for expected_eviction, incoming_row in zip(
        original_high_noise_rows[:transition_row_count],
        original_high_noise_rows[:transition_row_count],
        strict=True,
    ):
        actual_eviction = (
            high_noise_market_maker.append_completed_observation(incoming_row)
        )
        assert actual_eviction == expected_eviction
    assert high_noise_market_maker.snapshot() == (
        original_high_noise_rows[transition_row_count:]
        + original_high_noise_rows[:transition_row_count]
    )
    high_transition_fast = high_noise_market_maker.estimates()
    high_transition_readable = fit_market_maker_regressions(
        high_noise_market_maker.snapshot()
    )
    for fast_coefficient, readable_coefficient in zip(
        (
            high_transition_fast.xi_0_hat,
            high_transition_fast.xi_1_hat,
            high_transition_fast.gamma_0_hat,
            high_transition_fast.gamma_1_hat,
        ),
        (
            high_transition_readable.xi_0_hat,
            high_transition_readable.xi_1_hat,
            high_transition_readable.gamma_0_hat,
            high_transition_readable.gamma_1_hat,
        ),
        strict=True,
    ):
        assert isclose(
            fast_coefficient,
            readable_coefficient,
            rel_tol=1e-9,
            abs_tol=1e-8,
        )

    assert nash_prehistory.rows != cartel_prehistory.rows
    assert not isclose(
        nash_prehistory.benchmark_trading_intensity,
        cartel_prehistory.benchmark_trading_intensity,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    # Invalid balance designs must fail before a partial history is produced.
    # 无效平衡设计必须在产生部分历史之前报错。
    for bad_level_count in (1, 3, 2.5, True):
        try:
            build_balanced_gaussian_noise_levels(
                parameters.noise_std,
                bad_level_count,  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            bad_level_count_was_rejected = True
        else:
            bad_level_count_was_rejected = False
        assert bad_level_count_was_rejected

    try:
        build_synthetic_market_maker_prehistory(
            parameters,
            benchmark_name="unknown",
        )
    except ValueError:
        unknown_benchmark_was_rejected = True
    else:
        unknown_benchmark_was_rejected = False
    assert unknown_benchmark_was_rejected

    incompatible_parameter_sets = (
        replace(parameters, market_maker_window=9_999),
        replace(
            parameters,
            num_value_points=2,
            market_maker_window=30,
        ),
    )
    for incompatible_parameters in incompatible_parameter_sets:
        try:
            build_synthetic_market_maker_prehistory(
                incompatible_parameters,
                benchmark_name="nash",
            )
        except ValueError:
            incompatible_design_was_rejected = True
        else:
            incompatible_design_was_rejected = False
        assert incompatible_design_was_rejected

    print("Step 24C: Initial market-maker history / 步骤 24C：做市商初始历史")
    print("Paper status / 原文状态: D_0 construction is unspecified / D_0 构造未说明")
    print("Replication assumption A3 / 复现假设 A3: balanced Nash-consistent prehistory")
    print(f"Rows loaded / 已载入行数: {nash_prehistory.window_size:,}")
    print(
        "Rows per fundamental value / 每个基本价值的行数: "
        f"{parameters.market_maker_window // parameters.num_value_points:,}"
    )
    print(f"Target noise std / 目标噪声标准差: {parameters.noise_std:.6f}")
    print(
        "Worst complete-prefix variance error / 完整前缀最大方差误差: "
        f"{_maximum_pair_prefix_variance_error(nash_prehistory.balanced_noise_levels, parameters.noise_std):.3%}"
    )
    print(f"Nash lambda / Nash 价格冲击: {nash_prehistory.benchmark_price_impact:.12f}")
    print(f"Nash chi / Nash 交易强度: {nash_prehistory.benchmark_trading_intensity:.9f}")
    print(f"Recovered xi_0 / 恢复的 xi_0: {nash_estimates.xi_0_hat:.9f}")
    print(f"Recovered xi_1 / 恢复的 xi_1: {nash_estimates.xi_1_hat:.9f}")
    print(f"Recovered gamma_0 / 恢复的 gamma_0: {nash_estimates.gamma_0_hat:.9f}")
    print(f"Recovered gamma_1 / 恢复的 gamma_1: {nash_estimates.gamma_1_hat:.12f}")
    print(
        "Cartel sensitivity initializer also recovered its intended OLS. / "
        "Cartel 敏感性初始化器也恢复了预期 OLS。"
    )
    print(f"Cartel gamma_1 / Cartel gamma_1: {cartel_estimates.gamma_1_hat:.12f}")
    print(
        "High-noise Nash gamma_1 / 高噪声 Nash gamma_1: "
        f"{high_noise_estimates.gamma_1_hat:.12f}"
    )
    print(
        "First live row correctly evicted the oldest synthetic row. / "
        "第一条真实记录正确淘汰了最旧合成记录。"
    )
    print(
        "High-noise multi-block transition also preserved OLS parity. / "
        "高噪声多区块过渡也保持了 OLS 一致性。"
    )
    print(
        "Expanding-window sensitivity remains pending; do not call A3 paper-confirmed. / "
        "扩展窗口敏感性检验仍待完成；不得把 A3 称为论文已确认设定。"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
