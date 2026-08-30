"""Step 13: build one 31-point market-price grid for every value.

步骤 13：为每个基本价值分别建立一个 31 点市场价格网格。

Run / 运行:
    py -3 steps/step_13_price_grid.py

The market maker will later produce a continuous price.  The Q-learning state
must be finite, so the previous market price will be represented using one of
31 indexes.  The numerical meaning of an index is conditional on the previous
fundamental value: row k is P(v_k).

做市商以后会产生连续价格。Q-learning 状态必须是有限的，因此上一期市场价格使用
31 个编号之一表示；该编号的具体价格取决于上一期基本价值：第 k 行是 P(v_k)。
"""

from collections.abc import Sequence
from math import isclose, isfinite
from pathlib import Path
import sys


# Allow both direct execution and later imports. / 同时支持直接运行与后续导入复用。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from src.step01_value_grid import build_value_grid, discrete_value_std
from step_10_fixed_point_solver import solve_benchmark_fixed_point
from step_12_action_grid import calculate_orders_for_value


def calculate_unwidened_price_bounds(
    value_mean: float,
    number_of_speculators: int,
    nash_price_impact: float,
    benchmark_orders: list[float],
    noise_std: float,
    noise_multiple: float = 1.96,
) -> tuple[float, float]:
    """Return the paper's p_L and p_H before the final iota widening.

    返回最终使用 iota 扩展之前的论文价格边界 p_L 和 p_H。

    p_L = v_bar + lambda^N [I min(x^M,x^N) - 1.96 sigma_u]
    p_H = v_bar + lambda^N [I max(x^M,x^N) + 1.96 sigma_u]

    We reproduce the printed coefficient 1.96 exactly.  Statistically it gives
    the central 95-percent normal interval, although the paper's nearby prose
    calls the endpoints the 5th and 95th percentiles.

    我们严格复现公式中写出的 1.96。统计上它对应正态分布中央 95% 区间，尽管论文
    附近文字把端点称作第 5 与第 95 百分位。

    ``benchmark_orders`` contains Nash and cartel orders across the complete
    fundamental-value grid. / benchmark_orders 包含整个基本价值网格上的纳什与
    卡特尔订单。
    """

    if not isinstance(number_of_speculators, int) or number_of_speculators < 1:
        raise ValueError("I must be a positive integer. / I 必须是正整数。")
    if nash_price_impact <= 0.0:
        raise ValueError("lambda^N must be positive. / lambda^N 必须大于零。")
    if not benchmark_orders:
        raise ValueError("benchmark_orders cannot be empty. / 基准订单不能为空。")
    if noise_std <= 0.0 or noise_multiple <= 0.0:
        raise ValueError(
            "Noise standard deviation and multiplier must be positive. / "
            "噪声标准差与倍数必须大于零。"
        )

    smallest_benchmark_order = min(benchmark_orders)
    largest_benchmark_order = max(benchmark_orders)

    lower_price = value_mean + nash_price_impact * (
        number_of_speculators * smallest_benchmark_order
        - noise_multiple * noise_std
    )
    upper_price = value_mean + nash_price_impact * (
        number_of_speculators * largest_benchmark_order
        + noise_multiple * noise_std
    )

    if lower_price >= upper_price:
        raise ValueError("p_L must be below p_H. / p_L 必须小于 p_H。")

    return lower_price, upper_price


def build_price_grid(
    lower_price: float,
    upper_price: float,
    grid_widening: float,
    number_of_prices: int,
) -> list[float]:
    """Widen [p_L,p_H] on both sides and return equally spaced prices.

    在 [p_L,p_H] 两端进行扩展，并返回等间距价格。

    final lower = p_L - iota(p_H-p_L)
    final upper = p_H + iota(p_H-p_L)
    """

    if lower_price >= upper_price:
        raise ValueError("p_L must be below p_H. / p_L 必须小于 p_H。")
    if grid_widening < 0.0:
        raise ValueError("iota cannot be negative. / iota 不能为负。")
    if not isinstance(number_of_prices, int) or number_of_prices < 2:
        raise ValueError(
            "number_of_prices must be an integer of at least 2. / "
            "价格点数量必须是至少为 2 的整数。"
        )

    original_width = upper_price - lower_price
    final_lower_price = lower_price - grid_widening * original_width
    final_upper_price = upper_price + grid_widening * original_width
    spacing = (
        final_upper_price - final_lower_price
    ) / (
        number_of_prices - 1
    )

    return [
        final_lower_price + price_index * spacing
        for price_index in range(number_of_prices)
    ]


def build_price_grids_by_value(
    *,
    value_grid: Sequence[float],
    value_mean: float,
    number_of_speculators: int,
    nash_price_impact: float,
    nash_intensity: float,
    cartel_intensity: float,
    noise_std: float,
    grid_widening: float,
    number_of_prices: int,
) -> list[list[float]]:
    """Return one price-grid row P(v_k) for every value v_k.

    为每个基本价值 v_k 返回一行价格网格 P(v_k)。

    The Nash and cartel orders used to construct a row must come from the SAME
    fundamental value. Orders from different values must not be pooled into one
    global interval. / 建立某一行时使用的 Nash 与 cartel 订单必须来自同一个
    基本价值，不能把不同价值的订单混成一条全局区间。
    """

    if len(value_grid) < 1:
        raise ValueError("value_grid cannot be empty. / 价值网格不能为空。")
    if not all(isfinite(float(value)) for value in value_grid):
        raise ValueError(
            "value_grid must contain finite values. / 价值网格必须只包含有限数值。"
        )
    if not isfinite(nash_intensity) or not isfinite(cartel_intensity):
        raise ValueError(
            "Benchmark intensities must be finite. / 基准交易强度必须是有限数值。"
        )

    price_grids: list[list[float]] = []
    for fundamental_value in value_grid:
        benchmark_orders_at_this_value = calculate_orders_for_value(
            float(fundamental_value),
            value_mean,
            [cartel_intensity, nash_intensity],
        )
        lower_price, upper_price = calculate_unwidened_price_bounds(
            value_mean,
            number_of_speculators,
            nash_price_impact,
            benchmark_orders_at_this_value,
            noise_std,
        )
        price_grids.append(
            build_price_grid(
                lower_price,
                upper_price,
                grid_widening,
                number_of_prices,
            )
        )

    return price_grids


def main() -> None:
    """Build and validate low- and high-noise P(v). / 建立并验证两种环境的 P(v)。"""

    # A small independent hand check / 一个独立、可以手算的小测试。
    toy_lower, toy_upper = calculate_unwidened_price_bounds(
        value_mean=1.0,
        number_of_speculators=2,
        nash_price_impact=0.5,
        benchmark_orders=[-4.0, -2.0, 2.0, 4.0],
        noise_std=1.0,
    )
    assert isclose(toy_lower, -3.98, abs_tol=1e-12)
    assert isclose(toy_upper, 5.98, abs_tol=1e-12)
    toy_grid = build_price_grid(toy_lower, toy_upper, 0.1, 5)
    expected_toy_grid = [-4.976, -1.988, 1.0, 3.988, 6.976]
    assert all(
        isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(toy_grid, expected_toy_grid, strict=True)
    )

    parameters = PaperParameters()
    value_grid = build_value_grid(
        parameters.value_mean,
        parameters.value_std,
        parameters.num_value_points,
    )
    discrete_fundamental_std = discrete_value_std(
        value_grid,
        parameters.value_mean,
    )

    print("Step 13: Value-specific price grids / 步骤 13：分价值价格网格")
    print(
        f"Stored shape / 保存形状: {parameters.num_value_points} value rows x "
        f"{parameters.num_price_points} price points"
    )
    print(
        "Each row uses Nash/cartel orders from the same v_k. / "
        "每一行只使用同一个 v_k 下的 Nash/cartel 订单。"
    )

    widths_by_environment: list[list[float]] = []
    for environment_name, noise_std in (
        ("LOW NOISE / 低噪声", parameters.noise_std),
        ("HIGH NOISE / 高噪声", 100.0),
    ):
        nash_solution = solve_benchmark_fixed_point(
            "nash",
            parameters.num_speculators,
            noise_std,
            discrete_fundamental_std,
            parameters.investor_slope,
            parameters.pricing_error_weight,
        )
        cartel_solution = solve_benchmark_fixed_point(
            "cartel",
            parameters.num_speculators,
            noise_std,
            discrete_fundamental_std,
            parameters.investor_slope,
            parameters.pricing_error_weight,
        )
        price_grids = build_price_grids_by_value(
            value_grid=value_grid,
            value_mean=parameters.value_mean,
            number_of_speculators=parameters.num_speculators,
            nash_price_impact=nash_solution["price_impact"],
            nash_intensity=nash_solution["intensity"],
            cartel_intensity=cartel_solution["intensity"],
            noise_std=noise_std,
            grid_widening=parameters.grid_widening,
            number_of_prices=parameters.num_price_points,
        )
        assert len(price_grids) == parameters.num_value_points

        print(f"\n{environment_name}: sigma_u = {noise_std:.1f}")
        print("  v_k          P(v_k)[0]      P(v_k)[30]     one price step")
        environment_widths: list[float] = []
        for value_index, (fundamental_value, price_grid) in enumerate(
            zip(value_grid, price_grids, strict=True)
        ):
            benchmark_orders = calculate_orders_for_value(
                float(fundamental_value),
                parameters.value_mean,
                [cartel_solution["intensity"], nash_solution["intensity"]],
            )
            lower_price, upper_price = calculate_unwidened_price_bounds(
                parameters.value_mean,
                parameters.num_speculators,
                nash_solution["price_impact"],
                benchmark_orders,
                noise_std,
            )
            original_width = upper_price - lower_price
            expected_final_lower = lower_price - parameters.grid_widening * original_width
            expected_final_upper = upper_price + parameters.grid_widening * original_width
            spacing = price_grid[1] - price_grid[0]

            assert len(price_grid) == parameters.num_price_points
            assert isclose(price_grid[0], expected_final_lower, abs_tol=1e-12)
            assert isclose(price_grid[-1], expected_final_upper, abs_tol=1e-12)
            assert all(
                isclose(
                    price_grid[index + 1] - price_grid[index],
                    spacing,
                    abs_tol=1e-12,
                )
                for index in range(len(price_grid) - 1)
            )
            environment_widths.append(price_grid[-1] - price_grid[0])
            print(
                f"  V[{value_index}]={float(fundamental_value): .6f}  "
                f"{price_grid[0]: .9f}  {price_grid[-1]: .9f}  {spacing:.9f}"
            )

        # P(v_low) and P(v_high) are mirror rows around v_bar. / 围绕均值
        # 对称的两个价值，其价格行也互为镜像。
        for value_index, price_grid in enumerate(price_grids):
            mirror_grid = price_grids[-1 - value_index]
            assert all(
                isclose(
                    price + mirror_grid[-1 - price_index],
                    2.0 * parameters.value_mean,
                    abs_tol=1e-10,
                )
                for price_index, price in enumerate(price_grid)
            )
        widths_by_environment.append(environment_widths)

    assert all(
        high_width > low_width
        for low_width, high_width in zip(
            widths_by_environment[0],
            widths_by_environment[1],
            strict=True,
        )
    )
    print(
        "\nHigh noise widens every row through 1.96 sigma_u. / "
        "高噪声通过 1.96 sigma_u 扩大每一行。"
    )
    print(
        "Step 14 will choose the row using v_(t-1). / "
        "第 14 步将根据 v_(t-1) 选择价格行。"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
