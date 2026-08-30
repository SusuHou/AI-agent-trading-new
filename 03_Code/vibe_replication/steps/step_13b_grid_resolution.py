"""Step 13B: check whether one action step is visible on the price grid.

步骤 13B：检查一档 action 是否能在价格网格上被看见。

Run / 运行:
    py -3 -X utf8 steps/step_13b_grid_resolution.py

Paper notation / 论文符号:
    n_x = number of allowed action points for one value / 每个价值下的动作点数
    n_p = number of allowed price points for one value / 每个价值下的价格点数

Important / 重要:
    n_x and n_p count POINTS, not intervals.  Therefore n_x points create
    n_x-1 action intervals, and n_p points create n_p-1 price intervals.
    / n_x 与 n_p 数的是“点”，不是“间隔”。所以 n_x 个点形成 n_x-1 个
    action 间隔，n_p 个点形成 n_p-1 个价格间隔。
"""

from math import ceil, isfinite
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from src.step01_value_grid import build_value_grid, discrete_value_std
from step_10_fixed_point_solver import solve_benchmark_fixed_point
from step_12_action_grid import (
    build_action_multiplier_grid,
    calculate_orders_for_value,
)
from step_13_price_grid import (
    build_price_grid,
    calculate_unwidened_price_bounds,
)


def diagnose_one_value(
    *,
    parameters: PaperParameters,
    fundamental_value: float,
    noise_std: float,
    nash_intensity: float,
    cartel_intensity: float,
    nash_price_impact: float,
) -> dict[str, float | int]:
    """Calculate both one-step widths for one fixed fundamental value.

    对一个固定基本价值，计算 action 一档与 price 一档的宽度。

    The returned ratio is

        one-action price movement / one price-grid interval.

    A ratio near one means that moving one adjacent action changes the
    continuous price by about one price-grid interval. / ratio 接近 1 表示
    一位 trader 改变一档 action，大约会让连续价格移动一档 price grid。
    """

    action_multipliers = build_action_multiplier_grid(
        nash_intensity,
        cartel_intensity,
        parameters.grid_widening,
        parameters.num_action_points,
    )
    action_choices = calculate_orders_for_value(
        fundamental_value,
        parameters.value_mean,
        action_multipliers,
    )
    action_step = abs(action_choices[1] - action_choices[0])
    if action_step == 0.0:
        raise ValueError(
            "At v=v_bar every multiplier produces order zero, so a one-action "
            "price movement is undefined. / 当 v=v_bar 时所有乘数都产生零订单，"
            "因此无法定义一档 action 的价格变化。"
        )

    # These are the two paper benchmark orders at this SAME value. / 这是同一个
    # v 下的 cartel 与 Nash 基准订单；不能把其他 v 的订单混进来。
    benchmark_orders = calculate_orders_for_value(
        fundamental_value,
        parameters.value_mean,
        [cartel_intensity, nash_intensity],
    )
    lower_price, upper_price = calculate_unwidened_price_bounds(
        parameters.value_mean,
        parameters.num_speculators,
        nash_price_impact,
        benchmark_orders,
        noise_std,
    )
    price_grid = build_price_grid(
        lower_price,
        upper_price,
        parameters.grid_widening,
        parameters.num_price_points,
    )

    price_step = price_grid[1] - price_grid[0]
    one_action_price_movement = nash_price_impact * action_step
    final_price_width = price_grid[-1] - price_grid[0]
    ratio = one_action_price_movement / price_step

    if not all(
        isfinite(value) and value > 0.0
        for value in (
            action_step,
            price_step,
            one_action_price_movement,
            final_price_width,
            ratio,
        )
    ):
        raise ArithmeticError("A grid-resolution number is invalid. / 网格分辨率数值无效。")

    # This is a diagnostic, not a replacement for the paper's n_p=31.  It is
    # the smallest integer point count whose interval is no wider than the
    # one-action price movement. / 这是诊断值，不替代论文的 n_p=31；它表示让
    # price interval 不宽于一档 action 价格变化时所需的最小整数点数。
    suggested_price_points = (
        ceil(final_price_width / one_action_price_movement) + 1
    )

    return {
        "fundamental_value": float(fundamental_value),
        "number_of_action_points_n_x": parameters.num_action_points,
        "number_of_action_intervals": parameters.num_action_points - 1,
        "number_of_price_points_n_p": parameters.num_price_points,
        "number_of_price_intervals": parameters.num_price_points - 1,
        "one_action_order_step": action_step,
        "one_action_price_movement": one_action_price_movement,
        "one_price_grid_step": price_step,
        "movement_to_grid_step_ratio": ratio,
        "smallest_price_point_count_for_step_visibility": (
            suggested_price_points
        ),
    }


def build_resolution_table(
    parameters: PaperParameters,
    noise_std: float,
) -> list[dict[str, float | int]]:
    """Return one diagnostic row for every value-grid point. / 每个价值点返回一行诊断。"""

    value_grid = build_value_grid(
        parameters.value_mean,
        parameters.value_std,
        parameters.num_value_points,
    )
    value_grid_std = discrete_value_std(value_grid, parameters.value_mean)
    nash = solve_benchmark_fixed_point(
        "nash",
        parameters.num_speculators,
        noise_std,
        value_grid_std,
        parameters.investor_slope,
        parameters.pricing_error_weight,
    )
    cartel = solve_benchmark_fixed_point(
        "cartel",
        parameters.num_speculators,
        noise_std,
        value_grid_std,
        parameters.investor_slope,
        parameters.pricing_error_weight,
    )
    return [
        diagnose_one_value(
            parameters=parameters,
            fundamental_value=float(value),
            noise_std=noise_std,
            nash_intensity=nash["intensity"],
            cartel_intensity=cartel["intensity"],
            nash_price_impact=nash["price_impact"],
        )
        for value in value_grid
    ]


def main() -> None:
    """Print the paper counts and the resulting per-value widths. / 打印点数与分价值宽度。"""

    parameters = PaperParameters()
    print("Step 13B: grid resolution / 步骤 13B：网格分辨率")
    print(
        f"n_x = {parameters.num_action_points} action points -> "
        f"{parameters.num_action_points - 1} intervals / 动作间隔"
    )
    print(
        f"n_p = {parameters.num_price_points} price points -> "
        f"{parameters.num_price_points - 1} intervals / 价格间隔"
    )

    for environment_name, noise_std in (
        ("LOW NOISE / 低噪声", parameters.noise_std),
        ("HIGH NOISE / 高噪声", 100.0),
    ):
        rows = build_resolution_table(parameters, noise_std)
        print(f"\n{environment_name}: sigma_u = {noise_std}")
        print(
            "v          one-action dp    price-grid dp    ratio    suggested n_p"
        )
        for row in rows:
            print(
                f"{row['fundamental_value']: .6f}   "
                f"{row['one_action_price_movement']: .9f}    "
                f"{row['one_price_grid_step']: .9f}   "
                f"{row['movement_to_grid_step_ratio']: .3f}       "
                f"{row['smallest_price_point_count_for_step_visibility']:>4}"
            )

        if noise_std == parameters.noise_std:
            # Paper footnote-25 check for the low-noise baseline. / 对低噪声
            # 基准执行论文脚注 25 的机械检查。
            assert all(
                0.9 <= float(row["movement_to_grid_step_ratio"]) <= 1.2
                for row in rows
            )

    print("\nValidation passed / 验证通过")


if __name__ == "__main__":
    main()
