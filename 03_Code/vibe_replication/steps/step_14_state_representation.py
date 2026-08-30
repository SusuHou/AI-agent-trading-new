"""Step 14: turn the paper's market state into integer indexes.

步骤 14：把论文中的市场状态转换成整数编号。

Run / 运行:
    py -3 steps/step_14_state_representation.py

Paper rule / 论文规则:
    s_t = (p_{t-1}, v_{t-1}, v_t) in P x V x V

Important replication choice / 重要复现选择:
The paper does not say how a continuous realized price is mapped to the finite
price grid P.  We use the nearest price-grid point, clip observations outside
the grid to its endpoints, and choose the lower point at an exact midpoint.

论文没有说明如何把连续的实际价格映射到有限价格网格 P。我们的明确选择是：映射
到最近的价格点；超出网格时映射到端点；恰好位于中点时选择较低的价格点。

Only the STATE uses the mapped price.  Later calculations of investor demand,
profit, and market-maker history must continue to use the continuous price.
The full simulation should also count how often endpoint clipping occurs.

只有“状态”使用映射后的价格。之后的信息不敏感投资者需求、利润和做市商历史仍须
使用连续价格。完整模拟还应记录价格落到网格外并被截断的频率。
"""

from bisect import bisect_left
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
from step_13_price_grid import build_price_grids_by_value


def _require_strictly_increasing(
    grid: Sequence[float],
    grid_name: str,
) -> None:
    """Check once that a grid is usable. / 检查网格是否可以安全使用。"""

    if len(grid) == 0:
        raise ValueError(f"{grid_name} cannot be empty. / {grid_name} 不能为空。")
    if not all(isfinite(point) for point in grid):
        raise ValueError(
            f"{grid_name} must contain finite numbers. / "
            f"{grid_name} 必须只包含有限数值。"
        )
    if any(grid[index] >= grid[index + 1] for index in range(len(grid) - 1)):
        raise ValueError(
            f"{grid_name} must be strictly increasing. / "
            f"{grid_name} 必须严格递增。"
        )


def fundamental_value_to_index(
    fundamental_value: float,
    value_grid: Sequence[float],
    tolerance: float = 1e-12,
) -> int:
    """Return the index of a value that is already on V.

    返回一个已经位于价值网格 V 上的基本价值编号。

    Unlike price, fundamental value is drawn directly FROM V in the final
    simulation.  Therefore an off-grid value is an error; it is not silently
    rounded to a different economic state.

    与价格不同，正式模拟中的基本价值直接从 V 抽取。因此，网格外价值表示代码有
    错误；我们不会悄悄把它舍入成另一个经济状态。
    """

    if len(value_grid) == 0:
        raise ValueError("value_grid cannot be empty. / 价值网格不能为空。")
    if tolerance < 0.0:
        raise ValueError("tolerance cannot be negative. / 容差不能为负。")

    for value_index, grid_value in enumerate(value_grid):
        if isclose(
            fundamental_value,
            grid_value,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            return value_index

    raise ValueError(
        "Fundamental value is not on V. / 基本价值不在价值网格 V 上。"
    )


def continuous_price_to_index(
    continuous_price: float,
    price_grid: Sequence[float],
) -> int:
    """Map a continuous price to the nearest point in P.

    把连续价格映射到价格网格 P 中最近的点。

    ``bisect_left`` finds where the price would be inserted into the ordered
    grid.  We then compare the neighboring points. / ``bisect_left`` 找到该
    价格在有序网格中的插入位置，然后我们比较左右两个相邻点。
    """

    if not isfinite(continuous_price):
        raise ValueError("Price must be finite. / 价格必须是有限数值。")
    _require_strictly_increasing(price_grid, "price_grid")

    insertion_index = bisect_left(price_grid, continuous_price)

    # Outside the grid: use the nearest endpoint. / 网格之外：使用最近端点。
    if insertion_index == 0:
        return 0
    if insertion_index == len(price_grid):
        return len(price_grid) - 1

    lower_index = insertion_index - 1
    upper_index = insertion_index
    distance_to_lower = continuous_price - price_grid[lower_index]
    distance_to_upper = price_grid[upper_index] - continuous_price

    # Floating-point arithmetic may represent a computed midpoint a few bits
    # away from an exact tie.  Treat numerically equal distances as a tie and
    # choose the lower index. / 浮点数可能让计算出的中点偏离极小几位；距离在
    # 数值上相等时仍按并列处理，并选择较低编号。
    if distance_to_lower <= distance_to_upper or isclose(
        distance_to_lower,
        distance_to_upper,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        return lower_index
    return upper_index


def validate_price_grids_by_value(
    price_grids: Sequence[Sequence[float]],
    number_of_values: int,
    number_of_prices: int,
) -> tuple[tuple[float, ...], ...]:
    """Validate and freeze a rectangular n_v by n_p price-grid matrix.

    验证并冻结一个 n_v 行、n_p 列的价格网格矩阵。

    A flat global grid is rejected deliberately.  Failing loudly prevents an
    old experiment from silently using the wrong state encoding. / 我们故意拒绝
    旧的一维全局网格，避免旧实验悄悄使用错误的状态编码。
    """

    if number_of_values < 1 or number_of_prices < 2:
        raise ValueError("Grid dimensions are invalid. / 网格维度无效。")
    if len(price_grids) != number_of_values:
        raise ValueError(
            "price_grids must have one row per value. / "
            "price_grids 必须为每个价值提供一行。"
        )

    frozen_rows: list[tuple[float, ...]] = []
    for value_index, row in enumerate(price_grids):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(
                "Each price-grid row must be a sequence. / 每个价格行必须是序列。"
            )
        if len(row) != number_of_prices:
            raise ValueError(
                f"Price row {value_index} must contain {number_of_prices} "
                f"points. / 价格行 {value_index} 必须包含 {number_of_prices} 个点。"
            )
        converted_row = tuple(float(price) for price in row)
        _require_strictly_increasing(
            converted_row,
            f"price_grids[{value_index}]",
        )
        frozen_rows.append(converted_row)

    return tuple(frozen_rows)


def number_of_price_points(
    price_grids: Sequence[Sequence[float]],
) -> int:
    """Return n_p, the number of columns, not the number of rows.

    返回列数 n_p，而不是价格矩阵的行数。
    """

    if len(price_grids) < 1:
        raise ValueError("price_grids cannot be empty. / 价格网格不能为空。")
    first_row = price_grids[0]
    if isinstance(first_row, (str, bytes)) or not isinstance(first_row, Sequence):
        raise TypeError(
            "Expected a two-dimensional price grid. / 需要二维价格网格。"
        )
    if len(first_row) < 1:
        raise ValueError("Price rows cannot be empty. / 价格行不能为空。")
    if any(len(row) != len(first_row) for row in price_grids):
        raise ValueError("All price rows must have equal length. / 所有价格行必须等长。")
    return len(first_row)


def build_state_indexes(
    previous_price: float,
    previous_value: float,
    current_value: float,
    price_grids: Sequence[Sequence[float]],
    value_grid: Sequence[float],
) -> tuple[int, int, int]:
    """Return (previous-price, previous-value, current-value) indexes.

    返回（上一期价格、上一期价值、本期价值）的三个编号。
    """

    previous_value_index = fundamental_value_to_index(
        previous_value,
        value_grid,
    )
    current_value_index = fundamental_value_to_index(
        current_value,
        value_grid,
    )
    if len(price_grids) != len(value_grid):
        raise ValueError(
            "Price rows must match value-grid points. / 价格行数必须与价值点数一致。"
        )
    selected_price_row = price_grids[previous_value_index]
    return (
        continuous_price_to_index(previous_price, selected_price_row),
        previous_value_index,
        current_value_index,
    )


def number_of_states(number_of_prices: int, number_of_values: int) -> int:
    """Return |P x V x V| = n_p * n_v * n_v. / 返回状态总数。"""

    if number_of_prices < 1 or number_of_values < 1:
        raise ValueError("Grid sizes must be positive. / 网格大小必须为正。")
    return number_of_prices * number_of_values * number_of_values


def encode_state_index(
    state_indexes: tuple[int, int, int],
    number_of_prices: int,
    number_of_values: int,
) -> int:
    """Compress three indexes into one unique state number.

    把三个编号压缩成一个唯一的状态编号。

    Formula / 公式:
        ((price_index * n_v) + previous_value_index) * n_v
        + current_value_index
    """

    price_index, previous_value_index, current_value_index = state_indexes
    if not 0 <= price_index < number_of_prices:
        raise ValueError("Invalid price index. / 价格编号无效。")
    if not 0 <= previous_value_index < number_of_values:
        raise ValueError("Invalid previous-value index. / 上期价值编号无效。")
    if not 0 <= current_value_index < number_of_values:
        raise ValueError("Invalid current-value index. / 本期价值编号无效。")

    return (
        (price_index * number_of_values + previous_value_index)
        * number_of_values
        + current_value_index
    )


def decode_state_index(
    state_index: int,
    number_of_prices: int,
    number_of_values: int,
) -> tuple[int, int, int]:
    """Reverse one state number back into its three indexes.

    把一个状态编号还原成三个编号。
    """

    total_states = number_of_states(number_of_prices, number_of_values)
    if not 0 <= state_index < total_states:
        raise ValueError("Invalid state index. / 状态编号无效。")

    states_per_price = number_of_values * number_of_values
    price_index, remainder = divmod(state_index, states_per_price)
    previous_value_index, current_value_index = divmod(
        remainder,
        number_of_values,
    )
    return price_index, previous_value_index, current_value_index


def build_paper_price_grids(
    parameters: PaperParameters,
    value_grid: Sequence[float],
    noise_std: float,
) -> list[list[float]]:
    """Reuse Steps 10-13 to assemble all rows P(v_k). / 建立全部 P(v_k)。"""

    value_grid_std = discrete_value_std(value_grid, parameters.value_mean)
    nash_solution = solve_benchmark_fixed_point(
        "nash",
        parameters.num_speculators,
        noise_std,
        value_grid_std,
        parameters.investor_slope,
        parameters.pricing_error_weight,
    )
    cartel_solution = solve_benchmark_fixed_point(
        "cartel",
        parameters.num_speculators,
        noise_std,
        value_grid_std,
        parameters.investor_slope,
        parameters.pricing_error_weight,
    )

    return build_price_grids_by_value(
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


def build_paper_price_grid(
    parameters: PaperParameters,
    value_grid: Sequence[float],
    noise_std: float,
) -> list[list[float]]:
    """Compatibility name returning the new two-dimensional P(v).

    为旧 import 名称保留的兼容入口；返回值已经是新的二维 P(v)。

    New code should prefer ``build_paper_price_grids``. / 新代码应优先使用复数名称。
    """

    return build_paper_price_grids(parameters, value_grid, noise_std)


def main() -> None:
    """Build example states and run transparent checks. / 建立示例状态并检查。"""

    parameters = PaperParameters()
    value_grid = build_value_grid(
        parameters.value_mean,
        parameters.value_std,
        parameters.num_value_points,
    )
    low_noise_price_grids = build_paper_price_grids(
        parameters,
        value_grid,
        parameters.noise_std,
    )
    high_noise_price_grids = build_paper_price_grids(
        parameters,
        value_grid,
        100.0,
    )
    low_noise_price_grids = validate_price_grids_by_value(
        low_noise_price_grids,
        parameters.num_value_points,
        parameters.num_price_points,
    )
    high_noise_price_grids = validate_price_grids_by_value(
        high_noise_price_grids,
        parameters.num_value_points,
        parameters.num_price_points,
    )

    # 1. Every allowed value returns to its own index. / 每个允许价值都返回原编号。
    for expected_index, fundamental_value in enumerate(value_grid):
        assert fundamental_value_to_index(fundamental_value, value_grid) == expected_index

    # 2. Every point round-trips inside its OWN row. / 每个点在自己的价格行中返回原编号。
    for price_grids in (low_noise_price_grids, high_noise_price_grids):
        for price_grid in price_grids:
            for expected_index, grid_price in enumerate(price_grid):
                assert continuous_price_to_index(grid_price, price_grid) == expected_index

            spacing = price_grid[1] - price_grid[0]
            midpoint = (price_grid[10] + price_grid[11]) / 2.0
            tiny_move = spacing / 1_000_000.0
            assert continuous_price_to_index(midpoint - tiny_move, price_grid) == 10
            assert continuous_price_to_index(midpoint + tiny_move, price_grid) == 11
            assert continuous_price_to_index(midpoint, price_grid) == 10
            assert continuous_price_to_index(price_grid[0] - spacing, price_grid) == 0
            assert continuous_price_to_index(price_grid[-1] + spacing, price_grid) == 30

    # 3. A concrete paper-calibration state. / 一个使用论文校准值的具体状态。
    previous_value_index = 2
    previous_price_index = 16
    previous_value = value_grid[previous_value_index]
    current_value = value_grid[7]
    previous_price = low_noise_price_grids[previous_value_index][previous_price_index]
    state_tuple = build_state_indexes(
        previous_price,
        previous_value,
        current_value,
        low_noise_price_grids,
        value_grid,
    )
    assert state_tuple == (16, 2, 7)

    flat_state = encode_state_index(
        state_tuple,
        parameters.num_price_points,
        parameters.num_value_points,
    )
    assert flat_state == 1627
    assert decode_state_index(
        flat_state,
        parameters.num_price_points,
        parameters.num_value_points,
    ) == state_tuple

    # 4. Exhaustively prove that all 3,100 tuples have unique numbers and
    # decode correctly. / 穷举验证全部 3,100 个状态的编号唯一且可以还原。
    all_flat_indexes: set[int] = set()
    for price_index in range(parameters.num_price_points):
        for previous_value_index in range(parameters.num_value_points):
            for current_value_index in range(parameters.num_value_points):
                original_tuple = (
                    price_index,
                    previous_value_index,
                    current_value_index,
                )
                encoded = encode_state_index(
                    original_tuple,
                    parameters.num_price_points,
                    parameters.num_value_points,
                )
                assert decode_state_index(
                    encoded,
                    parameters.num_price_points,
                    parameters.num_value_points,
                ) == original_tuple
                all_flat_indexes.add(encoded)

    total_states = number_of_states(
        parameters.num_price_points,
        parameters.num_value_points,
    )
    assert total_states == 3_100
    assert len(all_flat_indexes) == total_states
    assert min(all_flat_indexes) == 0
    assert max(all_flat_indexes) == 3_099

    # 5. An arbitrary off-grid fundamental value must fail clearly.
    # 任意网格外基本价值必须明确报错。
    try:
        fundamental_value_to_index(123.456, value_grid)
    except ValueError:
        off_grid_value_was_rejected = True
    else:
        off_grid_value_was_rejected = False
    assert off_grid_value_was_rejected

    mapped_price_index = state_tuple[0]
    mapped_price = low_noise_price_grids[state_tuple[1]][mapped_price_index]

    print("Step 14: Row-aware state representation / 步骤 14：按价值选行的状态表示")
    print("Paper state / 论文状态: s_t = (p_(t-1), v_(t-1), v_t)")
    print(
        "Stored price-grid shape / 保存的价格网格形状: "
        f"({parameters.num_value_points}, {parameters.num_price_points})"
    )
    print(
        "State-index shape / 状态编号形状: "
        f"({parameters.num_price_points}, {parameters.num_value_points}, "
        f"{parameters.num_value_points})"
    )
    print(f"Number of possible states / 可能状态总数: {total_states}")
    print("\nExample / 示例:")
    print(f"  Continuous previous price / 连续上期价格: {previous_price:.6f}")
    print(
        "  Selected row and nearest price / 选择的价格行与最近价格点: "
        f"P(V[{state_tuple[1]}])[{mapped_price_index}] = {mapped_price:.9f}"
    )
    print(
        "  Previous fundamental and index / 上期基本价值与编号: "
        f"V[2] = {previous_value:.9f}"
    )
    print(
        "  Current fundamental and index / 本期基本价值与编号: "
        f"V[7] = {current_value:.9f}"
    )
    print(f"  Three-index state / 三编号状态: {state_tuple}")
    print(f"  One-number state / 单一状态编号: {flat_state}")
    print(
        "\nReplication choice / 复现选择: first select P(v_(t-1)), then use "
        "the nearest point; clip at row boundaries; lower point at an exact "
        "tie / 先选择 P(v_(t-1))，再取最近点；按该行边界截断；精确中点取低点"
    )
    print(
        "Continuous p is preserved for economic calculations. / "
        "经济计算仍保留并使用连续价格 p。"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
