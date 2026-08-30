"""Step 12: build the paper's 15-choice action grid.

步骤 12：建立论文中的 15 个交易动作。

Run / 运行:
    py -3 steps/step_12_action_grid.py

The economic action is the raw order x.  Its 15 allowed values depend on the
current fundamental value v.  For readable code, we first build 15 reusable
multipliers c_j and then calculate each raw order as

    x_j(v) = (v - v_bar) * c_j.

经济动作是实际订单 x。15 个合法订单量取决于当前基本价值 v。为了让代码容易读，
我们先建立 15 个可复用的乘数 c_j，再计算 x_j(v) = (v-v_bar)c_j。

This representation does not force the learned policy to be linear: the later
AI may choose a different action index j in every state.

这种表示并不强迫 AI 学到线性策略：以后 AI 可以在不同状态选择不同的动作编号 j。
"""

from math import isclose
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


def build_action_multiplier_grid(
    nash_intensity: float,
    cartel_intensity: float,
    grid_widening: float,
    number_of_actions: int,
) -> list[float]:
    """Return equally spaced action multipliers c_1,...,c_n.

    返回等间距的动作乘数 c_1,...,c_n。

    The paper starts with the gap between the Nash and cartel intensities and
    widens the interval by iota times that gap on BOTH sides:

        gap   = chi^N - chi^M
        lower = chi^M - iota * gap
        upper = chi^N + iota * gap

    论文先计算纳什与卡特尔交易强度之间的差距，然后在两端各扩大 iota 倍差距。
    """

    if nash_intensity <= 0.0 or cartel_intensity <= 0.0:
        raise ValueError(
            "Both benchmark intensities must be positive. / "
            "两个基准交易强度都必须大于零。"
        )
    if nash_intensity <= cartel_intensity:
        raise ValueError(
            "The paper calibration requires chi^N > chi^M. / "
            "论文校准要求 chi^N 大于 chi^M。"
        )
    if grid_widening < 0.0:
        raise ValueError("iota cannot be negative. / iota 不能为负。")
    if not isinstance(number_of_actions, int) or number_of_actions < 2:
        raise ValueError(
            "number_of_actions must be an integer of at least 2. / "
            "动作数量必须是至少为 2 的整数。"
        )

    benchmark_gap = nash_intensity - cartel_intensity
    lower_multiplier = cartel_intensity - grid_widening * benchmark_gap
    upper_multiplier = nash_intensity + grid_widening * benchmark_gap
    spacing = (
        upper_multiplier - lower_multiplier
    ) / (
        number_of_actions - 1
    )

    return [
        lower_multiplier + action_index * spacing
        for action_index in range(number_of_actions)
    ]


def calculate_orders_for_value(
    fundamental_value: float,
    value_mean: float,
    action_multipliers: list[float],
) -> list[float]:
    """Convert the 15 multipliers into raw order choices for one value.

    把 15 个乘数转换成某个基本价值下的实际订单选择。

    Positive x means buy; negative x means short; zero means no trade.
    / x 为正表示买入，为负表示卖空，等于零表示不交易。
    """

    if not action_multipliers:
        raise ValueError(
            "action_multipliers cannot be empty. / 动作乘数不能为空。"
        )

    value_signal = fundamental_value - value_mean
    return [
        value_signal * multiplier
        for multiplier in action_multipliers
    ]


def main() -> None:
    """Build and validate both paper action grids. / 建立并验证两种论文环境的动作网格。

    Beginner note / 初学者提示:
    The model logic is only the two functions above. Everything below is a
    transparent test and printout, not another economic rule.

    真正的模型逻辑只有上面两个函数。下面全部是公开的测试和打印，不是新的经济规则。
    """

    # First use a tiny hand-check that does not depend on Step 10.
    # 先做一个不依赖第 10 步、可以手算的小测试。
    toy_grid = build_action_multiplier_grid(
        nash_intensity=4.0,
        cartel_intensity=2.0,
        grid_widening=0.1,
        number_of_actions=5,
    )
    expected_toy_grid = [1.8, 2.4, 3.0, 3.6, 4.2]
    assert all(
        isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(toy_grid, expected_toy_grid)
    )

    toy_buy_orders = calculate_orders_for_value(1.5, 1.0, toy_grid)
    toy_short_orders = calculate_orders_for_value(0.5, 1.0, toy_grid)
    assert all(
        isclose(buy, -short, abs_tol=1e-12)
        for buy, short in zip(toy_buy_orders, toy_short_orders)
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

    print("Step 12: Action grid / 步骤 12：动作网格")
    print(
        "Paper baseline number of actions n_x / 论文基准动作数量: "
        f"{parameters.num_action_points}"
    )
    print(
        "Grid widening iota / 网格扩展参数: "
        f"{parameters.grid_widening:.1f}"
    )
    print(
        "Important / 重要: the AI later chooses one raw order x from the "
        "15 choices belonging to the current v. / AI 以后从当前 v 对应的 15 个"
        "实际订单 x 中选择一个。"
    )

    noise_environments = (
        ("LOW NOISE / 低噪声", parameters.noise_std),
        ("HIGH NOISE / 高噪声", 100.0),
    )

    for environment_name, noise_std in noise_environments:
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

        nash_intensity = nash_solution["intensity"]
        cartel_intensity = cartel_solution["intensity"]
        action_multipliers = build_action_multiplier_grid(
            nash_intensity,
            cartel_intensity,
            parameters.grid_widening,
            parameters.num_action_points,
        )
        action_orders_by_value = [
            calculate_orders_for_value(
                float(fundamental_value),
                parameters.value_mean,
                action_multipliers,
            )
            for fundamental_value in value_grid
        ]

        benchmark_gap = nash_intensity - cartel_intensity
        expected_lower = (
            cartel_intensity - parameters.grid_widening * benchmark_gap
        )
        expected_upper = (
            nash_intensity + parameters.grid_widening * benchmark_gap
        )
        spacing = action_multipliers[1] - action_multipliers[0]

        # Structural checks / 结构检查。
        assert len(action_multipliers) == parameters.num_action_points
        assert len(action_orders_by_value) == parameters.num_value_points
        assert all(
            len(one_value_orders) == parameters.num_action_points
            for one_value_orders in action_orders_by_value
        )
        assert isclose(action_multipliers[0], expected_lower, abs_tol=1e-12)
        assert isclose(action_multipliers[-1], expected_upper, abs_tol=1e-12)
        assert all(
            isclose(
                action_multipliers[index + 1] - action_multipliers[index],
                spacing,
                abs_tol=1e-12,
            )
            for index in range(len(action_multipliers) - 1)
        )
        assert action_multipliers[0] < cartel_intensity
        assert cartel_intensity < nash_intensity
        assert nash_intensity < action_multipliers[-1]

        # The value grid is symmetric, so opposite value signals must produce
        # opposite raw orders at the same action index.
        # 价值网格是对称的，因此相反的价值信号在相同动作编号下必须产生相反订单。
        for low_value_orders, high_value_orders in zip(
            action_orders_by_value,
            reversed(action_orders_by_value),
        ):
            assert all(
                isclose(low_order, -high_order, abs_tol=1e-10)
                for low_order, high_order in zip(
                    low_value_orders,
                    high_value_orders,
                )
            )

        # If v equals its mean, every allowed order becomes zero.  The paper's
        # ten-point grid does not contain v_bar, but this checks the formula.
        # 若 v 等于均值，全部订单都为零。十点价值网格没有均值点；这里只检查公式。
        mean_value_orders = calculate_orders_for_value(
            parameters.value_mean,
            parameters.value_mean,
            action_multipliers,
        )
        assert all(isclose(order, 0.0, abs_tol=1e-12) for order in mean_value_orders)

        lowest_value = float(value_grid[0])
        highest_value = float(value_grid[-1])
        lowest_value_orders = action_orders_by_value[0]
        highest_value_orders = action_orders_by_value[-1]
        middle_action_index = parameters.num_action_points // 2

        print(f"\n{environment_name}: sigma_u = {noise_std:.1f}")
        print(f"  Cartel intensity chi^M / 卡特尔强度: {cartel_intensity:.12f}")
        print(f"  Nash intensity chi^N / 纳什强度: {nash_intensity:.12f}")
        print(
            "  Multiplier interval after widening / 扩展后的乘数区间: "
            f"[{action_multipliers[0]:.12f}, {action_multipliers[-1]:.12f}]"
        )
        print(f"  Equal spacing / 等间距: {spacing:.12f}")
        print(
            "  Raw action-grid shape (values x actions) / "
            f"实际动作表形状（价值数 x 动作数）: "
            f"{len(action_orders_by_value)} x {len(action_multipliers)}"
        )

        print("  All 15 multiplier indexes / 全部 15 个乘数编号:")
        for action_index, multiplier in enumerate(action_multipliers):
            print(f"    index {action_index:2d}: {multiplier:.9f}")

        print(
            f"  Example when v = {highest_value:.6f} (> v_bar): buy orders / 买入订单"
        )
        for action_index in (0, middle_action_index, parameters.num_action_points - 1):
            print(
                f"    index {action_index:2d}: "
                f"x = {highest_value_orders[action_index]:.9f}"
            )

        print(
            f"  Example when v = {lowest_value:.6f} (< v_bar): short orders / 卖空订单"
        )
        for action_index in (0, middle_action_index, parameters.num_action_points - 1):
            print(
                f"    index {action_index:2d}: "
                f"x = {lowest_value_orders[action_index]:.9f}"
            )

    print(
        "\nThe Nash and cartel benchmarks are inside the permitted interval, "
        "but they need not be exact grid points. / 纳什与卡特尔基准位于合法区间内，"
        "但不一定恰好等于某个离散网格点。"
    )
    print(
        "The action grid is fixed once parameters are fixed; no random seed is "
        "used here. / 参数固定后，动作网格就是固定的；本步骤不使用随机种子。"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
