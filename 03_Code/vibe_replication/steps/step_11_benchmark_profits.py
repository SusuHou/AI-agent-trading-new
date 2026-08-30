"""Step 11: calculate Nash and perfect-cartel benchmark profits.

步骤 11：计算纳什与完全卡特尔基准利润。

Run / 运行:
    py -3 steps/step_11_benchmark_profits.py

Step 10 solved the equilibrium price-impact coefficients lambda^N and
lambda^M.  This step uses those coefficients to calculate the unconditional
expected profit of ONE informed speculator in each benchmark.

第 10 步求出了均衡价格冲击系数 lambda^N 和 lambda^M。本步骤使用这些系数，
计算每个基准中“一位”知情投机者的无条件预期利润。
"""

from math import fsum, isclose
from pathlib import Path
import sys
from typing import Iterable


# Let this file find the reusable code in both src and steps when it is run
# directly.  This path setup changes no economic assumption.
# 直接运行本文件时，让它能找到 src 和 steps 中已经验证的代码；路径设置不改变经济假设。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from src.step01_value_grid import build_value_grid, discrete_value_std
from step_05_speculator_profit import calculate_profit
from step_10_fixed_point_solver import solve_benchmark_fixed_point


def calculate_nash_benchmark_profit(
    discrete_fundamental_std: float,
    number_of_speculators: int,
    nash_price_impact: float,
) -> float:
    """Return expected Nash profit for one informed speculator.

    返回一位知情投机者的纳什预期利润。

    pi^N = sigma_v_hat^2 / ((I + 1)^2 * lambda^N)
    """

    if discrete_fundamental_std <= 0.0:
        raise ValueError(
            "sigma_v_hat must be positive. / sigma_v_hat 必须大于零。"
        )
    if not isinstance(number_of_speculators, int) or number_of_speculators < 1:
        raise ValueError("I must be a positive integer. / I 必须是正整数。")
    if nash_price_impact <= 0.0:
        raise ValueError(
            "lambda^N must be positive. / lambda^N 必须大于零。"
        )

    fundamental_variance = discrete_fundamental_std**2
    return fundamental_variance / (
        (number_of_speculators + 1) ** 2 * nash_price_impact
    )


def calculate_cartel_benchmark_profit(
    discrete_fundamental_std: float,
    number_of_speculators: int,
    cartel_price_impact: float,
) -> float:
    """Return expected perfect-cartel profit for one member.

    返回完全卡特尔中一位成员的预期利润。

    pi^M = sigma_v_hat^2 / (4 * I * lambda^M)

    This is profit PER MEMBER, not the cartel's joint profit. / 这是每位成员的
    利润，不是整个卡特尔的共同利润。
    """

    if discrete_fundamental_std <= 0.0:
        raise ValueError(
            "sigma_v_hat must be positive. / sigma_v_hat 必须大于零。"
        )
    if not isinstance(number_of_speculators, int) or number_of_speculators < 1:
        raise ValueError("I must be a positive integer. / I 必须是正整数。")
    if cartel_price_impact <= 0.0:
        raise ValueError(
            "lambda^M must be positive. / lambda^M 必须大于零。"
        )

    fundamental_variance = discrete_fundamental_std**2
    return fundamental_variance / (
        4.0 * number_of_speculators * cartel_price_impact
    )


def calculate_direct_expected_profit(
    value_grid: Iterable[float],
    noise_orders: Iterable[float],
    value_mean: float,
    number_of_speculators: int,
    trading_intensity: float,
    price_impact: float,
) -> float:
    """Average the original one-period payoff over values and noise shocks.

    在价值点与噪声冲击上，直接平均最初的单期利润公式。

    For every pair (v, u), this function rebuilds the full benchmark path:
        x = chi(v-v_bar)
        y = I*x + u
        p = v_bar + lambda*y
        profit = (v-p)*x

    对每一组 (v, u)，本函数重新计算完整基准路径。这样可以独立检查上面两个
    简化后的利润公式，而不是用同一个公式检查自己。
    """

    values = [float(value) for value in value_grid]
    noises = [float(noise) for noise in noise_orders]

    if not values:
        raise ValueError("value_grid cannot be empty. / 价值网格不能为空。")
    if not noises:
        raise ValueError("noise_orders cannot be empty. / 噪声订单不能为空。")
    if not isinstance(number_of_speculators, int) or number_of_speculators < 1:
        raise ValueError("I must be a positive integer. / I 必须是正整数。")
    if trading_intensity <= 0.0 or price_impact <= 0.0:
        raise ValueError(
            "chi and lambda must be positive. / chi 和 lambda 必须大于零。"
        )

    average_noise = fsum(noises) / len(noises)
    if not isclose(average_noise, 0.0, abs_tol=1e-12):
        raise ValueError(
            "This expected-profit check requires mean-zero noise. / "
            "这个预期利润检查要求噪声均值为零。"
        )

    realized_profits: list[float] = []

    for fundamental_value in values:
        value_signal = fundamental_value - value_mean
        one_trader_order = trading_intensity * value_signal

        for noise_order in noises:
            total_order_flow = (
                number_of_speculators * one_trader_order + noise_order
            )
            price = value_mean + price_impact * total_order_flow
            profit = calculate_profit(
                fundamental_value,
                price,
                one_trader_order,
            )
            realized_profits.append(profit)

    return fsum(realized_profits) / len(realized_profits)


def main() -> None:
    """Solve, calculate, and validate both paper benchmarks. / 求解并验证两个论文基准。"""

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
    fundamental_variance = discrete_fundamental_std**2

    print("Step 11: Benchmark expected profits / 步骤 11：基准预期利润")
    print(
        "Profit unit / 利润单位: expected profit PER informed speculator / "
        "每位知情投机者的预期利润"
    )
    print(
        "Discrete variance sigma_v_hat^2 / 离散基本价值方差: "
        f"{fundamental_variance:.12f}"
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

        nash_formula_profit = calculate_nash_benchmark_profit(
            discrete_fundamental_std,
            parameters.num_speculators,
            nash_solution["price_impact"],
        )
        cartel_formula_profit = calculate_cartel_benchmark_profit(
            discrete_fundamental_std,
            parameters.num_speculators,
            cartel_solution["price_impact"],
        )

        # The pair (-sigma_u, +sigma_u) has mean zero.  Profit is linear in
        # u, so this symmetric pair checks the exact noise expectation without
        # a slow random simulation.  It is a test device, not a replacement
        # for the Gaussian noise process used later.
        # (-sigma_u, +sigma_u) 的平均值为零。利润对 u 是线性的，因此这一对冲击
        # 可以精确检查噪声期望，无需随机模拟；它只是测试工具，不会取代后面的高斯噪声过程。
        symmetric_noise_orders = (-noise_std, noise_std)

        nash_direct_profit = calculate_direct_expected_profit(
            value_grid,
            symmetric_noise_orders,
            parameters.value_mean,
            parameters.num_speculators,
            nash_solution["intensity"],
            nash_solution["price_impact"],
        )
        cartel_direct_profit = calculate_direct_expected_profit(
            value_grid,
            symmetric_noise_orders,
            parameters.value_mean,
            parameters.num_speculators,
            cartel_solution["intensity"],
            cartel_solution["price_impact"],
        )

        assert isclose(
            nash_formula_profit,
            nash_direct_profit,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        assert isclose(
            cartel_formula_profit,
            cartel_direct_profit,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        assert cartel_formula_profit > nash_formula_profit

        cartel_joint_profit = (
            parameters.num_speculators * cartel_formula_profit
        )

        print(f"\n{environment_name}: sigma_u = {noise_std:.1f}")
        print("  Nash / 纳什:")
        print(
            "    lambda^N from Step 10 / 第 10 步的 lambda^N: "
            f"{nash_solution['price_impact']:.12f}"
        )
        print(
            "    Formula profit per trader / 公式计算的每人利润: "
            f"{nash_formula_profit:.12f}"
        )
        print(
            "    Direct value-and-noise check / 价值与噪声直接检查: "
            f"{nash_direct_profit:.12f}"
        )
        print(
            "    Formula-minus-direct residual / 公式减直接计算的残差: "
            f"{nash_formula_profit - nash_direct_profit:.2e}"
        )

        print("  Perfect cartel / 完全卡特尔:")
        print(
            "    lambda^M from Step 10 / 第 10 步的 lambda^M: "
            f"{cartel_solution['price_impact']:.12f}"
        )
        print(
            "    Formula profit per member / 公式计算的每位成员利润: "
            f"{cartel_formula_profit:.12f}"
        )
        print(
            "    Direct value-and-noise check / 价值与噪声直接检查: "
            f"{cartel_direct_profit:.12f}"
        )
        print(
            "    Formula-minus-direct residual / 公式减直接计算的残差: "
            f"{cartel_formula_profit - cartel_direct_profit:.2e}"
        )
        print(
            "    Joint profit of all cartel members / 卡特尔全部成员共同利润: "
            f"{cartel_joint_profit:.12f}"
        )
        print(
            "  Per-trader cartel advantage over Nash / 每人卡特尔利润减纳什利润: "
            f"{cartel_formula_profit - nash_formula_profit:.12f}"
        )

    print(
        "\nThe noise trader is included in the direct checks. Its mean-zero "
        "profit effect cancels in expectation, while sigma_u still changes "
        "the Step 10 equilibrium lambda. / 直接检查包含噪声交易者。其均值为零的利润"
        "影响在取期望时相互抵消，但 sigma_u 仍会改变第 10 步的均衡 lambda。"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
