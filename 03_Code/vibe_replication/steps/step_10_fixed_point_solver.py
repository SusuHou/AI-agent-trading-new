"""Step 10: solve the Nash and cartel price-impact fixed points.

步骤 10：求解纳什与卡特尔价格冲击的不动点。

Run / 运行:
    py -3 steps/step_10_fixed_point_solver.py

Steps 8 and 9 treated lambda^N and lambda^M as easy test inputs.  This step
connects those trading-intensity formulas to the market maker, the noise
trader, and the information-insensitive investors, and solves lambda itself.

第 8、9 步把 lambda^N 和 lambda^M 当作容易手算的测试输入。本步骤把交易强度公式
与做市商、噪声交易者和信息不敏感投资者连接起来，并真正求出 lambda。
"""

from math import isclose, sqrt
from pathlib import Path
import sys


# When this file is run directly from the steps folder, this small path setup
# lets Python find the reusable Step 1 code in src.  It changes no economics.
# 直接运行本文件时，这几行让 Python 能找到 src 中可复用的第 1 步代码；它不改变经济模型。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.parameters import PaperParameters
from src.step01_value_grid import build_value_grid, discrete_value_std
from step_08_nash_benchmark import calculate_nash_intensity
from step_09_cartel_benchmark import calculate_cartel_intensity


def calculate_gamma(
    number_of_speculators: int,
    trading_intensity: float,
    noise_std: float,
    discrete_fundamental_std: float,
) -> float:
    """Return the inference slope gamma from the paper's fixed point.

    返回论文不动点中的推断斜率 gamma。

    gamma = (I * chi) / ((I * chi)^2 + (sigma_u / sigma_v_hat)^2)

    Gamma tells us how strongly the market maker's expected value responds to
    total order flow. / Gamma 表示做市商对总订单流中的价值信息反应多强。
    """

    if not isinstance(number_of_speculators, int) or number_of_speculators < 1:
        raise ValueError(
            "I must be a positive integer. / I 必须是正整数。"
        )
    if trading_intensity <= 0.0:
        raise ValueError(
            "Trading intensity chi must be positive. / 交易强度 chi 必须大于零。"
        )
    if noise_std <= 0.0 or discrete_fundamental_std <= 0.0:
        raise ValueError(
            "Both standard deviations must be positive. / 两个标准差都必须大于零。"
        )

    aggregate_informed_intensity = (
        number_of_speculators * trading_intensity
    )
    noise_to_value_ratio = noise_std / discrete_fundamental_std

    return aggregate_informed_intensity / (
        aggregate_informed_intensity**2 + noise_to_value_ratio**2
    )


def calculate_implied_price_impact(
    gamma: float,
    investor_slope: float,
    pricing_error_weight: float,
) -> float:
    """Return the market maker's implied lambda.

    返回做市商公式所隐含的 lambda。

    lambda = (theta * gamma + xi) / (theta + xi^2)

    The information-insensitive investors enter through xi. / 信息不敏感投资者通过
    xi 进入这个公式。
    """

    if gamma < 0.0:
        raise ValueError("Gamma must be non-negative. / Gamma 不能为负。")
    if investor_slope < 0.0:
        raise ValueError(
            "Xi cannot be negative. / Xi 不能为负数。"
        )
    if pricing_error_weight <= 0.0:
        raise ValueError(
            "Theta must be positive. / Theta 必须大于零。"
        )

    return (
        pricing_error_weight * gamma + investor_slope
    ) / (
        pricing_error_weight + investor_slope**2
    )


def calculate_benchmark_intensity(
    benchmark: str,
    number_of_speculators: int,
    price_impact: float,
) -> float:
    """Reuse Step 8 or Step 9 to calculate chi from a candidate lambda.

    重用第 8 或第 9 步，根据候选 lambda 计算 chi。
    """

    if benchmark == "nash":
        return calculate_nash_intensity(
            number_of_speculators,
            price_impact,
        )
    if benchmark == "cartel":
        return calculate_cartel_intensity(
            number_of_speculators,
            price_impact,
        )

    raise ValueError(
        "benchmark must be 'nash' or 'cartel'. / "
        "benchmark 必须是 'nash' 或 'cartel'。"
    )


def calculate_fixed_point_residual(
    candidate_price_impact: float,
    benchmark: str,
    number_of_speculators: int,
    noise_std: float,
    discrete_fundamental_std: float,
    investor_slope: float,
    pricing_error_weight: float,
) -> float:
    """Return candidate lambda minus the lambda implied by the other equations.

    返回“候选 lambda 减去其他方程隐含的 lambda”。真正的不动点残差等于零。
    """

    intensity = calculate_benchmark_intensity(
        benchmark,
        number_of_speculators,
        candidate_price_impact,
    )
    gamma = calculate_gamma(
        number_of_speculators,
        intensity,
        noise_std,
        discrete_fundamental_std,
    )
    implied_price_impact = calculate_implied_price_impact(
        gamma,
        investor_slope,
        pricing_error_weight,
    )

    return candidate_price_impact - implied_price_impact


def calculate_contraction_slope_bound(
    benchmark: str,
    number_of_speculators: int,
    investor_slope: float,
    pricing_error_weight: float,
) -> float:
    """Return a sufficient upper bound for a simple contraction check.

    返回简单压缩映射检查所需的斜率上界。论文参数下该数小于 1；严格正参数下，
    残差函数的整体形状也保证只有一个正根。
    """

    if benchmark == "nash":
        aggregate_coefficient = (
            number_of_speculators / (number_of_speculators + 1)
        )
    elif benchmark == "cartel":
        aggregate_coefficient = 0.5
    else:
        raise ValueError(
            "benchmark must be 'nash' or 'cartel'. / "
            "benchmark 必须是 'nash' 或 'cartel'。"
        )

    denominator = pricing_error_weight + investor_slope**2
    return pricing_error_weight / (
        denominator * aggregate_coefficient
    )


def solve_price_impact_by_bisection(
    benchmark: str,
    number_of_speculators: int,
    noise_std: float,
    discrete_fundamental_std: float,
    investor_slope: float,
    pricing_error_weight: float,
    tolerance: float = 1e-18,
    maximum_iterations: int = 200,
) -> float:
    """Solve lambda = implied_lambda with a readable bisection algorithm.

    使用容易理解的二分法求解 lambda = implied_lambda。

    Bisection repeatedly keeps the half-interval that contains the zero of the
    residual. / 二分法每次保留“包含残差零点”的那一半区间。
    """

    if tolerance <= 0.0:
        raise ValueError("Tolerance must be positive. / 容差必须大于零。")
    if maximum_iterations < 1:
        raise ValueError(
            "maximum_iterations must be positive. / 最大迭代次数必须大于零。"
        )

    if not isinstance(number_of_speculators, int) or number_of_speculators < 1:
        raise ValueError(
            "I must be a positive integer. / I 必须是正整数。"
        )
    if noise_std <= 0.0 or discrete_fundamental_std <= 0.0:
        raise ValueError(
            "Both standard deviations must be positive. / 两个标准差都必须大于零。"
        )
    if investor_slope < 0.0 or pricing_error_weight <= 0.0:
        raise ValueError(
            "Xi must be non-negative and theta positive. / Xi 必须非负，theta 必须大于零。"
        )

    denominator = pricing_error_weight + investor_slope**2
    noise_to_value_ratio = noise_std / discrete_fundamental_std

    # The paper includes an efficient-price experiment with xi=0. At that
    # boundary, lambda=gamma and the positive fixed point has a closed form.
    # Evaluating the generic lower bracket at lambda=0 would ask Steps 8/9 to
    # divide by zero, so solve this valid boundary analytically instead.
    # / 论文包含 xi=0 的有效价格实验。此时 lambda=gamma，正不动点
    # 有闭式解。通用区间的下界 lambda=0 会导致除零，因此单独精确求解。
    if investor_slope == 0.0:
        if benchmark == "nash":
            aggregate_coefficient = (
                number_of_speculators / (number_of_speculators + 1)
            )
        elif benchmark == "cartel":
            aggregate_coefficient = 0.5
        else:
            raise ValueError(
                "benchmark must be 'nash' or 'cartel'. / "
                "benchmark 必须是 'nash' 或 'cartel'。"
            )
        return (
            sqrt(
                aggregate_coefficient
                * (1.0 - aggregate_coefficient)
            )
            / noise_to_value_ratio
        )

    # For strictly positive xi, the paper's residual has exactly one
    # positive root.  Also, gamma lies between zero and 1/(2r), where
    # r = sigma_u/sigma_v_hat.  These facts give a guaranteed bracket.
    # 对严格为正的 xi，论文残差函数只有一个正根；同时 gamma 位于 0 与 1/(2r)
    # 之间，因此可以得到可靠且只包含一个经济有效解的求根区间。
    maximum_gamma = 1.0 / (2.0 * noise_to_value_ratio)
    lower_bound = investor_slope / denominator
    upper_bound = (
        investor_slope + pricing_error_weight * maximum_gamma
    ) / denominator

    lower_residual = calculate_fixed_point_residual(
        lower_bound,
        benchmark,
        number_of_speculators,
        noise_std,
        discrete_fundamental_std,
        investor_slope,
        pricing_error_weight,
    )
    upper_residual = calculate_fixed_point_residual(
        upper_bound,
        benchmark,
        number_of_speculators,
        noise_std,
        discrete_fundamental_std,
        investor_slope,
        pricing_error_weight,
    )

    if lower_residual > 0.0 or upper_residual < 0.0:
        raise RuntimeError(
            "The fixed point was not bracketed. / 求根区间没有包住不动点。"
        )

    for _ in range(maximum_iterations):
        midpoint = (lower_bound + upper_bound) / 2.0
        midpoint_residual = calculate_fixed_point_residual(
            midpoint,
            benchmark,
            number_of_speculators,
            noise_std,
            discrete_fundamental_std,
            investor_slope,
            pricing_error_weight,
        )

        if (
            abs(midpoint_residual) <= tolerance
            or (upper_bound - lower_bound) / 2.0 <= tolerance
        ):
            return midpoint

        if midpoint_residual < 0.0:
            lower_bound = midpoint
        else:
            upper_bound = midpoint

    raise RuntimeError(
        "Bisection did not converge. / 二分法没有收敛。"
    )


def solve_benchmark_fixed_point(
    benchmark: str,
    number_of_speculators: int,
    noise_std: float,
    discrete_fundamental_std: float,
    investor_slope: float,
    pricing_error_weight: float,
) -> dict[str, float]:
    """Solve one benchmark and return all quantities needed for validation.

    求解一个基准，并返回验证所需的全部数值。
    """

    price_impact = solve_price_impact_by_bisection(
        benchmark,
        number_of_speculators,
        noise_std,
        discrete_fundamental_std,
        investor_slope,
        pricing_error_weight,
    )
    intensity = calculate_benchmark_intensity(
        benchmark,
        number_of_speculators,
        price_impact,
    )
    gamma = calculate_gamma(
        number_of_speculators,
        intensity,
        noise_std,
        discrete_fundamental_std,
    )
    implied_price_impact = calculate_implied_price_impact(
        gamma,
        investor_slope,
        pricing_error_weight,
    )
    fixed_point_residual = price_impact - implied_price_impact

    if benchmark == "nash":
        identity_residual = (
            (number_of_speculators + 1) * price_impact * intensity - 1.0
        )
    else:
        identity_residual = (
            2.0 * number_of_speculators * price_impact * intensity - 1.0
        )

    return {
        "price_impact": price_impact,
        "intensity": intensity,
        "gamma": gamma,
        "implied_price_impact": implied_price_impact,
        "fixed_point_residual": fixed_point_residual,
        "identity_residual": identity_residual,
        "contraction_slope_bound": calculate_contraction_slope_bound(
            benchmark,
            number_of_speculators,
            investor_slope,
            pricing_error_weight,
        ),
    }


def main() -> None:
    """Solve and validate both paper noise environments. / 求解并验证论文的两种噪声环境。"""

    parameters = PaperParameters()

    # Reuse Step 1 rather than typing sigma_v_hat by hand.
    # 重用第 1 步，而不是手动输入 sigma_v_hat。
    value_grid = build_value_grid(
        parameters.value_mean,
        parameters.value_std,
        parameters.num_value_points,
    )
    discrete_fundamental_std = discrete_value_std(
        value_grid,
        parameters.value_mean,
    )
    assert isclose(
        discrete_fundamental_std,
        0.937969795249,
        abs_tol=1e-12,
    )

    print("Step 10: Coupled fixed-point solver / 步骤 10：联立不动点求解器")
    print(
        "Discrete fundamental std sigma_v_hat from Step 1 / "
        f"来自第 1 步的离散标准差: {discrete_fundamental_std:.12f}"
    )
    print(
        f"Information-insensitive investor slope xi / 信息不敏感投资者斜率: "
        f"{parameters.investor_slope:.1f}"
    )
    print(
        f"Market-maker pricing-error weight theta / 做市商定价误差权重: "
        f"{parameters.pricing_error_weight:.1f}"
    )

    noise_environments = (
        ("LOW NOISE / 低噪声", parameters.noise_std),
        ("HIGH NOISE / 高噪声", 100.0),
    )

    for environment_name, noise_std in noise_environments:
        print(f"\n{environment_name}: sigma_u = {noise_std:.1f}")

        solutions = {
            "Nash / 纳什": solve_benchmark_fixed_point(
                "nash",
                parameters.num_speculators,
                noise_std,
                discrete_fundamental_std,
                parameters.investor_slope,
                parameters.pricing_error_weight,
            ),
            "Cartel / 卡特尔": solve_benchmark_fixed_point(
                "cartel",
                parameters.num_speculators,
                noise_std,
                discrete_fundamental_std,
                parameters.investor_slope,
                parameters.pricing_error_weight,
            ),
        }

        for benchmark_name, solution in solutions.items():
            assert solution["price_impact"] > 0.0
            assert solution["intensity"] > 0.0
            assert abs(solution["fixed_point_residual"]) <= 1e-13
            assert abs(solution["identity_residual"]) <= 1e-12
            assert solution["contraction_slope_bound"] < 1.0

            print(f"  {benchmark_name}:")
            print(
                "    Price impact lambda / 价格冲击: "
                f"{solution['price_impact']:.12f}"
            )
            print(
                "    Inference slope gamma / 推断斜率: "
                f"{solution['gamma']:.12f}"
            )
            print(
                "    Trading intensity chi / 交易强度: "
                f"{solution['intensity']:.9f}"
            )
            print(
                "    Fixed-point residual / 不动点残差: "
                f"{solution['fixed_point_residual']:.2e}"
            )
            print(
                "    Step 8/9 identity residual / 第 8/9 步恒等式残差: "
                f"{solution['identity_residual']:.2e}"
            )
            print(
                "    Contraction slope bound (< 1 here) / "
                "本参数下的压缩斜率上界: "
                f"{solution['contraction_slope_bound']:.9f}"
            )

    # Paper-valid boundary xi=0: the analytical solution must also satisfy the
    # original coupled equations. / 论文中有效的边界 xi=0：闭式解也必须满足原联立方程。
    zero_xi_solutions = {
        "Nash / 纳什": solve_benchmark_fixed_point(
            "nash",
            parameters.num_speculators,
            parameters.noise_std,
            discrete_fundamental_std,
            0.0,
            parameters.pricing_error_weight,
        ),
        "Cartel / 卡特尔": solve_benchmark_fixed_point(
            "cartel",
            parameters.num_speculators,
            parameters.noise_std,
            discrete_fundamental_std,
            0.0,
            parameters.pricing_error_weight,
        ),
    }
    print("\nZERO INSENSITIVE-INVESTOR SLOPE / 信息不敏感投资者斜率为零: xi = 0")
    for benchmark_name, solution in zero_xi_solutions.items():
        assert solution["price_impact"] > 0.0
        assert abs(solution["fixed_point_residual"]) <= 1e-13
        assert abs(solution["identity_residual"]) <= 1e-12
        assert isclose(
            solution["price_impact"],
            solution["gamma"],
            rel_tol=1e-12,
        )
        print(
            f"  {benchmark_name}: lambda={solution['price_impact']:.9f}, "
            f"gamma={solution['gamma']:.9f}, "
            f"residual={solution['fixed_point_residual']:.2e}"
        )

    print(
        "\nLambda is now solved rather than assumed. / "
        "现在 lambda 已由联立方程求出，不再是假设的测试输入。"
    )
    print(
        "Benchmark profits are deliberately postponed to Step 11. / "
        "基准利润特意留到第 11 步。"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
