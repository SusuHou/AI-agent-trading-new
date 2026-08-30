"""Theoretical benchmark speculators and the price-impact fixed point.

理论基准交易者（Nash / 完全 cartel）以及 lambda 的联立不动点。

    chi^N = 1 / ((I+1) lambda^N)          chi^M = 1 / (2 I lambda^M)
    gamma = I chi / ((I chi)^2 + (sigma_u / sigma_v_hat)^2)
    lambda = (theta gamma + xi) / (theta + xi^2)              (OA IA.2.9 / IA.2.12)
    pi^N   = sigma_v_hat^2 / ((I+1)^2 lambda^N)               (OA Prop IA.1)
    pi^M   = sigma_v_hat^2 / (4 I lambda^M)                   (OA Prop IA.2)
"""

from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class Benchmark:
    """Solved coefficients for one benchmark. / 一个基准的求解结果。"""

    name: str
    intensity: float      # chi
    price_impact: float   # lambda
    gamma: float
    residual: float

    def order(self, value: float, value_mean: float) -> float:
        """x^B(v) = chi (v - v_bar)."""
        return self.intensity * (value - value_mean)

    def price(self, total_order_flow: float, value_mean: float) -> float:
        """p^B(y) = v_bar + lambda y."""
        return value_mean + self.price_impact * total_order_flow


def _aggregate_coefficient(name: str, number_of_speculators: int) -> float:
    if name == "nash":
        return number_of_speculators / (number_of_speculators + 1)
    if name == "cartel":
        return 0.5
    raise ValueError("benchmark must be 'nash' or 'cartel'")


def intensity_from_lambda(name: str, number_of_speculators: int, price_impact: float) -> float:
    if name == "nash":
        return 1.0 / ((number_of_speculators + 1) * price_impact)
    if name == "cartel":
        return 1.0 / (2.0 * number_of_speculators * price_impact)
    raise ValueError("benchmark must be 'nash' or 'cartel'")


def gamma_from_intensity(number_of_speculators: int, intensity: float, noise_ratio: float) -> float:
    aggregate = number_of_speculators * intensity
    return aggregate / (aggregate * aggregate + noise_ratio * noise_ratio)


def implied_lambda(gamma: float, investor_slope: float, pricing_error_weight: float) -> float:
    return (pricing_error_weight * gamma + investor_slope) / (
        pricing_error_weight + investor_slope * investor_slope
    )


def residual(name, candidate, number_of_speculators, noise_ratio, investor_slope, pricing_error_weight):
    chi = intensity_from_lambda(name, number_of_speculators, candidate)
    gamma = gamma_from_intensity(number_of_speculators, chi, noise_ratio)
    return candidate - implied_lambda(gamma, investor_slope, pricing_error_weight)


def solve(
    name: str,
    number_of_speculators: int,
    noise_std: float,
    discrete_value_std: float,
    investor_slope: float,
    pricing_error_weight: float,
    tolerance: float = 1e-18,
) -> Benchmark:
    """Bisection on the unique positive root of lambda = implied_lambda(lambda).

    对 lambda 的唯一正根做二分法。xi = 0 时有闭式解 sqrt(a(1-a)) / r。
    """
    if noise_std <= 0 or discrete_value_std <= 0 or pricing_error_weight <= 0 or investor_slope < 0:
        raise ValueError("invalid inputs to the fixed point")
    noise_ratio = noise_std / discrete_value_std
    aggregate = _aggregate_coefficient(name, number_of_speculators)

    if investor_slope == 0.0:
        price_impact = sqrt(aggregate * (1.0 - aggregate)) / noise_ratio
    else:
        denominator = pricing_error_weight + investor_slope * investor_slope
        lower = investor_slope / denominator                          # gamma = 0
        upper = (investor_slope + pricing_error_weight / (2 * noise_ratio)) / denominator  # gamma max
        f_lower = residual(name, lower, number_of_speculators, noise_ratio, investor_slope, pricing_error_weight)
        f_upper = residual(name, upper, number_of_speculators, noise_ratio, investor_slope, pricing_error_weight)
        if f_lower > 0 or f_upper < 0:
            raise RuntimeError("fixed point not bracketed")
        for _ in range(300):
            mid = 0.5 * (lower + upper)
            f_mid = residual(name, mid, number_of_speculators, noise_ratio, investor_slope, pricing_error_weight)
            if abs(f_mid) <= tolerance or 0.5 * (upper - lower) <= tolerance:
                break
            if f_mid < 0:
                lower = mid
            else:
                upper = mid
        price_impact = mid

    chi = intensity_from_lambda(name, number_of_speculators, price_impact)
    gamma = gamma_from_intensity(number_of_speculators, chi, noise_ratio)
    return Benchmark(
        name=name,
        intensity=chi,
        price_impact=price_impact,
        gamma=gamma,
        residual=price_impact - implied_lambda(gamma, investor_slope, pricing_error_weight),
    )


def expected_profit(benchmark: Benchmark, number_of_speculators: int, discrete_value_std: float) -> float:
    """Unconditional per-speculator profit, closed form. / 每位投机者的无条件预期利润。"""
    variance = discrete_value_std * discrete_value_std
    if benchmark.name == "nash":
        return variance / ((number_of_speculators + 1) ** 2 * benchmark.price_impact)
    return variance / (4.0 * number_of_speculators * benchmark.price_impact)


def matched_path_profit(
    benchmark: Benchmark,
    values: np.ndarray,
    noise_orders: np.ndarray,
    number_of_speculators: int,
    value_mean: float,
) -> np.ndarray:
    """OA IA.4.2 / IA.4.3: benchmark profit on the SAME realized (v_t, u_t) path.

    在同一条已实现 (v_t, u_t) 路径上重建基准利润；向量化。
    """
    x = benchmark.intensity * (np.asarray(values) - value_mean)
    y = number_of_speculators * x + np.asarray(noise_orders)
    p = value_mean + benchmark.price_impact * y
    return (np.asarray(values) - p) * x
