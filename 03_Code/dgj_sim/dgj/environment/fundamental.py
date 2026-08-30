"""Fundamental value v_t. / 基本价值 v_t。

Paper 4.2: v_k = v_bar + sigma_v * Phi^-1((2k-1)/(2 n_v)), k = 1..n_v, equal probability.
Footnote 24: all benchmarks use the discrete std sigma_v_hat (0.938 at n_v = 10).
"""

from statistics import NormalDist

import numpy as np


def value_grid(mean: float, std: float, num_points: int) -> np.ndarray:
    """Equal-probability Gaussian quantile grid V. / 等概率高斯分位数网格。"""
    if std <= 0 or num_points < 2:
        raise ValueError("std must be positive and num_points >= 2")
    normal = NormalDist()
    probabilities = (2 * np.arange(1, num_points + 1) - 1) / (2 * num_points)
    return mean + std * np.array([normal.inv_cdf(float(p)) for p in probabilities])


def discrete_std(grid: np.ndarray, mean: float) -> float:
    """sigma_v_hat = sqrt(mean((v_k - v_bar)^2)), footnote 24. / 离散标准差。"""
    grid = np.asarray(grid, dtype=float)
    return float(np.sqrt(np.mean((grid - mean) ** 2)))
