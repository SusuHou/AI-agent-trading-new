"""Step 1: discretize the fundamental value. / 第一步：离散化基本价值。"""

from statistics import NormalDist

import numpy as np

from .parameters import PaperParameters


def build_value_grid(mean: float, std: float, num_points: int) -> np.ndarray:
    """Return the paper's equal-probability Gaussian grid. / 返回论文的等概率高斯网格。

    Formula: v_k = mean + std * Phi^-1((2k-1)/(2n)).
    Here Phi^-1 converts a probability into a standard-normal quantile.
    / 公式：v_k = mean + std * Phi^-1((2k-1)/(2n))。
    Phi^-1 把概率转换为标准正态分布的分位数。
    """
    if std <= 0:
        raise ValueError("std must be positive / 标准差必须为正")
    if num_points < 2:
        raise ValueError("num_points must be at least 2 / 网格至少需要 2 点")

    normal = NormalDist()
    probabilities = (2 * np.arange(1, num_points + 1) - 1) / (2 * num_points)
    quantiles = np.array([normal.inv_cdf(float(p)) for p in probabilities])
    return mean + std * quantiles


def discrete_value_std(value_grid: np.ndarray, mean: float) -> float:
    """Compute sigma_v_hat from footnote 24. / 计算脚注 24 的 sigma_v_hat。

    The paper uses the known grid mean rather than estimating a sample mean.
    / 论文使用已知的网格均值，而不是重新估计样本均值。
    """
    grid = np.asarray(value_grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("value_grid must be a non-empty 1D array / 价值网格必须是一维且非空")
    return float(np.sqrt(np.mean((grid - mean) ** 2)))


def main() -> None:
    """Print Step 1 results for visual checking. / 打印第一步结果，方便人工核对。"""
    parameters = PaperParameters()
    grid = build_value_grid(
        parameters.value_mean,
        parameters.value_std,
        parameters.num_value_points,
    )
    sigma_hat = discrete_value_std(grid, parameters.value_mean)

    print("Fundamental-value grid / 基本价值网格:")
    print(np.array2string(grid, precision=6))
    print(f"Number of points / 网格点数: {grid.size}")
    print(f"Grid mean / 网格均值: {grid.mean():.6f}")
    print(f"Discrete std sigma_v_hat / 离散标准差: {sigma_hat:.6f}")


if __name__ == "__main__":
    main()

