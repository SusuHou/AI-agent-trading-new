"""D_0: the market maker's history before period 0 — replication choice A3.

论文没有说明 t=0 时那 T_m 条历史从何而来。基准选择：与指定基准（Nash 或
cartel）一致的确定性平衡合成历史。做市商只拿到这些行，系数仍由它自己用 OLS 估计。

Rows are ordered in blocks of (one +/- noise pair) x (every value), so that
whichever prefix the rolling window later evicts stays mean-zero in noise.
"""

from statistics import NormalDist

import numpy as np

from dgj.environment.insensitive_investors import demand
from dgj.players.benchmarks import Benchmark


def balanced_noise_levels(noise_std: float, count: int) -> np.ndarray:
    """count symmetric Gaussian quantiles scaled to population std == noise_std."""
    if count < 2 or count % 2:
        raise ValueError("count must be an even integer >= 2")
    normal = NormalDist()
    half = count // 2
    magnitudes = np.array(
        [normal.inv_cdf((2 * (k + 1) - 1) / (2 * count)) for k in range(half, count)]
    )
    magnitudes *= noise_std / np.sqrt(np.mean(magnitudes ** 2))
    # greedy ordering keeps every prefix's second moment near sigma_u^2
    remaining = list(magnitudes)
    ordered = []
    cumulative = 0.0
    while remaining:
        target = (len(ordered) + 1) * noise_std * noise_std
        pick = min(range(len(remaining)), key=lambda k: (abs(cumulative + remaining[k] ** 2 - target), k))
        m = remaining.pop(pick)
        ordered.append(m)
        cumulative += m * m
    levels = np.empty(count)
    levels[0::2] = -np.array(ordered)
    levels[1::2] = np.array(ordered)
    return levels


def build_rows(
    benchmark: Benchmark,
    value_grid: np.ndarray,
    value_mean: float,
    investor_slope: float,
    noise_std: float,
    number_of_speculators: int,
    window: int,
) -> np.ndarray:
    """(T_m, 4) array of (v, p, z, y) rows consistent with ``benchmark``."""
    n_v = len(value_grid)
    if window % n_v or (window // n_v) % 2:
        raise ValueError("A3 requires T_m divisible by n_v with an even quotient")
    levels = balanced_noise_levels(noise_std, window // n_v)
    rows = np.empty((window, 4))
    r = 0
    for pair in range(0, len(levels), 2):
        for v in value_grid:
            x = benchmark.order(float(v), value_mean)
            for u in levels[pair:pair + 2]:
                y = number_of_speculators * x + u
                p = benchmark.price(y, value_mean)
                rows[r] = (v, p, demand(p, investor_slope, value_mean), y)
                r += 1
    return rows
