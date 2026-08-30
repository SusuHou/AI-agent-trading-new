"""Action space X(v): n_x orders between the cartel and Nash strategies. / 动作空间。

Paper 4.2: for v > v_bar the interval is [x^M - iota(x^N - x^M), x^N + iota(x^N - x^M)];
for v < v_bar the mirror image. Both are produced by n_x multipliers
c_j in [chi^M - iota*gap, chi^N + iota*gap] applied as x_j(v) = (v - v_bar) c_j.
"""

import numpy as np


def multipliers(nash_intensity: float, cartel_intensity: float, widening: float, num_actions: int) -> np.ndarray:
    if nash_intensity <= cartel_intensity:
        raise ValueError("calibration requires chi^N > chi^M")
    gap = nash_intensity - cartel_intensity
    return np.linspace(
        cartel_intensity - widening * gap,
        nash_intensity + widening * gap,
        num_actions,
    )


def orders_table(value_grid: np.ndarray, value_mean: float, action_multipliers: np.ndarray) -> np.ndarray:
    """orders[v_idx, action] = raw order x. / 每个价值点下每个动作的实际订单。"""
    signal = np.asarray(value_grid, dtype=float) - value_mean
    return np.outer(signal, np.asarray(action_multipliers, dtype=float))
