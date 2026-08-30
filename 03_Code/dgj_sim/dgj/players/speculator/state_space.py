"""State s_t = (p_{t-1}, v_{t-1}, v_t) as one integer. / 状态编码。

    id = ((p_idx * n_v) + v_lag) * n_v + v_cur

Consequence used by the kernel: the n_v possible next states (same p_t, v_t,
different v_{t+1}) are one CONTIGUOUS block of rows starting at
``next_block(p_idx, v_cur, n_v)``.

Replication choice A2: a continuous price maps to the nearest grid point,
clipped at the ends, ties to the lower point.
"""

import numpy as np

from dgj._jit import njit


def number_of_states(num_prices: int, num_values: int) -> int:
    return num_prices * num_values * num_values


@njit
def encode(p_idx: int, v_lag: int, v_cur: int, n_v: int) -> int:
    return (p_idx * n_v + v_lag) * n_v + v_cur


def decode(state_id: int, num_values: int) -> tuple[int, int, int]:
    p_idx, remainder = divmod(int(state_id), num_values * num_values)
    v_lag, v_cur = divmod(remainder, num_values)
    return p_idx, v_lag, v_cur


@njit
def next_block(p_idx: int, v_cur: int, n_v: int) -> int:
    """First row of the n_v contiguous next states (v' = 0..n_v-1)."""
    return (p_idx * n_v + v_cur) * n_v


@njit
def price_to_index(price: float, price_grid: np.ndarray) -> int:
    """A2: nearest grid point; clip outside; exact midpoint -> lower index."""
    n = price_grid.shape[0]
    if price <= price_grid[0]:
        return 0
    if price >= price_grid[n - 1]:
        return n - 1
    lo = 0
    hi = n - 1
    while hi - lo > 1:               # invariant: P[lo] < price <= P[hi]
        mid = (lo + hi) // 2
        if price_grid[mid] < price:
            lo = mid
        else:
            hi = mid
    if price - price_grid[lo] <= price_grid[hi] - price:
        return lo
    return hi
