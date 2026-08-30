"""Q-learning pieces of the informed speculator. / 知情投机者的 Q-learning。

    Q_0(s, x)  = 1/((1-rho) n_x) sum_{x_-i in X} [v - (v_bar + lambda^N (x + (I-1) x_-i))] x   (p.25)
    eps_t(v)   = exp(-beta t(v))                                                              (4.3)
    action     = argmax Q w.p. 1-eps, uniform on X w.p. eps                                   (2.6)
    Q update   = (1-a) Q + a [pi + rho * mean_{v'} max_x' Q((p_t, v_t, v'), x')]  (2.4) + OA p.60
"""

import math

import numpy as np

from dgj._jit import njit
from dgj.players.speculator import policy as policy_module


def initial_q_block(
    value_grid: np.ndarray,
    value_mean: float,
    orders: np.ndarray,
    number_of_speculators: int,
    nash_price_impact: float,
    discount_factor: float,
) -> np.ndarray:
    """(n_v, n_x) block of Q_0; it does not depend on p_{t-1} or v_{t-1}."""
    n_v, n_x = orders.shape
    block = np.empty((n_v, n_x))
    for k in range(n_v):
        v = float(value_grid[k])
        own = orders[k, :]                       # x
        other_mean = orders[k, :].mean()         # uniform opponent -> mean of x_-i
        # sum over x_-i is linear in x_-i, so the average opponent order suffices
        price = value_mean + nash_price_impact * (own + (number_of_speculators - 1) * other_mean)
        block[k, :] = (v - price) * own / (1.0 - discount_factor)
    return block


def initial_q_table(block: np.ndarray, number_of_speculators: int, num_prices: int) -> np.ndarray:
    """Tile the block over all (p, v_lag) and agents -> (I, S, n_x)."""
    n_v, n_x = block.shape
    per_agent = np.tile(block, (num_prices * n_v, 1))          # rows ordered (p, v_lag, v_cur)
    return np.repeat(per_agent[None, :, :], number_of_speculators, axis=0).copy()


@njit
def exploration_rate(visits_of_current_value: int, exploration_decay: float) -> float:
    return math.exp(-exploration_decay * visits_of_current_value)


@njit
def choose_action(policy_mask: int, eps: float, mode_draw: float, action_draw: int, num_actions: int) -> int:
    """(2.6) with pre-drawn randomness. / 使用预先抽好的随机数实现 epsilon-greedy。

    mode_draw ~ U[0,1) decides explore/exploit; action_draw is a large uniform
    integer used both for the exploratory action and for exact-tie breaking.
    """
    if mode_draw < eps:
        return action_draw % num_actions
    ties = policy_module.popcount(policy_mask)
    if ties == 1:
        return policy_module.lowest_action(policy_mask)
    return policy_module.kth_action(policy_mask, action_draw % ties)


@njit
def expected_continuation(Q: np.ndarray, i: int, base: int, n_v: int) -> float:
    """mean over v' of max_x' Q[i, base + v', x'] — reads only OLD rows."""
    total = 0.0
    n_x = Q.shape[2]
    for k in range(n_v):
        row = base + k
        best = Q[i, row, 0]
        for j in range(1, n_x):
            if Q[i, row, j] > best:
                best = Q[i, row, j]
        total += best
    return total / n_v


@njit
def update(Q, i, s, a, profit, base, n_v, learning_rate, discount_factor):
    """(2.4): one cell only. Continuation is computed before the single write."""
    continuation = expected_continuation(Q, i, base, n_v)
    Q[i, s, a] = (1.0 - learning_rate) * Q[i, s, a] + learning_rate * (
        profit + discount_factor * continuation
    )
