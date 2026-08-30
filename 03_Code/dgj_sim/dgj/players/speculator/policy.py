"""Greedy policy as a bit mask of exact argmax actions. / 贪心策略位掩码。

policy[i, s] has bit j set iff Q[i, s, j] == max_j Q[i, s, :].
A period changes the policy iff any mask changes (a tie set that only
re-orders internally is NOT a change). Convergence = no change for
``convergence_periods`` consecutive periods (paper p.26).
"""

import numpy as np

from dgj._jit import njit


@njit
def argmax_mask(Q: np.ndarray, i: int, s: int) -> int:
    best = Q[i, s, 0]
    mask = 1
    for j in range(1, Q.shape[2]):
        q = Q[i, s, j]
        if q > best:
            best = q
            mask = 1 << j
        elif q == best:
            mask |= 1 << j
    return mask


@njit
def lowest_action(mask: int) -> int:
    """Measurement tie rule: lowest set bit. / 测量期并列规则：最小动作编号。"""
    j = 0
    while (mask >> j) & 1 == 0:
        j += 1
    return j


@njit
def kth_action(mask: int, k: int) -> int:
    """Training tie rule: k-th set bit (k already reduced mod the tie count)."""
    j = 0
    seen = -1
    while True:
        if (mask >> j) & 1:
            seen += 1
            if seen == k:
                return j
        j += 1


@njit
def popcount(mask: int) -> int:
    count = 0
    while mask:
        count += mask & 1
        mask >>= 1
    return count


def initial_policy(Q: np.ndarray) -> np.ndarray:
    """Vectorised argmax masks for every (agent, state). / 初始策略。"""
    is_max = Q == Q.max(axis=2, keepdims=True)
    bits = (1 << np.arange(Q.shape[2], dtype=np.int64))
    return (is_max * bits).sum(axis=2).astype(np.int64)


def actions_from_mask(mask: int, num_actions: int) -> tuple[int, ...]:
    return tuple(j for j in range(num_actions) if (int(mask) >> j) & 1)
