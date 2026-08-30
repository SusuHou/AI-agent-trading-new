"""Adaptive market maker, equations (4.1)-(4.2). / 自适应做市商。

Rolling dataset D_t = {(v, p, z, y)} over the last T_m periods (ring buffer
``hist``) plus running centred OLS statistics (``stats``) so that every period
costs O(1):

    z = xi_0 - xi_1 p         v = gamma_0 + gamma_1 y                       (4.1)
    lambda_hat = (theta gamma_1 + xi_1) / (theta + xi_1^2),  p_hat(y) = gamma_0 + lambda_hat y   (4.2)

THE RULE / 规则:  ``quote`` reads stats as they are BEFORE this period's row
is appended by ``observe``.  The kernel calls quote first, observe last.
"""

import numpy as np

from dgj._jit import njit

# hist columns / 历史列
H_V, H_P, H_Z, H_Y = 0, 1, 2, 3
# stats slots / 统计量槽位
S_N, S_MP, S_MZ, S_SPP, S_SPZ, S_MY, S_MV, S_SYY, S_SYV = range(9)
STATS_SIZE = 9
# cursor slots shared with the kernel (defined once here, imported by protocol)
C_P_IDX, C_V_LAG, C_V_CUR, C_STREAK, C_T, C_HEAD, C_NROWS, C_SINCE_RESYNC, C_APPENDS = range(9)
CURSOR_SIZE = 9


def new_history(window: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((window, 4)), np.zeros(STATS_SIZE)


@njit
def _pair_add(stats, ix_mx, ix_my, ix_sxx, ix_sxy, n_new, x, y):
    dx = x - stats[ix_mx]
    dy = y - stats[ix_my]
    mx = stats[ix_mx] + dx / n_new
    my = stats[ix_my] + dy / n_new
    stats[ix_sxx] += dx * (x - mx)
    stats[ix_sxy] += dx * (y - my)
    stats[ix_mx] = mx
    stats[ix_my] = my


@njit
def _pair_remove(stats, ix_mx, ix_my, ix_sxx, ix_sxy, n_new, x, y):
    mx_old = stats[ix_mx]
    my_old = stats[ix_my]
    mx = mx_old - (x - mx_old) / n_new
    my = my_old - (y - my_old) / n_new
    stats[ix_sxx] -= (x - mx_old) * (x - mx)
    stats[ix_sxy] -= (x - mx_old) * (y - my)
    stats[ix_mx] = mx
    stats[ix_my] = my


@njit
def add(stats, v, p, z, y):
    n_new = stats[S_N] + 1.0
    _pair_add(stats, S_MP, S_MZ, S_SPP, S_SPZ, n_new, p, z)
    _pair_add(stats, S_MY, S_MV, S_SYY, S_SYV, n_new, y, v)
    stats[S_N] = n_new


@njit
def remove(stats, v, p, z, y):
    n_new = stats[S_N] - 1.0
    if n_new <= 0.5:
        for k in range(STATS_SIZE):
            stats[k] = 0.0
        return
    _pair_remove(stats, S_MP, S_MZ, S_SPP, S_SPZ, n_new, p, z)
    _pair_remove(stats, S_MY, S_MV, S_SYY, S_SYV, n_new, y, v)
    stats[S_N] = n_new


@njit
def rebuild(stats, hist, nrows):
    """Exact two-pass recomputation; limits floating-point drift. / 精确重建。"""
    for k in range(STATS_SIZE):
        stats[k] = 0.0
    if nrows == 0:
        return
    mp = 0.0
    mz = 0.0
    my = 0.0
    mv = 0.0
    for r in range(nrows):
        mp += hist[r, H_P]
        mz += hist[r, H_Z]
        my += hist[r, H_Y]
        mv += hist[r, H_V]
    mp /= nrows
    mz /= nrows
    my /= nrows
    mv /= nrows
    spp = 0.0
    spz = 0.0
    syy = 0.0
    syv = 0.0
    for r in range(nrows):
        dp = hist[r, H_P] - mp
        dy = hist[r, H_Y] - my
        spp += dp * dp
        spz += dp * (hist[r, H_Z] - mz)
        syy += dy * dy
        syv += dy * (hist[r, H_V] - mv)
    stats[S_N] = nrows
    stats[S_MP] = mp
    stats[S_MZ] = mz
    stats[S_SPP] = spp
    stats[S_SPZ] = spz
    stats[S_MY] = my
    stats[S_MV] = mv
    stats[S_SYY] = syy
    stats[S_SYV] = syv


@njit
def coefficients(stats, pricing_error_weight):
    """(xi_0, xi_1, gamma_0, gamma_1, lambda_hat) from the current stats."""
    if stats[S_N] < 2.0 or stats[S_SPP] <= 0.0 or stats[S_SYY] <= 0.0:
        raise ValueError("OLS is not identified: fewer than two rows or no variation")
    raw_slope = stats[S_SPZ] / stats[S_SPP]        # z = intercept + raw_slope * p
    xi_1 = -raw_slope                              # paper writes z = xi_0 - xi_1 p
    xi_0 = stats[S_MZ] - raw_slope * stats[S_MP]
    gamma_1 = stats[S_SYV] / stats[S_SYY]
    gamma_0 = stats[S_MV] - gamma_1 * stats[S_MY]
    lam = (pricing_error_weight * gamma_1 + xi_1) / (pricing_error_weight + xi_1 * xi_1)
    return xi_0, xi_1, gamma_0, gamma_1, lam


@njit
def quote(stats, total_order_flow, pricing_error_weight):
    """(4.2): continuous p_hat_t(y_t) from PAST data only. Returns (p, lambda_hat)."""
    xi_0, xi_1, gamma_0, gamma_1, lam = coefficients(stats, pricing_error_weight)
    return gamma_0 + lam * total_order_flow, lam


@njit
def observe(hist, stats, cursor, window, v, p, z, y):
    """Append the COMPLETED row (v_t, p_t, z_t, y_t) -> D_{t+1}. Evicts the oldest when full."""
    head = cursor[C_HEAD]
    nrows = cursor[C_NROWS]
    if nrows == window:
        remove(stats, hist[head, H_V], hist[head, H_P], hist[head, H_Z], hist[head, H_Y])
    hist[head, H_V] = v
    hist[head, H_P] = p
    hist[head, H_Z] = z
    hist[head, H_Y] = y
    cursor[C_HEAD] = (head + 1) % window
    if nrows < window:
        cursor[C_NROWS] = nrows + 1
    add(stats, v, p, z, y)
    cursor[C_APPENDS] += 1
    cursor[C_SINCE_RESYNC] += 1
    if cursor[C_SINCE_RESYNC] >= window:
        rebuild(stats, hist, cursor[C_NROWS])
        cursor[C_SINCE_RESYNC] = 0


@njit
def preload(hist, stats, cursor, window, rows):
    """Feed D_0 rows through the same observe() path the live market uses."""
    for r in range(rows.shape[0]):
        observe(hist, stats, cursor, window, rows[r, 0], rows[r, 1], rows[r, 2], rows[r, 3])


def snapshot(hist: np.ndarray, cursor: np.ndarray) -> np.ndarray:
    """Rows oldest -> newest (Python helper for tests/metrics). / 从旧到新的快照。"""
    nrows = int(cursor[C_NROWS])
    head = int(cursor[C_HEAD])
    if nrows < hist.shape[0]:
        return hist[:nrows].copy()
    return np.concatenate([hist[head:], hist[:head]], axis=0)


def batch_coefficients(rows: np.ndarray, pricing_error_weight: float):
    """Independent NumPy OLS oracle on a (n, 4) array of (v, p, z, y)."""
    v, p, z, y = rows[:, H_V], rows[:, H_P], rows[:, H_Z], rows[:, H_Y]
    z_fit = np.linalg.lstsq(np.column_stack([np.ones_like(p), p]), z, rcond=None)[0]
    v_fit = np.linalg.lstsq(np.column_stack([np.ones_like(y), y]), v, rcond=None)[0]
    xi_0, xi_1 = float(z_fit[0]), float(-z_fit[1])
    gamma_0, gamma_1 = float(v_fit[0]), float(v_fit[1])
    lam = (pricing_error_weight * gamma_1 + xi_1) / (pricing_error_weight + xi_1 * xi_1)
    return xi_0, xi_1, gamma_0, gamma_1, lam
