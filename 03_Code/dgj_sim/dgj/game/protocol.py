"""One market period in the paper's order — the compiled kernel. / 一期的因果顺序：内核。

Paper p.23 protocol, per period t, given s_t = (p_{t-1}, v_{t-1}, v_t):

    1. speculators choose x_{i,t}   (eps from PAST visits of v_t; count v_t once afterwards)
    2. noise trader submits u_t;  y_t = sum_i x_{i,t} + u_t
    3. market maker quotes p_t from D_t (PAST rows only) and current y_t
    4. investors submit z_t = -xi (p_t - v_bar);  profits pi_{i,t} = (v_t - p_t) x_{i,t}
    5. the completed row (v_t, p_t, z_t, y_t) joins the history -> D_{t+1}
    6. v_{t+1} is drawn; s_{t+1} formed; each Q_i updated in ONE cell (expected over v')

Measurement mode (after convergence): steps 1 uses the frozen greedy policy,
no exploration, no visit count, no Q update; the market maker keeps rolling.
"""

import numpy as np

from dgj._jit import njit
from dgj.environment.insensitive_investors import demand
from dgj.players.market_maker import adaptive
from dgj.players.market_maker.adaptive import (
    C_P_IDX, C_V_LAG, C_V_CUR, C_STREAK, C_T,
)
from dgj.players.speculator import policy as policy_module
from dgj.players.speculator import q_learning
from dgj.players.speculator.state_space import encode, next_block, price_to_index

# measurement row layout / 测量行的列
COL_V, COL_U, COL_Y, COL_P, COL_LAM, COL_Z = range(6)
FIXED_COLUMNS = 6


def row_width(number_of_speculators: int) -> int:
    return FIXED_COLUMNS + 2 * number_of_speculators


def col_x(i: int) -> int:
    return FIXED_COLUMNS + i


def col_pi(i: int, number_of_speculators: int) -> int:
    return FIXED_COLUMNS + number_of_speculators + i


@njit
def run_periods(
    Q, visits, policy, cursor, hist, stats, orders, value_grid, price_grid,  # price_grid: (n_v, n_p)
    value_draws, noise_draws, mode_draws, action_draws,
    learning_rate, discount_factor, exploration_decay, pricing_error_weight,
    investor_slope, value_mean, window,
    n_periods, learning, streak_target, out_rows,
):
    """Run up to n_periods; returns (periods_run, converged). / 运行若干期。

    learning=True : training (steps 1-6 fully); stops early when the joint
                    greedy policy has been unchanged for streak_target periods.
    learning=False: frozen-policy measurement; writes one row per period into
                    out_rows if it has room.
    """
    I = Q.shape[0]
    n_v = value_grid.shape[0]
    n_x = Q.shape[2]
    record = out_rows.shape[0] >= n_periods and out_rows.shape[0] > 0
    acts = np.empty(I, dtype=np.int64)
    x = np.empty(I)
    pi = np.empty(I)

    for k in range(n_periods):
        p_idx = cursor[C_P_IDX]
        v_lag = cursor[C_V_LAG]
        v_cur = cursor[C_V_CUR]
        s = encode(p_idx, v_lag, v_cur, n_v)

        # 1. choose / 选动作 --------------------------------------------
        if learning:
            eps = q_learning.exploration_rate(visits[v_cur], exploration_decay)
            for i in range(I):
                acts[i] = q_learning.choose_action(
                    policy[i, s], eps, mode_draws[i, k], action_draws[i, k], n_x
                )
            visits[v_cur] += 1                       # once per period, after both chose
        else:
            for i in range(I):
                acts[i] = policy_module.lowest_action(policy[i, s])

        # 2. noise arrives / 噪声到达 -------------------------------------
        u = noise_draws[k]
        y = u
        for i in range(I):
            x[i] = orders[v_cur, acts[i]]
            y += x[i]

        # 3. price from PAST data + current y / 用旧数据定价 ---------------
        p, lam = adaptive.quote(stats, y, pricing_error_weight)

        # 4. investors and profits / 投资者与利润 --------------------------
        z = demand(p, investor_slope, value_mean)
        v = value_grid[v_cur]
        for i in range(I):
            pi[i] = (v - p) * x[i]

        # 5. completed row joins history / 本期记录进入历史 -----------------
        adaptive.observe(hist, stats, cursor, window, v, p, z, y)

        # 6. next state and Q update / 下一状态与 Q 更新 ------------------
        v_next = value_draws[k]
        p_next = price_to_index(p, price_grid[v_cur])     # P(v_t): grid of the value p_t formed under
        if learning:
            base = next_block(p_next, v_cur, n_v)
            changed = False
            for i in range(I):
                q_learning.update(Q, i, s, acts[i], pi[i], base, n_v, learning_rate, discount_factor)
                mask = policy_module.argmax_mask(Q, i, s)
                if mask != policy[i, s]:
                    policy[i, s] = mask
                    changed = True
            if changed:
                cursor[C_STREAK] = 0
            else:
                cursor[C_STREAK] += 1

        if record:
            out_rows[k, COL_V] = v
            out_rows[k, COL_U] = u
            out_rows[k, COL_Y] = y
            out_rows[k, COL_P] = p
            out_rows[k, COL_LAM] = lam
            out_rows[k, COL_Z] = z
            for i in range(I):
                out_rows[k, FIXED_COLUMNS + i] = x[i]
                out_rows[k, FIXED_COLUMNS + I + i] = pi[i]

        cursor[C_P_IDX] = p_next
        cursor[C_V_LAG] = v_cur
        cursor[C_V_CUR] = v_next
        cursor[C_T] += 1

        if learning and cursor[C_STREAK] >= streak_target:
            return k + 1, True
    return n_periods, False
