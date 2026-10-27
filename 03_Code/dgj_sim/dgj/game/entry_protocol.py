"""Training kernel with staggered entry. / 带进入时机的训练内核。

Same period order as ``protocol.run_periods`` (paper p.23), with three changes for the
entry-timing experiments (EXPERIMENT_PLAN_PAPER.md, items 1-3):

    * agent i is ACTIVE from period ``entry_period[i]`` on; before that it submits a zero
      order, earns nothing, and its Q-table, visit counts and policy are untouched
      (a fresh entrant arrives with Dou et al.'s initial Q);
    * exploration counts are PER AGENT, visits[i, v], so an entrant's epsilon restarts
      at 1 (with both agents active from period 0 the two rows are identical, and the
      kernel reproduces ``protocol.run_periods`` exactly);
    * eps = max(exp(-beta t_i(v)), exploration_floor).

Convergence streak: reset when an ACTIVE agent's greedy policy changes. The kernel stops
when the streak reaches ``streak_target`` and either every agent is active or
``require_all_active`` is False (used for the lone-incumbent phase of stage 3).
``use_lagged_price=False`` fixes the price index at 0 (diagnostic T2, as Dou's negative control).
``explore_turns=True``: only trader (t mod I) may explore in period t, with probability min(1, I*eps) (test M3).
"""

import numpy as np

from dgj._jit import njit
from dgj.environment.insensitive_investors import demand
from dgj.players.market_maker import adaptive
from dgj.players.market_maker.adaptive import C_P_IDX, C_V_LAG, C_V_CUR, C_STREAK, C_T
from dgj.players.speculator import policy as policy_module
from dgj.players.speculator import q_learning
from dgj.players.speculator.state_space import encode, next_block, price_to_index


@njit
def run_periods_entry(
    Q, visits, policy, cursor, hist, stats, orders, value_grid, price_grid,
    value_draws, noise_draws, mode_draws, action_draws,
    learning_rate, discount_factor, exploration_decay, exploration_floor,
    pricing_error_weight, investor_slope, value_mean, window,
    n_periods, streak_target, entry_period, require_all_active, use_lagged_price, explore_turns,
):
    """Run up to n_periods of training; returns (periods_run, converged)."""
    I = Q.shape[0]
    n_v = value_grid.shape[0]
    n_x = Q.shape[2]
    acts = np.zeros(I, dtype=np.int64)
    x = np.zeros(I)
    pi = np.zeros(I)
    active = np.zeros(I, dtype=np.bool_)

    for k in range(n_periods):
        t = cursor[C_T]
        all_active = True
        for i in range(I):
            active[i] = t >= entry_period[i]
            if not active[i]:
                all_active = False

        p_idx = cursor[C_P_IDX]
        v_lag = cursor[C_V_LAG]
        v_cur = cursor[C_V_CUR]
        s = encode(p_idx, v_lag, v_cur, n_v)

        # 1. active agents choose; each uses its own past visits of v_t
        for i in range(I):
            if active[i]:
                eps = q_learning.exploration_rate(visits[i, v_cur], exploration_decay)
                if eps < exploration_floor:
                    eps = exploration_floor
                if explore_turns:
                    # Lambin test (M3): only one trader may explore in a period (turns by period
                    # index); its probability is scaled by I so each trader's expected
                    # exploration is unchanged. / 轮流探索：每期只有一人可探索，概率乘 I。
                    if t % I == i:
                        eps = min(1.0, I * eps)
                    else:
                        eps = 0.0
                acts[i] = q_learning.choose_action(policy[i, s], eps, mode_draws[i, k], action_draws[i, k], n_x)
        for i in range(I):
            if active[i]:
                visits[i, v_cur] += 1

        # 2. order flow (an inactive agent submits nothing)
        u = noise_draws[k]
        y = u
        for i in range(I):
            if active[i]:
                x[i] = orders[v_cur, acts[i]]
            else:
                x[i] = 0.0
            y += x[i]

        # 3-5. price, investors, profits, history
        p, lam = adaptive.quote(stats, y, pricing_error_weight)
        z = demand(p, investor_slope, value_mean)
        v = value_grid[v_cur]
        for i in range(I):
            pi[i] = (v - p) * x[i]
        adaptive.observe(hist, stats, cursor, window, v, p, z, y)

        # 6. next state; Q update for active agents only
        v_next = value_draws[k]
        p_next = price_to_index(p, price_grid[v_cur]) if use_lagged_price else 0
        base = next_block(p_next, v_cur, n_v)
        changed = False
        for i in range(I):
            if active[i]:
                q_learning.update(Q, i, s, acts[i], pi[i], base, n_v, learning_rate, discount_factor)
                mask = policy_module.argmax_mask(Q, i, s)
                if mask != policy[i, s]:
                    policy[i, s] = mask
                    changed = True
        if changed:
            cursor[C_STREAK] = 0
        else:
            cursor[C_STREAK] += 1

        cursor[C_P_IDX] = p_next
        cursor[C_V_LAG] = v_cur
        cursor[C_V_CUR] = v_next
        cursor[C_T] += 1

        if cursor[C_STREAK] >= streak_target and (all_active or not require_all_active):
            return k + 1, True
    return n_periods, False


@njit
def play_frozen(
    policy, cursor, hist, stats, orders, value_grid, price_grid,
    value_draws, noise_draws, pricing_error_weight, investor_slope, value_mean, window,
    n_periods, active, use_lagged_price, out_actions, out_values,
):
    """Frozen greedy play (lowest-index tie rule), no learning; agents with active[i] False
    submit nothing. Records each active agent's action index and v_t. Market maker keeps rolling.
    / 冻结策略下的实际交易：用于衡量在位者真正下多大的单。
    """
    I = policy.shape[0]
    n_v = value_grid.shape[0]
    for k in range(n_periods):
        s = encode(cursor[C_P_IDX], cursor[C_V_LAG], cursor[C_V_CUR], n_v)
        v_cur = cursor[C_V_CUR]
        y = noise_draws[k]
        x_own = np.zeros(I)
        for i in range(I):
            if active[i]:
                a = policy_module.lowest_action(policy[i, s])
                out_actions[i, k] = a
                x_own[i] = orders[v_cur, a]
                y += x_own[i]
            else:
                out_actions[i, k] = -1
        out_values[k] = value_grid[v_cur]
        p, lam = adaptive.quote(stats, y, pricing_error_weight)
        z = demand(p, investor_slope, value_mean)
        adaptive.observe(hist, stats, cursor, window, value_grid[v_cur], p, z, y)
        cursor[C_P_IDX] = price_to_index(p, price_grid[v_cur]) if use_lagged_price else 0
        cursor[C_V_LAG] = v_cur
        cursor[C_V_CUR] = value_draws[k]
        cursor[C_T] += 1
