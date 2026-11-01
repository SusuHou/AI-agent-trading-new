"""Equilibrium test: exact best response of one trader to the other's frozen greedy policy. / 最优反应检验。

Claim C2 (equilibrium selection vs incomplete learning), S8 item 5. Calvano et al. (2020)-style test on a
converged session:

    * the rival plays its greedy policy (lowest-index tie rule, as in measurement);
    * the market maker's pricing rule is FROZEN at its converged coefficients, p = gamma_0 + lambda_hat * y
      (ASSUMPTION: in the simulator the rule keeps re-estimating on a 10,000-period window; a deviation would
      move it slowly; the test ignores that feedback);
    * the trader then faces a single-agent MDP on the simulator's own state space s = (P-bin of p_{t-1},
      v_{t-1}, v_t) and action grid. With u ~ N(0, sigma_u^2), p is Gaussian given (s, a): mean
      mu = gamma_0 + lambda (x_a + x_rival), sd lambda * sigma_u; the next P-bin follows exactly from the
      nearest-point partition of the price grid of v_t (Phi at the cell midpoints, outer bins open-ended, as
      price_to_index clips); v_{t+1} is uniform on the value grid. Expected profit (v - mu) x_a is exact.
    * value iteration gives V*(s), Q*(s, a); policy evaluation gives the learned policy's own value V_L(s).

Reported per trader: share of states (all, and visited on the measured path) where the learned action is
optimal (Q*(s, a_L) >= V*(s) - tol |V*(s)|), whether it is optimal in every visited state ("Nash on path")
and in every state ("subgame perfect"), and the profit gain of the optimal deviation on the path:
sum_s w(s) (V*(s) - V_L(s)) / sum_s w(s) V_L(s), w = visit frequency on the measured path.
"""
from math import erf, sqrt

import numpy as np

from dgj.game import protocol
from dgj.players.speculator.state_space import price_to_index

TOL = 1e-6          # relative tolerance for "learned action is optimal"


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def greedy_actions(policy_row: np.ndarray) -> np.ndarray:
    """Lowest set bit of each mask (measurement tie rule)."""
    m = policy_row.astype(np.int64)
    return np.array([(int(x) & -int(x)).bit_length() - 1 for x in m], dtype=np.int64)


def transition_model(orders, value_grid, price_grid, rival_actions, gamma_0, lam, noise_std):
    """r[s, a] and P[s, a, k] (next P-bin) for every state s = (p, v_lag, v_cur) and own action a."""
    n_v, n_x = orders.shape
    n_p = price_grid.shape[1]
    S = n_p * n_v * n_v
    v_cur = np.arange(S) % n_v
    x_own = orders[v_cur, :]                                   # (S, A)
    x_riv = orders[v_cur, rival_actions][:, None]              # (S, 1)
    mu = gamma_0 + lam * (x_own + x_riv)                       # (S, A)
    v = value_grid[v_cur][:, None]
    reward = (v - mu) * x_own
    sd = abs(lam) * noise_std
    mids = (price_grid[:, 1:] + price_grid[:, :-1]) / 2.0     # (n_v, n_p - 1): nearest-point cell edges
    edges = mids[v_cur]                                        # (S, n_p - 1)
    cdf = _norm_cdf((edges[:, None, :] - mu[:, :, None]) / sd)  # (S, A, n_p - 1)
    zeros = np.zeros(cdf.shape[:2] + (1,))
    ones = np.ones(cdf.shape[:2] + (1,))
    P = np.diff(np.concatenate([zeros, cdf, ones], axis=2), axis=2)   # (S, A, n_p)
    return reward, P, v_cur


def _continuation(V, n_p, n_v, v_cur):
    C = V.reshape(n_p, n_v, n_v).mean(axis=2)                 # C[k, v_t] = mean over v_{t+1}
    return C[:, v_cur].T                                       # (S, n_p): C[k, v_cur(s)]


def solve(reward, P, v_cur, n_p, n_v, rho, own_actions=None, tol=1e-10, max_iter=20_000):
    """Value iteration (own_actions None) or policy evaluation (own_actions given)."""
    S, A = reward.shape
    V = np.zeros(S)
    scale = max(1.0, float(np.abs(reward).max()) / (1 - rho))
    for _ in range(max_iter):
        Q = reward + rho * np.einsum("sak,sk->sa", P, _continuation(V, n_p, n_v, v_cur))
        V_new = Q.max(axis=1) if own_actions is None else Q[np.arange(S), own_actions]
        if np.max(np.abs(V_new - V)) <= tol * scale:
            return V_new, Q
        V = V_new
    raise RuntimeError("value iteration did not converge")


def path_state_weights(rows, value_grid, price_grid, n_p, value_mean, use_lagged_price=True):
    """Visit counts of states on the measured path (rows from Session.measure)."""
    n_v = len(value_grid)
    vi = np.abs(value_grid[None, :] - rows[:, protocol.COL_V][:, None]).argmin(axis=1)
    p_idx = np.array([price_to_index(float(p), price_grid[k]) for p, k in zip(rows[:-1, protocol.COL_P], vi[:-1])])
    if not use_lagged_price:
        p_idx[:] = 0
    s = (p_idx * n_v + vi[:-1]) * n_v + vi[1:]                  # state of period t+1
    return np.bincount(s, minlength=n_p * n_v * n_v).astype(float)


def best_response_test(grids, policy, i, gamma_0, lam, noise_std, rho, weights):
    n_v, n_x = grids.orders.shape
    n_p = grids.price_grid.shape[1]
    rival = greedy_actions(policy[1 - i])
    own = greedy_actions(policy[i])
    reward, P, v_cur = transition_model(grids.orders, grids.value_grid, grids.price_grid, rival, gamma_0, lam, noise_std)
    V_star, Q_star = solve(reward, P, v_cur, n_p, n_v, rho)
    V_L, _ = solve(reward, P, v_cur, n_p, n_v, rho, own_actions=own)
    S = len(V_star)
    ok = Q_star[np.arange(S), own] >= V_star - TOL * np.maximum(1.0, np.abs(V_star))
    visited = weights > 0
    w = weights / weights.sum()
    gain = float((w * (V_star - V_L)).sum() / (w * V_L).sum())
    best = Q_star.argmax(axis=1)
    return {
        "share_optimal_all_states": float(ok.mean()),
        "share_optimal_visited_states": float(ok[visited].mean()),
        "share_optimal_visit_weighted": float((w * ok).sum()),
        "nash_on_path": bool(ok[visited].all()),
        "subgame_perfect": bool(ok.all()),
        "deviation_gain_on_path": gain,
        "visited_states": int(visited.sum()),
        "mean_abs_action_gap_visited": float(np.abs(best - own)[visited].mean()),
    }
