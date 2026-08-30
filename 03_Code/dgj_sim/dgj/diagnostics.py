"""Diagnostics that explain results rather than produce them. / 解释性诊断。

detectability(): how often a ONE-STEP deviation by one speculator moves the
price-grid index of p_{t-1} that the other speculator will see next period.
This is the quantity footnote 25 is about; it must be near 1 for price-trigger
strategies to be learnable at all.
"""

from dataclasses import dataclass

import numpy as np

from dgj.game.session import Grids
from dgj.players.speculator.state_space import price_to_index


@dataclass(frozen=True)
class Detectability:
    overall: float                  # P(index changes | opponent moves one action step)
    by_value: tuple[float, ...]     # same, conditional on v
    price_step: tuple[float, ...]   # grid spacing per v
    one_action_move: tuple[float, ...]  # |lambda * dx| per v


def detectability(grids: Grids, value_mean: float, noise_std: float, price_impact: float | None = None,
                  draws: int = 20_000, seed: int = 0) -> Detectability:
    """Monte Carlo over (v, own action, opponent action, u): does a one-step
    opponent deviation change price_to_index(p, P(v))?  Uses the Nash pricing
    rule p = v_bar + lambda y (what the market maker recovers from D_0)."""
    rng = np.random.default_rng(seed)
    lam = grids.nash.price_impact if price_impact is None else price_impact
    n_v, n_x = grids.orders.shape
    by_value, steps, moves = [], [], []
    hits_total = 0
    for k in range(n_v):
        own = rng.integers(n_x, size=draws)
        opp = rng.integers(n_x - 1, size=draws)          # deviation opp -> opp+1 always inside the grid
        u = rng.normal(0.0, noise_std, size=draws)
        hits = 0
        for j in range(draws):
            y0 = grids.orders[k, own[j]] + grids.orders[k, opp[j]] + u[j]
            y1 = grids.orders[k, own[j]] + grids.orders[k, opp[j] + 1] + u[j]
            p0 = value_mean + lam * y0
            p1 = value_mean + lam * y1
            hits += price_to_index(p0, grids.price_grid[k]) != price_to_index(p1, grids.price_grid[k])
        by_value.append(hits / draws)
        hits_total += hits
        steps.append(float(grids.price_grid[k, 1] - grids.price_grid[k, 0]))
        moves.append(float(abs(lam * (grids.orders[k, 1] - grids.orders[k, 0]))))
    return Detectability(hits_total / (draws * n_v), tuple(by_value), tuple(steps), tuple(moves))
