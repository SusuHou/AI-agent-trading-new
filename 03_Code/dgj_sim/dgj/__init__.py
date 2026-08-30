"""dgj: a compiled-kernel replication of Dou, Goldstein & Ji (2025).

Package layout mirrors the paper's actors / 包结构对应论文中的参与者:

    environment/   exogenous world: v_t grid, noise trader, insensitive investors
    players/       speculator (Q-learning), market_maker (adaptive OLS), benchmarks (Nash/cartel)
    game/          shocks (independent RNG streams), protocol (one period = the kernel), session (phases)
    metrics/       Delta^C, chi_hat^C, price informativeness — computed offline from saved rows
    experiments/   1,000-session fan-out and aggregation
"""

from dgj.config import ExperimentCell, PaperParameters

__all__ = ["ExperimentCell", "PaperParameters"]
