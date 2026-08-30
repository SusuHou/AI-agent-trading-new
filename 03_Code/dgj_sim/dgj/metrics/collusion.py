"""Delta^C — matched-path collusion capacity, OA IA.4.1–IA.4.3. / 合谋利润指标。

    Delta_i = (mean pi_i - mean pi^N) / (mean pi^M - mean pi^N)
    Delta^C = mean_i Delta_i

pi^N_t and pi^M_t are recomputed on the SAME realized (v_t, u_t) as the AI session.
"""

from dataclasses import dataclass

import numpy as np

from dgj.game import protocol
from dgj.game.session import Grids
from dgj.players import benchmarks


@dataclass(frozen=True)
class CollusionResult:
    mean_actual_profit: tuple[float, ...]
    mean_nash_profit: float
    mean_cartel_profit: float
    delta_by_agent: tuple[float, ...]
    delta_c: float
    profit_gain_vs_nash: float      # sum_i pi_i / sum_i pi^N  (OA "relative profit gain")


def compute(rows: np.ndarray, grids: Grids, number_of_speculators: int, value_mean: float) -> CollusionResult:
    v = rows[:, protocol.COL_V]
    u = rows[:, protocol.COL_U]
    I = number_of_speculators
    actual = tuple(float(rows[:, protocol.col_pi(i, I)].mean()) for i in range(I))
    nash = float(benchmarks.matched_path_profit(grids.nash, v, u, I, value_mean).mean())
    cartel = float(benchmarks.matched_path_profit(grids.cartel, v, u, I, value_mean).mean())
    gap = cartel - nash
    if gap <= 0:
        raise ArithmeticError("cartel-minus-Nash gap is not positive; Delta^C undefined")
    deltas = tuple((a - nash) / gap for a in actual)
    return CollusionResult(
        mean_actual_profit=actual,
        mean_nash_profit=nash,
        mean_cartel_profit=cartel,
        delta_by_agent=deltas,
        delta_c=float(np.mean(deltas)),
        profit_gain_vs_nash=float(sum(actual) / (I * nash)),
    )
