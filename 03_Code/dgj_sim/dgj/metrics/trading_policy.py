"""chi_hat^C (OA IA.4.4) and price informativeness I^C (OA IA.4.5). / 交易强度与价格信息效率。

    x_{i,t} = chi_{i,0} + chi_{i,1} v_t + e     (unrestricted, per agent)
    chi_hat^C = mean_i chi_{i,1}
    I^C = (I chi_hat^C)^2 (sigma_v_hat / sigma_u)^2
"""

from dataclasses import dataclass

import numpy as np

from dgj.game import protocol


@dataclass(frozen=True)
class TradingPolicyResult:
    intercept_by_agent: tuple[float, ...]
    slope_by_agent: tuple[float, ...]
    average_intensity: float
    price_informativeness: float


def compute(rows: np.ndarray, number_of_speculators: int, discrete_value_std: float, noise_std: float) -> TradingPolicyResult:
    v = rows[:, protocol.COL_V]
    if v.max() - v.min() <= 0:
        raise ArithmeticError("v_t does not vary; slope undefined")
    design = np.column_stack([np.ones_like(v), v])
    intercepts, slopes = [], []
    for i in range(number_of_speculators):
        coef = np.linalg.lstsq(design, rows[:, protocol.col_x(i)], rcond=None)[0]
        intercepts.append(float(coef[0]))
        slopes.append(float(coef[1]))
    chi = float(np.mean(slopes))
    informativeness = (number_of_speculators * chi) ** 2 * (discrete_value_std / noise_std) ** 2
    return TradingPolicyResult(tuple(intercepts), tuple(slopes), chi, float(informativeness))
