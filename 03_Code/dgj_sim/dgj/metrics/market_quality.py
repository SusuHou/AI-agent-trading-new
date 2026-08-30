"""Market liquidity (OA IA.4.6) and mispricing (OA IA.4.7) from saved rows. / 流动性与错误定价。

    L_t = 1 / |1 - xi * lambda_hat_t|                       averaged over the window
    E_t = (1 - lambda_hat_t * I * chi_hat^C) * |v_t - v_bar|  averaged over the window

The OA prints sums without 1/T but calls them averages; we report the mean.
The printed E_t has no absolute value around the loading; we report both and
flag periods where the loading is negative (then the two readings differ).
"""

from dataclasses import dataclass

import numpy as np

from dgj.game import protocol


@dataclass(frozen=True)
class LiquidityResult:
    mean_liquidity: float
    min_liquidity: float
    max_liquidity: float
    singular_periods: int          # xi * lambda_hat == 1 exactly


@dataclass(frozen=True)
class MispricingResult:
    mean_mispricing_paper: float        # signed loading, as printed
    mean_mispricing_absolute: float     # |loading|, Definition 3.4
    negative_loading_periods: int
    min_loading: float


def liquidity(rows: np.ndarray, investor_slope: float) -> LiquidityResult:
    lam = rows[:, protocol.COL_LAM]
    sensitivity = np.abs(1.0 - investor_slope * lam)
    singular = int((sensitivity == 0.0).sum())
    with np.errstate(divide="ignore"):
        L = np.where(sensitivity > 0, 1.0 / np.where(sensitivity > 0, sensitivity, 1.0), np.inf)
    return LiquidityResult(float(L.mean()), float(L.min()), float(L.max()), singular)


def mispricing(rows: np.ndarray, number_of_speculators: int, average_intensity: float, value_mean: float) -> MispricingResult:
    lam = rows[:, protocol.COL_LAM]
    deviation = np.abs(rows[:, protocol.COL_V] - value_mean)
    loading = 1.0 - lam * number_of_speculators * average_intensity
    return MispricingResult(
        mean_mispricing_paper=float((loading * deviation).mean()),
        mean_mispricing_absolute=float((np.abs(loading) * deviation).mean()),
        negative_loading_periods=int((loading < 0).sum()),
        min_loading=float(loading.min()),
    )
