"""Information-insensitive investors, equation (3.2): z_t = -xi (p_t - v_bar).

信息不敏感投资者：看到价格后才下单，因此 z_t 只能在 p_t 出现之后计算。
"""

from dgj._jit import njit


@njit
def demand(price: float, investor_slope: float, value_mean: float) -> float:
    return -investor_slope * (price - value_mean)
