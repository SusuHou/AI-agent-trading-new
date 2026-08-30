"""Noise trader: u_t ~ N(0, sigma_u^2), independent of everything. / 噪声交易者。

The actual draws come from a dedicated stream in game/shocks.py so that
sessions are reproducible; this module only owns the distribution.
"""

import numpy as np


def draw(generator: np.random.Generator, noise_std: float, size: int) -> np.ndarray:
    """One chunk of u_t draws. / 抽取一段 u_t。"""
    if noise_std <= 0:
        raise ValueError("sigma_u must be positive")
    return generator.normal(0.0, noise_std, size)
