import unittest
from math import sqrt

import numpy as np

import _setup  # noqa: F401
from dgj.environment import fundamental
from dgj.players import benchmarks


class TestBenchmarks(unittest.TestCase):
    def setUp(self):
        self.V = fundamental.value_grid(1.0, 1.0, 10)
        self.s = fundamental.discrete_std(self.V, 1.0)

    def _check(self, name, noise, xi=500.0, theta=0.1, I=2):
        b = benchmarks.solve(name, I, noise, self.s, xi, theta)
        self.assertLess(abs(b.residual), 1e-13)
        identity = (I + 1) * b.price_impact * b.intensity if name == "nash" else 2 * I * b.price_impact * b.intensity
        self.assertAlmostEqual(identity, 1.0, places=12)
        return b

    def test_fixed_points_both_noise_levels(self):
        for noise in (0.1, 100.0):
            n = self._check("nash", noise)
            m = self._check("cartel", noise)
            self.assertGreater(n.intensity, m.intensity)
            self.assertAlmostEqual(n.price_impact, 0.002, places=6)   # xi/(xi^2+theta) dominates

    def test_zero_xi_closed_form(self):
        b = self._check("nash", 0.1, xi=0.0)
        a = 2 / 3
        self.assertAlmostEqual(b.price_impact, sqrt(a * (1 - a)) / (0.1 / self.s), places=12)
        self.assertAlmostEqual(b.price_impact, b.gamma, places=12)

    def test_expected_profit_equals_matched_path_average(self):
        for noise in (0.1, 100.0):
            for name in ("nash", "cartel"):
                b = self._check(name, noise)
                v = np.repeat(self.V, 2)
                u = np.tile([-noise, noise], 10)
                direct = benchmarks.matched_path_profit(b, v, u, 2, 1.0).mean()
                closed = benchmarks.expected_profit(b, 2, self.s)
                self.assertAlmostEqual(direct, closed, places=12)
        self.assertGreater(
            benchmarks.expected_profit(self._check("cartel", 0.1), 2, self.s),
            benchmarks.expected_profit(self._check("nash", 0.1), 2, self.s),
        )


if __name__ == "__main__":
    unittest.main()
