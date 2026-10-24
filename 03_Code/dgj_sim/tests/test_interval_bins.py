"""per_value_interval: equal-width interval bins over Dou's per-value price interval (Atanasova 2026)."""

import unittest

import numpy as np

from dgj.config import ExperimentCell, PaperParameters
from dgj.game.session import build_grids
from dgj.players.speculator.state_space import price_to_index


class TestIntervalBins(unittest.TestCase):
    def setUp(self):
        self.pv = build_grids(ExperimentCell(price_grid="per_value"))
        self.iv = build_grids(ExperimentCell(price_grid="per_value_interval"))

    def test_per_value_unchanged(self):
        # the existing grid is still linspace(lo, hi, 31) with the same end points
        P = self.pv.price_grid
        self.assertTrue(np.allclose(np.diff(P, axis=1), (P[:, -1:] - P[:, :1]) / 30))

    def test_centres_of_31_equal_bins(self):
        lo, hi = self.pv.price_grid[:, 0], self.pv.price_grid[:, -1]
        w = (hi - lo) / 31
        expected = lo[:, None] + (np.arange(31) + 0.5) * w[:, None]
        self.assertTrue(np.allclose(self.iv.price_grid, expected))

    def test_nearest_centre_equals_interval_bin(self):
        rng = np.random.default_rng(0)
        for k in range(10):
            lo, hi = self.pv.price_grid[k, 0], self.pv.price_grid[k, -1]
            w = (hi - lo) / 31
            prices = rng.uniform(lo - 3 * w, hi + 3 * w, 20000)
            got = np.array([price_to_index(float(p), self.iv.price_grid[k]) for p in prices])
            want = np.clip(np.floor((prices - lo) / w), 0, 30).astype(int)
            self.assertTrue(np.array_equal(got, want))

    def test_resolution_matches_atanasova(self):
        # one action step moves the price ~1.1 bin widths (Atanasova 2026, p. 14 / Table A.8)
        g = self.iv
        lam = g.nash.price_impact
        step = np.diff(g.multipliers)[0]
        bins = [(g.price_grid[k, 1] - g.price_grid[k, 0]) for k in range(10)]
        ratio = max(lam * step * abs(v - 1.0) / b for v, b in zip(g.value_grid, bins))
        self.assertAlmostEqual(ratio, 1.10, delta=0.02)

    def test_other_grids_identical(self):
        self.assertTrue(np.array_equal(self.pv.orders, self.iv.orders))
        self.assertTrue(np.array_equal(self.pv.initial_q_block, self.iv.initial_q_block))


if __name__ == "__main__":
    unittest.main()
