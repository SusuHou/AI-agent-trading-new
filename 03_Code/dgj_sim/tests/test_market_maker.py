import unittest

import numpy as np

import _setup  # noqa: F401
from dgj.config import ExperimentCell
from dgj.game.session import build_grids
from dgj.players.market_maker import adaptive, prehistory
from dgj.players.market_maker.adaptive import CURSOR_SIZE, H_P, H_Y


class TestAdaptiveMarketMaker(unittest.TestCase):
    def _preloaded(self, rows, window):
        hist, stats = adaptive.new_history(window)
        cursor = np.zeros(CURSOR_SIZE, dtype=np.int64)
        adaptive.preload(hist, stats, cursor, window, np.ascontiguousarray(rows, dtype=float))
        return hist, stats, cursor

    def test_hand_example_from_paper_equations(self):
        rows = np.array([[0.80, 0.98, 10.0, -2.0], [0.90, 1.00, 0.0, 0.0],
                         [1.00, 1.02, -10.0, 2.0], [1.10, 1.04, -20.0, 4.0]])
        hist, stats, cursor = self._preloaded(rows, 4)
        xi0, xi1, g0, g1, lam = adaptive.coefficients(stats, 0.1)
        self.assertAlmostEqual(xi0, 500.0, places=9)
        self.assertAlmostEqual(xi1, 500.0, places=9)
        self.assertAlmostEqual(g0, 0.90, places=12)
        self.assertAlmostEqual(g1, 0.05, places=12)
        self.assertAlmostEqual(lam, 0.00200001919999232, places=15)
        p, lam_q = adaptive.quote(stats, 10.0, 0.1)
        self.assertAlmostEqual(p, 0.9200001919999232, places=15)
        self.assertEqual(lam_q, lam)

    def test_rolling_add_remove_matches_batch_ols(self):
        rng = np.random.default_rng(1)
        window = 17
        hist, stats = adaptive.new_history(window)
        cursor = np.zeros(CURSOR_SIZE, dtype=np.int64)
        for t in range(400):
            p = 0.9 + 0.2 * rng.random()
            y = rng.normal(0, 3)
            v = 0.95 + 0.04 * y + 0.01 * rng.normal()
            z = 500 - 500 * p + 0.03 * rng.normal()
            adaptive.observe(hist, stats, cursor, window, v, p, z, y)
            if cursor[adaptive.C_NROWS] >= 2:
                snap = adaptive.snapshot(hist, cursor)
                fast = adaptive.coefficients(stats, 0.1)
                slow = adaptive.batch_coefficients(snap, 0.1)
                np.testing.assert_allclose(fast, slow, rtol=1e-9, atol=1e-9)
        self.assertEqual(int(cursor[adaptive.C_NROWS]), window)
        self.assertEqual(int(cursor[adaptive.C_APPENDS]), 400)

    def test_prehistory_recovers_benchmark_coefficients(self):
        cell = ExperimentCell()
        g = build_grids(cell)
        p = cell.parameters
        rows = prehistory.build_rows(g.nash, g.value_grid, p.value_mean, p.investor_slope,
                                     p.noise_std, p.num_speculators, p.market_maker_window)
        self.assertEqual(rows.shape, (p.market_maker_window, 4))
        u = rows[:, H_Y] - p.num_speculators * g.nash.intensity * (rows[:, 0] - p.value_mean)
        self.assertAlmostEqual(float(u.mean()), 0.0, places=10)
        self.assertAlmostEqual(float(np.sqrt(np.mean(u ** 2))), p.noise_std, places=10)
        hist, stats, cursor = self._preloaded(rows, p.market_maker_window)
        xi0, xi1, g0, g1, lam = adaptive.coefficients(stats, p.pricing_error_weight)
        self.assertAlmostEqual(xi1, p.investor_slope, places=8)
        self.assertAlmostEqual(g0, p.value_mean, places=8)
        self.assertAlmostEqual(g1, g.nash.gamma, places=8)
        self.assertAlmostEqual(lam, g.nash.price_impact, places=10)
        self.assertAlmostEqual(float(adaptive.snapshot(hist, cursor)[0, H_P]), rows[0, H_P])


if __name__ == "__main__":
    unittest.main()
