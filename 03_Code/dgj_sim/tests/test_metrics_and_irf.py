import unittest

import numpy as np

import _setup  # noqa: F401
from dgj.config import ExperimentCell, PaperParameters
from dgj.game import irf, protocol
from dgj.game.session import Session
from dgj.metrics import market_quality
from dgj.players.market_maker.adaptive import C_APPENDS, H_P
from dgj.players.speculator import policy


def _rows(lam, v, I=2, p=1.0):
    rows = np.zeros((len(lam), protocol.row_width(I)))
    rows[:, protocol.COL_LAM] = lam
    rows[:, protocol.COL_V] = v
    rows[:, protocol.COL_P] = p
    return rows


class TestMarketQuality(unittest.TestCase):
    def test_liquidity_hand_values(self):          # step-32 demo: xi=2, lambda in {0, .25, -.25}
        r = market_quality.liquidity(_rows([0.0, 0.25, -0.25], [1.0, 1.0, 1.0]), 2.0)
        self.assertAlmostEqual(r.mean_liquidity, (1 + 2 + 2 / 3) / 3, places=12)
        self.assertEqual(r.singular_periods, 0)

    def test_mispricing_hand_value(self):          # step-33 demo: I=2, chi=.25, lambda=.5, v=3 -> 1.5
        r = market_quality.mispricing(_rows([0.5], [3.0]), 2, 0.25, 1.0)
        self.assertAlmostEqual(r.mean_mispricing_paper, 1.5, places=12)
        self.assertAlmostEqual(r.mean_mispricing_absolute, 1.5, places=12)
        self.assertEqual(r.negative_loading_periods, 0)


class TestIRF(unittest.TestCase):
    def setUp(self):
        cell = ExperimentCell(parameters=PaperParameters(market_maker_window=200, convergence_periods=3,
                                                         measurement_periods=200), label="irf_test")
        self.s = Session(cell, 0, 11)
        self.s.state.Q[:, :, 5] = 1e9                     # stable greedy policy -> converges in 3 periods
        self.s.state.policy[...] = policy.initial_policy(self.s.state.Q)
        self.assertTrue(self.s.train(chunk_size=10))
        self.fork = irf.take_fork(self.s)
        self.rows = self.s.measure()
        self.base = irf.long_run_baseline(self.rows, 2, 1.0)

    def test_shock_calibration_and_paired_paths(self):
        p = self.s.p
        shock = irf.calibrate_shock(self.base)
        self.assertAlmostEqual(shock * self.base.mean_lambda, 0.012 * self.base.mean_oriented_price, places=12)
        state_before = (self.s.state.cursor.copy(), self.s.state.hist.copy(), self.s.state.stats.copy())
        res = irf.run_irf(self.s, self.fork, self.base, paths=50, shock_magnitude=shock)
        # running the IRF must not touch the live session
        self.assertTrue(np.array_equal(self.s.state.cursor, state_before[0]))
        self.assertTrue(np.array_equal(self.s.state.hist, state_before[1]))
        self.assertTrue(np.array_equal(self.s.state.stats, state_before[2]))
        # before the shock, control and treatment coincide; at t=3 the price moves by lambda*shock (oriented)
        np.testing.assert_allclose(res.control_oriented_price[:2], res.treatment_oriented_price[:2], rtol=0, atol=1e-12)
        lam = self.rows[:, protocol.COL_LAM].mean()
        self.assertAlmostEqual(res.treatment_oriented_price[2] - res.control_oriented_price[2], lam * shock, delta=1e-6)
        self.assertEqual(res.paths, 50)
        # frozen single action: the t=4 response vs the common-random-number control is exactly zero
        np.testing.assert_allclose(res.response_vs_control, 0.0, atol=1e-12)
        self.assertEqual(res.mechanism, "over_pruning")
        # the long-run-normalized version carries |v - v_bar| sampling noise (why we do not classify on it)
        self.assertGreater(max(abs(r) for r in res.response_vs_long_run), 5e-5)

    def test_classifier_thresholds(self):
        self.assertEqual(irf.classify((1e-3, 2e-3)), "price_trigger")
        self.assertEqual(irf.classify((1e-6, -1e-6)), "over_pruning")
        self.assertEqual(irf.classify((1e-3, 1e-6)), "unclassified")
        self.assertEqual(irf.classify((5e-4, 5e-4)), "unclassified")     # strict inequalities

    def test_branch_uses_own_history(self):
        # the treatment's t=4 quote must come from a history containing the shocked t=3 row
        u = np.zeros(4); vidx = np.array([1, 8, 8, 8])
        control = irf._run_branch(self.s, self.fork, vidx, u)
        u2 = u.copy(); u2[2] += 5.0
        treatment = irf._run_branch(self.s, self.fork, vidx, u2)
        self.assertNotEqual(control[3, protocol.COL_LAM], treatment[3, protocol.COL_LAM])
        self.assertEqual(control[0, protocol.COL_P], treatment[0, protocol.COL_P])


if __name__ == "__main__":
    unittest.main()
