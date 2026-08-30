"""The four rules, tested directly on the kernel and the session. / 四条规则的直接测试。

1. within-period order: choose -> noise -> price from OLD data -> profits -> append -> Q update
2. the market maker never prices period t with period-t data
3. after convergence: Q, exploration, visit counts frozen; market maker keeps rolling
4. every session's randomness is independent and reproducible
"""

import os
import tempfile
import unittest

import numpy as np

import _setup  # noqa: F401
from dgj.config import ExperimentCell, PaperParameters
from dgj.game import protocol
from dgj.game.session import Session
from dgj.game.shocks import ShockStreams, Shocks
from dgj.players.market_maker import adaptive
from dgj.players.market_maker.adaptive import (
    C_APPENDS, C_P_IDX, C_T, C_V_CUR, C_V_LAG, H_P, H_V, H_Y, H_Z,
)
from dgj.players.speculator import policy, state_space

SMALL = PaperParameters(market_maker_window=200, convergence_periods=3, measurement_periods=20)
CELL = ExperimentCell(parameters=SMALL, label="test_small")
LONG = ExperimentCell(parameters=PaperParameters(market_maker_window=200), label="test_long")  # never converges in a short test


def one_shock(v_next: int, u: float, I: int = 2) -> Shocks:
    """A single-period shock with mode draws that force exploitation when eps ~ 0."""
    return Shocks(np.array([v_next], dtype=np.int64), np.array([u]),
                  np.full((I, 1), 0.99), np.zeros((I, 1), dtype=np.int64))


class TestRules1And2_PeriodOrder(unittest.TestCase):
    def test_price_uses_old_data_and_events_happen_in_order(self):
        sess = Session(CELL, 0, 123)
        st, g, p = sess.state, sess.grids, SMALL
        st.visits[:] = 10 ** 9                          # eps = exp(-500): pure exploitation
        p_idx, v_lag, v_cur = (int(st.cursor[k]) for k in (C_P_IDX, C_V_LAG, C_V_CUR))
        s = state_space.encode(p_idx, v_lag, v_cur, p.num_value_points)
        acts = [policy.lowest_action(int(st.policy[i, s])) for i in range(2)]

        # what the market maker knows BEFORE this period
        stats_before = st.stats.copy()
        snap_before = adaptive.snapshot(st.hist, st.cursor)
        _, _, g0, _, lam = adaptive.coefficients(stats_before, p.pricing_error_weight)
        np.testing.assert_allclose(
            adaptive.batch_coefficients(snap_before, p.pricing_error_weight), adaptive.coefficients(stats_before, p.pricing_error_weight), rtol=1e-9)
        Q_before, visits_before = st.Q.copy(), st.visits.copy()

        u, v_next = 0.05, 3
        x = [g.orders[v_cur, a] for a in acts]
        y = u + x[0] + x[1]
        expected_p = g0 + lam * y

        ran, converged = protocol.run_periods(*sess._kernel_args(one_shock(v_next, u), 1, True, Session._EMPTY_ROWS))
        self.assertEqual(ran, 1)
        self.assertFalse(converged)

        # (2) price came from OLD coefficients + current y
        newest = adaptive.snapshot(st.hist, st.cursor)[-1]
        self.assertAlmostEqual(newest[H_P], expected_p, places=13)
        self.assertAlmostEqual(newest[H_Y], y, places=13)
        # a "cheating" maker that included this period's row would have quoted differently
        _, _, g0c, _, lamc = adaptive.batch_coefficients(adaptive.snapshot(st.hist, st.cursor), p.pricing_error_weight)
        self.assertGreater(abs((g0c + lamc * y) - expected_p), 1e-12)

        # (4)+(5) the appended row is (v_t, p_t, z_t, y_t) with z computed from p_t
        v = g.value_grid[v_cur]
        self.assertEqual(newest[H_V], v)
        self.assertAlmostEqual(newest[H_Z], -p.investor_slope * (expected_p - p.value_mean), places=12)

        # (1) the visit counter moved exactly once, only for v_t
        diff = st.visits - visits_before
        self.assertEqual(diff[v_cur], 1)
        self.assertEqual(diff.sum(), 1)

        # (6) exactly one Q cell per agent changed, using the OLD next rows
        p_next = state_space.price_to_index(expected_p, g.price_grid[v_cur])
        base = state_space.next_block(p_next, v_cur, p.num_value_points)
        for i in range(2):
            changed = np.argwhere(st.Q[i] != Q_before[i]).tolist()
            self.assertEqual(changed, [[s, acts[i]]])
            cont = Q_before[i, base:base + p.num_value_points].max(axis=1).mean()
            profit = (v - expected_p) * x[i]
            expected_q = (1 - p.learning_rate) * Q_before[i, s, acts[i]] + p.learning_rate * (profit + p.discount_factor * cont)
            self.assertAlmostEqual(st.Q[i, s, acts[i]], expected_q, places=12)

        # state advanced: s_{t+1} = (grid(p_t), v_t, v_{t+1})
        self.assertEqual(int(st.cursor[C_P_IDX]), p_next)
        self.assertEqual(int(st.cursor[C_V_LAG]), v_cur)
        self.assertEqual(int(st.cursor[C_V_CUR]), v_next)
        self.assertEqual(int(st.cursor[C_T]), 1)
        self.assertEqual(int(st.cursor[C_APPENDS]), 1)

    def test_exploration_when_eps_is_one(self):
        sess = Session(CELL, 0, 5)
        st = sess.state
        self.assertTrue(np.all(st.visits == 0))       # eps = 1 on first visit
        shock = Shocks(np.array([0]), np.array([0.0]), np.zeros((2, 1)), np.array([[4], [11]], dtype=np.int64))
        v_cur = int(st.cursor[C_V_CUR])
        protocol.run_periods(*sess._kernel_args(shock, 1, True, Session._EMPTY_ROWS))
        newest = adaptive.snapshot(st.hist, st.cursor)[-1]
        expected_y = sess.grids.orders[v_cur, 4 % 15] + sess.grids.orders[v_cur, 11 % 15]
        self.assertAlmostEqual(newest[H_Y], expected_y, places=12)


class TestRule3_FreezeAfterConvergence(unittest.TestCase):
    def _stable_session(self, index):
        sess = Session(CELL, index, 123)
        sess.state.Q[:, :, 0] = 1e9                     # action 0 dominates everywhere
        sess.state.policy[...] = policy.initial_policy(sess.state.Q)
        return sess

    def test_training_stops_at_streak_and_measurement_freezes_learning(self):
        sess = self._stable_session(1)
        self.assertTrue(sess.train(chunk_size=10))
        self.assertEqual(sess.converged_at, 3)
        self.assertEqual(sess.phase, "converged")
        st = sess.state
        Q_c, visits_c, policy_c = st.Q.copy(), st.visits.copy(), st.policy.copy()
        appends_c, stats_c = int(st.cursor[C_APPENDS]), st.stats.copy()

        rows = sess.measure()
        self.assertEqual(rows.shape, (20, protocol.row_width(2)))
        self.assertTrue(np.array_equal(st.Q, Q_c))
        self.assertTrue(np.array_equal(st.visits, visits_c))
        self.assertTrue(np.array_equal(st.policy, policy_c))
        self.assertEqual(int(st.cursor[C_APPENDS]), appends_c + 20)   # maker kept rolling
        self.assertFalse(np.array_equal(st.stats, stats_c))
        self.assertEqual(sess.phase, "complete")
        # frozen greedy policy = action 0 for everyone; rows are internally consistent
        V = sess.grids.value_grid
        for r in rows:
            k = int(np.argmin(np.abs(V - r[protocol.COL_V])))
            for i in range(2):
                self.assertAlmostEqual(r[protocol.col_x(i)], sess.grids.orders[k, 0], places=12)
                self.assertAlmostEqual(r[protocol.col_pi(i, 2)], (r[protocol.COL_V] - r[protocol.COL_P]) * r[protocol.col_x(i)], places=12)
            self.assertAlmostEqual(r[protocol.COL_Z], -SMALL.investor_slope * (r[protocol.COL_P] - SMALL.value_mean), places=10)
            self.assertTrue(np.isfinite(r[protocol.COL_LAM]))

    def test_measure_before_convergence_is_rejected(self):
        sess = Session(CELL, 2, 123)
        with self.assertRaises(RuntimeError):
            sess.measure()


class TestRule4_Reproducibility(unittest.TestCase):
    def test_same_identity_same_path_different_index_different_path(self):
        a, b, c = Session(LONG, 0, 7), Session(LONG, 0, 7), Session(LONG, 1, 7)
        for s in (a, b, c):
            s.train(chunk_size=100, max_periods=300)
        for name in ("Q", "visits", "policy", "cursor", "hist", "stats"):
            self.assertTrue(np.array_equal(getattr(a.state, name), getattr(b.state, name)), name)
        self.assertFalse(np.array_equal(a.state.hist, c.state.hist))

    def test_streams_are_isolated_generators(self):
        s = ShockStreams(7, CELL, 0)
        first, second = s.draw(5, 10), s.draw(5, 10)
        direct = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            entropy=7, spawn_key=(CELL.key_uint32(), 0, 1)))).integers(10, size=10, dtype=np.int64)
        np.testing.assert_array_equal(np.concatenate([first.value_index, second.value_index]), direct)
        # chunk size does not change the sequence
        t = ShockStreams(7, CELL, 0).draw(10, 10)
        np.testing.assert_array_equal(t.noise, np.concatenate([first.noise, second.noise]))
        np.testing.assert_array_equal(t.mode, np.concatenate([first.mode, second.mode], axis=1))

    def test_checkpoint_roundtrip(self):
        original = Session(LONG, 2, 9)
        original.train(chunk_size=50, max_periods=150)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.npz")
            original.save_checkpoint(path)
            original.train(chunk_size=50, max_periods=300)
            resumed = Session(LONG, 2, 9)
            resumed.load_checkpoint(path)
            resumed.train(chunk_size=50, max_periods=300)
        self.assertTrue(np.array_equal(original.state.Q, resumed.state.Q))
        self.assertTrue(np.array_equal(original.state.cursor, resumed.state.cursor))
        self.assertTrue(np.array_equal(original.state.hist, resumed.state.hist))


if __name__ == "__main__":
    unittest.main()
