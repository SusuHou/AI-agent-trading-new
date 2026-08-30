"""Bit-level parity with the readable ``vibe_replication/steps`` oracle. / 与逐步版本对照。

Same grids, same D_0, same initial state, same supplied shocks, pure exploitation
(visit counters pre-set so eps ~ 0) -> identical prices, Q tables and history.
"""

import sys
import unittest

import numpy as np

import _setup
from dgj.config import ExperimentCell
from dgj.game import protocol
from dgj.game.session import Session, build_grids
from dgj.game.shocks import Shocks
from dgj.players.market_maker import adaptive
from dgj.players.market_maker.adaptive import C_P_IDX, C_V_CUR, C_V_LAG, H_P, H_Y
from dgj.players.speculator import policy


@unittest.skipUnless(_setup.HAVE_STEPS, "vibe_replication/steps not found next to dgj_sim")
class TestParityWithSteps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for path in (str(_setup.STEPS_ROOT), str(_setup.STEPS_DIR)):
            if path not in sys.path:
                sys.path.insert(0, path)
        import step_25_one_market_period as s25
        import step_26_reproducible_random_streams as s26
        from src.parameters import PaperParameters as StepParameters

        cls.s25, cls.s26 = s25, s26
        cls.step_params = StepParameters()
        cls.theirs = s25.build_paper_inputs(cls.step_params)
        cls.cell = ExperimentCell(price_grid="per_value")
        cls.grids = build_grids(cls.cell)

    def test_grids_and_initial_q_match(self):
        value_grid, price_grid, multipliers, q0, _ = self.theirs
        np.testing.assert_allclose(self.grids.value_grid, value_grid, atol=1e-12)
        np.testing.assert_allclose(self.grids.price_grid, price_grid, atol=1e-12)
        np.testing.assert_allclose(self.grids.multipliers, multipliers, atol=1e-12)
        from dgj.players.speculator import q_learning
        ours = q_learning.initial_q_table(self.grids.initial_q_block, 1, 31)[0]
        np.testing.assert_allclose(ours, q0, rtol=1e-10, atol=1e-10)

    def test_three_hundred_periods_identical(self):
        value_grid, price_grid, multipliers, q0, prehistory = self.theirs
        their = self.s26.build_randomized_paper_session(
            parameters=self.step_params, value_grid=value_grid, price_grid=price_grid,
            action_multipliers=multipliers, initial_q_table=q0, prehistory=prehistory,
            experiment_seed=1, experiment_cell_key="parity", session_index=0)
        rows = np.array([[r.fundamental_value_v, r.market_price_p, r.insensitive_order_z, r.informed_and_noise_order_y]
                         for r in prehistory.rows])
        ours = Session(self.cell, 0, 1, grids=self.grids, prehistory_rows=rows)
        p_idx, v_lag, v_cur = their.initial_state_indexes
        ours.state.cursor[C_P_IDX], ours.state.cursor[C_V_LAG], ours.state.cursor[C_V_CUR] = p_idx, v_lag, v_cur
        their.shared_value_visit_counts[:] = [10 ** 9] * 10
        ours.state.visits[:] = 10 ** 9
        if any(policy.popcount(int(m)) != 1 for m in ours.state.policy.ravel()):
            self.skipTest("exact Q_0 tie: tie-break randomness differs between implementations")

        rng = np.random.default_rng(42)
        for _ in range(300):
            u, v_next = float(rng.normal(0, 0.1)), int(rng.integers(10))
            trace = their.run_period_with_supplied_draws_for_test(u, v_next)
            shock = Shocks(np.array([v_next]), np.array([u]), np.full((2, 1), 0.99), np.zeros((2, 1), dtype=np.int64))
            protocol.run_periods(*ours._kernel_args(shock, 1, True, Session._EMPTY_ROWS))
            newest = adaptive.snapshot(ours.state.hist, ours.state.cursor)[-1]
            self.assertAlmostEqual(newest[H_P], trace.continuous_price_p, delta=1e-9)
            self.assertAlmostEqual(newest[H_Y], trace.total_order_flow_y, delta=1e-9)

        for i in range(2):
            np.testing.assert_allclose(ours.state.Q[i], their.traders[i].q_table, rtol=1e-9, atol=1e-9)
        self.assertEqual(ours.state.visits.tolist(), their.shared_value_visit_counts)
        their_hist = np.array([[r.fundamental_value_v, r.market_price_p, r.insensitive_order_z, r.informed_and_noise_order_y]
                               for r in their.market_maker.snapshot()])
        np.testing.assert_allclose(adaptive.snapshot(ours.state.hist, ours.state.cursor), their_hist, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
