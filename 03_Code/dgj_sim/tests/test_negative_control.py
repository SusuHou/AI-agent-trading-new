"""Negative control: no lagged price in the state (ledger I-14). / 阴性对照：状态中没有滞后价格。

Dou et al. (Section 5, printed p. 32) remove p_{t-1} from s_t; price-trigger monitoring is then
impossible. These tests check that the switch does exactly that and nothing else.
"""

import unittest
from dataclasses import replace

import numpy as np

import _setup  # noqa: F401
from dgj.config import ExperimentCell, PaperParameters
from dgj.game.session import Session
from dgj.game.shocks import ShockStreams
from dgj.players.market_maker.adaptive import C_P_IDX

SMALL = PaperParameters(market_maker_window=200)
BASE = ExperimentCell(parameters=SMALL, label="nc_base")
CONTROL = replace(BASE, state_includes_lagged_price=False, label="nc_control")


class TestNegativeControl(unittest.TestCase):
    def test_default_cells_keep_their_keys(self):
        # the clean per-value cohort on Narval was recorded under this key
        self.assertEqual(ExperimentCell().key(), "0af80038fb4ea945")
        self.assertNotIn("state_includes_lagged_price", ExperimentCell().to_dict())

    def test_switch_changes_the_cell_key(self):
        self.assertNotEqual(BASE.key(), CONTROL.key())
        self.assertIs(CONTROL.to_dict()["state_includes_lagged_price"], False)

    def test_shocks_are_common_with_the_baseline(self):
        a, b = ShockStreams(7, BASE, 2), ShockStreams(7, CONTROL, 2)
        x, y = a.draw(40, 10), b.draw(40, 10)
        for name in ("value_index", "noise", "mode", "action"):
            np.testing.assert_array_equal(getattr(x, name), getattr(y, name), name)

    def test_price_index_is_always_zero_and_only_those_states_learn(self):
        fresh = Session(CONTROL, 0, 11)
        q0 = fresh.state.Q.copy()
        self.assertEqual(int(fresh.state.cursor[C_P_IDX]), 0)
        s = Session(CONTROL, 0, 11)
        for k in range(1, 6):
            s.train(chunk_size=200, max_periods=200 * k)   # max_periods is cumulative
            self.assertEqual(int(s.state.cursor[C_P_IDX]), 0)
        n_v = SMALL.num_value_points
        p_zero_rows = n_v * n_v                     # state ids with p_idx = 0 come first
        self.assertFalse(np.array_equal(s.state.Q[:, :p_zero_rows], q0[:, :p_zero_rows]))
        np.testing.assert_array_equal(s.state.Q[:, p_zero_rows:], q0[:, p_zero_rows:])

    def test_baseline_still_uses_the_lagged_price(self):
        s = Session(BASE, 0, 11)
        seen = set()
        for k in range(1, 6):
            s.train(chunk_size=200, max_periods=200 * k)   # max_periods is cumulative
            seen.add(int(s.state.cursor[C_P_IDX]))
        self.assertGreater(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
