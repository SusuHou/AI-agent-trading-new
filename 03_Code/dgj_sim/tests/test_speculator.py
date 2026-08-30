from bisect import bisect_left
import unittest

import numpy as np

import _setup  # noqa: F401
from dgj.players.speculator import action_space, policy, q_learning, state_space


class TestStateSpace(unittest.TestCase):
    def test_encode_decode_and_contiguous_next_block(self):
        n_p, n_v = 31, 10
        seen = set()
        for p in range(n_p):
            for vl in range(n_v):
                for vc in range(n_v):
                    sid = state_space.encode(p, vl, vc, n_v)
                    self.assertEqual(state_space.decode(sid, n_v), (p, vl, vc))
                    self.assertEqual(sid, state_space.next_block(p, vl, n_v) + vc)
                    seen.add(sid)
        self.assertEqual(seen, set(range(state_space.number_of_states(n_p, n_v))))

    def test_price_to_index_matches_bisect_reference(self):
        P = np.linspace(-0.3, 2.3, 31)

        def reference(price):
            ins = bisect_left(P.tolist(), price)
            if ins == 0:
                return 0
            if ins == len(P):
                return len(P) - 1
            return ins - 1 if price - P[ins - 1] <= P[ins] - price else ins

        rng = np.random.default_rng(0)
        probes = np.concatenate([rng.uniform(-1, 3, 2000), P, (P[:-1] + P[1:]) / 2, P + 1e-9, P - 1e-9])
        for price in probes:
            self.assertEqual(state_space.price_to_index(float(price), P), reference(float(price)))


class TestActionSpaceAndQ(unittest.TestCase):
    def test_multipliers_and_mirror(self):
        m = action_space.multipliers(4.0, 2.0, 0.1, 5)
        np.testing.assert_allclose(m, [1.8, 2.4, 3.0, 3.6, 4.2], atol=1e-12)
        orders = action_space.orders_table(np.array([0.5, 1.5]), 1.0, m)
        np.testing.assert_allclose(orders[0], -orders[1], atol=1e-12)

    def test_initial_q_toy(self):
        # step-16 toy: v=2, v_bar=1, own=1, opponents {1,3}, I=2, lambda=0.25, rho=0.5 -> Q=0.5
        block = q_learning.initial_q_block(np.array([2.0]), 1.0, np.array([[1.0, 3.0]]), 2, 0.25, 0.5)
        self.assertAlmostEqual(block[0, 0], 0.5, places=12)
        table = q_learning.initial_q_table(block, 2, 3)
        self.assertEqual(table.shape, (2, 3 * 1 * 1, 2))
        np.testing.assert_allclose(table[1, 2], block[0])

    def test_choose_action_rules(self):
        self.assertEqual(q_learning.choose_action(0b010, 0.0, 0.5, 12345, 3), 1)      # unique argmax
        self.assertEqual(q_learning.choose_action(0b010, 1.0, 0.5, 7, 3), 7 % 3)      # explore
        self.assertEqual(q_learning.choose_action(0b101, 0.0, 0.5, 0, 3), 0)          # tie, k=0
        self.assertEqual(q_learning.choose_action(0b101, 0.0, 0.5, 1, 3), 2)          # tie, k=1
        self.assertEqual(policy.lowest_action(0b1100), 2)
        self.assertEqual(policy.popcount(0b1011), 3)

    def test_update_uses_old_rows_and_touches_one_cell(self):
        Q = np.array([[[10.0, 5.0, 0.0], [12.0, 20.0, 8.0], [3.0, 7.0, 6.0]]])
        before = Q.copy()
        q_learning.update(Q, 0, 0, 0, 2.0, 1, 2, 0.01, 0.95)     # base=1, n_v=2 -> cont=(20+7)/2
        self.assertAlmostEqual(Q[0, 0, 0], 10.04825, places=12)
        self.assertEqual(np.argwhere(Q != before).tolist(), [[0, 0, 0]])
        # self-transition: continuation must come from the old row
        Q2 = np.array([[[10.0, 5.0, 0.0]]])
        q_learning.update(Q2, 0, 0, 0, 2.0, 0, 1, 0.01, 0.95)
        self.assertAlmostEqual(Q2[0, 0, 0], 10.015, places=12)

    def test_policy_masks(self):
        Q = np.array([[[1.0, 5.0, 2.0], [5.0, 5.0, 1.0]]])
        masks = policy.initial_policy(Q)
        self.assertEqual(masks.tolist(), [[0b010, 0b011]])
        self.assertEqual(policy.argmax_mask(Q, 0, 1), 0b011)
        self.assertEqual(policy.actions_from_mask(0b011, 3), (0, 1))


if __name__ == "__main__":
    unittest.main()


class TestPriceGridResolution(unittest.TestCase):
    """Footnote 25: with P(v), one action step by one speculator moves p by about one price step."""

    def test_per_value_grid_resolves_one_action_step(self):
        from dgj.config import ExperimentCell
        from dgj.game.session import build_grids
        g = build_grids(ExperimentCell(price_grid="per_value"))
        step_x = g.multipliers[1] - g.multipliers[0]
        for k, v in enumerate(g.value_grid):
            price_step = g.price_grid[k, 1] - g.price_grid[k, 0]
            action_price_move = g.nash.price_impact * step_x * abs(v - 1.0)
            self.assertGreater(action_price_move / price_step, 0.5)
            self.assertLess(action_price_move / price_step, 2.0)
        g_global = build_grids(ExperimentCell(price_grid="global"))
        self.assertGreater((g_global.price_grid[0, 1] - g_global.price_grid[0, 0]) / (g.nash.price_impact * step_x * 1.755), 5.0)
