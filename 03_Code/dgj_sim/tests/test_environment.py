import unittest

import numpy as np

import _setup  # noqa: F401
from dgj.environment import fundamental
from dgj.environment.insensitive_investors import demand


class TestEnvironment(unittest.TestCase):
    def test_value_grid_matches_paper(self):
        V = fundamental.value_grid(1.0, 1.0, 10)
        self.assertEqual(len(V), 10)
        self.assertTrue(np.all(np.diff(V) > 0))
        self.assertAlmostEqual(float(V.mean()), 1.0, places=12)
        self.assertAlmostEqual(fundamental.discrete_std(V, 1.0), 0.937969795249, places=11)
        np.testing.assert_allclose(V + V[::-1], 2.0, atol=1e-12)   # symmetric about v_bar

    def test_insensitive_demand_signs(self):
        self.assertAlmostEqual(demand(1.01, 500.0, 1.0), -5.0)
        self.assertAlmostEqual(demand(0.99, 500.0, 1.0), 5.0)
        self.assertEqual(demand(1.00, 500.0, 1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
