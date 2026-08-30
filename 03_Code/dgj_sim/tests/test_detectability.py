"""Footnote 25 as a measurable quantity: can the other speculator SEE a one-step deviation?"""

import unittest

import _setup  # noqa: F401
from dgj.config import ExperimentCell, PaperParameters
from dgj.diagnostics import detectability
from dgj.game.session import build_grids


def _detect(noise, grid):
    g = build_grids(ExperimentCell(parameters=PaperParameters(noise_std=noise), price_grid=grid))
    return detectability(g, 1.0, noise, draws=1500)


class TestDeviationDetectability(unittest.TestCase):
    def test_low_noise_global_grid_hides_deviations(self):
        d = _detect(0.1, "global")
        self.assertLess(d.overall, 0.15)
        self.assertEqual(min(d.by_value), 0.0)          # values nearest v_bar: never detectable

    def test_low_noise_per_value_grid_reveals_deviations(self):
        d = _detect(0.1, "per_value")
        self.assertGreater(d.overall, 0.95)
        for step, move in zip(d.price_step, d.one_action_move):   # footnote 25: ~one grid step per action step
            self.assertGreater(move / step, 0.9)
            self.assertLess(move / step, 1.2)

    def test_high_noise_deviations_hidden_under_both_grids(self):
        # the 1.96*sigma_u term widens P(v) so much that price-trigger strategies cannot be learned:
        # exactly the paper's high-noise regime, where only over-pruning collusion survives
        self.assertLess(_detect(100.0, "global").overall, 0.15)
        self.assertLess(_detect(100.0, "per_value").overall, 0.30)


if __name__ == "__main__":
    unittest.main()
