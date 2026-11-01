"""Best-response test (C2) on a tiny grid: exact solution, Monte Carlo check of the transition model, CLI."""

import json
import os
import tempfile
import unittest

import numpy as np

from dgj.config import ExperimentCell, PaperParameters
from dgj.experiments import run_entry_cli
from dgj.game import best_response as br
from dgj.game.entry_session import EntryCell, EntrySession
from dgj.players.speculator.state_space import price_to_index

SEED = 20261101


def tiny_session(**changes):
    settings = {"num_price_points": 5, "num_value_points": 2, "num_action_points": 3,
                "convergence_periods": 200, "measurement_periods": 2_000, "exploration_decay": 1e-3, **changes}
    cell = EntryCell(base=ExperimentCell(parameters=PaperParameters(**settings)), action_grid="dou",
                     exploration_floor=0.0)
    s = EntrySession(cell, 0, SEED)
    s.advance(2_000_000, chunk_size=1_000)
    return s


class TestBestResponse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = tiny_session(noise_std=0.5)
        g, p = cls.s.grids, cls.s.p
        cls.gamma_0, cls.lam = 1.0, 0.05
        cls.rival = br.greedy_actions(cls.s.state.policy[1])
        cls.reward, cls.P, cls.v_cur = br.transition_model(g.orders, g.value_grid, g.price_grid, cls.rival,
                                                           cls.gamma_0, cls.lam, p.noise_std)
        cls.n_p, cls.n_v = g.price_grid.shape[1], len(g.value_grid)

    def test_transition_probabilities_sum_to_one(self):
        self.assertTrue(np.allclose(self.P.sum(axis=2), 1.0))
        self.assertTrue((self.P >= -1e-15).all())

    def test_best_policy_value_equals_optimum(self):
        V_star, Q_star = br.solve(self.reward, self.P, self.v_cur, self.n_p, self.n_v, 0.95)
        V_b, _ = br.solve(self.reward, self.P, self.v_cur, self.n_p, self.n_v, 0.95, own_actions=Q_star.argmax(axis=1))
        self.assertTrue(np.allclose(V_b, V_star, rtol=1e-8, atol=1e-8))
        self.assertTrue(np.allclose(Q_star.max(axis=1), V_star, rtol=1e-8, atol=1e-8))

    def test_monte_carlo_matches_exact_value(self):
        """Simulate the frozen environment with price_to_index (the kernel's mapping) and compare."""
        g, p = self.s.grids, self.s.p
        rho = 0.95
        V_star, Q_star = br.solve(self.reward, self.P, self.v_cur, self.n_p, self.n_v, rho)
        pol = Q_star.argmax(axis=1)
        rng = np.random.default_rng(1)
        s0 = 3
        n_ep, H = 4000, 160
        totals = np.empty(n_ep)
        for e in range(n_ep):
            s, tot, disc = s0, 0.0, 1.0
            for _ in range(H):
                pi, vl, vc = s // (self.n_v * self.n_v), (s // self.n_v) % self.n_v, s % self.n_v
                a, b = pol[s], self.rival[s]
                price = self.gamma_0 + self.lam * (g.orders[vc, a] + g.orders[vc, b] + rng.normal(0, p.noise_std))
                tot += disc * (g.value_grid[vc] - price) * g.orders[vc, a]
                disc *= rho
                k = price_to_index(price, g.price_grid[vc])
                s = (k * self.n_v + vc) * self.n_v + rng.integers(self.n_v)
            totals[e] = tot
        se = totals.std(ddof=1) / np.sqrt(n_ep)
        self.assertLess(abs(totals.mean() - V_star[s0]), 4 * se + abs(V_star[s0]) * rho ** H + 1e-9)

    def test_best_response_policy_scores_as_equilibrium(self):
        V_star, Q_star = br.solve(self.reward, self.P, self.v_cur, self.n_p, self.n_v, 0.95)
        policy = self.s.state.policy.copy()
        policy[0] = 1 << Q_star.argmax(axis=1)            # agent 0 plays its exact best response
        w = np.ones(len(V_star))
        r = br.best_response_test(self.s.grids, policy, 0, self.gamma_0, self.lam, self.s.p.noise_std, 0.95, w)
        self.assertEqual(r["share_optimal_all_states"], 1.0)
        self.assertTrue(r["subgame_perfect"] and r["nash_on_path"])
        self.assertAlmostEqual(r["deviation_gain_on_path"], 0.0, places=9)

    def test_cli_writes_br(self):
        with tempfile.TemporaryDirectory() as d:
            args = ["--out", d, "--session", "0", "--seed", str(SEED), "--entry-period", "0", "--floor", "0",
                    "--action-grid", "dou", "--convergence-periods", "5", "--measurement-periods", "1000",
                    "--beta", "1e-3", "--work-periods", "2000000", "--chunk-size", "1000", "--br-test"]
            self.assertEqual(run_entry_cli.main(args), 0)
            with open(os.path.join(d, "session_0000.json")) as f:
                r = json.load(f)
            self.assertEqual(len(r["br"]["agents"]), 2)
            for a in r["br"]["agents"]:
                self.assertTrue(0.0 <= a["share_optimal_all_states"] <= 1.0)
                self.assertGreaterEqual(a["deviation_gain_on_path"], -1e-9)


if __name__ == "__main__":
    unittest.main()
