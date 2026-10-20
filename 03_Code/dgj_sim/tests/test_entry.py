"""Entry-timing code: parity with Dou's kernel, inactive entrant, floor, wide grid, resume."""

import os
import tempfile
import unittest

import numpy as np

from dgj.config import ExperimentCell, PaperParameters
from dgj.game import entry_session
from dgj.game.entry_session import EntryCell, EntrySession, build_grids, NEVER
from dgj.game.session import Session
from dgj.players import benchmarks
from dgj.experiments import run_entry_cli

SEED = 20261019


def small_base(**changes) -> ExperimentCell:
    settings = {"convergence_periods": 2_000, "measurement_periods": 1_000, **changes}
    params = PaperParameters(**settings)
    return ExperimentCell(parameters=params)


class TestWideGrid(unittest.TestCase):
    def test_benchmarks_on_grid(self):
        g = build_grids(EntryCell(base=small_base()))
        m = g.multipliers
        self.assertEqual(len(m), 15)
        self.assertAlmostEqual(m[1], g.cartel.intensity, places=9)
        self.assertAlmostEqual(m[5], g.nash.intensity, places=9)
        mono = benchmarks.solve("nash", 1, 0.1, g.discrete_value_std, 500.0, 0.1).intensity
        self.assertAlmostEqual(mono, 2 * g.cartel.intensity, places=6)
        self.assertLess(abs(m[13] - mono), 1e-3)
        self.assertTrue(np.allclose(np.diff(m), (g.nash.intensity - g.cartel.intensity) / 4))

    def test_dou_grid_unchanged(self):
        base = small_base()
        a = build_grids(EntryCell(base=base, action_grid="dou"))
        from dgj.game.session import build_grids as dou
        b = dou(base)
        self.assertTrue(np.array_equal(a.orders, b.orders))
        self.assertTrue(np.array_equal(a.initial_q_block, b.initial_q_block))


class TestParity(unittest.TestCase):
    def test_simultaneous_no_floor_dou_grid_equals_baseline(self):
        base = small_base()
        ref = Session(base, 3, SEED)
        ref.train(chunk_size=1_000, max_periods=20_000)
        new = EntrySession(EntryCell(base=base, action_grid="dou", exploration_floor=0.0), 3, SEED)
        new.advance(20_000, chunk_size=1_000)
        self.assertEqual(ref.periods_completed, new.periods_completed)
        self.assertTrue(np.array_equal(ref.state.Q, new.state.Q))
        self.assertTrue(np.array_equal(ref.state.policy, new.state.policy))
        self.assertTrue(np.array_equal(new.state.visits[0], new.state.visits[1]))
        self.assertTrue(np.array_equal(ref.state.visits, new.state.visits[0]))


class TestEntrant(unittest.TestCase):
    def test_entrant_untouched_before_entry(self):
        cell = EntryCell(base=small_base(), entry_period=10_000)
        s = EntrySession(cell, 0, SEED)
        q1 = s.state.Q[1].copy()
        s.advance(10_000, chunk_size=1_000)
        self.assertTrue(np.array_equal(q1, s.state.Q[1]))
        self.assertEqual(int(s.state.visits[1].sum()), 0)
        self.assertEqual(int(s.state.visits[0].sum()), 10_000)
        s.advance(1_000, chunk_size=1_000)
        self.assertEqual(int(s.state.visits[1].sum()), 1_000)

    def test_no_stop_before_entry(self):
        # tiny streak target: the lone incumbent would "converge" early, but clock entry must wait
        cell = EntryCell(base=small_base(convergence_periods=5), entry_period=5_000, exploration_floor=0.0)
        s = EntrySession(cell, 1, SEED)
        s.advance(5_000, chunk_size=1_000)
        self.assertEqual(s.phase, "training")
        self.assertEqual(s.periods_completed, 5_000)

    def test_stage3_enters_on_convergence(self):
        cell = EntryCell(base=small_base(convergence_periods=50), entry_on_convergence=True,
                         alone_cap=100_000, exploration_floor=0.0)
        s = EntrySession(cell, 2, SEED)
        self.assertEqual(int(s.entry_period[1]), NEVER)
        while s.phase == "alone":
            s.advance(1_000, chunk_size=1_000)
        self.assertEqual(int(s.entry_period[1]), s.periods_completed)
        self.assertFalse(s.alone_capped)

    def test_stage3_cap(self):
        cell = EntryCell(base=small_base(), entry_on_convergence=True, alone_cap=3_000)
        s = EntrySession(cell, 2, SEED)
        s.advance(3_000, chunk_size=1_000)
        self.assertTrue(s.alone_capped)
        self.assertEqual(int(s.entry_period[1]), 3_000)


class TestFloor(unittest.TestCase):
    def test_floor_keeps_exploring(self):
        # beta huge -> Dou's eps ~ 0 after a few visits; with the floor the policy keeps being tested
        def run(floor):
            cell = EntryCell(base=small_base(exploration_decay=1.0), exploration_floor=floor)
            s = EntrySession(cell, 4, SEED)
            q0 = s.state.Q.copy()
            s.advance(20_000, chunk_size=1_000)
            return int((s.state.Q != q0).sum())
        self.assertGreater(run(0.2), run(0.0))


class TestResume(unittest.TestCase):
    def test_checkpoint_resume_identical(self):
        cell = EntryCell(base=small_base(), entry_period=4_000)
        a = EntrySession(cell, 5, SEED)
        a.advance(8_000, chunk_size=1_000)
        b = EntrySession(cell, 5, SEED)
        b.advance(3_000, chunk_size=1_000)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ck.npz")
            b.save_entry_checkpoint(path, chunk_size=1_000)
            c = EntrySession(cell, 5, SEED)
            c.load_entry_checkpoint(path, chunk_size=1_000)
        c.advance(5_000, chunk_size=1_000)
        self.assertTrue(np.array_equal(a.state.Q, c.state.Q))
        self.assertTrue(np.array_equal(a.state.cursor, c.state.cursor))


class TestCli(unittest.TestCase):
    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            args = ["--out", d, "--session", "0", "--seed", str(SEED), "--entry-period", "0",
                    "--convergence-periods", "5", "--measurement-periods", "1000",
                    "--beta", "1e-3", "--floor", "0", "--work-periods", "2000000", "--chunk-size", "1000"]
            self.assertEqual(run_entry_cli.main(args), 0)
            self.assertTrue(os.path.exists(os.path.join(d, "session_0000.json")))

    def test_clock_entry_periods(self):
        self.assertEqual(run_entry_cli.clock_entry_period(0.1, 5e-7, 10), 46_051_702)
        self.assertEqual(run_entry_cli.clock_entry_period(0.01, 5e-7, 10), 92_103_404)



class TestDiagnostics(unittest.TestCase):
    def test_default_key_unchanged_by_new_fields(self):
        # key of the Narval measure-first cell (submitted at a619f04) must not move
        base = ExperimentCell(parameters=PaperParameters())
        cell = EntryCell(base=base, entry_on_convergence=True, alone_cap=2_000_000_000)
        self.assertNotIn("incumbent_init", cell.to_dict()["entry"])
        self.assertNotIn("lagged_price", cell.to_dict()["entry"])

    def test_lone_init_favours_monopolist(self):
        cell = EntryCell(base=small_base(), incumbent_init="lone")
        s = EntrySession(cell, 0, SEED)
        from dgj.players.speculator import policy as pm
        idx = {pm.lowest_action(int(m)) for m in s.state.policy[0]}
        self.assertEqual(idx, {13})
        dou = EntrySession(EntryCell(base=small_base()), 0, SEED)
        self.assertEqual({pm.lowest_action(int(m)) for m in dou.state.policy[0]}, {4})
        self.assertTrue(np.array_equal(s.state.Q[1], dou.state.Q[1]))   # entrant keeps Dou's values

    def test_no_lagged_price_keeps_index_zero(self):
        cell = EntryCell(base=small_base(), lagged_price=False)
        s = EntrySession(cell, 0, SEED)
        s.advance(5_000, chunk_size=1_000)
        from dgj.players.speculator.state_space import decode
        touched = np.nonzero((s.state.Q[0] != EntrySession(cell, 0, SEED).state.Q[0]).any(axis=1))[0]
        self.assertTrue(all(decode(int(r), 10)[0] == 0 for r in touched))

class TestConcurrentStart(unittest.TestCase):
    def test_many_tasks_create_cell_json_together(self):
        import subprocess, sys
        with tempfile.TemporaryDirectory() as d:
            args = ["--out", d, "--seed", "1", "--entry-on-convergence", "--alone-cap", "3000",
                    "--stop-after-alone", "--convergence-periods", "5", "--work-periods", "3000",
                    "--chunk-size", "1000"]
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", PYTHONPATH=os.getcwd())
            procs = [subprocess.Popen([sys.executable, "-m", "dgj.experiments.run_entry_cli", "--session", str(k), *args],
                                      env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) for k in range(8)]
            codes = [p.wait() for p in procs]
            self.assertEqual(codes, [0] * 8, [p.stderr.read()[-300:] for p in procs])


if __name__ == "__main__":
    unittest.main()
