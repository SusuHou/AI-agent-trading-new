"""Entry-timing session: incumbent learns alone, a fresh entrant joins. / 进入时机实验的 session。

Design (EXPERIMENT_PLAN_PAPER.md, decided 2026-10-19):
    action grid  "wide": 15 points, step (chi^N - chi^M)/4, from chi^M - step; cartel (j=1),
                 Nash (j=5) and the lone monopolist 2 chi^M (j=13, within 1e-5 steps) lie on it.
                 "dou": the published grid (control for the code, not for the paper).
    floor        eps >= exploration_floor for both agents.
    entry        agent 0 (incumbent) active from period 0; agent 1 (entrant) from
                 ``entry_period`` (clock; 0 = simultaneous start, the control), or, when
                 ``entry_on_convergence`` is True, from the period at which the lone incumbent
                 meets Dou's convergence rule (capped at ``alone_cap``; capped sessions are flagged).
    after entry  train until the joint greedy policy is unchanged for convergence_periods,
                 then measure with the frozen policy exactly as in ``Session.measure``.

Both agents start from Dou et al.'s initial Q (computed for the grid in use). The random
streams are those of a 2-speculator ``ExperimentCell`` (spawn key = (session, stream)), so
all entry cells, the control and the lone-incumbent run share every shock.
"""

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import time

import numpy as np

from dgj.config import ExperimentCell
from dgj.game import entry_protocol, protocol
from dgj.game import session as base_session
from dgj.game.shocks import ShockStreams
from dgj.players.market_maker.adaptive import C_STREAK, C_T
from dgj.players.speculator import action_space, policy as policy_module, q_learning

ENTRY_SCHEMA_VERSION = 1
NEVER = np.iinfo(np.int64).max // 4


def aligned_fine_multipliers(nash_intensity: float, cartel_intensity: float, num_actions: int) -> np.ndarray:
    """Menu test (2026-10-24): step (chi^N - chi^M)/12 from one step below cartel, so cartel (j=1) and Nash (j=13)
    lie on the grid with a step close to Dou's; 15 points ~ Dou's range, 43 points reach the monopolist (j=37)."""
    if nash_intensity <= cartel_intensity:
        raise ValueError("calibration requires chi^N > chi^M")
    step = (nash_intensity - cartel_intensity) / 12.0
    return cartel_intensity + step * np.arange(-1, num_actions - 1)


def wide_multipliers(nash_intensity: float, cartel_intensity: float, num_actions: int) -> np.ndarray:
    """Step = (chi^N - chi^M)/4, first point one step below cartel. / 宽网格。"""
    if nash_intensity <= cartel_intensity:
        raise ValueError("calibration requires chi^N > chi^M")
    step = (nash_intensity - cartel_intensity) / 4.0
    return cartel_intensity + step * np.arange(-1, num_actions - 1)


@dataclass(frozen=True)
class EntryCell:
    base: ExperimentCell = ExperimentCell()
    action_grid: str = "wide"            # "wide" | "dou"
    exploration_floor: float = 0.01
    entry_period: int = 0                # clock entry of agent 1 (0 = simultaneous start)
    entry_on_convergence: bool = False   # stage 3: enter when the lone incumbent converges
    alone_cap: int = 0                   # stage 3 cap on the alone phase (periods)
    incumbent_init: str = "dou"          # "dou" (uniform rival, I = 2) | "lone" (no rival): diagnostic T1
    lagged_price: bool = True            # False: no p_{t-1} in the state: diagnostic T2
    reset_incumbent_at_entry: bool = False  # test A: incumbent's visit counts set to 0 at entry (eps back to 1)
    joint_cap: int = 0                   # >0: stop joint training this many periods after entry, measure, flag

    def __post_init__(self) -> None:
        if self.base.parameters.num_speculators != 2:
            raise ValueError("entry cells have exactly two speculators")
        if self.action_grid not in ("wide", "dou", "aligned_fine"):
            raise ValueError("action_grid must be 'wide', 'dou' or 'aligned_fine'")
        if not 0.0 <= self.exploration_floor < 1.0:
            raise ValueError("exploration_floor must lie in [0, 1)")
        if self.entry_period < 0:
            raise ValueError("entry_period must be non-negative")
        if self.entry_on_convergence and self.alone_cap <= 0:
            raise ValueError("entry_on_convergence needs a positive alone_cap")
        if not self.base.state_includes_lagged_price:
            raise ValueError("use EntryCell.lagged_price, not the base cell's flag")
        if self.incumbent_init not in ("dou", "lone"):
            raise ValueError("incumbent_init must be 'dou' or 'lone'")
        if self.joint_cap < 0:
            raise ValueError("joint_cap must be non-negative")

    def to_dict(self) -> dict:
        d = self.base.to_dict()
        d["entry"] = {
            "action_grid": self.action_grid,
            "exploration_floor": self.exploration_floor,
            "entry_period": self.entry_period,
            "entry_on_convergence": self.entry_on_convergence,
            "alone_cap": self.alone_cap,
            "schema": ENTRY_SCHEMA_VERSION,
        }
        # diagnostics recorded only when switched on, so existing cells keep their keys
        if self.incumbent_init != "dou":
            d["entry"]["incumbent_init"] = self.incumbent_init
        if not self.lagged_price:
            d["entry"]["lagged_price"] = False
        if self.reset_incumbent_at_entry:
            d["entry"]["reset_incumbent_at_entry"] = True
        if self.joint_cap:
            d["entry"]["joint_cap"] = self.joint_cap
        return d

    def key(self) -> str:
        payload = self.to_dict()
        payload.pop("label")
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def build_grids(cell: EntryCell) -> base_session.Grids:
    """Dou's grids, with the action grid (and so the initial Q) replaced when 'wide'.

    The price grid is Dou's per-value (or global) grid, unchanged.
    """
    g = base_session.build_grids(cell.base)
    if cell.action_grid == "dou":
        return g
    p = cell.base.parameters
    make = aligned_fine_multipliers if cell.action_grid == "aligned_fine" else wide_multipliers
    mult = make(g.nash.intensity, g.cartel.intensity, p.num_action_points)
    orders = action_space.orders_table(g.value_grid, p.value_mean, mult)
    q_block = q_learning.initial_q_block(
        g.value_grid, p.value_mean, orders, p.num_speculators, g.nash.price_impact, p.discount_factor
    )
    return replace(g, multipliers=mult, orders=orders, initial_q_block=q_block)


class EntrySession(base_session.Session):
    """Phases: alone (stage 3 only) -> training -> converged -> measurement -> complete."""

    def __init__(self, cell: EntryCell, session_index: int, experiment_seed: int) -> None:
        self.entry_cell = cell
        super().__init__(cell.base, session_index, experiment_seed, grids=build_grids(cell))
        I = self.p.num_speculators
        self.state.visits = np.zeros((I, self.p.num_value_points), dtype=np.int64)
        self.entry_period = np.zeros(I, dtype=np.int64)
        if cell.entry_on_convergence:
            self.entry_period[1] = NEVER
            self.phase = "alone"
        else:
            self.entry_period[1] = cell.entry_period
        if cell.incumbent_init == "lone":
            g = self.grids
            lone = q_learning.initial_q_block(
                g.value_grid, self.p.value_mean, g.orders, 1, g.nash.price_impact, self.p.discount_factor
            )
            self.state.Q[0] = np.tile(lone, (self.p.num_price_points * self.p.num_value_points, 1))
            self.state.policy[...] = policy_module.initial_policy(self.state.Q)
        if not cell.lagged_price:
            self.state.cursor[base_session.C_P_IDX] = 0
        self.alone_converged_at: int | None = None
        self.alone_capped = False
        self.incumbent_reset_at: int | None = None     # period at which test A reset the incumbent's visits
        self.joint_capped = False                      # True: measured at the cap, not at convergence

    # ------------------------------------------------------------------
    def _entry_args(self, shocks, n, require_all_active):
        st, g, p = self.state, self.grids, self.p
        return (
            st.Q, st.visits, st.policy, st.cursor, st.hist, st.stats, g.orders, g.value_grid, g.price_grid,
            shocks.value_index, shocks.noise, shocks.mode, shocks.action,
            p.learning_rate, p.discount_factor, p.exploration_decay, self.entry_cell.exploration_floor,
            p.pricing_error_weight, p.investor_slope, p.value_mean, p.market_maker_window,
            n, p.convergence_periods, self.entry_period, require_all_active, self.entry_cell.lagged_price,
        )

    def _enter_now(self, *, capped: bool) -> None:
        t = self.periods_completed
        self.entry_period[1] = t
        self.alone_converged_at = None if capped else t
        self.alone_capped = capped
        self.state.cursor[C_STREAK] = 0
        self.phase = "training"

    def advance(self, work_periods: int, *, chunk_size: int = 1_000_000) -> str:
        """Advance by at most ``work_periods``; returns the phase reached."""
        if work_periods % chunk_size:
            raise ValueError("work_periods must be a multiple of chunk_size")
        t0 = time.perf_counter()
        end = self.periods_completed + work_periods
        while self.phase in ("alone", "training") and self.periods_completed < end:
            if (self.entry_cell.reset_incumbent_at_entry and self.phase == "training"
                    and self.incumbent_reset_at is None and self.periods_completed >= self.entry_period[1]):
                self.state.visits[0, :] = 0           # incumbent explores afresh, like the entrant
                self.incumbent_reset_at = self.periods_completed
            n = chunk_size
            if self.phase == "alone":
                n = min(n, self.entry_cell.alone_cap - self.periods_completed)
            shocks = self.streams.draw(n, self.p.num_value_points)
            streak_before = int(self.state.cursor[C_STREAK])
            ran, converged = entry_protocol.run_periods_entry(*self._entry_args(shocks, n, self.phase == "training"))
            if int(self.state.cursor[C_STREAK]) < streak_before + ran:
                self.policy_changes_seen += 1
            if self.phase == "alone":
                if converged:
                    self._enter_now(capped=False)
                elif self.periods_completed >= self.entry_cell.alone_cap:
                    self._enter_now(capped=True)
                if self.phase != "alone":
                    break                      # return at the entry point (checkpoint / stop-after-alone)
            elif converged:
                self.converged_at = self.periods_completed
                self.phase = "converged"
            elif (self.entry_cell.joint_cap
                  and self.periods_completed - int(self.entry_period[1]) >= self.entry_cell.joint_cap):
                self.joint_capped = True             # common-horizon reading (Atanasova B.2): freeze and measure
                self.phase = "converged"
        self.wall_time["training"] += time.perf_counter() - t0
        return self.phase

    def measure(self, n_periods: int | None = None, *, chunk_size: int = 100_000) -> np.ndarray:
        """Frozen measurement with Dou's kernel (visits are not read when learning=False)."""
        visits = self.state.visits
        self.state.visits = visits[0].copy()
        try:
            return super().measure(n_periods, chunk_size=chunk_size)
        finally:
            self.state.visits = visits

    def incumbent_play(self, n_periods: int = 100_000) -> dict:
        """What the incumbent actually trades under its frozen greedy policy, alone.

        Runs on COPIES of the state (the session itself is not advanced). Returns the traded
        intensity chi_hat (least squares of x on v - v_bar) and the action frequencies.
        """
        st, g, p = self.state, self.grids, self.p
        cursor, hist, stats = st.cursor.copy(), st.hist.copy(), st.stats.copy()
        rng = np.random.default_rng([self.session_index, 7_777])          # not the training streams
        vals = rng.integers(p.num_value_points, size=n_periods).astype(np.int64)
        noise = rng.normal(0.0, p.noise_std, size=n_periods)
        acts = np.empty((p.num_speculators, n_periods), dtype=np.int64)
        v = np.empty(n_periods)
        active = np.array([True, False])
        entry_protocol.play_frozen(
            st.policy, cursor, hist, stats, g.orders, g.value_grid, g.price_grid, vals, noise,
            p.pricing_error_weight, p.investor_slope, p.value_mean, p.market_maker_window,
            n_periods, active, self.entry_cell.lagged_price, acts, v,
        )
        a = acts[0]
        x = g.orders[np.searchsorted(g.value_grid, v), a]
        dv = v - p.value_mean
        freq = np.bincount(a, minlength=g.orders.shape[1]) / n_periods
        return {"chi_hat": float((x * dv).sum() / (dv * dv).sum()),
                "action_freq": [round(float(f), 4) for f in freq],
                "modal_action": int(freq.argmax())}

    # ------------------------------------------------------------------
    def entry_manifest(self) -> dict:
        return {
            "entry_cell": self.entry_cell.to_dict(),
            "entry_cell_key": self.entry_cell.key(),
            "entry_period": int(self.entry_period[1]),
            "alone_converged_at": self.alone_converged_at,
            "alone_capped": self.alone_capped,
            "incumbent_reset_at": self.incumbent_reset_at,
            "joint_capped": self.joint_capped,
            "converged_at": self.converged_at,
            "periods_completed": self.periods_completed,
            "policy_changes_seen": self.policy_changes_seen,
            "multipliers": [float(m) for m in self.grids.multipliers],
            "nash_chi": self.grids.nash.intensity,
            "cartel_chi": self.grids.cartel.intensity,
            "wall_time_seconds": dict(self.wall_time),
        }

    def save_entry_checkpoint(self, path: str, *, chunk_size: int) -> None:
        st = self.state
        base_session.atomic_savez_compressed(
            path,
            entry_schema_version=ENTRY_SCHEMA_VERSION,
            entry_cell_key=self.entry_cell.key(),
            session_index=self.session_index,
            experiment_seed=self.streams.experiment_seed,
            chunk_size=chunk_size,
            Q=st.Q, visits=st.visits, policy=st.policy, cursor=st.cursor, hist=st.hist, stats=st.stats,
            entry_period=self.entry_period,
            rng=json.dumps(self.streams.state()),
            phase=self.phase,
            alone_converged_at=-1 if self.alone_converged_at is None else self.alone_converged_at,
            alone_capped=self.alone_capped,
            incumbent_reset_at=-1 if self.incumbent_reset_at is None else self.incumbent_reset_at,
            joint_capped=self.joint_capped,
            converged_at=-1 if self.converged_at is None else self.converged_at,
            policy_changes_seen=self.policy_changes_seen,
            wall_time=json.dumps(self.wall_time),
        )

    def load_entry_checkpoint(self, path: str, *, chunk_size: int) -> None:
        with np.load(path, allow_pickle=False) as d:
            checks = {
                "entry_schema_version": (int(d["entry_schema_version"]), ENTRY_SCHEMA_VERSION),
                "entry_cell_key": (str(d["entry_cell_key"]), self.entry_cell.key()),
                "session_index": (int(d["session_index"]), self.session_index),
                "experiment_seed": (int(d["experiment_seed"]), self.streams.experiment_seed),
                "chunk_size": (int(d["chunk_size"]), chunk_size),
            }
            bad = [f"{k}={a!r}, expected {b!r}" for k, (a, b) in checks.items() if a != b]
            if bad:
                raise ValueError("checkpoint identity mismatch: " + "; ".join(bad))
            st = self.state
            for name in ("Q", "visits", "policy", "cursor", "hist", "stats"):
                value = np.array(d[name])
                if value.shape != getattr(st, name).shape:
                    raise ValueError(f"checkpoint {name} has shape {value.shape}")
                getattr(st, name)[...] = value
            self.entry_period[...] = d["entry_period"]
            self.streams.restore(json.loads(str(d["rng"])))
            self.phase = str(d["phase"])
            a = int(d["alone_converged_at"])
            self.alone_converged_at = None if a < 0 else a
            self.alone_capped = bool(d["alone_capped"])
            r = int(d["incumbent_reset_at"]) if "incumbent_reset_at" in d.files else -1
            self.incumbent_reset_at = None if r < 0 else r
            self.joint_capped = bool(d["joint_capped"]) if "joint_capped" in d.files else False
            c = int(d["converged_at"])
            self.converged_at = None if c < 0 else c
            self.policy_changes_seen = int(d["policy_changes_seen"])
            self.wall_time = {k: float(v) for k, v in json.loads(str(d["wall_time"])).items()}
