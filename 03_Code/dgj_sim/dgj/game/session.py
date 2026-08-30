"""One simulation session: build -> train to convergence -> measure. / 一个 session。

Phases / 阶段:
    TRAINING     kernel with learning=True, in chunks, until the joint greedy
                 policy is unchanged for ``convergence_periods`` periods (T_c)
    MEASUREMENT  kernel with learning=False for ``measurement_periods`` periods;
                 Q, visits, policy frozen; environment and market maker continue
    COMPLETE     rows + manifest available

Everything mutable lives in a handful of arrays (SessionState) so the kernel
never sees a Python object and a checkpoint is one np.savez.
"""

from dataclasses import dataclass, field
import json
import os
import tempfile
import time
from typing import Any

import numpy as np

from dgj.config import ExperimentCell
from dgj.environment import fundamental
from dgj.game import protocol
from dgj.game.shocks import ShockStreams
from dgj.provenance import (
    SCIENTIFIC_ENGINE_VERSION,
    scientific_runtime_identity,
    scientific_source_fingerprint,
)
from dgj.players import benchmarks
from dgj.players.market_maker import adaptive, prehistory
from dgj.players.market_maker.adaptive import (
    C_APPENDS, C_P_IDX, C_STREAK, C_T, C_V_CUR, C_V_LAG, CURSOR_SIZE,
)
from dgj.players.speculator import action_space, policy as policy_module, q_learning
from dgj.players.speculator.state_space import number_of_states


CHECKPOINT_SCHEMA_VERSION = 1


def atomic_savez_compressed(path: str, **arrays: Any) -> None:
    """Write one ``.npz`` beside its target, then publish it atomically.

    A Slurm job can be stopped while NumPy is compressing a file. Writing
    directly to the final path could then leave a corrupt file that merely
    *looks* like a checkpoint.  The temporary file is created in the same
    directory so ``os.replace`` is one atomic filesystem operation.

    / 先在目标旁边写完整临时文件，再原子替换。这样超算任务中断时，不会把
    半截文件伪装成有效 checkpoint。
    """
    target = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    temporary_path: str | None = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            dir=directory,
            prefix=f".{os.path.basename(target)}.",
            # NumPy appends ".npz" to string paths that do not already end in
            # it. Keeping that suffix here ensures we replace the file we made.
            suffix=".tmp.npz",
        )
        os.close(descriptor)
        np.savez_compressed(temporary_path, **arrays)
        # Windows requires a writable descriptor for ``fsync``.
        with open(temporary_path, "rb+") as temporary:
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass


@dataclass
class Grids:
    """Everything fixed once the cell is fixed. / 参数固定后就固定的东西。"""

    value_grid: np.ndarray
    discrete_value_std: float
    nash: benchmarks.Benchmark
    cartel: benchmarks.Benchmark
    multipliers: np.ndarray
    orders: np.ndarray            # (n_v, n_x)
    price_grid: np.ndarray        # (n_v, n_p): row k is P(v_k); identical rows when global
    initial_q_block: np.ndarray   # (n_v, n_x)


def build_grids(cell: ExperimentCell) -> Grids:
    p = cell.parameters
    V = fundamental.value_grid(p.value_mean, p.value_std, p.num_value_points)
    sigma_v_hat = fundamental.discrete_std(V, p.value_mean)
    common = (p.num_speculators, p.noise_std, sigma_v_hat, p.investor_slope, p.pricing_error_weight)
    nash = benchmarks.solve("nash", *common)
    cartel = benchmarks.solve("cartel", *common)
    mult = action_space.multipliers(nash.intensity, cartel.intensity, p.grid_widening, p.num_action_points)
    orders = action_space.orders_table(V, p.value_mean, mult)

    # price grid, paper 4.2: p_H/L = v_bar + lambda^N (I max/min{x^M, x^N} +/- 1.96 sigma_u), then iota widening.
    # A5: max/min{x^M, x^N} is evaluated at each v (as for X), giving P(v) with one action step ~ one
    # price step (footnote 25). "global" takes max/min over the whole grid instead (the steps/ reading).
    P = np.empty((p.num_value_points, p.num_price_points))
    if cell.price_grid == "per_value":
        for k, v in enumerate(V):
            xs = (nash.order(float(v), p.value_mean), cartel.order(float(v), p.value_mean))
            lower = p.value_mean + nash.price_impact * (p.num_speculators * min(xs) - 1.96 * p.noise_std)
            upper = p.value_mean + nash.price_impact * (p.num_speculators * max(xs) + 1.96 * p.noise_std)
            width = upper - lower
            P[k] = np.linspace(lower - p.grid_widening * width, upper + p.grid_widening * width, p.num_price_points)
    elif cell.price_grid == "global":
        bench_orders = np.concatenate([nash.order(V, p.value_mean), cartel.order(V, p.value_mean)])
        lower = p.value_mean + nash.price_impact * (p.num_speculators * bench_orders.min() - 1.96 * p.noise_std)
        upper = p.value_mean + nash.price_impact * (p.num_speculators * bench_orders.max() + 1.96 * p.noise_std)
        width = upper - lower
        P[:] = np.linspace(lower - p.grid_widening * width, upper + p.grid_widening * width, p.num_price_points)
    else:
        raise ValueError("price_grid must be 'per_value' or 'global'")

    q_block = q_learning.initial_q_block(V, p.value_mean, orders, p.num_speculators, nash.price_impact, p.discount_factor)
    return Grids(V, sigma_v_hat, nash, cartel, mult, orders, P, q_block)


@dataclass
class SessionState:
    """The mutable arrays the kernel works on. / 内核操作的可变数组。"""

    Q: np.ndarray          # (I, S, n_x)
    visits: np.ndarray     # (n_v,) int64
    policy: np.ndarray     # (I, S) int64 bit masks
    cursor: np.ndarray     # (CURSOR_SIZE,) int64
    hist: np.ndarray       # (T_m, 4)
    stats: np.ndarray      # (9,)


@dataclass
class SessionResult:
    cell_key: str
    session_index: int
    converged_at: int               # T_c: number of training periods completed
    policy_changes_seen: int
    measurement_rows: np.ndarray    # (T, row_width)
    manifest: dict = field(default_factory=dict)


class Session:
    def __init__(
        self,
        cell: ExperimentCell,
        session_index: int,
        experiment_seed: int,
        *,
        grids: Grids | None = None,
        prehistory_rows: np.ndarray | None = None,
    ) -> None:
        self.cell = cell
        self.p = cell.parameters
        self.session_index = int(session_index)
        self.grids = grids if grids is not None else build_grids(cell)
        g = self.grids
        I = self.p.num_speculators
        S = number_of_states(self.p.num_price_points, self.p.num_value_points)

        # --- speculators
        Q = q_learning.initial_q_table(g.initial_q_block, I, self.p.num_price_points)
        policy = policy_module.initial_policy(Q)
        visits = np.zeros(self.p.num_value_points, dtype=np.int64)

        # --- market maker with D_0 (A3)
        hist, stats = adaptive.new_history(self.p.market_maker_window)
        cursor = np.zeros(CURSOR_SIZE, dtype=np.int64)
        if prehistory_rows is None:
            bench = g.nash if cell.prehistory == "nash" else g.cartel
            prehistory_rows = prehistory.build_rows(
                bench, g.value_grid, self.p.value_mean, self.p.investor_slope,
                self.p.noise_std, I, self.p.market_maker_window,
            )
        adaptive.preload(hist, stats, cursor, self.p.market_maker_window, np.ascontiguousarray(prehistory_rows, dtype=float))
        if int(cursor[adaptive.C_NROWS]) != self.p.market_maker_window:
            raise ValueError("D_0 must contain exactly T_m rows")
        cursor[C_APPENDS] = 0                 # count only live appends from here

        # --- randomness and initial state
        self.streams = ShockStreams(experiment_seed, cell, session_index)
        p_idx, v_lag, v_cur = self.streams.initial_state(self.p.num_price_points, self.p.num_value_points)
        cursor[C_P_IDX], cursor[C_V_LAG], cursor[C_V_CUR] = p_idx, v_lag, v_cur

        self.state = SessionState(Q=Q, visits=visits, policy=policy, cursor=cursor, hist=hist, stats=stats)
        self.phase = "training"
        self.converged_at: int | None = None
        self.policy_changes_seen = 0
        self._streak_before_chunk = 0
        self.wall_time = {"training": 0.0, "measurement": 0.0}

    # ------------------------------------------------------------------
    @property
    def periods_completed(self) -> int:
        return int(self.state.cursor[C_T])

    def _kernel_args(self, shocks, n, learning, out_rows):
        st = self.state
        g = self.grids
        p = self.p
        return (
            st.Q, st.visits, st.policy, st.cursor, st.hist, st.stats, g.orders, g.value_grid, g.price_grid,
            shocks.value_index, shocks.noise, shocks.mode, shocks.action,
            p.learning_rate, p.discount_factor, p.exploration_decay, p.pricing_error_weight,
            p.investor_slope, p.value_mean, p.market_maker_window,
            n, learning, p.convergence_periods, out_rows,
        )

    _EMPTY_ROWS = np.zeros((0, 1))

    def train(self, *, chunk_size: int = 1_000_000, max_periods: int | None = None) -> bool:
        """Run training chunks until convergence (True) or max_periods (False)."""
        if self.phase != "training":
            raise RuntimeError("session is not in the training phase")
        t0 = time.perf_counter()
        while True:
            if max_periods is not None:
                remaining = max_periods - self.periods_completed
                if remaining <= 0:
                    self.wall_time["training"] += time.perf_counter() - t0
                    return False
                n = min(chunk_size, remaining)
            else:
                n = chunk_size
            shocks = self.streams.draw(n, self.p.num_value_points)
            streak_before = int(self.state.cursor[C_STREAK])
            ran, converged = protocol.run_periods(*self._kernel_args(shocks, n, True, self._EMPTY_ROWS))
            # unused tail of the chunk is discarded on convergence; the stream simply moves on
            streak_after = int(self.state.cursor[C_STREAK])
            if streak_after < streak_before + ran:      # at least one reset happened in this chunk
                self.policy_changes_seen += 1
            if converged:
                self.converged_at = self.periods_completed
                self.phase = "converged"
                self.wall_time["training"] += time.perf_counter() - t0
                return True

    def measure(self, n_periods: int | None = None, *, chunk_size: int = 100_000) -> np.ndarray:
        """Frozen-policy measurement; returns (T, row_width) rows."""
        if self.phase != "converged":
            raise RuntimeError("measure only after convergence")
        T = self.p.measurement_periods if n_periods is None else int(n_periods)
        width = protocol.row_width(self.p.num_speculators)
        rows = np.empty((T, width))
        q_before = self.state.Q.copy() if T <= 1_000_000 else None
        visits_before = self.state.visits.copy()
        policy_before = self.state.policy.copy()
        self.phase = "measurement"
        t0 = time.perf_counter()
        done = 0
        while done < T:
            n = min(chunk_size, T - done)
            shocks = self.streams.draw(n, self.p.num_value_points)
            out = np.empty((n, width))
            protocol.run_periods(*self._kernel_args(shocks, n, False, out))
            rows[done:done + n] = out
            done += n
        self.wall_time["measurement"] += time.perf_counter() - t0
        # invariants of the frozen phase / 冻结阶段不变量
        if q_before is not None and not np.array_equal(q_before, self.state.Q):
            raise RuntimeError("Q changed during measurement")
        if not np.array_equal(visits_before, self.state.visits) or not np.array_equal(policy_before, self.state.policy):
            raise RuntimeError("visits or policy changed during measurement")
        self.phase = "complete"
        self.measurement_rows = rows
        return rows

    def run(self, *, chunk_size: int = 1_000_000, max_periods: int | None = None) -> SessionResult:
        if not self.train(chunk_size=chunk_size, max_periods=max_periods):
            raise TimeoutError(f"no convergence within {max_periods} periods")
        rows = self.measure()
        return SessionResult(
            cell_key=self.cell.key(),
            session_index=self.session_index,
            converged_at=int(self.converged_at),
            policy_changes_seen=self.policy_changes_seen,
            measurement_rows=rows,
            manifest=self.manifest(),
        )

    # ------------------------------------------------------------------
    def manifest(self) -> dict[str, Any]:
        g = self.grids
        return {
            "cell": self.cell.to_dict(),
            "cell_key": self.cell.key(),
            "shocks": self.streams.manifest(),
            "discrete_value_std": g.discrete_value_std,
            "nash": {"chi": g.nash.intensity, "lambda": g.nash.price_impact},
            "cartel": {"chi": g.cartel.intensity, "lambda": g.cartel.price_impact},
            "converged_at": self.converged_at,
            "periods_completed": self.periods_completed,
            "wall_time_seconds": dict(self.wall_time),
        }

    def save_checkpoint(self, path: str, *, training_chunk_size: int | None = None) -> None:
        """Save a self-identifying, atomic training checkpoint.

        ``training_chunk_size`` is part of reproducibility because an unused
        random tail is discarded when convergence occurs inside a chunk.
        / chunk size 会影响收敛时丢弃多少预抽随机数，因此也必须随 checkpoint
        保存并在续跑时核对。
        """
        if training_chunk_size is not None and training_chunk_size < 1:
            raise ValueError("training_chunk_size must be positive")
        st = self.state
        atomic_savez_compressed(
            path,
            checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
            scientific_engine_version=SCIENTIFIC_ENGINE_VERSION,
            scientific_source_fingerprint=scientific_source_fingerprint(),
            scientific_runtime_identity=json.dumps(
                scientific_runtime_identity(), sort_keys=True
            ),
            cell_key=self.cell.key(),
            session_index=self.session_index,
            experiment_seed=self.streams.experiment_seed,
            training_chunk_size=-1 if training_chunk_size is None else training_chunk_size,
            Q=st.Q,
            visits=st.visits,
            policy=st.policy,
            cursor=st.cursor,
            hist=st.hist,
            stats=st.stats,
            rng=json.dumps(self.streams.state()), phase=self.phase,
            converged_at=-1 if self.converged_at is None else self.converged_at,
            policy_changes_seen=self.policy_changes_seen,
            wall_time=json.dumps(self.wall_time),
        )

    def load_checkpoint(self, path: str, *, expected_training_chunk_size: int | None = None) -> None:
        """Validate checkpoint identity completely before restoring its state.

        Legacy checkpoints without identity metadata are rejected rather than
        silently mixed with another experiment. / 旧 checkpoint 若没有 cell、
        seed、session 等身份信息，会明确拒绝，而不是偷偷混入本次实验。
        """
        if expected_training_chunk_size is not None and expected_training_chunk_size < 1:
            raise ValueError("expected_training_chunk_size must be positive")
        st = self.state
        required = {
            "checkpoint_schema_version", "scientific_engine_version",
            "scientific_source_fingerprint", "scientific_runtime_identity",
            "cell_key", "session_index", "experiment_seed",
            "training_chunk_size", "Q", "visits", "policy", "cursor", "hist", "stats",
            "rng", "phase", "converged_at", "policy_changes_seen", "wall_time",
        }
        try:
            with np.load(path, allow_pickle=False) as data:
                missing = required.difference(data.files)
                if missing:
                    raise ValueError(
                        "legacy or incomplete checkpoint; missing fields: "
                        + ", ".join(sorted(missing))
                    )
                schema = int(data["checkpoint_schema_version"].item())
                if schema != CHECKPOINT_SCHEMA_VERSION:
                    raise ValueError(
                        f"unsupported checkpoint schema {schema}; expected {CHECKPOINT_SCHEMA_VERSION}"
                    )
                saved_engine_version = int(data["scientific_engine_version"].item())
                saved_source_fingerprint = str(
                    data["scientific_source_fingerprint"].item()
                )
                saved_runtime_identity = json.loads(
                    str(data["scientific_runtime_identity"].item())
                )
                saved_cell_key = str(data["cell_key"].item())
                saved_session_index = int(data["session_index"].item())
                saved_seed = int(data["experiment_seed"].item())
                saved_chunk_size = int(data["training_chunk_size"].item())
                identity_errors = []
                if saved_engine_version != SCIENTIFIC_ENGINE_VERSION:
                    identity_errors.append(
                        f"scientific_engine_version={saved_engine_version}, "
                        f"expected {SCIENTIFIC_ENGINE_VERSION}"
                    )
                current_fingerprint = scientific_source_fingerprint()
                if saved_source_fingerprint != current_fingerprint:
                    identity_errors.append(
                        "scientific_source_fingerprint differs from the current code"
                    )
                if saved_runtime_identity != scientific_runtime_identity():
                    identity_errors.append(
                        "scientific_runtime_identity differs from the current environment"
                    )
                if saved_cell_key != self.cell.key():
                    identity_errors.append(f"cell_key={saved_cell_key}, expected {self.cell.key()}")
                if saved_session_index != self.session_index:
                    identity_errors.append(
                        f"session_index={saved_session_index}, expected {self.session_index}"
                    )
                if saved_seed != self.streams.experiment_seed:
                    identity_errors.append(
                        f"experiment_seed={saved_seed}, expected {self.streams.experiment_seed}"
                    )
                if (
                    expected_training_chunk_size is not None
                    and saved_chunk_size != expected_training_chunk_size
                ):
                    identity_errors.append(
                        f"training_chunk_size={saved_chunk_size}, "
                        f"expected {expected_training_chunk_size}"
                    )
                if identity_errors:
                    raise ValueError("checkpoint identity mismatch: " + "; ".join(identity_errors))

                restored_arrays = {}
                for name in ("Q", "visits", "policy", "cursor", "hist", "stats"):
                    value = np.array(data[name], copy=True)
                    target = getattr(st, name)
                    if value.shape != target.shape or value.dtype != target.dtype:
                        raise ValueError(
                            f"checkpoint {name} has shape/dtype {value.shape}/{value.dtype}; "
                            f"expected {target.shape}/{target.dtype}"
                        )
                    restored_arrays[name] = value

                rng_state = json.loads(str(data["rng"].item()))
                if not isinstance(rng_state, list) or len(rng_state) != len(self.streams.state()):
                    raise ValueError("checkpoint RNG stream count does not match this session")
                phase = str(data["phase"].item())
                c = int(data["converged_at"].item())
                if phase not in {"training", "converged"}:
                    raise ValueError(f"checkpoint phase {phase!r} cannot be resumed")
                if (phase == "training" and c >= 0) or (phase == "converged" and c < 0):
                    raise ValueError("checkpoint phase and converged_at disagree")
                policy_changes_seen = int(data["policy_changes_seen"].item())
                wall_time = json.loads(str(data["wall_time"].item()))
                if (
                    not isinstance(wall_time, dict)
                    or set(wall_time) != {"training", "measurement"}
                    or any(float(value) < 0 for value in wall_time.values())
                ):
                    raise ValueError("checkpoint wall_time is invalid")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load checkpoint {path!r}: {error}") from error

        # Commit only after every field above has passed validation.
        for name, value in restored_arrays.items():
            getattr(st, name)[...] = value
        self.streams.restore(rng_state)
        self.phase = phase
        self.converged_at = None if c < 0 else c
        self.policy_changes_seen = policy_changes_seen
        self.wall_time = {key: float(value) for key, value in wall_time.items()}
