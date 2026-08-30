"""Impulse-response experiment and mechanism classification (paper §5, OA §4.5). / 脉冲响应与机制分类。

Protocol (paper Fig. 3, OA 4.5) as implemented here:
    * fork the converged market (frozen policy, live rolling market maker) into a
      control branch and a treatment branch that share the same u_t and v_{t+1}
      draws (common random numbers — replication choice);
    * local periods t = 1..4 are simulated from the fork; at t = 3 the treatment
      branch receives an ADVERSE noise shock with the sign of (v_3 - v_bar),
      added to the ordinary u_3, calibrated so the average oriented price rises
      by 1.2% of its long-run mean (replication choice: one common magnitude
      |u| = target * E[p_tilde] / E[lambda_hat] from the measurement window);
    * the t = 4 oriented order response of each speculator classifies the session.
      The paper normalizes by the long-run mean E[x_tilde]; with 10,000 random v
      paths that estimate carries ~1% sampling noise, far above the thresholds, so
      we classify on the response relative to the common-random-number CONTROL
      branch (identical draws, no shock) and report the long-run version too:
          both responses >  5e-4  -> price_trigger
          both |responses| < 5e-5 -> over_pruning
          otherwise               -> unclassified
    * 10,000 paths per session (PAPER_PATHS), averaged.

Orientation: p_tilde = (p - v_bar) * sign(v - v_bar), x_tilde = x * sign(v - v_bar).
"""

from dataclasses import dataclass

import numpy as np

from dgj.game import protocol
from dgj.game.session import Session, SessionState
from dgj.game.shocks import Shocks
from dgj.players.market_maker.adaptive import C_V_CUR

PAPER_PATHS = 10_000
PAPER_TARGET_DEVIATION = 0.012
PAPER_SHOCK_PERIOD = 3          # local period index (1-based)
PAPER_RESPONSE_PERIOD = 4
LOCAL_PERIODS = 4
LOW_THRESHOLD = 5e-5
HIGH_THRESHOLD = 5e-4


# ---------------------------------------------------------------------------
# long-run baseline from the measurement rows (Figure-3 denominators)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LongRunBaseline:
    mean_oriented_price: float
    mean_oriented_order: tuple[float, ...]
    mean_profit: tuple[float, ...]
    mean_lambda: float


def long_run_baseline(rows: np.ndarray, number_of_speculators: int, value_mean: float) -> LongRunBaseline:
    sign = np.sign(rows[:, protocol.COL_V] - value_mean)
    oriented_price = (rows[:, protocol.COL_P] - value_mean) * sign
    I = number_of_speculators
    return LongRunBaseline(
        mean_oriented_price=float(oriented_price.mean()),
        mean_oriented_order=tuple(float((rows[:, protocol.col_x(i)] * sign).mean()) for i in range(I)),
        mean_profit=tuple(float(rows[:, protocol.col_pi(i, I)].mean()) for i in range(I)),
        mean_lambda=float(rows[:, protocol.COL_LAM].mean()),
    )


def calibrate_shock(baseline: LongRunBaseline, target: float = PAPER_TARGET_DEVIATION) -> float:
    """|u_shock| such that lambda * |u| = target * E[p_tilde]."""
    if baseline.mean_oriented_price <= 0 or baseline.mean_lambda <= 0:
        raise ArithmeticError("shock calibration needs positive E[p_tilde] and E[lambda]")
    return target * baseline.mean_oriented_price / baseline.mean_lambda


# ---------------------------------------------------------------------------
# forking a converged market
# ---------------------------------------------------------------------------
@dataclass
class MarketFork:
    """The mutable part of a converged market: cursor, history, OLS stats. / 可变部分。"""

    cursor: np.ndarray
    hist: np.ndarray
    stats: np.ndarray


def take_fork(session: Session) -> MarketFork:
    if session.phase not in ("converged", "measurement", "complete"):
        raise RuntimeError("fork only a converged session (frozen policy)")
    st = session.state
    return MarketFork(st.cursor.copy(), st.hist.copy(), st.stats.copy())


def _branch_state(session: Session, fork: MarketFork) -> SessionState:
    st = session.state
    # Q, policy, visits are never written in measurement mode -> shared, not copied
    return SessionState(Q=st.Q, visits=st.visits, policy=st.policy,
                        cursor=fork.cursor.copy(), hist=fork.hist.copy(), stats=fork.stats.copy())


def _run_branch(session: Session, fork: MarketFork, value_index: np.ndarray, noise: np.ndarray) -> np.ndarray:
    I = session.p.num_speculators
    state = _branch_state(session, fork)
    shocks = Shocks(value_index=value_index, noise=noise,
                    mode=np.zeros((I, LOCAL_PERIODS)), action=np.zeros((I, LOCAL_PERIODS), dtype=np.int64))
    out = np.empty((LOCAL_PERIODS, protocol.row_width(I)))
    saved = session.state
    session.state = state                       # _kernel_args reads session.state
    try:
        protocol.run_periods(*session._kernel_args(shocks, LOCAL_PERIODS, False, out))
    finally:
        session.state = saved
    return out


# ---------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IRFResult:
    paths: int
    shock_magnitude: float
    # mean oriented price by local period, control and treatment, and normalized deviation
    control_oriented_price: np.ndarray        # (4,)
    treatment_oriented_price: np.ndarray      # (4,)
    normalized_price_deviation: np.ndarray    # (4,)  (treatment - baseline) / baseline
    # oriented orders at t = 4
    control_t4_oriented_order: tuple[float, ...]
    treatment_t4_oriented_order: tuple[float, ...]
    response_vs_long_run: tuple[float, ...]   # (treatment_t4 - E[x_tilde]) / E[x_tilde]
    response_vs_control: tuple[float, ...]    # (treatment_t4 - control_t4) / control_t4
    mechanism: str


def classify(responses, low: float = LOW_THRESHOLD, high: float = HIGH_THRESHOLD) -> str:
    r = tuple(float(x) for x in responses)
    if all(x > high for x in r):
        return "price_trigger"
    if all(abs(x) < low for x in r):
        return "over_pruning"
    return "unclassified"


def run_irf(session: Session, fork: MarketFork, baseline: LongRunBaseline, *,
            paths: int = PAPER_PATHS, shock_magnitude: float | None = None, irf_seed: int = 7) -> IRFResult:
    p, g = session.p, session.grids
    I = p.num_speculators
    if shock_magnitude is None:
        shock_magnitude = calibrate_shock(baseline)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        entropy=irf_seed, spawn_key=(session.cell.key_uint32(), session.session_index, 99))))
    u_all = rng.normal(0.0, p.noise_std, size=(paths, LOCAL_PERIODS))
    v_all = rng.integers(p.num_value_points, size=(paths, LOCAL_PERIODS), dtype=np.int64)

    c_price = np.zeros(LOCAL_PERIODS)
    t_price = np.zeros(LOCAL_PERIODS)
    c_x4 = np.zeros(I)
    t_x4 = np.zeros(I)
    v_first = g.value_grid[int(fork.cursor[C_V_CUR])]      # value at local t = 1
    for k in range(paths):
        u = u_all[k]
        vidx = v_all[k]
        # value at local t=3 is the draw made at the end of t=2
        v3 = g.value_grid[vidx[PAPER_SHOCK_PERIOD - 2]]
        sign3 = 1.0 if v3 > p.value_mean else -1.0
        u_treat = u.copy()
        u_treat[PAPER_SHOCK_PERIOD - 1] += sign3 * shock_magnitude

        control = _run_branch(session, fork, vidx, u)
        treatment = _run_branch(session, fork, vidx, u_treat)
        for t in range(LOCAL_PERIODS):
            s = 1.0 if control[t, protocol.COL_V] > p.value_mean else -1.0
            c_price[t] += (control[t, protocol.COL_P] - p.value_mean) * s
            t_price[t] += (treatment[t, protocol.COL_P] - p.value_mean) * s
        s4 = 1.0 if control[PAPER_RESPONSE_PERIOD - 1, protocol.COL_V] > p.value_mean else -1.0
        for i in range(I):
            c_x4[i] += control[PAPER_RESPONSE_PERIOD - 1, protocol.col_x(i)] * s4
            t_x4[i] += treatment[PAPER_RESPONSE_PERIOD - 1, protocol.col_x(i)] * s4
    c_price /= paths
    t_price /= paths
    c_x4 /= paths
    t_x4 /= paths
    base_x = np.array(baseline.mean_oriented_order)
    resp_long = tuple(float(x) for x in (t_x4 - base_x) / base_x)
    resp_ctrl = tuple(float(x) for x in (t_x4 - c_x4) / c_x4)
    return IRFResult(
        paths=paths, shock_magnitude=float(shock_magnitude),
        control_oriented_price=c_price, treatment_oriented_price=t_price,
        normalized_price_deviation=(t_price - baseline.mean_oriented_price) / baseline.mean_oriented_price,
        control_t4_oriented_order=tuple(float(x) for x in c_x4),
        treatment_t4_oriented_order=tuple(float(x) for x in t_x4),
        response_vs_long_run=resp_long, response_vs_control=resp_ctrl,
        mechanism=classify(resp_ctrl),      # control-based: common random numbers cancel the |v-v_bar| sampling noise
    )
