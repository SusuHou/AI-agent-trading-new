"""Run or resume one entry-timing session. / 运行或续跑一个进入时机 session。

    python -m dgj.experiments.run_entry_cli --out DIR --session K --seed S \
        --noise-std 0.1 --entry-period 46000000          # clock entry (0 = simultaneous control)
        --entry-on-convergence --alone-cap 2000000000    # stage 3
        [--stop-after-alone]                             # measure-first run: stop when the incumbent converges

Exit codes: 0 done (result written, or alone phase finished with --stop-after-alone);
75 safe pause (checkpoint written, resubmit); 2 error.
Outputs in DIR: cell.json, ckpt_K.npz, progress_K.json, session_K.json (final result).
"""

import argparse
import json
import math
import os
import signal
import sys
import time

from dgj.config import ExperimentCell, PaperParameters
from dgj.game.entry_session import EntryCell, EntrySession
from dgj.metrics import collusion

PAUSE = 75
_stop = False


def _on_signal(_signum, _frame) -> None:
    global _stop
    _stop = True


def _write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def clock_entry_period(epsilon: float, beta: float, num_values: int) -> int:
    """Period at which eps = exp(-beta t/n_v) reaches ``epsilon`` (values equally likely)."""
    return int(round(num_values * math.log(1.0 / epsilon) / beta))


def build_cell(a: argparse.Namespace) -> EntryCell:
    params = PaperParameters(
        num_speculators=2, noise_std=a.noise_std, exploration_decay=a.beta,
        learning_rate=a.alpha, discount_factor=a.rho, investor_slope=a.xi,
        convergence_periods=a.convergence_periods, measurement_periods=a.measurement_periods,
    )
    base = ExperimentCell(parameters=params, label=a.label, prehistory="nash", price_grid=a.price_grid)
    return EntryCell(
        base=base, action_grid=a.action_grid, exploration_floor=a.floor,
        entry_period=0 if a.entry_on_convergence else a.entry_period,
        entry_on_convergence=a.entry_on_convergence, alone_cap=a.alone_cap if a.entry_on_convergence else 0,
        incumbent_init=a.incumbent_init, lagged_price=not a.no_lagged_price,
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--label", default="entry")
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=5e-7)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--rho", type=float, default=0.95)
    p.add_argument("--xi", type=float, default=500.0)
    p.add_argument("--price-grid", default="per_value", choices=["per_value", "global"])
    p.add_argument("--action-grid", default="wide", choices=["wide", "dou"])
    p.add_argument("--floor", type=float, default=0.01)
    p.add_argument("--entry-period", type=int, default=0)
    p.add_argument("--entry-on-convergence", action="store_true")
    p.add_argument("--alone-cap", type=int, default=0)
    p.add_argument("--stop-after-alone", action="store_true")
    p.add_argument("--incumbent-init", default="dou", choices=["dou", "lone"], help="diagnostic T1")
    p.add_argument("--no-lagged-price", action="store_true", help="diagnostic T2")
    p.add_argument("--convergence-periods", type=int, default=1_000_000)
    p.add_argument("--measurement-periods", type=int, default=100_000)
    p.add_argument("--work-periods", type=int, default=5_000_000_000)
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument("--checkpoint-every", type=int, default=50, help="chunks between checkpoints")
    return p


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    for name in ("work_periods", "alone_cap", "entry_period"):
        if getattr(a, name) % a.chunk_size:
            print(f"ERROR: --{name.replace('_', '-')} must be a multiple of --chunk-size", file=sys.stderr)
            return 2
    if a.stop_after_alone and not a.entry_on_convergence:
        print("ERROR: --stop-after-alone needs --entry-on-convergence", file=sys.stderr)
        return 2
    cell = build_cell(a)
    os.makedirs(a.out, exist_ok=True)
    cell_path = os.path.join(a.out, "cell.json")
    expected = {"entry_cell": cell.to_dict(), "entry_cell_key": cell.key(), "seed": a.seed, "chunk_size": a.chunk_size}
    if os.path.exists(cell_path):
        with open(cell_path, encoding="utf-8") as f:
            if json.load(f) != expected:
                print(f"ERROR: {cell_path} describes a different cell or seed", file=sys.stderr)
                return 2
    else:
        _write_json(cell_path, expected)

    tag = f"{a.session:04d}"
    result_path = os.path.join(a.out, f"session_{tag}.json")
    ckpt_path = os.path.join(a.out, f"ckpt_{tag}.npz")
    progress_path = os.path.join(a.out, f"progress_{tag}.json")
    if os.path.exists(result_path):
        print(f"session {a.session}: result exists")
        return 0

    s = EntrySession(cell, a.session, a.seed)
    if os.path.exists(ckpt_path):
        s.load_entry_checkpoint(ckpt_path, chunk_size=a.chunk_size)
        print(f"session {a.session}: resumed at {s.periods_completed:,} periods, phase {s.phase}", flush=True)
    if a.stop_after_alone and s.phase != "alone":
        _write_json(progress_path, {"status": "alone_done", **s.entry_manifest()})
        print(f"session {a.session}: alone phase already finished")
        return 0

    signal.signal(signal.SIGTERM, _on_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_signal)

    end = s.periods_completed + a.work_periods
    chunks = 0
    t0 = time.time()
    while s.phase in ("alone", "training") and s.periods_completed < end and not _stop:
        before = s.phase
        s.advance(a.chunk_size, chunk_size=a.chunk_size)
        chunks += 1
        if before == "alone" and s.phase != "alone":
            print(f"session {a.session}: entrant enters at {s.periods_completed:,}"
                  f" ({'capped' if s.alone_capped else 'incumbent converged'})", flush=True)
            if a.stop_after_alone:
                s.save_entry_checkpoint(ckpt_path, chunk_size=a.chunk_size)
                _write_json(progress_path, {"status": "alone_done", "incumbent_play": s.incumbent_play(),
                                            **s.entry_manifest()})
                return 0
        if chunks % a.checkpoint_every == 0:
            s.save_entry_checkpoint(ckpt_path, chunk_size=a.chunk_size)
            _write_json(progress_path, {"status": "running", "rate_periods_per_s": chunks * a.chunk_size / max(time.time() - t0, 1e-9), **s.entry_manifest()})

    if s.phase in ("alone", "training"):
        s.save_entry_checkpoint(ckpt_path, chunk_size=a.chunk_size)
        _write_json(progress_path, {"status": "paused", **s.entry_manifest()})
        print(f"session {a.session}: paused at {s.periods_completed:,} ({s.phase})", flush=True)
        return PAUSE

    rows = s.measure()
    res = collusion.compute(rows, s.grids, 2, s.p.value_mean)
    payload = {
        "status": "complete",
        **s.entry_manifest(),
        "delta_c": res.delta_c,
        "delta_by_agent": list(res.delta_by_agent),
        "mean_actual_profit": list(res.mean_actual_profit),
        "mean_nash_profit": res.mean_nash_profit,
        "mean_cartel_profit": res.mean_cartel_profit,
        "mean_order_multiplier_by_agent": [
            float((rows[:, 6 + i] * (rows[:, 0] - s.p.value_mean)).sum() / ((rows[:, 0] - s.p.value_mean) ** 2).sum())
            for i in range(2)
        ],
    }
    _write_json(result_path, payload)
    _write_json(progress_path, {"status": "complete", **s.entry_manifest()})
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"session {a.session}: complete, delta_c={res.delta_c:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
