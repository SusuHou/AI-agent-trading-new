# dgj_sim — compiled-kernel replication of Dou, Goldstein & Ji (2025)

A rebuild of `../vibe_replication/steps` as a program that can actually run the
paper's experiments (10⁷–10¹⁰ periods per session, 1,000 sessions per cell).
The economics are the same; the difference is structure and speed.

把 `../vibe_replication/steps` 重组成一个真的能跑论文实验规模的程序。经济学完全相同，
不同的是结构和速度。

```
py -3 -m pip install numba          # optional but ~200x faster; pure-Python fallback otherwise
py -3 -m unittest discover -s tests -v
py -3 -m dgj.experiments.run_cell --sessions 4 --workers 4 --max-periods 20000000 --out outputs/low_noise
```

## The four rules and where they live / 四条规则在代码中的位置

| Rule | Enforced in |
|---|---|
| 1. Period order: choose → noise → price from old data → profits → append → Q update | `game/protocol.py::run_periods` (steps 1–6, in that order, nothing else) |
| 2. Market maker never prices period t with period-t data | `adaptive.quote` is called before `adaptive.observe`; `tests/test_protocol_order.py` proves the price equals the old-coefficient price and differs from the "cheating" price |
| 3. After convergence freeze Q / exploration / visit counts, maker keeps rolling | `run_periods(learning=False)` skips steps 1a and 6; `Session.measure` asserts Q/visits/policy unchanged and append count grew |
| 4. Independent, reproducible randomness per session | `game/shocks.py`: `SeedSequence(seed, spawn_key=(cell, session, stream))`, drawn in chunks outside the kernel |

## Layout = the paper's actors / 结构 = 论文的参与者

```
dgj/
├── config.py                   PaperParameters + ExperimentCell (labelled choices A2/A3/A4)
├── environment/                exogenous world — no decisions, no learning
│   ├── fundamental.py          v grid, sigma_v_hat                          (4.2, fn 24)
│   ├── noise_trader.py         u_t ~ N(0, sigma_u^2)
│   └── insensitive_investors.py z = -xi (p - v_bar)                          (3.2)
├── players/
│   ├── benchmarks.py           Nash / cartel: chi, lambda fixed point, pi^N pi^M, matched-path profit
│   ├── speculator/
│   │   ├── state_space.py      s = (p_{t-1}, v_{t-1}, v_t) -> int; price -> grid (A2)
│   │   ├── action_space.py     X(v): 15 orders between x^M and x^N
│   │   ├── q_learning.py       Q_0 (p.25), eps_t(v) (4.3), choose (2.6), update (2.4)+OA
│   │   └── policy.py           greedy argmax masks, tie rules, convergence streak
│   └── market_maker/
│       ├── adaptive.py         ring buffer D_t + O(1) OLS (4.1), quote (4.2), observe
│       ├── theoretical.py      (3.4) and p = v_bar + lambda y
│       └── prehistory.py       D_0 (A3)
├── game/
│   ├── shocks.py               independent streams, chunked draws (A4)
│   ├── protocol.py             ONE period — the @njit kernel
│   └── session.py              grids -> state arrays -> train -> measure -> result / checkpoint
├── metrics/                    Delta^C (IA.4.1-3), chi_hat^C (IA.4.4), I^C (IA.4.5) from saved rows
└── experiments/run_cell.py     N sessions in a process pool, per-session .npz, summary.json
```

## Build order (separable steps) / 可拆分的构建步骤

Each step is independently testable; each depends only on the ones above it.

1. `config.py` — parameters and cell identity. Test: hash stable, invalid values rejected.
2. `environment/fundamental.py` — grid and σ̂_v = 0.938. Test: `test_environment.py`.
3. `players/benchmarks.py` — fixed point, identities, profits. Test: `test_benchmarks.py`.
4. `players/speculator/{state_space,action_space}.py` — encodings and grids. Test: round trip over all 3,100 states; nearest-grid vs bisect reference.
5. `players/market_maker/adaptive.py` — O(1) rolling OLS. Test: equals batch `lstsq` after every add/evict.
6. `players/market_maker/prehistory.py` — D₀. Test: maker recovers ξ̂₁=500, λ̂=λ^N from the rows alone.
7. `players/speculator/{q_learning,policy}.py` — Q₀, ε, choose, update, masks. Test: hand examples (0.5, 10.04825, 10.015).
8. `game/shocks.py` — streams. Test: chunk-size invariance; per-stream isolation.
9. `game/protocol.py` — the kernel. Test: `test_protocol_order.py` rules 1–2 on one period.
10. `game/session.py` — phases and checkpoints. Test: rules 3–4, checkpoint round trip.
11. `tests/test_parity_vs_steps.py` — 300 periods bit-equal (1e-9) with the readable `steps/` oracle.
12. `metrics/`, `experiments/` — offline post-processing and fan-out.

## How the pieces stick together / 如何拼在一起

```
ExperimentCell ──build_grids()──▶ Grids (V, P, X, chi/lambda, Q0 block)      immutable per cell
        │
        └──Session(cell, k, seed)──▶ SessionState (Q, visits, policy, cursor, hist, stats)   mutable arrays
                     │                       ▲
                     │  ShockStreams.draw(n) ─┘ chunks of (v', u, mode_i, action_i)
                     ▼
        protocol.run_periods(state arrays, shocks, params, learning=True)   ← the only hot loop
                     │  returns when policy unchanged for 1,000,000 periods → T_c
                     ▼
        protocol.run_periods(..., learning=False, out_rows)   100,000 frozen-policy rows
                     ▼
        session_k.npz  (rows + manifest)  ──▶ metrics.collusion / trading_policy ──▶ summary.json
```

Design rules that make this work:

* **Actors are pure functions on arrays.** `q_learning.update`, `adaptive.quote`, `demand`, … take
  arrays/scalars and return scalars. They are readable one at a time, testable one at a time, and
  `numba` inlines them into the kernel.
* **The kernel is the only place that knows the period order.** Nothing else calls actors in sequence.
* **All randomness enters as arrays.** The kernel has no RNG, so results do not depend on numba,
  threads, or execution order; a checkpoint is just the arrays plus the generator states.
* **Measurement rows are saved raw.** Every OA §4.1 metric is a function of the rows, so metrics can
  be added or corrected without re-simulating.
* **`steps/` is the oracle, not the program.** The parity test inherits every hand check already done there.

## What is still a replication choice (the paper is silent) / 论文未说明之处

* A2 continuous price → nearest grid point, clipped, ties to the lower point.
* A3 D₀ = balanced Nash-consistent synthetic history (also `prehistory="cartel"`); run an
  expanding-window sensitivity before reporting.
* A4 seeding scheme.
* Measurement window = the 100,000 periods after T_c with Q, ε and counters frozen.
* Exact-tie rules: uniform among tied actions during training exploitation; lowest index when frozen.

## Running on a cluster / 上超算

```
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt   # Python >= 3.10
python -m unittest discover -s tests            # must pass on the cluster's Python before submitting
OUT=outputs/low_noise_per_value CELL_ARGS="--noise-std 0.1 --price-grid per_value" sbatch hpc/submit_array.slurm
python hpc/aggregate_dir.py outputs/low_noise_per_value
```

* One array task = one session (`dgj.experiments.run_session_cli`). Each writes its own
  `session_<k>.npz`; a checkpoint `ckpt_<k>.npz` is saved every `--checkpoint-every` chunks and
  resumed automatically, so a walltime kill costs at most one interval — just resubmit.
* Set `--max-periods` (default 5B ≈ 1.5–3 h at ~1 µs/period). Sessions that hit the cap exit with
  code 3 and are marked `censored` in the manifest; `aggregate_dir.py` reports how many.
* `NUMBA_CACHE_DIR` must be node-local (the script sets it) — a shared-filesystem cache races.
* Memory per session ≈ 200 MB (Q table 0.7 MB; 1M-period shock chunks ≈ 50 MB).
* Suggested first campaign: the two baseline cells (σ_u = 0.1 and 100) × the two price-grid
  readings (`per_value`, `global`), 100 sessions each, before spending on 1,000-session cells.

## Where each `vibe_replication/steps` file went / 每个 step 的去向

| steps/ | dgj_sim | kept |
|---|---|---|
| 01, src/step01 | `environment/fundamental.py` | formula |
| 02 | `environment/noise_trader.py`, `game/shocks.py` | distribution |
| 03, 04, 05, ckpt A | `environment/insensitive_investors.py` + three lines inside `game/protocol.py` | formulas |
| 06, 07 | `players/market_maker/theoretical.py` | (3.4) only |
| 08, 09, 10, 11 | `players/benchmarks.py` | fixed point, chi/lambda, profits, matched-path profit |
| 12, 13, 14, 15 | `players/speculator/{action_space,state_space}.py`, `game/session.build_grids` | grids, encoding, A2; **P(v) per value (A5)** |
| 16, 17, 18, 19, 20, 21, ckpt B | `players/speculator/{q_learning,policy}.py` | Q0, eps, choose, update, masks |
| 22, 23, 24, 24b, 24c | `players/market_maker/{adaptive,prehistory}.py` | ring buffer, O(1) OLS, (4.2), D0 (A3) |
| 25, 26 | `game/protocol.py` (kernel), `game/shocks.py`, `game/session.py` | period order, streams, session |
| 27, 28 | `game/protocol.py` streak + `game/session.py` phases | convergence, freeze |
| 29, 30, 31 | `metrics/{collusion,trading_policy}.py` | Delta^C, chi_hat, I^C |
| 32, 33 | `metrics/market_quality.py` | L^C, E^C |
| 34, 35a, 35b, 35c | `game/irf.py` | orientation, shock calibration, fork, paired paths, classifier, long-run baseline |
| every `main()`, receipt dataclass, provenance/atomicity check | `tests/` (as asserts) or deleted | — |

## Why the price grid must be per value (footnote 25), in one number / 为什么价格网格要按 v 分

`dgj/diagnostics.py::detectability` — probability that a ONE-STEP deviation by one speculator
changes the price-grid index the other speculator sees next period (`tests/test_detectability.py`):

| σ_u | global grid | per-value grid P(v) |
|---|---|---|
| 0.1 (low noise) | **0.068** (0.000 at the two values nearest v̄) | **1.000** |
| 100 (high noise) | 0.048 | 0.140 |

With the global grid a deviation is invisible in the state, so punishment cannot be learned and
sessions converge near Nash (Δ^C ≈ 0.1). With P(v) every deviation is visible at low noise
(Δ^C ≈ 0.35 in two paper-criterion sessions). At high noise the 1.96σ_u band swamps the grid
under either reading — which is the paper's own explanation of why the high-noise regime can only
collude through over-pruning, not price triggers.
