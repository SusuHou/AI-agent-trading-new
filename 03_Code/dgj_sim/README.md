# dgj_sim — compiled-kernel replication of Dou, Goldstein & Ji (2025)

A rebuild of `../vibe_replication/steps` as a program that can actually run the
paper's experiments (10⁷–10¹⁰ periods per session, 1,000 sessions per cell).
The economics are the same; the difference is structure and speed.

把 `../vibe_replication/steps` 重组成一个真的能跑论文实验规模的程序。经济学完全相同，
不同的是结构和速度。

```
py -3 -m pip install numba          # optional but ~200x faster; pure-Python fallback otherwise
py -3 -m unittest discover -s tests -v
py -3 -m dgj.experiments.run_cell --debug-only --sessions 4 --workers 4 --max-periods 20000000 --out outputs/low_noise
```

## Step 37A: freeze accepted session metrics / 冻结已验收的 session 指标

After strict aggregation has produced an accepted ``summary.json``, export its
``per_session`` records without recalculating any metric:

/严格汇总生成已验收的 ``summary.json`` 后，在不重新计算任何指标的情况下导出
``per_session`` 记录：

```bash
python hpc/export_summary_csv.py "$HIGH_OUT/summary.json" \
  --expected-sessions 1000 \
  --expected-noise-std 100
```

The command validates the complete session-index set, finite metrics, aggregate
totals, provenance counts, mechanism fields, ``cell.json``, and the requested
noise condition. It then publishes ``session_metrics.csv`` and
``session_metrics_receipt.json`` beside the summary. Existing identical files
are validated; different files are never overwritten. / 该命令验证全部 session
编号、有限数值、汇总一致性、provenance、mechanism、``cell.json`` 及噪声条件，
然后发布 CSV 和哈希 receipt；绝不覆盖内容不同的已有文件。

## Step 37B: high-noise descriptive evidence / 高噪声描述性证据

Use only the immutable Step 37A CSV and receipt; do not return to the session
``.npz`` files or recompute the economic metrics. / 只读取 Step 37A 已冻结的 CSV
和 receipt，不返回 session ``.npz``，也不重新计算经济指标：

```bash
python hpc/summarize_session_metrics.py \
  "$HIGH_OUT/session_metrics.csv" \
  --receipt "$HIGH_OUT/session_metrics_receipt.json" \
  --expected-sessions 1000 \
  --expected-noise-std 100 \
  --expected-label high_noise \
  --output-dir "$HIGH_OUT/step37b_high_noise"
```

The command re-hashes and validates the Step 37A pair, then publishes these
immutable artifacts without pandas, Matplotlib, or any extra HPC dependency:
/ 该命令重新验证 Step 37A 文件及其哈希，然后在不增加 pandas、Matplotlib 或
其他 HPC 依赖的情况下发布：

- ``descriptive_statistics.csv`` — eight metrics with mean, sample standard
  deviation, standard error, Type-7 quantiles, minimum, and maximum. /
  八个指标的均值、样本标准差、标准误、Type-7 分位数及范围。
- ``ecdf_data.csv`` — every accepted observation in exact deterministic ECDF
  order; this is the canonical plotting input for later figures. / 保留每个
  session 的精确 ECDF 绘图数据，不使用任意直方图分箱。
- ``provenance_counts.csv`` — current-schema and recovered legacy counts. /
  当前 schema 与恢复的 legacy session 数量。
- ``descriptive_report.md`` — a human-readable high-noise table. / 便于阅读的
  高噪声描述表。
- ``analysis_receipt.json`` — input/output hashes, definitions, runtime, and
  mechanism-availability status. / 输入输出哈希、统计定义、运行环境和机制可用性。

This is high-noise descriptive evidence only—not a completed low-versus-high
replication, confidence interval, hypothesis test, or paper comparison. Core
campaigns run with ``irf_paths=0``; therefore zero mechanism sessions means
"not measured," and Step 37B records shares as unavailable rather than three
false zero-percent results. / 本步骤只是高噪声描述证据，不是完整的高低噪声比较。
核心实验使用 ``irf_paths=0``，所以机制 session 为零表示“没有测量”，绝不表示
三种机制的真实占比都是零。

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
mkdir -p logs
OUT=outputs/low_noise_uncapped CELL_ARGS="--noise-std 0.1 --price-grid per_value --label low_noise" \
  WORK_PERIODS=5000000000 sbatch hpc/submit_array.slurm
# Submit the same command again after the array finishes: completed sessions skip,
# incomplete sessions load ckpt_<k>.npz and do another 5B-period computing shift.
python hpc/aggregate_dir.py outputs/low_noise_uncapped --expected-sessions 1000
```

* One array task = one session (`dgj.experiments.run_session_cli`). Each writes its own
  `session_<k>.npz` **only after genuine convergence**. While incomplete it atomically saves
  `ckpt_<k>.npz` plus `progress_<k>.json`; resubmitting the same session resumes exactly.
  / 只有真正收敛才写正式结果；未完成时只保存 checkpoint，重复提交即可续跑。
* `--work-periods` is additional work for one invocation, not a cumulative scientific cap.
  The default is 5B per invocation. Exit 75 means a safe incomplete pause; the Slurm wrapper
  converts it to task success. The session continues across later array submissions until the
  one-million-period policy-stability criterion is met.
* A per-session operating-system lock rejects overlapping array jobs for the same ID.
  Checkpoints also bind a SHA-256 fingerprint of the trajectory-defining source files;
  changing scientific code between rounds is rejected instead of silently mixed.
* The 8-hour wall time remains an engineering safeguard. Slurm sends `USR1` five minutes early;
  Python finishes its current checkpoint block and pauses safely. Reaching 50B cumulative periods
  emits a diagnostic-review warning but does not declare convergence or failure.
* `aggregate_dir.py` now refuses checkpoints, censored files, identity mismatches, and an incomplete
  expected cohort. It cannot silently mix cap-time policies into `Delta^C`.
* `NUMBA_CACHE_DIR` must be node-local (the script sets it) — a shared-filesystem cache races.
* Memory per session ≈ 200 MB (Q table 0.7 MB; 1M-period shock chunks ≈ 50 MB).
* Suggested first campaign: the two baseline cells (σ_u = 0.1 and 100) × the two price-grid
  readings (`per_value`, `global`), 100 sessions each, before spending on 1,000-session cells.

### Recovering an older capped cohort / 修复旧的 capped 实验

Do not overwrite the old evidence directory. The complete reconnect-safe protocol—including
independent SHA-256-verified copies, a live Slurm signal/checkpoint pilot, repeated recovery
rounds, and the later high-noise cell—is in [`hpc/NARVAL_UNCAPPED_RUN.md`](hpc/NARVAL_UNCAPPED_RUN.md).

/ 不覆盖旧目录。完整、可在重新 SSH 登录后继续的命令都在上述指南中；请不要从 README
拼接零散命令来运行正式实验。

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
