# Methodology and Replication Diary

**Project:** Replication of *AI-Powered Trading, Algorithmic Collusion, and Price Efficiency*  
**Repository:** `AI-agent-trading-new`  
**Readable reference implementation:** `03_Code/vibe_replication`  
**Candidate accelerated implementation:** `03_Code/dgj_sim`  
**Diary compiled:** 30 August 2026  
**Current empirical status:** core implementation constructed; the candidate low-noise Narval campaign produced 1,000 artifacts, but 178 sessions were right-censored and the cell failed the zero-censoring acceptance gate; recovery is planned, the high-noise campaign has not started, and no final paper result is claimed in this document.

## Audit note

This diary was compiled retrospectively from the paper and online appendix, the paper-to-code checklist, the source files, automated tests, saved local artifacts, the development journal, the learning log, Git history, and the operator's contemporaneous account of the Narval run. It is not a verbatim transcript of the development conversation, and it does not invent dates for steps that were not documented contemporaneously.

The following evidence labels are used throughout:

- **Paper rule:** a rule or equation stated in the paper or appendix.
- **Replication interpretation:** a choice required because the paper does not fully specify its software implementation.
- **Implementation verified:** the code passed the stated unit or integration checks at its own boundary.
- **Debug evidence:** a deliberately small run that tests the pipeline but is not a paper-scale result.
- **Candidate experiment:** a real computational run whose provenance or full acceptance checks are still pending.
- **Formal empirical result:** a result produced only after all required sessions, convergence checks, artifact checks, and aggregation checks pass.

The word **done** in the software history means only that a component passed its stated engineering checks. It does not mean that the empirical findings of the paper have been reproduced.

### Short glossary for a beginning reader

- **Q-learning:** a learning method that stores an estimated long-run discounted profit for every state-action pair and revises one estimate after each observed reward and next state.
- **OLS (ordinary least squares):** a method for fitting a straight line by choosing coefficients that minimize the sum of squared prediction errors.
- **Parity:** evidence that two implementations give the same economically relevant output when they receive the same inputs and supplied random draws.
- **Readable oracle or reference implementation:** the slower, transparent implementation against which an optimized implementation is checked; “oracle” does not mean that it is automatically free of mistakes.
- **Provenance:** the record of how an output was produced, including code version, parameters, environment, random identity, and execution command.
- **Artifact:** a saved file produced by a run, such as a checkpoint, session result, manifest, summary, or figure source-data file.
- **Checkpoint:** a saved copy of all state needed to stop a long session and continue it later without changing its path.
- **Censored session:** a session stopped by an operational cap before convergence; its true convergence time is therefore unknown rather than equal to the cap.
- **Experiment cell:** one fixed parameter and implementation specification, repeated across many independent sessions.
- **IRF (impulse-response function):** a comparison of how a converged market responds over subsequent periods when one controlled shock is introduced.

## 1. Research objective

The objective is to reconstruct the paper's artificial financial market and determine whether independently learning informed speculators reproduce the paper's low-noise and high-noise outcomes. The central baseline comparison uses two informed Q-learning speculators under two noise environments:

- low noise, with `sigma_u = 0.1`; and
- high noise, with `sigma_u = 100`.

The core experiment is designed to estimate the distribution of convergence times, collusion profitability, trading intensity, price informativeness, liquidity, and mispricing across independent sessions. A later mechanism experiment uses impulse-response functions (IRFs) to distinguish price-trigger behavior from over-pruning. The mechanism analysis is logically downstream of the core low/high experiment and is not required before the first baseline comparison can be reported.

The replication is not calibrated by changing parameters until its output resembles the paper. Parameters and interpretations are frozen before formal execution. If the result differs from the paper, the discrepancy is investigated rather than tuned away.

## 2. Sources and hierarchy of evidence

The source hierarchy is:

1. the main paper;
2. the online appendix;
3. the local full-text extractions and technical specification;
4. the readable step-by-step implementation and its tests;
5. the accelerated implementation, but only after parity with the readable implementation is established.

An older implementation can help locate a relevant passage, but it is not evidence that an equation is correct. The principal project records are:

- `docs/00_paper_to_code_checklist.md`;
- `docs/development_journal.md`;
- `docs/learning_log.md`;
- `steps/step_*.py`;
- `tests/test_*.py`; and
- `run_formal_experiment.py`.

## 3. Software-engineering method

The project was intentionally built in small, inspectable increments. Every economic component followed the same sequence:

```text
paper equation or rule
        -> one readable Python function
        -> a hand-calculable example
        -> executable assertions and, where material, dedicated tests
        -> integration with the next component
        -> a larger session-level test
```

This approach was selected because a single large simulation could produce plausible-looking output even when several individual equations were wrong. The readable `steps` implementation is therefore treated as the methodological oracle. Performance optimization is a separate concern and must not silently alter the economics.

Four engineering principles guided the build:

1. **One economic idea per early file.** A beginner can run and inspect each component independently.
2. **Exact economic quantities, finite learning spaces.** Realized prices remain continuous. Informed orders are selected from the finite 15-action grid and retained at their exact numerical values. Only the lagged continuous price is mapped again to a discrete index for the Q-state.
3. **Causal ordering.** Current-period data cannot be used to set the current-period price before those data exist.
4. **Reproducibility and provenance.** Parameters, random streams, checkpoints, source identities, and experiment status are recorded explicitly.

## 4. Model participants and one-period causal structure

The reconstructed market contains five types of objects or participants.

### 4.1 Fundamental value

The fundamental value `v_t` represents the asset's economic value in period `t`. It is exogenous to the agents and is drawn from a ten-point discretization of a normal distribution.

### 4.2 Informed Q-learning speculators

There are `I = 2` informed speculators in the baseline. Both observe the current fundamental value and the same market state, but they have separate Q-tables, separate action randomization, separate profits, and separate updates. They do not communicate directly.

### 4.3 Noise trader

The noise trader supplies an exogenous order `u_t`. It does not learn or maximize an objective. Its role is to make aggregate order flow an imperfect signal of informed trading.

### 4.4 Information-insensitive investors

These investors submit price-responsive demand `z_t`. They do not condition on the current private fundamental signal. Their demand helps the market maker absorb inventory and is present both in the market-maker objective and in the data used to estimate the investor-demand slope.

### 4.5 Adaptive market maker

The market maker observes aggregate informed-plus-noise order flow, estimates two linear relationships from a rolling window, and sets a continuous price. The market maker is adaptive because the OLS coefficients evolve with recent data, but it is not a Q-learning agent.

### 4.6 Period sequence

The final causal order for one training period is:

```text
1. The current state s_t is available.
2. Both informed agents choose their orders x_1,t and x_2,t.
3. The noise order u_t arrives.
4. Aggregate order flow y_t is calculated.
5. The market maker estimates its rule from prior history only.
6. The market maker sets the continuous price p_t.
7. Information-insensitive investors submit z_t.
8. The informed agents receive profits pi_1,t and pi_2,t.
9. The completed market row is added to the rolling history.
10. The next fundamental value v_(t+1) is drawn.
11. The next state s_(t+1) is constructed.
12. Each agent updates exactly one Q-table cell.
13. The joint greedy policy is checked for convergence.
```

This sequence is a methodological constraint. Reordering it could introduce look-ahead bias or allow one agent's Q update to affect the other agent's same-period learning target.

## 5. Baseline calibration

| Symbol | Baseline | Meaning |
|---|---:|---|
| `I` | 2 | Number of informed speculators |
| `v_bar` | 1 | Mean fundamental value |
| `sigma_v` | 1 | Standard deviation of the underlying continuous value distribution |
| `sigma_u` | 0.1 or 100 | Low- or high-noise standard deviation |
| `xi` | 500 | Slope of information-insensitive investor demand |
| `theta` | 0.1 | Market maker's weight on pricing error |
| `rho` | 0.95 | Speculator discount factor |
| `alpha` | 0.01 | Q-learning rate |
| `beta` | `5e-7` | Value-specific exploration decay |
| `n_v` | 10 | Number of fundamental-value points |
| `n_x` | 15 | Number of allowed action indexes |
| `n_p` | 31 | Number of price-state indexes per value |
| `iota` | 0.1 | Widening applied to action and price ranges |
| `T_m` | 10,000 | Market-maker rolling-window length |
| convergence streak | 1,000,000 | Consecutive periods with no joint greedy-policy change |
| measurement window | 100,000 | Frozen-policy periods after convergence |

The parameter object is frozen so that a parameter cannot be silently changed in one part of the simulation after an experiment begins.

---

# Part I. Chronological construction diary

## Entry 01 — Fundamental-value distribution and grid

**Purpose.** Define the possible fundamental values before introducing agents or randomness.

**Paper rule.** The continuous assumption is

```text
v_t ~ N(v_bar, sigma_v^2).
```

The simulation replaces this continuous distribution with `n_v = 10` equal-probability representative points:

```text
v_k = v_bar + sigma_v * Phi^(-1)((2k - 1)/(2n_v)),  k = 1,...,n_v.
```

The fraction `(2k - 1)/(2n_v)` selects the midpoint probability of each equal-probability bin. With ten bins, these probabilities are `0.05, 0.15, ..., 0.95`. The index `k` is only the position of a point in the grid; it is not a random seed and not an economic parameter.

The resulting baseline grid is:

| `k` | Midpoint probability | Fundamental value |
|---:|---:|---:|
| 1 | 0.05 | -0.644854 |
| 2 | 0.15 | -0.036433 |
| 3 | 0.25 | 0.325510 |
| 4 | 0.35 | 0.614680 |
| 5 | 0.45 | 0.874339 |
| 6 | 0.55 | 1.125661 |
| 7 | 0.65 | 1.385320 |
| 8 | 0.75 | 1.674490 |
| 9 | 0.85 | 2.036433 |
| 10 | 0.95 | 2.644854 |

**Implementation.** `src/step01_value_grid.py` constructs the grid. During a session, a random draw selects one of these ten fixed values with equal probability. Random seeds determine the reproducible sequence of selected indexes; they do not replace the ten numerical values.

**Validation.** The grid must contain exactly ten values, be symmetric around one, have mean one, and have discrete standard deviation

```text
sigma_v_hat = sqrt((1/n_v) * sum_k (v_k - v_bar)^2) ~= 0.938.
```

The validation confirms that the code represents the assumed discretized model. It does not establish that real asset fundamentals are normally distributed. That is an economic modeling assumption whose empirical adequacy would require external data and a separate model-validation exercise.

**Status.** Implementation verified.

## Entry 02 — Noise trader

**Purpose.** Add uninformed order-flow noise without yet combining it with agent actions.

**Paper rule.**

```text
u_t ~ N(0, sigma_u^2).
```

Low noise uses `sigma_u = 0.1`; high noise uses `sigma_u = 100`. The true population mean remains zero in both cases.

**Implementation.** The early executable is `steps/step_02_noise_trader_high noise.py`. A seeded generator produces a continuous Gaussian draw. The noise trader is not another strategic agent; it has no Q-table, objective, or memory.

**Validation diary.** With seed 42 and 100,000 draws, the observed sample statistics were:

| Environment | Sample mean | Sample standard deviation |
|---|---:|---:|
| `sigma_u = 0.1` | 0.000226 | 0.100263 |
| `sigma_u = 100` | 0.226079 | 100.262576 |

The larger high-noise sample mean did not imply a changed population mean. The same standardized draws were multiplied by `100/0.1 = 1,000`, so their small residual sample average also scaled by approximately 1,000. Relative to their own scales, the two sample means were the same. For 100,000 high-noise draws, the standard error of the sample mean was approximately `100/sqrt(100000) = 0.316`, so the observed 0.226 remained consistent with a zero population mean.

**Status.** Implementation verified; contemporaneous explanation preserved in `docs/learning_log.md`.

## Entry 03 — Aggregate informed-plus-noise order flow

**Purpose.** Combine strategic and exogenous orders into the quantity observed by the market maker.

**Paper rule.**

```text
y_t = sum_i x_i,t + u_t.
```

For the baseline two-agent case:

```text
y_t = x_1,t + x_2,t + u_t.
```

**Implementation.** `steps/step_03_total_order_flow.py` is a pure arithmetic function. It does not set a price and does not include information-insensitive demand `z_t`.

**Validation.** Fixed hand-selected orders were used so the total could be checked without random sampling.

**Status.** Implementation verified.

## Entry 04 — Information-insensitive investors

**Purpose.** Add the price-responsive investor sector that absorbs inventory around the mean value.

**Paper rule, equation (3.2).**

```text
z_t = -xi * (p_t - v_bar).
```

If price exceeds the mean value, these investors sell; if price is below the mean, they buy. At the mean price they submit zero net demand.

**Hand validation with `v_bar = 1` and `xi = 500`.**

```text
p_t = 1.01 -> z_t = -5  (sell)
p_t = 0.99 -> z_t = +5  (buy)
p_t = 1.00 -> z_t =  0
```

**Implementation.** `steps/step_04_information_insensitive_investors.py` retains this participant explicitly. It later enters three parts of the full model: realized demand, the market maker's inventory objective through `y_t + z_t`, and the rolling regression used to estimate the demand slope.

**Status.** Implementation verified.

## Entry 05 — Informed-speculator profit

**Purpose.** Define the immediate reward supplied to Q-learning.

**Paper rule, equation (3.1).**

```text
pi_i,t = (v_t - p_t) * x_i,t.
```

A positive order is a purchase and a negative order is a short sale. A trader profits by buying below value or shorting above value.

**Validation.** Four sign combinations were checked: buying an underpriced asset, buying an overpriced asset, shorting an overpriced asset, and shorting an underpriced asset. The observed hand-test profits were `+0.40, -0.40, +0.40, -0.40` respectively.

**Status.** Implementation verified.

## Integration Checkpoint A — Basic market arithmetic

Steps 3–5 were connected with fixed values before any market-maker learning or Q-learning was introduced. The recorded integration example produced:

```text
informed-plus-noise flow y_t =  1.50
insensitive-investor order z_t = -5.00
combined market order          = -3.50
trader 1 profit                =  0.38
trader 2 profit                = -0.19
```

The price was intentionally fixed in this checkpoint. The purpose was not to simulate a market, but to prove that the elementary arithmetic components agreed when connected.

## Entry 06 — Market-maker objective

**Purpose.** State what the market maker is minimizing before deriving or coding its price rule.

**Paper rule, equation (3.3).**

```text
min_p E[(y_t + z_t)^2 + theta * (p_t - v_t)^2 | y_t].
```

The first term penalizes residual inventory or order imbalance after information-insensitive demand. The second term penalizes pricing error relative to fundamental value. The parameter `theta` controls their relative weight.

**Implementation.** `steps/step_06_market_maker_objective.py` evaluates the objective for supplied candidate prices. A teaching example compared prices 1.00, 1.02, and 1.04 when `y_t = 10`. Price 1.02 allowed investor demand of -10 to offset the order flow and therefore had the smallest of the three displayed objective values.

The phrase “best of three” described only this hand-check. Production pricing does not search only these three prices; the analytical/adaptive formula is used later.

**Status.** Implementation verified as an objective evaluator.

## Entry 07 — Static theoretical market-maker price

**Purpose.** Convert the market-maker objective into a direct theoretical pricing equation for benchmark validation.

**Paper rule, equation (3.4).**

```text
p_t = [xi/(xi^2 + theta)] y_t
    + [xi^2/(xi^2 + theta)] v_bar
    + [theta/(xi^2 + theta)] E[v_t | y_t].
```

**Interpretation.** Price combines order flow, the unconditional mean fundamental, and the market maker's conditional expectation of value.

**Limit checks.** When `xi = 0`, information-insensitive investors do not respond to price and the formula reduces to `E[v_t | y_t]`. In the numerical illustration this conditional expectation was 1.20. When `xi` is extremely large, investor demand strongly enforces clearing around the mean and the price approaches `v_bar + y_t/xi`; the illustration produced 1.0000005. Noise affects price through observed aggregate order flow and through conditional inference; it is not assumed away.

**Status.** Implementation verified by direct calculation and limiting cases.

## Entry 08 — Nash benchmark

**Purpose.** Define the competitive informed-trading reference case.

**Paper rule.**

```text
x^N(v) = chi^N * (v - v_bar),
chi^N  = 1 / ((I + 1) * lambda^N).
```

`chi^N` is a trading-intensity coefficient. It is not one of the 15 discrete learned actions; it is a continuous theoretical benchmark used to construct grids and counterfactual profits.

**Implementation and validation.** `steps/step_08_nash_benchmark.py` checks the first-order-condition identity and provides continuous benchmark orders and prices. The coupled fixed point is solved in Step 10.

**Status.** Implementation verified.

## Entry 09 — Cartel benchmark

**Purpose.** Define the perfect-cooperation reference case.

**Paper rule.**

```text
x^M(v) = chi^M * (v - v_bar),
chi^M  = 1 / (2I * lambda^M).
```

The cartel benchmark maximizes joint informed-trader profit. It is not an assumption that the learning agents communicate; it is a normalization endpoint against which learned profitability is measured.

**Implementation and validation.** `steps/step_09_cartel_benchmark.py` verifies the cartel first-order condition before Step 10 solves the coupled coefficient system.

**Status.** Implementation verified.

## Entry 10 — Coupled fixed-point solver

**Purpose.** Solve for the benchmark price impact and trading intensity rather than inserting an arbitrary coefficient.

**Paper equations.**

```text
lambda = (theta * gamma + xi) / (theta + xi^2),

gamma  = I * chi / [(I * chi)^2 + (sigma_u/sigma_v_hat)^2].
```

The relevant Nash or cartel expression for `chi` is substituted into these equations. The resulting positive fixed point determines `lambda^N, chi^N` or `lambda^M, chi^M`.

**Numerical method.** `steps/step_10_fixed_point_solver.py` uses bisection for positive `xi`. Bisection was chosen because it is transparent, deterministic, and robust once a unique positive root is bracketed. Root solving occurs once when an experiment cell is created, so its speed is negligible relative to millions or billions of market periods. The paper-valid boundary `xi = 0` uses its exact positive analytical root.

**Validation.** The solver checks the original residual equations, the benchmark identities, the positive-root bracket, and the expected uniqueness on the economically admissible interval.

**Status.** Implementation verified.

## Entry 11 — Benchmark profits

**Purpose.** Produce the theoretical profitability endpoints used later in the normalized collusion measure.

**Paper rules.**

```text
pi^N = sigma_v_hat^2 / [(I + 1)^2 * lambda^N],

pi^M = sigma_v_hat^2 / [4I * lambda^M].
```

Both formulas are expected profit per informed trader or cartel member; cartel joint profit is `I*pi^M`.

**Validation.** `steps/step_11_benchmark_profits.py` compares each closed-form expression with direct expected-payoff calculations over every discrete value point and a symmetric noise pair `(-sigma_u,+sigma_u)`. The two-point noise construction is a validation device that exploits mean-zero linearity in benchmark profit; it does not replace Gaussian noise in simulation.

**Status.** Implementation verified.

## Entry 12 — Fifteen-action grid

**Purpose.** Convert a continuous trading problem into a finite set of choices for tabular Q-learning.

The paper baseline uses `n_x = 15`, not 21. Fifteen is a calibrated design choice from the paper rather than a number produced by random seeds.

**Construction.** First define the gap between benchmark intensities:

```text
gap   = chi^N - chi^M,
lower = chi^M - iota * gap,
upper = chi^N + iota * gap.
```

Fifteen equally spaced multipliers `c_j` are placed between the widened endpoints. For a particular fundamental value,

```text
x_j(v) = (v - v_bar) * c_j.
```

The action index is stable across values, although the raw order changes with `v`. This construction does not force the learned policy to be linear because an agent can select different indexes in different states.

Low- and high-noise cells solve different fixed points. They therefore receive different `chi^N`, `chi^M`, multiplier endpoints, and raw action grids. There is no single universal set of 15 numerical orders shared by both noise environments.

**Validation.** `steps/step_12_action_grid.py` checks exactly 15 multipliers, correct endpoints and spacing, and mirror-image buy/sell orders around the mean value.

**Status.** Implementation verified.

## Entry 13 — Value-specific 31-point price grids

**Purpose.** Map the continuous previous price into a finite Q-learning state without using that rounded price for economic calculations.

The paper uses `n_p = 31`, approximately `2n_x`. The bounds for one fixed value are based on the Nash price impact, the Nash and cartel orders at that same value, and a `1.96 sigma_u` noise band:

```text
p_L(v) = v_bar + lambda^N [I * min{x^M(v),x^N(v)} - 1.96 sigma_u],
p_H(v) = v_bar + lambda^N [I * max{x^M(v),x^N(v)} + 1.96 sigma_u].
```

The interval is widened by `iota` at both ends and divided into 31 equally spaced points.

The code follows the coefficient `1.96` printed in the paper. Nearby prose describes the bounds as the 5th and 95th percentiles, which would normally correspond to approximately `1.645` standard deviations. This textual inconsistency is retained in the record rather than silently changing the printed formula.

**Material implementation change.** The first implementation pooled price extremes across all fundamental values into one global grid. A resolution diagnostic showed that this grid could be too coarse: a unilateral one-action deviation often failed to change the previous-price state index, especially in the low-noise environment where the paper's price-trigger mechanism requires such a deviation to be visible. The implementation was therefore changed to a matrix `P(v_k)` with ten rows and 31 points per row.

This is supported by the paper's footnote explaining the relationship between `n_p` and `2n_x`. It does not leak new information because the previous fundamental value is already in the state. The state count remains `31 * 10 * 10 = 3,100`; only the numerical interpretation of a price index depends on the known previous value.

**Validation.** `steps/step_13_price_grid.py`, `steps/step_13b_grid_resolution.py`, and their tests verify shape `10 x 31`, strict ordering, equal row spacing, symmetry, and the one-action-step detectability diagnostic. Pre-change and post-change experiment results must never be pooled.

**Status.** Value-specific implementation verified. The current readable state code rejects a flat global grid; the former global version survives only in pre-refactor history/snapshots. The accelerated candidate implements a separate global option, but it requires its own validation before serving as the planned sensitivity.

## Entry 14 — Finite state representation

**Paper rule.**

```text
s_t = (p_(t-1), v_(t-1), v_t).
```

**Implementation.** A state contains three integer indexes: previous-price index, previous-value index, and current-value index. Continuous `p_(t-1)` is mapped to the nearest point in row `P(v_(t-1))`; out-of-range values are clipped to the row endpoints, and an exact midpoint is assigned to the lower index. Continuous prices remain untouched in profits, investor demand, and market-maker history.

The flattened state ID has 3,100 possible values. Encoding and decoding permit storage in a two-dimensional Q-table while preserving the economic tuple.

**Validation.** `steps/step_14_state_representation.py` and the row-aware tests check value round trips, price boundaries, clipping, exact midpoints, encoding/decoding, and the full state count. Formal simulations must record clipping rates because frequent clipping would indicate an inadequate state range.

**Status.** Implementation verified. The nearest-point, clipping, and lower-midpoint conventions are disclosed replication choices because the paper does not specify the exact software mapping.

## Entry 15 — Initial state

**Purpose.** Start each independent session from a valid state without privileging a particular previous price or value pair.

**Paper rule.** The initial state is sampled uniformly from the finite state space `P x V x V`.

**Implementation.** `steps/step_15_initial_state.py` draws valid indexes for the previous price, previous value, and current value. With value-specific price grids, the price index refers to the row belonging to the sampled previous value. This step consumes an injected random generator; Step 26 later assigns it a stream separated from values, noise, and agent actions.

**Validation.** Fixed seeds reproduce the same initial state, all indexes fall within their legal ranges, encoding/decoding agrees, and repeated draws do not access an invalid price row. Cross-stream independence is a Step 26 validation rather than a Step 15 claim.

**Status.** Implementation verified.

## Entry 16 — Initial Q-tables

**Purpose.** Give each informed agent a paper-based initial estimate rather than beginning every state-action value at an arbitrary constant.

**Paper rule.** Under a uniformly random opponent action and mean-zero noise, the initial value for agent `i` is

```text
Q_i,0(s,x)
  = 1 / [(1-rho)n_x]
    * sum_(x_-i in X)
      {v - [v_bar + lambda^N(x + (I-1)x_-i)]}x.
```

The expression averages the current one-period profit over all opponent actions and divides by `1-rho` to convert the stationary one-period estimate into a discounted starting value.

**Implementation.** `steps/step_16_initial_q_table.py` builds a table with 3,100 state rows and 15 action columns, or 46,500 cells per agent. The initial formula depends on current value and current action but not on previous price or previous value, so a smaller value-by-action block is calculated once and copied into all compatible states. Raw orders from `X(v)` are used; multiplier indexes are never mistaken for orders.

Step 16 constructs the common numerical initializer. Step 21 copies it into separate agent-owned arrays and proves that memory is not shared. Sharing one mutable NumPy array would make one agent's learning directly change the other agent and would violate independence.

**Validation.** A tiny opponent-action grid is calculated fully by hand, table dimensions are checked, and action-column meanings remain stable even when `v < v_bar` reverses the numerical order of raw orders.

**Status.** Implementation verified.

## Entry 17 — Meaning of a Q-value

**Purpose.** Record what the table represents before implementing choice and updating.

The Q-value is not merely today's profit. It is the expected discounted long-run profit from choosing action `x` in state `s` and then continuing optimally:

```text
Q_i(s,x) = E[pi_i,t + rho * V_i(s_(t+1)) | s_t=s, x_i,t=x],

V_i(s)   = max_x Q_i(s,x).
```

Thus the agent maximizes long-run discounted profit. Realized current profit is the immediate reward inside the learning target, not the entire objective.

**Implementation.** `steps/step_17_q_value_meaning.py` documents the row/column interpretation, provides safe accessors, validates dimensions, and supports both a flat table and a conceptual state-action tensor.

**Status.** Implementation verified as a semantic and dimensional boundary.

## Entry 18 — Epsilon-greedy choice

**Purpose.** Allow the agent to explore actions while gradually exploiting its current Q estimates.

**Paper rule, equation (2.6).** With probability `1-epsilon`, select a maximizing action; with probability `epsilon`, select a random action from the action set.

```text
action = random action,       with probability epsilon;
action = argmax_x Q(s,x),     otherwise.
```

The paper does not specify a tie rule when several actions have exactly the same maximal Q-value. During training, the replication selects uniformly among exact maximizers using that agent's private action/tie stream. Numerically close but unequal values are not treated as ties.

**Validation.** `steps/step_18_epsilon_greedy_action.py` forces the pure exploration and pure exploitation branches, checks random draws at probability boundaries, exercises the uniform-by-construction tie branch, and confirms that selections remain within the exact maximizing set. It is not a statistical goodness-of-fit test of tie frequencies.

**Status.** Implementation verified; the exact-tie rule is a disclosed replication choice.

## Entry 19 — Value-specific exploration counters

**Purpose.** Implement the paper's exploration decay separately for each possible current fundamental value.

**Paper rule, equation (4.3).**

```text
epsilon_t(v) = exp[-beta * n_t(v)],
```

where `n_t(v)` is the number of past market periods in which the system visited value `v`. There is no fixed ten-percent exploration rate. At a value's first visit, its past count is zero and therefore `epsilon = 1`.

**Implementation.** One shared vector of ten counters belongs to the market session, not one vector per trader. Both agents observe the same current value and use the same epsilon, but make independent random draws. The relevant counter is incremented once only after both agents have acted, so the agents face the same pre-period visit history.

For example, visits to value indexes `3, 3, 7` produce the vector

```text
[0, 0, 0, 2, 0, 0, 0, 1, 0, 0].
```

Here “index” means a zero-based position in the ten-value grid, and “counter” means the number of previous visits to that position.

**Validation.** `steps/step_19_value_specific_epsilon.py` checks the first visit, repeated visits, independence across value indexes, monotonic decay, and exactly one increment per complete market period. `checkpoint_b_exploration_choice.py` connects Steps 18 and 19 in a transparent action-selection period.

**Status.** Implementation verified.

## Entry 20 — Q-learning update

**Purpose.** Update exactly the state-action cell that was visited.

**Paper rule, equation (2.4).**

```text
Q_new(s_t,x_t)
  = (1-alpha)Q_old(s_t,x_t)
    + alpha[pi_t + rho * max_(x') Q_old(s_(t+1),x')].
```

This can be read as retaining most of the old estimate and incorporating a small share, `alpha`, of a new target containing current realized profit and discounted continuation value.

**Appendix acceleration.** The reported experiments average the continuation term over all equally likely next fundamental values:

```text
continuation
  = (1/n_v) * sum_(v' in V)
      max_(x') Q((p_t,v_t,v'),x').
```

This replaces only the realized-next-value continuation with its exact discrete expectation. The current realized profit remains unchanged.

**Implementation.** `steps/step_20_q_learning_update.py` implements both the direct equation and the appendix expected-next-value version. All required old Q rows are copied before the single write, which matters if a next state equals the current state.

**Validation.** The direct hand example updates `10` to `10.11`; the expected-continuation example updates `10` to `10.04825`. Full-table comparisons prove that only the visited cell changes.

**Status.** Implementation verified; the appendix acceleration is the primary reported-experiment path.

## Entry 21 — Two independent informed agents

**Purpose.** Connect two Q-learners without accidental shared mutable learning state.

**Implementation.** `steps/step_21_two_independent_q_traders.py` creates two `InformedQTrader` objects. The trader object is deliberately mutable because its Q-table must learn over time. Frozen dataclasses are used instead for immutable records such as completed action or update receipts. The use of `@dataclass` does not itself imply immutability; only `@dataclass(frozen=True)` prevents ordinary field reassignment.

Each trader owns:

- a separate Q-table;
- private action-selection randomness;
- its own action and profit;
- its own one-cell Q update.

The Step 21 demonstration can use one private generator per trader. The formal two-stream-per-agent split between mode and action/tie draws is established in Step 26. The two traders share the public state, the current value, epsilon, and the system-level value-visit counter. They do not share Q arrays and do not send messages to one another.

**Validation.** Mutating or updating trader 1 cannot alter trader 2's table, and vice versa. Both actions are selected before either learning update is committed.

**Status.** Implementation verified.

## Entry 22 — Rolling market-maker history

**Purpose.** Give the adaptive market maker a finite memory of recent market outcomes.

**Paper rule.** The maker stores the most recent `T_m = 10,000` observations:

```text
D_t = {(v,p,z,y)}.
```

**Implementation.** `steps/step_22_market_maker_rolling_history.py` implements a fixed-capacity first-in-first-out window. The current row cannot enter the data used to set its own price. It is appended only after price, demand, and profits have been calculated.

“Rolling” means that after the window is full, each new observation replaces the oldest observation. A non-rolling expanding history would give ancient observations permanent weight; the rolling version lets the estimated relationship adapt to the agents' more recent behavior. Even after agents stabilize, the rolling market maker remains statistically meaningful because the most recent 10,000 observations estimate the relationship induced by the stable strategy and current stochastic environment.

**Validation.** Capacity never exceeds `T_m`, ordering is preserved, the correct oldest row is removed, and a pre-pricing snapshot excludes the current row. Actual prior-only pricing is connected in the Step 24 toy pipeline and fully verified in Step 25.

**Status.** Implementation verified.

## Entry 23 — Market-maker OLS regressions

**Purpose.** Estimate investor demand and the informativeness of order flow from the rolling data.

**Paper linear models.**

```text
z = xi_0_hat - xi_1_hat * p + error_z,

v = gamma_0_hat + gamma_1_hat * y + error_v.
```

Generic OLS software writes the first equation as `z = intercept + raw_slope*p`. Therefore the paper's coefficient is recovered by

```text
xi_0_hat = intercept,
xi_1_hat = -raw_slope.
```

The second regression directly yields `gamma_0_hat` and `gamma_1_hat`.

**Replication interpretation.** The paper describes rolling linear regressions but does not name the estimator software, weighting, regularization, sign constraints, or degenerate-data behavior. The readable baseline uses ordinary unweighted least squares with an intercept, no regularization, and no silently imposed sign constraint. Degenerate inputs are rejected rather than repaired by an invented coefficient.

These alternatives would be economically material. Extra recency weights would make effective memory shorter than the stated 10,000-row window. Regularization would shrink the estimated slopes and therefore change `lambda_hat_t` and prices. A sign constraint could suppress unusual but genuine finite-sample estimates and introduce nonlinear boundary behavior. A fallback used when regressors have no variation could freeze an old price rule or inject an arbitrary new one. For this reason, none of these choices is introduced silently.

**Implementation and validation.** `steps/step_23_market_maker_ols.py` first implements the regression transparently and matches synthetic hand calculations and NumPy least squares.

**Status.** Readable OLS implementation verified; alternative weighted or regularized regressions are robustness questions, not baseline rules.

## Entry 24 — Adaptive market-maker price

**Purpose.** Turn the rolling estimates into the actual continuous price used each period.

**Paper pricing rule, equation (4.2), using the equation-(4.1) regression estimates.**

```text
lambda_hat_t
  = (theta * gamma_1_hat,t + xi_1_hat,t)
    / (theta + xi_1_hat,t^2),

p_t = gamma_0_hat,t + lambda_hat_t * y_t.
```

The two regressions play different roles. The `v`-on-`y` regression estimates how informative order flow is about value. The `z`-on-`p` regression estimates how strongly information-insensitive investors absorb price movements. Their slopes enter the price-impact formula, and the estimated value intercept anchors the price level.

The estimated `xi_0_hat` remains a regression diagnostic but does not enter the printed price formula. The current input is `y_t`, not `y_t+z_t`, because `z_t` does not exist until after price is announced. The output remains continuous and is not clipped or rounded to `P(v)`. The paper supplies no coefficient constraints, smoothing, clipping, or fallback rule, so the baseline imposes none and rejects unidentified or non-finite cases.

**Readable implementation.** `steps/step_24_adaptive_market_maker_price.py` consumes Step 23 estimates calculated from prior history together with a supplied current `y_t`.

**Efficient implementation.** Re-running batch OLS from scratch every period would be unnecessarily slow. `steps/step_24b_fast_rolling_ols.py` maintains centered sufficient statistics and updates them in constant time when one row enters and one row leaves. It periodically rebuilds the statistics from the window to limit floating-point drift. The optimized coefficients and prices must match the readable OLS path.

**Initial-history issue.** The paper does not state how the first 10,000 rows exist before period zero. `steps/step_24c_initial_market_maker_history.py` implements a balanced Nash-consistent synthetic prehistory that contains exactly 10,000 rows and allows the market maker to recover the intended coefficients through its own OLS. It deterministically crosses every value with a symmetric, rescaled Gaussian-quantile noise design; it is not one random draw of 10,000 iid historical periods. This exactly controls the initial noise mean, population standard deviation, and value-noise covariance. A cartel-consistent initializer also exists for sensitivity analysis. This is assumption A3, not a paper fact.

**Validation.** Hand price calculations, rolling-versus-batch OLS parity after every tested insertion/removal, low- and high-noise prehistory recovery, FIFO ordering, and periodic-rebuild parity all pass their encoded tests.

**Status.** Implementation verified. Formal results must later test sensitivity to cartel-consistent and expanding-window starts.

## Entry 25 — One complete training period

**Purpose.** Join all earlier actors and equations into one deterministic causal protocol.

**Implementation.** `steps/step_25_one_market_period.py` implements events 1–12 of the thirteen-event sequence stated in Section 4.6 of this diary. The market maker prices using the old history and current `y_t`; continuous price determines `z_t` and profits; the completed row then enters history; `v_(t+1)` forms the next state; and each agent updates one private Q cell using the appendix continuation expectation. Step 27 subsequently appends the convergence check as event 13.

Both agents' learning targets are prepared before either private table is modified. Changing only the realized next fundamental value changes the realized next-state label but cannot change the current price, profit, or expected-next-value target.

**Embedded executable self-validation.** A line-by-line deterministic trace checks prior-only pricing, no look-ahead, one shared visit increment, continuous-price economics, FIFO history replacement, prepare-then-commit Q updates, and next-state construction. Five deliberately invalid calls verify that rejected inputs mutate no random stream, counter, table, or history row. These checks run when the step module is executed directly; they are not all separate `unittest` cases.

**Status.** Implementation verified as the readable one-period oracle.

## Entry 26 — Reproducible random streams

**Purpose.** Make stochastic paths reproducible and independent of parallel scheduling.

**Implementation.** `steps/step_26_reproducible_random_streams.py` derives stable experiment-cell and session identities and then seven named SHA-256 child streams:

1. initial state;
2. fundamental values;
3. noise orders;
4. trader 1 exploration mode;
5. trader 1 random/tie action;
6. trader 2 exploration mode; and
7. trader 2 random/tie action.

Values are sampled uniformly from the ten fixed grid points. Noise remains continuous Gaussian. Each random variable is drawn only at its causal point in the period. No seed depends on wall-clock time, process ID, task completion order, or Python's unstable default hash.

**Embedded executable self-validation.** Identical experiment-cell/session identities replay the full path and final mutable state exactly; changed identities differ; consuming one stream cannot move another stream; all planned session and child-stream seeds are unique; and a fixed-draw lean path matches Step 25. These direct checks are distinct from the tests collected automatically by `unittest discover`.

**Status.** Implementation verified. The particular hash-based seed derivation is a disclosed replication choice because the paper does not publish seeds or an RNG engine.

## Entry 27 — Convergence tracker

**Purpose.** Implement the paper's requirement that all agents' best strategies remain unchanged for 1,000,000 consecutive periods.

The source paper additionally states that all 1,000 sessions continue running until every session satisfies this criterion, and reports realized convergence times ranging from approximately 20 million to 50 billion periods. Accordingly, convergence—not a pre-set cumulative period count—is the scientific stopping rule. Finite Slurm wall times and period budgets apply only to individual computing invocations; an incomplete session must preserve its exact checkpoint and resume in a later invocation.

“Unchanged” does not mean that each agent literally selects the same action every realized period. It means that, for every possible state, the complete set of Q-maximizing actions remains unchanged for both agents.

**Implementation.** Before period zero, save each agent's state-to-exact-maximizer-set mapping. After both Q updates, inspect the two rows that could have changed. If neither agent's maximizer set changes, increment one shared stability streak. If either changes, reset the streak to zero. A period containing the change is not counted as stable. Q-values may move without resetting the streak if their exact maximizing sets remain unchanged.

**Embedded executable self-validation.** `steps/step_27_convergence_tracker.py` checks the known toy streak `[1,0,1,2,3]`, exact ties, threshold boundaries, equivalence between efficient row-local checking and full-policy checking, and same-seed parity with Step 26. These assertions run on direct execution rather than through a dedicated Step-27 test file.

**Status.** Implementation verified. Exact tie-set and off-by-one conventions are disclosed because the paper does not define them in software terms.

## Entry 28 — Training, measurement, and completion

**Purpose.** Separate learning from the post-convergence sample used to estimate research outcomes.

**Implementation.** `steps/step_28_session_phases.py` defines the exclusive state machine

```text
TRAINING -> MEASUREMENT -> COMPLETE.
```

The period that reaches the one-million stability streak remains a training period. Measurement covers exactly the next 100,000 periods. At convergence, each state is assigned one frozen greedy action; exact ties use the lowest action index. During measurement:

- Q-tables are read-only;
- exploration is disabled;
- value-visit counts and trader RNGs are frozen;
- fundamentals and noise continue without a reset;
- the rolling market maker continues to learn from its own recent rows; and
- raw completed rows stream to attached metric sinks.

**Embedded executable self-validation.** A toy `K=2, T=3` controller produces training periods `[0,1]`, measurement periods `[2,3,4]`, and five total periods. Direct assertions reject overlap, stale frozen policies, direct kernel bypass, extra periods, missing sinks, and treating a debug cap as convergence. Later integration tests also exercise this boundary, but there is no dedicated Step-28 test file.

**Status.** Implementation verified. Freezing agents while allowing the market maker to roll is a paper-supported replication interpretation; continued background Q-updating and a frozen maker remain named sensitivities.

---

# Part II. Post-convergence outcome construction

## Entry 29 — Matched-path collusion profitability

**Purpose.** Compare learned profits with Nash and cartel profits under exactly the same realized fundamentals and noise shocks.

For every frozen-policy measurement row, the counterfactual benchmark `B` in `{N,M}` is reconstructed continuously:

```text
x^B_t  = chi^B * (v_t - v_bar),
y^B_t  = I * x^B_t + u_t,
p^B_t  = v_bar + lambda^B * y^B_t,
pi^B_t = (v_t - p^B_t) * x^B_t.
```

The benchmark is not rounded to an action or price grid and does not reuse the adaptive learned market maker. This preserves the theoretical meaning of Nash and cartel.

For each agent,

```text
Delta_i^C
  = [mean(pi_i^C) - mean(pi_i^N)]
    / [mean(pi_i^M) - mean(pi_i^N)],

Delta^C = (1/I) * sum_i Delta_i^C.
```

The software never clips a valid `Delta^C` to `[0,1]`. A negative value means the learned strategy earned less than the matched-path Nash benchmark; a value above one means it exceeded the matched-path cartel normalization. Neither is automatically a bug. If the finite-sample denominator is non-positive or numerically indistinguishable from zero, the metric is reported as undefined rather than repaired with an absolute value or arbitrary epsilon.

**Embedded executable self-validation.** `steps/step_29_matched_path_collusion_profitability.py` reproduces hand examples, including `Delta_1^C=0`, `Delta_2^C=1`, and `Delta^C=0.5`, and matches Step 11's closed forms in low-noise, high-noise, and `xi=0` checks. These assertions run when the module is executed directly and are not a dedicated Step-29 `unittest` file.

**Status.** Metric implementation verified; an individual debug value is not a cross-session paper result.

## Entry 30 — Trading intensity

**Purpose.** Estimate how strongly each informed agent's learned order responds to fundamental value.

**Appendix rule.** For each agent separately, estimate

```text
x_i,t = chi_i,0_hat + chi_i,1_hat * v_t + error_i,t.
```

The session trading-intensity estimate is

```text
chi_hat^C = (1/I) * sum_i chi_i,1_hat.
```

The regressions use signed raw orders and continuous value numbers. They do not use action indexes, total order flow, theoretical benchmark orders, or a forced theoretical intercept.

**Implementation.** `steps/step_30_trading_intensity.py` maintains centered constant-memory OLS moments for each agent and receives the identical Step 28 rows used by the collusion scorer.

**Validation.** Hand intercepts and slopes match independent NumPy OLS, including a complete 100,000-row stream. Duplicate sinks, cross-session wiring, and inconsistencies among value, action index, and raw order are rejected.

**Status.** Implementation verified.

## Entry 31 — Price informativeness

**Purpose.** Convert informed trading intensity into the paper's signal-to-noise measure.

**Appendix rule.**

```text
I^C = (I * chi_hat^C)^2 * (sigma_v_hat/sigma_u)^2.
```

The calculation uses the known standard deviation of the discrete value grid and the configured noise standard deviation. It does not replace either with a realized finite-sample standard deviation.

**Validation.** `steps/step_31_price_informativeness.py` reproduces a hand example with signal variance 6.25, noise variance 0.25, and informativeness 25. Provenance checks prevent the statistic from being combined with another session's trading-intensity estimate or a rebound parameter object.

**Status.** Implementation verified; the measure is not clipped to one.

## Entry 32 — Market liquidity

**Purpose.** Measure how effectively the market absorbs an order in each period.

**Appendix rule.**

```text
L_t^C = 1 / |1 - xi * lambda_hat_t|.
```

The `lambda_hat_t` used is the prior-history coefficient that actually set period `t`'s price. The structural parameter `xi` is used rather than the maker's estimated `xi_1_hat`. The reciprocal is applied period by period before aggregation; using an average lambda inside one reciprocal would be a different statistic.

The appendix prose describes average liquidity, while the displayed aggregation omits `1/T`. The replication therefore retains both the literal sum and the arithmetic mean over the exact 100,000 measurement rows, using the mean as the principal session quantity.

**Numerical treatment.** A true representable zero in `1-xi*lambda_hat_t` is recorded as positive infinity with diagnostics. No epsilon or clipping is added. A finite near-zero gap is inverted as written.

**Validation.** `steps/step_32_market_liquidity.py` reproduces period values `(1,2,2/3)`, sum `11/3`, and mean `11/9`, agrees with an independent inventory derivative, and matches batch aggregation over 100,000 rows.

**Status.** Implementation verified; the paper's sum-versus-mean inconsistency remains disclosed.

## Entry 33 — Mispricing

**Purpose.** Reproduce the paper's pricing-error measure while exposing rather than hiding a sign ambiguity.

The economic definition is

```text
E_t^C = |E[p_t | v_t] - v_t|.
```

The appendix's printed expansion is

```text
(1 - lambda_hat_t * I * chi_hat^C) * |v_t - v_bar|,
```

but it does not print an absolute value around the first factor. The expressions agree when that loading is non-negative. If it is negative while the value deviation is positive, the printed expansion can become negative even though the definition is an absolute error.

**Implementation.** `steps/step_33_mispricing.py` stores only `(lambda_hat_t, |v_t-v_bar|)` during measurement. After Step 30 estimates `chi_hat^C` from the complete window, it replays these compact pairs and computes the printed signed expansion and the absolute-valued version of that expansion. The latter is treated as Definition 3.4 under the paper's centering derivation. It is not an independent nonparametric or direct estimate of `E[p_t | v_t]`, and the implementation does not retain a price-regression intercept for such an estimate. It records negative-loading counts, actual formula disagreements, and first affected periods. The main value is left undefined if the two expansions disagree; the code does not silently insert an absolute value.

**Validation.** A hand example gives loading 0.75 and mispricing 1.5. The compact replay matches an independent batch calculation over 100,000 pairs.

**Status.** Implementation verified with an explicit ambiguity guard.

---

# Part III. Impulse-response and mechanism methodology

## Entry 34 — Pure mechanism-classification contract

**Purpose.** Encode the response orientation, shock target, and mechanism thresholds before constructing expensive stochastic IRF paths.

Prices and orders are oriented by `sign(v_t-v_bar)` so that economically comparable adverse movements have the same direction across values above and below the mean. A noise shock is added at local `t=3`; the agents' endogenous order response is measured at `t=4`. The target is a 1.2 percent normalized oriented-price deviation at the cell-pooled finite-sample level after pooling all valid session/path calibration receipts. It is not a requirement that every path or every session individually move by exactly 1.2 percent.

The visually verified thresholds are

```text
x_low  = 5e-5,
x_high = 5e-4.
```

A session is classified as `price_trigger` only if both agents' normalized responses are strictly above the high threshold. It is `over_pruning` only if both absolute responses are strictly below the low threshold. Exact equality, mixed agent outcomes, and all remaining cases are `unclassified`. Agents are never averaged before this two-agent rule is applied.

The appendix's low-response expression is missing parentheses. The replication reads it as

```text
abs((x_tilde - E[x_tilde]) / E[x_tilde]) < 5e-5,
```

because the literal alternative is dimensionally inconsistent and conflicts with the correctly normalized expression elsewhere in the paper.

**Validation.** `steps/step_34_mechanism_classifier.py` checks hand calibration, shock direction, exact and adjacent floating-point thresholds, mixed agents, invalid domains, and the positive-lambda requirement.

**Status.** Implementation verified as a pure contract. Step 34 alone produces no path, impulse response, or empirical mechanism frequency.

## Entry 35A — Lossless converged-market checkpoint

**Purpose.** Create a safe common origin for measurement and independent control/treatment branches.

**Implementation.** `steps/step_35a_converged_market_checkpoint.py` captures the session after convergence and before measurement row zero. It preserves both Q-tables, policy masks and frozen actions, current market state, visit counts, the exact 10,000-row maker history, centered OLS statistics and rebuild phase, and all seven RNG states including the cached second Gaussian variate. Array payloads and the whole envelope receive SHA-256 checks.

Restoration creates detached new arrays and objects. It intentionally does not copy a live controller, convergence tracker, observer, callback, or metric sink.

**Validation.** Tests establish exact next-period parity, Gaussian-cache restoration, exact continuation across OLS resynchronization, branch independence, source non-mutation, and rejection of wrong-time, tampered, or internally inconsistent checkpoints.

**Status.** Infrastructure verified; no shock or IRF result is produced.

## Entry 35B — One paired control/treatment IRF path

**Purpose.** Prove the causal mechanics of a single four-period response path before scaling to 10,000 paths.

The converged checkpoint is treated as local `t=0`; new transactions occur at `t=1,2,3,4`. Path-specific seeds are derived from stable experiment, source-session, path-index, and stream identities using full SHA-256 values. Ordinary noise and next values are drawn once and supplied to both branches. Only treatment receives the additive adverse noise shock at `t=3`.

The Q-policy remains frozen, while each branch's rolling market maker evolves using only its own rows. Treatment `p_3` is mapped into treatment's `t=4` state, permitting a learned price-trigger response.

**Replication interpretations.** The branches carry checkpoint `p_0, v_0, v_1`; ordinary draws use common random numbers; Q remains frozen; and the maker continues rolling. The paper does not provide all of these software details, so they remain disclosed and subject to sensitivity analysis.

**Validation.** `steps/step_35b_paired_irf_path.py` tests pre-shock identity, exactly one treatment shock, independent histories, branch-order invariance, complete `p_3 -> s_4 -> x_4` wiring, calibration integrity, and child-stream uniqueness.

**Status.** One-path reference verified; it is not a 10,000-path result.

## Entry 35C — Session-specific long-run IRF baseline

**Purpose.** Estimate the long-run denominators used to normalize the short-run IRF for the same converged session.

Over the exact 100,000 Step 28 measurement rows, prices and each agent's orders are oriented before averaging. The collector computes compensated constant-memory estimates of

```text
E[p_tilde], E[x_tilde_i], E[pi_i],
```

plus long-run lambda diagnostics. It binds these values to the live measurement sink, exact Step 35A checkpoint, session identity, period boundaries, parameters, grid, and row digest.

The finite-sample estimator is a replication choice: the same session's frozen-policy measurement window is used, not the four-period unshocked control branch.

**Validation.** `steps/step_35c_irf_long_run_baseline.py` matches hand and batch means and rejects detached scorers, manually inserted rows, false fan-out membership, swapped sinks, and stale receipts.

**Status.** Denominator infrastructure verified; it applies no shock and produces no mechanism label.

## Entry 35D — Unshocked `t=3` calibration paths

**Purpose.** Estimate each session's actual unshocked `t=3` price level and price-impact coefficient before choosing one common cell shock.

For each session, exactly 10,000 unshocked continuations run from the Step 35A origin through local `t=3`. The reducer records the pre-append `lambda_hat_3` actually used for pricing and the unshocked oriented price `p_tilde_3^0`.

For efficiency, a detached branch is restored once. After each path, a reversible transaction undoes the three appended/evicted maker rows, centered OLS statistics, rebuild phase, counters, market state, and draw mode. Only compensated sums, diagnostics, counts, and digests remain in memory.

**Validation.** The reversible path matches a fresh restore, including across OLS rebuild points and interrupted-path recovery. Canonical indexes `0,...,9999` are checked exactly.

**Status.** Per-session calibration code verified. It applies no shock and does not establish a 1,000-session result.

## Entry 35E — Cell-level common shock calibration

**Purpose.** Convert 1,000 valid same-cell Step 35D receipts into one adverse shock magnitude shared across all sessions in that cell.

The primary finite-sample exact-level formula is

```text
m_level
  = [(1 + 0.012)E[p_tilde] - E[p_tilde_3^0]]
    / E[actual lambda_hat_3].
```

The older increment shortcut is retained as a named sensitivity:

```text
m_increment
  = 0.012 E[p_tilde] / E[actual lambda_hat_3].
```

The two are identical only when the finite-sample unshocked `t=3` mean equals the long-run price mean. If the unshocked level is already at or above the target, the positive adverse-shock equation is rejected rather than repaired with an absolute value.

Pooling uses raw measurement and path counts. Session ordering, identities, base streams, cell equality, and receipt checksums are validated before one immutable magnitude is produced.

**Validation.** `steps/step_35e_cell_shock_calibration.py` includes a hand case where the exact-level magnitude 0.236 reaches the target but the shortcut 0.036 does not.

**Status.** Calibration algorithm verified; the full set of 1,000 genuine session receipts has not been verified in this diary.

## Entry 35F — Paired response and mechanism classification

**Purpose.** Apply the common shock, estimate each session's `t=4` response, and classify its mechanism.

For every session, 10,000 paired paths are run. Treatment alone receives the common `t=3` shock. The primary response for each informed trader is

```text
r_i
  = [mean_paths(x_tilde^T_i,4) - E_long-run(x_tilde_i)]
    / E_long-run(x_tilde_i).
```

These two trader-specific responses are passed to the strict Step 34 classifier. The paired treatment-minus-control response

```text
r_i^TC
  = [mean_paths(x_tilde^T_i,4) - mean_paths(x_tilde^C_i,4)]
    / E_long-run(x_tilde_i)
```

is retained only as a sensitivity and is not the primary classifier input. Session labels are counted at cell level; paths from different sessions are not pooled and reclassified as one observation.

The runner verifies that its control branch exactly reproduces the Step 35D digest and that

```text
p_tilde_3^T - p_tilde_3^0 = lambda_hat_3 * m.
```

Both branches are rolled back after every path.

**Validation.** `steps/step_35f_paired_response_and_classification.py` includes a genuine one-session, 100-path debug run that hits `1.200000000%` and exercises source-chain and rollback checks.

**Status.** Runner verified at debug scale. The formal `1,000 sessions x 10,000 paths` mechanism experiment remains unverified. Steps 35D–35F currently implement `t=3` shock calibration, the `t=4` order response, and session mechanism labels; they do not yet persist and aggregate complete `t=1,...,4` price, profit, and order-response trajectories of the kind required for a full Figure-3-style IRF plot.

---

# Part IV. Reproducible experiment orchestration

## Entry 36A — One-session result-row smoke test

**Purpose.** Prove that the measurement and metric components can emit structured JSON/CSV before spending compute on learned sessions.

`steps/step_36a_one_session_result_row.py` uses a deliberately stable synthetic policy to exercise the pipeline. Its artifacts store parameters, seed and grid identity, counts, metrics, and timing, but explicitly set flags showing that they are neither learned research evidence nor a paper result.

**Status.** Engineering smoke test verified. Its synthetic `Delta^C` or trading intensity must never be cited as empirical evidence.

## Entry 36B — Immutable experiment manifest

**Purpose.** Freeze the scientific plan and task identities before distributed execution.

`steps/step_36b_experiment_manifest.py` creates deterministic cell and session manifests, derived random streams, parameter and grid identities, source identities, and artifact paths. PAPER mode contains 1,000 tasks per cell. Operational period budgets are recorded separately from the uncapped scientific convergence criterion. Manifests are written atomically, silent overwrite is refused, and tampering changes the checksum.

`src/source_manifests.py` separates the execution-source closure from the result-pipeline closure so an unrelated orchestration-file edit does not invalidate scientific checkpoints, while a change to causal model code does.

**Status.** Manifest infrastructure verified; it performs no learning itself.

## Entry 36C — Exact mid-training resume

**Purpose.** Continue a long training session after wall-time limits without changing its stochastic or learning path.

`steps/step_36c_exact_training_resume.py` saves only between complete periods and serializes every causal component: market state, both agents, Q-tables, visits, policy/convergence state, rolling history and OLS accumulators, controller phase, and all RNG states. The envelope is checksummed and bound to its task, source, Python/NumPy/platform metadata, and native byte order.

**Validation.** An uninterrupted 40-period run equals 19 periods plus save/load plus 21 periods. Tests cover future path equality, detached ownership, OLS rebuild continuation, unchanged global RNG, atomic adoption, and tamper/corruption/wrong-task rejection.

**Status.** Exact-resume implementation verified. The checksum detects accidental changes; it is not authentication against a malicious party.

## Entry 36D — Single-session training runner

**Purpose.** Run one planned session from fresh state or checkpoint until convergence, a safe finite work boundary, or a recorded failure.

`steps/step_36d_single_session_training_runner.py` distinguishes `incomplete`, `converged`, and `failed`. A finite invocation budget limits only one scheduler call and never changes the paper's convergence rule. Checkpoint cadence is global, the newest two checkpoints are retained together with any evidence-pinned source, and a failed attempt leaves the last durable checkpoint as the trusted restart point.

The runner stops at convergence immediately before measurement row zero so the Step 35A origin can be preserved.

**Status.** Operational runner verified; it produces no post-convergence metric by itself.

## Entry 36E — Complete measurement runner

**Purpose.** Turn one converged learned session into a durable, auditable result bundle.

`steps/step_36e_complete_measurement_runner.py` attaches all measurement sinks before row zero, persists and reloads the convergence origin, runs exactly 100,000 frozen-policy rows, and produces Steps 29–33 plus the Step 35C baseline and provenance receipts. If measurement fails partway, partial results are discarded and the complete measurement phase restarts deterministically from its origin.

Artifacts use restricted decoding, path containment, immutable bundles, checksums, and cross-receipt validation. The convergence checkpoint needed by later IRF work remains pinned.

**Status.** Per-session pipeline verified. One bundle is not a 1,000-session cell result.

## Entry 36F — Persisted calibration bridge

**Purpose.** Recover the live object and provenance relationships required for Step 35D/35F after a process boundary.

`steps/step_36f_persisted_calibration_bridge.py` retains the Step 36E evidence, convergence origin, replay source, and byte fingerprints. A fresh process replays the saved measurement rows, requires exact reproduction of the Step 36E metrics and Step 35C baseline, reconstructs the matching live checkpoint/scorer chain, reruns Step 35D from path zero, and requires exact receipt equality.

**Status.** Single-session bridge verified. A full manager that gathers 1,000 bridges, performs one Step 35E calibration, distributes Step 35F jobs, and accounts for failures remains future work.

## Formal core experiment runner

`run_formal_experiment.py` defines one immutable family with:

```text
1,000 low-noise sessions  (sigma_u = 0.1)
1,000 high-noise sessions (sigma_u = 100)
1,000,000 unchanged-policy convergence periods
100,000 frozen-policy measurement periods
```

Array indexes `0..999` map to low-noise sessions and `1000..1999` to high-noise sessions. Each worker owns an exclusive task lock, runs or resumes one Step 36E session, and publishes a status. The verified collector refuses a cell with fewer than 1,000 valid completed bundles, preserves unclamped `Delta^C`, uses standard Type-7 percentiles, and does not invent missing mispricing values.

The core runner deliberately stops before the full-cell IRF mechanism experiment. This allows the baseline low/high comparison to be completed first and the same converged sessions to be reused later.

At the audited baseline, `vibe_replication` does not yet contain a production 2,000-session Slurm array wrapper for this evidence-rich runner; its only committed Slurm wrapper is the Step 36G benchmark. The separate `dgj_sim/hpc/submit_array.slurm` launches the accelerated candidate format and is not automatically interchangeable with `run_formal_experiment.py`, its checksummed Step 36E bundles, or its strict collector.

**Status.** Formal orchestration implemented in the readable track; its paper-scale empirical campaign has not been validated in this diary.

## Entry 36G — Narval throughput benchmark

**Purpose.** Measure the exact readable formal training loop on one scheduled Narval CPU before selecting production wall time and chunk size.

`steps/step_36g_narval_throughput_benchmark.py` creates an isolated non-production PAPER-mode sandbox, runs a warm-up chunk, saves an exact checkpoint, and times a resumed chunk through the real Step 36D path. It checks `fresh -> resumed`, the exact checkpoint handoff, period accounting, zero measurement rows, and isolation from the production artifact root.

The general defaults are 1,000 warm-up plus 10,000 measured periods. The first 30-minute Slurm wrapper deliberately uses 25 warm-up plus 250 measured periods for each noise cell. Low and high cells run in separate processes. Reports store environment, source and commit identity, elapsed times, training and end-to-end throughput, peak resident memory, execution scope, and a checksum.

Every report sets `research_result=false` and `paper_results_ready=false`. The field `linear_extrapolation_seconds_per_million_at_observed_rate` is populated only when the Slurm environment has been verified; local-smoke reports store `None`. Even when populated, it is a linear scheduling estimate, not a measured convergence time, because policy changes can repeatedly reset the one-million stability streak.

**Validation.** Nine focused Step 36G tests and the documented full readable suite of 262 tests passed locally at the validated baseline. A three-period local connection smoke established exact resume and zero measurement rows. The Narval report itself must still be inspected before a cluster throughput claim is recorded.

**Status.** Benchmark code and Slurm wrapper verified; cluster evidence remains separate from the current accelerated candidate campaign.

---

# Part V. Accelerated implementation diary

## Entry dated 30 August 2026 — Addition of `dgj_sim`

### Reason for the second implementation

The readable `vibe_replication/steps` code was designed for explanation and auditability. Its many objects, receipts, checks, and Python-level operations make it too slow for sessions that may require billions of periods. Commit

```text
9ab452fb6dc54e7ce25a9ec9417e346aa177d366
```

added `03_Code/dgj_sim` as a candidate accelerated implementation. This commit is later than the Step 36G commits and contains them; it does not remove the readable reference implementation.

The accelerated engine reorganizes the same intended economics into small functions operating on explicit arrays and a single hot period loop that Numba can compile. Mutations are confined to documented state arrays, including the Q tables and market-maker statistics. The readable `steps` code remains the oracle. Speed does not make the accelerated engine authoritative; parity does.

The relevant committed lineage is:

| Commit | Recorded role |
|---|---|
| `5deb44e` | Readable Repository Baseline R0 through Step 36F |
| `7f915c0` | Step 36G Narval throughput benchmark implementation |
| `c7708ba` | Step 36G validation documentation |
| `9ab452f` | Addition of the separate `dgj_sim` engine and candidate HPC scripts |

### Accelerated architecture

The main modules are organized by economic role:

```text
dgj/config.py                       frozen parameters and experiment-cell identity
dgj/environment/                    value, noise, insensitive investor demand
dgj/players/benchmarks.py           Nash/cartel fixed points and profits
dgj/players/speculator/             state, actions, Q-learning, policy masks
dgj/players/market_maker/           prehistory, rolling OLS, adaptive price
dgj/game/protocol.py                compiled within-period causal order
dgj/game/session.py                 training, convergence, measurement, checkpoint
dgj/game/irf.py                     accelerated impulse-response operations
dgj/metrics/                        post-convergence statistics
dgj/experiments/                    one-session and cell runners
```

The accelerated period kernel retains the required order: choose actions, add noise, price from old data, calculate investor demand and profits, append the completed row, form the next state, and update learning. Randomness is generated outside the compiled kernel in independent arrays so scheduling and compilation cannot change draw order.

The accelerated engine uses NumPy's PCG64 generator through `SeedSequence(entropy, spawn_key=(cell32, session, stream))`. Here `cell32` is derived from the SHA-256 experiment-cell key, while session and stream identifiers are supplied as spawn-key coordinates. This is distinct from the readable engine's seven named SHA-256-derived stream seeds; the two implementations therefore require behavioral parity tests rather than an assumption that their raw random sequences are identical.

Numba is optional at import time; without it, the same functions fall back to ordinary Python. The fallback is useful for correctness tests but not expected to be practical for billion-period sessions.

### Experiment-cell identity

`dgj/config.py` stores paper parameters and disclosed choices in an immutable `ExperimentCell`. Its stable hash excludes only the descriptive label. The candidate baseline selects:

```text
prehistory = nash
price mapping = nearest
price grid = per_value
training tie rule = uniform
measurement tie rule = lowest_index
```

Low and high cells must differ only in `sigma_u` and their descriptive cell identity. A different grid, prehistory, tie rule, or parameter is a different experiment and must be stored separately.

The fields `price_mapping`, `training_tie_rule`, and `measurement_tie_rule` are included in the candidate cell hash, but the current accelerated kernels do not dynamically dispatch alternative implementations for these labels. Their behavior is presently hardcoded to nearest-price mapping, uniform training ties, and lowest-index measurement ties. Until dispatch and tests are added, changing one of these labels may change the cell key without changing the executed behavior; such a cell cannot be treated as a valid sensitivity experiment.

### Accelerated parity evidence

The accelerated repository contains 33 test methods covering the environment, benchmarks, rolling OLS, price-grid detectability, state and action spaces, Q updates, random streams, period order, freeze behavior, metrics, IRFs, and checkpoints. The parity tests compare grids and initial Q values with the readable engine and compare a 300-period supplied path to tight numerical tolerance. That supplied-path test first sets visit counters to `10^9`, which effectively removes exploration. It therefore tests a nearly deterministic execution path, not stochastic training-path parity. It may also skip when the initial Q table contains exact ties, and it is conditionally skipped if the readable `steps` directory is unavailable.

This is meaningful but not yet sufficient for formal acceptance. Three hundred periods do not cross the entire multi-billion-period convergence path, every checkpoint cadence, every OLS resynchronization, or every rare numerical edge case. Before formal interpretation, selected same-seed sessions must also demonstrate checkpoint/restart parity and invariant aggregate metrics.

Contemporaneous terminal output during preparation of this diary reported execution of all 33 methods. Thirty-two passed; the checkpoint-roundtrip test ended in a Windows temporary-filesystem `PermissionError`. No numerical assertion failure was reported, but the blocked test does not count as a pass. A clean compute-node test record must therefore be archived before the accelerated checkpoint path is described as verified on Narval.

### Candidate local pilot results

Four stored one-session pilots use the value-specific grid and paper-length convergence/measurement criteria:

| Noise | Experiment seed | Sessions | Convergence period | `Delta^C` | Trading intensity |
|---|---:|---:|---:|---:|---:|
| 0.1 | 20260828 | 1 | 3,091,841,692 | 0.319221 | 158.113 |
| 100 | 20260828 | 1 | 1,535,973,503 | 0.747638 | 141.859 |
| 0.1 | 20260829 | 1 | 2,926,303,926 | 0.356050 | 156.210 |
| 100 | 20260829 | 1 | 1,444,311,010 | 0.750344 | 141.666 |

These stored artifacts report convergence under the programmed stability criterion and contain measurement files for these particular paths. They are consistent with the accelerated engine reaching that criterion, but they are not independent proof and do not estimate the population distribution because there are only two sessions per noise regime.

The four source summaries are:

```text
outputs/pilot_exact_per_value_sigma_u_0p1/summary.json
outputs/pilot_exact_per_value_sigma_u_100/summary.json
outputs/pilot_rerun_per_value_seed_20260829_sigma_u_0p1/summary.json
outputs/pilot_rerun_per_value_seed_20260829_sigma_u_100/summary.json
```

In these pilots, high-noise `Delta^C` exceeded low-noise `Delta^C`, contrary to the expected ordering under the current reading of the paper. This discrepancy is not resolved by altering the code to force the expected sign. It motivates the planned independent-session campaign, uncertainty quantification, parity audit, and sensitivity checks.

The pilot summaries show zero for all mechanism shares because their per-session mechanism field is null. Null-labelled sessions remain in the denominator but enter none of the three category numerators; consequently, the displayed shares sum to zero and are not valid mechanism estimates.

### Accelerated output and checkpoint design

`dgj.experiments.run_session_cli` runs one session and writes:

```text
cell.json
ckpt_<session>.npz       while incomplete
session_<session>.npz    after terminal processing
```

The result contains 100,000 measurement rows, convergence status, a manifest, and optional IRF arrays. The array script maps one task to one session and checkpoints after groups of chunks. Chunk size is scientifically relevant in this implementation: the session runner pre-draws an entire random chunk and discards the unused tail if convergence is reached inside it. Changing chunk size can therefore change the post-convergence random continuation and measurement rows even when the same policy converges. The revised result manifest records chunk size, checkpoint cadence, per-invocation work budget, absence of a cumulative scientific cap, and IRF path count. The checkpoint separately binds and verifies cell key, session index, experiment seed, schema, and training chunk size.

The committed original launcher defaulted to a cumulative limit of five billion training periods, although the exact command-line override, if any, used by the first Narval job has not yet been recovered from its submission receipt. This value was an engineering default rather than a rule in the source paper. The first low-noise campaign showed that treating any cumulative computational limit as a terminal scientific outcome generated material right-censoring. The revised protocol therefore removes a cumulative scientific endpoint while retaining finite, checkpointed Slurm invocations. An invocation that ends before convergence writes no measurement artifact, preserves its checkpoint, and resumes under the same session identity, seed, economic kernel, parameters, environment, random-stream state, chunk size, and scientific-source fingerprint. Fifty billion periods, the upper end of the convergence range reported by the paper, is retained as an operator-enforced diagnostic review point rather than an automatic declaration of convergence or failure.

### Accelerated-engine risks discovered before aggregation

The version of `dgj_sim` used by the first low-noise campaign did not yet provide the same evidence protections as the readable Step 36 pipeline. The following risks are retained as a historical audit record; the locally corrected lifecycle described below addresses the operational items, but it was not yet committed, deployed, or live-tested on Narval at the time of this entry:

1. **Environment is inherited.** Module and virtual-environment activation lines in `hpc/submit_array.slurm` are commented. A task may therefore use a different Python, NumPy, Numba, or LLVM environment from the one intended.
2. **Dependencies are not pinned exactly.** `requirements.txt` specifies `numpy>=1.26` and `numba>=0.59`, not immutable versions.
3. **Artifacts omit full computational provenance.** Session manifests do not contain the Git commit, package versions, hostname, Slurm IDs, or hardware identity.
4. **Shared checkout risk.** Array tasks import from the current shared checkout when they start. Editing or pulling that checkout while tasks remain queued could make different sessions use different source code.
5. **Output-directory reuse is insufficiently guarded.** An existing `cell.json` is not compared strictly with new command-line arguments, and an existing result is skipped. Reusing a directory can therefore mix incompatible cells.
6. **Writes are not atomic.** A killed job can leave a partial checkpoint or result. File existence alone is not proof of integrity.
7. **Censored sessions are dangerous.** Unless `--no-measure-if-censored` is supplied, the CLI measures the current unconverged policy, writes a result with `converged_at=-1`, removes its checkpoint, and exits with code 3. A later rerun sees the result and exits without continuing.
8. **Aggregation accepts incomplete evidence.** The current aggregator summarizes however many session files exist and includes censored sessions in its means. It does not require all 1,000 indexes or zero censoring.
9. **The saved `summary.json` omits the added censorship count.** `aggregate_dir.py` adds `censored_sessions` only to its in-memory/stdout object after the underlying summary has already been written.
10. **Mechanism shares can be misleading.** Null mechanism labels cause the three printed shares to sum below one instead of raising an error.
11. **IRFs may run unintentionally.** The CLI default is the paper's 10,000 IRF paths unless `--irf-paths 0` is supplied. A nominal core run can therefore perform later mechanism work and take longer than expected.
12. **Numba cache collision is possible.** The candidate cache path contains the array job ID but not the individual task ID, so simultaneous tasks on one node may share a cache directory.
13. **The Slurm log directory must pre-exist.** Slurm can attempt to open stdout/stderr paths before the job script runs, so creating `logs/` inside the script may be too late.
14. **Submission-directory dependence is hidden.** The script sets `PYTHONPATH=$PWD`; it must therefore be submitted from `03_Code/dgj_sim` unless the path logic is made absolute.
15. **The allocation account is implicit.** The script does not record an `--account` value, so account selection depends on the submission environment.
16. **Array tasks can race on `cell.json`.** Multiple tasks may attempt to create the same non-atomic file concurrently.
17. **Checkpoint identity is not validated before loading.** The checkpoint does not bind and verify cell key, parameters, experiment seed, and session index before state restoration.
18. **The cache uses `TMPDIR`.** The script does not explicitly prefer Narval's job-local `SLURM_TMPDIR`, reducing isolation between tasks.

The accelerated `dgj/game/irf.py` is not yet equivalent to the formal readable IRF method in A20–A21. It classifies treatment responses relative to the paired control, uses the shortcut calibration `target * E[p_tilde] / E[lambda]`, and defaults to `irf_seed=7`. It therefore produces exploratory mechanism evidence, not the primary finite-sample level-target and long-run-normalization specification. For the present core low/high campaign, `--irf-paths 0` is mandatory. Formal IRFs must wait for implementation and parity of the A20–A21 definitions.

Commit `9ab452f` also contains a commit-message claim of 1,000 low-noise sessions with mean `Delta^C = 0.295` and ten high-noise sessions with mean `Delta^C = 0.707`. No corresponding production artifacts are present in the audited tree, so the claim is unverified and is not used as evidence. The same message gives the interval `2000000–2000099`, which contains 100 integer session identifiers rather than ten; that inconsistency must be resolved from original run receipts if those results are ever recovered.

These points do not prove that the economic calculations are wrong. They define the evidence gate that must be passed before the accelerated files are treated as formal research output.

### Post-campaign correction to the accelerated runner

After the 17.8 percent censoring rate was observed, the accelerated lifecycle was corrected locally rather than increasing one cumulative cap to another. `run_session_cli` now accepts `--work-periods`, which limits only the additional work performed by the current invocation. At the end of an incomplete invocation it atomically writes an identity-bound checkpoint and a progress receipt, leaves the phase as training, performs no measurement or IRF, creates no `session_XXXX.npz`, and returns the standard temporary-failure code 75. The Slurm wrapper treats this as a safe computing pause. Resubmission loads the checkpoint and adds another work slice; it does not compare the cumulative period counter with a terminal scientific maximum.

Measurement and result publication are reachable only from a genuinely converged phase. Checkpoints and result files are written beside their destinations and atomically replaced after complete serialization. Existing result files are validated for schema, convergence, internal period counts, measurement shape, full cell identity, seed, session identity, scientific-source fingerprint, and Python/NumPy/Numba runtime identity before they are skipped. A per-session operating-system lock rejects overlapping Slurm jobs, and a fingerprint of the trajectory-defining source files prevents continuation under silently changed code. `cell.json` is atomically published and checked against every task's command-line cell. The core IRF default is now zero, the Numba cache includes the individual array-task identity and prefers `SLURM_TMPDIR`, and formal strict aggregation—with the mandatory `--expected-sessions 1000` option—refuses any checkpoint, censored result, identity mismatch, missing index, or unexpected index. Each `summary.json` separately records the SHA-256 identity of the collusion, trading-policy, market-quality, and aggregation code, so later metric-code revisions cannot masquerade as the original analysis. Mixed or absent IRF evidence yields a null mechanism-share field rather than a misleading partial share. The historical capped outputs remain unchanged as audit evidence; accepted legacy artifacts are independently copied and SHA-256-verified into a new recovery cohort.

The corrected local suite contains 44 passing automated tests, including deterministic pause/resume parity, prompt scheduler-stop handling, convergence-checkpoint recovery, the enforced 50-billion-period review boundary, identity and chunk-size checks, scientific-source mismatch rejection, overlapping-owner rejection, stale-progress repair, strict censored-result rejection, exact recovery-index enforcement, and recovery-copy/hash verification. This is implementation evidence, not deployment evidence: a commit and push, a Narval pull, environment receipt, and one-task live `USR1` checkpoint/resume pilot are still required before the repaired campaign is launched. The legacy `run_cell` interface remains available only behind an explicit `--debug-only` acknowledgement and is not authorized for formal cluster experiments.

---

# Part VI. Narval deployment diary

## Connection and repository state

The project was cloned to the Narval scratch filesystem. The operator verified the repository commit as:

```text
9ab452fb6dc54e7ce25a9ec9417e346aa177d366
Add dgj_sim engine and Narval HPC scripts
```

Git history confirmed that this commit contains the earlier Step 36G implementation. The available Alliance allocations shown by `sacctmgr` were:

```text
def-cbravo_cpu
def-cbravo_gpu
```

The CPU account is appropriate for the one-CPU jobs described here.

The first activated home virtual environment did not contain NumPy. A dedicated Python 3.13 environment with the required package version was therefore planned for Step 36G. The final environment actually inherited by the current accelerated jobs has not yet been independently recovered in this diary and must be archived from job logs or the compute environment.

## Current run status as of the latest diary update

The low-noise array was reported under Slurm job `2099411`, with results in `$SCRATCH/ai-trading-runs/low_noise_per_value`. The directory contained exactly 1,000 `session_XXXX.npz` artifacts and no remaining ordinary checkpoints. The first unfiltered aggregation then reported 178 censored sessions. Thus 822 sessions met the programmed convergence criterion before the operational cap, while 178, or 17.8 percent, did not.

The existing aggregator nevertheless included every cap-time policy in its economic means. Its reported `Delta^C = 0.4155713011` and the other market statistics are therefore diagnostic mixed-sample quantities, not formal low-noise replication estimates. The reported convergence median is also invalid because the aggregator inserts censored values as `-1` before taking the ordinary median. The separate experiment record is `docs/experiments/2026-08-30_low_noise_campaign_log.md`.

The exact submission command, final per-task `sacct` receipt, package environment, operational cap, chunk/checkpoint arguments, and IRF argument still require archival confirmation. The correct status is **candidate low-noise campaign completed at the artifact level but rejected by the zero-censoring gate**. The 178 paths must be rerun from their original deterministic identities and then continued through finite checkpointed invocations until they satisfy the unchanged-policy criterion. Because the original capped runner deleted their cap-time checkpoints, recovery requires deterministic replay from period zero and continuation beyond the original cumulative cap when necessary; the committed default was five billion periods, but its application to job `2099411` remains subject to receipt confirmation. The original artifacts remain preserved audit evidence of the first capped campaign. The high-noise campaign has not yet been reported as started.

The original output directory and source checkout identity must be preserved. Recovery and high-noise work should use new output directories and a frozen committed source tree so that no existing evidence is overwritten or silently mixed.

## Post-run acceptance protocol

Low and high noise must pass the following gate independently before aggregation.

### 1. Scheduler and provenance gate

Archive:

- Git commit and clean/dirty status;
- exact `sbatch` command;
- Slurm job and array IDs;
- `squeue` and final `sacct` state;
- requested and observed CPU, memory, and wall time;
- stdout and stderr logs;
- hostname or node information;
- Python, NumPy, Numba, llvmlite/LLVM versions; and
- environment/module list;
- submission working directory and allocation account;
- training chunk size and checkpoint cadence;
- per-invocation period budget and Slurm wall time, cumulative periods reached, checkpoint cadence, diagnostic review thresholds, and confirmation that no cumulative scientific cap was applied; and
- IRF path count, IRF seed, and every other mechanism argument, including an explicit `--irf-paths 0` receipt for the core campaign.

Every task expected to produce a formal session must end in an understood state. `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, nonzero exit codes, and missing logs require investigation.

### 2. Cell-configuration gate

Low and high outputs must reside in separate new directories. Their `cell.json` files must match on every field except the intended noise standard deviation and descriptive label. The expected baseline includes:

```text
I = 2
rho = 0.95
xi = 500
n_v/n_x/n_p = 10/15/31
prehistory = nash
price grid = per_value
convergence periods = 1,000,000
measurement periods = 100,000
```

Under the audited candidate configuration, the expected label-independent cell keys are `0af80038fb4ea945` for low noise and `6d2e57c5993e710e` for high noise. A different key is not automatically wrong, but it signals a different parameter or implementation choice that must be explained before comparison.

### 3. Completeness gate

For a 1,000-session cell, require exactly

```text
session_0000.npz, ..., session_0999.npz.
```

There must be no missing, duplicated, unexpected, or mixed indexes. A scheduler array ending is not sufficient evidence because some tasks may have failed or timed out.

### 4. File-integrity gate

Every archive must open with `allow_pickle=False`, and every stored array must be read to trigger ZIP/CRC checks. For the two-agent baseline, the measurement array must have shape `(100000, 10)` and contain finite values. Required keys are `rows`, `converged_at`, and `manifest`, plus the expected IRF arrays if IRFs were intentionally enabled.

### 5. Convergence gate

Primary formal results require:

```text
converged_at >= 0
manifest.censored = false
censored sessions = 0
```

Manifest and array convergence values must agree, and the completed-period count must equal convergence time plus 100,000 measurement periods. A censored file must be quarantined rather than aggregated or silently overwritten.

### 6. Identity and random-stream gate

Each file index must equal the manifest session index. The cell key, experiment seed, parameters, and seven random streams must match the cell plan. Session identities must be unique.

### 7. Economic accounting gate

All 100,000 raw measurement rows in every accepted session must reproduce the defining identities:

```text
y_t = sum_i x_i,t + u_t,
z_t = -xi(p_t-v_bar),
pi_i,t = (v_t-p_t)x_i,t.
```

Metric calculations should be rerun from raw rows rather than trusted only because a summary file exists.

### 8. Reproducibility gate

A selected subset of session identities should be rerun from fresh state and, separately, from checkpoints. Their numerical arrays and normalized manifest content must match the originals under the recorded environment. Byte-for-byte equality of the outer `.npz` ZIP archives is not required because archive metadata can differ even when every stored array and manifest field is identical.

### 9. Aggregation gate

Only after all earlier gates pass should cell-level summaries be calculated. Formal reporting should include:

- number of planned, completed, converged, censored, and excluded sessions;
- mean, median, and quantiles of `Delta^C`;
- convergence-time distribution;
- trading intensity and benchmark coefficients;
- price informativeness;
- liquidity and singularity diagnostics;
- mispricing and ambiguity diagnostics; and
- mechanism shares only when valid mechanism receipts exist.

One value from one session is never treated as the low- or high-noise result.

## Read-only operational checks

The following commands do not change a running experiment and should be captured in the run record:

```bash
git rev-parse HEAD
git status --short --untracked-files=no
squeue -u "$USER" -o "%.18i %.30j %.8T %.10M %.10L %R"
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ReqMem,MaxRSS,ExitCode
```

After the arrays stop, the basic file-count check is:

```bash
find "$LOW_OUT"  -maxdepth 1 -type f -name 'session_[0-9][0-9][0-9][0-9].npz' | wc -l
find "$HIGH_OUT" -maxdepth 1 -type f -name 'session_[0-9][0-9][0-9][0-9].npz' | wc -l
find "$LOW_OUT"  -maxdepth 1 -type f -name 'ckpt_[0-9][0-9][0-9][0-9].npz' | wc -l
find "$HIGH_OUT" -maxdepth 1 -type f -name 'ckpt_[0-9][0-9][0-9][0-9].npz' | wc -l
```

Counts alone are not acceptance; every archive and manifest still requires the integrity, convergence, and identity checks above. `aggregate_dir.py` must not be run as the formal summarizer until that gate has been enforced, because its current implementation accepts partial and censored inputs.

## Planned sequence after both noise campaigns

The next sequence is:

```text
finish low-noise campaign
        -> finish high-noise campaign
        -> archive scheduler and environment provenance
        -> validate every session and reject censoring/corruption
        -> establish accelerated/readable parity
        -> aggregate low and high independently
        -> compare distributions with the paper
        -> reproduce the core baseline figure and source-data table
        -> run pre-specified grid/prehistory sensitivities
        -> run and classify full IRFs if mechanism analysis remains in scope
        -> produce final figures, tables, and written results
```

If the currently running campaign contains fewer than 1,000 valid independent sessions per cell, it is retained as a pilot and used to refine scheduling. If it already contains 1,000 sessions per cell, it can become a formal candidate only if the source, configuration, environment, and post-run gates can be reconstructed and passed. A failed gate may require a rerun from a corrected, frozen commit; it must not be hidden by selective exclusion.

---

# Part VII. Replication decision register

This section records choices that software must make even though the paper does not fully specify them. They must remain visible in the methodology and, where material, be examined through sensitivity analysis.

## A1 — Training-time Q ties

The paper states `argmax` but gives no tie rule. The baseline selects uniformly among exact maximizers using the acting agent's private tie stream. Near-equal values are not ties. Measurement ties are handled separately under A9.

## A2 — Mapping continuous price into the finite state

Select row `P(v_(t-1))`, map to the nearest point, clip outside the row to its endpoint, and choose the lower index at an exact midpoint. Retain the continuous price for all economic calculations. Record clipping rates in formal runs.

## A3 — Initial market-maker history

The paper does not describe the initial 10,000 observations. The baseline uses a balanced Nash-consistent synthetic prehistory from which the maker recovers coefficients through its own OLS. Required sensitivities are cartel-consistent and expanding-window starts.

## A4 — Random-stream architecture

The paper does not publish seeds, generator, or stream assignment. The readable engine uses stable experiment/cell/session identities and seven private SHA-256-derived streams. Its formal artifacts must record the experiment seed, cell key, session index, derived seeds, derivation version, Python version, and RNG engine. The accelerated engine instead uses PCG64 through NumPy `SeedSequence` with `(cell32, session, stream)` as spawn-key coordinates; only `cell32` comes from the SHA-256 cell key. These architectures are documented separately and require outcome-level parity checks.

## A5 — Economically admissible fixed-point root

For positive inputs, select the unique positive root inside the analytical gamma-bound interval. Bisection is the transparent numerical solver. The `xi=0` boundary uses its exact positive root and must satisfy the original equations.

## A6 — High response threshold OCR ambiguity

Visual inspection of the original appendix page established `x_low=5e-5` and `x_high=10*x_low=5e-4`. Both rules apply strict inequalities to both informed agents.

## A7 — Definition of unchanged policy

The policy is the complete mapping from every state to its exact set of maximizing actions. A period is stable only if both agents' sets remain unchanged after both Q updates. Any set change resets the shared streak to zero; the changed period is not counted as stable.

## A8 — Convergence-to-measurement boundary

The convergence-reaching period remains training. Measure exactly `T_c+1,...,T_c+100000`, carry market state and environment streams forward without reset, freeze Q/exploration/visits, and allow the maker to keep rolling. Sensitivities should include continued background Q-updating and a frozen maker.

## A9 — Measurement-time Q ties

At convergence, select the lowest index in each exact maximizing set once and use that pure action throughout measurement. This consumes no trader RNG. A future sensitivity should sample uniformly from the unchanged maximizing set.

## A10 — Undefined collusion denominator

If `mean(pi^M)-mean(pi^N)` is non-positive or smaller than 64 floating-point units at the observed profit scale, report `Delta^C` as undefined. Do not take an absolute value, add epsilon, or clip a valid result.

## A11 — Trading-intensity regression

Estimate one unweighted OLS with a free intercept for each agent, then average agent slopes. Do not impose the theoretical intercept. If measured values lack meaningful variation, report the slope as undefined.

## A12 — Liquidity sum versus mean

The prose says average but the display omits `1/T`. Use the arithmetic mean over exactly the 100,000 measurement rows as the main `L^C`, while preserving the literal unnormalised sum.

## A13 — Liquidity singularity

Evaluate `1-xi*lambda_hat_t` with one-rounding fused arithmetic where available. A true zero becomes positive infinity with diagnostics. Do not add epsilon, impose a tolerance, or clip a large finite reciprocal.

## A14 — Mispricing sum versus mean

Use the arithmetic mean over exactly the measurement window as `E^C` and preserve the literal unnormalised sum, following the same disclosed interpretation as A12.

## A15 — Missing absolute value in printed mispricing expansion

Compute and retain both the printed signed expression and the economically absolute definition. If they disagree, do not silently correct the paper or report negative mispricing; mark the main value undefined and record the affected periods.

## A16 — Missing parentheses in the low-response rule

Use `abs((x_tilde-E[x_tilde])/E[x_tilde])<5e-5`. This matches the dimensionally coherent normalized response shown elsewhere. The interpretation remains disclosed.

## A17 — Construction of IRF paths

The primary implementation uses lossless converged checkpoints, same-session long-run denominators, common ordinary draws, an additive treatment shock only at `t=3`, frozen Q/policy/visits, and independently rolling makers. Remaining sensitivities include fork timing, shared checkpoint `v_1`, frozen maker, continued Q-learning, pooled denominators, and the shortcut shock.

## A18 — Meaning of cloning a converged session

Clone all causal mutable state: Q, policies, values, visits, exact rolling rows/statistics/rebuild phase, and all RNG states. Restore into detached objects and never copy live controllers or sinks. Bind exact continuation to platform and source identities.

## A19 — Timing of one paired path

Treat the completed convergence outcome as local `t=0`, execute transactions `t=1..4`, and shock only treatment at `t=3`. Carry checkpoint `p_0,v_0,v_1`, share ordinary draws across branches, and keep each maker on its own history. The four-period unshocked branch is not the long-run expectation.

## A20 — Finite samples used for IRF normalization

Use each session's own 100,000 frozen-policy rows for long-run denominators. Pool price moments by raw row counts for cell shock calibration. Classify each session after averaging its 10,000 treatment paths and normalizing by that session's own long-run order means. A pooled-order denominator is a sensitivity only.

## A21 — Meaning of the 1.2 percent shock target

Use the exact finite-sample price-level target rather than only a treatment-control increment. Retain the increment shortcut and its achieved level as a sensitivity. Reject a cell when a positive adverse shock cannot solve the level target.

## A22 — Global versus value-specific price grid

The primary reading uses one 31-point row `P(v_k)` for each fixed value. This restores the footnote's one-action-step/one-price-step resolution in low noise without adding information to the state. The global grid is a named sensitivity and old global-grid results cannot be pooled with the new model.

## A23 — Scientific source fingerprint

The readable formal pipeline separately hashes the exact recursive execution-source closure and result-pipeline closure using normalized bytes and domain-separated SHA-256. A model-code change invalidates affected evidence; an unrelated orchestration addition does not.

## A24 — Persisted bridge from completed sessions to mechanism work

Step 36F stores the Step 35D receipt plus references and fingerprints for measurement evidence, convergence origin, and replay source. Reload rebuilds the scorer/checkpoint chain, reruns Step 35D, and requires exact equality. Full 1,000-session IRF orchestration remains open.

---

# Part VIII. Validation philosophy and evidence ledger

## Validation hierarchy

Validation was deliberately layered:

1. **Hand arithmetic.** Small numbers test the intended formula independently of other modules.
2. **Unit tests.** Boundary, sign, shape, and mutation behavior are checked for one function or class.
3. **Integration checkpoints.** Previously verified components are connected using fixed inputs.
4. **Path parity.** A lean or accelerated path is compared with a readable oracle under identical supplied randomness.
5. **Serialization parity.** Uninterrupted execution is compared with save/load/resume execution.
6. **Full measurement parity.** Online constant-memory metrics are compared with independent batch calculations over 100,000 rows.
7. **Experiment acceptance.** Scheduler, environment, session completeness, convergence, artifact integrity, provenance, and aggregation are checked before any research claim.

Passing a lower layer does not imply passing a higher one. For example, a correct Q-update unit test does not establish that a session converges, and a converged session does not establish a population result.

## Dedicated automated evidence for Steps 35A–36G

The later evidence pipeline has dedicated test modules in addition to the readable examples. Their principal scope is recorded here so that the word “verified” can be traced to an actual file:

| Step | Validation file(s) | Principal evidence encoded |
|---|---|---|
| 35A | `tests/test_step35a_converged_market_checkpoint.py` | lossless state capture, detached restore, continuation, and tamper rejection |
| 35B | `tests/test_step35b_paired_irf_path.py` | paired-path timing, common draws, treatment-only shock, branch isolation, and `p_3 -> s_4 -> x_4` wiring |
| 35C | `tests/test_step35c_irf_long_run_baseline.py` | long-run finite-sample means, sink/checkpoint binding, and receipt integrity |
| 35D | `tests/test_step35d_unshocked_t3_calibration_paths.py`; `tests/test_step35d_reversible_transactions.py` | canonical path accounting and exact rollback, including OLS rebuild boundaries |
| 35E | `tests/test_step35e_cell_shock_calibration.py` | cell pooling, exact-level calibration, shortcut sensitivity, and invalid-target rejection |
| 35F | `tests/test_step35f_paired_response_and_classification.py` | paired response construction, classifier thresholds, rollback, and debug-scale achieved target |
| 36A | `tests/test_step36a_one_session_result_row.py` | structured smoke artifacts and explicit non-research flags |
| 36B | `tests/test_step36b_experiment_manifest.py`; `tests/test_source_manifests.py` | deterministic task plans, immutable manifests, overwrite/tamper guards, and separated source closures |
| 36C | `tests/test_step36c_exact_training_resume.py` | uninterrupted-versus-resumed path equality, ownership separation, and corruption/wrong-task rejection |
| 36D | `tests/test_step36d_single_session_training_runner.py` | fresh/resumed work boundaries, durable status, checkpoint retention, and failure handling |
| 36E | `tests/test_step36e_complete_measurement_runner.py` | exact 100,000-row measurement, deterministic restart, metrics, evidence bundles, and checkpoint pinning |
| 36F | `tests/test_step36f_persisted_calibration_bridge.py` | cross-process reconstruction, exact metric replay, source binding, and Step-35D receipt equality |
| 36G | `tests/test_step36g_narval_throughput_benchmark.py` | sandbox isolation, exact checkpoint handoff, period accounting, one-CPU guard, and report semantics |

Steps 25–29 instead rely substantially on embedded assertions that run when their modules are executed directly, together with indirect coverage in later integration tests. The `unittest discover` count below must not be read as evidence that every embedded `main()` assertion was executed in that command.

## Recorded readable-engine validations

### Repository Baseline R0

On 30 August 2026, the readable implementation through Step 36F was validated locally with:

```text
Operating system: Windows
Python: 3.13.1
NumPy: 2.5.1
Command: py -3 -X utf8 -m unittest discover -s tests -v
Result: 253 tests passed; 0 failures; 0 errors; 0 skips
Runtime: approximately 68.7 seconds
```

This demonstrates consistency with 253 encoded checks, not paper-scale convergence or empirical replication.

### Step 36G validation

The Step 36G implementation was validated with nine focused tests and the expanded full suite:

```text
Focused Step 36G: 9/9 passed
Full readable suite: 262/262 passed
Full-suite runtime: 238.662 seconds
```

A real local connection smoke ran one fresh training period and resumed the same checkpoint for exactly two more periods. It ended with three cumulative training periods, zero measurement rows, a local-smoke scope, no million-period extrapolation, and research flags set to false.

### Accelerated-engine validation attempt during diary preparation

The `dgj_sim` suite contains 33 test methods. Contemporaneous terminal output during this diary preparation reported 32 passes and one checkpoint-roundtrip error caused by a Windows temporary-filesystem permission failure. No numerical assertion failure was reported, but the blocked test is not counted as passed. This incident reinforces the need to record the actual compute-node filesystem and environment rather than relying only on source-level claims.

## Validation record template for future runs

Every future entry should record:

```text
Date and time:
Git commit:
Scientific source hash:
Python and package versions:
Operating system / host / Slurm job:
Exact command:
Parameters and experiment-cell key:
Output directory:
Observed result:
Pass/fail status:
What this proves:
What this does not prove:
```

Every experiment must also be labeled as one of:

```text
engineering smoke test
debug-scale experiment
candidate formal experiment
validated formal paper-scale experiment
```

---

# Part IX. Planned analysis and reporting

## Core low/high comparison

The first inferential deliverable will compare validated low-noise and high-noise cells. The source-data table will contain one row per independent session, with at least:

```text
cell and session identity
convergence period
censoring and inclusion status
Delta^C and component profits
trading intensity
price informativeness
liquidity diagnostics
mispricing diagnostics
software and source provenance
```

Cell summaries will report the number of sessions and distributional uncertainty, not only a mean. The primary visual output will reproduce the paper's baseline low/high comparison from this validated table. Plot code will consume immutable source data rather than rerun the simulation.

The inferential specification is not yet frozen. Before inspecting completed cell summaries, the project must pre-register the primary low-versus-high estimands, the uncertainty or confidence-interval method, the treatment of undefined and excluded sessions, the exact paper comparison targets, and a quantitative rule for classifying each finding as numerically reproduced, qualitatively matched, or not reproduced. Freezing these choices before seeing the aggregate results prevents the reporting rule from being adjusted to favor a desired conclusion.

## Discrepancy protocol

If the low/high ordering or magnitude differs from the paper, the investigation order is:

1. confirm session count and censoring;
2. confirm exact parameters and cell keys;
3. confirm price-grid interpretation and clipping/detectability;
4. confirm Q initialization and expected-next-value update;
5. confirm training and measurement tie rules;
6. confirm rolling OLS timing and initial history;
7. confirm convergence boundaries and frozen-policy measurement;
8. confirm random-stream independence and reproducibility;
9. confirm matched-path benchmark reconstruction; and
10. run pre-specified sensitivities without overwriting the primary specification.

No result is discarded merely because its sign is surprising.

## Mechanism and robustness sequence

After the core comparison is secure:

1. complete full-cell Step 35D–35F orchestration;
2. estimate price-trigger, over-pruning, and unclassified session shares;
3. compare value-specific and global price grids;
4. compare Nash, cartel, and expanding-window maker initialization;
5. examine alternative measurement tie behavior, continued Q updating, and frozen-maker measurement;
6. reproduce additional comparative statics in `I`, `rho`, `xi`, and other paper settings; and
7. generate the remaining figures and any appendix-oriented tables.

## Reproducible reporting rule

Every final figure or table must be linked to:

- an immutable session-level source-data file;
- an experiment receipt or provenance manifest;
- the exact code commit and environment;
- the aggregation and plotting script; and
- a short statement of exclusions, censoring, and unresolved assumptions.

The final paper will distinguish clearly between a result reproduced numerically, a qualitative pattern only, a result that differs from the source paper, and a result that remains unverified.

---

# Closing methodological reflection

The replication was built by treating economic modeling, learning dynamics, measurement, and computation as separate layers. The market equations were first implemented in hand-checkable functions. The finite state and action spaces were then built from theoretical benchmarks. Q-learning was introduced only after the market arithmetic and state indexing were stable. The adaptive market maker was first written as transparent OLS and only later optimized. A complete causal period preceded randomness, convergence, and measurement. Metrics were attached only after the frozen-policy session lifecycle was explicit. IRF logic was decomposed into checkpointing, paired paths, long-run denominators, calibration, and classification before any paper-scale mechanism run was attempted. Finally, experiment manifests, exact resume, artifact validation, and cluster deployment were treated as part of the methodology rather than incidental software work.

This approach provides strong internal computational evidence, but internal correctness is not the same as external validity or empirical replication. The first low-noise Narval campaign exposed 17.8 percent right-censoring and therefore remains diagnostic while the affected paths are recovered. The high-noise campaign has not yet been reported as started. Formal conclusions will be written only after both cells pass the complete acceptance protocol and their validated session-level distributions are compared with the paper.
