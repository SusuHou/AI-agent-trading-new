# Experiment Log: Baseline Low-Noise Campaign and Censoring Diagnosis

**Project:** Replication of *AI-Powered Trading, Algorithmic Collusion, and Price Efficiency*  
**Experiment cell:** Baseline low-noise environment  
**Noise standard deviation:** `sigma_u = 0.1`  
**Execution platform:** Narval cluster, Digital Research Alliance of Canada  
**Slurm array job reported by the operator:** `2099411`  
**Result directory:** `$SCRATCH/ai-trading-runs/low_noise_per_value`  
**Date recorded:** 30 August 2026  
**Evidence status:** Candidate experiment; diagnostic but not yet a valid formal cell result

## 1. Purpose of the run

The purpose of this campaign was to estimate the distribution of learned outcomes across 1,000 independent sessions of the paper's baseline low-noise market. Each session contained two independently learning informed speculators, a Gaussian noise trader, price-responsive information-insensitive investors, and an adaptive market maker estimated from a rolling history. The intended convergence criterion required the complete greedy policies of both informed agents to remain unchanged for 1,000,000 consecutive periods. Once convergence was reached, the agents' policies were frozen and the market was measured for a further 100,000 periods.

This run was intended to provide the low-noise half of the central low-versus-high comparison. It was not intended to calibrate the code until it produced a preferred value of the collusion statistic.

## 2. Intended configuration

The intended scientific configuration was:

| Setting | Value |
|---|---:|
| Informed speculators, `I` | 2 |
| Fundamental-value grid points, `n_v` | 10 |
| Action points, `n_x` | 15 |
| Price-state points per value, `n_p` | 31 |
| Noise standard deviation, `sigma_u` | 0.1 |
| Investor-demand slope, `xi` | 500 |
| Discount factor, `rho` | 0.95 |
| Learning rate, `alpha` | 0.01 |
| Exploration-decay parameter, `beta` | `5e-7` |
| Market-maker memory, `T_m` | 10,000 periods |
| Price grid | Value-specific |
| Market-maker prehistory | Nash-consistent |
| Required policy-stability streak | 1,000,000 periods |
| Post-convergence measurement window | 100,000 periods |
| Planned sessions | 1,000 |

The accelerated implementation was run from `03_Code/dgj_sim`. The Narval checkout had previously been reported at commit `9ab452fb6dc54e7ce25a9ec9417e346aa177d366`, but the exact commit, dirty status, package versions, submission command, maximum-period argument, chunk size, checkpoint cadence, and IRF argument inherited by job `2099411` must still be preserved or reconfirmed from the job receipt and logs. The committed launcher defaults to a five-billion-period operational cap, one-million-period training chunks, and 50 chunks between checkpoints. These defaults should not be attributed to the completed run without checking its actual submission environment.

## 3. Completion evidence initially observed

After the array left the live Slurm queue, the result directory contained:

```text
session files: 1,000
checkpoint files: 0
```

This established that one terminal result artifact existed for every planned session index and that no ordinary checkpoint file remained. It did not, by itself, establish convergence. The accelerated command-line runner can reach its operational period cap, measure the current unconverged greedy policy, save a full `session_XXXX.npz` file, mark it as censored, delete the checkpoint, and exit with a nonzero status. Consequently, file counts are a necessary but insufficient completion test.

## 4. Initial aggregate output

The existing aggregator read all 1,000 session artifacts and reported:

| Statistic | Reported value |
|---|---:|
| Sessions found | 1,000 |
| Mean collusion statistic, `Delta^C` | 0.4155713011 |
| 1st percentile of `Delta^C` | 0.2301128507 |
| 99th percentile of `Delta^C` | 0.7665601368 |
| Mean learned trading intensity | 154.1427149267 |
| Nash trading intensity | 166.6666333333 |
| Cartel trading intensity | 124.9999500000 |
| Mean price informativeness | 8,368,025.9747 |
| Reported median convergence time | 1,848,170,478 periods |
| Mean liquidity | 4,101,188.9068 |
| Mean mispricing | 0.2965388949 |
| Reported price-trigger share | 0.741 |
| Reported over-pruning share | 0.000 |
| Reported unclassified share | 0.259 |
| Censored sessions | 178 |

The aggregate command therefore established the following operational result:

```text
1,000 artifacts were produced.
822 sessions satisfied the convergence criterion before the cap.
178 sessions were right-censored at the operational cap.
The observed censoring rate was 17.8 percent.
```

## 5. Why the printed economic statistics are not formal results

The current aggregation function does not exclude censored sessions. It calculates collusion, trading intensity, informativeness, liquidity, mispricing, and mechanism labels from every `session_*.npz` file that it finds. For a capped session, the saved 100,000-period sample was generated from the greedy policy that happened to exist at the cap, even though that policy had not satisfied the one-million-period stability criterion.

The reported `Delta^C = 0.4156` therefore combines 822 policies that satisfied the programmed stability criterion with 178 cap-time policies. The direction of the resulting bias is not known. Dropping the 178 sessions and averaging only the 822 would not solve the primary problem because convergence speed may be correlated with the learned strategy. An 822-session statistic would condition on convergence before the cap and could therefore be selection-biased. Such a calculation may be retained as a clearly labelled diagnostic, but it cannot replace the planned 1,000-session estimand.

The reported convergence median is also unsuitable for formal interpretation. The aggregator represents a censored convergence time as `-1` and includes those values in an ordinary numerical median. Thus `1.848 billion periods` is neither the median among converged sessions nor a censoring-aware estimate of the population median.

The mechanism shares are not treated as formal evidence. First, the accelerated IRF implementation is exploratory and does not yet reproduce every finite-sample calibration and normalization rule of the readable Step-35 specification. Second, the capped sessions were subjected to mechanism calculations from policies that had not converged. The displayed 74.1 percent price-trigger share must therefore not be used in the essay as a replicated mechanism estimate.

The remaining market-quality quantities are also provisional because the same mixed sample enters their averages. The very large informativeness value is not, by itself, proof of a coding error: the statistic scales inversely with the low noise variance. The very large mean liquidity is potentially dominated by observations for which `|1-xi*lambda_hat|` is close to zero and requires distributional and near-singularity diagnostics rather than interpretation through its mean alone.

## 6. Meaning of non-convergence in this experiment

The 178 sessions should not be described as having proven incapable of convergence. They are more precisely described as right-censored: their convergence times exceed the period observed under the operational cap, but their eventual convergence times remain unknown.

The policy-stability requirement is substantially stricter than repeatedly selecting the same realized action. Each agent has a policy over 3,100 market states, and every state admits 15 actions. After each Q update, the convergence tracker asks whether the exact set of maximizing actions changed for either agent. A change in one state for one agent resets the shared stability streak to zero. For example, a session can reach a streak of 999,999 periods and then return to zero if a rarely visited state causes one agent's maximizing action to switch.

Several features can generate a long convergence-time tail:

1. **Simultaneous multi-agent learning.** Each trader's policy affects the other trader's rewards and state transitions, so the learning environment is not stationary from either trader's perspective.
2. **An adaptive market maker.** The rolling OLS estimates continue to change with the most recent 10,000 observations, changing prices and rewards even when trading behavior is nearly stable.
3. **A constant learning rate.** With `alpha = 0.01`, Q-values can continue to move. When two actions have similar values, small reward changes can reverse their ranking without producing a large economic change.
4. **Uneven state visitation.** The 3,100 states are not visited uniformly. A rarely reached price-value configuration may update late and reset an otherwise long policy-stability streak.
5. **Path dependence.** Early exploration, noise orders, and interaction between the two learners can place different session seeds on learning paths with very different stabilization times.

The fact that 822 sessions converged under the same implementation shows that the convergence tracker can reach its target. The remaining 178 may reflect a genuine long tail, persistent policy oscillation, an operational cap that is too small, or an implementation issue affecting particular paths. The aggregate output alone cannot distinguish these explanations. The last reported stability streaks in the corresponding Slurm logs should be inspected: a streak near one million suggests near-convergence at the cap, whereas repeated low streaks suggest continuing policy changes.

Late random exploration is not automatically the main explanation. Exploration is value-specific and decays exponentially with visits. At billions of periods it is expected to be extremely small for frequently visited values. The more plausible late-stage mechanisms are changes in the ranking of nearly tied actions, updates in rare states, and continued interaction with the rolling market maker. This remains a hypothesis until the censored-session logs or additional diagnostic runs are examined.

## 7. Formal status decision

The low-noise campaign failed the adopted zero-censoring acceptance gate. It is classified as a candidate diagnostic campaign rather than a validated formal result.

The defensible research statement is:

> Of 1,000 low-noise sessions, 822 satisfied the programmed convergence criterion before the operational period cap, while 178 sessions, or 17.8 percent, were right-censored. Because the existing aggregator included cap-time policies in the economic summaries, the resulting collusion and market-quality statistics are treated as diagnostic rather than as final replication estimates.

The experiment is not discarded. It provides evidence about the convergence-time distribution, exposes a flaw in the handling and aggregation of censored runs, and identifies the session identities that require further execution.

## 8. Recovery decision

The original output directory will be preserved unchanged as evidence of the first capped campaign. The 178 censored artifacts will not be deleted, silently replaced, or pooled as converged observations.

Because the existing runner deleted the checkpoints associated with capped results, those 178 paths cannot resume from their original cap-time states. They must restart from their original experiment seed and session indexes. Same-seed execution is intended to reproduce the original paths, conditional on first verifying the original code, arguments, cell identity, chunk size, and software environment. The recovery runs must use:

- the same economic kernel, equations, RNG derivation, chunk size, parameters, and state-transition logic, together with a separately identified orchestration-only runner revision and deterministic prefix-parity validation against the original implementation;
- the same experiment seed and session indexes;
- the same low-noise cell and cell key;
- the same value-specific price grid and Nash prehistory;
- the same one-million-period convergence requirement;
- the same `chunk_size = 1,000,000`, because changing chunk size can change the unused random tail discarded at convergence and therefore the measurement continuation; and
- a new output directory so that the original evidence is never overwritten.

Subject to confirming that the original operational cap was the launcher's default of five billion periods, the initial recovery proposal used staged ten- and twenty-billion-period cumulative caps. This historical proposal is superseded by the protocol correction documented in Section 13. Recovery now uses finite per-invocation work slices with no cumulative scientific cap.

After recovery, a new evidence-controlled directory will contain exactly one validated result per session index: an independently copied and hash-verified artifact for each of the 822 initially converged sessions and a newly recovered artifact for each of the 178 initially censored sessions. A selection manifest will record the source path, convergence time, and cryptographic hash of every copied artifact. Formal aggregation will proceed only if the composed cohort contains exactly 1,000 valid sessions, zero censored sessions, correct identities, finite `(100000, 10)` measurement arrays, and no duplicate or missing index. Core measurement rows remain comparable because the optional accelerated IRF runs only after measurement; however, if the original and recovery artifacts contain different IRF path counts, all mechanism-share fields from the composed cohort will be excluded rather than interpreted.

## 9. Consequence for the high-noise campaign

The full 1,000-session high-noise campaign should not be launched under the same unguarded five-billion-period cap. The historical proposal for ten- and twenty-billion-period pilot caps is superseded by Section 13. Any high-noise pilot or full campaign must use finite checkpointed invocations, no cumulative scientific cap, no measurement before genuine convergence, and `--irf-paths 0` for the core comparison.

This pilot is an operational sizing exercise, not a substitute for the full high-noise cell. A larger per-invocation work budget changes only how much is computed before the next durable pause; it does not change the scientific stopping rule or force an early-converging session to continue.

## 10. Outstanding evidence to archive

Before this entry is used in the essay, the following evidence should be attached or confirmed:

- the final `sacct` task-state and exit-code counts for Slurm job `2099411`;
- the exact Git commit and tracked working-tree status used by the jobs;
- the exact `sbatch` command and exported variables;
- the original cumulative cap (committed launcher default: five billion periods, pending job-receipt confirmation), and for all revised runs the per-invocation budget, Slurm wall time, cumulative periods, checkpoint cadence, review thresholds, and absence of a cumulative scientific cap;
- Python, NumPy, Numba, and llvmlite versions from the compute environment;
- the SHA-256 hash of `cell.json` and the expected cell key;
- the list of the 178 censored session indexes;
- the original aggregate terminal output; and
- the repair and immutable-selection receipts.

## 11. Essay-ready methodological paragraph

> We initially executed 1,000 independent sessions in the low-noise environment. Although every session produced a terminal artifact, 178 sessions reached the operational period cap before satisfying the requirement that both agents' complete greedy policies remain unchanged for one million consecutive periods. The original aggregation routine nevertheless evaluated the cap-time policies and combined them with the 822 sessions that satisfied the programmed stability criterion. We therefore classified the first aggregate as diagnostic and did not treat its collusion or market-quality statistics as replication estimates. The censored session identities were preserved for planned same-seed recovery through finite checkpointed invocations, to be continued until the original convergence criterion is satisfied, conditional on verification of the original economic kernel, parameters, environment, random identities, state and action grids, and chunk size. Formal aggregation remains deferred until one validated criterion-satisfying artifact is available for every planned session index.

## 12. Results-language paragraph for the interim record

> The initial low-noise campaign produced 1,000 artifacts, of which 822 converged before the operational cap and 178 were right-censored, corresponding to a censoring rate of 17.8 percent. The unfiltered aggregation yielded a mean normalized collusion statistic of 0.416, with 1st and 99th percentiles of 0.230 and 0.767, respectively. Because these statistics combined converged strategies with policies observed only at the cap, they are reported solely as diagnostic evidence. No formal claim regarding the magnitude of low-noise collusion, its comparison with the high-noise environment, or the prevalence of a particular collusive mechanism is based on this mixed sample.

## 13. Post-campaign correction to the stopping protocol

A subsequent audit of the source paper established that the accelerated launcher's committed five-billion-period default was an engineering setting, not part of the paper's convergence definition; whether job `2099411` inherited that exact default remains pending confirmation from its archived submission receipt. The paper requires all simulation sessions to continue until each satisfies the one-million-period unchanged-policy criterion and reports realized convergence times extending to approximately fifty billion periods. The staged ten- and twenty-billion cumulative caps proposed earlier in this log are therefore superseded as final stopping boundaries.

The revised protocol separates the scientific stopping rule from scheduler safeguards. Each session is logically continued until convergence but is executed through finite Slurm invocations with fixed wall time, period budget, chunk size, and checkpoint cadence. When an invocation ends before convergence, it preserves the exact checkpoint, emits no measurement sample, and resumes under the same session identity and random streams. Fifty billion periods is an operator-enforced diagnostic review point: the runner records and warns that review is due, but it does not automatically classify the session as converged or failed or prevent a later resubmission.

The formal HPC entry point has been modified locally accordingly and its accelerated-engine suite passes 44 automated tests. The former cumulative `--max-periods` option now fails in `run_session_cli` with an explanatory error and is replaced there by an additional-work option, `--work-periods`; the separate local/debug `run_cell` entry point requires an explicit `--debug-only` acknowledgement and is not authorized for formal cluster experiments. An incomplete HPC invocation writes only `ckpt_XXXX.npz` and `progress_XXXX.json`, returns a safe-pause status, and leaves the session in the training phase. Only genuine convergence permits the 100,000-period measurement window and atomic publication of `session_XXXX.npz`. Checkpoints bind the full cell identity, experiment seed, session index, schema, training chunk size, engine version, a SHA-256 fingerprint of trajectory-defining source files, and Python/NumPy/Numba runtime versions. A per-session operating-system lock rejects overlapping array jobs. Completed results are checked for schema, convergence, internal period counts, identities, row shape, source fingerprint, and runtime identity before reuse; stale progress is repaired if a prior process stopped after atomic result publication. Formal strict aggregation rejects unfinished checkpoints, censored artifacts, and identity mismatches; full-cohort completeness is additionally enforced by the required command-line argument `--expected-sessions 1000`. Recovery copies accepted legacy artifacts independently and verifies source and destination hashes. Commit, push, Narval pull, environment receipt, and a live one-task Slurm signal/resume smoke test remain pending at the time of this entry.

The efficient 822-plus-178 recovery remains a mixed-provenance design. The 822 legacy NPZ files do not themselves embed the old Git commit, runtime versions, or training chunk size. Their reuse is supported by independent artifact hashes, the reported old checkout `9ab452fb6dc54e7ce25a9ec9417e346aa177d366`, the hard-coded one-million-period chunk in that checkout's Slurm launcher, and a core-equivalence audit showing that the trajectory equations and `Session` initialization, training, and measurement logic are unchanged. The historical Python/NumPy/Numba runtime receipt remains unknown. Accordingly, the repaired summary must report 822 `legacy_unversioned` sessions and 178 `current_schema` sessions and cannot be described as a homogeneous current-provenance cohort. A fully fresh 1,000-session low-noise rerun under the corrected runner is the preferred design for the strongest publication claim. Offline summaries additionally record a separate fingerprint of the metric and aggregation code.

This revision was adopted after observing 17.8 percent right-censoring in the initial low-noise campaign and is therefore documented as a post-campaign correction rather than a preregistered choice. It changes the computational termination protocol but does not alter the economic parameters, random identities, model equations, grids, or convergence criterion. If an administrative hard stop ultimately becomes unavoidable, the affected sessions will remain right-censored, will not be measured or included in the primary economic aggregate, and will be reported separately.
