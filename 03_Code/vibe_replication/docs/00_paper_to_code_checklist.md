# Paper-to-Code Checklist / 论文到代码核对表

This is the single roadmap for the replication. Code is added only in the order
listed here, and a block is complete only after its validation passes. / 这是本次复现的
唯一路线图。代码只按这里的顺序加入；只有验证通过后，一个模块才算完成。

## Source of truth / 唯一论文依据

- Main paper / 主论文: `AI-Powered Trading, Algorithmic Collusion, and Price Efficiency`
- Main text extraction / 主论文文本: `../../../ai-trading-collusion/docs/paper_full_text.txt`
- Online appendix / 在线附录: `../../../ai-trading-collusion/docs/online_appendix_full_text.txt`
- Existing technical extraction / 现有技术摘录: `../../../ai-trading-collusion/docs/paper_spec.md`

The old implementation may help us locate details, but it is not evidence that a
formula is correct. We validate against the paper and appendix. / 旧实现可以帮助定位
细节，但不能作为公式正确的证据；最终依据是论文和附录。

## Status labels / 状态说明

- `DONE`: paper checked, code readable, test passed / 已核对论文、代码可读、测试通过
- `DRAFT`: code exists but has not passed our agreed review / 有草稿，但尚未完成共同核对
- `TODO`: not coded / 尚未编码
- `OPEN`: paper or implementation decision still needs investigation / 仍需查明

## Ordered implementation checklist / 按顺序实施的核对表

| Order / 顺序 | Paper rule / 论文规则 | Planned code block / 计划代码块 | Required validation / 必须验证 | Status / 状态 |
|---:|---|---|---|---|
| 1 | Fundamental value: `v_t ~ N(v_bar, sigma_v^2)`; discretize with `v_k = v_bar + sigma_v Phi^-1((2k-1)/(2n_v))`; baseline `n_v=10`; `sigma_v_hat ~= 0.938` | Fundamental-value grid / 基本价值网格 | 10 values; probabilities 0.05...0.95; symmetry; mean 1; discrete std 0.938 | `DONE` |
| 2 | Noise order: `u_t ~ N(0, sigma_u^2)` | Noise-trader rule / 噪声交易者规则 | Fixed-number test first; distribution and seed test later | `DONE` |
| 3 | Informed-plus-noise flow: `y_t = sum_i x_i,t + u_t` | Total order flow / 总订单流 | Hand calculation with fixed orders | `DONE` |
| 4 | Information-insensitive demand, eq. (3.2): `z_t = -xi(p_t-v_bar)` | Insensitive-investor rule / 信息不敏感投资者规则 | Above-mean price gives selling; below-mean price gives buying; magnitude test | `DONE` |
| 5 | Speculator payoff, eq. (3.1): `pi_i,t = (v_t-p_t)x_i,t` | Profit function / 投机者利润函数 | Four hand cases: buy/short crossed with under/overpricing | `DONE` |
| 6 | Market-maker objective, eq. (3.3): minimize `E[(y_t+z_t)^2 + theta(p_t-v_t)^2 | y_t]` | Objective evaluator / 做市商目标函数 | Compare candidate prices in a fixed numerical example | `DONE` |
| 7 | Theoretical price, eq. (3.4) | Static theoretical price rule / 静态理论定价 | Direct hand calculation and limiting cases | `DONE` |
| 8 | Nash benchmark: `x^N(v)=chi^N(v-v_bar)` and `chi^N=1/((I+1)lambda^N)` | Nash benchmark calculator / 纳什基准 | First-order-condition identity now; coupled fixed-point residual after Order 10 | `DONE` |
| 9 | Cartel benchmark: `x^M(v)=chi^M(v-v_bar)` and `chi^M=1/(2I lambda^M)` | Cartel benchmark calculator / 卡特尔基准 | Cartel first-order-condition identity now; coupled fixed-point residual after Order 10 | `DONE` |
| 10 | Appendix fixed points: `lambda=(theta gamma+xi)/(theta+xi^2)` and `gamma=I chi/((I chi)^2+(sigma_u/sigma_v_hat)^2)` | Numerical fixed-point solver / 数值不动点求解器 | Residual near zero; Step 8/9 identities agree; unique positive root bracketed | `DONE`: bisection for `xi>0`; the paper-valid `xi=0` boundary uses its exact positive root and passes the original residual equations |
| 11 | Benchmark profits: `pi^N=sigma_v_hat^2/((I+1)^2 lambda^N)` and `pi^M=sigma_v_hat^2/(4I lambda^M)` | Benchmark profit functions / 基准利润函数 | Formula result equals direct expected-payoff calculation | `DONE` |
| 12 | Action grid: baseline `n_x=15`, Nash-to-cartel range widened by `iota=0.1` | Action/intensity grid / 动作强度网格 | Exactly 15 ordered choices; correct endpoints and spacing | `DONE` |
| 13 | Value-specific price grid: for each fixed `v_k`, use that same value's Nash/cartel orders in the paper's `p^L(v_k),p^H(v_k)` bounds, widen by `iota`, and create `n_p=31` points | Price-grid matrix `P(v)` / 分价值价格网格矩阵 | Shape `n_v x n_p = 10 x 31`; every row strictly increasing and equally spaced; paired-value rows are mirror images; official row spacing matches the Step-13B footnote-25 diagnostic | `DONE`; old pooled global grid retired |
| 14 | State: `s_t=(p_{t-1},v_{t-1},v_t)` | State representation and indexes / 状态与索引 | Value-to-index round trip; boundary tests; table shape | `DONE` |
| 15 | Initial state is uniform on `P x V x V` | Initial-state initializer / 初始状态 | Fixed-seed reproducibility and valid-index tests | `DONE` |
| 16 | Paper's discounted initial Q-value under uniformly random opponent actions and mean noise zero | Q-table initializer / Q 表初始化 | Tiny grid that can be calculated completely by hand | `DONE` |
| 17 | Bellman/Q meaning, eqs. (2.1)-(2.3) | Q-value interpretation / Q 值定义 | Document every array dimension before updating it | `DONE` |
| 18 | Epsilon-greedy action, eq. (2.6) | Action chooser / 动作选择器 | Force exploration and exploitation branches separately | `DONE` |
| 19 | Value-dependent exploration, eq. (4.3): `epsilon_t(v)=exp(-beta t(v))`, where `t(v)` counts the system's past visits to value `v` | One shared vector of per-value visit counters / 一组全市场共享的分价值访问计数器 | First visit uses epsilon 1; visiting one value changes only that value's next epsilon; increment once after both traders act | `DONE` |
| 20 | Q update, eq. (2.4), plus Online Appendix 4.3 expected-`v_(t+1)` acceleration used in reported experiments | One-agent Q update with direct and accelerated continuation / 单智能体直接版与加速版 Q 更新 | Direct hand calculation `10 -> 10.11`; expected-continuation hand calculation `10 -> 10.04825`; only the visited cell changes | `DONE` |
| 21 | `I=2` independent informed AI speculators | Two-agent container / 双智能体 | Independent Q-table copies, action draws, profits, and updates; one shared system-level value-visit counter; updating either trader cannot mutate the other / 独立 Q 表、动作抽签、利润与更新；共享系统价值计数器；任一交易者更新不能修改另一位 | `DONE` |
| 22 | Market-maker history `D_t={v,p,z,y}` over `T_m=10,000` | Rolling history / 滚动历史 | Window never exceeds `T_m`; oldest row leaves correctly | `DONE` |
| 23 | Rolling OLS: `z=xi_0-xi_1 p+error` and `v=gamma_0+gamma_1 y+error` | Readable OLS first / 首先实现可读 OLS | Match a direct OLS calculation on synthetic data | `DONE` |
| 24 | Adaptive price, eqs. (4.1)-(4.2): `lambda_hat=(theta gamma_1_hat+xi_1_hat)/(theta+xi_1_hat^2)` and `p_hat=gamma_0_hat+lambda_hat y` | Adaptive market maker / 自适应做市商 | Hand calculation; rolling and direct OLS parity | `DONE`: exact price formula passes; centered O(1) rolling OLS matches the readable OLS and resulting continuous price in every tested window |
| 25 | Exact per-period protocol: agents act; noise arrives; maker prices; insensitive investors trade; profits realize; next value arrives; Q and history update | One-period environment / 单期环境 | Fully deterministic period traced line by line | `DONE`: prior-only OLS pricing, continuous-price demand/profits, one shared visit increment, prepare-then-commit private Q updates, FIFO history replacement, no-look-ahead checks, and five mutation-free rejection tests all pass |
| 26 | Independent random paths for values, noise, and exploration | Reproducible random streams / 可复现随机过程 | Same session identity gives identical full path; different session/cell differs; named streams cannot cross-consume | `DONE`: seven SHA-256-derived named streams, exact causal draws, 1,000-session uniqueness audit, Step-25 parity, once-only full-Q validation, and trace-free hot mode all pass |
| 27 | Convergence: all agents' argmax policies unchanged for 1,000,000 consecutive periods | Convergence tracker / 收敛跟踪器 | Toy policy-change sequence with known answer | `DONE`: full joint greedy policy built once; exact-tie maximizer sets compared after Q updates; any agent change resets the shared streak to zero; only each agent's updated row is rescanned; Step-26 path parity passes |
| 28 | Measurement window: next 100,000 periods after convergence | Session phases / session 阶段 | Training and measurement periods cannot overlap | `DONE`: explicit `TRAINING -> MEASUREMENT -> COMPLETE` controller; convergence period remains training; exactly the next `T=100,000` periods are measured; frozen greedy agents with no Q/exploration/visit updates; value/noise and rolling maker continue; raw rows stream to an online sink without internal history |
| 29 | Appendix IA.4.1-IA.4.3 matched-path Nash/cartel scoring and `Delta^C` | Collusion profitability / 合谋利润 | Same realized `(v_t,u_t)` path used for all three strategies | `DONE`: continuous theoretical Nash/cartel actions and prices are reconstructed on every Step-28 row; per-agent actual and benchmark profits are accumulated online; time means, each `Delta_i^C`, and then `Delta^C` follow the appendix order; no row history or result clipping |
| 30 | Appendix IA.4.4 OLS trading intensity `chi_hat^C` | Trading-intensity metric / 交易强度指标 | Recover each agent's unrestricted intercept and slope, then average slopes | `DONE`: each agent is regressed separately as `x_i,t = chi_i,0 + chi_i,1 v_t + error`; `chi_hat^C` is the mean slope; the same Step-28 rows fan out to Steps 29-30; centered constant-memory moments match independent NumPy OLS on all 100,000 rows; duplicate/cross-session sinks and inconsistent value/action/order rows are rejected; the pre-training parameter and value-grid context is frozen in the receipt |
| 31 | Appendix IA.4.5 price informativeness `I^C=(I chi_hat^C)^2(sigma_v_hat/sigma_u)^2` | Informativeness metric / 价格信息效率 | Exact signal-variance/noise-variance hand calculation and Step-30 integration | `DONE`: pure formula returns 25 in the hand example; the bound Step-30 scorer/controller plus pre-training context snapshot prevents cross-session or post-measurement parameter mixing; uses the discrete value-grid std and configured noise std, never realized sample standard deviations; frozen result preserves all inputs, intermediate variances, grid, measurement bounds, and seed provenance |
| 32 | Appendix IA.4.6 period liquidity `L_t^C=1/abs(1-xi lambda_hat_t)`; prose calls the session result an average although the displayed aggregation omits `1/T` | Liquidity metric / 流动性指标 | Direct inventory-sensitivity check; period-first average; exact and near-singularity tests | `DONE`: uses configured structural `xi` and the period-specific prior-history `lambda_hat_t` already used for pricing; fused multiply-add preserves finite near-singular gaps; exact zero is explicitly recorded as `+infinity`; the frozen receipt stores both the appendix's literal sum and our disclosed arithmetic mean over exactly the Step-28 measurement rows; 100,000-row online aggregation matches an independent batch sum |
| 33 | Appendix IA.4.7 mispricing: definition `E_t^C=abs(E[p_t|v_t]-v_t)`; printed expansion `(1-lambda_hat_t I chi_hat^C) abs(v_t-v_bar)` has no absolute value around its first factor | Mispricing metric / 错误定价指标 | Hand formula; 100,000-pair replay; Step-29/30/32/33 path parity; negative-loading ambiguity guard | `DONE`: stores only each period's prior-history `lambda_hat_t` and `abs(v_t-v_bar)` until the full-window Step-30 `chi_hat^C` is known (1.6 MB for 100,000 rows), then replays once; preserves both the printed signed expression and Definition 3.4 absolute error; reports the metric when both expressions agree on every measured term, while a negative loading combined with a positive value deviation requires an explicit research decision; separately records the raw negative-loading count, disagreement count, and first affected periods; stores both literal sum and disclosed exact-T arithmetic mean |
| 34 | Appendix Section 4.5: orient prices/orders by `sign(v_t-v_bar)`; introduce an adverse noise shock at `t=3`; target a `1.2%` normalized oriented-price deviation across noise environments; classify both agents' `t=4` normalized oriented-order responses with strict thresholds | Shock-calibration and mechanism-classifier contract / 冲击校准与机制分类器契约 | Hand calibration; shock signs; orient-before-average; exact and adjacent-float thresholds; known price-trigger, over-pruning, and unclassified cases; invalid domains; unverified-provenance flags | `DONE AS A PURE, AUDITABLE CONTRACT`: visually verified `x_underbar=5e-5` and `x_bar=10*x_underbar=5e-4`; both agents must pass the relevant strict inequality and agents are never averaged before classification. As disclosed replication completion rules, exact threshold equality and cases satisfying neither condition receive `unclassified`. This pure layer records shock sign/addition and numerical/reset choices as replication interpretations and deliberately marks naked caller-supplied aggregates as unverified. Steps 35A-F now provide the checkpoint, long-run baseline, exact-level calibration, executed paired paths, and trusted session/cell wrappers; Step 34 by itself still cannot claim a generated IRF result. |
| 35A | Start the IRF infrastructure from a genuinely converged market without allowing treatment/control branches to share mutable state | Lossless converged checkpoint and detached restore / 无损收敛快照与独立恢复 | Exact source/clone next-period parity; Gaussian cache; exact rolling-OLS resynchronization phase; two-branch independence; wrong-time and tamper rejection | `DONE AS INFRASTRUCTURE, NOT AN IRF RESULT`: captures only after the convergence period and before the first measurement row; stores both Q-tables, frozen policy masks/actions, current market state, visit counts, exact 10,000-row maker history plus both centered-statistic accumulators and rebuild phase, and all seven RNG states; array payloads are immutable bytes with nested and whole-checkpoint SHA-256 checks. Restore creates a detached frozen-policy session with new Q memory, maker, lists, and RNG objects, and copies no controller, tracker, observer, or sink. Fourteen dedicated tests prove lossless continuation and non-contamination, including an integrated source/restore path across a real OLS resynchronization and rejection of saved statistics or counters inconsistent with their rows/history. No shock, common-random-number convention, path reset policy, or 10,000-path claim is introduced yet. |
| 35B | Paper IRF timing: completed convergence outcome at local `t=0`, ordinary stochastic continuation, adverse noise shock at `t=3`, and endogenous response at `t=4` | One paired control-treatment IRF path / 一条配对的对照—实验 IRF 路径 | Carried `t=0 -> t=1` state continuity; exact pre-shock parity; one additive treatment shock; common ordinary draws; `p_3 -> s_4 -> x_4` trigger wiring; branch-order invariance; calibration-tamper rejection | `DONE AS A ONE-PATH CORRECTNESS REFERENCE, NOT A FORMAL IRF RESULT`: executes new transactions at local `t=1,2,3,4` while carrying the completed `p_0,v_0` outcome and checkpoint `v_1`; derives stable path streams from experiment/session/path identities with full 256-bit SHA-256 seeds; draws each ordinary `u_t` and next-value index once and supplies it to both independent branches; adds the adverse shock only to treatment `u_3`; keeps Q/policy frozen and rolling OLS adaptive as disclosed choices. A state-dependent fixture proves that different treatment/control `p_3` grid indexes become their respective `t=4` states and select different saved actions. Twenty-six focused tests also reject forged calibration dataclass copies, distinguish the paper's `1.2%` target from sensitivity targets, prove each maker uses its own history, and audit 20,000 unique within-session child-stream identities without claiming those paths were run. No long-run baseline, verified cell-level calibration, 10,000-path aggregation, or mechanism classification is issued. |
| 35C | Figure 3 normalizes oriented prices, profits, and oriented orders by long-run expectations `E[p_tilde]`, `E[pi_i]`, and `E[x_tilde_i]` | One-session long-run IRF baseline collector / 单个 session 的长期 IRF 基准收集器 | Orient each price/order row before averaging; hand means; compensated summation; independent batch parity; detached-scorer/fake-fan-out/live-delivery rejection; stale-receipt detection | `DONE AS DENOMINATOR INFRASTRUCTURE, NOT AN IRF RESULT`: as a disclosed replication interpretation, uses the exact post-convergence frozen-policy Step-28 measurement window from the same session (`100,000` rows in paper mode), rather than the four-period unshocked Step-35B control branch. Before accepting a row, it captures and binds the exact Step-35A convergence checkpoint that Step 35D must reuse. The Step-28 sink is immutable for the controller lifetime; official fan-out membership is derived from the same private sink tuple it actually executes; only genuine Python `MethodType` objects can claim bound-scorer identity; and the scorer accepts a row only during live delivery of that session's just-completed period. It streams rows in constant memory, computes per-session `E[p_tilde]`, each agent's `E[x_tilde_i]` and `E[pi_i]`, plus long-run lambda diagnostics, and binds the result to the checkpoint/build, session identity, period boundaries, parameters, grid, and scored-field digest. Fourteen focused tests prove orient-before-average arithmetic, online/batch equality, deterministic replay, checkpoint/sink provenance, and rejection of detached-scorer, swap-and-restore, manual-row, look-alike fan-out, and forged-`__self__`/`__func__` attacks. An unkeyed receipt checksum detects stale ordinary `dataclasses.replace` copies but is explicitly not authentication; Step 35D must consume the live scorer result and exact matching checkpoint. The paper defines the expectations but not their finite-sample estimator or pooling convention; the same-session `100,000`-row choice is therefore labeled as ours, while separate flags verify only the paper-scale `100,000` measurement and `1,000,000` convergence thresholds plus provenance. Long-run lambda is diagnostic only: Step 35C is never marked ready for shock calibration because Step 34 requires actual IRF-path `lambda_hat_3`. It applies no shock, runs zero IRF paths, performs no experiment-cell pooling/calibration, and issues no mechanism classification. |
| 35D | Each converged session averages `10,000` stochastic continuations; the adverse shock will enter at local `t=3` | Efficient unshocked `t=3` calibration-moment pass / 高效无冲击 t=3 校准统计路径 | Hot-path versus fresh-restore parity; exact OLS/history rollback; interrupted-path recovery; canonical `0..9999` execution; compensated means; constant-memory stress | `DONE AS A PER-SESSION CALIBRATION PASS, NOT A SHOCKED IRF`: consumes the live Step-35C scorer and the exact checkpoint object it bound; restores one detached frozen-policy branch once; for every path runs ordinary draws only through local `t=1,2,3`; records the actual pre-append OLS `lambda_hat_3` used for pricing plus unshocked `p_tilde_3^0`; then reverses only the three appended/evicted immutable maker rows and restores exact OLS accumulators, counters, period, price, values, and draw mode. The online reducer retains no path list and records count, compensated means, minimum/nonpositive lambda diagnostics, and an executed-fields digest. A real 10,000-index scale test passes while honestly keeping paper-scale convergence/measurement, 1,000-session coverage, cross-session seed uniqueness, common shock calibration, shock application, t=4 response, classification, and figure readiness false. Step 35D also fixes the provenance wording: Step 35C's long-run sample and Step 35D's continuation sample are different samples from the same experiment cell, not one identical calibration sample. |
| 35E | Calibrate one uniform adverse `t=3` shock for all `N_sim=1,000` sessions in an experiment cell | Cell pooling and shock calibration / 实验单元汇总与冲击校准 | Exactly 1,000 distinct session receipts; cross-session seed audit; finite-sample level target versus increment shortcut; one immutable common magnitude | `DONE AS AUDITABLE CALIBRATION CODE, NOT AS A 1,000-SESSION RESULT`: validates and canonically orders same-cell Step-35D receipts, rebuilds every session seed manifest, rejects duplicate sessions/checkpoints/receipts/base streams, and pools long-run statistics by measurement-row counts and actual `t=3` statistics by path counts. The selected primary magnitude is `((1+0.012) E[p_tilde]-E[p_tilde_3^0])/E[actual lambda_hat_3]`; the old `0.012 E[p_tilde]/E[actual lambda_hat_3]` increment shortcut is retained as a named sensitivity and coincides only when the finite-sample unshocked `t=3` mean equals the long-run mean. The frozen receipt stores both results, the exact achieved level, pooled future Step-35F denominators, source/build/seed digests, and truthful paper-scale flags. It verifies unique cross-session session/base-stream seeds and unique path-seed namespaces without pretending that all 20 million SHA-256 child outputs were materialized and compared. Ten focused tests cover hand arithmetic, count weighting, actual-versus-long-run lambda, input order, invalid domains, mixed cells, duplicates, tampering, and debug-versus-formal claims. No shock or `t=4` path is executed yet. |
| 35F | Apply the calibrated adverse shock at `t=3`, measure both agents at `t=4`, and classify price-trigger/over-pruning | Paired response paths and mechanism classification / 配对反应路径与机制分类 | Same session/path schedules as calibration; control-treatment identity; exact 1.2% sample audit; strict per-agent thresholds | `DONE AS VERIFIED RUNNER AND DEBUG-SCALE EVIDENCE, NOT AS THE FORMAL 1,000-SESSION RESULT`: revalidates all ordered Step-35D sources by exactly rebuilding Step 35E, binds each live checkpoint and Step-35C scorer to one canonical session, and restores two independent reusable branches once per session. Each path uses the authenticated Step-35B schedule, applies the one exact-level magnitude only to treatment `u_3`, verifies pre-shock parity and `p_tilde_3^T-p_tilde_3^0=lambda_hat_3*m`, proves each `t=4` branch carries its own completed `t=3` maker row, then rolls both four-period transactions back exactly. A constant-memory reducer reproduces the exact Step-35D control digest and averages treatment `x_tilde_i,4`. Classification occurs once per session after path averaging, using that session's Step-35C `E[x_tilde_i]`; treatment-control response is retained only as a named sensitivity. The cell aggregator audits the actually executed 1.2% treatment-price level and counts session labels rather than reclassifying a pooled response. A genuine `1 session x 100 paths` demo hits `1.200000000%` and returns a debug price-trigger label; ten focused Step-35F tests plus the full 194-test suite pass. Formal flags remain false until canonical `1,000 x 10,000` paths and all formal source receipts are supplied. |
| 36 | Baseline `N_sim=1,000`; low/high `sigma_u`; comparative statics in `I`, `rho`, and other paper settings | Resumable experiment manager and HPC runner / 可恢复实验管理器与超算运行器 | Config saved; independent sessions; deterministic batches; checkpoint/resume; failed-session accounting | `FORMAL CORE RUNNER CONNECTED; EMPIRICAL RUN NOT YET EXECUTED`: `run_formal_experiment.py` freezes two exact PAPER-mode cells (`sigma_u=0.1` and `100`), each with 1,000 independent sessions, a 1,000,000-period unchanged-policy convergence requirement, and a 100,000-period measurement window. Array indices `0..999` map to low noise and `1000..1999` to high noise; each worker owns one atomic lock, resumes exact Step-36E state, and array jobs require a finite per-invocation period budget while the scientific plan remains uncapped. Checkpoint retention is bounded to the newest two files plus any evidence-pinned source. Collection refuses fewer than 1,000 validated bundles, preserves unclamped `Delta^C`, reports cross-session Type-7 percentiles, and never imputes undefined mispricing. This core path deliberately stops at Step 36E: Step35D-F mechanism/IRF analysis can reuse the same completed sessions later and does not block the core low/high comparison. A23 is resolved, and all 253 automated tests pass. The formal plans are orchestration evidence, not simulation results; pure-Python throughput remains a deployment constraint before the full cluster run. / `正式核心 RUNNER 已接通；尚未执行实证长跑`：入口已固定低/高噪声各 1,000 个正式 session、100 万期策略稳定判据和 10 万期测量窗口；支持确定的超算数组映射、精确续跑、有限 checkpoint 保留及严格的 1,000-session 汇总。核心路径故意停在 Step 36E，之后可复用同一批 session 做 Step35D-F 机制分析。当前没有把计划冒充结果；正式集群运行前仍需解决纯 Python 吞吐量。 |
| 37 | Reproduce the paper's figures and tables from verified experiment receipts | Figures, tables, and paper comparison / 图表与原文比较 | Scripted outputs; uncertainty bands; source-data manifests; numerical comparison tolerances | `TODO` |

## Open decisions: do not guess silently / 未决事项：不得悄悄猜测

| ID | Question / 问题 | Current status / 当前状态 |
|---|---|---|
| A1 | During Q-learning, if several actions share the maximum Q-value, how is one action selected? / Q-learning 训练中多个动作 Q 值并列时如何选择？ | `RESOLVED AS AN EXPLICIT TRAINING-PHASE CHOICE`: the paper and appendix specify `argmax` but no tie rule. During training, select uniformly among exact maximizers using the agent's injected RNG; near-equal values are not ties. Measurement-time limit-policy ties are a separate A9 choice. / `作为训练阶段的明确实现选择已解决`：论文与附录只写了 `argmax`，没有并列规则。训练期使用 agent 的独立随机流，在精确并列最大动作中等概率选择；近似相等不算并列。测量期极限策略的并列处理另见 A9。 |
| A2 | How exactly is a continuous realized price mapped to `P` for the next state: nearest point, bins, or another rule? / 连续价格如何映射到价格网格？ | `RESOLVED AS AN EXPLICIT IMPLEMENTATION CHOICE`: first select the row `P(v_(t-1))`, because the state already contains `v_(t-1)`, then use the nearest point in that row, clip outside observations to that row's endpoints, and choose the lower point at an exact midpoint. Thus `s_(t+1)=(p_t,v_t,v_(t+1))` maps `p_t` in row `P(v_t)`. Record clipping rates in full simulations and retain continuous prices for demand, profits, and market-maker history. / `作为明确的实现选择已解决`：先根据状态中已有的 `v_(t-1)` 选择价格行 `P(v_(t-1))`，再映射到该行最近点；超出范围时使用该行端点，精确中点取较低编号。因此下一状态 `s_(t+1)=(p_t,v_t,v_(t+1))` 使用 `P(v_t)` 映射 `p_t`。完整模拟记录截断率；需求、利润和做市商历史仍使用连续价格。 |
| A3 | How is the market maker's initial `T_m` history created before the first adaptive price? / 做市商最初的历史窗口如何生成？ | `BASELINE IMPLEMENTED; SENSITIVITY PENDING`: the paper and appendix are silent. Our explicit baseline preloads exactly `T_m` balanced Nash-consistent synthetic `(v,p,z,y)` rows before `t=0`; the maker receives only rows and recovers coefficients through its own rolling OLS. Low/high-noise recovery and a cartel-consistent initializer pass. This is our replication choice, not a paper rule. Before final results, compare full outcomes against cartel and expanding-window starts. / `基准已实现；敏感性结果待完成`：论文及附录没有说明。我们的明确基准是在 `t=0` 前载入恰好 `T_m` 条使用平衡设计且与 Nash 一致的合成 `(v,p,z,y)`；做市商只接收记录，并通过自己的滚动 OLS 恢复系数。低/高噪声恢复及 cartel 初始化器均已通过。这是我们的复现选择，不是论文规则。正式结果前仍须比较 cartel 与扩展窗口初始化下的完整实验结果。 |
| A4 | Which random draws share a generator, and how are session seeds derived? / 不同随机过程如何分配随机数生成器？ | `RESOLVED AS AN EXPLICIT IMPLEMENTATION CHOICE`: the paper specifies iid exogenous shocks, independent agent randomization, and 1,000 independent sessions, but discloses no seeds, RNG engine, or stream allocation. Derive a stable experiment-cell seed and session seed with named SHA-256, then use seven private streams: initial state, values, noise, and separate mode plus action/tie streams for each trader. Record the experiment seed, explicit cell key, session index, all derived seeds, derivation version, Python version, and RNG engine. Different experiment cells are independent by default; intentional common-random-number comparisons require an explicitly shared cell identity. / `作为明确的实现选择已解决`：论文规定外生冲击 iid、agent 独立随机化以及 1,000 个独立 session，但没有公布种子、RNG 引擎或随机流分配。使用命名 SHA-256 稳定地产生实验单元种子与 session 种子，再分配七条私有流：初始状态、价值、噪声，以及每位 trader 各自的模式流和动作/并列流。保存实验种子、明确的实验单元标签、session 编号、全部派生种子、推导版本、Python 版本和 RNG 引擎。不同实验单元默认独立；若以后有意使用共同随机数，必须明确共享单元身份。 |
| A5 | If a benchmark fixed-point equation has multiple numerical roots, which root is economically admissible? / 不动点有多个数值根时选哪个？ | `RESOLVED FOR POSITIVE INPUTS`: the residual's shape gives exactly one positive root inside the analytical gamma-bound interval; bisection selects that root |
| A6 | The appendix text around the high shock-response threshold contains an OCR ambiguity. / 附录高阈值文本存在 OCR 歧义。 | `RESOLVED BY VISUAL INSPECTION OF THE ORIGINAL APPENDIX PAGE`: `x_underbar=5e-5` and `x_bar=10*x_underbar=5e-4`. Both session-level rules use both informed agents and strict inequalities. / `通过目视核对原始附录页面解决`：`x_underbar=5e-5`，`x_bar=10*x_underbar=5e-4`；两个 session 层面的规则都要求两位知情 agent 同时通过严格不等式。 |
| A7 | What exactly counts as an unchanged policy period, and how do exact ties enter convergence? / 怎样才算一个“策略不变时期”，精确并列如何进入收敛判断？ | `RESOLVED AS AN EXPLICIT IMPLEMENTATION CHOICE`: before period 0, save every agent's full state-to-exact-maximizer-set mapping. After both Q updates in each completed period, compare only the state rows that could have changed. If every set is unchanged, add one; if any set changes, reset the shared streak to zero, so the changed period is not counted as stable. Reach convergence after exactly 1,000,000 no-change comparisons. The paper specifies full optimal-strategy stability but does not define tie handling or this off-by-one convention. / `作为明确的实现选择已解决`：第 0 期前保存每位 agent 从全部状态到“精确最大动作集合”的完整映射。每个完整时期的两次 Q 更新之后，只比较本期可能变化的状态行；全部集合不变则加一，任一集合变化则把共享连续计数归零，因此变化当期不算稳定期。恰好完成 1,000,000 次“不变比较”后判定收敛。论文规定完整最优策略必须稳定，但没有说明并列处理及这一差一约定。 |
| A8 | What exactly happens at the convergence-to-measurement boundary? / 从收敛切换到测量时究竟发生什么？ | `RESOLVED AS A PAPER-SUPPORTED REPLICATION INTERPRETATION`: the appendix says `T=100,000` periods after convergence but prints internally inconsistent inclusive summation bounds. Treat the convergence-reaching period as training and measure exactly `T_c+1,...,T_c+100,000`. Carry forward the same market state, environment RNG streams, and rolling-maker history without reset. Freeze Q at `Q_(T_c)`, disable exploration and visit counting, and use the frozen greedy policy. Continue random `v,u` and update the rolling maker once per measurement period. The frozen-agent switch is strongly supported by the appendix's limit-policy definition but is not stated as literal author code; later sensitivity should compare continued background Q-updating and a frozen maker. / `作为原文支持的复现解释已解决`：附录文字说收敛后测量 `T=100,000` 期，但打印的两端包含求和边界内部矛盾。把达到收敛的时期只算训练，并准确测量 `T_c+1,...,T_c+100,000`。不重置市场状态、环境随机流或滚动做市商历史；冻结 `Q_(T_c)`、关闭探索与访问计数，并使用固定贪心策略。价值与噪声继续随机，做市商每个测量期继续滚动更新一次。冻结 agent 的解释得到附录极限策略定义的强支持，但并非作者逐字的软件指令；以后应与继续后台 Q 更新和冻结做市商进行敏感性比较。 |
| A9 | If the frozen limit policy has an exact Q tie, which one action is used during measurement? / 固定极限策略存在精确 Q 并列时，测量期使用哪个动作？ | `RESOLVED AS AN EXPLICIT MEASUREMENT-PHASE CHOICE`: choose the lowest action index once at convergence and keep that pure action fixed. This consumes no trader RNG and makes the limit strategy reproducible. The paper does not specify this; final robustness checks must compare uniform random selection within the unchanged exact-maximizer set. / `作为测量阶段的明确实现选择已解决`：收敛时一次性选取并列集合中的最小动作编号，并固定使用这个纯动作；不消耗 trader RNG，且极限策略可复现。论文没有规定这一点；最终稳健性检验必须比较“在不变精确最大集合中等概率随机选择”。 |
| A10 | What should software do if a finite matched path gives `mean(pi^M)-mean(pi^N)` equal to, below, or numerically indistinguishable from zero? / 若有限样本路径上 `mean(pi^M)-mean(pi^N)` 等于零、小于零或数值上无法与零区分，软件如何处理？ | `RESOLVED AS AN EXPLICIT NUMERICAL SAFEGUARD`: the paper does not specify this edge case. Do not take an absolute value, add an arbitrary epsilon, or fabricate a score. Raise an explicit undefined-metric error when the denominator is non-positive or no larger than 64 floating-point units at the observed profit scale. Never clip a valid `Delta^C` to `[0,1]`. / `作为明确数值保护已解决`：论文没有规定该边界情形。不取绝对值、不任意加 epsilon、不伪造分数。分母非正，或不高于当前利润尺度下 64 个浮点单位时，明确报告指标未定义。有效 `Delta^C` 绝不截断到 `[0,1]`。 |
| A11 | Should trading-intensity OLS pool agents or impose the theoretical intercept `-v_bar*chi`? / 交易强度 OLS 应合并 agent，还是强制理论截距 `-v_bar*chi`？ | `RESOLVED FROM APPENDIX IA.4.4`: estimate one ordinary unweighted regression with a free intercept for each agent, then average the agent slopes. The theoretical intercept relation is recorded only as a diagnostic, not imposed. Centered online moments are algebraically equivalent to batch OLS and avoid storing 100,000 rows. If measured values have no numerically meaningful variation, report the metric as undefined rather than inventing a slope. / `依据附录 IA.4.4 已解决`：对每位 agent 分别进行含自由截距、无权重的普通回归，再平均各 agent 的斜率。理论截距关系只作为诊断，不强制加入回归。中心化在线矩与批量 OLS 在代数上等价，并避免保存十万行；若测量价值没有数值上有意义的变化，则明确报告指标未定义，不伪造斜率。 |
| A12 | Does Appendix IA.4.6 define session liquidity as a sum or an average, and which periods enter? / 附录 IA.4.6 的 session 流动性究竟是求和还是平均，包含哪些时期？ | `RESOLVED AS A DISCLOSED PAPER-SUPPORTED INTERPRETATION`: the prose says “average market liquidity,” but the displayed equation omits `1/T` and prints inclusive-looking `T_c,...,T_c+T` bounds. Report the arithmetic mean of the exact `T` Step-28 measurement rows as `L^C`, while also preserving the literal unnormalised sum in the receipt. This follows the prose and stated `T=100,000`; the original display inconsistency remains disclosed. / `作为公开说明且有原文支持的解释已解决`：文字明确说“平均市场流动性”，但展示式漏写 `1/T`，且打印出看似两端包含的 `T_c,...,T_c+T`。以 Step-28 恰好 `T` 条测量记录的算术平均作为 `L^C`，同时在 receipt 中保留字面未标准化求和。该选择遵循文字与明确的 `T=100,000`；原展示式的不一致仍被公开记录。 |
| A13 | How should IA.4.6 handle `1-xi*lambda_hat_t=0` or a very small finite gap? / IA.4.6 遇到 `1-xi*lambda_hat_t=0` 或很小的有限差时如何处理？ | `RESOLVED AS AN EXPLICIT NUMERICAL REPRESENTATION`: use fused multiply-add to evaluate `1-xi*lambda_hat_t` with one rounding; do not add epsilon, impose a tolerance, or clip a large finite result. A true representable zero is recorded as extended-real `+infinity`, with singular-period counts and the first affected period in the receipt. A nonzero finite gap is inverted as written, even if liquidity is extremely large; reciprocal overflow is tagged separately. / `作为明确数值表示已解决`：使用融合乘加只舍入一次地计算 `1-xi*lambda_hat_t`；不加 epsilon、不设人为容差，也不截断有限大数。真正可表示的零记录为扩展实数 `+infinity`，并在 receipt 中保存奇点期数及首个受影响时期。任何非零有限差都按原式取倒数，即使流动性极大；倒数溢出另行标记。 |
| A14 | Does Appendix IA.4.7 define session mispricing as a sum or an average, and which periods enter? / 附录 IA.4.7 的 session 错误定价究竟是求和还是平均，包含哪些时期？ | `RESOLVED AS THE SAME DISCLOSED EXACT-T INTERPRETATION AS A12`: the prose calls the outcome “average mispricing,” but the display again omits `1/T` and uses inclusive-looking bounds. Report the arithmetic mean over the exact `T` Step-28 measurement rows as `E^C`, and preserve the literal unnormalised sum in the receipt. / `采用与 A12 相同的公开 exact-T 解释已解决`：文字称其为“平均错误定价”，展示式仍漏写 `1/T` 并使用看似两端包含的边界。以 Step-28 恰好 `T` 条测量记录的算术平均作为 `E^C`，同时在 receipt 中保留字面未标准化求和。 |
| A15 | Why does IA.4.7 define an absolute error but omit an absolute value around `1-lambda_hat_t I chi_hat^C` in its printed expansion? / 为什么 IA.4.7 先定义绝对误差，却在展开式的 `1-lambda_hat_t I chi_hat^C` 外漏掉绝对值？ | `RESOLVED AS AN AUDITED DOMAIN CHECK, NOT A SILENT CORRECTION`: theoretical equilibria make the loading non-negative, but the appendix gives no finite-simulation guarantee or software rule. Compute and retain both the printed signed expression and Definition 3.4's absolute-error expression. They agree when the loading is non-negative and also when `v_t=v_bar`, because both measured terms are then zero. A negative loading combined with a positive value deviation makes the expressions disagree: preserve both diagnostics, leave the primary metric undefined, and require an explicit researcher decision rather than silently adding an absolute value or reporting negative “mispricing.” Record the raw negative-loading count separately from the actual formula-disagreement count and their first affected periods. Because final `chi_hat^C` is known only after Step 30 uses the full window, buffer only `(lambda_hat_t, abs(v_t-v_bar))` as two float64 values per row and replay once; this is 1.6 MB at `T=100,000`. / `作为经过审计的定义域检查解决，而不是静默修正`：理论均衡会使该系数非负，但附录没有给出有限模拟保证或软件规则。代码同时计算并保留原文带符号展开式与定义 3.4 的绝对误差式。系数非负时两式一致；`v_t=v_bar` 时两项同为零，因此即使系数为负也仍一致。负系数与正价值偏差同时出现时两式才会不同：此时保留两类诊断、把主要指标留作未定义，并要求研究者明确选择，绝不偷偷补绝对值或报告负“错误定价”。原始负系数期数与真正公式不一致期数分开记录，并保存各自首个受影响时期。由于最终 `chi_hat^C` 必须等 Step 30 使用完整窗口后才能知道，每期只缓存 `(lambda_hat_t, abs(v_t-v_bar))` 两个 float64 并在结束后重放一次；`T=100,000` 时为 1.6 MB。 |
| A16 | The appendix's low-response condition is printed with missing parentheses, which literally resembles `abs(x_tilde-E[x_tilde]/E[x_tilde])<x_underbar`. How should it be read? / 附录低反应条件漏印括号，字面近似 `abs(x_tilde-E[x_tilde]/E[x_tilde])<x_underbar`，应如何理解？ | `RESOLVED AS A DOCUMENTED PAPER-SUPPORTED INTERPRETATION`: use `abs((x_tilde-E[x_tilde])/E[x_tilde])<5e-5`. The literal alternative reduces part of the expression to one and is dimensionally inconsistent; Figure 3 prints the normalized order-response formula correctly, and Section 4.5 describes an insignificant *change*. The receipt records that the parentheses are a replication interpretation rather than pretending the appendix typography is unambiguous. / `作为有原文支持且公开记录的解释解决`：采用 `abs((x_tilde-E[x_tilde])/E[x_tilde])<5e-5`。字面替代式会把其中一部分约成 1，量纲不一致；图 3 正确打印了标准化订单反应公式，第 4.5 节也描述的是不显著的“变化”。receipt 明确记录括号属于复现解释，不假装附录排版没有歧义。 |
| A17 | How are the 10,000 IRF paths, long-run expectations, control/shock branches, market-maker state, and common random numbers constructed after convergence? Does learning or rolling OLS continue? / 收敛后如何构造 10,000 条 IRF 路径、长期均值、对照/冲击分支、做市商状态及共同随机数？学习或滚动 OLS 是否继续？ | `PRIMARY RUNNER RESOLVED THROUGH SHOCKED RESPONSE; FORMAL SCALE AND SENSITIVITIES REMAIN`: Steps 35A-35F now implement lossless detached checkpoints, same-session long-run denominators, reversible unshocked calibration, exact-level cell calibration, paired shocked `t=1..4` paths, actual-target auditing, and per-session mechanism labels. Q/policy/visits/internal RNGs remain frozen; each branch's rolling OLS continues on its own rows; ordinary draws are common and only treatment receives the signed additive `t=3` shock. Step 35F replays and matches the exact Step-35D control digest, classifies only after averaging each session's paths, and aggregates labels as cell shares. These are disclosed replication interpretations because the paper does not specify checkpoint/reset/CRN mechanics. Still open are the formal `1,000 x 10,000` execution, HPC resume layer, and the `t=0`/shared-`v_1`, frozen-maker, continued-Q, pooled-denominator, and shortcut-shock sensitivities. / `主要受冲击 runner 已解决；正式规模与敏感性仍待完成`：第 35A-35F 步已接通无损快照、同-session 长期分母、可回滚无冲击校准、exact-level 单元校准、配对受冲击 `t=1..4` 路径、实际目标核对和逐-session 分类。仍待正式 `1,000 x 10,000` 运行、HPC 恢复层及各项实现选择敏感性。 |
| A18 | What exactly does “clone the converged session” mean in code? / 代码中的“克隆收敛 session”究竟复制什么？ | `RESOLVED AS LOSSLESS REPLICATION INFRASTRUCTURE`: capture the first frozen-policy boundary after convergence and preserve full Q tables, policy masks/actions, market state, visit counts, exact rolling-OLS rows/statistics/rebuild phase, and all seven RNG states including Python `gauss()`'s cached normal variate. Restore into a detached session and never copy the old controller token, tracker, observer, or metric sink. This is a safety/provenance contract, not a claim that the paper requires carrying every component into its IRF. Exact continuation checks Python/NumPy versions, Python implementation, OS/machine, native byte order, and the A23 versioned execution/result source identities; the environment and checked build must therefore be pinned before HPC transfer. / `作为无损复现基础设施已解决`：在收敛后的首个固定策略边界保存完整 Q 表、策略 masks/动作、市场状态、访问计数、精确滚动 OLS 历史/统计量/重建阶段，以及七条随机状态（包括 Python `gauss()` 缓存的第二个正态数）。恢复到脱离旧控制图的新 session，绝不复制旧 controller token、tracker、observer 或指标 sink。精确续跑会核对 Python/NumPy、操作系统/机器、字节序，以及 A23 的带版本执行/结果源码身份；因此转移到超算前必须固定软件环境与已核对源码。 |
| A19 | How is one paired IRF path timed, randomized, and separated from the paper's long-run baseline? / 一条配对 IRF 路径如何计时、随机化，并与论文长期基准区分？ | `RESOLVED FOR THE STEP-35B REFERENCE; SENSITIVITY REMAINS`: interpret the already completed convergence outcome as local `t=0`, execute new transactions at `t=1..4`, and apply the shock only at `t=3`. Carry `p_0,v_0,v_1` from the checkpoint; the paper does not settle this fork rule, so shared checkpoint `v_1` is disclosed and retained for sensitivity analysis. Use full 256-bit SHA-256 path seeds derived from stable IRF-experiment seed, source-session seed, path index, and stream name; the checkpoint digest is provenance only. Draw ordinary `u_t` and next `v` once, give both branches the same draw, and add the signed shock only in treatment. Keep policy frozen and each maker rolling on its own rows. The unshocked branch is not `E[x_tilde_i]`, and Step 35B never classifies. / `针对第 35B 步参考路径已解决；敏感性仍待完成`：把已经完成的收敛结果解释为局部 `t=0`，新交易发生在 `t=1..4`，冲击只发生在 `t=3`。checkpoint 中的 `p_0,v_0,v_1` 被继承；原文没有确定这一分叉规则，因此“各路径共享 checkpoint 的 `v_1`”被明确披露并保留为敏感性问题。路径种子使用稳定的 IRF 实验种子、来源 session 种子、路径编号和流名称，经完整 256 位 SHA-256 派生；checkpoint 哈希只用于来源记录。普通 `u_t` 与下一价值每期只抽一次并同时交给两分支，仅实验组增加带符号冲击。策略冻结，各做市商只滚动自己的记录。未冲击分支不是 `E[x_tilde_i]`，第 35B 步绝不分类。 |
| A20 | From which finite sample are Figure 3's `E[p_tilde]`, `E[pi_i]`, and `E[x_tilde_i]` estimated? / 图 3 中的 `E[p_tilde]`、`E[pi_i]` 与 `E[x_tilde_i]` 应从哪个有限样本估计？ | `PRIMARY ESTIMATORS RESOLVED; POOLED-DENOMINATOR SENSITIVITY REMAINS`: every session contributes its own exact Step-28 frozen-policy measurement rows. Step 35E raw-count weights session price moments to calibrate and audit the one cell-wide shock. For Appendix-4.5 mechanism labels, Step 35F follows the paper's session-level wording: first average that session's 10,000 treatment paths, then normalize each agent by the same session's Step-35C `E[x_tilde_i]`, and only afterward count labels across 1,000 sessions. A pooled cell order denominator is retained only as a named Figure-level sensitivity; it cannot replace the primary session label. At formal scale equal row/path counts make raw-row weighting and equal weighting of session means coincide for linear levels, but not necessarily for ratios. / `主要估计量已解决；合并分母敏感性仍保留`：第 35E 步按原始计数汇总价格以校准统一冲击；第 35F 步先在每个 session 内平均一万条实验路径，再除以该 session 自己的 Step-35C 长期订单均值，最后跨一千个 session 统计标签。实验单元合并订单分母仅保留为敏感性，不能替代主分类。 |
| A21 | Is the `1.2%` shock calibrated as a treatment-control increment or as the plotted price level relative to the long-run mean? / `1.2%` 冲击究竟按实验组—对照组增量校准，还是按相对长期均值的图中价格水平校准？ | `RESOLVED AS A DISCLOSED EXACT-LEVEL PRIMARY INTERPRETATION`: the paper prints the level normalization `(p_tilde_3-E[p_tilde])/E[p_tilde]` but does not disclose its numerical routine. Step 35E therefore selects `((1+0.012)E[p_tilde]-E[p_tilde_3^0])/E[actual lambda_hat_3]` as the primary finite-sample magnitude. It also reports the old increment shortcut `0.012 E[p_tilde]/E[actual lambda_hat_3]`, its achieved level, and the difference. The two coincide only if the finite-sample unshocked `t=3` mean equals the long-run mean. If the unshocked level is already at or above the 1.2% target, a positive adverse shock cannot solve the exact-level equation and the code rejects the cell instead of taking an absolute value. / `作为明确披露的 exact-level 主要解释已解决`：原文打印价格水平标准化，却未公开数值程序。因此第 35E 步选用 exact-level 有限样本公式，并同时报告旧增量 shortcut、其实现的价格水平及差异；只有无冲击 t=3 均值等于长期均值时两者才相同。若无冲击水平已达到或超过目标，正逆向冲击无解，代码会拒绝而不是取绝对值。 |
| A22 | Are the 31 numerical price points one pooled global grid or one row conditional on each fixed fundamental value? / 31 个数值价格点是一条全局网格，还是每个固定基本价值各有一行？ | `RESOLVED AS A PAPER-SUPPORTED IMPLEMENTATION INTERPRETATION`: use `P(v_k)` with 31 points for every fixed `v_k`. Footnote 25 states that `n_p` is chosen near `2n_x` so a one-grid-point change in one agent's order moves price by one price-grid point, holding other conditions fixed. The equality is recovered in the low-noise calibration only when the bounds and spacing are calculated at the same fixed `v_k`; pooling extreme orders across all values makes the grid too coarse and hides unilateral deviations. This does not add information because `v_(t-1)` is already an observed state component, and the state/Q dimensions remain `31 x 10 x 10 = 3,100`. High-noise rows remain coarse under `n_p=31`, as expected from the paper's over-pruning region. The global-grid implementation is preserved only in the pre-refactor source snapshot and must not be pooled with new results. / `作为原文支持的实现解释已解决`：对每个固定 `v_k` 分别使用含 31 点的 `P(v_k)`。脚注 25 说明选择 `n_p` 接近 `2n_x`，是为了在其他条件固定时，一位 agent 的订单改变一档可使价格网格移动一档；只有在同一个固定 `v_k` 下计算边界和间距，低噪声校准才满足这一关系。把所有价值的极端订单汇总会使网格过粗，并隐藏单边偏离。此做法没有额外泄露信息，因为 `v_(t-1)` 本来就在状态中；状态与 Q 表规模仍为 `31 x 10 x 10 = 3,100`。在 `n_p=31` 下高噪声行仍然较粗，这与论文的过度剪枝区域一致。旧全局网格仅保留在改造前代码快照中，不能与新结果混合。 |
| A23 | Which source files should determine a persisted checkpoint's implementation fingerprint? / 哪些源码应决定持久化 checkpoint 的实现指纹？ | `RESOLVED BEFORE FORMAL EXECUTION`: `src/source_manifests.py` freezes a 27-file Step-28 execution closure and a disjoint 19-file result-pipeline layer. Paths and normalized source bytes use domain-separated SHA-256 hashes; plans and tasks store the manifest, execution, result, and combined identities. Independent AST tests require the execution list to equal the exact recursive local-import closure and require every result root's local dependencies to be covered. Adding an unrelated root orchestration file leaves the scientific hashes unchanged. Affected Step35A/36B-F schemas were advanced, and Step36E evidence now truthfully records A23 as resolved. / `已在正式运行前解决`：明确的 27 文件执行闭包与 19 文件结果管线分别取带域、规范换行的 SHA-256；计划、任务与证据保存这些身份。独立 AST 测试会拒绝遗漏或多余依赖，而新增无关调度文件不会让长跑 checkpoint 无故失效。 |
| A24 | How can Step 35E pool 1,000 completed sessions and then let separate Step-35F jobs recover each session's exact trusted baseline/checkpoint chain? / 第 35E 步汇总一千个 session 后，独立的第 35F 任务如何恢复每个 session 的精确可信基准与 checkpoint 链？ | `PER-SESSION BRIDGE RESOLVED BY STEP 36F; FULL-CELL ORCHESTRATION OPEN`: Step 36F persists the exact Step-35D receipt together with references and byte fingerprints for the complete Step-36E evidence, convergence origin, and any Step-36C replay source. Its reload adapter deterministically rebuilds the live scorer/checkpoint identity chain, reruns Step 35D from path zero, and requires exact receipt equality. A real fresh-process test then reconstructs a debug Step-35E context and runs Step 35F successfully. The original Step-36E evidence remains unchanged and honestly keeps its older A24 flags false; the later Step-36F envelope proves the additional per-session facts. What remains is the cell-level manager that collects every canonical session bridge, performs one Step-35E calibration, dispatches all separate Step-35F workers, and accounts for failures across 1,000 sessions. / `Step 36F 已解决单 session bridge；完整实验单元调度仍待完成`：36F 保存精确 Step-35D receipt，并记录 Step-36E evidence、收敛 origin 与可选 Step-36C 重放起点的路径及逐字节指纹。重载 adapter 会重建实时 scorer/checkpoint 身份链、从路径 0 重跑 Step-35D，并要求 receipt 完全一致；新进程测试随后已真实建立 debug Step-35E context 并运行 Step-35F。剩余工作是收集全部标准 session bridge、统一执行一次 Step-35E、分发所有 Step-35F worker，并对 1,000 sessions 做失败核算。 |

## Current gate / 当前关口

Orders 1-5 and Integration Checkpoint A have been run and reviewed successfully.
Checkpoint A connects `y_t`, `z_t`, and both speculators' profits in one fixed,
hand-checkable example. / 第 1-5 项以及整合检查点 A 已经成功运行并共同核对。
检查点 A 使用一个固定、可以手算的例子，把 `y_t`、`z_t` 和两位投机者的利润
连接起来。

Orders 6-36F and the formal Step-36 core entrypoint are implemented and validated through their explicitly stated
scope boundaries. Order 19 maintains ten shared
system-level value counters and computes epsilon from the selected value's
past visits. After both traders act, it increments that counter once for the
market period. Integration Checkpoint B connects Orders 18 and 19 in one
transparent action-selection period. Order 20 validates both the direct
realized-next-state equation and the appendix acceleration used in the reported
experiments, which averages continuation over possible next values. Both prove
that only the visited state-action cell changes. Order 21 places two independent
informed Q-traders around Steps 18-20: their Q-tables, random draws, profits,
and updates are private, while epsilon and the system value counter are shared.
Order 22 stores completed `(v_t,p_t,z_t,y_t)` rows in a capacity-`T_m` rolling
window and prevents the current row from entering its own pricing data. The
paper does not disclose how the first `T_m=10,000` observations are initialized.
Decision A3 now implements a balanced Nash-consistent synthetic prehistory of
exactly 10,000 rows. The maker receives only rows and recovers the intended
Nash coefficients through its own rolling OLS; low/high-noise cases and a
cartel-consistent initializer pass. A3 remains explicitly our replication
choice, and full-outcome sensitivity to cartel and expanding-window starts is
still required. Order 23 estimates the two linear models with readable,
unregularized intercept OLS, explicitly
mapping the raw `z`-on-`p` slope to `xi_1_hat` with a sign reversal and matching
NumPy least squares. Order 24A implements and hand-validates the exact adaptive
price formula, preserves the continuous price, and connects Orders 22-24 without
look-ahead. Order 24B now maintains the same two OLS regressions with centered
O(1) add/remove updates, periodically rebuilds them to limit floating-point
drift, and matches both the readable coefficients and the resulting adaptive
price across every tested rolling window. Order 24 and the baseline A3
initializer are therefore complete. Order 25 now connects the full deterministic
period in the paper's order: the traders choose before noise; the maker estimates
from prior history and prices observed `y_t`; insensitive demand and profits use
the continuous price; the completed row then replaces the oldest history row;
and each private Q-table updates one cell using the appendix expected-next-value
acceleration. Changing only realized `v_(t+1)` changes the realized next state but
not the current outcome or accelerated Q target, confirming no look-ahead. The
period first checks grid/window/Q coherence, and five deliberately invalid calls
confirm that rejected input changes no RNG, counter, Q-table, or history row.
Step 25 retains full-table checking as a readable diagnostic. Order 26 now
implements explicit decision A4 with stable experiment-cell/session identities
and seven named SHA-256 child streams. Values are drawn uniformly from V, noise
remains continuous Gaussian, and each trader owns separate mode and action/tie
streams. Noise and the next value are drawn only at their causal positions. Two
same-identity sessions reproduce their complete eight-period market paths and
final mutable states exactly; another session and another experiment cell differ.
All 1,000 planned sessions and 7,000 child streams have unique seeds. A fixed-draw
lean period matches the Step-25 oracle exactly. Full Q validation runs once at
session construction, while a receipt-free path avoids per-period trace allocation.
Order 27 implements decision A7. It snapshots both agents' complete greedy
policies once, represents exact ties as maximizer sets, and then checks only the
two Q-rows that can change in each completed period. Any agent's policy change
resets the one shared no-change streak to zero; Q-value movement that preserves
the maximizing set does not reset it. A hand-known streak `[1,0,1,2,3]`, tie-set
changes, threshold boundaries, safe pre-period attachment, and a same-seed
Step-26 path-parity test all pass. Order 28 implements decisions A8-A9 with an
exclusive `TRAINING -> MEASUREMENT -> COMPLETE` controller. The convergence
period is training only; the next exactly 100,000 periods use one frozen greedy
action per agent and state, while value/noise streams and rolling OLS continue
without reset. Q tables become read-only, visits and trader RNGs freeze, raw
rows are streamed rather than retained, and bulk runs require a sink. Controller
ownership prevents direct kernel bypass; stale policy masks, extra periods,
debug-cap pseudo-convergence, and sink failures are rejected without a false
completion receipt. A `K=2,T=3` test produces training `[0,1]`, measurement
`[2,3,4]`, and five total periods. Order 29 streams those rows into a
session-bound, constant-memory scorer. On every identical realized `(v_t,u_t)`,
it reconstructs continuous theoretical Nash and cartel actions, flows, prices,
and per-agent profits; it never reuses the AI's adaptive OLS price or snaps a
benchmark to a grid. A hand example gives `Delta_1^C=0`, `Delta_2^C=1`, and
`Delta^C=0.5`; low-noise, high-noise, and the paper's `xi=0` matched-path
checks independently equal the Step-11 closed forms. The scorer verifies the successful Step-28 controller,
exact row count, global-period boundaries, session seed identity, fixed-point
residuals, and coefficient provenance before issuing a frozen result. Order 30
uses the same completed Step-28 rows through a deterministic fan-out. For each
agent separately, it estimates the unrestricted OLS line
`x_i,t = chi_i,0 + chi_i,1 v_t + error`, then averages the slopes to obtain
`chi_hat^C`. It consumes the signed realized orders and continuous fundamental
values—not action indexes, total flow, or benchmark orders. Centered online
moments use constant memory and match an independent NumPy batch regression.
Incomplete sessions and paths without meaningful value variation cannot issue a
result. The scorer also cross-checks each value, action index, and raw order;
the controller rejects duplicate or cross-session metric wiring before an
expensive run begins. It also freezes the complete parameter object and exact
value grid before period zero, so later attribute rebinding cannot change the
meaning of its result. A full 100,000-row test matches batch OLS. Order 31 is
pure post-processing of that completed Step-30 result; it does not replay the
measurement path. It computes the IA.4.5 signal-to-noise ratio using the known
discrete-grid standard deviation and configured noise standard deviation, never
their realized sample counterparts. An exact example independently obtains
signal variance `6.25`, noise variance `0.25`, and informativeness `25`. The
frozen result records the agent slopes, grid, all formula inputs/intermediates,
measurement boundaries, and seed provenance. Cross-session scorer/controller
pairs, post-measurement parameter/grid rebinding, incomplete sessions, undefined
standard deviations, and numerical overflow/underflow are rejected; valid
results are not clipped to one. Order 32 is a sibling online sink attached
before Step-28 measurement begins. Each period uses the configured structural
`xi` and the exact prior-history `lambda_hat_t` that priced that period, then
applies IA.4.6 before time aggregation; it never substitutes the rolling
`xi_1_hat`, queries the already-updated maker, or applies the reciprocal to an
average lambda. A three-period hand example gives liquidity `(1,2,2/3)`, literal
sum `11/3`, and arithmetic mean `11/9`; an independent inventory derivative
gives the same middle-period value `2`. Fused multiply-add prevents
`1-500*0.002` from being falsely rounded to an exact zero. True zero gaps are
recorded as extended-real infinity, while finite near-singular results remain
unclipped. The receipt preserves both the appendix's printed unnormalised sum
and the disclosed mean over exactly the Step-28 rows, plus measurement bounds,
seed/parameter provenance, and singular diagnostics. Its 100,000-row
constant-memory result matches an independent batch sum. Order 33 receives the
same Step-28 path but defers its final calculation until Step 30 estimates the
full-window `chi_hat^C`. It stores only two float64 values per period—the
prior-history `lambda_hat_t` and `abs(v_t-v_bar)`—then replays those compact
pairs once, requiring 1.6 MB for 100,000 periods. A hand example gives loading
`0.75` and mispricing `1.5`; a 100,000-pair result matches an independent batch
sum. The receipt preserves both IA.4.7's printed signed expansion and Definition
3.4's absolute error. It reports `E^C` when both formulas agree in every measured
term; a negative loading with positive value deviation instead records the count
and first affected period and requires an explicit research choice.
Steps 29, 30, 32, and 33 consume one identical fan-out path. Order 34 now
implements the Appendix Section 4.5 pure arithmetic contract. It first orients
prices and orders with `sign(v_t-v_bar)`. A hand calibration with
`E[p_tilde]=2`, `E[lambda_hat_3]=0.5`, and positive minimum lambda `0.5`
gives the replication protocol's common cell-level shock magnitude `0.048`, an
implied oriented price increment `0.024`, and implied target `1.2%`. As disclosed
interpretations, the shock sign follows `v_3-v_bar` and it is added to ordinary
noise. At `t=4`,
both normalized agent responses must be strictly above `5e-4` for
price-trigger or both absolute responses strictly below `5e-5` for
over-pruning; exact boundaries, mixed agents, and all other cases are
unclassified under our completion rule. Twenty-seven isolated tests cover the
arithmetic, adjacent-float thresholds, invalid domains, positive-lambda guard,
cell-wide magnitude reuse, immutable receipts, unverified provenance flags, and
the disclosed low-rule parenthesis interpretation. This step deliberately does
not claim to have generated or verified the 10,000 IRF paths: A17 records the missing
paper conventions and safe checkpoint/clone requirements. Order 35A now resolves
the safety half of that gate. It captures exactly between the convergence period
and first measurement row, stores immutable byte snapshots of both Q-tables and
the frozen policy, preserves all seven RNG states including the Gaussian cache,
and saves the rolling maker's 10,000 rows together with its exact centered OLS
accumulators and next-resynchronization phase. Each restore owns fresh mutable
objects and copies no controller, tracker, observer, or sink. Fourteen focused
tests prove source/clone next-period parity, source nonmutation, two-branch
independence, exact OLS continuation across a rebuild, and tamper/wrong-time
rejection. Step 35A applies no shock and claims no 10,000-path result. Step 35B
then restores one independent control-treatment pair, carries the completed
local-`t=0` outcome, and executes local `t=1..4`. A path-specific external driver
draws ordinary noise and next values once for both branches; treatment alone
receives one additive adverse shock at `t=3`. The learned policy remains frozen,
while each rolling maker updates only from its own rows. A state-dependent test
proves the complete `p_3 -> s_4 -> x_4` trigger channel, and calibration receipts
are recomputed so changed frozen copies cannot forge arithmetic or provenance.
Twenty-six focused tests pass. This remains one correctness path: no 10,000-path
execution, authenticated cell calibration, or classifier result exists yet.
Step 35C now collects the separate long-run denominator that Figure 3 requires.
As a disclosed interpretation of an unspecified estimator, each session streams
its exact frozen-policy Step-28 measurement window (100,000 rows in paper mode),
orients every row before averaging, and reports per-session `E[p_tilde]`, both
`E[x_tilde_i]`, `E[pi_i]`, and long-run lambda diagnostics in constant memory.
Before the first row it captures the exact Step-35A checkpoint that Step 35D must
reuse and authenticates controller sink/fan-out membership. It never uses the
four-period control branch as a denominator. Fourteen focused tests pass, including
an independent batch oracle and rejection of formerly successful detached-scorer,
sink-swap, manual-row, and fake-fan-out provenance attacks. The unkeyed receipt
checksum detects stale copies but is not authentication; Step 35D must use the live
scorer result and matching checkpoint. Long-run lambda remains a
diagnostic and cannot authorize the actual `t=3` shock calibration. This is still
not a shock experiment: Step 35C executes zero IRF paths,
does no cell pooling or calibration, and cannot classify a mechanism. Step 35D
now consumes that live scorer/checkpoint pair and executes an unshocked
calibration pass through local t=3. A reusable transaction logs only three
maker-row replacements and restores the exact history, OLS accumulators,
resynchronization phase, and session state after each path. The reducer retains
only compensated sums, diagnostics, and a digest. Twenty-four focused Step-35D tests
(including the reversible layer) pass; one test truly executes indexes 0..9999
without turning that count-only success into a shocked-IRF, formal-convergence,
1,000-session, calibration, classification, or figure claim. It collects both
actual pre-append `lambda_hat_3` and unshocked `p_tilde_3^0`, because the paper's
printed level normalization and the provisional increment shortcut need not be
identical in a finite sample. Step 35E now pools verified same-cell receipts,
audits cross-session identities and seed namespaces, and selects the disclosed
exact-level finite-sample rule while retaining the increment shortcut as a
sensitivity. Step 35F then runs two reusable, independent four-period branches,
reproduces the Step-35D control digest, applies the authenticated shock only at
`t=3`, and averages each session's treatment `t=4` orders before classification.
Its genuine one-session debug run hits the executed 1.2% price target exactly;
ten focused tests reject altered seeds, shocks, path arithmetic, source chains,
and incomplete transaction cleanup. This proves the runner, not the paper's
formal empirical result: the `1,000 x 10,000` cell execution, HPC orchestration,
and full implementation-choice sensitivity analysis remain for Step 36/37.
/ 第 6-36F 项以及正式 Step-36 核心入口已经在各自明确限定的范围内实现并验证。第 19 项维护十个全市场共享的价值计数器，先根据当前
价值点的过去访问次数计算 epsilon；等两个交易者都选完动作后，再为本期把该计数
增加一次。整合检查点 B 已把第 18 与第 19 项连接成一个透明的动作选择期。第 20 项
同时验证了“使用实际下一状态”的直接方程版本，以及论文报告实验采用的附录加速版本；
后者对所有可能的下一价值计算平均延续价值。两个版本都证明只有被访问的状态-动作
格子改变。第 21 项已经把两位独立知情 Q 交易者连接到第 18-20 步：各自拥有 Q 表、
随机抽签、利润与更新，同时共享 epsilon 和系统价值计数器。第 22 项把已经完成的
`(v_t,p_t,z_t,y_t)` 保存在容量为 `T_m` 的滚动窗口，并阻止本期记录参与决定自己的
价格。论文没有说明最初 `T_m=10,000` 条记录如何初始化。A3 现在已经实现恰好一万条、
使用平衡设计且与 Nash 一致的合成前历史。做市商只接收记录，并通过自己的滚动 OLS
恢复预期的 Nash 系数；低/高噪声情形和 cartel 初始化器均已通过。A3 仍明确属于我们的
复现选择，正式结果前仍须检验 cartel 与扩展窗口初始化下的完整结果敏感性。第 23 项使用可读、
无正则化且含截距的 OLS 估计两个线性模型，明确把 `z` 对 `p` 的原始斜率取相反数得到
`xi_1_hat`，并与 NumPy 最小二乘结果匹配。第 24A 项实现并手算验证精确的自适应定价
公式，保留连续价格，并在没有前视偏差的情况下连接第 22-24 项。第 24B 项使用中心化
O(1) 加入/移除更新维护相同的两条 OLS，定期重建以限制浮点漂移，并在每个测试滚动
窗口中同时匹配可读系数与最终自适应价格。因此第 24 项和 A3 基准初始化器现已完成。
第 25 项现已按照论文时序连接完整的确定性时期：交易者先选择，之后噪声到达；做市商只用
旧历史估计关系，并根据观察到的 `y_t` 定价；信息不敏感需求和利润使用连续价格；完整本期
记录随后替换最旧历史记录；两张私有 Q 表各自使用附录的下一价值期望加速，只更新一个格子。
只改变实际 `v_(t+1)` 会改变实际下一状态，却不改变本期结果或加速 Q 目标，因此没有前视。
每期先检查网格、窗口和 Q 表的一致性；五个故意无效的调用进一步证明，被拒绝的输入不会
改变随机数、计数器、Q 表或历史记录。第 25 步保留完整 Q 表检查，作为易读的诊断版本。
第 26 步现已按照明确决定 A4，使用稳定的
实验单元/session 身份和七条命名 SHA-256 子流。价值从 V 等概率抽取，噪声保持连续正态；
每位 trader 分别拥有模式流和动作/并列流。噪声与下一价值只在各自的因果位置抽取。两个身份
相同的 session 会精确重放完整八期市场轨迹及最终可变状态；不同 session 与不同实验单元会得到
不同路径。计划中的 1,000 个 session 和 7,000 条子流种子均不重复。固定抽样下的精简时期与
第 25 步完全一致。完整 Q 检查只在 session 建立时运行一次；无流水单路径也避免逐期分配轨迹
对象。第 27 步实现了决定 A7：先一次性保存两位 agent 的完整贪心策略，以最大动作集合表示精确
并列；之后每个完整时期只检查可能变化的两行 Q。任一 agent 的策略改变都会把一个共享连续计数
归零；若 Q 数值变化但最大动作集合不变，则不会归零。已知答案 `[1,0,1,2,3]`、并列集合变化、
阈值边界、第 0 期前安全连接，以及与第 26 步同种子路径完全一致的测试均已通过。第 28 步根据
A8-A9 建立独占的 `TRAINING -> MEASUREMENT -> COMPLETE` controller：收敛当期只属于训练；
之后恰好 100,000 期为每位 agent、每个状态使用一个固定贪心动作，同时不重置地继续价值/噪声
随机流与滚动 OLS。Q 表变为只读，访问计数与 trader RNG 冻结；原始行通过 sink 流式输出而非
保存在内部，批量运行必须连接 sink。controller 所有权阻止直接绕过市场内核；过期策略 mask、
额外时期、把调试上限冒充收敛，以及 sink 失败都不会产生虚假完成总结。`K=2,T=3` 测试准确得到
训练 `[0,1]`、测量 `[2,3,4]`，总计五期。第 29 项把这些行流式送入与指定 session 绑定的固定内存计分器。
对每一组完全相同的已实现 `(v_t,u_t)`，它重建连续理论 Nash 与 cartel 动作、订单流、价格及每位 agent 利润；
不复用 AI 的自适应 OLS 价格，也不把理论基准吸附到网格。手算例得到 `Delta_1^C=0`、`Delta_2^C=1`、
`Delta^C=0.5`；低噪声、高噪声与论文的 `xi=0` 同路径检查也独立等于第 11 步闭式解。生成冻结结果前，计分器会核对 Step-28 成功
controller、精确行数、全局时期边界、session 种子身份、不动点残差与系数来源。第 30 项通过确定性的
fan-out 使用同一批完整 Step-28 记录；它对每位 agent 分别估计非约束 OLS
`x_i,t = chi_i,0 + chi_i,1 v_t + error`，再平均各自斜率得到 `chi_hat^C`。输入是实际带符号订单和连续
基本价值，而不是动作编号、总订单流或理论基准订单。中心化在线矩只占固定内存，并与独立 NumPy 批量
回归一致。未完成的 session 或基本价值没有有效变化的路径不能生成结果。scorer 还会交叉核对每个价值、
动作编号与原始订单；controller 会在昂贵运行开始前拒绝重复或跨 session 的指标接线。它还会在第 0 期前
冻结完整参数对象与精确价值网格，因此事后重新绑定属性不能改变结果含义。完整十万行测试与
批量 OLS 一致。第 31 项只对已经完成的 Step-30 结果进行后处理，不重新播放测量路径。它按照 IA.4.5，
使用已知离散价值网格标准差与设定噪声标准差计算信噪比，绝不使用二者的实现样本标准差。独立手算准确得到
知情订单流方差 `6.25`、噪声方差 `0.25` 和价格信息效率 `25`。冻结结果保存各 agent 斜率、完整网格、公式
全部输入与中间量、测量边界及种子来源。跨 session scorer/controller 配对、未完成 session、无定义标准差
、测量后参数/网格重绑及数值溢出或下溢都会被拒绝；有效结果不截断到 1。第 32 项是在 Step-28
测量开始前连接的平级在线 sink。每一期都使用设定的结构参数 `xi`，以及真正为该期定价、由旧历史得到的
`lambda_hat_t`，先应用 IA.4.6 再进行时间汇总；绝不把滚动 `xi_1_hat` 当作结构参数，不查询已经加入本期
记录后的 maker，也不对平均 lambda 取倒数。三期手算例得到流动性 `(1,2,2/3)`、字面求和 `11/3`、
算术平均 `11/9`；独立库存导数也得到中间一期的流动性 `2`。融合乘加避免把 `1-500*0.002` 错误舍入为
精确零；真正零差记录为扩展实数无穷，接近奇点但有限的结果不被截断。冻结 receipt 同时保存附录展示的
未标准化求和与我们公开采用的恰好 Step-28 测量记录平均值，并保存时期边界、种子/参数来源及奇点诊断。
十万条固定内存汇总与独立批量求和一致。第 33 项读取同一条 Step-28 路径，但要等 Step 30
使用完整测量窗口估计出 `chi_hat^C` 后才进行最终计算。它每期只保存两个 float64——由旧历史
估计、当期真正使用的 `lambda_hat_t` 与 `abs(v_t-v_bar)`——随后只重放一次；十万期占 1.6 MB。
手算例的系数为 `0.75`、错误定价为 `1.5`，十万紧凑数对也与独立批量求和一致。冻结 receipt
同时保留 IA.4.7 的原文带符号展开式与定义 3.4 的绝对误差式；两式在每个测量项中都一致时才正式报告
`E^C`。若负系数与正价值偏差同时出现，则记录期数与首个受影响时期，并要求研究者明确决定。第 29、30、32、33 项消费同一个 fan-out 路径。
第 34 项现已实现附录第 4.5 节的纯计算契约：先用 `sign(v_t-v_bar)` 调整价格与订单方向；
手算 `E[p_tilde]=2`、`E[lambda_hat_3]=0.5`、最小正 lambda `0.5`，得到本复现协议的
实验单元统一冲击幅度 `0.048`、公式隐含方向调整价格增量 `0.024` 与隐含目标 `1.2%`。
作为公开的复现解释，冲击符号跟随 `v_3-v_bar`，并加到普通噪声上。`t=4` 时，
两位 agent 的标准化反应都必须严格高于 `5e-4` 才属于 price-trigger；两者绝对值都必须
严格低于 `5e-5` 才属于 over-pruning；恰好等于边界、两位 agent 不一致及其他情况都属于
本复现补全规则下的 unclassified。二十七项独立测试覆盖手算、相邻浮点阈值、无效定义域、
正 lambda 保护、跨 session 统一幅度复用、不可变 receipt、未验证来源标记，以及公开说明的
低反应括号解释。本步骤明确不声称已经生成或验证 10,000 条 IRF 路径；
A17 记录了论文未公开的约定。第 35A 项现已解决其中的“安全复制”部分：只在收敛期结束、第一条
测量记录之前保存；把两张 Q 表、固定策略、当前市场状态、七条随机流（包括正态抽样缓存），以及
做市商 10,000 行历史、精确 OLS 累加器和下一次重同步阶段全部无损保存。每个恢复分支都新建自己的
可变对象，绝不复制旧 controller、tracker、observer 或 sink。十四项专项测试证明下一期与源 session
完全一致、源 session 不被改变、两个分支互不污染、跨 OLS 重建点仍精确续跑，并会拒绝错误时点和
被篡改的快照。第 35A 项尚未施加冲击，也不声称已产生 10,000 条路径。第 35B 项随后恢复一组
完全独立的对照—实验分支，继承已经完成的局部 `t=0` 结果，并执行局部 `t=1..4`。逐路径外部
driver 每期只抽一次普通噪声和下一价值，同时交给两个分支；只有实验组在 `t=3` 增加一次逆向
冲击。学习策略保持冻结，而每个滚动做市商只使用自己的记录更新。状态依赖测试已经证明完整的
`p_3 -> s_4 -> x_4` 触发链；校准凭证会被重新计算，因此即使用冻结 dataclass 复制并篡改字段，
也不能伪造算术或来源。二十六项专项测试通过。本步骤仍只代表一条正确性路径：尚未执行 10,000
条路径，没有经过来源认证的实验单元校准，也没有分类结果。第 35C 步现在单独收集图 3 所需的
长期分母。作为对原文未说明估计量的一项公开解释，每个 session 流式读取自己恰好一个固定策略
Step-28 测量窗口（论文模式为 100,000 行）；每行先调整方向再求平均，并以固定内存报告单-session
的 `E[p_tilde]`、两位 agent 的 `E[x_tilde_i]`、`E[pi_i]` 与长期 lambda 诊断。第一条记录前，
它保存 Step 35D 必须复用的精确 Step-35A checkpoint，并核验 controller sink/fan-out 确实包含本 scorer。
它绝不把四期对照分支当作分母。十四项专项测试通过，其中包括独立批量 oracle，以及拒绝过去可以成功的
detached-scorer、sink 换出再换回、手工插入记录和伪 fan-out 来源攻击。无密钥 receipt 校验码只能发现
过期副本，不构成认证；第 35D 步必须使用实时 scorer 结果及匹配 checkpoint。长期 lambda 只作诊断，不能授权真正的
`t=3` 冲击校准。但这仍不是冲击实验：第 35C 步执行零条 IRF 路径，不做实验单元汇总或校准，也不能分类机制。
第 35D 步现在读取这对实时 scorer/checkpoint，运行到局部 t=3 的无冲击校准路径。可复用事务每条路径只记录
三次做市商历史替换，并精确恢复历史顺序、OLS 累加器、重同步阶段和 session 状态；汇总器只保留补偿求和、
诊断计数与摘要。包括底层回滚在内的二十四项 Step-35D 专项测试通过，其中一项真实执行了 `0..9999` 全部编号，
但只确认路径数量，不声称已完成冲击 IRF、论文收敛规模、1,000 sessions、统一校准、分类或图表。本步骤同时收集
追加本期记录之前实际使用的 `lambda_hat_3` 与无冲击 `p_tilde_3^0`，因为原文打印的水平标准化与暂定的增量简式
在有限样本中不一定相同。第 35E 步现在按来源行数汇总同一实验单元的 Step-35C/35D 凭证，重建 session
种子清单并拒绝重复或混合单元；它选择 exact-level 公式作为主校准，同时保留旧增量 shortcut 及其实际水平偏差。
十项专项测试通过，包括 `P=3,P_3^0=2.8,lambda_3=1` 的手算：主幅度 `0.236` 正好命中 `1.2%`，
而 shortcut `0.036` 不能命中水平目标。短调试批次只能验证代码；只有标准 `0..999` 的一千份正式 receipt、
每份一万条路径及论文规模长期样本全部满足时，凭证才会声明可进入正式 Step 35F。第 35F 步现已实现专用的
exact-level 配对 runner：每个 session 只恢复一次两个独立分支，每条路径执行 `t=1..4` 后精确回滚；它会重建
第 35D 步的完整对照路径摘要，只在实验组 `t=3` 加入统一冲击，并证明两个分支在 `t=4` 分别携带自己的
`t=3` 做市商记录。分类发生在每个 session 的路径平均之后，分母使用该 session 自己第 35C 步的长期订单均值；
之后实验单元只统计 session 标签占比，不把所有路径混在一起重新分类。真实的单-session、100-路径调试运行在
实际受冲击路径上命中 `1.200000000%`，十项专项测试通过。但这只证明 runner 与审计链正确；正式
`1,000 x 10,000` 运行、HPC 管理以及 A3 的完整敏感性检验仍属于第 36/37 步。

The formal core entrypoint is now `run_formal_experiment.py`. It creates one
immutable low/high experiment family, assigns 2,000 distinct session seeds and
14,000 distinct child-stream seeds, runs or resumes one Step-36E session per
worker, reports six operational states, and publishes a core receipt only after
all 1,000 canonical sessions in that cell reload and validate. Its automated
validation is not a reduced economic experiment: mocks and metadata checks test
orchestration without training an agent. The full suite contains 253 passing
tests. Formal empirical values remain absent until the workers actually finish.
/ 正式核心入口现为 `run_formal_experiment.py`：它固定一个低/高噪声实验 family，
分配 2,000 个互异 session 种子与 14,000 个互异子随机流种子，每个 worker 只运行或
续跑一个 Step-36E session，并且只有该 cell 的 1,000 份标准 evidence 全部重读验证
后才发布核心 receipt。这里通过的 253 项自动测试不是缩小版经济实验；它们只用
metadata 与 mock 核对调度代码。worker 真正完成以前，仍没有正式实证数值。
