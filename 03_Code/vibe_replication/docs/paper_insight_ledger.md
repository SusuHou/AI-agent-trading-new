# Paper Insight Ledger / 论文灵感台账

Working notes captured **while** the research happens, indexed by where each
item is expected to land in the written paper. This is not a results file and
not a validation log — it is the place where an observation is recorded before
it is lost. / 研究进行中随手记录的洞见，按其在论文中的预期去处编号。这不是结果
文件，也不是验证日志；它的作用是在洞见丢失之前先把它记下来。

- Validation Q&A belongs in [`learning_log.md`](learning_log.md).
- Implementation decisions belong in [`00_paper_to_code_checklist.md`](00_paper_to_code_checklist.md) (A1–A24).
- Run receipts and campaign history belong in [`experiments/`](experiments/).
- 验证问答见 `learning_log.md`；实现决定见 A1–A24 核对表；运行记录见 `experiments/`。

## How to add an entry / 记录格式

Copy the template. Keep the status label honest — an entry that overstates its
evidence is worse than no entry. / 复制模板；状态标签必须诚实，夸大证据的条目比
没有条目更糟。

```markdown
## I-NN — Short title / 简短标题

- **Status / 状态**: `VERIFIED` | `HYPOTHESIS` | `OPEN` | `HOUSEKEEPING`
- **Date / 日期**:
- **Destination / 论文去处**:

**Claim / 论点**

**Evidence / 证据**  (file:line, command, or number — never "I recall that")

**Outstanding / 待办**
```

Status meanings / 状态含义:

| Label | Meaning / 含义 |
|---|---|
| `VERIFIED` | Reproduced from data or paper text in this repo / 已从本仓库的数据或论文原文复现 |
| `HYPOTHESIS` | Mechanism is plausible and a concrete test is named, but not run / 机制合理且已指明检验方法，但尚未执行 |
| `OPEN` | Idea worth developing; no evidence yet / 值得发展的想法，尚无证据 |
| `HOUSEKEEPING` | Repo/documentation defect, not a research finding / 仓库或文档缺陷，非研究发现 |

## Index / 索引

| ID | Title / 标题 | Status | Destination / 去处 |
|---|---|---|---|
| I-01 | High-noise cell replicates on three independent quantities | `VERIFIED` | Results |
| I-02 | Low-noise cell does not reproduce the paper's `Delta^C` | `VERIFIED` | Results / Limitations |
| I-03 | `Delta^C` amplifies profit error by roughly 8x | `VERIFIED` | Methodology (measurement) |
| I-04 | The paper's own numbers validate our benchmark solver | `VERIFIED` | Replication validation |
| I-05 | Footnote 25 holds exactly in low noise; A22 is vindicated | `VERIFIED` | Methodology (A22) |
| I-06 | `T_c` has a constant hazard after exploration dies | `HYPOTHESIS` | Methodology / Appendix |
| I-07 | Censoring may be selection-biased toward low `Delta^C` | `HYPOTHESIS` | Results / Limitations |
| I-08 | `beta` is an economic parameter, not a technical one | `VERIFIED` | Methodology + Discussion |
| I-09 | `L^C` is near-deterministic under the baseline calibration | `VERIFIED` | Results (market quality) |
| I-10 | The Grossman–Stiglitz inversion | `OPEN` | Discussion |
| I-11 | Documentation lags the codebase | `HOUSEKEEPING` | — |

---

## I-01 — High-noise cell replicates on three independent quantities / 高噪声环境在三个独立数字上复现成功

- **Status / 状态**: `VERIFIED`
- **Date / 日期**: 2026-08-31
- **Destination / 论文去处**: Results — replication validation / 结果章，复现验证

**Claim / 论点**

The `sigma_u = 10^2` cell reproduces the paper's reported profit decomposition
to within 2% on three quantities that are not free parameters and are not
mechanically linked to one another. / `sigma_u = 10^2` 环境在三个既非自由参数、
彼此也无机械联系的数字上，与论文报告的利润分解吻合到 2% 以内。

**Evidence / 证据**

Paper (`docs/paper_full_text.txt:2195–2197`): "each informed AI speculator earns
about 54 on average, derived from average losses of 88 from information-
insensitive investors and 20 from noise traders. Market makers again break even."

| Quantity / 数量 | Paper / 论文 | `pilot_exact_per_value_sigma_u_100` |
|---|---|---|
| Speculator profit / 投机者利润 | ~54 | 53.41, 53.34 |
| Investor loss / 投资者损失 | ~88 | 86.59 |
| Noise-trader loss / 噪声交易者损失 | ~20 | 20.16 |

Reproduce / 复现:

```bash
cd 03_Code/dgj_sim && python3 -c "
import numpy as np
r = np.load('outputs/pilot_exact_per_value_sigma_u_100/session_0000.npz')['rows']
v, u, p, z = r[:,0], r[:,1], r[:,3], r[:,5]
print('speculator', r[:,8].mean(), r[:,9].mean())
print('investors ', (z*(v-p)).mean())
print('noise     ', (u*(v-p)).mean())
"
```

**Outstanding / 待办**

Single-session pilot. Confirm against the full 1,000-session high-noise cohort
before this wording enters the paper. / 目前只是单 session pilot；写入论文前须以
完整 1,000 session 高噪声队列确认。

---

## I-02 — Low-noise cell does not reproduce the paper's `Delta^C` / 低噪声环境未能复现论文的合谋指标

- **Status / 状态**: `VERIFIED`
- **Date / 日期**: 2026-08-31
- **Destination / 论文去处**: Results / Limitations / 结果章与局限章

**Claim / 论点**

At `sigma_u = 10^-1` the paper reports `Delta^C ~ 0.75`; our sessions produce
0.32–0.42. Profit *levels* are within ~6%, but the normalized statistic is off
by a factor of 2.3. / 在 `sigma_u = 10^-1` 下论文报告 `Delta^C ~ 0.75`，我们得到
0.32–0.42。利润**水平**只差约 6%，但标准化指标差了 2.3 倍。

**Evidence / 证据**

Paper (`docs/paper_full_text.txt:1578–1582`): "low noise trading risk
(specifically, `sigma_u = 10^-1`), the average value of `Delta^C` ... is
approximately 0.75 ... average trading profits that are about 10% higher than
those in the non-collusive equilibrium benchmark."
Paper (`:2190–2192`): "each informed AI speculator earns an average profit of
approximately 54, totaling a loss of about 108 for information-insensitive
investors."

| Quantity | Paper | Our pilot | 1,000-session mixed campaign |
|---|---|---|---|
| `Delta^C` | ~0.75 | 0.319 / 0.356 | 0.4156 (diagnostic only) |
| Profit / 利润 | ~54 | 50.65, 50.81 | — |
| Investor loss | ~108 | 101.42 | — |
| Gain vs Nash | ~10% | 4.0% | — |
| Noise trader | ~0 | 0.0002 | — |

**Outstanding / 待办**

Do not write this up as "failed to replicate" until I-07 is tested. The
censoring-selection explanation would resolve it without any code defect. /
在 I-07 检验完成前，不要写成「复现失败」；删失选择性偏差可以在不存在代码缺陷的
情况下解释这个差距。

---

## I-03 — `Delta^C` amplifies profit error by roughly 8x / 合谋指标把利润误差放大约八倍

- **Status / 状态**: `VERIFIED`
- **Date / 日期**: 2026-08-31
- **Destination / 论文去处**: Methodology — measurement properties / 方法章，指标性质

**Claim / 论点**

Under the baseline calibration the entire Nash-to-cartel profit window is only
12.5% of the profit level. `Delta^C` therefore divides by a very small number,
and a 1% error in realized profit becomes 0.083 of `Delta^C`. This is a property
of the estimand, not of any implementation. / 基准校准下，Nash 到 cartel 的整个
利润区间只有利润水平的 12.5%。`Delta^C` 因此除以一个很小的数：利润上 1% 的误差
会变成 `Delta^C` 上 0.083 的误差。这是指标本身的性质，与实现无关。

**Evidence / 证据**

Backed out of `pilot_exact_per_value_sigma_u_0p1/summary.json`
(`delta_c = 0.31922098`, `profit_gain_vs_nash = 1.03990315`, mean profit
50.73015):

```
pi^N          = 50.73015 / 1.03990315 = 48.7835
pi^M - pi^N   = (50.73015 - 48.7835) / 0.31922098 = 6.0982
pi^M          = 54.8817
window / pi^N = 6.0982 / 48.7835 = 12.5%
```

Cross-checked against the closed forms `pi^N = sigma_v_hat^2 / ((I+1)^2 lambda^N)`
= 48.88 and `pi^M = sigma_v_hat^2 / (4 I lambda^M)` = 54.99. / 与闭式解交叉验证一致。

Amplification / 放大倍数: reaching `Delta^C = 0.75` requires profit 53.3572,
i.e. +5.18%. So `0.4308 / 0.0518 = 8.3x`.

**Outstanding / 待办**

Worth a short methodological paragraph regardless of how I-02 resolves — it
tells any future replicator that `Delta^C` is a fragile target. / 无论 I-02 如何
收场都值得写一段方法论说明：它提醒后续复现者 `Delta^C` 是一个脆弱的目标。

---

## I-04 — The paper's own numbers validate our benchmark solver / 论文自身的数字反过来验证了我们的基准求解器

- **Status / 状态**: `VERIFIED`
- **Date / 日期**: 2026-08-31
- **Destination / 论文去处**: Replication validation / 复现验证

**Claim / 论点**

Feeding the paper's reported low-noise profit (~54) into *our independently
solved* benchmarks yields `Delta^C ~ 0.80`, consistent with the paper's reported
0.75 to within rounding. The Nash and cartel fixed-point solvers are therefore
correct; the discrepancy in I-02 lies in the learned policy, not the benchmarks.
/ 把论文报告的低噪声利润（约 54）代入**我们独立求解**的基准，得到
`Delta^C ~ 0.80`，与论文的 0.75 在舍入范围内一致。因此 Nash 与 cartel 不动点
求解器是正确的；I-02 的差距出在学到的策略上，不在基准上。

**Evidence / 证据**

`(54 - 48.7835) / 6.0982 = 0.855`; using the paper's "+10% over Nash" instead,
`48.7835 * 1.10 = 53.66` gives `(53.66 - 48.7835) / 6.0982 = 0.80`.

**Outstanding / 待办**

None. This is a clean positive result and should be stated explicitly — it
narrows the search space for I-02. / 无。这是一个干净的正面结果，应明确写出，它
缩小了 I-02 的排查范围。

---

## I-05 — Footnote 25 holds exactly in low noise; A22 is vindicated / 脚注 25 在低噪声下精确成立，A22 选择得到验证

- **Status / 状态**: `VERIFIED`
- **Date / 日期**: 2026-08-31
- **Destination / 论文去处**: Methodology — defends decision A22 / 方法章，为 A22 辩护

**Claim / 论点**

The paper's footnote 25 requires that a one-grid-point change in one
speculator's order move the price by one price-grid point. Under the
**per-value** grid `P(v_k)` this holds to within 7% at every value point in the
low-noise cell. The pooled global grid does not achieve this. A22 was the right
call and the price grid is **not** the cause of I-02. / 论文脚注 25 要求一位投机者
改变一档订单时，价格恰好移动一个价格档位。在**按价值划分**的网格 `P(v_k)` 下，
低噪声环境的每一个价值点都在 7% 误差内满足该性质；全局合并网格做不到。A22 的
选择是对的，价格网格**不是** I-02 的原因。

**Evidence / 证据**

Paper (`docs/paper_full_text.txt:1245`, footnote 25): "Our choice of `n_p ~ 2 n_x`
ensures that, all else equal, a one-grid point change in one informed AI
speculator's order flow will result in a change in price `p_t` over the grid
defined by `P`."

| `v` | `|v - v_bar|` | low noise ratio | high noise ratio |
|---|---|---|---|
| -0.645 | 1.645 | 1.068 | 0.278 |
| 0.326 | 0.674 | 1.064 | 0.134 |
| 0.874 | 0.126 | 1.033 | 0.028 |
| 1.126 | 0.126 | 1.033 | 0.028 |
| 1.674 | 0.674 | 1.064 | 0.134 |
| 2.645 | 1.645 | 1.068 | 0.278 |

Reproduce / 复现:

```bash
cd 03_Code/dgj_sim && python3 -c "
from dgj.config import ExperimentCell
from dgj.game.session import build_grids
for su in (0.1, 100.0):
    g = build_grids(ExperimentCell().with_parameters(noise_std=su))
    step = g.multipliers[1] - g.multipliers[0]
    print(f'sigma_u={su}')
    for k in range(10):
        dp_grid = g.price_grid[k,1] - g.price_grid[k,0]
        dp_act  = g.nash.price_impact * (g.value_grid[k] - 1.0) * step
        print(f'  v={g.value_grid[k]:7.3f}  ratio={abs(dp_act/dp_grid):.3f}')
"
```

**Second use: this table quantifies the paper's own mechanism boundary /
第二个用途：此表把论文自身的机制边界数值化**

Proposition 3.1 states that price-trigger collusion is theoretically infeasible
under high noise because noise destroys the price's ability to reveal
deviations. The paper's wording (`docs/paper_full_text.txt:205`): noise reduces
"informativeness and **rendering prices ineffective for detecting deviations**."

The ratio measures exactly that capability. A price-trigger strategy requires a
rival's one-step deviation to move the realized price into a *different* price
bucket; otherwise the observing agent's state `(p_{t-1}, v_{t-1}, v_t)` is
bit-for-bit identical whether the rival cooperated or cheated, and no trigger
strategy can condition on the deviation. The high-noise ratios of 0.03–0.28 say
the price state falls short of that resolution by a factor of 3 to 30. /
价格触发策略要求对手挪动一档动作后，实际价格落入**不同**的价格桶；否则观察方的
状态逐位相同，触发策略无从条件化。高噪声下 0.03–0.28 表示价格状态的分辨率不足
3 至 30 倍。

The mechanism is transparent in the arithmetic: the numerator (`lambda * dx`,
the price move from one action step) does not depend on `sigma_u` at all, while
the denominator (bucket width) must stretch to cover `+/- 1.96 sigma_u` with only
`n_p = 31` buckets. At `v = 1.674` the numerator is 4.818e-3 in **both** cells;
the bucket widens from 4.528e-3 to 3.586e-2, a factor of 7.9. / 机制在算术上一目
了然：分子与 `sigma_u` 无关，分母却必须用 31 个桶覆盖 `+/- 1.96 sigma_u`。

This makes the table a publishable numerical counterpart to a proposition the
paper argues only in theory. / 因此该表是论文仅以理论论证的命题的可发表数值对应物。

**Outstanding / 待办**

Quote the table in the methodology section as the defence of A22, and consider a
second use in the results or discussion as the measured counterpart to
Proposition 3.1. Verify against Proposition 3.1's exact statement before
claiming correspondence. / 在方法章引用此表作为 A22 的辩护；并考虑在结果或讨论章
作为 Proposition 3.1 的实测对应物再次使用。宣称对应关系前须核对该命题原文。

---

## I-06 — `T_c` has a constant hazard after exploration dies / 探索消失后收敛风险率恒定

- **Status / 状态**: `HYPOTHESIS`
- **Date / 日期**: 2026-08-31
- **Destination / 论文去处**: Methodology / Appendix — justifies survival treatment / 方法章或附录，为生存分析辩护

**Claim / 论点**

Convergence time is not a "wait until learning finishes" process. It is a
memoryless waiting time, so `T_c` should be approximately exponential. / 收敛时间
不是「等到学习完成」的过程，而是一个无记忆等待时间，因此 `T_c` 应近似服从指数分布。

**Mechanism / 机制**

1. `exploration_rate()` depends **only** on the visit count — no `Q` values. With
   `beta = 5e-7` and `t(v) ~ t/10`, the effective rate is `5e-8`, so at
   `t = 10^9`, `eps = e^-50 ~ 2e-22`. Exploration is finished. / 探索率只依赖访问
   次数，与 `Q` 无关；`t = 10^9` 时 `eps ~ 2e-22`，探索彻底结束。
2. `q_learning.update` writes **one cell**. Once `a*` is fixed, the other
   `n_x - 1` actions' `Q` values are frozen permanently. / 更新只写一个格子；`a*`
   固定后，其余动作的 `Q` 永久冻结。
3. The greedy action's `Q` is an EWMA of a *noisy* target (`u_t` is redrawn each
   period), so it never settles. Stationary jitter
   `sd = sigma_target * sqrt(alpha/(2-alpha)) = 0.0709 sigma_target`
   (simulated 0.0707 over 2M steps). / 贪心动作的 `Q` 是带噪目标的指数移动平均，
   永不收敛，稳定抖动标准差为 `0.0709 sigma_target`。
4. A flip is that jitter crossing below a frozen runner-up. Both the jitter and
   the frozen values are constant, so the per-visit flip probability is a
   **constant**. / 翻转即抖动下穿冻结的次优值；两者皆为常数，故每次访问的翻转
   概率恒定。

The knife-edge is near a 4-sd lead / 临界点约在 4 个标准差:

| Lead / 领先差距 | Flip rate per visit | `P(10^6 clean)` |
|---|---|---|
| 1.1 sd | 1.2e-2 | 0 |
| 2.0 sd | 3.1e-3 | 0 |
| 3.0 sd | 2.8e-4 | 1e-123 |
| 4.2 sd | 2.3e-6 | 0.097 |

**Consistency check / 一致性核对**

Paper (`docs/paper_full_text.txt:1256–1258`): "convergence occurs within a range
of approximately 20 million to 50 billion periods" — a 2,500x spread, which a
constant-hazard (exponential) waiting time produces naturally and a
"learning-completes-at-a-typical-time" model does not. / 论文报告收敛期数从 2000 万
到 500 亿，跨度 2500 倍；恒定风险率的指数等待时间自然产生这种跨度，而「学习在某个
典型时刻完成」的模型不会。

**Outstanding / 待办**

Not yet tested on real data. Test: Q–Q plot of observed `converged_at` against an
exponential distribution across the converged cohort. If it holds, censored runs
can be handled by survival analysis instead of being rerun to completion. / 尚未
在真实数据上检验。检验方法：对已收敛队列的 `converged_at` 作指数分布 Q–Q 图。
若成立，删失样本可用生存分析处理，无需全部跑完。

⚠️ The 20-million lower bound is *below* the `~3e8` point where `eps` becomes
negligible, so the constant-hazard story cannot be the whole account across all
parameterizations. It should be stated as applying to the baseline cells. /
2000 万这个下界低于 `eps` 可忽略的 `~3e8`，因此恒定风险率不能解释全部参数配置；
陈述时应限定在基准环境。

---

## I-07 — Censoring may be selection-biased toward low `Delta^C` / 删失可能系统性偏向低合谋样本

- **Status / 状态**: `HYPOTHESIS`
- **Date / 日期**: 2026-08-31
- **Destination / 论文去处**: Results / Limitations — potentially resolves I-02 / 结果与局限章，可能直接解决 I-02

**Claim / 论点**

If price-trigger equilibria take longer to form than the alternatives, then the
5e9 operational cap removed disproportionately many **collusive** sessions. The
low-noise mean would then be downward biased, and I-02 would be a sampling
artefact rather than a replication failure. / 若价格触发式均衡形成更慢，则 5e9 的
运行上限会不成比例地砍掉**更合谋**的 session。低噪声均值因此向下偏，I-02 就是抽样
假象而非复现失败。

**Supporting pattern / 支持性迹象**

- Low-noise pilot converged at `3.09e9` (under the cap) with `Delta^C = 0.319`.
- High-noise pilot converged at `1.54e9` with `Delta^C = 0.748`.
- The mixed 1,000-session mean (0.4156) is *above* the uncensored pilot value —
  direction consistent with a positive `T_c`–`Delta^C` relationship.
- The campaign log already anticipated this: "convergence speed may be
  correlated with the learned strategy" (`experiments/2026-08-30_low_noise_campaign_log.md:88`).

**Test / 检验**

`corr(converged_at, delta_c)` over the converged cohort, plus a scatter plot. If
significantly positive, report the low-noise estimate with a survival-corrected
mean and state the bias direction explicitly. / 对已收敛队列计算
`corr(converged_at, delta_c)` 并作散点图；若显著为正，则以生存校正后的均值报告
低噪声结果，并明确说明偏差方向。

**Outstanding / 待办**

Requires the per-session table (`summary.json` or Step 37A `session_metrics.csv`)
to be brought into the repo from Narval. Not currently present. / 需要把 Narval
上的逐 session 表带回仓库；目前不在仓库内。

---

## I-08 — `beta` is an economic parameter, not a technical one / 探索衰减参数是经济参数而非技术参数

- **Status / 状态**: `VERIFIED` (mechanism); proposed robustness check is `OPEN`
- **Date / 日期**: 2026-09-01
- **Destination / 论文去处**: Methodology + Discussion / 方法章与讨论章

**Claim / 论点**

Exploration stops on a fixed clock, not on any measure of learning progress.
`beta` therefore determines how much of the action space remains permanently
mis-evaluated — i.e. how biased, and hence how collusive, the agents end up.
It is load-bearing for the economics, not a convergence-speed knob. / 探索按固定
时钟停止，与学习进展无关。因此 `beta` 决定了动作空间中有多少部分被永久错误评估，
也就决定了 agent 最终有多偏、有多合谋。它承载经济含义，不是收敛加速旋钮。

**Evidence / 证据**

```python
# dgj/players/speculator/q_learning.py
@njit
def exploration_rate(visits_of_current_value: int, exploration_decay: float) -> float:
    return math.exp(-exploration_decay * visits_of_current_value)
```

The only argument is a visit count. No `Q`, no convergence state. An agent that
has found the optimum and an agent that has merely run out of clock behave
identically thereafter. / 唯一的输入是访问次数，没有 `Q`，没有收敛状态。真正找到
最优的 agent 与只是时钟走完的 agent，此后行为完全相同。

This is precisely the paper's own theoretical construct — Definition 3.3
(`docs/paper_full_text.txt`, p.19): a collusive equilibrium in which speculators
"systematically undervalue aggressive trading strategies due to an incorrect
outcome evaluation system. This system remains uncorrected as learning is
confined to outcomes observed along the equilibrium path." And p.19 on
experience-based equilibrium: beliefs "about off-path outcomes need not align
with expected discounted cash flows under the true distribution, allowing for
significant biases." / 这正是论文自己的理论构造。

**Why this explains the rising branch of the U / 为何解释 U 型的上升支**

Under high noise the profit signal is buried in `u_t`, so distinguishing two
actions requires *more* exploratory draws — but `beta` is fixed, so the budget is
the same. More noise therefore means less identification per unit of exploration,
more permanently frozen errors, more collusion. The paper states the same
(`:1535–1541`): the asymmetry becomes "increasingly difficult to correct through
exploration updates." / 高噪声下利润信号被淹没，区分两个动作需要更多次探索，但
`beta` 固定、预算不变。噪声越大，单位探索识别出的信息越少，永久冻结的错误越多，
合谋越强。论文原文表述一致。

**Outstanding / 待办**

Proposed robustness check: vary `beta` and observe `Delta^C`. If `Delta^C` moves
with `beta`, that **empirically demonstrates** collusion arises from insufficient
exploration rather than merely citing the theory. A few dozen sessions per point
should reveal the trend — far cheaper than a 1,000-session cell.
First verify whether the paper already reports a `beta` sensitivity; if not, this
is an incremental contribution. / 建议的稳健性检验：改变 `beta` 观察 `Delta^C`。
若二者联动，即**实证**了合谋源于探索不足，而非仅引用理论。每个点几十个 session
即可看出趋势，远比 1,000 session 便宜。先核查论文是否已做 `beta` 敏感性；若无，
这是增量贡献。

---

## I-09 — `L^C` is near-deterministic under the baseline calibration / 流动性指标在基准校准下近乎确定

- **Status / 状态**: `VERIFIED`
- **Date / 日期**: 2026-08-31
- **Destination / 论文去处**: Results — market quality, with an interpretive caution / 结果章，市场质量，附解释警告

**Claim / 论点**

Because `1 - xi*lambda_hat = theta(1 - xi*gamma_hat_1) / (theta + xi^2)`, the
liquidity metric reduces to `L^C ~ 2.5e6 / |1 - xi*gamma_hat_1|`. Its level is
dominated by the constant `(theta + xi^2)/theta = 2.5e6`, and it carries almost
no independent information beyond `chi_hat` and `sigma_u`. / 由于该恒等式，流动性
指标化简为 `L^C ~ 2.5e6 / |1 - xi*gamma_hat_1|`；其量级由常数 2.5e6 主导，除
`chi_hat` 与 `sigma_u` 外几乎不携带独立信息。

**Evidence / 证据**

Over 100,000 measurement periods in both pilot cells, `lambda_hat` varies only in
the 12th decimal, and `1 - xi*lambda_hat` is **negative in 100% of periods**
(`xi*gamma_hat_1 ~ 1.54–1.58 > 1`). No period crosses the pole. / 在两个 pilot 的
十万个测量期中，`lambda_hat` 仅在第 12 位小数变动，`1 - xi*lambda_hat` 在 100% 的
期数中为负，无一期穿过奇点。

| Cell | `lambda_hat` mean | `1 - xi*lambda_hat` | `L^C` mean | `L^C` median |
|---|---|---|---|---|
| `sigma_u=0.1` | 0.002000000460 | -2.301e-7 | 4.345e6 | 4.341e6 |
| `sigma_u=100` | 0.002000000429 | -2.145e-7 | 4.663e6 | 4.661e6 |

**Independent validation of the market maker / 做市商的独立验证**

Recovering `gamma_1` from theory,
`gamma = I*chi / ((I*chi)^2 + (sigma_u/sigma_v_hat)^2)`, gives 0.003165 (low) and
0.003089 (high); the values implied by the measured `lambda_hat` are 0.00315 and
0.00307. Agreement to three significant figures means the rolling OLS converges
to exactly where theory says it should. This is strong evidence the market-maker
kernel is correct. / 由理论式反推 `gamma_1` 与实测值三位有效数字吻合，说明滚动 OLS
收敛到理论预测处，是做市商内核正确的有力证据。

**Outstanding / 待办**

Check whether the paper reports `L^C` in the same order of magnitude. If it does
not, there is an interpretation gap to locate. Consider reporting
`|1 - xi*gamma_hat_1|` alongside `L^C`, since that is where the economics lives.
/ 核对论文报告的 `L^C` 量级是否相同；若不同则存在解释差异需定位。建议在报告
`L^C` 的同时报告 `|1 - xi*gamma_hat_1|`，经济含义在后者。

---

## I-10 — The Grossman–Stiglitz inversion / 对 Grossman-Stiglitz 权衡的反转

- **Status / 状态**: `OPEN`
- **Date / 日期**: 2026-08-31
- **Destination / 论文去处**: Discussion / 讨论章

**Claim / 论点**

In the classical Grossman–Stiglitz trade-off, informativeness and arbitrageur
rents are *complements*: rents are the payment for producing information. Here
they are *substitutes* — the rents come from **not** producing information
(over-pruning) or from mutual deterrence (price triggers), and both destroy
informativeness. / 在经典的 Grossman-Stiglitz 权衡中，信息效率与套利者租金是
互补的：租金是生产信息的报酬。而在这里两者是替代的——租金来自**不**生产信息
（过度剪枝）或相互威慑（价格触发），两者都在摧毁信息效率。

**Supporting observation / 支持性观察**

`I^C` is 8.8e6 (low noise) versus 7.08 (high noise) — six orders of magnitude —
yet `Delta^C` is *higher* in the high-noise cell. The environment with the most
"efficient" prices has the least collusion. / 价格信息效率相差六个数量级，但合谋
程度在高噪声环境**更高**。价格最「有效」的环境恰恰合谋最少。

**Argument to develop / 待展开的论点**

The efficient-market hypothesis quietly assumes arbitrageurs are *competitive*
and treats informativeness as a by-product of that competition. Once arbitrageurs
are learning algorithms, competition is no longer primitive — it is an
equilibrium selection. "Market efficiency" is then not a testable state of the
world but a contingency depending on arbitrageurs *failing* to coordinate. /
有效市场假说默认套利者是**竞争性**的，并把信息效率视为竞争的副产品。一旦套利者是
学习型算法，竞争就不再是原始设定，而是均衡选择的结果。「市场有效」于是不是一个可
检验的世界状态，而是一个依赖于套利者**协调失败**的偶然性。

**Outstanding / 待办**

Draft 2–3 pages connecting to Grossman & Stiglitz (1980), Kyle (1985), and the
algorithmic-collusion literature (Calvano et al., 2020). Check that this framing
does not merely restate the source paper's own discussion. / 起草 2–3 页，接上
Grossman-Stiglitz (1980)、Kyle (1985) 与算法合谋文献 (Calvano et al., 2020)；
须确认此框架并非只是复述原文自身的讨论。

---

## I-11 — Documentation lags the codebase / 文档落后于代码

- **Status / 状态**: `HOUSEKEEPING`
- **Date / 日期**: 2026-08-31

Three concrete defects observed / 三处具体缺陷:

1. `docs/development_journal.md` states "the verified market kernel is still pure
   Python and has no Numba backend" and lists the accelerated backend as
   `[PLANNED]`. `03_Code/dgj_sim` exists, is tested, and has run a full campaign.
   `vibe_replication/README.md` carries the same stale claim. / 开发日记与 README
   仍称没有 Numba 后端，但 `dgj_sim` 已存在、已测试并已跑完一轮实验。
2. `dgj/config.py` annotates `price_grid` with `# A5`; the per-value price-grid
   decision is **A22** (A5 is the fixed-point root selection). / `config.py` 把
   `price_grid` 标注为 A5，实际应为 A22。
3. The "822 converged / 178 censored" figures quoted throughout the docs describe
   the **first capped campaign** (Slurm job `2099411`, committed 2026-08-30). If
   the recovery has since completed, no record of it exists in the repository. /
   文档中反复引用的 822/178 描述的是**首轮带上限的实验**；若恢复运行已完成，仓库
   内没有任何记录。
