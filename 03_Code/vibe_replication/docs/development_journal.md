# Development Journal / 开发日志

> **Audit note / 审计说明**
>
> This journal was created retrospectively on 2026-08-30. Earlier milestones
> are reconstructed from the code, automated tests, paper-to-code checklist,
> saved artifacts, and the one contemporaneous learning-log entry. They are
> not a complete transcript of the earlier Codex conversation. File
> modification times show only when a file was last changed; they do not prove
> the exact time at which a research step was completed. / 本日志于 2026-08-30
> 事后建立。此前里程碑根据现存代码、自动测试、论文核对表、保存的产物以及一条
> 当时记录的学习日志重建；它不是此前 Codex 对话的完整逐字记录。文件修改时间只
> 表示最后修改时间，不能证明某项研究步骤准确的完成时刻。

## How to read this journal / 如何阅读本日志

- `[CONTEMPORANEOUS]`: recorded when the event occurred / 当时记录。
- `[RECONSTRUCTED—VERIFIED]`: reconstructed later and supported by current
  code plus automated validation / 事后重建，并有当前代码和自动验证支持。
- `[RECONSTRUCTED—UNVERIFIED]`: remembered or inferred but not independently
  demonstrated / 来自回忆或推断，尚无独立证据。
- `[PLANNED]`: future work; not yet completed / 未来工作，尚未完成。

`DONE` means only that a component passed the engineering checks stated for
its own boundary. It does **not** mean that the paper's formal empirical result
has been reproduced. / `DONE` 只表示某个模块在自己明确限定的范围内通过工程
检查，**不表示**论文的正式实证结果已经复现。

## Repository Baseline R0 / 仓库基准 R0

Status: `[RECONSTRUCTED—VERIFIED]` / 状态：`[事后重建—已验证]`

This baseline freezes the readable reference implementation through Step 36F
and the separate formal core experiment entrypoint. It deliberately excludes
temporary files, Python bytecode, rendered previews, generated experiment
results, checkpoints, and bundles. / 本基准固定截至 Step 36F 的可读参考实现，
以及独立的正式核心实验入口；它有意排除临时文件、Python 字节码、渲染预览、
生成的实验结果、checkpoint 与 bundle。

Verified local environment / 已验证的本地环境：

- Operating system / 操作系统：Windows.
- Python / Python：3.13.1.
- NumPy / NumPy：2.5.1.
- Test command / 测试命令：
  `py -3 -X utf8 -m unittest discover -s tests -v`.
- Observed result on 2026-08-30 / 2026-08-30 实际结果：
  `253 tests; 253 passed; 0 failures; 0 errors; 0 skips`.
- Runtime of that validation / 本次验证耗时：approximately 68.7 seconds / 约 68.7 秒。

What this proves / 这能证明：the currently connected modules satisfy the 253
encoded unit and integration checks on this machine / 当前连接的模块在本机满足
253 项已编码的单元及整合检查。

What this does not prove / 这不能证明：the model is a perfect interpretation
of every ambiguous sentence in the paper, the full experiment will converge,
or the paper's numerical findings have been reproduced / 不能证明我们对论文所有
模糊表述的解释都唯一正确，也不能证明完整实验一定收敛或论文数值结果已经复现。

## Step 36G pre-deployment validation / Step 36G 部署前验证

Status: `[CONTEMPORANEOUS]` / 状态：`[当时记录]`

- Date / 日期：2026-08-30.
- Tested implementation commit / 被测试的实现 commit：
  `7f915c05c87720348d789ba48447cbcac79af9af`.
- Python / Python：3.13.1.
- NumPy / NumPy：2.5.1.
- Focused command / 专项命令：
  `py -3 -X utf8 -m unittest tests.test_step36g_narval_throughput_benchmark -v`.
- Focused result / 专项结果：`9 tests; 9 passed`.
- Full command / 完整命令：
  `py -3 -X utf8 -m unittest discover -s tests -v`.
- Full result / 完整结果：`262 tests; 262 passed; 0 failures; 0 errors; 0 skips`.
- Full-suite runtime / 完整套件耗时：238.662 seconds / 238.662 秒。
- Real local connection smoke / 真实本地连接 smoke：one low-noise paper-mode
  session ran `1` fresh period and exactly resumed the same checkpoint for `2`
  periods; the persisted receipt reloaded with `execution_scope` equal to
  `local_connection_smoke`, cumulative training period `3`, zero measurement
  rows, `research_result=false`, and no million-period extrapolation / 一个低噪声
  paper-mode session 先 fresh 运行 1 期，再从同一 checkpoint 精确续跑 2 期；
  持久化报告成功重读，明确标为本地连接 smoke、累计训练 3 期、测量记录为零、
  `research_result=false`，且不发布百万期外推。

What this proves / 这能证明：the Step-36G Python boundary, exact-resume audit,
low/high forwarding, CLI, report semantics/checksum, source-manifest separation,
and all earlier encoded checks pass locally / Step 36G 的 Python 边界、精确续跑
审计、低/高噪声转发、CLI、报告语义/校验、源码清单隔离，以及此前所有已编码检查
在本机通过。

What this does not prove / 这不能证明：the Slurm wrapper has run on Narval,
Narval's throughput or memory use, convergence time, `Delta^C`, or any paper
finding / 尚不能证明 Slurm wrapper 已在 Narval 实跑，也没有 Narval 吞吐率、内存、
收敛时长、`Delta^C` 或任何论文发现。

## Reconstructed milestone map / 重建的里程碑地图

All entries in this section are `[RECONSTRUCTED—VERIFIED]` unless stated
otherwise. Detailed equations, assumptions, tests, and unresolved decisions
are maintained in [the paper-to-code checklist](00_paper_to_code_checklist.md).
/ 除非另有标注，本节均属于“事后重建—已验证”。详细方程、假设、测试及未决
选择记录在[论文到代码核对表](00_paper_to_code_checklist.md)中。

1. **Steps 1–5 — Market primitives / 市场基础量。** Fundamental-value grid,
   noise order, total order flow, information-insensitive investor demand, and
   informed-speculator profit / 基本价值网格、噪声订单、总订单流、信息不敏感
   投资者需求与知情投机者利润。
2. **Steps 6–11 — Theory and benchmarks / 理论与基准。** Market-maker
   objective and theoretical price; Nash/cartel strategies, fixed points, and
   benchmark profits / 做市商目标与理论价格，以及 Nash/cartel 策略、不动点和
   基准利润。
3. **Steps 12–17 — Finite learning problem / 有限学习问题。** Action grids,
   value-specific price grids, state representation, initial state, Q-table
   initialization, and Q-value interpretation / 动作网格、按价值划分的价格网格、
   状态表示、初始状态、Q 表初始化与 Q 值含义。
4. **Steps 18–21 — Learning traders / 学习型交易者。** Epsilon-greedy
   choice, value-specific exploration counters, Q-learning update, and two
   independent informed traders / epsilon-greedy 选择、按价值计数的探索率、
   Q-learning 更新与两位独立知情交易者。
5. **Steps 22–28 — Adaptive market and session lifecycle / 自适应市场与
   session 生命周期。** Rolling history, two OLS regressions, adaptive pricing,
   efficient rolling updates, initial history, one complete period, reproducible
   random streams, convergence tracking, and measurement phases / 滚动历史、
   两个 OLS 回归、自适应定价、高效滚动更新、初始历史、完整单期、可复现随机流、
   收敛追踪与测量阶段。
6. **Steps 29–34 — Paper outcome pipeline / 论文结果管线。** Matched-path
   collusion profitability, trading intensity, price informativeness,
   liquidity, mispricing, and the mechanism-classification contract / 同路径
   合谋利润、交易强度、价格信息效率、流动性、错误定价与机制分类契约。
7. **Steps 35A–35F — IRF/mechanism infrastructure / IRF 与机制基础设施。**
   Converged checkpoints, paired paths, long-run baseline, unshocked calibration
   paths, cell-level shock calibration arithmetic, and paired-response
   classification / 收敛 checkpoint、配对路径、长期基线、无冲击校准路径、
   cell 级冲击校准运算及配对反应分类。
8. **Steps 36A–36F — Persistence and experiment infrastructure / 持久化与
   实验基础设施。** Session result rows, experiment manifests, exact resume,
   one-session training, complete measurement, and the persisted per-session
   Step-35D calibration bridge / session 结果行、实验 manifest、精确续跑、
   单 session 训练、完整测量及持久化的单 session Step-35D 校准桥。
9. **Unnumbered formal core runner / 未编号正式核心 runner。**
   `run_formal_experiment.py` creates and manages the low/high-noise core plans.
   It intentionally stops at Step 36E, so the core results can be produced
   before the optional IRF/mechanism analysis / `run_formal_experiment.py`
   建立并管理低/高噪声核心计划，并有意停在 Step 36E，让核心结果可以先于可选的
   IRF/机制分析产生。
10. **Step 36G — Target-machine throughput benchmark / 目标机器吞吐率测试。**
    The code and Slurm wrapper now isolate a fresh formal paper-mode sandbox,
    run a short warm-up, exactly resume the same session for the measured
    chunk, and publish a checksum-protected operational receipt. Low/high cells
    use separate Python processes. The Narval job itself is still
    `[PLANNED]`; therefore no Narval rate or total runtime is reported yet. /
    代码与 Slurm 包装现已建立隔离的正式 paper-mode sandbox，先短暂预热，再精确
    续跑同一 session 的计时 chunk，并发布带校验的运行报告；低/高噪声使用不同
    Python 进程。Narval job 本身仍属 `[PLANNED]`，因此目前没有 Narval 速度或总
    运行时间数值。

## Two experiment tracks / 两条实验路线

The project now contains two related but distinct tracks. Confusing them would
make the completion status look more advanced than it is. / 当前项目包含两条相关
但不同的路线；若把它们混在一起，会让完成状态显得比实际更超前。

### Core low/high experiment / 核心低噪声与高噪声实验

- Components / 组成：Steps 36B–36E plus `run_formal_experiment.py`.
- Intended scale / 目标规模：1,000 low-noise sessions and 1,000 high-noise
  sessions / 低噪声与高噪声各 1,000 个 session。
- Current status / 当前状态：`[PLANNED]` empirical run. Formal plans and resume
  machinery exist, but the 2,000 formal sessions have not been completed / 正式
  计划和续跑机制已存在，但 2,000 个正式 session 尚未完成。

### IRF and mechanism experiment / IRF 与机制实验

- Components / 组成：Steps 35A–35F plus the Step-36F persistence bridge.
- Current status / 当前状态：the per-session bridge is verified; full-cell
  orchestration across completed sessions remains `[PLANNED]` / 单 session 桥已
  验证；跨全部已完成 session 的完整 cell 调度仍属计划工作。

## Material implementation change / 重大实现变更

Status: `[RECONSTRUCTED—VERIFIED]` / 状态：`[事后重建—已验证]`

The price-state discretization was changed from one global price grid to a
value-specific family `P(v)`. The purpose is to make the movement caused by a
one-step action deviation visible in the lagged-price state at the intended
resolution. The pre-change files are preserved outside the tracked baseline in
`backups/before_per_value_price_grid_20260829.zip`. Results produced before and
after this change must not be pooled as if they came from the same model. / 价格
状态离散化已经从一个全局价格网格改为按基本价值划分的一组 `P(v)`。其目的在于：
让偏离一档动作所造成的价格变化，能以预期分辨率进入滞后价格状态。修改前文件保存在
`backups/before_per_value_price_grid_20260829.zip`，但该备份不进入源码基准。修改前后
产生的结果不能被当作同一个模型的数据混合汇总。

## Decision register / 研究决定登记

Decisions A1–A24 are maintained in
[the paper-to-code checklist](00_paper_to_code_checklist.md). Each decision must
continue to distinguish: what the paper states, what it leaves unspecified,
the implementation selected here, the likely effect on results, and the
required sensitivity analysis. / A1–A24 保存在
[论文到代码核对表](00_paper_to_code_checklist.md)中。每项决定必须继续区分：论文明确说了什么、没有
说明什么、本复现选择了什么、可能怎样影响结果，以及需要什么敏感性分析。

## Validation ledger / 验证登记格式

Every future validation should append an entry containing / 今后的每次验证都应追加：

```text
Date and time / 日期时间:
Commit hash / commit 哈希:
Python and dependency versions / Python 与依赖版本:
Command / 命令:
Observed output / 实际输出:
Passed or failed / 通过或失败:
What this proves / 这能证明什么:
What this does not prove / 这不能证明什么:
```

Every experiment run must also be labelled as one of / 每次实验运行还必须标为：

- engineering smoke test / 工程冒烟测试；
- debug-scale experiment / 调试规模实验；
- formal paper-scale experiment / 论文正式规模实验。

Synthetic Step-36A numbers and small Step-35F demonstrations are pipeline
evidence and must never be reported as paper findings. / Step 36A 的合成数值及
Step 35F 的小规模演示只属于管线证据，绝不能作为论文发现报告。

## Next work after Baseline R0 / R0 之后的工作

1. `[CODE VERIFIED; NARVAL RUN PLANNED]` Submit Step 36G and benchmark the exact
   formal market loop on the target machine / 提交 Step 36G，在目标机器上测量
   正式市场循环的真实吞吐量。
2. `[PLANNED]` Add an accelerated backend while retaining the readable Python
   path, then require controlled same-seed parity / 保留可读 Python 路径并增加加速
   后端，再要求受控的同 seed 一致性验证。
3. `[WRAPPER IMPLEMENTED; COMPUTE-NODE TEST PLANNED]` Pin and test the cluster
   environment and Slurm packaging / 包装已经实现；仍需在计算节点固定并测试环境。
4. `[PLANNED]` Run and collect 1,000 low-noise plus 1,000 high-noise formal
   sessions / 运行并汇总低、高噪声各 1,000 个正式 session。
5. `[PLANNED]` Add the full-cell IRF manager if the mechanism analysis is in
   scope / 若需要机制分析，再加入完整 cell 的 IRF manager。
6. `[PLANNED]` Produce paper figures/tables and compare them with the source
   paper / 制作论文图表并与原文比较。
