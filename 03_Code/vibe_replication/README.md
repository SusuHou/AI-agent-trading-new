# Vibe Replication / 直觉式逐步复现

This folder rebuilds Dou, Goldstein, and Ji (2025) in small, testable steps.  We
do not move to the next component until the current component agrees with the
paper. / 本目录把 Dou、Goldstein 与 Ji（2025）的模型拆成可测试的小步骤；当前模块与
论文核对一致后，才进入下一模块。

## Python version / Python 版本

Use Python **3.13 or newer**. Step 32 deliberately uses `math.fma` to evaluate
`1 - xi * lambda_hat_t` with one rounding; older Python versions can falsely
turn a finite near-singular liquidity value into an exact zero. The local
development interpreter is Python 3.13. When moving the project to a cluster,
create or load a Python 3.13+ environment before installing the remaining
dependencies. / 请使用 **Python 3.13 或更高版本**。第 32 步特意使用 `math.fma`
只舍入一次地计算 `1 - xi * lambda_hat_t`；旧版本可能把接近奇点但仍有限的流动性
错误变成精确零。当前本地开发解释器是 Python 3.13；迁移到超算时，应先建立或加载
Python 3.13+ 环境，再安装其余依赖。

Install the pinned scientific dependency from this directory: / 在本目录安装
已经固定版本的科学计算依赖：

```powershell
py -3 -m pip install -r requirements.txt
```

## Repository baseline and development record / 仓库基准与开发记录

Repository Baseline R0 freezes the readable Python implementation through
Step 36F plus the formal core runner. The reconstructed history, evidence
labels, validation boundary, two experiment tracks, and remaining work are in
[`docs/development_journal.md`](docs/development_journal.md). / 仓库基准 R0
固定截至 Step 36F 的可读 Python 实现以及正式核心 runner。重建历史、证据标签、
验证边界、两条实验路线与剩余工作记录在
[`docs/development_journal.md`](docs/development_journal.md)。

Generated results and checkpoints are intentionally excluded from Git. They
are machine- and run-specific evidence, not source code; formal results must be
archived separately with their manifest and commit hash. / 生成的结果和 checkpoint
有意不进入 Git：它们是特定机器和特定运行的证据，不是源码。正式结果必须连同
manifest 和 commit hash 另行归档。

## Step 1 / 第一步

The first step contains: / 第一步包括：

- the paper's baseline parameters / 论文基准参数；
- the Gaussian-quantile fundamental-value grid / 高斯分位数基本价值网格；
- the discrete standard deviation `sigma_v_hat` / 离散标准差 `sigma_v_hat`；
- tests derived directly from the paper / 直接来自论文的测试。

Run the demonstration from this directory: / 在本目录运行演示：

```powershell
python -m src.step01_value_grid
```

Run the tests: / 运行测试：

```powershell
py -3 -m unittest discover -s tests -v
```

## Step 36A experiment smoke row / 第 36A 步实验冒烟结果行

Before launching expensive Q-learning experiments, Step 36A checks that one
completed session can feed every existing metric and become one saved table
row. It deliberately uses a synthetic stable policy, so the output is pipeline
evidence only—not a paper replication finding. / 在启动昂贵的 Q-learning 实验前，
第 36A 步先检查一个完整 session 能否进入所有现有指标，并保存成一行表格。它故意
使用合成稳定策略，因此输出只证明管线接通，不是论文复现发现。

```powershell
py -3 -X utf8 steps/step_36a_one_session_result_row.py
py -3 -X utf8 -m unittest tests.test_step36a_one_session_result_row -v
```

The matching JSON and CSV files are written under
`results/step36a_engineering_smoke/`. / 对应 JSON 与 CSV 文件会写入
`results/step36a_engineering_smoke/`。

## Formal low/high core experiment / 正式低/高噪声核心实验

The formal entrypoint is `run_formal_experiment.py`. It does not create a
smaller economic experiment. Its two plans always contain the paper-scale
settings below. / 正式入口是 `run_formal_experiment.py`。它不会建立缩小版经济
实验；两个计划始终使用下列论文规模设定：

- 1,000 independent sessions per noise environment / 每个噪声环境 1,000 个独立 session；
- `sigma_u=0.1` and `sigma_u=100` / 低噪声与高噪声；
- 1,000,000 consecutive periods without a greedy-policy change / 最优策略连续 100 万期不变；
- 100,000 frozen-policy measurement periods / 收敛后测量 10 万期。

Freeze and verify both formal plans once: / 首先一次性固定并核对两个正式计划：

```powershell
py -3 -X utf8 run_formal_experiment.py init
```

Run one operational chunk of a real formal session: / 运行某个真实正式 session 的
一个可恢复计算片段：

```powershell
py -3 -X utf8 run_formal_experiment.py run-session --cell low --session-index 0 --period-budget 100000 --checkpoint-interval 100000
```

`--period-budget` limits only this invocation. It does **not** cap or alter the
scientific experiment; the next invocation resumes the exact saved Q tables,
market-maker history, random streams, and convergence counter. / `--period-budget`
只限制本次进程，不限制或改变正式实验；下次会精确恢复 Q 表、做市商历史、随机流与
收敛计数器。

Check progress without changing it: / 只读查看进度：

```powershell
py -3 -X utf8 run_formal_experiment.py status --cell all
```

For a SLURM array, indices `0..999` mean low noise and `1000..1999` mean high
noise. Every array worker must receive a finite `--period-budget` chosen to
finish before its wall-time; the runner automatically reads
`SLURM_ARRAY_TASK_ID`. / 在 SLURM 数组中，`0..999` 对应低噪声，
`1000..1999` 对应高噪声。每个 worker 都必须设置能在 wall-time 之前完成的
`--period-budget`；入口会自动读取 `SLURM_ARRAY_TASK_ID`。

Only after all 1,000 sessions in a cell are complete may it be collected: /
只有一个 cell 的 1,000 个 session 全部完成后才能汇总：

```powershell
py -3 -X utf8 run_formal_experiment.py collect --cell low
py -3 -X utf8 run_formal_experiment.py collect --cell high
```

The core run stops at Step 36E and reports `Delta^C`, trading intensity, price
informativeness, liquidity, mispricing diagnostics, profits, and convergence
periods. Steps 35D--35F are the later IRF/mechanism analysis and can reuse the
same completed sessions. / 核心实验停在 Step 36E，报告合谋利润、交易强度、价格
信息效率、流动性、错误定价诊断、利润与收敛期数；Step 35D--35F 属于之后的
IRF/机制分析，可复用同一批已完成 session。

Current limitation: the verified market kernel is still pure Python and has no
Numba backend. The plans and resume/HPC worker are ready, but launching all
2,000 jobs should wait for a cluster-specific throughput measurement and an
acceleration decision. / 当前限制：已验证的市场 kernel 仍是纯 Python，没有
Numba 后端。计划、续跑与 HPC worker 已接通，但启动全部 2,000 个任务前，必须先
在目标超算测量吞吐量并决定加速方案。

## Step 36G Narval throughput benchmark / 第 36G 步 Narval 吞吐率测试

Step 36G measures the **same formal training entrypoint** on one scheduled
Narval CPU before we choose wall time, chunk size, or whether acceleration is
necessary. The Python tool defaults to a 1,000-period warm-up and 10,000-period
measured resume. The first 30-minute Slurm pilot explicitly uses a conservative
25+250 periods per cell; after observing Narval speed, we can request suitable
wall time and repeat with a longer sample. Low and high noise are launched in
separate Python processes. / 第 36G 步在一颗由 Slurm 分配的 Narval CPU 上测量
**同一个正式训练入口**，然后我们才决定 wall time、每个 chunk 多长，以及是否必须
加速。Python 工具默认预热 1,000 期并续跑计时 10,000 期；首次 30 分钟 Slurm
pilot 明确使用保守的每个 cell `25+250` 期。看到 Narval 实测速度后，再申请合适的
wall time 并扩大样本。低噪声与高噪声分别使用新的 Python 进程。

This benchmark does not change any economic parameter, cannot reach the
1,000,000-period convergence criterion, writes zero measurement rows, and
marks every report `research_result=false`. A Slurm report includes a clearly
named linear extrapolation to one million periods; a local smoke report does
not. This extrapolation is not a measured million-period duration and total
convergence time remains unknown until agents actually converge. / 它不修改任何
经济参数，不可能在 benchmark 期数内达到 100 万期稳定判据，写入零条测量记录，
并把报告明确标记为 `research_result=false`。Slurm 报告会给出名称明确的“百万期
线性外推”，本地 smoke 报告不会；它不是实际跑完百万期的时长。agent 真正收敛前，
总训练时间仍然未知。

On Narval, after the pinned environment exists and this commit has been pulled,
submit from `03_Code/vibe_replication`: / 在 Narval 已建立固定环境并拉取本 commit
以后，从 `03_Code/vibe_replication` 提交：

```bash
sbatch --account=YOUR_REAL_ALLOCATION hpc/step_36g_narval_benchmark.slurm
```

Replace `YOUR_REAL_ALLOCATION` with the allocation shown by your Alliance
account; it is intentionally not guessed in the script. Check the queue and,
after completion, inspect resource use with: / 请把 `YOUR_REAL_ALLOCATION` 换成
Alliance 账户显示的真实 allocation；脚本故意不猜。查看队列和任务完成后的资源用量：

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ReqMem,MaxRSS
```

The two checksum-protected reports are saved below
`$SCRATCH/vibe_replication_step36g/<commit>/job_<job-id>/`. Generated benchmark
data are not committed to Git. / 两份带校验的报告保存在
`$SCRATCH/vibe_replication_step36g/<commit>/job_<job-id>/` 下；benchmark 生成数据
不进入 Git。
