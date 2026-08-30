# Narval protocol: repair low noise, then run high noise

The scientific stopping rule is convergence: both agents' complete greedy
policies must remain unchanged for 1,000,000 consecutive periods. A finite
Slurm job is only a computing shift. If that shift ends first, Python saves an
exact checkpoint and the next shift resumes it.

科学停止规则是“真正收敛”：两个 agent 在所有状态下的完整最优策略连续 1,000,000
期不变。单次 Slurm job 只是一次计算班次；班次先结束时，只保存 checkpoint，下次续跑。

Important / 重要：

- Never submit a new round while the previous array is still running. A
  per-session operating-system lock rejects accidental overlap.
- Never edit or pull the checkout between rounds. Every checkpoint is bound to
  a SHA-256 fingerprint of the scientific source code.
- `WORK_PERIODS=5000000000` means “up to 5B additional periods in this job,”
  not “stop the scientific session at 5B.”
- Core low/high runs use `--irf-paths 0`; mechanism/IRF analysis is a later,
  separate experiment.

## 1. Update, verify Numba, test, and record provenance / 更新、测试与留档

Run from a fresh SSH login. / 每次重新登录后都可从这里开始。

```bash
cd "$SCRATCH/AI-agent-trading-new"
git pull --ff-only

# Formal runs must have no modified tracked code. Untracked log files are okay.
test -z "$(git status --porcelain --untracked-files=no)" || {
  echo "STOP: tracked files are modified"; exit 1;
}

cd 03_Code/dgj_sim
source "$HOME/venv-aitrading/bin/activate"

python -c 'import sys,numpy,numba; from dgj._jit import HAVE_NUMBA; print(sys.version); print("NumPy",numpy.__version__,"Numba",numba.__version__,"HAVE_NUMBA",HAVE_NUMBA); assert HAVE_NUMBA'
python -m unittest discover -s tests -p 'test_*.py' -v
mkdir -p logs "$SCRATCH/ai-trading-runs/campaign_receipts"

RECEIPT="$SCRATCH/ai-trading-runs/campaign_receipts/code_$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  date -u --iso-8601=seconds
  git rev-parse HEAD
  git status --short
  sha256sum hpc/submit_array.slurm
  python -c 'import sys,numpy,numba; from dgj.provenance import scientific_identity; print(sys.version); print("NumPy",numpy.__version__,"Numba",numba.__version__); print(scientific_identity())'
} | tee "$RECEIPT"
```

Do not submit if the test suite or `HAVE_NUMBA` check fails. / 测试或 Numba
检查失败时不要提交；纯 Python 会慢得不适合正式实验。

## 2. Create a clean low-noise recovery cohort / 建立低噪声修复目录

The old capped directory remains audit evidence. The helper independently
copies and SHA-256-verifies the 822 genuinely converged files. It omits the 178
censored IDs, which must deterministically replay from period zero because the
old runner deleted their final checkpoints.

旧目录不改。脚本独立复制并核对 822 个真正收敛文件；原来 178 个 censored ID 因没有
可用 checkpoint，使用相同 session ID 和 seed 从第 0 期重跑。

The commit shown below is the reported old campaign checkout and the audited
core-equivalence base. The old NPZ files do **not** embed commit, runtime, or
chunk identity, and no historical runtime receipt has yet been recovered; the
composed cohort must therefore be reported as mixed/partially unversioned.

```bash
cd "$SCRATCH/AI-agent-trading-new/03_Code/dgj_sim"
source "$HOME/venv-aitrading/bin/activate"

OLD_OUT="$SCRATCH/ai-trading-runs/low_noise_per_value"
NEW_OUT="$SCRATCH/ai-trading-runs/low_noise_per_value_uncapped_recovery"

python hpc/prepare_recovery.py "$OLD_OUT" "$NEW_OUT" \
  --expect-sessions 1000 --expect-censored 178 --training-chunk-size 1000000 \
  --legacy-source-commit 9ab452fb6dc54e7ce25a9ec9417e346aa177d366

echo "copied results: $(find "$NEW_OUT" -maxdepth 1 -type f -name 'session_*.npz' | wc -l)"
echo "IDs to replay:  $(wc -l < "$NEW_OUT/recovery_session_ids.txt")"
```

The copy-and-hash pass reads every accepted file twice and can take several
minutes; it prints a progress line every 50 files. / 复制并核对 822 个文件可能需要
几分钟；脚本每完成 50 个文件会显示一次进度。

Required output: `copied results: 822` and `IDs to replay: 178`. The helper
refuses an existing target directory, wrong ID set, wrong identities, invalid
rows, or an unexpected censor count.

## 3. One-task live signal/resume pilot / 先做一个真实暂停测试

This 15-minute pilot uses one of the censored identities. It verifies that
Slurm's `USR1` warning reaches Python, produces a loadable checkpoint, and is
mapped to a successful computing shift. The later full array simply resumes
this same ID, so the pilot is not wasted.

```bash
cd "$SCRATCH/AI-agent-trading-new/03_Code/dgj_sim"
source "$HOME/venv-aitrading/bin/activate"
mkdir -p logs

OLD_OUT="$SCRATCH/ai-trading-runs/low_noise_per_value"
NEW_OUT="$SCRATCH/ai-trading-runs/low_noise_per_value_uncapped_recovery"
PILOT_ID=$(head -n 1 "$NEW_OUT/recovery_session_ids.txt")
SEED=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment_seed"])' "$OLD_OUT/cell.json")
LABEL=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["cell"]["label"])' "$OLD_OUT/cell.json")
[[ "$LABEL" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "STOP: unsafe label"; exit 1; }
CELL_ARGS="--noise-std 0.1 --speculators 2 --rho 0.95 --xi 500 --convergence-periods 1000000 --measurement-periods 100000 --prehistory nash --price-grid per_value --label $LABEL --irf-paths 0"

SUBMISSION=$(OUT="$NEW_OUT" SEED="$SEED" WORK_PERIODS=50000000000 \
  CELL_ARGS="$CELL_ARGS" sbatch --parsable --time=00:15:00 \
  --signal=USR1@300 --array="$PILOT_ID" hpc/submit_array.slurm)
PILOT_JOB_ID=${SUBMISSION%%;*}
echo "$PILOT_JOB_ID" | tee -a "$NEW_OUT/submission_job_ids.txt"
echo "pilot job: $PILOT_JOB_ID; session: $PILOT_ID"
```

Wait until `squeue -j "$PILOT_JOB_ID"` is empty, then validate:

```bash
NEW_OUT="$SCRATCH/ai-trading-runs/low_noise_per_value_uncapped_recovery"
PILOT_ID=$(head -n 1 "$NEW_OUT/recovery_session_ids.txt")
PILOT_JOB_ID=$(head -n 1 "$NEW_OUT/submission_job_ids.txt")
sacct -j "$PILOT_JOB_ID" -X -n -P --format=JobIDRaw,State,ExitCode,Elapsed,MaxRSS \
  | tee "$NEW_OUT/sacct_${PILOT_JOB_ID}.txt"
python hpc/verify_checkpoint.py "$NEW_OUT" "$PILOT_ID" \
  --expect-stop-reason scheduler_signal --expect-job-id "$PILOT_JOB_ID"
```

Proceed only if `sacct` reports success and the verifier prints `VALID
checkpoint`. / 只有两项都通过才进行 178-session 正式提交。

## 4. Submit one low-noise work round / 提交一轮低噪声计算

This block is reconnect-safe: it reconstructs every shell variable. / 这个区块在
重新 SSH 登录后也能直接运行，不依赖上次 shell 变量。

```bash
cd "$SCRATCH/AI-agent-trading-new/03_Code/dgj_sim"
source "$HOME/venv-aitrading/bin/activate"
mkdir -p logs

OLD_OUT="$SCRATCH/ai-trading-runs/low_noise_per_value"
NEW_OUT="$SCRATCH/ai-trading-runs/low_noise_per_value_uncapped_recovery"
IDS=$(tr -d '\n' < "$NEW_OUT/recovery_array.txt")
SEED=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment_seed"])' "$OLD_OUT/cell.json")
LABEL=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["cell"]["label"])' "$OLD_OUT/cell.json")
[[ -n "$IDS" && "$LABEL" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "STOP: invalid recovery metadata"; exit 1; }
CELL_ARGS="--noise-std 0.1 --speculators 2 --rho 0.95 --xi 500 --convergence-periods 1000000 --measurement-periods 100000 --prehistory nash --price-grid per_value --label $LABEL --irf-paths 0"

SUBMISSION=$(OUT="$NEW_OUT" SEED="$SEED" WORK_PERIODS=5000000000 \
  CELL_ARGS="$CELL_ARGS" sbatch --parsable --array="${IDS}%50" hpc/submit_array.slurm)
JOB_ID=${SUBMISSION%%;*}
echo "$JOB_ID" | tee -a "$NEW_OUT/submission_job_ids.txt"
echo "submitted low-noise recovery job $JOB_ID"
```

Five billion is the maximum additional work in this invocation only. A session
that needs 12B may use 5B + 5B + 2B over three rounds. / 50 亿只是本轮工作量；
它不是 session 的累计停止上限。

## 5. Monitor and resume low noise / 查看并续跑低噪声

```bash
NEW_OUT="$SCRATCH/ai-trading-runs/low_noise_per_value_uncapped_recovery"
JOB_ID=$(tail -n 1 "$NEW_OUT/submission_job_ids.txt")
squeue -j "$JOB_ID" -o '%.18i %.2t %.10M %.10L %R'
```

After the array disappears from `squeue`:

```bash
NEW_OUT="$SCRATCH/ai-trading-runs/low_noise_per_value_uncapped_recovery"
JOB_ID=$(tail -n 1 "$NEW_OUT/submission_job_ids.txt")
sacct -j "$JOB_ID" -X -n -P --format=JobIDRaw,State,ExitCode,Elapsed,MaxRSS \
  | tee "$NEW_OUT/sacct_${JOB_ID}.txt"
echo "results:     $(find "$NEW_OUT" -maxdepth 1 -type f -name 'session_*.npz' | wc -l)"
echo "checkpoints: $(find "$NEW_OUT" -maxdepth 1 -type f -name 'ckpt_*.npz' | wc -l)"
python hpc/list_review_due.py "$NEW_OUT"
```

`Slurm COMPLETED` means only that one computing shift ended safely. A
`ckpt_XXXX.npz` means that scientific session is still unfinished.

If checkpoints remain, wait until the previous array is completely finished,
then run the **entire reconnect-safe block in Step 4 again**. Completed IDs are
validated and skipped; checkpointed IDs add another work slice. The same-source
fingerprint and session lock prevent unsafe mixing or overlap.

若仍有 checkpoint，确认上一轮完全结束，再完整复制 Step 4。不要只复制最后一行，因为
重新登录后 `IDS`、`SEED`、`LABEL` 等变量都不存在。

If a session reaches 50B cumulative training periods, the runner now refuses
another ordinary resume for that ID. Inspect its progress, Q/policy diagnostics,
and logs and record the decision. Only after that review, submit the listed IDs
with `ALLOW_AFTER_REVIEW=1`; do not set this variable pre-emptively. Fifty
billion is not automatic convergence or failure.

After the review has been written down, use this separate sparse-array command;
never export `ALLOW_AFTER_REVIEW=1` for the ordinary full list.

```bash
cd "$SCRATCH/AI-agent-trading-new/03_Code/dgj_sim"
source "$HOME/venv-aitrading/bin/activate"
mkdir -p logs
OLD_OUT="$SCRATCH/ai-trading-runs/low_noise_per_value"
NEW_OUT="$SCRATCH/ai-trading-runs/low_noise_per_value_uncapped_recovery"
REVIEW_IDS=$(python hpc/list_review_due.py "$NEW_OUT" --ids-only)
SEED=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment_seed"])' "$OLD_OUT/cell.json")
LABEL=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["cell"]["label"])' "$OLD_OUT/cell.json")
CELL_ARGS="--noise-std 0.1 --speculators 2 --rho 0.95 --xi 500 --convergence-periods 1000000 --measurement-periods 100000 --prehistory nash --price-grid per_value --label $LABEL --irf-paths 0"
[[ -n "$REVIEW_IDS" ]] || { echo "No reviewed IDs to submit"; exit 1; }
SUBMISSION=$(OUT="$NEW_OUT" SEED="$SEED" WORK_PERIODS=5000000000 \
  ALLOW_AFTER_REVIEW=1 CELL_ARGS="$CELL_ARGS" sbatch --parsable \
  --array="${REVIEW_IDS}%10" hpc/submit_array.slurm)
REVIEW_JOB_ID=${SUBMISSION%%;*}
echo "$REVIEW_JOB_ID" | tee -a "$NEW_OUT/submission_job_ids.txt"
```

## 6. Accept and aggregate low noise / 正式验收低噪声

Do not aggregate until file counts are exactly 1,000 results and zero
checkpoints. Counts alone are not enough; the strict command validates every
identity, convergence field, row shape, and finite value.

```bash
(
  set -o pipefail
  cd "$SCRATCH/AI-agent-trading-new/03_Code/dgj_sim"
  source "$HOME/venv-aitrading/bin/activate"
  NEW_OUT="$SCRATCH/ai-trading-runs/low_noise_per_value_uncapped_recovery"
  python hpc/aggregate_dir.py "$NEW_OUT" --expected-sessions 1000 \
    | tee "$NEW_OUT/aggregate_stdout.txt"
)
```

The repaired cohort mixes 822 schema-less legacy files with 178 current-schema
files. Its core measures (`Delta C`, trading intensity, informativeness,
liquidity, and mispricing) may be analyzed only as a **mixed-provenance cohort**
backed by the recorded core-equivalence audit—not as a homogeneous
current-provenance run. The old runtime remains an explicit limitation unless
an historical receipt is recovered. For the strongest publication claim,
rerun all 1,000 low-noise IDs from scratch in a new current-schema directory.
Mechanism shares are deliberately `null` because IRF evidence is absent or
mixed; do not interpret them.

### 6B. Optional homogeneous low-noise rerun / 可选：全新重跑 1,000 个低噪声

For a publication cohort with no legacy-provenance qualification, start a new
directory and run all 1,000 IDs with the corrected schema. This costs more but
is the strongest design. / 若要最严格的论文证据，用新目录从头跑全部 1,000 个。

```bash
(
set -euo pipefail
cd "$SCRATCH/AI-agent-trading-new/03_Code/dgj_sim"
source "$HOME/venv-aitrading/bin/activate"
mkdir -p logs
LOW_CLEAN="$SCRATCH/ai-trading-runs/low_noise_per_value_uncapped_clean"
[[ ! -e "$LOW_CLEAN" ]] || { echo "STOP: $LOW_CLEAN already exists"; exit 1; }
LOW_ARGS="--noise-std 0.1 --speculators 2 --rho 0.95 --xi 500 --convergence-periods 1000000 --measurement-periods 100000 --prehistory nash --price-grid per_value --label low_noise_clean --irf-paths 0"
SUBMISSION=$(OUT="$LOW_CLEAN" SEED=20260828 WORK_PERIODS=5000000000 \
  CELL_ARGS="$LOW_ARGS" sbatch --parsable --array='0-999%50' hpc/submit_array.slurm)
LOW_CLEAN_JOB_ID=${SUBMISSION%%;*}
mkdir -p "$LOW_CLEAN"
echo "$LOW_CLEAN_JOB_ID" | tee -a "$LOW_CLEAN/submission_job_ids.txt"
)
```

Resume it exactly like the high-noise cell below (same directory and IDs,
`noise-std 0.1`) until 1,000 results and zero checkpoints, then use strict
aggregation with `--expected-sessions 1000`.

## 7. First high-noise submission / 首次提交高噪声

Start only after Step 6 succeeds. This subshell deliberately refuses to reuse
an existing directory; failure does not close the SSH session.

```bash
(
set -euo pipefail
cd "$SCRATCH/AI-agent-trading-new/03_Code/dgj_sim"
source "$HOME/venv-aitrading/bin/activate"
mkdir -p logs

HIGH_OUT="$SCRATCH/ai-trading-runs/high_noise_per_value_uncapped"
[[ ! -e "$HIGH_OUT" ]] || { echo "STOP: $HIGH_OUT already exists"; exit 1; }
HIGH_ARGS="--noise-std 100 --speculators 2 --rho 0.95 --xi 500 --convergence-periods 1000000 --measurement-periods 100000 --prehistory nash --price-grid per_value --label high_noise --irf-paths 0"

SUBMISSION=$(OUT="$HIGH_OUT" SEED=20260828 WORK_PERIODS=5000000000 \
  CELL_ARGS="$HIGH_ARGS" sbatch --parsable --array='0-999%50' hpc/submit_array.slurm)
HIGH_JOB_ID=${SUBMISSION%%;*}
mkdir -p "$HIGH_OUT"
echo "$HIGH_JOB_ID" | tee -a "$HIGH_OUT/submission_job_ids.txt"
echo "submitted high-noise job $HIGH_JOB_ID"
)
```

## 8. Resume and aggregate high noise / 续跑并汇总高噪声

After each high-noise array is completely absent from `squeue`, inspect counts.
If checkpoints or missing results remain, use this reconnect-safe block:

```bash
cd "$SCRATCH/AI-agent-trading-new/03_Code/dgj_sim"
source "$HOME/venv-aitrading/bin/activate"
mkdir -p logs
HIGH_OUT="$SCRATCH/ai-trading-runs/high_noise_per_value_uncapped"
[[ -f "$HIGH_OUT/cell.json" ]] || { echo "STOP: high-noise cell does not exist"; exit 1; }
HIGH_ARGS="--noise-std 100 --speculators 2 --rho 0.95 --xi 500 --convergence-periods 1000000 --measurement-periods 100000 --prehistory nash --price-grid per_value --label high_noise --irf-paths 0"

SUBMISSION=$(OUT="$HIGH_OUT" SEED=20260828 WORK_PERIODS=5000000000 \
  CELL_ARGS="$HIGH_ARGS" sbatch --parsable --array='0-999%50' hpc/submit_array.slurm)
HIGH_JOB_ID=${SUBMISSION%%;*}
echo "$HIGH_JOB_ID" | tee -a "$HIGH_OUT/submission_job_ids.txt"
echo "resubmitted high-noise job $HIGH_JOB_ID"
```

Also run `python hpc/list_review_due.py "$HIGH_OUT"`. If it lists IDs, review
them first and submit only that sparse list with `ALLOW_AFTER_REVIEW=1`, using
the same pattern as the low-noise review block above.

When there are exactly 1,000 results and zero checkpoints:

```bash
(
  set -o pipefail
  cd "$SCRATCH/AI-agent-trading-new/03_Code/dgj_sim"
  source "$HOME/venv-aitrading/bin/activate"
  HIGH_OUT="$SCRATCH/ai-trading-runs/high_noise_per_value_uncapped"
  python hpc/aggregate_dir.py "$HIGH_OUT" --expected-sessions 1000 \
    | tee "$HIGH_OUT/aggregate_stdout.txt"
)
```

Only after both strict aggregations pass should the low/high comparison figure
or table be generated. / 只有低噪声和高噪声都通过严格验收，才制作最终对比图表。
