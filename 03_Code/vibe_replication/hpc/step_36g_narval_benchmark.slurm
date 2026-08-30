#!/bin/bash
# Step 36G: exact formal-loop throughput benchmark on one Narval compute node.
# Step 36G：在一台 Narval 计算节点上测量正式训练循环的吞吐率。
#
# Submit from 03_Code/vibe_replication after replacing YOUR_ACCOUNT below at
# submission time; no account is invented inside this file. / 请从
# 03_Code/vibe_replication 提交，并在提交命令中填写真实 allocation；本文件不猜账号。
#
#   sbatch --account=YOUR_ACCOUNT hpc/step_36g_narval_benchmark.slurm

#SBATCH --job-name=vibe36g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=hpc/logs/step36g-%j.out
#SBATCH --error=hpc/logs/step36g-%j.err

set -Eeuo pipefail

BENCHMARK_JOB_ROOT=""
record_exit_status() {
    exit_code=$?
    set +e
    if [[ -n "${BENCHMARK_JOB_ROOT}" && -d "${BENCHMARK_JOB_ROOT}" ]]; then
        if [[ ${exit_code} -eq 0 ]]; then
            job_state="PASS"
        else
            job_state="FAIL"
        fi
        printf 'state=%s\nexit_code=%s\n' "${job_state}" "${exit_code}" \
            > "${BENCHMARK_JOB_ROOT}/job_exit_status.txt"
    fi
    trap - EXIT
    exit "${exit_code}"
}
trap record_exit_status EXIT

# Fail before any calculation when this is not a scheduled compute job.
# 若不是 scheduler 分配的计算任务，在计算开始前立即停止。
: "${SLURM_JOB_ID:?SLURM_JOB_ID is missing; submit with sbatch / 请用 sbatch 提交}"
: "${SLURM_TMPDIR:?SLURM_TMPDIR is missing / 缺少 SLURM_TMPDIR}"
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is missing / 缺少 SLURM_SUBMIT_DIR}"
: "${SCRATCH:?SCRATCH is missing / 缺少 SCRATCH}"

if [[ ! -f "${SLURM_SUBMIT_DIR}/run_formal_experiment.py" ]]; then
    echo "Submit from 03_Code/vibe_replication. / 请从 03_Code/vibe_replication 提交。" >&2
    exit 2
fi

REPOSITORY_ROOT="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)"
COMMIT_HASH="$(git -C "${REPOSITORY_ROOT}" rev-parse HEAD)"
if [[ -n "$(git -C "${REPOSITORY_ROOT}" status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked files are dirty; commit them first. / 已跟踪文件有改动；请先 commit。" >&2
    exit 2
fi
for required_path in \
    "03_Code/vibe_replication/steps/step_36g_narval_throughput_benchmark.py" \
    "03_Code/vibe_replication/tests/test_step36g_narval_throughput_benchmark.py" \
    "03_Code/vibe_replication/hpc/step_36g_narval_benchmark.slurm"
do
    if ! git -C "${REPOSITORY_ROOT}" cat-file -e "${COMMIT_HASH}:${required_path}"; then
        echo "Step 36G is absent from commit ${COMMIT_HASH}: ${required_path} / Step 36G 尚未 commit。" >&2
        exit 2
    fi
done

# Load the same pinned scientific environment used by the benchmark receipt.
# 加载与 benchmark 报告一致的固定科学环境。
module load python/3.13.2
VIBE_ENV_PATH="${VIBE_R0_ENV_PATH:-${SCRATCH}/vibe-r0-env}"
if [[ ! -f "${VIBE_ENV_PATH}/bin/activate" ]]; then
    echo "Missing virtual environment: ${VIBE_ENV_PATH} / 找不到虚拟环境。" >&2
    exit 2
fi
source "${VIBE_ENV_PATH}/bin/activate"
PYTHON_BIN="${VIBE_ENV_PATH}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Venv Python is not executable: ${PYTHON_BIN} / venv Python 不可执行。" >&2
    exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUTF8=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PYTHON_BIN}" -c 'import sys, numpy as np; assert sys.version_info >= (3, 13), sys.version; assert np.__version__ == "2.5.1", np.__version__; print("Python:", sys.version); print("NumPy:", np.__version__)'
module list 2>&1
lscpu | grep -E 'Model name|Socket|Core|Thread|CPU max MHz' || true

# Run a detached local clone of the committed bytes, not an uncommitted
# login-node working tree. Keeping ``.git`` lets the Python receipt independently
# verify the commit and clean tracked state. / 运行 commit 的本地 detached clone，
# 不运行登录节点工作区中的临时改动；保留 ``.git`` 让 Python 报告再次独立核对
# commit 与已跟踪文件是否干净。
NODE_SOURCE_ROOT="${SLURM_TMPDIR}/vibe36g-source"
git clone --no-local --no-checkout "${REPOSITORY_ROOT}" "${NODE_SOURCE_ROOT}"
git -C "${NODE_SOURCE_ROOT}" -c advice.detachedHead=false checkout --detach "${COMMIT_HASH}"
if [[ -n "$(git -C "${NODE_SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]; then
    echo "Detached benchmark clone is unexpectedly dirty. / detached benchmark clone 意外含有改动。" >&2
    exit 2
fi
NODE_PROJECT_ROOT="${NODE_SOURCE_ROOT}/03_Code/vibe_replication"
cd "${NODE_PROJECT_ROOT}"

echo "Step 36G commit / Step 36G commit: ${COMMIT_HASH}"
echo "Compute host / 计算节点: $(hostname)"
echo "Running the full validation suite first. / 首先运行完整验证套件。"
"${PYTHON_BIN}" -X utf8 -m unittest discover -s tests -v

# A new job directory is never reused. Each cell then gets a separate Python
# process so interpreter warm state cannot masquerade as a low/high economic
# difference. / 每个 job 使用全新目录；low/high 分别启动 Python 进程，避免解释器
# 缓存被误解为噪声环境造成的经济差异。
BENCHMARK_PARENT="${SCRATCH}/vibe_replication_step36g/${COMMIT_HASH}"
BENCHMARK_JOB_CANDIDATE="${BENCHMARK_PARENT}/job_${SLURM_JOB_ID}"
mkdir -p "${BENCHMARK_PARENT}"
if ! mkdir "${BENCHMARK_JOB_CANDIDATE}"; then
    echo "Refusing existing job output: ${BENCHMARK_JOB_CANDIDATE} / 拒绝复用已有 job 输出。" >&2
    exit 2
fi
# Assign the trap-owned path only after this job created it successfully.
# 只有本 job 成功建立目录后，才把它交给 EXIT trap 写状态。
BENCHMARK_JOB_ROOT="${BENCHMARK_JOB_CANDIDATE}"

# The first pilot is deliberately small enough for a 30-minute allocation.
# Override these environment variables only when requesting matching walltime.
# 首次 pilot 故意保持在 30 分钟内；若扩大期数，也必须同时申请相应 walltime。
WARMUP_PERIODS="${STEP36G_WARMUP_PERIODS:-25}"
MEASURED_PERIODS="${STEP36G_MEASURED_PERIODS:-250}"
if ! [[ "${WARMUP_PERIODS}" =~ ^[1-9][0-9]*$ && "${MEASURED_PERIODS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Benchmark periods must be positive integers. / benchmark 期数必须是正整数。" >&2
    exit 2
fi
echo "Warm-up periods / 预热期数: ${WARMUP_PERIODS}"
echo "Measured periods / 计时期数: ${MEASURED_PERIODS}"

echo "Benchmarking low noise in a fresh process. / 在新进程测试低噪声。"
"${PYTHON_BIN}" -X utf8 steps/step_36g_narval_throughput_benchmark.py \
    --cell low \
    --benchmark-root "${BENCHMARK_JOB_ROOT}/low" \
    --warmup-periods "${WARMUP_PERIODS}" \
    --measured-periods "${MEASURED_PERIODS}"

echo "Benchmarking high noise in a fresh process. / 在新进程测试高噪声。"
"${PYTHON_BIN}" -X utf8 steps/step_36g_narval_throughput_benchmark.py \
    --cell high \
    --benchmark-root "${BENCHMARK_JOB_ROOT}/high" \
    --warmup-periods "${WARMUP_PERIODS}" \
    --measured-periods "${MEASURED_PERIODS}"

echo "Step 36G completed. / Step 36G 已完成。"
echo "Reports / 报告目录: ${BENCHMARK_JOB_ROOT}"
