"""Measure the exact formal training path before launching 2,000 sessions.

在启动 2,000 个正式 session 前，测量同一条正式训练路径的真实速度。

This is an *operational benchmark*, not a smaller economic experiment.  It
uses ``run_formal_experiment.run_one_session`` with the paper-mode plan, first
for a short warm-up and then for a timed exact-resume chunk.  The output tells
us how many training periods one allocated CPU can execute per second (the
provided Slurm wrapper targets Narval).  It does not tell us when Q-learning
will converge and it is not a paper result.

这是一个“运行速度测试”，不是缩小版经济实验。它调用正式入口，先短暂预热，
再从 checkpoint 精确续跑并计时。输出只能回答“一颗已分配 CPU 每秒能跑多少期”，
不能回答 Q-learning 何时收敛，也不能当作论文结果。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping
import argparse
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = PROJECT_ROOT / "steps"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


import numpy as np

import run_formal_experiment as formal
from steps.step_36b_experiment_manifest import load_experiment_cell_plan
from steps.step_36d_single_session_training_runner import (
    FRESH_START,
    INCOMPLETE,
    RESUMED_START,
    STATUS_FILE_NAME,
    PersistedTrainingStatus,
    load_training_status,
)


SCHEMA_VERSION = "step36g-narval-throughput-benchmark-v1"
BENCHMARK_MODE = "exact-formal-training-throughput-only"
REPORT_FILE_NAME = "step36g_throughput_report.json"
SCOPE_MARKER_FILE_NAME = ".step36g_benchmark_scope.json"
# Keep this internal name short.  Checkpoint filenames are necessarily long,
# and a verbose parent name can cross Windows' traditional 260-character path
# limit during the optional local smoke test.  Isolation comes from the new
# outer root and its scope marker, not from a long name. / 内部名保持简短；
# checkpoint 文件名本来就长，冗长父目录会让可选的 Windows smoke test 超过传统
# 260 字符限制。隔离依靠全新外层目录及 scope marker，而不是长名字。
FORMAL_SANDBOX_DIRECTORY_NAME = "formal"

# These defaults are deliberately far below the paper's 1,000,000-period
# convergence streak.  Therefore the benchmark cannot accidentally enter the
# 100,000-row measurement phase. / 这些默认值远小于论文要求的 100 万期稳定
# streak，所以 benchmark 不会意外进入 10 万条测量阶段。
DEFAULT_WARMUP_PERIODS = 1_000
DEFAULT_MEASURED_PERIODS = 10_000
DEFAULT_CHECKPOINT_INTERVAL = 1_000_000
DEFAULT_SESSION_INDEX = 0
LOCAL_SAFETY_MAX_PERIODS = 100
PINNED_NUMPY_VERSION = "2.5.1"

SLURM_ENVIRONMENT_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_CPUS_PER_TASK",
    "SLURM_MEM_PER_NODE",
    "SLURM_JOB_PARTITION",
    "SLURM_CLUSTER_NAME",
)
THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

Clock = Callable[[], float]


@dataclass(frozen=True)
class BenchmarkEnvironment:
    """Small, whitelisted machine record; never dump the full environment.

    小型、白名单式机器记录；绝不把包含秘密的全部环境变量写入报告。
    """

    hostname: str
    operating_system: str
    operating_system_release: str
    machine: str
    processor: str
    python_executable: str
    python_version: str
    python_implementation: str
    numpy_version: str
    logical_cpu_count: int | None
    process_id: int
    git_commit: str | None
    git_tracked_files_clean: bool | None
    loaded_modules: str | None
    slurm: tuple[tuple[str, str], ...]
    thread_limits: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ThroughputBenchmarkReport:
    """Checksum-protected receipt for one noise cell. / 单个噪声 cell 的带校验报告。"""

    schema_version: str
    benchmark_mode: str
    execution_scope: str
    slurm_compute_node_verified: bool
    cluster_name: str | None
    cell: str
    noise_standard_deviation: float
    session_index: int
    task_id: str
    task_sha256: str
    plan_sha256: str
    run_config_sha256: str
    family_manifest_sha256: str
    execution_source_sha256: str
    result_pipeline_source_sha256: str
    step36g_source_sha256: str
    benchmark_root: str
    isolated_formal_artifact_root: str
    warmup_period_budget: int
    measured_period_budget: int
    checkpoint_interval_periods: int
    warmup_start_mode: str
    warmup_ending_verified_period: int
    warmup_training_elapsed_seconds: float
    warmup_training_periods_per_second: float
    warmup_end_to_end_elapsed_seconds: float
    measured_start_mode: str
    measured_starting_verified_period: int
    measured_ending_verified_period: int
    measured_training_elapsed_seconds: float
    measured_training_periods_per_second: float
    measured_end_to_end_elapsed_seconds: float
    measured_end_to_end_periods_per_second: float
    linear_extrapolation_seconds_per_million_at_observed_rate: float | None
    outer_status: str
    outer_phase: str
    outer_stop_reason: str
    inner_stop_reason: str
    measurement_rows_committed: int
    peak_resident_set_megabytes: float | None
    created_at_utc: str
    environment: BenchmarkEnvironment
    economic_parameters_changed_for_benchmark: bool
    convergence_observed: bool
    research_result: bool
    paper_results_ready: bool
    total_formal_runtime_known: bool
    report_sha256: str


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    """Return deterministic JSON text. / 返回确定不变的 JSON 文本。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _sha256_json(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalized_file_sha256(path: Path) -> str:
    """Hash source with platform-independent line endings. / 用跨平台换行取源码哈希。"""

    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer. / {name} 必须是正整数。")
    return value


def _git_text(*arguments: str) -> str | None:
    """Read one harmless Git fact without invoking a shell. / 不经过 shell 读取 Git 信息。"""

    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def collect_environment(environment: Mapping[str, str] | None = None) -> BenchmarkEnvironment:
    """Collect only reproducibility fields that are safe to publish.

    只收集可公开、且复现速度测试所需的环境字段。
    """

    source = os.environ if environment is None else environment
    commit = _git_text("rev-parse", "HEAD")
    tracked_status = _git_text("status", "--porcelain", "--untracked-files=no")
    return BenchmarkEnvironment(
        hostname=socket.gethostname(),
        operating_system=platform.system(),
        operating_system_release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor(),
        python_executable=sys.executable,
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        numpy_version=np.__version__,
        logical_cpu_count=os.cpu_count(),
        process_id=os.getpid(),
        git_commit=commit,
        git_tracked_files_clean=(None if tracked_status is None else tracked_status == ""),
        loaded_modules=source.get("LOADEDMODULES"),
        slurm=tuple((key, source[key]) for key in SLURM_ENVIRONMENT_KEYS if key in source),
        thread_limits=tuple((key, source[key]) for key in THREAD_ENVIRONMENT_KEYS if key in source),
    )


def validate_execution_environment(
    environment: BenchmarkEnvironment,
    *,
    allow_non_slurm: bool,
    total_period_budget: int,
) -> None:
    """Refuse a misleading or unsafe benchmark setup. / 拒绝误导或不安全的测试环境。"""

    if sys.version_info < (3, 13):
        raise RuntimeError("Step 36G requires Python 3.13+. / Step 36G 要求 Python 3.13+。")
    if environment.numpy_version != PINNED_NUMPY_VERSION:
        raise RuntimeError(
            f"NumPy must equal {PINNED_NUMPY_VERSION}; observed {environment.numpy_version}. "
            f"/ NumPy 必须是 {PINNED_NUMPY_VERSION}；当前为 {environment.numpy_version}。"
        )

    slurm = dict(environment.slurm)
    thread_limits = dict(environment.thread_limits)
    if "SLURM_JOB_ID" not in slurm:
        if not allow_non_slurm:
            raise RuntimeError(
                "Run this benchmark through Slurm on a compute node. / "
                "请通过 Slurm 在计算节点上运行此 benchmark。"
            )
        if total_period_budget > LOCAL_SAFETY_MAX_PERIODS:
            raise RuntimeError(
                f"Non-Slurm safety limit is {LOCAL_SAFETY_MAX_PERIODS} total periods. "
                f"/ 非 Slurm 安全上限是合计 {LOCAL_SAFETY_MAX_PERIODS} 期。"
            )
        return

    if slurm.get("SLURM_CPUS_PER_TASK") != "1":
        raise RuntimeError("Benchmark must request exactly one CPU. / benchmark 必须只申请一颗 CPU。")
    for key in THREAD_ENVIRONMENT_KEYS:
        if thread_limits.get(key) != "1":
            raise RuntimeError(f"{key} must equal 1. / {key} 必须等于 1。")
    if environment.git_tracked_files_clean is not True:
        raise RuntimeError(
            "The benchmark checkout must have no tracked-file changes. / "
            "benchmark 的 checkout 不能有已跟踪文件改动。"
        )


def _paths_overlap(first: Path, second: Path) -> bool:
    a = first.resolve()
    b = second.resolve()
    return a == b or a in b.parents or b in a.parents


def prepare_isolated_roots(benchmark_root: Path) -> tuple[Path, Path]:
    """Create a new benchmark root and an isolated inner formal root.

    建立全新的 benchmark 根目录，以及隔离的正式 runner 内层目录。

    Existing directories are rejected rather than erased.  This makes an
    accidental overwrite impossible. / 已存在目录会被拒绝而不是删除，避免覆盖。
    """

    if not isinstance(benchmark_root, Path):
        raise TypeError("benchmark_root must be pathlib.Path. / benchmark_root 必须是 pathlib.Path。")
    outer = benchmark_root.resolve()
    production = formal.DEFAULT_ARTIFACT_ROOT.resolve()
    if _paths_overlap(outer, production):
        raise ValueError(
            "Benchmark output must not overlap the production artifact root. / "
            "benchmark 输出不能与正式实验目录重叠。"
        )
    if outer.exists():
        raise FileExistsError(
            f"Refusing to reuse benchmark root: {outer} / 拒绝复用 benchmark 目录：{outer}"
        )
    outer.mkdir(parents=True)
    inner = outer / FORMAL_SANDBOX_DIRECTORY_NAME
    marker = {
        "schema_version": SCHEMA_VERSION,
        "scope": BENCHMARK_MODE,
        "research_result": False,
        "paper_results_ready": False,
    }
    (outer / SCOPE_MARKER_FILE_NAME).write_text(
        _canonical_json(marker, indent=2) + "\n",
        encoding="utf-8",
    )
    return outer, inner


def _training_status_path(formal_root: Path, cell: str, relative_task_directory: str) -> Path:
    """Locate Step 36D's validated status inside the isolated tree.

    在隔离目录中定位 Step 36D 的已验证状态文件。
    """

    return (
        formal_root
        / formal.CELL_ARTIFACT_RELATIVE_PATHS[cell]
        / Path(relative_task_directory)
        / STATUS_FILE_NAME
    )


def _validate_outer_result(
    result: Mapping[str, object],
    *,
    cell: str,
    session_index: int,
    expected_ending_period: int,
) -> None:
    expected = {
        "cell": cell,
        "session_index": session_index,
        "status": INCOMPLETE,
        "phase": "training",
        "stop_reason": "training_not_yet_converged",
        "training_periods_verified": expected_ending_period,
        "measurement_rows_committed": 0,
        "complete_evidence_available": False,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(
                f"Formal runner returned unexpected {key}: {result.get(key)!r}; expected {value!r}. "
                f"/ 正式 runner 的 {key} 不符合预期。"
            )


def _validate_training_attempt(
    status: PersistedTrainingStatus,
    *,
    expected_start_mode: str,
    expected_start: int,
    expected_budget: int,
    expected_end: int,
    expected_attempt_number: int,
) -> None:
    scientific = status.scientific_outcome
    attempt = status.attempt
    checks = {
        "status": (scientific.status, INCOMPLETE),
        "stop_reason": (scientific.stop_reason, "invocation_period_budget_reached"),
        "training_stage_only": (scientific.training_stage_only, True),
        "measurement_periods_completed": (scientific.measurement_periods_completed, 0),
        "attempt_number": (attempt.attempt_number, expected_attempt_number),
        "start_mode": (attempt.start_mode, expected_start_mode),
        "starting_training_period": (attempt.starting_training_period, expected_start),
        "successful_periods_this_attempt": (
            attempt.successful_periods_this_attempt,
            expected_budget,
        ),
        "ending_verified_period": (attempt.ending_verified_period, expected_end),
        "invocation_period_budget": (attempt.invocation_period_budget, expected_budget),
    }
    for name, (observed, expected) in checks.items():
        if observed != expected:
            raise RuntimeError(
                f"Training status mismatch for {name}: {observed!r}; expected {expected!r}. "
                f"/ training status 的 {name} 不符合预期。"
            )
    if expected_start_mode == FRESH_START and attempt.input_checkpoint is not None:
        raise RuntimeError("A fresh attempt cannot have an input checkpoint. / fresh 调用不能有输入 checkpoint。")
    if expected_start_mode == RESUMED_START and attempt.input_checkpoint is None:
        raise RuntimeError("A resumed attempt must name its input checkpoint. / resumed 调用必须记录输入 checkpoint。")
    if attempt.elapsed_seconds <= 0.0 or attempt.periods_per_second is None:
        raise RuntimeError("Training timer did not produce a positive rate. / 训练计时没有产生正速度。")
    if not math.isclose(
        attempt.periods_per_second,
        expected_budget / attempt.elapsed_seconds,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise RuntimeError("Step 36D throughput arithmetic is inconsistent. / Step 36D 吞吐率算术不一致。")


def _validate_resume_handoff(
    warmup_status: PersistedTrainingStatus,
    measured_status: PersistedTrainingStatus,
) -> None:
    """Prove the measured call loaded the exact warm-up checkpoint.

    证明计时调用加载的正是预热调用保存的那个 checkpoint，而不只是期数碰巧相同。
    """

    warmup_checkpoint = warmup_status.scientific_outcome.latest_mid_training_checkpoint
    measured_input = measured_status.attempt.input_checkpoint
    if warmup_checkpoint is None or measured_input is None:
        raise RuntimeError("Exact resume lacks a checkpoint reference. / 精确续跑缺少 checkpoint 引用。")
    if measured_input != warmup_checkpoint:
        raise RuntimeError(
            "Measured run did not resume the exact warm-up checkpoint. / "
            "计时调用没有续跑预热阶段的同一个 checkpoint。"
        )


def _peak_resident_set_megabytes() -> float | None:
    """Return a diagnostic RSS value when the OS exposes one. / 若系统支持则返回 RSS 诊断值。"""

    try:
        import resource

        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError, OSError, ValueError):
        return None
    # Linux reports KiB; macOS reports bytes. Narval is Linux.
    divisor = 1024.0 if platform.system() != "Darwin" else 1024.0 * 1024.0
    return raw / divisor


def _report_without_checksum(report: ThroughputBenchmarkReport) -> dict[str, object]:
    payload = asdict(report)
    payload.pop("report_sha256")
    return payload


def validate_report(report: ThroughputBenchmarkReport) -> None:
    """Validate identity, arithmetic, and honesty flags. / 验证身份、算术与诚实标记。"""

    if not isinstance(report, ThroughputBenchmarkReport):
        raise TypeError("report has the wrong type. / report 类型错误。")
    if report.schema_version != SCHEMA_VERSION or report.benchmark_mode != BENCHMARK_MODE:
        raise ValueError("Unsupported Step 36G report schema. / 不支持的 Step 36G report 格式。")
    if report.execution_scope not in ("slurm_compute_benchmark", "local_connection_smoke"):
        raise ValueError("Unknown Step 36G execution scope. / 未知 Step 36G 执行范围。")
    slurm_fields = dict(report.environment.slurm)
    environment_is_slurm = "SLURM_JOB_ID" in slurm_fields
    if report.slurm_compute_node_verified != environment_is_slurm:
        raise ValueError("Slurm verification flag disagrees with the environment. / Slurm 验证标记与环境不一致。")
    if (report.execution_scope == "slurm_compute_benchmark") != environment_is_slurm:
        raise ValueError("Execution scope disagrees with Slurm evidence. / 执行范围与 Slurm 证据不一致。")
    if report.cluster_name != slurm_fields.get("SLURM_CLUSTER_NAME"):
        raise ValueError("Cluster label disagrees with Slurm evidence. / cluster 标签与 Slurm 证据不一致。")
    if report.cell not in formal.CELL_NOISE_STDS:
        raise ValueError("Unknown experiment cell. / 未知实验 cell。")
    if not math.isclose(
        report.noise_standard_deviation,
        formal.CELL_NOISE_STDS[report.cell],
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("Noise standard deviation differs from the formal cell. / 噪声标准差与正式 cell 不同。")
    _positive_integer(report.warmup_period_budget, "warmup_period_budget")
    _positive_integer(report.measured_period_budget, "measured_period_budget")
    _positive_integer(
        report.checkpoint_interval_periods,
        "checkpoint_interval_periods",
    )
    if report.warmup_period_budget + report.measured_period_budget >= formal.FORMAL_STABILITY_STREAK:
        raise ValueError("Benchmark is too long to guarantee training-only scope. / benchmark 太长，无法保证只在训练阶段。")
    if report.warmup_start_mode != FRESH_START:
        raise ValueError("Warm-up must start fresh. / 预热必须从 fresh 开始。")
    if report.measured_start_mode != RESUMED_START:
        raise ValueError("Measured call must resume. / 计时调用必须是 resumed。")
    if report.warmup_ending_verified_period != report.warmup_period_budget:
        raise ValueError("Warm-up period accounting is inconsistent. / 预热期数核算不一致。")
    if report.measured_starting_verified_period != report.warmup_ending_verified_period:
        raise ValueError("Measured start does not follow warm-up. / 计时起点没有紧接预热终点。")
    if report.measured_ending_verified_period - report.measured_starting_verified_period != report.measured_period_budget:
        raise ValueError("Measured period difference is inconsistent. / 被测时期差不一致。")
    positive_finite_values = (
        report.warmup_training_elapsed_seconds,
        report.warmup_training_periods_per_second,
        report.warmup_end_to_end_elapsed_seconds,
        report.measured_training_elapsed_seconds,
        report.measured_training_periods_per_second,
        report.measured_end_to_end_elapsed_seconds,
        report.measured_end_to_end_periods_per_second,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive_finite_values):
        raise ValueError("All Step 36G timing values must be positive and finite. / 所有 Step 36G 计时值必须为正且有限。")
    arithmetic_checks = (
        (
            report.warmup_training_periods_per_second,
            report.warmup_period_budget / report.warmup_training_elapsed_seconds,
        ),
        (
            report.measured_training_periods_per_second,
            report.measured_period_budget / report.measured_training_elapsed_seconds,
        ),
        (
            report.measured_end_to_end_periods_per_second,
            report.measured_period_budget / report.measured_end_to_end_elapsed_seconds,
        ),
    )
    if any(
        not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=0.0)
        for observed, expected in arithmetic_checks
    ):
        raise ValueError("Step 36G timing arithmetic is inconsistent. / Step 36G 计时算术不一致。")
    projection = report.linear_extrapolation_seconds_per_million_at_observed_rate
    if environment_is_slurm:
        if projection is None or not math.isclose(
            projection,
            1_000_000.0 / report.measured_training_periods_per_second,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError("Slurm linear extrapolation is inconsistent. / Slurm 线性外推算术不一致。")
    elif projection is not None:
        raise ValueError("A local smoke test cannot publish a million-period projection. / 本地 smoke test 不能发布百万期外推。")
    hash_fields = (
        report.task_sha256,
        report.plan_sha256,
        report.run_config_sha256,
        report.family_manifest_sha256,
        report.execution_source_sha256,
        report.result_pipeline_source_sha256,
        report.step36g_source_sha256,
    )
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hash_fields):
        raise ValueError("A Step 36G source/identity hash is invalid. / 某个 Step 36G 身份哈希无效。")
    if report.environment.numpy_version != PINNED_NUMPY_VERSION:
        raise ValueError("Report used the wrong NumPy version. / report 使用了错误 NumPy 版本。")
    if environment_is_slurm and (
        report.environment.git_commit is None
        or re.fullmatch(r"[0-9a-f]{40}", report.environment.git_commit) is None
        or report.environment.git_tracked_files_clean is not True
    ):
        raise ValueError("Slurm report lacks clean Git provenance. / Slurm report 缺少干净的 Git 来源。")
    if report.peak_resident_set_megabytes is not None and (
        not math.isfinite(report.peak_resident_set_megabytes)
        or report.peak_resident_set_megabytes < 0.0
    ):
        raise ValueError("Peak RSS diagnostic is invalid. / peak RSS 诊断值无效。")
    if any(
        (
            report.economic_parameters_changed_for_benchmark,
            report.convergence_observed,
            report.research_result,
            report.paper_results_ready,
            report.total_formal_runtime_known,
        )
    ):
        raise ValueError("Step 36G honesty flags are invalid. / Step 36G 诚实标记无效。")
    if (
        report.outer_status != INCOMPLETE
        or report.outer_phase != "training"
        or report.outer_stop_reason != "training_not_yet_converged"
        or report.inner_stop_reason != "invocation_period_budget_reached"
        or report.measurement_rows_committed != 0
    ):
        raise ValueError("Step 36G crossed its training-only boundary. / Step 36G 越过了仅训练边界。")
    expected_digest = _sha256_json(_report_without_checksum(report))
    if report.report_sha256 != expected_digest:
        raise ValueError("Step 36G report checksum failed. / Step 36G report 校验失败。")


def save_report(report: ThroughputBenchmarkReport, path: Path) -> Path:
    """Write once; never replace another benchmark. / 只写一次；绝不覆盖另一份 benchmark。"""

    validate_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical_json(asdict(report), indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def load_report(path: Path) -> ThroughputBenchmarkReport:
    """Reload and checksum-check one report. / 重读并校验一份报告。"""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        environment_raw = raw.pop("environment")
        environment = BenchmarkEnvironment(
            **{
                **environment_raw,
                "slurm": tuple(tuple(item) for item in environment_raw["slurm"]),
                "thread_limits": tuple(
                    tuple(item) for item in environment_raw["thread_limits"]
                ),
            }
        )
        report = ThroughputBenchmarkReport(environment=environment, **raw)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("Cannot read a complete Step 36G report. / 无法读取完整 Step 36G report。") from error
    validate_report(report)
    return report


def run_throughput_benchmark(
    *,
    benchmark_root: Path,
    cell: str,
    warmup_periods: int = DEFAULT_WARMUP_PERIODS,
    measured_periods: int = DEFAULT_MEASURED_PERIODS,
    checkpoint_interval_periods: int = DEFAULT_CHECKPOINT_INTERVAL,
    session_index: int = DEFAULT_SESSION_INDEX,
    allow_non_slurm: bool = False,
    environment: BenchmarkEnvironment | None = None,
    clock: Clock = time.perf_counter,
) -> tuple[ThroughputBenchmarkReport, Path]:
    """Run one fresh-process cell benchmark and publish its receipt.

    运行一个单 cell、单进程 benchmark，并发布带校验报告。
    """

    if cell not in formal.CELL_NOISE_STDS:
        raise ValueError("cell must be 'low' or 'high'. / cell 必须是 low 或 high。")
    warmup = _positive_integer(warmup_periods, "warmup_periods")
    measured = _positive_integer(measured_periods, "measured_periods")
    checkpoint = _positive_integer(
        checkpoint_interval_periods,
        "checkpoint_interval_periods",
    )
    if isinstance(session_index, bool) or not isinstance(session_index, int) or not 0 <= session_index < formal.FORMAL_SESSION_COUNT:
        raise ValueError("session_index must lie in [0, 999]. / session_index 必须在 [0, 999]。")
    total = warmup + measured
    if total >= formal.FORMAL_STABILITY_STREAK:
        raise ValueError(
            "Warm-up plus measured periods must stay below the convergence streak. / "
            "预热期加计时期必须小于收敛 streak。"
        )
    observed_environment = collect_environment() if environment is None else environment
    validate_execution_environment(
        observed_environment,
        allow_non_slurm=allow_non_slurm,
        total_period_budget=total,
    )
    outer_root, formal_root = prepare_isolated_roots(benchmark_root)

    family = formal.initialize_family(formal_root)
    plan = load_experiment_cell_plan(formal_root / formal.PLAN_RELATIVE_PATHS[cell])
    task = plan.tasks[session_index]
    status_path = _training_status_path(
        formal_root,
        cell,
        task.relative_artifact_directory,
    )

    warmup_wall_start = float(clock())
    warmup_result = formal.run_one_session(
        formal_root,
        cell=cell,
        session_index=session_index,
        checkpoint_interval_periods=checkpoint,
        invocation_period_budget=warmup,
        retry_failed=False,
    )
    warmup_wall_end = float(clock())
    if warmup_wall_end <= warmup_wall_start:
        raise RuntimeError("Warm-up wall timer must advance. / 预热 wall timer 必须前进。")
    _validate_outer_result(
        warmup_result,
        cell=cell,
        session_index=session_index,
        expected_ending_period=warmup,
    )
    warmup_status = load_training_status(
        status_path,
        expected_task=task,
        expected_config=plan.config,
    )
    _validate_training_attempt(
        warmup_status,
        expected_start_mode=FRESH_START,
        expected_start=0,
        expected_budget=warmup,
        expected_end=warmup,
        expected_attempt_number=1,
    )

    measured_wall_start = float(clock())
    measured_result = formal.run_one_session(
        formal_root,
        cell=cell,
        session_index=session_index,
        checkpoint_interval_periods=checkpoint,
        invocation_period_budget=measured,
        retry_failed=False,
    )
    measured_wall_end = float(clock())
    if measured_wall_end <= measured_wall_start:
        raise RuntimeError("Measured wall timer must advance. / 正式计时 wall timer 必须前进。")
    expected_end = warmup + measured
    _validate_outer_result(
        measured_result,
        cell=cell,
        session_index=session_index,
        expected_ending_period=expected_end,
    )
    measured_status = load_training_status(
        status_path,
        expected_task=task,
        expected_config=plan.config,
    )
    _validate_training_attempt(
        measured_status,
        expected_start_mode=RESUMED_START,
        expected_start=warmup,
        expected_budget=measured,
        expected_end=expected_end,
        expected_attempt_number=2,
    )
    _validate_resume_handoff(warmup_status, measured_status)

    warmup_attempt = warmup_status.attempt
    measured_attempt = measured_status.attempt
    measured_wall_elapsed = measured_wall_end - measured_wall_start
    measured_wall_rate = measured / measured_wall_elapsed
    slurm_fields = dict(observed_environment.slurm)
    slurm_verified = "SLURM_JOB_ID" in slurm_fields
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_mode": BENCHMARK_MODE,
        "execution_scope": (
            "slurm_compute_benchmark" if slurm_verified else "local_connection_smoke"
        ),
        "slurm_compute_node_verified": slurm_verified,
        "cluster_name": slurm_fields.get("SLURM_CLUSTER_NAME"),
        "cell": cell,
        "noise_standard_deviation": plan.config.parameters.noise_std,
        "session_index": session_index,
        "task_id": task.task_id,
        "task_sha256": task.task_sha256,
        "plan_sha256": plan.plan_sha256,
        "run_config_sha256": plan.run_config_sha256,
        "family_manifest_sha256": family["family_manifest_sha256"],
        "execution_source_sha256": family["execution_source_sha256"],
        "result_pipeline_source_sha256": family["result_pipeline_source_sha256"],
        "step36g_source_sha256": _normalized_file_sha256(Path(__file__)),
        "benchmark_root": str(outer_root),
        "isolated_formal_artifact_root": str(formal_root),
        "warmup_period_budget": warmup,
        "measured_period_budget": measured,
        "checkpoint_interval_periods": checkpoint,
        "warmup_start_mode": warmup_attempt.start_mode,
        "warmup_ending_verified_period": warmup_attempt.ending_verified_period,
        "warmup_training_elapsed_seconds": warmup_attempt.elapsed_seconds,
        "warmup_training_periods_per_second": warmup_attempt.periods_per_second,
        "warmup_end_to_end_elapsed_seconds": warmup_wall_end - warmup_wall_start,
        "measured_start_mode": measured_attempt.start_mode,
        "measured_starting_verified_period": measured_attempt.starting_training_period,
        "measured_ending_verified_period": measured_attempt.ending_verified_period,
        "measured_training_elapsed_seconds": measured_attempt.elapsed_seconds,
        "measured_training_periods_per_second": measured_attempt.periods_per_second,
        "measured_end_to_end_elapsed_seconds": measured_wall_elapsed,
        "measured_end_to_end_periods_per_second": measured_wall_rate,
        "linear_extrapolation_seconds_per_million_at_observed_rate": (
            1_000_000.0 / measured_attempt.periods_per_second
            if slurm_verified
            else None
        ),
        "outer_status": str(measured_result["status"]),
        "outer_phase": str(measured_result["phase"]),
        "outer_stop_reason": str(measured_result["stop_reason"]),
        "inner_stop_reason": measured_status.scientific_outcome.stop_reason,
        "measurement_rows_committed": int(measured_result["measurement_rows_committed"]),
        "peak_resident_set_megabytes": _peak_resident_set_megabytes(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": observed_environment,
        "economic_parameters_changed_for_benchmark": False,
        "convergence_observed": False,
        "research_result": False,
        "paper_results_ready": False,
        "total_formal_runtime_known": False,
    }
    report = ThroughputBenchmarkReport(
        **payload,
        report_sha256=_sha256_json(
            {
                **payload,
                "environment": asdict(observed_environment),
            }
        ),
    )
    validate_report(report)
    report_path = save_report(report, outer_root / REPORT_FILE_NAME)
    if load_report(report_path) != report:
        raise RuntimeError("Saved report did not round-trip. / 保存的 report 未能一致重读。")
    return report, report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one exact formal training cell on an allocated Slurm CPU. / "
            "在 Slurm 计算节点上测试一个正式训练 cell 的速度。"
        )
    )
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--cell", choices=("low", "high"), required=True)
    parser.add_argument("--warmup-periods", type=int, default=DEFAULT_WARMUP_PERIODS)
    parser.add_argument("--measured-periods", type=int, default=DEFAULT_MEASURED_PERIODS)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
    )
    parser.add_argument("--session-index", type=int, default=DEFAULT_SESSION_INDEX)
    parser.add_argument(
        "--allow-non-slurm",
        action="store_true",
        help=(
            "Allow at most 100 total periods for local code validation only. / "
            "只允许本地最多合计 100 期的代码验证。"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report, report_path = run_throughput_benchmark(
        benchmark_root=args.benchmark_root,
        cell=args.cell,
        warmup_periods=args.warmup_periods,
        measured_periods=args.measured_periods,
        checkpoint_interval_periods=args.checkpoint_interval,
        session_index=args.session_index,
        allow_non_slurm=args.allow_non_slurm,
    )
    summary = {
        "step": "36G",
        "cell": report.cell,
        "execution_scope": report.execution_scope,
        "measured_training_periods_per_second": report.measured_training_periods_per_second,
        "linear_extrapolation_seconds_per_million_at_observed_rate": (
            report.linear_extrapolation_seconds_per_million_at_observed_rate
        ),
        "report_path": str(report_path),
        "research_result": report.research_result,
        "paper_results_ready": report.paper_results_ready,
        "total_formal_runtime_known": report.total_formal_runtime_known,
    }
    print(_canonical_json(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
