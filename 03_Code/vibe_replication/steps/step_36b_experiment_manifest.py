"""Step 36B: build and save an auditable experiment-cell plan.

第 36B 步：建立并保存一个可审查的实验单元计划。

Run the small demonstration / 运行小型演示:
    py -3 -X utf8 steps/step_36b_experiment_manifest.py

What this step does / 本步骤做什么:
    * fixes every scientific setting before a long run starts;
      / 在长时间运行前锁定全部科学设定；
    * creates one deterministic job identity for every independent session;
      / 为每个独立 session 建立确定的任务身份；
    * assigns the seven Step-26 random streams without consuming an RNG;
      / 不消耗随机数，预先分配第 26 步的七条随机流；
    * saves the plan atomically and refuses silent overwrite or tampering.
      / 原子保存计划，并拒绝静默覆盖或篡改。

Strict boundary / 严格边界:
    A plan is NOT a completed experiment. This file trains no Q-learner,
    launches no SLURM job, and produces no paper result. Exact within-session
    checkpoint/resume is the next substep. / “计划”不等于“完成实验”。本文件
    不训练 Q-learner、不启动 SLURM，也不产生论文结果；session 内精确断点续跑
    是下一小步。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from numbers import Real
from pathlib import Path, PurePosixPath
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from src.source_manifests import (
    LOADED_EXECUTION_SOURCE_SHA256,
    LOADED_RESULT_PIPELINE_SOURCE_SHA256,
    LOADED_SOURCE_SCOPE_MANIFEST_SHA256,
    SOURCE_SCOPE_MANIFEST_VERSION,
)
from step_26_reproducible_random_streams import (
    MAX_UINT64_PLUS_ONE,
    SEED_DERIVATION_VERSION,
    SessionSeedManifest,
    build_session_seed_manifest,
)
from step_27_convergence_tracker import PAPER_UNCHANGED_PERIODS
from step_28_session_phases import PAPER_MEASUREMENT_PERIODS
from step_36a_one_session_result_row import (
    PRICE_GRID_ENCODING,
    _atomic_text_write,
)
from steps.step_34_mechanism_classifier import PAPER_PATHS_PER_SESSION
from steps.step_35a_converged_market_checkpoint import (
    LOADED_IMPLEMENTATION_TREE_SHA256,
)
from steps.step_35e_cell_shock_calibration import (
    PAPER_SESSIONS_PER_EXPERIMENT_CELL,
)
from steps.step_35f_paired_response_and_classification import (
    STEP35F_PROTOCOL_VERSION,
)


PLAN_SCHEMA_VERSION = "step36b-experiment-cell-plan-v3-formal-runner"
TASK_SCHEMA_VERSION = "step36b-session-task-v2-layered-source-scope"
DEBUG_MODE = "debug"
PAPER_MODE = "paper"
ALLOWED_MODES = (DEBUG_MODE, PAPER_MODE)

# These labels make implementation choices visible in every saved plan.
# / 这些标签让每个已保存计划都明确披露实现选择。
INITIAL_Q_PROTOCOL = "paper-uniform-opponent-zero-noise-v1"
INITIAL_STATE_PROTOCOL = "uniform-price-value-value-grid-v1"
MARKET_MAKER_PREHISTORY_PROTOCOL = "balanced-nash-consistent-v1"
LEARNING_PROTOCOL = "steps26-28-tabular-q-learning-v1"
EXACT_TRAINING_RESUME_GRANULARITY = "between-completed-training-periods-v1"


@dataclass(frozen=True)
class ExperimentCellConfig:
    """Scientific settings shared by every session in one experiment cell.

    一个实验单元中所有 session 共同使用的科学设定。
    """

    mode: str
    experiment_cell_key: str
    parameters: PaperParameters
    experiment_seed: int
    irf_experiment_seed: int
    session_count: int
    convergence_periods_required: int
    measurement_periods_required: int
    irf_paths_per_session: int
    mechanism_analysis_enabled: bool = True
    initial_q_protocol: str = INITIAL_Q_PROTOCOL
    initial_state_protocol: str = INITIAL_STATE_PROTOCOL
    market_maker_prehistory_protocol: str = MARKET_MAKER_PREHISTORY_PROTOCOL
    price_grid_encoding: str = PRICE_GRID_ENCODING
    learning_protocol: str = LEARNING_PROTOCOL
    irf_protocol: str = STEP35F_PROTOCOL_VERSION
    seed_derivation_version: str = SEED_DERIVATION_VERSION


@dataclass(frozen=True)
class ExperimentExecutionPolicy:
    """Operational limits kept separate from the scientific identity.

    与科学身份分开保存的运行限制。

    A debug cap may stop work early, but it can only create an incomplete job.
    It can never manufacture a result. / 调试上限可以提前停止计算，但只能产生
    “未完成”任务，绝不能把它伪装成结果。
    """

    maximum_training_periods: int | None = None
    resume_granularity: str = EXACT_TRAINING_RESUME_GRANULARITY
    within_session_checkpointing_available: bool = True
    persisted_post_convergence_bundle_available: bool = False
    formal_session_runner_available: bool = False
    hpc_array_dispatch_available: bool = False


@dataclass(frozen=True)
class SessionTaskManifest:
    """One immutable, replayable job specification. / 一个不可修改、可重放的任务说明。"""

    schema_version: str
    task_id: str
    session_index: int
    run_config_sha256: str
    implementation_tree_sha256: str
    source_scope_manifest_version: str
    source_scope_manifest_sha256: str
    execution_source_sha256: str
    result_pipeline_source_sha256: str
    seed_manifest: SessionSeedManifest
    relative_artifact_directory: str
    task_sha256: str


@dataclass(frozen=True)
class ExperimentCellPlan:
    """Saved plan plus honest readiness flags. / 已保存计划及诚实的就绪标记。"""

    schema_version: str
    config: ExperimentCellConfig
    execution_policy: ExperimentExecutionPolicy
    experiment_cell_sha256: str
    run_config_sha256: str
    implementation_tree_sha256: str
    source_scope_manifest_version: str
    source_scope_manifest_sha256: str
    execution_source_sha256: str
    result_pipeline_source_sha256: str
    tasks: tuple[SessionTaskManifest, ...]
    task_count: int
    unique_session_seed_count: int
    unique_child_seed_count: int
    tasks_sha256: str
    formal_mode_requested: bool
    paper_scale_counts_requested: bool
    uncapped_training_requested: bool
    formal_session_runner_connected: bool
    within_session_checkpointing_available: bool
    persisted_post_convergence_bundle_available: bool
    hpc_array_dispatch_available: bool
    research_result: bool
    paper_results_ready: bool
    plan_sha256: str


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    """Encode deterministic JSON. / 编码为确定的 JSON。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _json_ready(value: object) -> object:
    """Convert dataclasses and tuples to plain JSON data. / 转成普通 JSON 数据。"""

    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return json.loads(_canonical_json(value))


def _sha256_json(value: object) -> str:
    """Hash canonical JSON. / 对规范 JSON 计算 SHA-256 指纹。"""

    return sha256(_canonical_json(_json_ready(value)).encode("utf-8")).hexdigest()


def _positive_integer(number: int, label: str, maximum: int) -> int:
    """Validate a non-Boolean positive integer. / 检查非布尔正整数。"""

    if isinstance(number, bool) or not isinstance(number, int):
        raise TypeError(f"{label} must be an integer. / {label} 必须是整数。")
    if not 1 <= number <= maximum:
        raise ValueError(
            f"{label} must lie in [1, {maximum}]. / {label} 必须位于 [1, {maximum}]。"
        )
    return number


def _uint64(number: int, label: str) -> int:
    """Validate one unsigned 64-bit seed. / 检查一个 64 位无符号种子。"""

    if isinstance(number, bool) or not isinstance(number, int):
        raise TypeError(f"{label} must be an integer. / {label} 必须是整数。")
    if not 0 <= number < MAX_UINT64_PLUS_ONE:
        raise ValueError(f"{label} must lie in [0, 2^64). / {label} 必须位于 [0, 2^64)。")
    return number


def _validate_parameters(parameters: PaperParameters) -> None:
    """Reject hidden NaN/Inf and type mistakes missed by simple comparisons.

    拒绝简单大小比较可能漏掉的 NaN、Inf 和类型错误。
    """

    if not isinstance(parameters, PaperParameters):
        raise TypeError("parameters must be PaperParameters. / parameters 类型错误。")

    integer_names = (
        "num_speculators",
        "num_value_points",
        "num_action_points",
        "num_price_points",
        "market_maker_window",
    )
    for name in integer_names:
        value = getattr(parameters, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"parameters.{name} must be an integer. / 参数 {name} 必须是整数。")

    for name, value in asdict(parameters).items():
        if name in integer_names:
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"parameters.{name} must be real. / 参数 {name} 必须是实数。")
        if not isfinite(float(value)):
            raise ValueError(f"parameters.{name} must be finite. / 参数 {name} 必须是有限数。")

    # PaperParameters already checks the main economic restrictions. These two
    # fields need explicit guards here. / PaperParameters 已检查主要经济限制；
    # 这两个字段还需要在此明确检查。
    if parameters.exploration_decay <= 0:
        raise ValueError("exploration_decay must be positive. / exploration_decay 必须为正。")
    if parameters.grid_widening < 0:
        raise ValueError("grid_widening cannot be negative. / grid_widening 不能为负。")


def _validate_config_and_policy(
    config: ExperimentCellConfig,
    policy: ExperimentExecutionPolicy,
) -> None:
    """Validate inputs before making any task. / 建立任务前检查全部输入。"""

    if not isinstance(config, ExperimentCellConfig):
        raise TypeError("config has the wrong type. / config 类型错误。")
    if not isinstance(policy, ExperimentExecutionPolicy):
        raise TypeError("execution_policy has the wrong type. / execution_policy 类型错误。")
    if config.mode not in ALLOWED_MODES:
        raise ValueError(f"mode must be one of {ALLOWED_MODES}. / mode 必须是 {ALLOWED_MODES} 之一。")
    if not isinstance(config.experiment_cell_key, str) or not config.experiment_cell_key:
        raise ValueError("experiment_cell_key cannot be empty. / 实验单元标签不能为空。")
    if config.experiment_cell_key.strip() != config.experiment_cell_key:
        raise ValueError("experiment_cell_key cannot have outer spaces. / 实验单元标签首尾不能有空格。")

    _validate_parameters(config.parameters)
    _uint64(config.experiment_seed, "experiment_seed")
    _uint64(config.irf_experiment_seed, "irf_experiment_seed")
    _positive_integer(
        config.session_count,
        "session_count",
        PAPER_SESSIONS_PER_EXPERIMENT_CELL,
    )
    _positive_integer(
        config.convergence_periods_required,
        "convergence_periods_required",
        PAPER_UNCHANGED_PERIODS,
    )
    _positive_integer(
        config.measurement_periods_required,
        "measurement_periods_required",
        PAPER_MEASUREMENT_PERIODS,
    )
    _positive_integer(
        config.irf_paths_per_session,
        "irf_paths_per_session",
        PAPER_PATHS_PER_SESSION,
    )
    if not isinstance(config.mechanism_analysis_enabled, bool):
        raise TypeError("mechanism_analysis_enabled must be bool. / 机制分析开关必须是 bool。")
    if config.parameters.num_speculators != 2:
        raise ValueError(
            "The verified session engine currently requires I=2; I-comparative "
            "statics are not connected yet. / 当前已验证 session 引擎只支持 I=2；"
            "I 的比较静态尚未接通。"
        )

    expected_protocols = {
        "initial_q_protocol": INITIAL_Q_PROTOCOL,
        "initial_state_protocol": INITIAL_STATE_PROTOCOL,
        "market_maker_prehistory_protocol": MARKET_MAKER_PREHISTORY_PROTOCOL,
        "price_grid_encoding": PRICE_GRID_ENCODING,
        "learning_protocol": LEARNING_PROTOCOL,
        "irf_protocol": STEP35F_PROTOCOL_VERSION,
        "seed_derivation_version": SEED_DERIVATION_VERSION,
    }
    for name, expected in expected_protocols.items():
        if getattr(config, name) != expected:
            raise ValueError(
                f"{name} must equal {expected!r}. / {name} 必须等于 {expected!r}。"
            )

    if policy.maximum_training_periods is not None:
        _positive_integer(
            policy.maximum_training_periods,
            "maximum_training_periods",
            2**63 - 1,
        )
    if policy.resume_granularity != EXACT_TRAINING_RESUME_GRANULARITY:
        raise ValueError("Resume must occur between fully completed training periods. / 必须在完整训练期之间恢复。")
    for name in (
        "within_session_checkpointing_available",
        "persisted_post_convergence_bundle_available",
        "formal_session_runner_available",
        "hpc_array_dispatch_available",
    ):
        if not isinstance(getattr(policy, name), bool):
            raise TypeError(f"{name} must be bool. / {name} 必须是 bool。")
    if not policy.within_session_checkpointing_available:
        raise ValueError("Step 36C exact training checkpointing must remain enabled. / 必须启用第 36C 步精确训练 checkpoint。")
    if (
        policy.formal_session_runner_available
        and not policy.persisted_post_convergence_bundle_available
    ):
        raise ValueError(
            "The formal runner requires persisted complete measurement evidence. / "
            "正式 runner 要求保存完整测量 evidence。"
        )
    if (
        policy.hpc_array_dispatch_available
        and not policy.formal_session_runner_available
    ):
        raise ValueError(
            "HPC array selection requires the formal session runner. / "
            "HPC array 选择要求正式 session runner。"
        )

    if config.mode == PAPER_MODE:
        paper_counts = (
            config.session_count == PAPER_SESSIONS_PER_EXPERIMENT_CELL
            and config.convergence_periods_required == PAPER_UNCHANGED_PERIODS
            and config.measurement_periods_required == PAPER_MEASUREMENT_PERIODS
            and config.irf_paths_per_session == PAPER_PATHS_PER_SESSION
            and config.mechanism_analysis_enabled
        )
        if not paper_counts:
            raise ValueError("Paper mode requires every exact paper-scale count. / 论文模式要求全部精确论文规模计数。")
        if policy.maximum_training_periods is not None:
            raise ValueError(
                "Paper mode cannot cap training: a timeout must remain incomplete. / "
                "论文模式不能限制训练期数；超时任务必须保持未完成。"
            )


def _scientific_cell_payload(config: ExperimentCellConfig) -> dict[str, object]:
    """Scientific identity excluding random roots. / 不含随机根种子的科学身份。"""

    payload = asdict(config)
    payload.pop("experiment_seed")
    payload.pop("irf_experiment_seed")
    return payload


def _task_without_checksum(task: SessionTaskManifest) -> dict[str, object]:
    payload = asdict(task)
    payload.pop("task_sha256")
    return payload


def _plan_without_checksum(plan: ExperimentCellPlan) -> dict[str, object]:
    payload = asdict(plan)
    payload.pop("plan_sha256")
    return payload


def _build_session_task(
    *,
    config: ExperimentCellConfig,
    run_config_sha256: str,
    session_index: int,
) -> SessionTaskManifest:
    """Build one job without drawing any random number. / 不抽随机数地建立一个任务。"""

    seed_manifest = build_session_seed_manifest(
        config.experiment_seed,
        config.experiment_cell_key,
        session_index,
    )
    identity_payload = {
        "schema_version": TASK_SCHEMA_VERSION,
        "run_config_sha256": run_config_sha256,
        "implementation_tree_sha256": LOADED_IMPLEMENTATION_TREE_SHA256,
        "source_scope_manifest_version": SOURCE_SCOPE_MANIFEST_VERSION,
        "source_scope_manifest_sha256": LOADED_SOURCE_SCOPE_MANIFEST_SHA256,
        "execution_source_sha256": LOADED_EXECUTION_SOURCE_SHA256,
        "result_pipeline_source_sha256": LOADED_RESULT_PIPELINE_SOURCE_SHA256,
        "session_index": session_index,
        "session_seed": seed_manifest.session_seed,
    }
    identity = _sha256_json(identity_payload)
    task_without_checksum = SessionTaskManifest(
        schema_version=TASK_SCHEMA_VERSION,
        task_id=f"session-{session_index:04d}-{identity[:16]}",
        session_index=session_index,
        run_config_sha256=run_config_sha256,
        implementation_tree_sha256=LOADED_IMPLEMENTATION_TREE_SHA256,
        source_scope_manifest_version=SOURCE_SCOPE_MANIFEST_VERSION,
        source_scope_manifest_sha256=(
            LOADED_SOURCE_SCOPE_MANIFEST_SHA256
        ),
        execution_source_sha256=LOADED_EXECUTION_SOURCE_SHA256,
        result_pipeline_source_sha256=(
            LOADED_RESULT_PIPELINE_SOURCE_SHA256
        ),
        seed_manifest=seed_manifest,
        relative_artifact_directory=(
            PurePosixPath("sessions") / f"session_{session_index:04d}_{identity[:12]}"
        ).as_posix(),
        task_sha256="",
    )
    checksum = _sha256_json(_task_without_checksum(task_without_checksum))
    return SessionTaskManifest(
        **{
            **_task_without_checksum(task_without_checksum),
            "seed_manifest": seed_manifest,
            "task_sha256": checksum,
        }
    )


def validate_session_task_manifest(task: SessionTaskManifest) -> None:
    """Validate one task independently before loading its checkpoint.

    在读取该任务的 checkpoint 前，独立检查一个任务。
    """

    if not isinstance(task, SessionTaskManifest):
        raise TypeError("task must be SessionTaskManifest. / task 类型错误。")
    if task.schema_version != TASK_SCHEMA_VERSION:
        raise ValueError("Task schema is unsupported. / 任务格式不支持。")
    for value, label in (
        (task.run_config_sha256, "run_config_sha256"),
        (task.implementation_tree_sha256, "implementation_tree_sha256"),
        (task.source_scope_manifest_sha256, "source_scope_manifest_sha256"),
        (task.execution_source_sha256, "execution_source_sha256"),
        (task.result_pipeline_source_sha256, "result_pipeline_source_sha256"),
        (task.task_sha256, "task_sha256"),
    ):
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{label} is not a SHA-256 digest. / {label} 不是 SHA-256 摘要。")
    if (
        task.implementation_tree_sha256 != LOADED_IMPLEMENTATION_TREE_SHA256
        or task.source_scope_manifest_version
        != SOURCE_SCOPE_MANIFEST_VERSION
        or task.source_scope_manifest_sha256
        != LOADED_SOURCE_SCOPE_MANIFEST_SHA256
        or task.execution_source_sha256
        != LOADED_EXECUTION_SOURCE_SHA256
        or task.result_pipeline_source_sha256
        != LOADED_RESULT_PIPELINE_SOURCE_SHA256
    ):
        raise RuntimeError("Task belongs to a different checked source build. / 任务属于不同的已核对源码版本。")
    if isinstance(task.session_index, bool) or not isinstance(task.session_index, int) or task.session_index < 0:
        raise ValueError("Task session_index is invalid. / 任务 session_index 无效。")
    expected_manifest = build_session_seed_manifest(
        task.seed_manifest.experiment_seed,
        task.seed_manifest.experiment_cell_key,
        task.session_index,
    )
    if task.seed_manifest != expected_manifest:
        raise ValueError("Task seed manifest is inconsistent. / 任务种子清单不一致。")
    identity = _sha256_json(
        {
            "schema_version": TASK_SCHEMA_VERSION,
            "run_config_sha256": task.run_config_sha256,
            "implementation_tree_sha256": task.implementation_tree_sha256,
            "source_scope_manifest_version": task.source_scope_manifest_version,
            "source_scope_manifest_sha256": task.source_scope_manifest_sha256,
            "execution_source_sha256": task.execution_source_sha256,
            "result_pipeline_source_sha256": task.result_pipeline_source_sha256,
            "session_index": task.session_index,
            "session_seed": task.seed_manifest.session_seed,
        }
    )
    expected_task_id = f"session-{task.session_index:04d}-{identity[:16]}"
    expected_directory = (
        PurePosixPath("sessions")
        / f"session_{task.session_index:04d}_{identity[:12]}"
    ).as_posix()
    if task.task_id != expected_task_id or task.relative_artifact_directory != expected_directory:
        raise ValueError("Task identity or relative directory is inconsistent. / 任务身份或相对目录不一致。")
    if _sha256_json(_task_without_checksum(task)) != task.task_sha256:
        raise ValueError("Task checksum failed. / 任务校验失败。")


def validate_session_task_for_config(
    task: SessionTaskManifest,
    config: ExperimentCellConfig,
) -> None:
    """Prove that one task was generated from the expected cell config.

    证明某个任务确实由预期的实验单元配置生成。

    A task stores a compact config hash rather than duplicating every setting.
    This function recomputes that hash and the whole deterministic task record.
    / task 只保存精简的配置哈希，不重复保存每个参数；本函数会重新计算该哈希
    以及完整的确定性任务记录。
    """

    validate_session_task_manifest(task)
    # The execution policy is deliberately absent from a task's scientific
    # identity. A default valid policy lets us validate only the config here.
    # / 运行策略故意不属于 task 的科学身份；这里用默认合法策略只验证 config。
    _validate_config_and_policy(config, ExperimentExecutionPolicy())
    if task.session_index >= config.session_count:
        raise ValueError(
            "Task index lies outside the expected experiment cell. / "
            "任务编号超出预期实验单元。"
        )
    expected_run_config_sha256 = _sha256_json(
        {
            "config": asdict(config),
            "implementation_tree_sha256": LOADED_IMPLEMENTATION_TREE_SHA256,
        }
    )
    expected_task = _build_session_task(
        config=config,
        run_config_sha256=expected_run_config_sha256,
        session_index=task.session_index,
    )
    if task != expected_task:
        raise ValueError(
            "Task does not belong to the expected experiment config. / "
            "任务不属于预期实验配置。"
        )


def build_experiment_cell_plan(
    config: ExperimentCellConfig,
    execution_policy: ExperimentExecutionPolicy | None = None,
) -> ExperimentCellPlan:
    """Build the canonical ordered task plan for one cell.

    为一个实验单元建立规范、按顺序排列的任务计划。
    """

    if execution_policy is None:
        execution_policy = ExperimentExecutionPolicy()
    _validate_config_and_policy(config, execution_policy)

    experiment_cell_sha256 = _sha256_json(_scientific_cell_payload(config))
    run_config_sha256 = _sha256_json(
        {
            "config": asdict(config),
            "implementation_tree_sha256": LOADED_IMPLEMENTATION_TREE_SHA256,
        }
    )
    tasks = tuple(
        _build_session_task(
            config=config,
            run_config_sha256=run_config_sha256,
            session_index=session_index,
        )
        for session_index in range(config.session_count)
    )
    for task in tasks:
        validate_session_task_manifest(task)
    session_seeds = {task.seed_manifest.session_seed for task in tasks}
    child_seeds = {
        child_seed
        for task in tasks
        for child_seed in task.seed_manifest.child_seeds()
    }
    if len(session_seeds) != len(tasks):
        raise RuntimeError("Session-seed collision detected. / 检测到 session 种子碰撞。")
    if len(child_seeds) != 7 * len(tasks):
        raise RuntimeError("Child-stream seed collision detected. / 检测到子随机流种子碰撞。")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise RuntimeError("Task-identity collision detected. / 检测到任务身份碰撞。")
    tasks_sha256 = _sha256_json([task.task_sha256 for task in tasks])
    paper_counts = (
        config.session_count == PAPER_SESSIONS_PER_EXPERIMENT_CELL
        and config.convergence_periods_required == PAPER_UNCHANGED_PERIODS
        and config.measurement_periods_required == PAPER_MEASUREMENT_PERIODS
        and config.irf_paths_per_session == PAPER_PATHS_PER_SESSION
        and config.mechanism_analysis_enabled
    )

    without_plan_checksum = ExperimentCellPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        config=config,
        execution_policy=execution_policy,
        experiment_cell_sha256=experiment_cell_sha256,
        run_config_sha256=run_config_sha256,
        implementation_tree_sha256=LOADED_IMPLEMENTATION_TREE_SHA256,
        source_scope_manifest_version=SOURCE_SCOPE_MANIFEST_VERSION,
        source_scope_manifest_sha256=(
            LOADED_SOURCE_SCOPE_MANIFEST_SHA256
        ),
        execution_source_sha256=LOADED_EXECUTION_SOURCE_SHA256,
        result_pipeline_source_sha256=(
            LOADED_RESULT_PIPELINE_SOURCE_SHA256
        ),
        tasks=tasks,
        task_count=len(tasks),
        unique_session_seed_count=len(session_seeds),
        unique_child_seed_count=len(child_seeds),
        tasks_sha256=tasks_sha256,
        formal_mode_requested=(config.mode == PAPER_MODE),
        paper_scale_counts_requested=paper_counts,
        uncapped_training_requested=(execution_policy.maximum_training_periods is None),
        # This flag is an explicit requested capability. Step 36F requires it
        # before publishing a post-convergence bridge; older Step 36E-only
        # plans may honestly leave it false. / 这个标记是明确请求的能力；
        # Step 36F 发布收敛后 bridge 前要求它为 true，旧的 Step 36E-only
        # 计划可以如实保持 false。
        formal_session_runner_connected=(
            execution_policy.formal_session_runner_available
        ),
        within_session_checkpointing_available=(
            execution_policy.within_session_checkpointing_available
        ),
        persisted_post_convergence_bundle_available=(
            execution_policy.persisted_post_convergence_bundle_available
        ),
        hpc_array_dispatch_available=(
            execution_policy.hpc_array_dispatch_available
        ),
        # A plan is metadata, never empirical evidence. / 计划只是元数据，绝不是实证结果。
        research_result=False,
        paper_results_ready=False,
        plan_sha256="",
    )
    plan_sha256 = _sha256_json(_plan_without_checksum(without_plan_checksum))
    return ExperimentCellPlan(
        **{
            **_plan_without_checksum(without_plan_checksum),
            "config": config,
            "execution_policy": execution_policy,
            "tasks": tasks,
            "plan_sha256": plan_sha256,
        }
    )


def validate_experiment_cell_plan(plan: ExperimentCellPlan) -> None:
    """Rebuild a plan and require byte-level logical equality.

    重新建立计划，并要求逻辑内容逐项完全相同。
    """

    if not isinstance(plan, ExperimentCellPlan):
        raise TypeError("plan has the wrong type. / plan 类型错误。")
    expected = build_experiment_cell_plan(plan.config, plan.execution_policy)
    if _json_ready(plan) != _json_ready(expected):
        raise ValueError("Experiment plan failed deterministic validation. / 实验计划未通过确定性验证。")


def select_tasks_for_shard(
    plan: ExperimentCellPlan,
    *,
    shard_count: int,
    shard_index: int,
) -> tuple[SessionTaskManifest, ...]:
    """Select a deterministic slice; this does not launch an HPC job.

    确定性选择一个任务切片；本函数不会启动 HPC 任务。
    """

    validate_experiment_cell_plan(plan)
    checked_count = _positive_integer(shard_count, "shard_count", plan.task_count)
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise TypeError("shard_index must be an integer. / shard_index 必须是整数。")
    if not 0 <= shard_index < checked_count:
        raise ValueError("shard_index is outside the shard range. / shard_index 超出切片范围。")
    return plan.tasks[shard_index::checked_count]


def _plan_json_dictionary(plan: ExperimentCellPlan) -> dict[str, object]:
    ready = _json_ready(plan)
    if not isinstance(ready, dict):
        raise RuntimeError("Internal plan conversion failed. / 内部计划转换失败。")
    return ready


def save_experiment_cell_plan(plan: ExperimentCellPlan, path: Path) -> Path:
    """Atomically save once; identical replay is a no-op, conflict is rejected.

    原子保存一次；完全相同的重放不重复写，冲突则拒绝。
    """

    validate_experiment_cell_plan(plan)
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path. / path 必须是 pathlib.Path。")
    expected_dictionary = _plan_json_dictionary(plan)
    if path.exists():
        existing = load_experiment_cell_plan(path)
        if _plan_json_dictionary(existing) == expected_dictionary:
            return path
        raise FileExistsError(
            "A different valid plan already exists at this path. / 此路径已经存在另一个有效计划。"
        )
    text = _canonical_json(expected_dictionary, indent=2) + "\n"
    _atomic_text_write(path, text)
    return path


def _config_from_dictionary(dictionary: dict[str, object]) -> ExperimentCellConfig:
    data = dict(dictionary)
    parameters = data.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Saved parameters are malformed. / 已保存参数格式错误。")
    data["parameters"] = PaperParameters(**parameters)
    return ExperimentCellConfig(**data)


def load_experiment_cell_plan(path: Path) -> ExperimentCellPlan:
    """Load, rebuild, and reject tampered or stale plans.

    读取并重建计划；拒绝被篡改或对应旧源码的计划。
    """

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path. / path 必须是 pathlib.Path。")
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read a complete plan JSON file. / 无法读取完整计划 JSON。") from error
    if not isinstance(saved, dict):
        raise ValueError("Saved plan must be a JSON object. / 已保存计划必须是 JSON 对象。")

    try:
        config_dictionary = saved["config"]
        policy_dictionary = saved["execution_policy"]
        if not isinstance(config_dictionary, dict) or not isinstance(policy_dictionary, dict):
            raise TypeError
        config = _config_from_dictionary(config_dictionary)
        policy = ExperimentExecutionPolicy(**policy_dictionary)
        expected = build_experiment_cell_plan(config, policy)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Saved plan schema or values are invalid. / 已保存计划结构或数值无效。") from error

    if saved != _plan_json_dictionary(expected):
        raise ValueError(
            "Saved plan was tampered with or belongs to a different source build. / "
            "已保存计划被篡改，或属于不同的源码版本。"
        )
    return expected


def build_debug_example_plan() -> ExperimentCellPlan:
    """Make a tiny plan for validation, not research. / 建立只用于验证的小型计划。"""

    config = ExperimentCellConfig(
        mode=DEBUG_MODE,
        experiment_cell_key="step36b-debug-low-noise",
        parameters=PaperParameters(noise_std=0.1),
        experiment_seed=20260829,
        irf_experiment_seed=20260830,
        session_count=3,
        convergence_periods_required=5,
        measurement_periods_required=200,
        irf_paths_per_session=100,
    )
    policy = ExperimentExecutionPolicy(maximum_training_periods=5)
    return build_experiment_cell_plan(config, policy)


def main() -> None:
    """Save and reload the three-task demonstration. / 保存并重读三个任务的演示。"""

    plan = build_debug_example_plan()
    # Include the checked source/config identity in the demo filename. If code
    # changes later, the old audit artifact is preserved instead of silently
    # overwritten. / 演示文件名包含已核对的源码/配置身份；以后代码改变时，旧的
    # 审计文件会被保留，而不是被静默覆盖。
    output_path = (
        PROJECT_ROOT
        / "results"
        / "step36b_debug_plan"
        / f"experiment_plan_{plan.run_config_sha256[:12]}.json"
    )
    save_experiment_cell_plan(plan, output_path)
    replay = load_experiment_cell_plan(output_path)
    assert replay == plan

    print("Step 36B: experiment manifest / 第 36B 步：实验任务清单")
    print(f"Tasks planned / 已规划任务: {plan.task_count}")
    print(f"Unique session seeds / 唯一 session 种子: {plan.unique_session_seed_count}")
    print(f"Unique child streams / 唯一子随机流: {plan.unique_child_seed_count}")
    print(f"Cell fingerprint / 实验单元指纹: {plan.experiment_cell_sha256}")
    print(f"Run fingerprint / 本次运行指纹: {plan.run_config_sha256}")
    for task in plan.tasks:
        print(f"  {task.task_id} -> {task.relative_artifact_directory}")
    print(f"Saved plan / 已保存计划: {output_path}")
    print(
        "Boundary / 边界: zero sessions were trained; research_result=false. "
        "/ 尚未训练任何 session；research_result=false。"
    )


if __name__ == "__main__":
    main()
