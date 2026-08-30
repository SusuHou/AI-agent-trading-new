"""Versioned source scopes for resumable formal experiments.

正式可续跑实验所使用的、带版本源码范围。

Why explicit lists / 为什么使用明确清单:
    A training checkpoint must become stale when market logic changes, but it
    must not become stale merely because a plotting or orchestration file was
    added later.  These lists freeze that boundary before formal execution.
    / 市场逻辑改变时，训练 checkpoint 必须失效；但以后新增绘图或调度文件时，
    checkpoint 不应无故失效。正式运行前用这些清单固定边界。
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_SCOPE_MANIFEST_VERSION = "formal-session-source-scope-v1"
SOURCE_SCOPE_MANIFEST_FILE = "src/source_manifests.py"

# Exact recursive local-import closure of Step 28, the market/session engine.
# / Step 28 市场/session 引擎的精确本地 import 闭包。
EXECUTION_SOURCE_FILES = (
    "src/parameters.py",
    "src/step01_value_grid.py",
    "steps/step_03_total_order_flow.py",
    "steps/step_04_information_insensitive_investors.py",
    "steps/step_05_speculator_profit.py",
    "steps/step_08_nash_benchmark.py",
    "steps/step_09_cartel_benchmark.py",
    "steps/step_10_fixed_point_solver.py",
    "steps/step_12_action_grid.py",
    "steps/step_13_price_grid.py",
    "steps/step_14_state_representation.py",
    "steps/step_15_initial_state.py",
    "steps/step_16_initial_q_table.py",
    "steps/step_17_q_value_meaning.py",
    "steps/step_18_epsilon_greedy_action.py",
    "steps/step_19_value_specific_epsilon.py",
    "steps/step_20_q_learning_update.py",
    "steps/step_21_two_independent_q_traders.py",
    "steps/step_22_market_maker_rolling_history.py",
    "steps/step_23_market_maker_ols.py",
    "steps/step_24_adaptive_market_maker_price.py",
    "steps/step_24b_fast_rolling_ols.py",
    "steps/step_24c_initial_market_maker_history.py",
    "steps/step_25_one_market_period.py",
    "steps/step_26_reproducible_random_streams.py",
    "steps/step_27_convergence_tracker.py",
    "steps/step_28_session_phases.py",
)

# Files that turn one session into checkpoints, measurements, and mechanism
# evidence. They may change without changing the 27-file market transition
# closure, but a complete session artifact must still bind them. / 这些文件把
# 单个 session 变成 checkpoint、测量与机制证据；它们不一定改变 27 文件的市场
# 转移闭包，但完整 session artifact 仍必须绑定它们。
RESULT_PIPELINE_SOURCE_FILES = (
    "steps/step_11_benchmark_profits.py",
    "steps/step_29_matched_path_collusion_profitability.py",
    "steps/step_30_trading_intensity.py",
    "steps/step_31_price_informativeness.py",
    "steps/step_32_market_liquidity.py",
    "steps/step_33_mispricing.py",
    "steps/step_34_mechanism_classifier.py",
    "steps/step_35a_converged_market_checkpoint.py",
    "steps/step_35b_paired_irf_path.py",
    "steps/step_35c_irf_long_run_baseline.py",
    "steps/step_35d_unshocked_t3_calibration_paths.py",
    "steps/step_35e_cell_shock_calibration.py",
    "steps/step_35f_paired_response_and_classification.py",
    "steps/step_36a_one_session_result_row.py",
    "steps/step_36b_experiment_manifest.py",
    "steps/step_36c_exact_training_resume.py",
    "steps/step_36d_single_session_training_runner.py",
    "steps/step_36e_complete_measurement_runner.py",
    "steps/step_36f_persisted_calibration_bridge.py",
)


def _validated_paths(relative_names: tuple[str, ...]) -> tuple[Path, ...]:
    """Resolve one sorted, duplicate-free, workspace-contained list.

    解析一份已排序、无重复且不越出项目目录的清单。
    """

    if tuple(sorted(relative_names)) != relative_names:
        raise RuntimeError("Source manifest must be sorted. / 源码清单必须排序。")
    if len(set(relative_names)) != len(relative_names):
        raise RuntimeError("Source manifest contains duplicates. / 源码清单含重复项。")
    root = PROJECT_ROOT.resolve()
    paths: list[Path] = []
    for relative_name in relative_names:
        relative = PurePosixPath(relative_name)
        if relative.is_absolute() or any(
            part in ("", ".", "..") for part in relative.parts
        ):
            raise RuntimeError("Source manifest path is unsafe. / 源码清单路径不安全。")
        path = root.joinpath(*relative.parts).resolve()
        if root not in path.parents or not path.is_file():
            raise RuntimeError(
                f"Source manifest file is missing or outside the project: {relative_name}. "
                f"/ 源码清单文件丢失或越界：{relative_name}。"
            )
        paths.append(path)
    return tuple(paths)


def _source_digest(
    relative_names: tuple[str, ...],
    *,
    domain: bytes,
) -> str:
    """Hash normalized source bytes and their explicit names. / 哈希规范源码与明确文件名。"""

    digest = sha256(domain)
    digest.update(SOURCE_SCOPE_MANIFEST_VERSION.encode("utf-8"))
    for relative_name, path in zip(
        relative_names,
        _validated_paths(relative_names),
        strict=True,
    ):
        name = relative_name.encode("utf-8")
        source = path.read_bytes().replace(b"\r\n", b"\n")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(source).to_bytes(8, "big"))
        digest.update(source)
    return digest.hexdigest()


LOADED_EXECUTION_SOURCE_SHA256 = _source_digest(
    EXECUTION_SOURCE_FILES,
    domain=b"vibe-replication.formal-execution-sources.v1\0",
)
LOADED_RESULT_PIPELINE_SOURCE_SHA256 = _source_digest(
    RESULT_PIPELINE_SOURCE_FILES,
    domain=b"vibe-replication.formal-result-pipeline-sources.v1\0",
)
LOADED_SOURCE_SCOPE_MANIFEST_SHA256 = _source_digest(
    (SOURCE_SCOPE_MANIFEST_FILE,),
    domain=b"vibe-replication.source-scope-manifest-file.v1\0",
)

_combined = sha256(b"vibe-replication.formal-session-combined-sources.v1\0")
_combined.update(SOURCE_SCOPE_MANIFEST_VERSION.encode("utf-8"))
_combined.update(bytes.fromhex(LOADED_SOURCE_SCOPE_MANIFEST_SHA256))
_combined.update(bytes.fromhex(LOADED_EXECUTION_SOURCE_SHA256))
_combined.update(bytes.fromhex(LOADED_RESULT_PIPELINE_SOURCE_SHA256))
LOADED_COMBINED_SESSION_SOURCE_SHA256 = _combined.hexdigest()

