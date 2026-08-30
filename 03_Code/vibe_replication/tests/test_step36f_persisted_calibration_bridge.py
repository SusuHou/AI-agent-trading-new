"""Focused tests for the persisted Step-36F calibration bridge.

第 36F 步持久化校准桥的专项测试。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = PROJECT_ROOT / "steps"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from step_25_one_market_period import build_paper_inputs
from step_26_reproducible_random_streams import build_randomized_paper_session
from step_28_session_phases import SessionPhaseController
from steps.step_36b_experiment_manifest import (
    DEBUG_MODE,
    ExperimentCellConfig,
    ExperimentExecutionPolicy,
    build_experiment_cell_plan,
)
import steps.step_36d_single_session_training_runner as step36d
import steps.step_36e_complete_measurement_runner as step36e
import steps.step_36f_persisted_calibration_bridge as step36f


TEST_ARTIFACT_PARENT = PROJECT_ROOT / "results" / "step36f_test_artifacts"


@contextmanager
def _owned_test_directory(label: str) -> Iterator[Path]:
    """Own and clean one narrow test directory. / 独占并清理一个很窄的测试目录。"""

    if not label or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
        for character in label
    ):
        raise ValueError("Unsafe test label. / 测试标签不安全。")
    TEST_ARTIFACT_PARENT.mkdir(parents=True, exist_ok=True)
    directory = TEST_ARTIFACT_PARENT / label
    if directory.exists():
        raise RuntimeError(
            f"Refusing to reuse stale test directory: {directory}. / "
            f"拒绝复用遗留测试目录：{directory}。"
        )
    directory.mkdir()
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        if (
            resolved.parent != TEST_ARTIFACT_PARENT.resolve()
            or resolved.name != label
        ):
            raise AssertionError(
                "Refusing to clean an unowned path. / 拒绝清理非本测试路径。"
            )
        for child in sorted(
            resolved.rglob("*"),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
            else:
                raise AssertionError(
                    "Unexpected artifact type. / 测试产物类型异常。"
                )
        resolved.rmdir()


def _stable_training_controller(
    config,
    task,
    *,
    measurement_sink_protocol_id,
    measurement_sink_factory,
) -> SessionPhaseController:
    """Create a deterministic policy that converges immediately in tiny tests.

    建立一个在小型测试中立刻收敛的确定性策略。
    """

    del measurement_sink_protocol_id
    if measurement_sink_factory is None:
        raise AssertionError("Step 36F must keep the real measurement pipeline.")
    values, prices, actions, initial_q, prehistory = build_paper_inputs(
        config.parameters
    )
    stable_q = np.zeros_like(initial_q)
    stable_q[:, 0] = 1_000_000_000.0
    session = build_randomized_paper_session(
        parameters=config.parameters,
        value_grid=values,
        price_grid=prices,
        action_multipliers=actions,
        initial_q_table=stable_q,
        prehistory=prehistory,
        experiment_seed=task.seed_manifest.experiment_seed,
        experiment_cell_key=task.seed_manifest.experiment_cell_key,
        session_index=task.session_index,
    )
    sink = measurement_sink_factory(session)
    return SessionPhaseController.create_for_fresh_session(
        session,
        convergence_periods_required=config.convergence_periods_required,
        measurement_periods_required=config.measurement_periods_required,
        measurement_sink=sink,
    )


@contextmanager
def _stable_controller_patches() -> Iterator[None]:
    """Patch both import sites used by training and replay. / 同时替换训练与重放的导入点。"""

    # Step 36D calls the function through its own module, while Step 36E keeps
    # a direct imported reference for deterministic replay.  Tests must patch
    # both names so training and replay start from the same controlled Q table.
    # / Step 36D 通过自己的模块调用；Step 36E 为确定性重放保存了直接导入的
    # 引用。因此测试必须同时替换两个名字，训练与重放才会从同一张 Q 表开始。
    with (
        patch.object(
            step36d,
            "build_fresh_training_controller",
            side_effect=_stable_training_controller,
        ),
        patch.object(
            step36e,
            "build_fresh_training_controller",
            side_effect=_stable_training_controller,
        ),
    ):
        yield


class PersistedCalibrationBridgeTests(unittest.TestCase):
    """Test persistence, replay provenance, and fail-closed behavior.

    测试持久化、重放来源，以及遇到问题时拒绝继续的行为。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.parameters = PaperParameters(
            noise_std=0.1,
            market_maker_window=20,
        )

    def _plan(self, label: str, *, paths: int = 2):
        """Build one tiny non-research experiment cell. / 建立一个小型非研究实验格。"""

        config = ExperimentCellConfig(
            mode=DEBUG_MODE,
            experiment_cell_key=f"step36f-test-{label}",
            parameters=self.parameters,
            experiment_seed=36_600_101,
            irf_experiment_seed=36_600_102,
            session_count=1,
            convergence_periods_required=1,
            measurement_periods_required=20,
            irf_paths_per_session=paths,
        )
        return build_experiment_cell_plan(
            config,
            ExperimentExecutionPolicy(
                maximum_training_periods=20,
                persisted_post_convergence_bundle_available=True,
            ),
        )

    def _run(self, plan, artifact_root: Path):
        """Run Step 36F under a controlled training policy. / 用受控训练策略运行 36F。"""

        with _stable_controller_patches():
            return step36f.run_and_persist_session_calibration_bridge(
                plan,
                plan.tasks[0],
                artifact_root=artifact_root,
                checkpoint_interval_periods=5,
            )

    def test_happy_path_saves_loads_and_keeps_debug_claims_honest(self) -> None:
        """A complete bridge survives reload but never calls debug output research.

        完整 bridge 可以重读，但绝不把 debug 输出称为研究结果。
        """

        plan = self._plan("happy", paths=3)
        task = plan.tasks[0]
        with _owned_test_directory("happy") as artifact_root:
            execution = self._run(plan, artifact_root)

            self.assertEqual(execution.status, step36f.COMPLETE)
            self.assertIsNotNone(execution.bridge)
            self.assertIsNotNone(execution.measurement_execution)
            self.assertIsNotNone(execution.verified_live_source)
            self.assertTrue(execution.bridge_path.is_file())
            bridge = execution.bridge
            assert bridge is not None
            self.assertEqual(bridge.calibration_paths_requested, 3)
            self.assertEqual(bridge.calibration_paths_executed, 3)
            self.assertTrue(bridge.exact_complete_measurement_replay_verified)
            self.assertTrue(bridge.live_checkpoint_and_scorer_identity_verified)
            self.assertTrue(bridge.a24_per_session_chain_complete)
            self.assertFalse(bridge.a24_full_cell_bridge_complete)
            self.assertFalse(bridge.cell_shock_calibrated)
            self.assertFalse(bridge.step35f_run)
            self.assertFalse(bridge.research_result)
            self.assertFalse(bridge.paper_results_ready)

            with self.assertRaises(ValueError):
                step36f.load_step36f_session_calibration_bridge(
                    execution.bridge_path,
                    expected_plan=plan,
                    expected_task=task,
                    artifact_root=artifact_root,
                )
            loaded = step36f.load_step36f_session_calibration_bridge(
                execution.bridge_path,
                expected_plan=plan,
                expected_task=task,
                artifact_root=artifact_root,
                trusted_local_file=True,
            )
            self.assertEqual(loaded, bridge)

    def test_completed_bridge_is_byte_idempotent(self) -> None:
        """A second call reads the immutable result and does not rerun Step 36E.

        第二次调用只读取不可变结果，不会重新运行 Step 36E。
        """

        plan = self._plan("idempotent")
        with _owned_test_directory("idempotent") as artifact_root:
            first = self._run(plan, artifact_root)
            bytes_before = first.bridge_path.read_bytes()

            with patch.object(
                step36f,
                "run_complete_measurement_task",
                side_effect=AssertionError("completed bridge must not rerun Step 36E"),
            ):
                second = step36f.run_and_persist_session_calibration_bridge(
                    plan,
                    plan.tasks[0],
                    artifact_root=artifact_root,
                    checkpoint_interval_periods=5,
                )

            self.assertEqual(second.status, step36f.COMPLETE)
            self.assertEqual(second.bridge, first.bridge)
            self.assertIsNone(second.measurement_execution)
            self.assertIsNone(second.verified_live_source)
            self.assertEqual(second.bridge_path.read_bytes(), bytes_before)

    def test_reconstruction_reproduces_saved_calibration_and_live_identity(self) -> None:
        """Disk-only input recreates the genuine live scorer and exact checkpoint.

        仅靠硬盘证据，也能重建真正的实时 scorer 与精确 checkpoint。
        """

        plan = self._plan("reconstruct", paths=2)
        task = plan.tasks[0]
        with _owned_test_directory("reconstruct") as artifact_root:
            execution = self._run(plan, artifact_root)
            bridge = execution.bridge
            assert bridge is not None

            with _stable_controller_patches():
                source = step36f.reconstruct_verified_step35f_session_source(
                    bridge,
                    plan=plan,
                    task=task,
                    artifact_root=artifact_root,
                )

            self.assertEqual(source.bridge, bridge)
            self.assertEqual(
                source.evidence.evidence_sha256,
                bridge.complete_evidence_sha256,
            )
            self.assertEqual(
                source.checkpoint.checkpoint_sha256,
                bridge.convergence_checkpoint_sha256,
            )
            self.assertEqual(
                source.baseline_scorer.verified_live_result_for_step35d(
                    source.checkpoint
                ),
                bridge.calibration_receipt.long_run_baseline_receipt,
            )

    def test_fresh_python_process_reconstructs_and_runs_step35f(self) -> None:
        """A separate interpreter can recover the chain and execute Step 35F.

        一个全新的 Python 解释器可以恢复证据链并实际执行 Step 35F。
        """

        # This fixed debug seed is known to admit the paper's positive adverse
        # shock calibration; another perfectly valid short debug sample may
        # already sit beyond the +1.2% target and is then economically
        # undefined for Step 35E. / 这个固定 debug seed 可以校准论文所需的
        # 正逆向冲击；另一个完全有效但很短的样本可能已经超过 +1.2% 目标，
        # 此时 Step 35E 在经济上本来就无定义。
        config = ExperimentCellConfig(
            mode=DEBUG_MODE,
            experiment_cell_key="step36f-test-cross-process",
            parameters=self.parameters,
            experiment_seed=36_600_001,
            irf_experiment_seed=36_600_002,
            session_count=1,
            convergence_periods_required=1,
            measurement_periods_required=20,
            irf_paths_per_session=3,
        )
        plan = build_experiment_cell_plan(
            config,
            ExperimentExecutionPolicy(
                maximum_training_periods=20,
                persisted_post_convergence_bundle_available=True,
            ),
        )
        task = plan.tasks[0]
        with _owned_test_directory("cross-process") as artifact_root:
            # Do not patch the market builder here: monkeypatches cannot and
            # should not cross a process boundary. The one-period debug
            # convergence threshold keeps this genuine run small. / 此处不替换
            # 市场 builder：monkeypatch 不会也不应跨进程；一时期 debug 收敛
            # 阈值使真实运行仍然很小。
            execution = step36f.run_and_persist_session_calibration_bridge(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=5,
            )
            bridge = execution.bridge
            assert bridge is not None

            child_code = r"""
import json
from pathlib import Path
import sys
from src.parameters import PaperParameters
from steps.step_35e_cell_shock_calibration import calibrate_experiment_cell_uniform_shock
from steps.step_35f_paired_response_and_classification import prepare_verified_step35f_cell_context, run_step35f_session_response_paths
from steps.step_36b_experiment_manifest import DEBUG_MODE, ExperimentCellConfig, ExperimentExecutionPolicy, build_experiment_cell_plan
import steps.step_36f_persisted_calibration_bridge as step36f

root = Path(sys.argv[1])
config = ExperimentCellConfig(
    mode=DEBUG_MODE,
    experiment_cell_key="step36f-test-cross-process",
    parameters=PaperParameters(noise_std=0.1, market_maker_window=20),
    experiment_seed=36_600_001,
    irf_experiment_seed=36_600_002,
    session_count=1,
    convergence_periods_required=1,
    measurement_periods_required=20,
    irf_paths_per_session=3,
)
plan = build_experiment_cell_plan(
    config,
    ExperimentExecutionPolicy(
        maximum_training_periods=20,
        persisted_post_convergence_bundle_available=True,
    ),
)
task = plan.tasks[0]
bridge = step36f.load_step36f_session_calibration_bridge(
    step36f._bridge_path(root, task),
    expected_plan=plan,
    expected_task=task,
    artifact_root=root,
    trusted_local_file=True,
)
source = step36f.reconstruct_verified_step35f_session_source(
    bridge,
    plan=plan,
    task=task,
    artifact_root=root,
)
calibration = calibrate_experiment_cell_uniform_shock(
    (bridge.calibration_receipt,),
    expected_session_count=1,
)
context = prepare_verified_step35f_cell_context(
    calibration,
    (bridge.calibration_receipt,),
)
response = run_step35f_session_response_paths(
    source.checkpoint,
    baseline_scorer=source.baseline_scorer,
    context=context,
    path_count=3,
)
print(json.dumps({
    "checkpoint": source.checkpoint.checkpoint_sha256,
    "step35d": bridge.calibration_receipt.receipt_payload_sha256,
    "step35f_paths": response.paths_executed,
    "step35f_receipt": response.receipt_payload_sha256,
}))
"""
            completed = subprocess.run(
                [sys.executable, "-X", "utf8", "-c", child_code, str(artifact_root)],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                self.fail(
                    "Fresh-process Step 35F failed. / 新进程 Step 35F 失败。\n"
                    + completed.stderr
                )
            child_result = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(
                child_result["checkpoint"],
                bridge.convergence_checkpoint_sha256,
            )
            self.assertEqual(
                child_result["step35d"],
                bridge.calibration_receipt.receipt_payload_sha256,
            )
            self.assertEqual(child_result["step35f_paths"], 3)
            self.assertRegex(child_result["step35f_receipt"], r"^[0-9a-f]{64}$")

    def test_tampered_bridge_and_missing_dependency_fail_closed(self) -> None:
        """Changed metadata or a missing retained file is rejected before replay.

        metadata 被修改或依赖文件丢失时，必须在重放前拒绝继续。
        """

        plan = self._plan("tamper")
        task = plan.tasks[0]
        with _owned_test_directory("tamper") as artifact_root:
            execution = self._run(plan, artifact_root)
            bridge = execution.bridge
            assert bridge is not None

            dishonest = replace(
                bridge,
                research_result=True,
                bridge_sha256="",
            )
            dishonest = replace(
                dishonest,
                bridge_sha256=step36f._digest_dataclass(
                    dishonest,
                    "bridge_sha256",
                ),
            )
            with self.assertRaises(ValueError):
                step36f.validate_step36f_session_calibration_bridge(
                    dishonest,
                    expected_plan=plan,
                    expected_task=task,
                )

            origin_record = next(
                record
                for record in bridge.retained_artifacts
                if record.role == step36f.ROLE_CONVERGENCE_ORIGIN
            )
            origin_path = artifact_root / Path(origin_record.relative_path)
            origin_path.unlink()
            with self.assertRaises(ValueError):
                step36f.load_step36f_session_calibration_bridge(
                    execution.bridge_path,
                    expected_plan=plan,
                    expected_task=task,
                    artifact_root=artifact_root,
                    trusted_local_file=True,
                )

    def test_calibration_failure_publishes_no_bridge(self) -> None:
        """A Step-35D error leaves no partial bridge to mistake for COMPLETE.

        Step-35D 出错时不能留下会被误认为 COMPLETE 的半成品 bridge。
        """

        plan = self._plan("failure")
        with _owned_test_directory("failure") as artifact_root:
            with (
                _stable_controller_patches(),
                patch.object(
                    step36f,
                    "run_unshocked_t3_calibration_paths",
                    side_effect=RuntimeError("injected Step-35D failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected Step-35D failure"),
            ):
                step36f.run_and_persist_session_calibration_bridge(
                    plan,
                    plan.tasks[0],
                    artifact_root=artifact_root,
                    checkpoint_interval_periods=5,
                )

            self.assertEqual(
                tuple(artifact_root.rglob(step36f.BRIDGE_FILE_NAME)),
                (),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
