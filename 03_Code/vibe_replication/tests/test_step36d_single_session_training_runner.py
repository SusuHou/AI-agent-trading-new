"""Focused tests for the Step-36D one-session runner. / 第 36D 步单 session runner 测试。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

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
from step_28_session_phases import SessionPhase, SessionPhaseController
from steps.step_36b_experiment_manifest import (
    DEBUG_MODE,
    ExperimentCellConfig,
    ExperimentExecutionPolicy,
    build_experiment_cell_plan,
)
import steps.step_36c_exact_training_resume as step36c
import steps.step_36d_single_session_training_runner as step36d


TEST_OUTPUT_PARENT = PROJECT_ROOT / "results"


@contextmanager
def _owned_test_directory(label: str) -> Iterator[Path]:
    """Create and remove only one uniquely owned result tree. / 只清理本测试独占的结果树。"""

    # Keep the name short because the real checkpoint filename is long and
    # classic Windows paths may still have a 260-character ceiling. / 名字要短，
    # 因为真实 checkpoint 文件名很长，而传统 Windows 路径可能仍有 260 字符上限。
    directory = TEST_OUTPUT_PARENT / f"s36d-{label[:4]}-{uuid4().hex[:6]}"
    directory.mkdir()
    try:
        yield directory
    finally:
        # Resolve and check the exact parent before deleting anything.  The
        # runner legitimately creates nested task/checkpoint directories.
        # / 删除前核对精确父目录；runner 会正常建立嵌套任务和 checkpoint 目录。
        resolved = directory.resolve()
        if (
            resolved.parent != TEST_OUTPUT_PARENT.resolve()
            or not resolved.name.startswith("s36d-")
        ):
            raise AssertionError("Refusing to clean an unowned test path. / 拒绝清理非本测试路径。")
        children = sorted(
            resolved.rglob("*"),
            key=lambda child: len(child.parts),
            reverse=True,
        )
        for child in children:
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
            else:
                raise AssertionError("Unexpected test artifact type. / 测试产物类型异常。")
        resolved.rmdir()


class _FakeTimer:
    """Return predetermined clock readings. / 返回预先指定的时钟读数。"""

    def __init__(self, *readings: float) -> None:
        self._readings = iter(readings)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return next(self._readings)


class SingleSessionTrainingRunnerTests(unittest.TestCase):
    """Exercise scientific and operational Step-36D boundaries. / 检验第 36D 步边界。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.parameters = PaperParameters(
            noise_std=0.1,
            market_maker_window=20,
        )
        cls.config = ExperimentCellConfig(
            mode=DEBUG_MODE,
            experiment_cell_key="step36d-unit-low-noise",
            parameters=cls.parameters,
            experiment_seed=36_000_011,
            irf_experiment_seed=36_000_012,
            session_count=1,
            convergence_periods_required=1_000,
            measurement_periods_required=7,
            irf_paths_per_session=1,
        )
        (
            cls.value_grid,
            cls.price_grid,
            cls.action_multipliers,
            cls.initial_q_table,
            cls.prehistory,
        ) = build_paper_inputs(cls.parameters)

    def _plan(self, cap: int):
        """Build one valid debug plan with an operational cap. / 建立带运行上限的调试计划。"""

        return build_experiment_cell_plan(
            self.config,
            ExperimentExecutionPolicy(maximum_training_periods=cap),
        )

    def _stable_controller(
        self,
        config: ExperimentCellConfig,
        task,
    ) -> SessionPhaseController:
        """Make action zero uniquely best so convergence is controlled. / 固定唯一最优动作以控制收敛。"""

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
        return SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=config.convergence_periods_required,
            measurement_periods_required=config.measurement_periods_required,
            measurement_sink=None,
        )

    def test_fresh_and_resumed_runs_have_identical_scientific_state(self) -> None:
        """A disk interruption changes operations, not market science. / 写盘中断不能改变市场科学状态。"""

        plan = self._plan(cap=6)
        task = plan.tasks[0]
        with (
            _owned_test_directory("continuous") as continuous_root,
            _owned_test_directory("resumed") as resumed_root,
        ):
            continuous = step36d.run_training_task(
                plan,
                task,
                artifact_root=continuous_root,
                checkpoint_interval_periods=2,
            )
            first_piece = step36d.run_training_task(
                plan,
                task,
                artifact_root=resumed_root,
                checkpoint_interval_periods=2,
                invocation_period_budget=3,
            )
            resumed = step36d.run_training_task(
                plan,
                task,
                artifact_root=resumed_root,
                checkpoint_interval_periods=2,
            )

            self.assertEqual(first_piece.status.scientific_outcome.status, step36d.INCOMPLETE)
            self.assertEqual(first_piece.status.scientific_outcome.verified_training_periods, 3)
            self.assertEqual(continuous.status.scientific_outcome, resumed.status.scientific_outcome)
            self.assertEqual(resumed.status.attempt.start_mode, step36d.RESUMED_START)
            self.assertEqual(resumed.status.attempt.starting_training_period, 3)
            self.assertIsNotNone(continuous.controller)
            self.assertIsNotNone(resumed.controller)
            continuous_checkpoint = step36c.capture_mid_training_checkpoint(
                continuous.controller,
                task=task,
                expected_config=self.config,
            )
            resumed_checkpoint = step36c.capture_mid_training_checkpoint(
                resumed.controller,
                task=task,
                expected_config=self.config,
            )
            self.assertEqual(continuous_checkpoint, resumed_checkpoint)

    def test_checkpoint_cadence_uses_global_periods_and_flushes_terminal_state(self) -> None:
        """Resume cadence is global; every incomplete stop is durable. / 续跑按全局时期保存，停止点必须落盘。"""

        plan = self._plan(cap=8)
        task = plan.tasks[0]
        with _owned_test_directory("cadence") as artifact_root:
            first = step36d.run_training_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=3,
                invocation_period_budget=5,
            )
            second = step36d.run_training_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=3,
            )

            self.assertEqual(first.status.attempt.checkpoint_periods_written, (3, 5))
            self.assertEqual(second.status.attempt.checkpoint_periods_written, (6, 8))
            self.assertEqual(second.status.scientific_outcome.stop_reason, "absolute_training_cap_reached")
            checkpoint_directory = (
                artifact_root
                / Path(*task.relative_artifact_directory.split("/"))
                / step36d.CHECKPOINT_DIRECTORY_NAME
            )
            periods = sorted(
                int(step36d.CHECKPOINT_FILE_PATTERN.fullmatch(path.name).group("period"))
                for path in checkpoint_directory.glob("*.checkpoint")
            )
            # The attempt receipt records every write, while disk retains only
            # the newest two exact resume points. / attempt receipt 仍记录每次
            # 写入，但硬盘只保留最新两个精确恢复点。
            self.assertEqual(periods, [6, 8])

    def test_retention_keeps_latest_two_plus_explicit_evidence_source(self) -> None:
        """Evidence may pin one older checkpoint beyond the rolling two.

        证据可以在滚动保留的最新两份之外，固定保护一份更旧的 checkpoint。
        """

        plan = self._plan(cap=8)
        task = plan.tasks[0]
        with _owned_test_directory("retention-evidence") as artifact_root:
            step36d.run_training_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=2,
                invocation_period_budget=2,
            )
            task_directory = artifact_root / Path(
                *task.relative_artifact_directory.split("/")
            )
            checkpoint_directory = (
                task_directory / step36d.CHECKPOINT_DIRECTORY_NAME
            )
            protected = next(checkpoint_directory.glob("*.checkpoint"))

            step36d.run_training_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=2,
                invocation_period_budget=6,
            )
            # Period 2 was normally pruned once periods 6 and 8 existed.  This
            # standalone helper test recreates its role using the then-current
            # oldest file before adding later files. / 当 6、8 期存在后，第 2
            # 期通常会被清理；这里先保护当时的最旧文件，测试额外证据集合。
            self.assertFalse(protected.exists())

            # Run a fresh directory where the old file is protected before
            # pruning later files. / 在新目录中，让旧文件在后续清理前受保护。
        with _owned_test_directory("retention-protected") as artifact_root:
            checkpoint_directory = artifact_root / "managed"
            checkpoint_directory.mkdir()
            paths = []
            for period in (2, 4, 6, 8):
                path = checkpoint_directory / (
                    f"training_period_{period:012d}_{period:016x}.checkpoint"
                )
                path.write_bytes(b"test-only")
                paths.append(path)
            retained = step36d.prune_training_checkpoints(
                checkpoint_directory,
                protected_checkpoint_paths=(paths[0],),
            )
            self.assertEqual(
                {path.name for path in retained},
                {paths[0].name, paths[2].name, paths[3].name},
            )
            self.assertTrue(paths[0].exists())
            self.assertFalse(paths[1].exists())

    def test_convergence_at_cap_runs_zero_measurements_and_is_idempotent(self) -> None:
        """Convergence wins at the exact cap and rerunning does nothing. / 正好在上限收敛，重跑不做任何事。"""

        converging_config = replace(
            self.config,
            experiment_cell_key="step36d-unit-converging",
            convergence_periods_required=1,
            measurement_periods_required=3,
        )
        plan = build_experiment_cell_plan(
            converging_config,
            ExperimentExecutionPolicy(maximum_training_periods=1),
        )
        task = plan.tasks[0]
        controller = self._stable_controller(converging_config, task)
        with _owned_test_directory("converged") as artifact_root:
            with patch.object(
                step36d,
                "build_fresh_training_controller",
                return_value=controller,
            ):
                first = step36d.run_training_task(
                    plan,
                    task,
                    artifact_root=artifact_root,
                    checkpoint_interval_periods=10,
                )

            self.assertEqual(first.status.scientific_outcome.status, step36d.CONVERGED)
            self.assertEqual(first.status.scientific_outcome.verified_training_periods, 1)
            self.assertEqual(first.status.scientific_outcome.convergence_period_index, 0)
            self.assertTrue(first.status.scientific_outcome.ready_for_measurement)
            self.assertEqual(first.status.scientific_outcome.measurement_periods_completed, 0)
            self.assertIs(first.controller.phase, SessionPhase.MEASUREMENT)
            self.assertEqual(first.controller.measurement_periods_completed, 0)
            self.assertIsNotNone(first.converged_checkpoint)
            before = {
                path.relative_to(artifact_root): path.read_bytes()
                for path in artifact_root.rglob("*")
                if path.is_file()
            }

            def forbidden_timer() -> float:
                raise AssertionError("An idempotent converged call must not start timing. / 已收敛重调不应启动计时。")

            second = step36d.run_training_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=10,
                timer=forbidden_timer,
            )
            after = {
                path.relative_to(artifact_root): path.read_bytes()
                for path in artifact_root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(second.status, first.status)
            self.assertIsNone(second.controller)
            self.assertIsNone(second.converged_checkpoint)
            self.assertEqual(after, before)

    def test_failure_reports_only_the_last_durable_checkpoint(self) -> None:
        """An exception discards later live mutations and persists FAILED. / 异常会丢弃落盘点之后的实时改动。"""

        plan = self._plan(cap=10)
        task = plan.tasks[0]

        def fail_after_period_three(controller: SessionPhaseController) -> None:
            if controller.training_periods_completed == 3:
                raise RuntimeError("deliberate hook failure")

        with _owned_test_directory("failure") as artifact_root:
            with self.assertRaises(step36d.TrainingTaskExecutionError) as caught:
                step36d.run_training_task(
                    plan,
                    task,
                    artifact_root=artifact_root,
                    checkpoint_interval_periods=2,
                    period_completed_hook=fail_after_period_three,
                    timer=_FakeTimer(10.0, 13.0),
                )
            status = caught.exception.status
            self.assertEqual(status.scientific_outcome.status, step36d.FAILED)
            self.assertEqual(status.scientific_outcome.verified_training_periods, 2)
            self.assertEqual(status.scientific_outcome.latest_mid_training_checkpoint.period_number, 2)
            self.assertEqual(status.attempt.successful_periods_this_attempt, 3)
            self.assertEqual(status.attempt.ending_verified_period, 2)
            self.assertTrue(status.attempt.live_state_discarded_after_failure)
            self.assertEqual(status.attempt.failure_type, "RuntimeError")
            loaded = step36d.load_training_status(
                caught.exception.status_path,
                expected_task=task,
                expected_config=self.config,
            )
            self.assertEqual(loaded, status)

            # Without explicit retry authority, FAILED is an idempotent return.
            # / 未明确允许 retry 时，FAILED 状态只能原样返回。
            returned = step36d.run_training_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=2,
                timer=lambda: (_ for _ in ()).throw(AssertionError("timer called")),
            )
            self.assertEqual(returned.status, status)
            self.assertIsNone(returned.controller)

    def test_fake_timer_separates_throughput_from_scientific_state(self) -> None:
        """Injected time gives exact operational metadata. / 注入时钟给出精确运行元数据。"""

        plan = self._plan(cap=10)
        task = plan.tasks[0]
        timer = _FakeTimer(100.0, 104.0)
        with _owned_test_directory("timer") as artifact_root:
            execution = step36d.run_training_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=99,
                invocation_period_budget=2,
                timer=timer,
            )
            self.assertEqual(timer.calls, 2)
            self.assertEqual(execution.status.attempt.elapsed_seconds, 4.0)
            self.assertEqual(execution.status.attempt.successful_periods_this_attempt, 2)
            self.assertEqual(execution.status.attempt.periods_per_second, 0.5)
            self.assertEqual(execution.status.scientific_outcome.verified_training_periods, 2)

    def test_invalid_plan_task_and_path_fail_before_artifact_mutation(self) -> None:
        """Identity/path checks happen before any runner output. / 身份与路径错误必须在写文件前失败。"""

        plan = self._plan(cap=3)
        task = plan.tasks[0]
        with _owned_test_directory("preflight") as artifact_root:
            bad_plan = replace(plan, plan_sha256="f" * 64)
            with self.assertRaises(ValueError):
                step36d.run_training_task(
                    bad_plan,
                    task,
                    artifact_root=artifact_root,
                    checkpoint_interval_periods=1,
                )
            bad_task = replace(task, task_id="not-a-plan-member")
            with self.assertRaises(ValueError):
                step36d.run_training_task(
                    plan,
                    bad_task,
                    artifact_root=artifact_root,
                    checkpoint_interval_periods=1,
                )
            unsafe_task = replace(task, relative_artifact_directory="../escape")
            with self.assertRaises(ValueError):
                step36d._task_artifact_directory(artifact_root, unsafe_task)
            self.assertEqual(list(artifact_root.iterdir()), [])

    def test_call_at_existing_absolute_cap_is_a_training_no_op(self) -> None:
        """A resumed task already at cap executes zero market periods. / 已到上限的续跑任务不执行市场期。"""

        plan = self._plan(cap=2)
        task = plan.tasks[0]
        with _owned_test_directory("cap-no-op") as artifact_root:
            first = step36d.run_training_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=10,
            )
            checkpoint_reference = first.status.scientific_outcome.latest_mid_training_checkpoint
            second = step36d.run_training_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=10,
                timer=_FakeTimer(20.0, 20.0),
            )
            self.assertEqual(second.status.scientific_outcome, first.status.scientific_outcome)
            self.assertEqual(second.status.attempt.start_mode, step36d.RESUMED_START)
            self.assertEqual(second.status.attempt.starting_training_period, 2)
            self.assertEqual(second.status.attempt.successful_periods_this_attempt, 0)
            self.assertEqual(second.status.attempt.checkpoint_periods_written, ())
            self.assertEqual(second.status.scientific_outcome.latest_mid_training_checkpoint, checkpoint_reference)


if __name__ == "__main__":
    unittest.main()
