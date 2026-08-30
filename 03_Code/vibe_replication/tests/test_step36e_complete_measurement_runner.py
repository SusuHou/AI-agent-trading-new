"""Focused tests for the Step-36E complete-measurement runner.

第 36E 步完整测量 runner 的专项测试。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
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
from step_28_session_phases import SessionPhase, SessionPhaseController
from steps.step_36b_experiment_manifest import (
    DEBUG_MODE,
    ExperimentCellConfig,
    ExperimentExecutionPolicy,
    build_experiment_cell_plan,
)
import steps.step_36d_single_session_training_runner as step36d
import steps.step_36e_complete_measurement_runner as step36e


TEST_ARTIFACT_PARENT = (
    PROJECT_ROOT / "results" / "step36e_test_artifacts"
)


@contextmanager
def _owned_test_directory(label: str) -> Iterator[Path]:
    """Own and clean one deterministic directory, never a broad result tree.

    独占并清理一个确定性目录，绝不清理宽泛的 results 目录。
    """

    if not label or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in label):
        raise ValueError("Test label is unsafe. / 测试标签不安全。")
    TEST_ARTIFACT_PARENT.mkdir(parents=True, exist_ok=True)
    directory = TEST_ARTIFACT_PARENT / label
    if directory.exists():
        raise RuntimeError(
            f"Refusing to reuse stale test directory: {directory}. "
            f"/ 拒绝复用遗留测试目录：{directory}。"
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


class _FakeTimer:
    """Return predetermined readings. / 返回预先指定的时钟读数。"""

    def __init__(self, *readings: float) -> None:
        self._readings = iter(readings)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return next(self._readings)


def _stable_training_controller(
    config,
    task,
    *,
    measurement_sink_protocol_id,
    measurement_sink_factory,
) -> SessionPhaseController:
    """Build a controlled Q-table whose greedy policy cannot move in three periods.

    建立短期内不会改变贪心策略的 Q 表，让测试能够精确控制收敛时期。
    """

    del measurement_sink_protocol_id
    if measurement_sink_factory is None:
        raise AssertionError("Step 36E must attach its real sink before period zero.")
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


class CompleteMeasurementRunnerTests(unittest.TestCase):
    """Exercise all-or-nothing measurement and replay boundaries.

    检验测量的全有或全无提交，以及失败后的完整重放边界。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.parameters = PaperParameters(
            noise_std=0.1,
            market_maker_window=20,
        )

    def _plan(
        self,
        label: str,
        *,
        convergence_periods: int = 1,
        measurement_periods: int = 20,
        session_count: int = 1,
        cap: int = 20,
    ):
        """Build one tiny deterministic debug cell. / 建立一个确定性的小型调试实验格。"""

        config = ExperimentCellConfig(
            mode=DEBUG_MODE,
            experiment_cell_key=f"step36e-test-{label}",
            parameters=self.parameters,
            experiment_seed=36_200_001,
            irf_experiment_seed=36_200_002,
            session_count=session_count,
            convergence_periods_required=convergence_periods,
            measurement_periods_required=measurement_periods,
            irf_paths_per_session=1,
        )
        return build_experiment_cell_plan(
            config,
            ExperimentExecutionPolicy(maximum_training_periods=cap),
        )

    def test_complete_run_commits_exact_rows_and_cross_checked_receipts(self) -> None:
        """A success commits exactly T rows and every component receipt agrees.

        成功时只提交恰好 T 条，并要求全部组件 receipt 相互一致。
        """

        plan = self._plan("complete-evidence")
        task = plan.tasks[0]
        timer = _FakeTimer(10.0, 24.0)
        with _owned_test_directory("complete-evidence") as artifact_root:
            execution = step36e.run_complete_measurement_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=5,
                timer=timer,
            )

            outcome = execution.status.scientific_outcome
            evidence = execution.evidence
            self.assertEqual(outcome.status, step36e.COMPLETE)
            self.assertEqual(outcome.phase, step36e.MEASUREMENT_COMPLETE_PHASE)
            self.assertEqual(outcome.committed_measurement_rows, 20)
            self.assertIsNotNone(evidence)
            assert evidence is not None
            self.assertEqual(evidence.complete_measurement_rows_committed, 20)
            self.assertEqual(evidence.partial_measurement_rows_committed, 0)
            self.assertEqual(evidence.phase_receipt.measurement_periods_completed, 20)
            self.assertEqual(evidence.profitability_receipt.measurement_periods_scored, 20)
            self.assertEqual(evidence.trading_intensity_receipt.measurement_periods_scored, 20)
            self.assertEqual(evidence.market_liquidity_receipt.measurement_periods_scored, 20)
            self.assertEqual(evidence.mispricing_receipt.measurement_periods_scored, 20)
            self.assertEqual(evidence.irf_long_run_baseline_receipt.measurement_periods_scored, 20)
            self.assertEqual(evidence.learned_session_result.measurement_periods_completed, 20)
            self.assertTrue(
                evidence.learned_session_result.a23_source_fingerprint_scope_resolved
            )
            self.assertTrue(evidence.measurement_restart_not_mid_window_resume)
            self.assertFalse(evidence.research_result)
            self.assertFalse(evidence.paper_results_ready)
            step36e.validate_complete_measurement_evidence(
                evidence,
                expected_plan=plan,
                expected_task=task,
            )
            # The formal aggregator uses this read-only public boundary rather
            # than rerunning the market. / 正式汇总器使用这个只读
            # 公共边界，而不是重新运行市场。
            loaded = step36e.load_completed_measurement_evidence(
                plan,
                task,
                artifact_root=artifact_root,
            )
            self.assertEqual(loaded, evidence)
            self.assertEqual(timer.calls, 2)
            self.assertEqual(execution.status.attempt.elapsed_seconds, 14.0)
            self.assertEqual(execution.status.attempt.measurement_rows_per_second, 20 / 14)

    def test_incomplete_training_resume_matches_uninterrupted_science(self) -> None:
        """A training checkpoint changes operations, not the final science.

        训练 checkpoint 只改变运行过程，不改变最终科学结果。
        """

        plan = self._plan(
            "training-resume",
            convergence_periods=3,
            measurement_periods=20,
            cap=10,
        )
        task = plan.tasks[0]
        with (
            _owned_test_directory("training-continuous") as continuous_root,
            _owned_test_directory("training-resumed") as resumed_root,
            patch.object(
                step36d,
                "build_fresh_training_controller",
                side_effect=_stable_training_controller,
            ),
        ):
            continuous = step36e.run_complete_measurement_task(
                plan,
                task,
                artifact_root=continuous_root,
                checkpoint_interval_periods=2,
            )
            first_piece = step36e.run_complete_measurement_task(
                plan,
                task,
                artifact_root=resumed_root,
                checkpoint_interval_periods=2,
                invocation_training_period_budget=2,
            )
            resumed = step36e.run_complete_measurement_task(
                plan,
                task,
                artifact_root=resumed_root,
                checkpoint_interval_periods=2,
            )

            self.assertEqual(
                first_piece.status.scientific_outcome.status,
                step36e.INCOMPLETE,
            )
            self.assertEqual(
                first_piece.status.scientific_outcome.training_periods_verified,
                2,
            )
            self.assertEqual(first_piece.status.scientific_outcome.committed_measurement_rows, 0)
            self.assertEqual(resumed.status.scientific_outcome.status, step36e.COMPLETE)
            self.assertEqual(
                resumed.status.attempt.start_mode,
                step36e.CONVERGENCE_CONTINUATION_START,
            )
            resumed_evidence = resumed.evidence
            continuous_evidence = continuous.evidence
            self.assertIsNotNone(resumed_evidence)
            self.assertIsNotNone(continuous_evidence)
            assert resumed_evidence is not None
            assert continuous_evidence is not None

            replay_source = (
                resumed_evidence.convergence_origin.replay_mid_training_checkpoint
            )
            self.assertIsNotNone(replay_source)
            assert replay_source is not None
            replay_path = resumed_root / Path(
                *replay_source.relative_path.split("/")
            )
            self.assertTrue(replay_path.is_file())
            self.assertLessEqual(
                len(list(replay_path.parent.glob("*.checkpoint"))),
                step36d.TRAINING_CHECKPOINTS_TO_RETAIN,
            )

            # The canonical Step-35A digest depends only on explicit values,
            # so the row-zero scientific checkpoint and every resulting metric
            # must match exactly.  The outer origin/evidence wrappers correctly
            # remain different because one records a Step-36C resume source and
            # the other records a fresh start. / Step-35A 规范摘要只取决于明确
            # 数值，所以第 0 条前的科学 checkpoint 与全部指标必须完全一致。
            # 外层 origin/evidence 则应不同，因为一个记录 Step-36C 续跑来源，
            # 另一个记录 fresh start。
            self.assertEqual(
                resumed_evidence.convergence_origin.checkpoint,
                continuous_evidence.convergence_origin.checkpoint,
            )
            self.assertEqual(
                resumed_evidence.phase_receipt,
                continuous_evidence.phase_receipt,
            )
            self.assertEqual(
                resumed_evidence.profitability_receipt,
                continuous_evidence.profitability_receipt,
            )
            self.assertEqual(
                resumed_evidence.trading_intensity_receipt,
                continuous_evidence.trading_intensity_receipt,
            )
            self.assertEqual(
                resumed_evidence.price_informativeness_receipt,
                continuous_evidence.price_informativeness_receipt,
            )
            self.assertEqual(
                resumed_evidence.market_liquidity_receipt,
                continuous_evidence.market_liquidity_receipt,
            )
            self.assertEqual(
                resumed_evidence.mispricing_receipt,
                continuous_evidence.mispricing_receipt,
            )
            self.assertEqual(
                resumed_evidence.irf_long_run_baseline_receipt,
                continuous_evidence.irf_long_run_baseline_receipt,
            )
            self.assertEqual(
                resumed_evidence.learned_session_result,
                continuous_evidence.learned_session_result,
            )

    def test_measurement_retry_replays_row_zero_and_matches_control(self) -> None:
        """A partial measurement is discarded; retry reproduces row zero and science.

        半截测量会被丢弃；重试会重新产生第 0 条并复现相同科学结果。
        """

        plan = self._plan("measurement-replay")
        task = plan.tasks[0]
        first_attempt_rows = []
        retry_rows = []

        def fail_after_three(controller, observation) -> None:
            first_attempt_rows.append(observation)
            if controller.measurement_periods_completed == 3:
                raise RuntimeError("deliberate measurement interruption")

        def record_retry(controller, observation) -> None:
            del controller
            retry_rows.append(observation)

        with (
            _owned_test_directory("measurement-control") as control_root,
            _owned_test_directory("measurement-retry") as retry_root,
        ):
            control = step36e.run_complete_measurement_task(
                plan,
                task,
                artifact_root=control_root,
                checkpoint_interval_periods=5,
            )
            with self.assertRaises(step36e.CompleteMeasurementTaskError) as caught:
                step36e.run_complete_measurement_task(
                    plan,
                    task,
                    artifact_root=retry_root,
                    checkpoint_interval_periods=5,
                    measurement_period_completed_hook=fail_after_three,
                )
            failed = caught.exception.status
            self.assertEqual(failed.scientific_outcome.status, step36e.FAILED)
            self.assertEqual(failed.scientific_outcome.committed_measurement_rows, 0)
            self.assertEqual(failed.attempt.measurement_rows_delivered_this_attempt, 3)
            self.assertTrue(failed.attempt.live_runtime_discarded_after_failure)
            self.assertTrue(
                failed.scientific_outcome.replay_entire_window_required_after_failure
            )

            retried = step36e.run_complete_measurement_task(
                plan,
                task,
                artifact_root=retry_root,
                checkpoint_interval_periods=5,
                retry_failed=True,
                measurement_period_completed_hook=record_retry,
            )
            self.assertEqual(retried.status.scientific_outcome.status, step36e.COMPLETE)
            self.assertEqual(
                retried.status.attempt.start_mode,
                step36e.MEASUREMENT_REPLAY_START,
            )
            self.assertEqual(len(retry_rows), 20)
            self.assertEqual(first_attempt_rows[0], retry_rows[0])
            self.assertEqual(first_attempt_rows, retry_rows[:3])
            self.assertEqual(retried.evidence, control.evidence)

    def test_failed_status_without_retry_is_idempotent_and_commits_nothing(self) -> None:
        """FAILED is a no-op until retry authority is explicit. / 未明确 retry 前，FAILED 必须保持不动。"""

        plan = self._plan("failed-idempotent")
        task = plan.tasks[0]

        def fail_first_row(controller, observation) -> None:
            del controller, observation
            raise RuntimeError("stop after first delivered row")

        with _owned_test_directory("failed-idempotent") as artifact_root:
            with self.assertRaises(step36e.CompleteMeasurementTaskError) as caught:
                step36e.run_complete_measurement_task(
                    plan,
                    task,
                    artifact_root=artifact_root,
                    checkpoint_interval_periods=5,
                    measurement_period_completed_hook=fail_first_row,
                )
            before = {
                path.relative_to(artifact_root): path.read_bytes()
                for path in artifact_root.rglob("*")
                if path.is_file()
            }

            def forbidden_timer() -> float:
                raise AssertionError("FAILED no-op must not start timing. / FAILED 空操作不应启动计时。")

            returned = step36e.run_complete_measurement_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=5,
                timer=forbidden_timer,
            )
            after = {
                path.relative_to(artifact_root): path.read_bytes()
                for path in artifact_root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(returned.status, caught.exception.status)
            self.assertIsNone(returned.evidence)
            self.assertIsNone(returned.controller)
            self.assertEqual(after, before)
            evidence_files = list(
                artifact_root.rglob("evidence_*.bundle")
            )
            self.assertEqual(evidence_files, [])

    def test_complete_terminal_call_is_byte_idempotent(self) -> None:
        """A completed task reloads evidence without running another market row.

        已完成任务只重读 evidence，不再运行任何市场时期。
        """

        plan = self._plan("complete-idempotent")
        task = plan.tasks[0]
        with _owned_test_directory("complete-idempotent") as artifact_root:
            first = step36e.run_complete_measurement_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=5,
            )
            before = {
                path.relative_to(artifact_root): path.read_bytes()
                for path in artifact_root.rglob("*")
                if path.is_file()
            }

            def forbidden_timer() -> float:
                raise AssertionError("COMPLETE no-op must not start timing. / COMPLETE 空操作不应启动计时。")

            second = step36e.run_complete_measurement_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=5,
                timer=forbidden_timer,
                measurement_period_completed_hook=lambda *_: (_ for _ in ()).throw(
                    AssertionError("COMPLETE no-op delivered a row")
                ),
            )
            after = {
                path.relative_to(artifact_root): path.read_bytes()
                for path in artifact_root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(second.status, first.status)
            self.assertEqual(second.evidence, first.evidence)
            self.assertIsNone(second.controller)
            self.assertIsNone(second.pipeline)
            self.assertEqual(after, before)

    def test_corrupted_origin_fails_before_timer_and_additional_mutation(self) -> None:
        """A damaged replay origin cannot start an explicit retry.

        损坏的重放原点不能启动明确 retry。
        """

        plan = self._plan("corrupt-origin")
        task = plan.tasks[0]

        def fail_first_row(controller, observation) -> None:
            del controller, observation
            raise RuntimeError("create replay-required status")

        with _owned_test_directory("corrupt-origin") as artifact_root:
            with self.assertRaises(step36e.CompleteMeasurementTaskError) as caught:
                step36e.run_complete_measurement_task(
                    plan,
                    task,
                    artifact_root=artifact_root,
                    checkpoint_interval_periods=5,
                    measurement_period_completed_hook=fail_first_row,
                )
            reference = caught.exception.status.scientific_outcome.convergence_origin_reference
            self.assertIsNotNone(reference)
            assert reference is not None
            origin_path = artifact_root.joinpath(*reference.relative_path.split("/"))
            original = origin_path.read_bytes()
            origin_path.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
            before = {
                path.relative_to(artifact_root): path.read_bytes()
                for path in artifact_root.rglob("*")
                if path.is_file()
            }

            def forbidden_timer() -> float:
                raise AssertionError("Corruption must fail before timing. / 损坏必须在计时前失败。")

            with self.assertRaises(ValueError):
                step36e.run_complete_measurement_task(
                    plan,
                    task,
                    artifact_root=artifact_root,
                    checkpoint_interval_periods=5,
                    retry_failed=True,
                    timer=forbidden_timer,
                )
            after = {
                path.relative_to(artifact_root): path.read_bytes()
                for path in artifact_root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)

    def test_wrong_task_and_unsafe_path_fail_before_artifact_mutation(self) -> None:
        """Task identity and path containment are preflight checks.

        task 身份与路径范围必须在写入 artifact 前检查。
        """

        plan = self._plan("preflight", session_count=2)
        task = plan.tasks[0]
        wrong_task = plan.tasks[1]
        with _owned_test_directory("preflight") as artifact_root:
            foreign_plan = self._plan("foreign-plan")
            with self.assertRaises(ValueError):
                step36e.run_complete_measurement_task(
                    foreign_plan,
                    wrong_task,
                    artifact_root=artifact_root,
                    checkpoint_interval_periods=5,
                )
            unsafe_task = replace(task, relative_artifact_directory="../escape")
            with self.assertRaises(ValueError):
                step36e._task_directory(artifact_root, unsafe_task)
            self.assertEqual(list(artifact_root.iterdir()), [])

    def test_preexisting_step36d_convergence_is_replayed_and_completed(self) -> None:
        """A separately completed Step 36D remains usable by Step 36E.

        先单独完成的 Step 36D 仍可由 Step 36E 重放并完成测量。
        """

        plan = self._plan("preexisting-converged")
        task = plan.tasks[0]
        with _owned_test_directory("preexisting-converged") as artifact_root:
            factory = step36e.MeasurementPipelineFactory()
            training = step36d.run_training_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=2,
                measurement_sink_protocol_id=(
                    step36e.MEASUREMENT_PIPELINE_PROTOCOL_ID
                ),
                measurement_sink_factory=factory,
            )
            self.assertEqual(
                training.status.scientific_outcome.status,
                step36d.CONVERGED,
            )
            terminal_reload = step36d.run_training_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=2,
                measurement_sink_protocol_id=(
                    step36e.MEASUREMENT_PIPELINE_PROTOCOL_ID
                ),
                measurement_sink_factory=(
                    step36e.MeasurementPipelineFactory()
                ),
            )
            self.assertIsNone(terminal_reload.controller)
            execution = step36e.run_complete_measurement_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=2,
            )
            self.assertEqual(
                execution.status.scientific_outcome.status,
                step36e.COMPLETE,
            )
            self.assertEqual(
                execution.status.attempt.start_mode,
                step36e.MEASUREMENT_REPLAY_START,
            )
            self.assertIsNotNone(execution.evidence)

    def test_missing_terminal_replay_source_fails_closed_with_status(self) -> None:
        """A missing Step-36C source cannot publish a fabricated origin.

        Step-36C 重放起点丢失时，不能发布伪造 origin，并须保存 FAILED 状态。
        """

        plan = self._plan(
            "missing-terminal-source",
            convergence_periods=3,
            cap=10,
        )
        task = plan.tasks[0]
        with _owned_test_directory("missing-terminal-source") as artifact_root:
            training = step36d.run_training_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=1,
                measurement_sink_protocol_id=(
                    step36e.MEASUREMENT_PIPELINE_PROTOCOL_ID
                ),
                measurement_sink_factory=(
                    step36e.MeasurementPipelineFactory()
                ),
            )
            scientific = training.status.scientific_outcome
            source = scientific.latest_mid_training_checkpoint
            self.assertEqual(scientific.status, step36d.CONVERGED)
            self.assertIsNotNone(source)
            assert source is not None
            source_path = step36e._safe_artifact_path(
                artifact_root,
                source.relative_path,
            )
            source_path.unlink()
            with self.assertRaises(
                step36e.CompleteMeasurementTaskError
            ) as caught:
                step36e.run_complete_measurement_task(
                    plan,
                    task,
                    artifact_root=artifact_root,
                    checkpoint_interval_periods=1,
                )
            failed = caught.exception.status
            self.assertEqual(
                failed.scientific_outcome.status,
                step36e.FAILED,
            )
            self.assertEqual(
                failed.scientific_outcome.training_periods_verified,
                scientific.verified_training_periods,
            )
            self.assertEqual(
                failed.attempt.training_periods_at_end,
                scientific.verified_training_periods,
            )
            self.assertIsNone(
                failed.scientific_outcome.convergence_origin_reference
            )

    def test_recomputed_wrappers_cannot_hide_contradictory_science(self) -> None:
        """Checksums cannot make mutually inconsistent summaries valid.

        即使重新计算 checksum，互相矛盾的科学摘要也不能通过验证。
        """

        plan = self._plan("contradictory-evidence")
        task = plan.tasks[0]
        with _owned_test_directory("contradictory-evidence") as artifact_root:
            execution = step36e.run_complete_measurement_task(
                plan,
                task,
                artifact_root=artifact_root,
                checkpoint_interval_periods=5,
            )
            evidence = execution.evidence
            self.assertIsNotNone(evidence)
            assert evidence is not None

            altered_result_draft = replace(
                evidence.learned_session_result,
                mean_actual_profit_by_agent=tuple(
                    value + 123.0
                    for value in evidence.learned_session_result.mean_actual_profit_by_agent
                ),
                result_sha256="",
            )
            altered_result = replace(
                altered_result_draft,
                result_sha256=step36e._digest_dataclass(
                    altered_result_draft,
                    "result_sha256",
                ),
            )
            altered_evidence_draft = replace(
                evidence,
                learned_session_result=altered_result,
                evidence_sha256="",
            )
            altered_evidence = replace(
                altered_evidence_draft,
                evidence_sha256=step36e._digest_dataclass(
                    altered_evidence_draft,
                    "evidence_sha256",
                ),
            )
            with self.assertRaises(ValueError):
                step36e.validate_complete_measurement_evidence(
                    altered_evidence,
                    expected_plan=plan,
                    expected_task=task,
                )

            bad_attempt = replace(
                execution.status.attempt,
                failure_type="GhostFailure",
                failure_message=None,
            )
            bad_status = step36e._build_status(
                execution.status.scientific_outcome,
                bad_attempt,
            )
            with self.assertRaises(ValueError):
                step36e.validate_measurement_status(
                    bad_status,
                    expected_task=task,
                    expected_config=plan.config,
                )

    def test_public_steps_share_one_class_identity(self) -> None:
        """Package imports must not create duplicate Step-36 class types.

        package 导入不能制造两套不同的 Step-36 class 类型。
        """

        import steps.step_36b_experiment_manifest as step36b
        import steps.step_36c_exact_training_resume as step36c

        self.assertIs(
            step36e.ExperimentCellPlan,
            step36b.ExperimentCellPlan,
        )
        self.assertIs(
            step36d.MidTrainingCheckpoint,
            step36c.MidTrainingCheckpoint,
        )
        self.assertIs(
            step36e.TrainingCheckpointReference,
            step36d.TrainingCheckpointReference,
        )


if __name__ == "__main__":
    unittest.main()
