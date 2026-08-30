"""Focused tests for exact Step-36C training restart. / Step 36C 精确训练续跑测试。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import random
import subprocess
import sys
import unittest
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


SPLIT_PERIOD = 19  # Odd: Python random.gauss has one cached normal. / 奇数期会留下正态缓存。
TAIL_PERIODS = 7
CONVERGENCE_PERIODS_REQUIRED = 1_000
MEASUREMENT_PERIODS_REQUIRED = 7
TEST_OUTPUT_DIRECTORY = PROJECT_ROOT / "results"


@contextmanager
def _isolated_disk_test_directory() -> Iterator[Path]:
    """Create one ordinary, uniquely named test directory. / 建立唯一的普通测试目录。"""

    # Windows TemporaryDirectory applies permissions that this managed
    # workspace cannot reuse for child files.  An ordinary mkdir keeps this a
    # real disk round-trip without that platform-only ACL problem.
    # / Windows TemporaryDirectory 的权限会令当前受管工作区无法在其中
    # 再建文件；普通 mkdir 仍然是真实磁盘测试，并避开该 ACL 问题。
    directory = TEST_OUTPUT_DIRECTORY / f"step36c-test-{uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        for child in directory.iterdir():
            if not child.is_file():
                raise AssertionError(
                    "Step-36C disk test created an unexpected directory. / "
                    "Step 36C 磁盘测试创建了意外的子目录。"
                )
            child.unlink()
        directory.rmdir()


def _numpy_global_state_equal(
    left: tuple[object, ...],
    right: tuple[object, ...],
) -> bool:
    """Compare NumPy's legacy global RNG state safely. / 安全比较 NumPy 全局随机状态。"""

    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _runtime_audit(controller: SessionPhaseController) -> tuple[object, ...]:
    """Read every causal training field without advancing it. / 不推进市场，读取全部训练状态。"""

    session = controller.session
    return (
        controller.phase,
        controller.measurement_periods_required,
        controller.measurement_periods_completed,
        controller.measurement_first_period_index,
        controller.final_receipt,
        controller.failure_period_index,
        id(getattr(controller, "_controller_token")),
        session.period_number,
        session.previous_price,
        session.previous_value,
        session.current_value,
        tuple(session.shared_value_visit_counts),
        session.all_random_states(),
        session.market_maker.export_state(),
        tuple(
            (
                trader.q_table.dtype.str,
                trader.q_table.shape,
                trader.q_table.tobytes(order="C"),
                bool(trader.q_table.flags.writeable),
            )
            for trader in session.traders
        ),
        controller.tracker.export_training_state(),
        id(session.after_q_update_observer.__self__),
        id(getattr(session, "_phase_controller_token")),
    )


class ExactTrainingResumeTests(unittest.TestCase):
    """Exercise one genuine disk interruption, not an in-memory shortcut. / 测试真实写盘中断。"""

    @classmethod
    def setUpClass(cls) -> None:
        # A small T_m makes the seven-period tail cross an OLS resynchronization.
        # / 小型 T_m 让续跑路径跨过一次 OLS 重同步。
        cls.parameters = PaperParameters(
            noise_std=0.1,
            market_maker_window=20,
        )
        cls.config = ExperimentCellConfig(
            mode=DEBUG_MODE,
            experiment_cell_key="step36c-unit-low-noise",
            parameters=cls.parameters,
            experiment_seed=36_000_003,
            irf_experiment_seed=36_000_004,
            session_count=2,
            convergence_periods_required=CONVERGENCE_PERIODS_REQUIRED,
            measurement_periods_required=MEASUREMENT_PERIODS_REQUIRED,
            irf_paths_per_session=1,
            mechanism_analysis_enabled=True,
        )
        cls.plan = build_experiment_cell_plan(
            cls.config,
            ExperimentExecutionPolicy(maximum_training_periods=100),
        )
        cls.task, cls.other_task = cls.plan.tasks
        (
            cls.value_grid,
            cls.price_grid,
            cls.action_multipliers,
            cls.initial_q_table,
            cls.prehistory,
        ) = build_paper_inputs(cls.parameters)

    def _controller(
        self,
        *,
        stable_policy: bool = False,
        convergence_periods_required: int = CONVERGENCE_PERIODS_REQUIRED,
    ) -> SessionPhaseController:
        initial_q = self.initial_q_table
        if stable_policy:
            initial_q = np.zeros_like(self.initial_q_table)
            initial_q[:, 0] = 1_000_000_000.0
        session = build_randomized_paper_session(
            parameters=self.parameters,
            value_grid=self.value_grid,
            price_grid=self.price_grid,
            action_multipliers=self.action_multipliers,
            initial_q_table=initial_q,
            prehistory=self.prehistory,
            experiment_seed=self.task.seed_manifest.experiment_seed,
            experiment_cell_key=self.task.seed_manifest.experiment_cell_key,
            session_index=self.task.session_index,
        )
        return SessionPhaseController.create_for_fresh_session(
            session,
            convergence_periods_required=convergence_periods_required,
            measurement_periods_required=MEASUREMENT_PERIODS_REQUIRED,
            measurement_sink=None,
        )

    def _run_training_periods(
        self,
        controller: SessionPhaseController,
        count: int,
    ) -> None:
        for _ in range(count):
            self.assertIsNone(controller.run_next_period())
            self.assertIs(controller.phase, SessionPhase.TRAINING)

    def _capture(
        self,
        controller: SessionPhaseController,
    ) -> step36c.MidTrainingCheckpoint:
        return step36c.capture_mid_training_checkpoint(
            controller,
            task=self.task,
            expected_config=self.config,
        )

    def test_real_disk_resume_matches_after_every_future_period(self) -> None:
        """Continuous and resumed paths match exactly after an odd split. / 奇数切点后逐期完全相同。"""

        uninterrupted = self._controller()
        interrupted = self._controller()
        self._run_training_periods(uninterrupted, SPLIT_PERIOD)
        self._run_training_periods(interrupted, SPLIT_PERIOD)
        self.assertEqual(self._capture(uninterrupted), self._capture(interrupted))

        # One gauss draw per period means an odd split retains gauss_next.
        # / 每期一次 gauss；奇数切点应保留下一枚缓存正态数。
        noise_state = interrupted.session.all_random_states()[2]
        self.assertIsNotNone(noise_state[2])

        checkpoint = self._capture(interrupted)
        with _isolated_disk_test_directory() as temporary_directory:
            path = temporary_directory / "period_19.checkpoint"
            step36c.save_mid_training_checkpoint(
                checkpoint,
                path,
                expected_task=self.task,
                expected_config=self.config,
                expected_measurement_sink_protocol_id=(
                    step36c.NO_MEASUREMENT_SINK_PROTOCOL
                ),
            )
            loaded = step36c.load_mid_training_checkpoint(
                path,
                expected_task=self.task,
                expected_config=self.config,
                expected_measurement_sink_protocol_id=(
                    step36c.NO_MEASUREMENT_SINK_PROTOCOL
                ),
                trusted_local_file=True,
            )
        resumed = step36c.restore_mid_training_controller(
            loaded,
            expected_task=self.task,
            expected_config=self.config,
            expected_measurement_sink_protocol_id=(
                step36c.NO_MEASUREMENT_SINK_PROTOCOL
            ),
        )

        starting_resynchronizations = (
            uninterrupted.session.market_maker.resynchronization_count
        )
        for _ in range(TAIL_PERIODS):
            self.assertIsNone(uninterrupted.run_next_period())
            self.assertIsNone(resumed.run_next_period())
            self.assertEqual(self._capture(uninterrupted), self._capture(resumed))
        self.assertGreater(
            resumed.session.market_maker.resynchronization_count,
            starting_resynchronizations,
        )

    def test_capture_is_read_only_and_does_not_touch_global_rngs(self) -> None:
        """Checkpointing is observation, not another market event. / 保存只是观察，不是额外市场期。"""

        controller = self._controller()
        self._run_training_periods(controller, 3)
        before = _runtime_audit(controller)
        python_global_before = random.getstate()
        numpy_global_before = np.random.get_state()
        checkpoint = self._capture(controller)
        after = _runtime_audit(controller)

        self.assertEqual(before, after)
        self.assertEqual(random.getstate(), python_global_before)
        self.assertTrue(
            _numpy_global_state_equal(np.random.get_state(), numpy_global_before)
        )
        self.assertEqual(checkpoint.payload.period_number, 3)

    def test_restore_rebuilds_independent_runtime_ownership(self) -> None:
        """No mutable memory, RNG, tracker, or token leaks across restart. / 恢复后不共享可变对象。"""

        source = self._controller()
        self._run_training_periods(source, 5)
        restored = step36c.restore_mid_training_controller(
            self._capture(source),
            expected_task=self.task,
            expected_config=self.config,
            expected_measurement_sink_protocol_id=(
                step36c.NO_MEASUREMENT_SINK_PROTOCOL
            ),
        )

        self.assertIsNot(source, restored)
        self.assertIsNot(source.session, restored.session)
        self.assertIsNot(source.tracker, restored.tracker)
        self.assertIsNot(
            source.session.market_maker,
            restored.session.market_maker,
        )
        self.assertIsNot(
            source.session.shared_value_visit_counts,
            restored.session.shared_value_visit_counts,
        )
        self.assertIsNot(
            getattr(source, "_controller_token"),
            getattr(restored, "_controller_token"),
        )
        for left, right in zip(
            source.session.traders,
            restored.session.traders,
            strict=True,
        ):
            self.assertIsNot(left, right)
            self.assertFalse(np.shares_memory(left.q_table, right.q_table))

        source_generators = (
            source.session.streams.initial_state_generator,
            source.session.streams.value_generator,
            source.session.streams.noise_generator,
            source.session.traders[0].mode_random_generator,
            source.session.traders[0].action_random_generator,
            source.session.traders[1].mode_random_generator,
            source.session.traders[1].action_random_generator,
        )
        restored_generators = (
            restored.session.streams.initial_state_generator,
            restored.session.streams.value_generator,
            restored.session.streams.noise_generator,
            restored.session.traders[0].mode_random_generator,
            restored.session.traders[0].action_random_generator,
            restored.session.traders[1].mode_random_generator,
            restored.session.traders[1].action_random_generator,
        )
        self.assertTrue(
            all(
                left is not right
                for left, right in zip(
                    source_generators,
                    restored_generators,
                    strict=True,
                )
            )
        )
        self.assertIs(
            restored.session.after_q_update_observer.__self__,
            restored.tracker,
        )
        self.assertIs(
            getattr(restored.session, "_phase_controller_token"),
            getattr(restored, "_controller_token"),
        )

    def test_loader_rejects_untrusted_tampered_truncated_and_wrong_task(self) -> None:
        """Disk evidence fails closed before use. / 磁盘证据损坏或任务错误时必须拒绝。"""

        controller = self._controller()
        self._run_training_periods(controller, 3)
        checkpoint = self._capture(controller)
        with _isolated_disk_test_directory() as temporary_directory:
            directory = temporary_directory
            valid_path = directory / "valid.checkpoint"
            step36c.save_mid_training_checkpoint(
                checkpoint,
                valid_path,
                expected_task=self.task,
                expected_config=self.config,
                expected_measurement_sink_protocol_id=(
                    step36c.NO_MEASUREMENT_SINK_PROTOCOL
                ),
            )
            with self.assertRaises(ValueError):
                step36c.load_mid_training_checkpoint(
                    valid_path,
                    expected_task=self.task,
                    expected_config=self.config,
                    expected_measurement_sink_protocol_id=(
                        step36c.NO_MEASUREMENT_SINK_PROTOCOL
                    ),
                )
            with self.assertRaises(ValueError):
                step36c.load_mid_training_checkpoint(
                    valid_path,
                    expected_task=self.other_task,
                    expected_config=self.config,
                    expected_measurement_sink_protocol_id=(
                        step36c.NO_MEASUREMENT_SINK_PROTOCOL
                    ),
                    trusted_local_file=True,
                )

            original_bytes = valid_path.read_bytes()
            truncated_path = directory / "truncated.checkpoint"
            truncated_path.write_bytes(original_bytes[:-31])
            tampered_path = directory / "tampered.checkpoint"
            tampered = bytearray(original_bytes)
            tampered[-1] ^= 1
            tampered_path.write_bytes(tampered)
            for invalid_path in (truncated_path, tampered_path):
                with self.assertRaises(ValueError):
                    step36c.load_mid_training_checkpoint(
                        invalid_path,
                        expected_task=self.task,
                        expected_config=self.config,
                        expected_measurement_sink_protocol_id=(
                            step36c.NO_MEASUREMENT_SINK_PROTOCOL
                        ),
                        trusted_local_file=True,
                    )

    def test_tracker_mask_and_streak_tampering_are_rejected(self) -> None:
        """A recomputed outer checksum cannot legitimize false tracker history. / 重算外层摘要也不能伪造 tracker。"""

        controller = self._controller()
        self._run_training_periods(controller, 5)
        checkpoint = self._capture(controller)
        tracker_state = checkpoint.payload.tracker_state

        changed_mask_bytes = bytearray(tracker_state.policy_mask_bytes)
        changed_mask_bytes[0] ^= 1
        changed_mask_state = replace(
            tracker_state,
            policy_mask_bytes=bytes(changed_mask_bytes),
            policy_mask_sha256=sha256(bytes(changed_mask_bytes)).hexdigest(),
        )
        changed_mask_payload = replace(
            checkpoint.payload,
            tracker_state=changed_mask_state,
        )
        changed_mask_checkpoint = step36c.MidTrainingCheckpoint(
            payload=changed_mask_payload,
            checkpoint_sha256=step36c._payload_digest(changed_mask_payload),
        )
        with self.assertRaises(ValueError):
            step36c.verify_mid_training_checkpoint(
                changed_mask_checkpoint,
                expected_task=self.task,
                expected_config=self.config,
                expected_measurement_sink_protocol_id=(
                    step36c.NO_MEASUREMENT_SINK_PROTOCOL
                ),
            )

        changed_streak_state = replace(
            tracker_state,
            unchanged_periods=tracker_state.required_unchanged_periods,
        )
        changed_streak_payload = replace(
            checkpoint.payload,
            tracker_state=changed_streak_state,
        )
        changed_streak_checkpoint = step36c.MidTrainingCheckpoint(
            payload=changed_streak_payload,
            checkpoint_sha256=step36c._payload_digest(changed_streak_payload),
        )
        with self.assertRaises(ValueError):
            step36c.verify_mid_training_checkpoint(
                changed_streak_checkpoint,
                expected_task=self.task,
                expected_config=self.config,
                expected_measurement_sink_protocol_id=(
                    step36c.NO_MEASUREMENT_SINK_PROTOCOL
                ),
            )

    def test_capture_rejects_wrong_task_fresh_phase_and_bad_accounting(self) -> None:
        """Only one consistent, completed, unfinished training boundary is legal. / 只接受完整且未收敛的训练边界。"""

        fresh = self._controller()
        with self.assertRaises(RuntimeError):
            self._capture(fresh)

        controller = self._controller()
        self._run_training_periods(controller, 2)
        with self.assertRaises(ValueError):
            step36c.capture_mid_training_checkpoint(
                controller,
                task=self.other_task,
                expected_config=self.config,
            )

        controller.session.shared_value_visit_counts[0] += 1
        try:
            with self.assertRaises(RuntimeError):
                self._capture(controller)
        finally:
            controller.session.shared_value_visit_counts[0] -= 1

        stable = self._controller(
            stable_policy=True,
            convergence_periods_required=1,
        )
        self.assertIsNone(stable.run_next_period())
        self.assertIs(stable.phase, SessionPhase.MEASUREMENT)
        with self.assertRaises(RuntimeError):
            self._capture(stable)

    def test_capture_binds_live_runtime_to_expected_scientific_config(self) -> None:
        """The same seed identity cannot relabel a high-noise market as low noise. / 同一种子不能把高噪声市场冒充低噪声。"""

        high_noise_parameters = PaperParameters(
            noise_std=100.0,
            market_maker_window=20,
        )
        values, prices, actions, initial_q, prehistory = build_paper_inputs(
            high_noise_parameters
        )
        high_noise_session = build_randomized_paper_session(
            parameters=high_noise_parameters,
            value_grid=values,
            price_grid=prices,
            action_multipliers=actions,
            initial_q_table=initial_q,
            prehistory=prehistory,
            experiment_seed=self.task.seed_manifest.experiment_seed,
            experiment_cell_key=self.task.seed_manifest.experiment_cell_key,
            session_index=self.task.session_index,
        )
        high_noise_controller = SessionPhaseController.create_for_fresh_session(
            high_noise_session,
            convergence_periods_required=CONVERGENCE_PERIODS_REQUIRED,
            measurement_periods_required=MEASUREMENT_PERIODS_REQUIRED,
        )
        self.assertIsNone(high_noise_controller.run_next_period())
        with self.assertRaises(ValueError):
            step36c.capture_mid_training_checkpoint(
                high_noise_controller,
                task=self.task,
                expected_config=self.config,
            )

        wrong_threshold = self._controller(
            convergence_periods_required=CONVERGENCE_PERIODS_REQUIRED + 1
        )
        self._run_training_periods(wrong_threshold, 1)
        with self.assertRaises(ValueError):
            self._capture(wrong_threshold)

    def test_capture_rejects_active_market_maker_transaction(self) -> None:
        """A live rollback transaction cannot disappear across restart. / 活动回滚事务不能在恢复时消失。"""

        controller = self._controller()
        self._run_training_periods(controller, 2)
        token = controller.session.market_maker.begin_reversible_append_transaction(
            max_appends=1
        )
        try:
            with self.assertRaises(RuntimeError):
                self._capture(controller)
        finally:
            controller.session.market_maker.rollback_reversible_append_transaction(
                token
            )

    def test_builtin_wire_loads_in_fresh_package_import_process(self) -> None:
        """Disk codec does not depend on the checkpoint class's import name. / 磁盘格式不依赖 class 导入名称。"""

        controller = self._controller()
        self._run_training_periods(controller, 3)
        checkpoint = self._capture(controller)
        with _isolated_disk_test_directory() as temporary_directory:
            path = temporary_directory / "cross_process.checkpoint"
            step36c.save_mid_training_checkpoint(
                checkpoint,
                path,
                expected_task=self.task,
                expected_config=self.config,
                expected_measurement_sink_protocol_id=(
                    step36c.NO_MEASUREMENT_SINK_PROTOCOL
                ),
            )
            child_code = f"""
from pathlib import Path
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(STEPS_DIRECTORY)!r})
from steps import step_36c_exact_training_resume as c
parameters = c.PaperParameters(noise_std=0.1, market_maker_window=20)
config = c.ExperimentCellConfig(
    mode=c.DEBUG_MODE,
    experiment_cell_key='step36c-unit-low-noise',
    parameters=parameters,
    experiment_seed=36_000_003,
    irf_experiment_seed=36_000_004,
    session_count=2,
    convergence_periods_required=1_000,
    measurement_periods_required=7,
    irf_paths_per_session=1,
    mechanism_analysis_enabled=True,
)
plan = c.build_experiment_cell_plan(
    config,
    c.ExperimentExecutionPolicy(maximum_training_periods=100),
)
task = plan.tasks[0]
loaded = c.load_mid_training_checkpoint(
    Path({str(path)!r}),
    expected_task=task,
    expected_config=config,
    expected_measurement_sink_protocol_id=c.NO_MEASUREMENT_SINK_PROTOCOL,
    trusted_local_file=True,
)
restored = c.restore_mid_training_controller(
    loaded,
    expected_task=task,
    expected_config=config,
    expected_measurement_sink_protocol_id=c.NO_MEASUREMENT_SINK_PROTOCOL,
)
restored.run_next_period()
print(c.capture_mid_training_checkpoint(
    restored,
    task=task,
    expected_config=config,
).checkpoint_sha256)
"""
            completed = subprocess.run(
                [sys.executable, "-X", "utf8", "-c", child_code],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

            parent_loaded = step36c.load_mid_training_checkpoint(
                path,
                expected_task=self.task,
                expected_config=self.config,
                expected_measurement_sink_protocol_id=(
                    step36c.NO_MEASUREMENT_SINK_PROTOCOL
                ),
                trusted_local_file=True,
            )
            parent_restored = step36c.restore_mid_training_controller(
                parent_loaded,
                expected_task=self.task,
                expected_config=self.config,
                expected_measurement_sink_protocol_id=(
                    step36c.NO_MEASUREMENT_SINK_PROTOCOL
                ),
            )
            parent_restored.run_next_period()
            expected_digest = self._capture(parent_restored).checkpoint_sha256
            self.assertEqual(completed.stdout.strip().splitlines()[-1], expected_digest)

    def test_atomic_write_adopts_one_complete_stale_stage(self) -> None:
        """A crash after closing the stage does not block every future retry.

        若进程在关闭完整暂存文件后退出，后续重试不会永远被阻塞。
        """

        with _isolated_disk_test_directory() as directory:
            target = directory / (
                "training_period_000000000019_"
                "0123456789abcdef.checkpoint"
            )
            data = b"complete-stage-from-an-interrupted-write"
            stage = step36c._atomic_staging_path(target, data)
            stage.write_bytes(data)

            step36c._atomic_binary_write(target, data)

            self.assertEqual(target.read_bytes(), data)
            self.assertFalse(stage.exists())


if __name__ == "__main__":
    unittest.main()
