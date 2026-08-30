"""Independent integration tests for Step 35F. / 第 35F 步独立整合测试。

The fixture is intentionally small but genuine: it trains a stable debug
session, captures a live convergence checkpoint, scores Step 35C, runs the
same unshocked paths in Step 35D, calibrates Step 35E, and finally executes
the paired Step-35F branches. / 测试规模刻意缩小，但接线是真实的：先训练一个
稳定调试 session，捕获实时收敛快照，完成第 35C 步评分，再由第 35D 步运行
同一批无冲击路径、第 35E 步校准，最后执行第 35F 步的配对分支。
"""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = PROJECT_ROOT / "steps"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from step_28_session_phases import SessionPhase
from steps.step_34_mechanism_classifier import PAPER_TARGET_PRICE_DEVIATION
from steps.step_35c_irf_long_run_baseline import _build_demo_controller
from steps.step_35d_unshocked_t3_calibration_paths import (
    run_unshocked_t3_calibration_paths,
)
from steps.step_35e_cell_shock_calibration import (
    calibrate_experiment_cell_uniform_shock,
)
import steps.step_35f_paired_response_and_classification as step35f
from steps.step_35f_paired_response_and_classification import (
    OnlinePairedT4Moments,
    ReusablePairedT4Workspace,
    aggregate_step35f_experiment_cell,
    prepare_verified_step35f_cell_context,
    run_step35f_session_response_paths,
    validate_step35f_cell_receipt,
    validate_step35f_session_receipt,
    validate_verified_step35f_cell_context,
)


class TestStep35FPairedResponseAndClassification(unittest.TestCase):
    """Exercise real path execution, aggregation, and provenance checks.

    使用真实路径执行、汇总和来源核对测试第 35F 步。
    """

    IRF_SEED = 20_260_835
    DEBUG_PATHS = 24

    @classmethod
    def setUpClass(cls) -> None:
        """Build the genuine pipeline once so the tests stay fast.

        只建立一次真实流程，使整组测试保持快速。
        """

        controller, scorer = _build_demo_controller()
        while controller.phase is SessionPhase.TRAINING:
            if controller.training_periods_completed >= 5:
                raise AssertionError("The stable Step-35F fixture did not converge.")
            controller.run_next_period()
        checkpoint = scorer.capture_and_bind_convergence_checkpoint(controller)
        controller.run_until_complete()
        baseline = scorer.finalize(controller)
        source = run_unshocked_t3_calibration_paths(
            checkpoint,
            baseline_scorer=scorer,
            irf_experiment_seed=cls.IRF_SEED,
            path_count=cls.DEBUG_PATHS,
        )
        calibration = calibrate_experiment_cell_uniform_shock(
            (source,),
            expected_session_count=1,
        )
        context = prepare_verified_step35f_cell_context(calibration, (source,))
        session_result = run_step35f_session_response_paths(
            checkpoint,
            baseline_scorer=scorer,
            context=context,
        )
        cell_result = aggregate_step35f_experiment_cell(
            context,
            (session_result,),
        )

        cls.controller = controller
        cls.scorer = scorer
        cls.checkpoint = checkpoint
        cls.baseline = baseline
        cls.source = source
        cls.calibration = calibration
        cls.context = context
        cls.session_result = session_result
        cls.cell_result = cell_result

    def test_executed_treatment_paths_hit_the_one_point_two_percent_target(self) -> None:
        """The exact Step-35E level calibration must work on Step-35F paths.

        第 35E 步的精确“价格水平”校准必须在第 35F 步真实路径上命中 1.2%。
        """

        expected_treatment_level = (
            (1.0 + PAPER_TARGET_PRICE_DEVIATION)
            * self.cell_result.pooled_long_run_mean_oriented_price
        )
        self.assertAlmostEqual(
            self.cell_result.pooled_mean_treatment_t3_oriented_price,
            expected_treatment_level,
            places=12,
        )
        self.assertAlmostEqual(
            self.cell_result.achieved_normalized_treatment_price_level_deviation,
            PAPER_TARGET_PRICE_DEVIATION,
            places=12,
        )
        self.assertTrue(
            self.cell_result.exact_paper_target_achieved_on_executed_paths
        )
        validate_step35f_session_receipt(self.session_result)
        validate_step35f_cell_receipt(self.cell_result)

    def test_workspace_replay_is_identical_and_reset_is_exact(self) -> None:
        """Running path zero twice must give the same immutable result.

        连续两次运行第零条路径，必须得到完全相同的不可修改结果。
        """

        workspace = ReusablePairedT4Workspace(
            self.checkpoint,
            baseline_scorer=self.scorer,
            context=self.context,
        )
        first = workspace.run_path(0)
        workspace.verify_exact_checkpoint_reset()
        repeated = workspace.run_path(0)
        workspace.verify_exact_checkpoint_reset()
        workspace.close_and_verify()
        self.assertEqual(repeated, first)
        self.assertEqual(workspace.paths_completed, 2)
        self.assertEqual(workspace.rollbacks_completed, 4)
        self.assertFalse(workspace.is_poisoned)

    def test_each_session_uses_its_own_step35c_order_denominators(self) -> None:
        """Recompute both responses from this session's live Step-35C means.

        使用该 session 自己的实时第 35C 步均值，重新计算两位 agent 的反应。
        """

        own_denominators = tuple(self.baseline.mean_oriented_order_by_agent)
        self.assertEqual(
            self.session_result.session_long_run_mean_oriented_order_by_agent,
            own_denominators,
        )
        for agent_index in range(2):
            expected = (
                self.session_result.mean_treatment_t4_oriented_order_by_agent[
                    agent_index
                ]
                - own_denominators[agent_index]
            ) / own_denominators[agent_index]
            self.assertEqual(
                self.session_result.normalized_order_response_by_agent[agent_index],
                expected,
            )
            detail = self.session_result.normalized_response_details[agent_index]
            self.assertEqual(
                detail.long_run_mean_oriented_order,
                own_denominators[agent_index],
            )
        self.assertTrue(
            self.session_result.paper_primary_long_run_denominator_used
        )

    def test_changed_session_cell_or_context_receipts_are_rejected(self) -> None:
        """Frozen records are tamper-evident, not silently editable.

        frozen 凭证能发现篡改，不能被悄悄修改。
        """

        with self.assertRaises(FrozenInstanceError):
            self.session_result.paths_executed = 999  # type: ignore[misc]
        with self.assertRaises(ValueError):
            validate_step35f_session_receipt(
                replace(
                    self.session_result,
                    normalized_order_response_by_agent=(99.0, 99.0),
                )
            )
        with self.assertRaises(ValueError):
            validate_step35f_cell_receipt(
                replace(
                    self.cell_result,
                    price_trigger_session_count=(
                        self.cell_result.price_trigger_session_count + 1
                    ),
                )
            )
        with self.assertRaises(ValueError):
            validate_verified_step35f_cell_context(
                replace(self.context, source_manifest_sha256="0" * 64)
            )

    def test_context_rejects_a_different_genuine_step35d_source(self) -> None:
        """A valid receipt from another IRF seed cannot rebuild this calibration.

        来自另一 IRF seed 的有效凭证不能冒充本次校准的来源。
        """

        other_source = run_unshocked_t3_calibration_paths(
            self.checkpoint,
            baseline_scorer=self.scorer,
            irf_experiment_seed=self.IRF_SEED + 2,
            path_count=self.DEBUG_PATHS,
        )
        with self.assertRaises(ValueError):
            prepare_verified_step35f_cell_context(
                self.calibration,
                (other_source,),
            )

    def test_path_count_must_equal_the_step35d_source_count(self) -> None:
        """Step 35F cannot quietly use fewer paths than Step 35D.

        第 35F 步不能悄悄使用比第 35D 步更少的路径。
        """

        with self.assertRaisesRegex(ValueError, "same canonical path count|完全相同"):
            run_step35f_session_response_paths(
                self.checkpoint,
                baseline_scorer=self.scorer,
                context=self.context,
                path_count=self.DEBUG_PATHS - 1,
            )

    def test_debug_run_never_claims_formal_paper_scale(self) -> None:
        """One by 24 tests wiring; it is not 1,000 by 10,000 evidence.

        1 个 session 乘 24 条路径只能测试接线，不是 1,000 乘 10,000 的论文证据。
        """

        session = self.session_result
        cell = self.cell_result
        self.assertEqual(session.paths_executed, self.DEBUG_PATHS)
        self.assertFalse(session.paper_10000_paths_verified)
        self.assertFalse(session.paper_scale_long_run_source_verified)
        self.assertFalse(session.formal_session_classification_ready)
        self.assertTrue(session.session_classification_computed)
        self.assertFalse(cell.paper_1000_sessions_verified)
        self.assertFalse(cell.paper_10000_paths_per_session_verified)
        self.assertFalse(cell.formal_paper_mechanism_result_ready)
        self.assertEqual(cell.sessions_received, 1)
        self.assertEqual(
            cell.price_trigger_session_count
            + cell.over_pruning_session_count
            + cell.unclassified_session_count,
            1,
        )

    def test_forged_irf_seed_or_shock_cannot_cross_the_receipt_chain(self) -> None:
        """Even a recomputed local checksum cannot change provenance or shock.

        即使重新计算本地 checksum，也不能更改 IRF seed 或统一冲击幅度。
        """

        changed_seed = replace(
            self.session_result,
            irf_experiment_seed=123,
            receipt_payload_sha256="",
        )
        changed_seed = replace(
            changed_seed,
            receipt_payload_sha256=step35f._session_receipt_digest(changed_seed),
        )
        # The standalone session knows only that 123 is a valid uint64; the
        # cell source chain proves it is the wrong seed. / 单个 session 只知道
        # 123 是合法整数；实验单元来源链才能证明它不是本次 seed。
        validate_step35f_session_receipt(changed_seed)
        with self.assertRaises(ValueError):
            aggregate_step35f_experiment_cell(self.context, (changed_seed,))

        changed_shock = replace(
            self.session_result,
            selected_absolute_noise_shock=(
                2.0 * self.session_result.selected_absolute_noise_shock
            ),
            receipt_payload_sha256="",
        )
        changed_shock = replace(
            changed_shock,
            receipt_payload_sha256=step35f._session_receipt_digest(changed_shock),
        )
        with self.assertRaises(ValueError):
            validate_step35f_session_receipt(changed_shock)

    def test_public_online_reducer_rejects_inconsistent_path_arithmetic(self) -> None:
        """A hand-built path object cannot inject a false t=3 increment.

        手工构造的路径对象不能注入错误的 t=3 价格增量。
        """

        workspace = ReusablePairedT4Workspace(
            self.checkpoint,
            baseline_scorer=self.scorer,
            context=self.context,
        )
        path_result = workspace.run_path(0)
        workspace.close_and_verify()
        reducer = OnlinePairedT4Moments(
            workspace.schedule_context,
            selected_absolute_noise_shock=workspace.absolute_noise_shock,
        )
        bad = replace(
            path_result,
            paired_t3_oriented_price_increment=(
                path_result.paired_t3_oriented_price_increment + 0.1
            ),
        )
        with self.assertRaises(ValueError):
            reducer.add(bad)
        self.assertEqual(reducer.count, 0)

    def test_second_transaction_start_failure_rolls_back_the_first(self) -> None:
        """A treatment-start failure cannot leave control inside a transaction.

        实验组事务启动失败时，对照组不能遗留在未关闭事务中。
        """

        workspace = ReusablePairedT4Workspace(
            self.checkpoint,
            baseline_scorer=self.scorer,
            context=self.context,
        )
        with patch.object(
            workspace._treatment,
            "begin_reversible_frozen_supplied_path",
            side_effect=RuntimeError("injected treatment-start failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                workspace.run_path(0)
        self.assertTrue(workspace.is_poisoned)
        # Starting a new control transaction proves the previous one was
        # actually removed. / 能重新启动对照事务，证明旧事务已经被清除。
        token = workspace._control.begin_reversible_frozen_supplied_path(
            max_periods=1
        )
        self.assertEqual(
            workspace._control.rollback_reversible_frozen_supplied_path(token),
            0,
        )


if __name__ == "__main__":
    unittest.main()
