"""Tests for Step 36B's deterministic experiment plan. / Step 36B 测试。"""

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = PROJECT_ROOT / "steps"
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
from step_26_reproducible_random_streams import build_session_seed_manifest
from step_27_convergence_tracker import PAPER_UNCHANGED_PERIODS
from step_28_session_phases import PAPER_MEASUREMENT_PERIODS
from step_34_mechanism_classifier import PAPER_PATHS_PER_SESSION
from step_35e_cell_shock_calibration import PAPER_SESSIONS_PER_EXPERIMENT_CELL
from steps.step_36b_experiment_manifest import (
    DEBUG_MODE,
    PAPER_MODE,
    ExperimentCellConfig,
    ExperimentExecutionPolicy,
    build_experiment_cell_plan,
    load_experiment_cell_plan,
    save_experiment_cell_plan,
    select_tasks_for_shard,
    validate_experiment_cell_plan,
)


def make_debug_config(**changes: object) -> ExperimentCellConfig:
    """Return one tiny valid config. / 返回一个很小的有效配置。"""

    config = ExperimentCellConfig(
        mode=DEBUG_MODE,
        experiment_cell_key="step36b-test-low-noise",
        parameters=PaperParameters(noise_std=0.1),
        experiment_seed=123_456,
        irf_experiment_seed=654_321,
        session_count=3,
        convergence_periods_required=5,
        measurement_periods_required=21,
        irf_paths_per_session=11,
    )
    return replace(config, **changes)


class Step36BExperimentManifestTests(unittest.TestCase):
    """Dependency-free planning tests; no learner is trained. / 无依赖计划测试，不训练 agent。"""

    def test_deterministic_replay_and_three_sessions_have_twenty_one_streams(self) -> None:
        """Rebuilding the same plan must reproduce every task and seed exactly.

        相同输入重建计划时，每个任务和种子都必须完全一致。
        """

        config = make_debug_config()
        policy = ExperimentExecutionPolicy(maximum_training_periods=5)
        first = build_experiment_cell_plan(config, policy)
        replay = build_experiment_cell_plan(config, policy)

        self.assertEqual(replay, first)
        validate_experiment_cell_plan(first)
        self.assertEqual(first.task_count, 3)
        self.assertEqual(first.unique_session_seed_count, 3)
        self.assertEqual(first.unique_child_seed_count, 21)
        self.assertEqual(tuple(task.session_index for task in first.tasks), (0, 1, 2))
        self.assertEqual(len({task.task_id for task in first.tasks}), 3)
        self.assertEqual(
            len({task.relative_artifact_directory for task in first.tasks}),
            3,
        )

        all_child_seeds: set[int] = set()
        for task in first.tasks:
            expected_manifest = build_session_seed_manifest(
                config.experiment_seed,
                config.experiment_cell_key,
                task.session_index,
            )
            self.assertEqual(task.seed_manifest, expected_manifest)
            self.assertEqual(task.run_config_sha256, first.run_config_sha256)
            self.assertEqual(
                task.implementation_tree_sha256,
                first.implementation_tree_sha256,
            )
            self.assertEqual(len(task.task_sha256), 64)
            all_child_seeds.update(task.seed_manifest.child_seeds())
        self.assertEqual(len(all_child_seeds), 21)

    def test_operational_training_cap_changes_only_the_plan_identity(self) -> None:
        """An operational cap is not part of the scientific/run configuration.

        运行上限属于操作设置，不属于科学设定或运行配置身份。
        """

        config = make_debug_config()
        cap_five = build_experiment_cell_plan(
            config,
            ExperimentExecutionPolicy(maximum_training_periods=5),
        )
        cap_six = build_experiment_cell_plan(
            config,
            ExperimentExecutionPolicy(maximum_training_periods=6),
        )

        self.assertEqual(
            cap_five.experiment_cell_sha256,
            cap_six.experiment_cell_sha256,
        )
        self.assertEqual(cap_five.run_config_sha256, cap_six.run_config_sha256)
        self.assertEqual(cap_five.tasks_sha256, cap_six.tasks_sha256)
        self.assertEqual(cap_five.tasks, cap_six.tasks)
        self.assertNotEqual(cap_five.plan_sha256, cap_six.plan_sha256)

    def test_experiment_root_seed_changes_run_and_task_seeds_but_not_cell(self) -> None:
        """The seed root identifies a run, not its seed-free scientific cell.

        根种子改变“哪一次运行”，但不改变不含种子的科学实验单元。
        """

        first = build_experiment_cell_plan(make_debug_config())
        second = build_experiment_cell_plan(
            make_debug_config(experiment_seed=123_457)
        )

        self.assertEqual(
            first.experiment_cell_sha256,
            second.experiment_cell_sha256,
        )
        self.assertNotEqual(first.run_config_sha256, second.run_config_sha256)
        self.assertNotEqual(first.tasks_sha256, second.tasks_sha256)
        self.assertNotEqual(
            tuple(task.task_id for task in first.tasks),
            tuple(task.task_id for task in second.tasks),
        )
        self.assertNotEqual(
            tuple(task.seed_manifest.session_seed for task in first.tasks),
            tuple(task.seed_manifest.session_seed for task in second.tasks),
        )

    def test_low_and_high_noise_have_different_scientific_and_run_identities(self) -> None:
        """Changing sigma_u must never reuse a low-noise identity. / 改变 sigma_u 后不得复用低噪声身份。"""

        low = build_experiment_cell_plan(make_debug_config())
        high = build_experiment_cell_plan(
            make_debug_config(parameters=PaperParameters(noise_std=100.0))
        )

        self.assertNotEqual(low.experiment_cell_sha256, high.experiment_cell_sha256)
        self.assertNotEqual(low.run_config_sha256, high.run_config_sha256)
        self.assertNotEqual(low.tasks_sha256, high.tasks_sha256)

    def test_exact_paper_mode_allocates_1000_jobs_but_claims_no_result(self) -> None:
        """Planning formal scale is still zero completed simulations.

        即使规划了正式规模，也仍然没有完成任何模拟结果。
        """

        config = ExperimentCellConfig(
            mode=PAPER_MODE,
            experiment_cell_key="step36b-test-paper-low-noise",
            parameters=PaperParameters(noise_std=0.1),
            experiment_seed=20260829,
            irf_experiment_seed=20260830,
            session_count=PAPER_SESSIONS_PER_EXPERIMENT_CELL,
            convergence_periods_required=PAPER_UNCHANGED_PERIODS,
            measurement_periods_required=PAPER_MEASUREMENT_PERIODS,
            irf_paths_per_session=PAPER_PATHS_PER_SESSION,
        )
        plan = build_experiment_cell_plan(config)

        self.assertEqual(plan.task_count, 1_000)
        self.assertEqual(plan.unique_session_seed_count, 1_000)
        self.assertEqual(plan.unique_child_seed_count, 7_000)
        self.assertEqual(plan.tasks[0].session_index, 0)
        self.assertEqual(plan.tasks[-1].session_index, 999)
        self.assertTrue(plan.formal_mode_requested)
        self.assertTrue(plan.paper_scale_counts_requested)
        self.assertTrue(plan.uncapped_training_requested)
        self.assertFalse(plan.formal_session_runner_connected)
        self.assertTrue(plan.within_session_checkpointing_available)
        self.assertFalse(plan.persisted_post_convergence_bundle_available)
        self.assertFalse(plan.hpc_array_dispatch_available)
        self.assertFalse(plan.research_result)
        self.assertFalse(plan.paper_results_ready)

    def test_formal_policy_exposes_exact_scale_source_identity_and_readiness(self) -> None:
        """A formal plan advertises connected capabilities but no empirical result.

        正式计划应显示已接通的运行能力，但仍不得冒充实证结果。
        """

        config = ExperimentCellConfig(
            mode=PAPER_MODE,
            experiment_cell_key="step36b-test-formal-source-boundary",
            parameters=PaperParameters(noise_std=0.1),
            experiment_seed=20260831,
            irf_experiment_seed=20260901,
            session_count=PAPER_SESSIONS_PER_EXPERIMENT_CELL,
            convergence_periods_required=PAPER_UNCHANGED_PERIODS,
            measurement_periods_required=PAPER_MEASUREMENT_PERIODS,
            irf_paths_per_session=PAPER_PATHS_PER_SESSION,
        )
        policy = ExperimentExecutionPolicy(
            maximum_training_periods=None,
            within_session_checkpointing_available=True,
            persisted_post_convergence_bundle_available=True,
            formal_session_runner_available=True,
            hpc_array_dispatch_available=True,
        )
        plan = build_experiment_cell_plan(config, policy)

        validate_experiment_cell_plan(plan)
        self.assertEqual(plan.task_count, 1_000)
        self.assertEqual(plan.unique_session_seed_count, 1_000)
        self.assertEqual(plan.unique_child_seed_count, 7_000)
        self.assertTrue(plan.formal_mode_requested)
        self.assertTrue(plan.paper_scale_counts_requested)
        self.assertTrue(plan.uncapped_training_requested)
        self.assertTrue(plan.formal_session_runner_connected)
        self.assertTrue(plan.within_session_checkpointing_available)
        self.assertTrue(plan.persisted_post_convergence_bundle_available)
        self.assertTrue(plan.hpc_array_dispatch_available)
        # Readiness metadata is not a completed result. / 就绪元数据不是已完成结果。
        self.assertFalse(plan.research_result)
        self.assertFalse(plan.paper_results_ready)

        self.assertEqual(
            plan.source_scope_manifest_version,
            SOURCE_SCOPE_MANIFEST_VERSION,
        )
        self.assertEqual(
            plan.source_scope_manifest_sha256,
            LOADED_SOURCE_SCOPE_MANIFEST_SHA256,
        )
        self.assertEqual(
            plan.execution_source_sha256,
            LOADED_EXECUTION_SOURCE_SHA256,
        )
        self.assertEqual(
            plan.result_pipeline_source_sha256,
            LOADED_RESULT_PIPELINE_SOURCE_SHA256,
        )
        self.assertTrue(all(len(value) == 64 for value in (
            plan.source_scope_manifest_sha256,
            plan.execution_source_sha256,
            plan.result_pipeline_source_sha256,
        )))
        for task in (plan.tasks[0], plan.tasks[499], plan.tasks[-1]):
            self.assertEqual(
                task.source_scope_manifest_version,
                plan.source_scope_manifest_version,
            )
            self.assertEqual(
                task.source_scope_manifest_sha256,
                plan.source_scope_manifest_sha256,
            )
            self.assertEqual(
                task.execution_source_sha256,
                plan.execution_source_sha256,
            )
            self.assertEqual(
                task.result_pipeline_source_sha256,
                plan.result_pipeline_source_sha256,
            )

    def test_execution_policy_rejects_impossible_capability_dependencies(self) -> None:
        """A downstream capability cannot be true when its prerequisite is false.

        下游能力不能在它的前置能力为 false 时被标成 true。
        """

        config = make_debug_config()
        invalid_policies = (
            ExperimentExecutionPolicy(
                persisted_post_convergence_bundle_available=False,
                formal_session_runner_available=True,
            ),
            ExperimentExecutionPolicy(
                persisted_post_convergence_bundle_available=True,
                formal_session_runner_available=False,
                hpc_array_dispatch_available=True,
            ),
            ExperimentExecutionPolicy(
                within_session_checkpointing_available=False,
            ),
        )
        for policy in invalid_policies:
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError):
                    build_experiment_cell_plan(config, policy)

        for field_name in (
            "within_session_checkpointing_available",
            "persisted_post_convergence_bundle_available",
            "formal_session_runner_available",
            "hpc_array_dispatch_available",
        ):
            with self.subTest(non_boolean_field=field_name):
                with self.assertRaises(TypeError):
                    build_experiment_cell_plan(
                        config,
                        replace(
                            ExperimentExecutionPolicy(),
                            **{field_name: 1},
                        ),
                    )

    def test_json_round_trip_is_idempotent_and_rejects_conflict_and_tamper(self) -> None:
        """Saved plans are immutable audit records. / 已保存计划是不可静默改写的审计记录。"""

        plan = build_experiment_cell_plan(
            make_debug_config(),
            ExperimentExecutionPolicy(maximum_training_periods=5),
        )
        directory = PROJECT_ROOT / "results" / f"step36b_test_{uuid4().hex}"
        path = directory / "experiment_plan.json"
        try:
            save_experiment_cell_plan(plan, path)
            original_bytes = path.read_bytes()
            loaded = load_experiment_cell_plan(path)
            self.assertEqual(loaded, plan)

            # Exact replay is a no-op. / 完全相同的重放不重复写文件。
            save_experiment_cell_plan(plan, path)
            self.assertEqual(path.read_bytes(), original_bytes)

            conflicting = build_experiment_cell_plan(
                make_debug_config(),
                ExperimentExecutionPolicy(maximum_training_periods=6),
            )
            with self.assertRaises(FileExistsError):
                save_experiment_cell_plan(conflicting, path)
            self.assertEqual(path.read_bytes(), original_bytes)

            tampered = json.loads(original_bytes.decode("utf-8"))
            tampered["task_count"] = 999
            path.write_text(json.dumps(tampered), encoding="utf-8")
            tampered_bytes = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "tampered|篡改"):
                load_experiment_cell_plan(path)
            with self.assertRaises(ValueError):
                save_experiment_cell_plan(plan, path)
            self.assertEqual(path.read_bytes(), tampered_bytes)
        finally:
            if path.exists():
                path.unlink()
            if directory.exists():
                directory.rmdir()

    def test_input_validation_rejects_nonfinite_bool_counts_and_unsupported_i(self) -> None:
        """Invalid inputs must fail before any task is built. / 无效输入必须在建立任务前失败。"""

        with self.assertRaisesRegex(ValueError, "finite|有限"):
            build_experiment_cell_plan(
                make_debug_config(
                    parameters=PaperParameters(value_mean=float("nan"))
                )
            )
        with self.assertRaises(TypeError):
            build_experiment_cell_plan(make_debug_config(experiment_seed=True))
        with self.assertRaises(TypeError):
            build_experiment_cell_plan(make_debug_config(session_count=True))

        invalid_counts = (
            {"session_count": 0},
            {"session_count": PAPER_SESSIONS_PER_EXPERIMENT_CELL + 1},
            {"convergence_periods_required": 0},
            {"measurement_periods_required": PAPER_MEASUREMENT_PERIODS + 1},
            {"irf_paths_per_session": 0},
        )
        for changes in invalid_counts:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    build_experiment_cell_plan(make_debug_config(**changes))

        with self.assertRaisesRegex(ValueError, "I=2"):
            build_experiment_cell_plan(
                make_debug_config(parameters=PaperParameters(num_speculators=3))
            )

    def test_sharding_is_deterministic_disjoint_and_complete(self) -> None:
        """Round-robin shards must partition canonical task order exactly.

        轮转分片必须不重不漏地覆盖规范任务顺序。
        """

        plan = build_experiment_cell_plan(make_debug_config(session_count=10))
        shards = tuple(
            select_tasks_for_shard(plan, shard_count=3, shard_index=index)
            for index in range(3)
        )
        replay = tuple(
            select_tasks_for_shard(plan, shard_count=3, shard_index=index)
            for index in range(3)
        )

        self.assertEqual(replay, shards)
        self.assertEqual(
            tuple(task.session_index for task in shards[0]),
            (0, 3, 6, 9),
        )
        self.assertEqual(
            tuple(task.session_index for task in shards[1]),
            (1, 4, 7),
        )
        self.assertEqual(
            tuple(task.session_index for task in shards[2]),
            (2, 5, 8),
        )
        shard_sets = tuple({task.task_id for task in shard} for shard in shards)
        self.assertTrue(shard_sets[0].isdisjoint(shard_sets[1]))
        self.assertTrue(shard_sets[0].isdisjoint(shard_sets[2]))
        self.assertTrue(shard_sets[1].isdisjoint(shard_sets[2]))
        self.assertEqual(
            set().union(*shard_sets),
            {task.task_id for task in plan.tasks},
        )

        with self.assertRaises(ValueError):
            select_tasks_for_shard(plan, shard_count=3, shard_index=3)
        with self.assertRaises(TypeError):
            select_tasks_for_shard(plan, shard_count=True, shard_index=0)


if __name__ == "__main__":
    unittest.main()
