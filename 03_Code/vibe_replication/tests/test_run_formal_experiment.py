"""Focused orchestration tests for ``run_formal_experiment.py``.

``run_formal_experiment.py`` 的专项调度测试。

These tests build metadata and use mocks; they never train a market session.
/ 这些测试只建立元数据并使用 mock；绝不训练市场 session。
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from hashlib import sha256
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse
import json
import os
import sys
import unittest
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = PROJECT_ROOT / "steps"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


import run_formal_experiment as formal


TEST_ARTIFACT_PARENT = PROJECT_ROOT / "results" / "formal_core_runner_test_artifacts"


@contextmanager
def _owned_workspace_directory(prefix: str):
    """Create and safely clean one test-owned workspace directory.

    建立并安全清理一个由本测试独占的工作区目录。
    """

    TEST_ARTIFACT_PARENT.mkdir(parents=True, exist_ok=True)
    directory = TEST_ARTIFACT_PARENT / f"{prefix}-{uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        if resolved.parent != TEST_ARTIFACT_PARENT.resolve():
            raise AssertionError("Refusing unsafe test cleanup. / 拒绝不安全的测试清理。")
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
                raise AssertionError("Unexpected test artifact type. / 测试 artifact 类型异常。")
        resolved.rmdir()


def _arguments(**changes: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "cell": None,
        "session_index": None,
        "array_index": None,
    }
    values.update(changes)
    return argparse.Namespace(**values)


class FormalCoreRunnerTests(unittest.TestCase):
    """No-market tests for the formal core entrypoint. / 正式核心入口的无市场测试。"""

    def test_exact_plans_and_immutable_family_round_trip(self) -> None:
        """Both saved plans must carry exact PAPER_MODE counts and flags.

        两份保存计划都必须带有精确 PAPER_MODE 规模与运行标记。
        """

        with _owned_workspace_directory("plan") as root:
            manifest = formal.initialize_family(root)
            replay = formal.load_family_manifest(root)
            self.assertEqual(replay, manifest)
            self.assertEqual(manifest["session_count_per_cell"], 1_000)
            self.assertEqual(manifest["convergence_unchanged_periods"], 1_000_000)
            self.assertEqual(manifest["measurement_periods_per_session"], 100_000)
            self.assertEqual(
                manifest["irf_paths_per_session_reserved_for_later_steps"],
                10_000,
            )
            self.assertEqual(
                manifest["unique_session_seed_count_across_cells"],
                2_000,
            )
            self.assertEqual(
                manifest["unique_child_stream_seed_count_across_cells"],
                14_000,
            )
            self.assertFalse(manifest["step35e_common_shock_calibrated"])
            self.assertFalse(manifest["mechanism_results_included"])

            for cell, sigma_u in (("low", 0.1), ("high", 100.0)):
                plan_path = formal._checked_relative_child(
                    root,
                    formal.PLAN_RELATIVE_PATHS[cell],
                )
                plan = formal.load_experiment_cell_plan(plan_path)
                self.assertEqual(plan.config.mode, formal.PAPER_MODE)
                self.assertEqual(plan.config.parameters.noise_std, sigma_u)
                self.assertEqual(plan.task_count, 1_000)
                self.assertEqual(plan.unique_child_seed_count, 7_000)
                self.assertTrue(plan.paper_scale_counts_requested)
                self.assertTrue(plan.uncapped_training_requested)
                self.assertTrue(plan.formal_session_runner_connected)
                self.assertTrue(plan.within_session_checkpointing_available)
                self.assertTrue(plan.persisted_post_convergence_bundle_available)
                self.assertTrue(plan.hpc_array_dispatch_available)
                self.assertFalse(plan.research_result)

    def test_runner_hash_normalizes_windows_crlf(self) -> None:
        """One source has one runner identity on Windows and Linux/HPC.

        同一源码在 Windows 与 Linux/超算上具有同一个 runner 身份。
        """

        with patch.object(Path, "read_bytes", return_value=b"a\r\nb\r\n"):
            observed = formal._runner_sha256()
        self.assertEqual(observed, sha256(b"a\nb\n").hexdigest())

    def test_explicit_and_array_target_mapping_and_slurm_conflicts(self) -> None:
        """0..999 maps low; 1000..1999 maps high. / 0..999 映射 low；1000..1999 映射 high。"""

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                formal._resolve_run_target(
                    _arguments(cell="high", session_index=7)
                ),
                ("high", 7, None),
            )
            for array_index, expected in (
                (0, ("low", 0, 0)),
                (999, ("low", 999, 999)),
                (1000, ("high", 0, 1000)),
                (1999, ("high", 999, 1999)),
            ):
                self.assertEqual(
                    formal._resolve_run_target(
                        _arguments(array_index=array_index)
                    ),
                    expected,
                )

        with patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "1003"}, clear=True):
            self.assertEqual(
                formal._resolve_run_target(_arguments()),
                ("high", 3, 1003),
            )
            with self.assertRaises(ValueError):
                formal._resolve_run_target(
                    _arguments(cell="low", session_index=3)
                )
            with self.assertRaises(ValueError):
                formal._resolve_run_target(_arguments(array_index=1004))

    def test_type7_and_undefined_mispricing_are_explicit(self) -> None:
        """Undefined mispricing is not silently dropped from the formal mean.

        未定义 mispricing 不得被悄悄排除后伪装成正式均值。
        """

        self.assertAlmostEqual(formal._type7((0.0, 10.0), 0.01), 0.1)
        self.assertAlmostEqual(formal._type7((0.0, 10.0), 0.50), 5.0)
        self.assertAlmostEqual(formal._type7((0.0, 10.0), 0.99), 9.9)

        incomplete_population = formal._summary(
            "reported_average_mispricing",
            (1.0, None, 3.0),
        )
        self.assertEqual(incomplete_population.total_session_count, 3)
        self.assertEqual(incomplete_population.defined_session_count, 2)
        self.assertEqual(incomplete_population.undefined_session_count, 1)
        self.assertIsNone(incomplete_population.formal_all_session_mean)
        self.assertEqual(incomplete_population.mean_of_defined_sessions, 2.0)
        self.assertFalse(incomplete_population.undefined_values_imputed)
        self.assertIn("type 7", incomplete_population.quantile_method)

        complete_population = formal._summary("delta_c_unclamped", (1.0, 3.0))
        self.assertEqual(complete_population.formal_all_session_mean, 2.0)

    def test_exclusive_lock_conflicts_and_removes_only_its_own_token(self) -> None:
        """Two workers cannot own one session; token ownership controls cleanup.

        两个 worker 不能独占同一 session；cleanup 由 token 所有权控制。
        """

        task = SimpleNamespace(
            relative_artifact_directory="sessions/session_0000_fake",
            task_id="fake-task",
            task_sha256="f" * 64,
        )
        with _owned_workspace_directory("lock") as root:
            lock_path = formal._worker_lock_path(root, task)
            with formal._exclusive_session_claim(root, task):
                self.assertTrue(lock_path.is_file())
                with self.assertRaises(formal.SessionLockConflictError):
                    with formal._exclusive_session_claim(root, task):
                        pass
            self.assertFalse(lock_path.exists())

            with formal._exclusive_session_claim(root, task):
                record = json.loads(lock_path.read_text(encoding="utf-8"))
                record["claim_token"] = "owned-by-a-different-worker"
                lock_path.write_text(json.dumps(record), encoding="utf-8")
            self.assertTrue(lock_path.is_file())
            lock_path.unlink()

    def test_period_budget_is_forwarded_without_changing_plan(self) -> None:
        """The invocation budget reaches Step 36E as an operational argument.

        单次调用 budget 只作为操作参数传给 Step 36E。
        """

        task = SimpleNamespace(task_id="task", task_sha256="a" * 64)
        plan = SimpleNamespace(
            tasks=(task,),
            execution_policy=SimpleNamespace(maximum_training_periods=None),
        )
        scientific = SimpleNamespace(
            status=formal.INCOMPLETE,
            phase="training",
            stop_reason="training_not_yet_converged",
            training_periods_verified=123,
            committed_measurement_rows=0,
        )
        execution = SimpleNamespace(
            status=SimpleNamespace(scientific_outcome=scientific),
            evidence=None,
        )
        with (
            patch.object(formal, "_load_cell", return_value=(plan, Path("cell"))),
            patch.object(formal, "_exclusive_session_claim") as claim,
            patch.object(
                formal,
                "run_complete_measurement_task",
                return_value=execution,
            ) as run,
        ):
            claim.return_value.__enter__.return_value = Path("lock")
            result = formal.run_one_session(
                Path("root"),
                cell="low",
                session_index=0,
                checkpoint_interval_periods=99,
                invocation_period_budget=17,
                retry_failed=False,
            )
        self.assertEqual(result["status"], formal.INCOMPLETE)
        self.assertEqual(
            run.call_args.kwargs["invocation_training_period_budget"],
            17,
        )
        self.assertIsNone(plan.execution_policy.maximum_training_periods)

    def test_main_returns_complete_incomplete_and_failed_exit_codes(self) -> None:
        """Scheduler-visible exit codes distinguish outcomes. / 调度器可见退出码能区分结果。"""

        base = {
            "cell": "low",
            "session_index": 0,
            "task_id": "task",
            "phase": "training",
            "stop_reason": "test",
            "training_periods_verified": 1,
            "measurement_rows_committed": 0,
            "complete_evidence_available": False,
        }
        command = [
            "run-session",
            "--artifact-root",
            ".",
            "--cell",
            "low",
            "--session-index",
            "0",
            "--period-budget",
            "1",
        ]
        with patch.dict(os.environ, {}, clear=True):
            for status, expected in (
                (formal.COMPLETE, formal.EXIT_COMPLETE),
                (formal.INCOMPLETE, formal.EXIT_INCOMPLETE),
                (formal.FAILED, formal.EXIT_FAILED),
            ):
                with (
                    patch.object(
                        formal,
                        "run_one_session",
                        return_value={**base, "status": status},
                    ),
                    redirect_stdout(StringIO()),
                ):
                    self.assertEqual(formal.main(command), expected)

    def test_array_worker_requires_an_operational_period_budget(self) -> None:
        """Array jobs must yield before scheduler walltime. / array 任务必须在调度时限前主动退出。"""

        with (
            patch.dict(os.environ, {}, clear=True),
            redirect_stdout(StringIO()),
            patch("sys.stderr", StringIO()),
        ):
            with self.assertRaises(SystemExit) as stopped:
                formal.main(["run-session", "--array-index", "0"])
        self.assertEqual(stopped.exception.code, 2)

        outcome = {
            "cell": "low",
            "session_index": 0,
            "status": formal.INCOMPLETE,
        }
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(formal, "run_one_session", return_value=outcome) as run,
            redirect_stdout(StringIO()),
        ):
            exit_code = formal.main(
                [
                    "run-session",
                    "--array-index",
                    "0",
                    "--period-budget",
                    "500",
                ]
            )
        self.assertEqual(exit_code, formal.EXIT_INCOMPLETE)
        self.assertEqual(run.call_args.kwargs["invocation_period_budget"], 500)

    def test_collect_refuses_missing_complete_evidence(self) -> None:
        """Collection cannot promote a planned-but-missing cell. / 汇总不能提升只有计划却缺证据的 cell。"""

        tasks = tuple(SimpleNamespace() for _ in range(1_000))
        plan = SimpleNamespace(tasks=tasks)
        with (
            patch.object(formal, "load_family_manifest", return_value={}),
            patch.object(
                formal,
                "_load_cell",
                return_value=(plan, Path("missing-cell")),
            ),
            patch.object(
                formal,
                "load_completed_measurement_evidence",
                side_effect=FileNotFoundError("missing"),
            ),
        ):
            with self.assertRaises(FileNotFoundError):
                formal.collect_cell(Path("root"), "low")


if __name__ == "__main__":
    unittest.main(verbosity=2)
