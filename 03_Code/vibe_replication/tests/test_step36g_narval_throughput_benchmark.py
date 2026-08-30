"""Focused tests for the Step-36G Narval throughput benchmark.

Step 36G Narval 吞吐率 benchmark 的专项测试。

The tests mock the expensive market loop.  They test orchestration and audit
boundaries; they do not pretend to measure Narval from a Windows laptop.
/ 测试会替换昂贵市场循环，只验证调度与审计边界；不会假装 Windows 笔记本测到了
Narval 的速度。
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch
from uuid import uuid4
import json
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = PROJECT_ROOT / "steps"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.source_manifests import EXECUTION_SOURCE_FILES, RESULT_PIPELINE_SOURCE_FILES
import steps.step_36g_narval_throughput_benchmark as step36g


TEST_ARTIFACT_PARENT = PROJECT_ROOT / "results" / "step36g_test_artifacts"


@contextmanager
def _owned_test_path(label: str):
    """Yield one non-existing, uniquely owned path and clean it safely.

    提供一个尚不存在、由本测试独占的路径，并在结束后安全清理。
    """

    TEST_ARTIFACT_PARENT.mkdir(parents=True, exist_ok=True)
    path = TEST_ARTIFACT_PARENT / f"{label}-{uuid4().hex}"
    try:
        yield path
    finally:
        if path.exists():
            resolved = path.resolve()
            if resolved.parent != TEST_ARTIFACT_PARENT.resolve():
                raise AssertionError("Refusing unsafe cleanup. / 拒绝不安全清理。")
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
                    raise AssertionError("Unexpected artifact type. / 意外的 artifact 类型。")
            resolved.rmdir()


def _environment(*, slurm: bool, clean: bool = True, cpus: str = "1") -> step36g.BenchmarkEnvironment:
    slurm_items = (
        (
            ("SLURM_JOB_ID", "12345"),
            ("SLURM_CPUS_PER_TASK", cpus),
            ("SLURM_JOB_NODELIST", "fake-node"),
        )
        if slurm
        else ()
    )
    return step36g.BenchmarkEnvironment(
        hostname="fake-node",
        operating_system="Linux",
        operating_system_release="test",
        machine="x86_64",
        processor="fake-cpu",
        python_executable="/fake/python",
        python_version="3.13.2",
        python_implementation="CPython",
        numpy_version=step36g.PINNED_NUMPY_VERSION,
        logical_cpu_count=1,
        process_id=111,
        git_commit="a" * 40,
        git_tracked_files_clean=clean,
        loaded_modules="python/3.13.2",
        slurm=slurm_items,
        thread_limits=tuple((key, "1") for key in step36g.THREAD_ENVIRONMENT_KEYS),
    )


def _attempt(
    *,
    number: int,
    start_mode: str,
    start: int,
    budget: int,
    elapsed: float,
    input_checkpoint,
):
    return SimpleNamespace(
        attempt_number=number,
        start_mode=start_mode,
        starting_training_period=start,
        successful_periods_this_attempt=budget,
        ending_verified_period=start + budget,
        invocation_period_budget=budget,
        elapsed_seconds=elapsed,
        periods_per_second=budget / elapsed,
        input_checkpoint=input_checkpoint,
    )


def _status(attempt, *, latest_checkpoint):
    return SimpleNamespace(
        scientific_outcome=SimpleNamespace(
            status=step36g.INCOMPLETE,
            stop_reason="invocation_period_budget_reached",
            training_stage_only=True,
            measurement_periods_completed=0,
            latest_mid_training_checkpoint=latest_checkpoint,
        ),
        attempt=attempt,
    )


def _outer(cell: str, ending_period: int):
    return {
        "cell": cell,
        "session_index": 0,
        "task_id": "task-0",
        "status": step36g.INCOMPLETE,
        "phase": "training",
        "stop_reason": "training_not_yet_converged",
        "training_periods_verified": ending_period,
        "measurement_rows_committed": 0,
        "complete_evidence_available": False,
    }


class NarvalThroughputBenchmarkTests(unittest.TestCase):
    """Test fail-closed operational behavior without running Q-learning.

    不实际运行 Q-learning，测试遇到不一致时是否会拒绝继续。
    """

    def test_non_slurm_requires_explicit_small_local_override(self) -> None:
        local = _environment(slurm=False)
        with self.assertRaises(RuntimeError):
            step36g.validate_execution_environment(
                local,
                allow_non_slurm=False,
                total_period_budget=2,
            )
        step36g.validate_execution_environment(
            local,
            allow_non_slurm=True,
            total_period_budget=step36g.LOCAL_SAFETY_MAX_PERIODS,
        )
        with self.assertRaises(RuntimeError):
            step36g.validate_execution_environment(
                local,
                allow_non_slurm=True,
                total_period_budget=step36g.LOCAL_SAFETY_MAX_PERIODS + 1,
            )

    def test_slurm_requires_one_cpu_thread_limits_and_clean_source(self) -> None:
        step36g.validate_execution_environment(
            _environment(slurm=True),
            allow_non_slurm=False,
            total_period_budget=11_000,
        )
        with self.assertRaises(RuntimeError):
            step36g.validate_execution_environment(
                _environment(slurm=True, cpus="2"),
                allow_non_slurm=False,
                total_period_budget=11_000,
            )
        with self.assertRaises(RuntimeError):
            step36g.validate_execution_environment(
                _environment(slurm=True, clean=False),
                allow_non_slurm=False,
                total_period_budget=11_000,
            )
        bad_threads = replace(
            _environment(slurm=True),
            thread_limits=(("OMP_NUM_THREADS", "2"),),
        )
        with self.assertRaises(RuntimeError):
            step36g.validate_execution_environment(
                bad_threads,
                allow_non_slurm=False,
                total_period_budget=11_000,
            )

    def test_output_root_must_be_new_and_separate_from_production(self) -> None:
        with self.assertRaises(ValueError):
            step36g.prepare_isolated_roots(step36g.formal.DEFAULT_ARTIFACT_ROOT)
        with self.assertRaises(ValueError):
            step36g.prepare_isolated_roots(
                step36g.formal.DEFAULT_ARTIFACT_ROOT / "nested-benchmark"
            )
        with self.assertRaises(ValueError):
            step36g.prepare_isolated_roots(
                step36g.formal.DEFAULT_ARTIFACT_ROOT.parent
            )
        with _owned_test_path("root") as root:
            root.mkdir()
            with self.assertRaises(FileExistsError):
                step36g.prepare_isolated_roots(root)

    def test_mocked_run_uses_fresh_then_exact_resume_and_saves_receipt(self) -> None:
        warmup = 3
        measured = 7
        family = {
            "family_manifest_sha256": "e" * 64,
            "execution_source_sha256": "f" * 64,
            "result_pipeline_source_sha256": "0" * 64,
        }
        warmup_checkpoint = SimpleNamespace(
            relative_path="sessions/session_0000/checkpoint.bin",
            period_number=warmup,
            checkpoint_sha256="9" * 64,
        )
        measured_checkpoint = SimpleNamespace(
            relative_path="sessions/session_0000/checkpoint-2.bin",
            period_number=warmup + measured,
            checkpoint_sha256="8" * 64,
        )
        warmup_status = _status(
            _attempt(
                number=1,
                start_mode=step36g.FRESH_START,
                start=0,
                budget=warmup,
                elapsed=1.5,
                input_checkpoint=None,
            ),
            latest_checkpoint=warmup_checkpoint,
        )
        measured_status = _status(
            _attempt(
                number=2,
                start_mode=step36g.RESUMED_START,
                start=warmup,
                budget=measured,
                elapsed=2.0,
                input_checkpoint=warmup_checkpoint,
            ),
            latest_checkpoint=measured_checkpoint,
        )
        for cell, noise_std in (("low", 0.1), ("high", 100.0)):
            with self.subTest(cell=cell), _owned_test_path(f"run-{cell}") as root:
                task = SimpleNamespace(
                    task_id=f"task-{cell}-0",
                    task_sha256="b" * 64,
                    relative_artifact_directory="sessions/session_0000",
                )
                config = SimpleNamespace(
                    parameters=SimpleNamespace(noise_std=noise_std)
                )
                plan = SimpleNamespace(
                    tasks=(task,),
                    config=config,
                    plan_sha256="c" * 64,
                    run_config_sha256="d" * 64,
                )
                with (
                    patch.object(step36g.formal, "initialize_family", return_value=family),
                    patch.object(step36g, "load_experiment_cell_plan", return_value=plan),
                    patch.object(
                        step36g.formal,
                        "run_one_session",
                        side_effect=(
                            _outer(cell, warmup),
                            _outer(cell, warmup + measured),
                        ),
                    ) as runner,
                    patch.object(
                        step36g,
                        "load_training_status",
                        side_effect=(warmup_status, measured_status),
                    ),
                    patch.object(step36g, "_peak_resident_set_megabytes", return_value=12.5),
                    patch.object(step36g, "_normalized_file_sha256", return_value="1" * 64),
                ):
                    timer = iter((0.0, 2.0, 3.0, 7.0))
                    report, report_path = step36g.run_throughput_benchmark(
                        benchmark_root=root,
                        cell=cell,
                        warmup_periods=warmup,
                        measured_periods=measured,
                        checkpoint_interval_periods=99,
                        allow_non_slurm=True,
                        environment=_environment(slurm=False),
                        clock=lambda: next(timer),
                    )

                formal_root = root / step36g.FORMAL_SANDBOX_DIRECTORY_NAME
                self.assertEqual(
                    runner.call_args_list,
                    [
                        call(
                            formal_root,
                            cell=cell,
                            session_index=0,
                            checkpoint_interval_periods=99,
                            invocation_period_budget=warmup,
                            retry_failed=False,
                        ),
                        call(
                            formal_root,
                            cell=cell,
                            session_index=0,
                            checkpoint_interval_periods=99,
                            invocation_period_budget=measured,
                            retry_failed=False,
                        ),
                    ],
                )
                self.assertEqual(report.cell, cell)
                self.assertEqual(report.noise_standard_deviation, noise_std)
                self.assertEqual(report.warmup_start_mode, step36g.FRESH_START)
                self.assertEqual(report.measured_start_mode, step36g.RESUMED_START)
                self.assertEqual(report.measured_starting_verified_period, warmup)
                self.assertEqual(report.measured_ending_verified_period, warmup + measured)
                self.assertAlmostEqual(report.measured_training_periods_per_second, 3.5)
                self.assertAlmostEqual(report.measured_end_to_end_periods_per_second, 1.75)
                self.assertEqual(report.measurement_rows_committed, 0)
                self.assertFalse(report.research_result)
                self.assertFalse(report.paper_results_ready)
                self.assertFalse(report.total_formal_runtime_known)
                self.assertEqual(report.execution_scope, "local_connection_smoke")
                self.assertFalse(report.slurm_compute_node_verified)
                self.assertIsNone(
                    report.linear_extrapolation_seconds_per_million_at_observed_rate
                )
                self.assertEqual(step36g.load_report(report_path), report)

    def test_resume_must_use_the_exact_warmup_checkpoint(self) -> None:
        """Matching scalar periods are insufficient without checkpoint identity.

        仅有相同期数还不够；必须是同一个 checkpoint 身份。
        """

        warmup_checkpoint = SimpleNamespace(
            relative_path="warmup.bin",
            period_number=3,
            checkpoint_sha256="1" * 64,
        )
        other_checkpoint = SimpleNamespace(
            relative_path="other.bin",
            period_number=3,
            checkpoint_sha256="2" * 64,
        )
        warmup_status = _status(
            _attempt(
                number=1,
                start_mode=step36g.FRESH_START,
                start=0,
                budget=3,
                elapsed=1.0,
                input_checkpoint=None,
            ),
            latest_checkpoint=warmup_checkpoint,
        )
        measured_status = _status(
            _attempt(
                number=2,
                start_mode=step36g.RESUMED_START,
                start=3,
                budget=7,
                elapsed=1.0,
                input_checkpoint=other_checkpoint,
            ),
            latest_checkpoint=other_checkpoint,
        )
        with self.assertRaises(RuntimeError):
            step36g._validate_resume_handoff(warmup_status, measured_status)

    def test_changed_report_value_fails_checksum(self) -> None:
        environment = _environment(slurm=False)
        payload = {
            "schema_version": step36g.SCHEMA_VERSION,
            "benchmark_mode": step36g.BENCHMARK_MODE,
            "execution_scope": "local_connection_smoke",
            "slurm_compute_node_verified": False,
            "cluster_name": None,
            "cell": "low",
            "noise_standard_deviation": 0.1,
            "session_index": 0,
            "task_id": "task",
            "task_sha256": "a" * 64,
            "plan_sha256": "b" * 64,
            "run_config_sha256": "c" * 64,
            "family_manifest_sha256": "d" * 64,
            "execution_source_sha256": "e" * 64,
            "result_pipeline_source_sha256": "f" * 64,
            "step36g_source_sha256": "0" * 64,
            "benchmark_root": "/benchmark",
            "isolated_formal_artifact_root": "/benchmark/sandbox",
            "warmup_period_budget": 3,
            "measured_period_budget": 7,
            "checkpoint_interval_periods": 99,
            "warmup_start_mode": step36g.FRESH_START,
            "warmup_ending_verified_period": 3,
            "warmup_training_elapsed_seconds": 1.5,
            "warmup_training_periods_per_second": 2.0,
            "warmup_end_to_end_elapsed_seconds": 2.0,
            "measured_start_mode": step36g.RESUMED_START,
            "measured_starting_verified_period": 3,
            "measured_ending_verified_period": 10,
            "measured_training_elapsed_seconds": 2.0,
            "measured_training_periods_per_second": 3.5,
            "measured_end_to_end_elapsed_seconds": 4.0,
            "measured_end_to_end_periods_per_second": 1.75,
            "linear_extrapolation_seconds_per_million_at_observed_rate": None,
            "outer_status": step36g.INCOMPLETE,
            "outer_phase": "training",
            "outer_stop_reason": "training_not_yet_converged",
            "inner_stop_reason": "invocation_period_budget_reached",
            "measurement_rows_committed": 0,
            "peak_resident_set_megabytes": None,
            "created_at_utc": "2026-08-30T00:00:00+00:00",
            "environment": environment,
            "economic_parameters_changed_for_benchmark": False,
            "convergence_observed": False,
            "research_result": False,
            "paper_results_ready": False,
            "total_formal_runtime_known": False,
        }
        report = step36g.ThroughputBenchmarkReport(
            **payload,
            report_sha256=step36g._sha256_json(
                {**payload, "environment": step36g.asdict(environment)}
            ),
        )
        step36g.validate_report(report)
        # Recomputing a checksum cannot legitimize inconsistent timing.
        # / 重算 checksum 也不能把自相矛盾的计时数据变正确。
        inconsistent = replace(
            report,
            measured_training_elapsed_seconds=4.0,
            report_sha256="",
        )
        inconsistent = replace(
            inconsistent,
            report_sha256=step36g._sha256_json(
                step36g._report_without_checksum(inconsistent)
            ),
        )
        with self.assertRaises(ValueError):
            step36g.validate_report(inconsistent)
        with _owned_test_path("tamper") as root:
            root.mkdir()
            path = step36g.save_report(report, root / step36g.REPORT_FILE_NAME)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["measured_training_periods_per_second"] = 999.0
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                step36g.load_report(path)

    def test_invalid_counts_fail_before_creating_output(self) -> None:
        """CLI-domain mistakes must not leave a benchmark directory.

        CLI 数值错误不能留下 benchmark 目录。
        """

        with _owned_test_path("invalid") as root:
            for changes in (
                {"warmup_periods": 0},
                {"measured_periods": -1},
                {"session_index": 1_000},
            ):
                arguments = {
                    "benchmark_root": root,
                    "cell": "low",
                    "warmup_periods": 3,
                    "measured_periods": 7,
                    "session_index": 0,
                    "allow_non_slurm": True,
                    "environment": _environment(slurm=False),
                }
                arguments.update(changes)
                with self.assertRaises(ValueError):
                    step36g.run_throughput_benchmark(**arguments)
                self.assertFalse(root.exists())

    def test_main_forwards_cli_arguments_and_prints_machine_json(self) -> None:
        """The Slurm-facing command line must remain wired to the runner.

        面向 Slurm 的命令行必须把参数正确交给 runner。
        """

        report = SimpleNamespace(
            cell="high",
            execution_scope="slurm_compute_benchmark",
            measured_training_periods_per_second=12.5,
            linear_extrapolation_seconds_per_million_at_observed_rate=80_000.0,
            research_result=False,
            paper_results_ready=False,
            total_formal_runtime_known=False,
        )
        output = StringIO()
        with (
            patch.object(
                step36g,
                "run_throughput_benchmark",
                return_value=(report, Path("/reports/high.json")),
            ) as runner,
            redirect_stdout(output),
        ):
            exit_code = step36g.main(
                [
                    "--benchmark-root",
                    "bench",
                    "--cell",
                    "high",
                    "--warmup-periods",
                    "25",
                    "--measured-periods",
                    "250",
                    "--checkpoint-interval",
                    "999",
                    "--session-index",
                    "4",
                ]
            )
        self.assertEqual(exit_code, 0)
        runner.assert_called_once_with(
            benchmark_root=Path("bench"),
            cell="high",
            warmup_periods=25,
            measured_periods=250,
            checkpoint_interval_periods=999,
            session_index=4,
            allow_non_slurm=False,
        )
        printed = json.loads(output.getvalue())
        self.assertEqual(printed["step"], "36G")
        self.assertEqual(printed["cell"], "high")
        self.assertFalse(printed["research_result"])

    def test_operational_files_are_outside_scientific_source_manifests(self) -> None:
        """Adding orchestration must not invalidate scientific checkpoints.

        增加调度文件不能让科学 checkpoint 无故失效。
        """

        flattened = set(EXECUTION_SOURCE_FILES) | set(RESULT_PIPELINE_SOURCE_FILES)
        self.assertNotIn("steps/step_36g_narval_throughput_benchmark.py", flattened)
        self.assertFalse(any("step_36g" in item for item in flattened))


if __name__ == "__main__":
    unittest.main()
