"""Regression tests for uncapped scientific sessions and finite job slices.

/ 验证：本次 job 可以安全暂停，但未收敛 session 绝不能被测量成正式结果。
"""

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import _setup  # noqa: F401
from dgj.config import ExperimentCell, PaperParameters
from dgj.experiments import run_session_cli as cli
from dgj.experiments.run_cell import aggregate
from dgj.game.session import Session, atomic_savez_compressed
from dgj.players.market_maker.adaptive import C_T


def command(out: str, *, convergence: int, measurement: int, work: int) -> list[str]:
    return [
        "--out", out,
        "--session", "0",
        "--seed", "314159",
        "--label", "cli_lifecycle_test",
        "--noise-std", "0.1",
        "--price-grid", "per_value",
        "--prehistory", "nash",
        "--convergence-periods", str(convergence),
        "--measurement-periods", str(measurement),
        "--work-periods", str(work),
        "--chunk-size", "5",
        "--checkpoint-every", "2",
        "--irf-paths", "0",
    ]


class TestResumableSessionCLI(unittest.TestCase):
    def test_work_budget_pauses_without_result_and_resume_is_additional(self):
        with tempfile.TemporaryDirectory() as out:
            arguments = command(out, convergence=1_000, measurement=5, work=20)
            self.assertEqual(cli.main(arguments), cli.SAFE_INCOMPLETE_EXIT)
            checkpoint = os.path.join(out, "ckpt_0000.npz")
            result = os.path.join(out, "session_0000.npz")
            progress = os.path.join(out, "progress_0000.json")
            self.assertTrue(os.path.exists(checkpoint))
            self.assertFalse(os.path.exists(result))
            with np.load(checkpoint, allow_pickle=False) as data:
                self.assertEqual(int(data["cursor"][C_T]), 20)
                self.assertEqual(str(data["phase"].item()), "training")
            with open(progress, encoding="utf-8") as handle:
                receipt = json.load(handle)
            self.assertEqual(receipt["status"], "incomplete")
            self.assertEqual(receipt["periods_completed"], 20)
            self.assertEqual(receipt["stop_reason"], "work_budget_reached")

            # The second invocation adds another 20 periods; it is not stuck at
            # an absolute 20-period cap. / 第二次是在 20 的基础上再跑 20。
            self.assertEqual(cli.main(arguments), cli.SAFE_INCOMPLETE_EXIT)
            resumed = Session(
                ExperimentCell(
                    parameters=PaperParameters(
                        noise_std=0.1,
                        convergence_periods=1_000,
                        measurement_periods=5,
                    ),
                    label="cli_lifecycle_test",
                    prehistory="nash",
                    price_grid="per_value",
                ),
                0,
                314159,
            )
            resumed.load_checkpoint(checkpoint, expected_training_chunk_size=5)
            self.assertEqual(resumed.periods_completed, 40)

            uninterrupted = Session(resumed.cell, 0, 314159)
            self.assertFalse(uninterrupted.train(chunk_size=5, max_periods=40))
            for name in ("Q", "visits", "policy", "cursor", "hist", "stats"):
                self.assertTrue(
                    np.array_equal(
                        getattr(resumed.state, name),
                        getattr(uninterrupted.state, name),
                    ),
                    name,
                )
            self.assertEqual(resumed.streams.state(), uninterrupted.streams.state())
            self.assertFalse(os.path.exists(result))

    def test_only_genuine_convergence_writes_result(self):
        with tempfile.TemporaryDirectory() as out:
            arguments = command(out, convergence=1, measurement=5, work=100)
            self.assertEqual(cli.main(arguments), 0)
            checkpoint = os.path.join(out, "ckpt_0000.npz")
            result = os.path.join(out, "session_0000.npz")
            self.assertFalse(os.path.exists(checkpoint))
            self.assertTrue(os.path.exists(result))
            with np.load(result, allow_pickle=False) as data:
                self.assertGreaterEqual(int(data["converged_at"].item()), 0)
                self.assertEqual(data["rows"].shape, (5, 10))
                manifest = json.loads(str(data["manifest"].item()))
            self.assertFalse(manifest["censored"])
            self.assertIsNone(manifest["runner"]["scientific_cumulative_period_cap"])
            parsed = cli._parser().parse_args(arguments)
            summary = aggregate(cli.build_cell(parsed), out)
            self.assertEqual(summary["sessions"], 1)
            self.assertEqual(summary["censored_sessions"], 0)
            with open(os.path.join(out, "progress_0000.json"), encoding="utf-8") as handle:
                receipt = json.load(handle)
            self.assertEqual(receipt["status"], "complete")

            # Simulate interruption after result publication but before cleanup.
            # Resubmission must delete the stale checkpoint and repair progress.
            checkpoint = os.path.join(out, "ckpt_0000.npz")
            with open(checkpoint, "wb") as handle:
                handle.write(b"stale")
            with open(os.path.join(out, "progress_0000.json"), "w", encoding="utf-8") as handle:
                json.dump({"status": "incomplete"}, handle)

            # A valid finished result is a safe, receipt-repairing no-op.
            self.assertEqual(cli.main(arguments), 0)
            self.assertFalse(os.path.exists(checkpoint))
            with open(os.path.join(out, "progress_0000.json"), encoding="utf-8") as handle:
                repaired = json.load(handle)
            self.assertEqual(repaired["status"], "complete")
            self.assertEqual(repaired["stop_reason"], "validated_existing_result")

    def test_checkpoint_identity_and_chunk_size_are_enforced(self):
        with tempfile.TemporaryDirectory() as out:
            arguments = command(out, convergence=1_000, measurement=5, work=20)
            self.assertEqual(cli.main(arguments), cli.SAFE_INCOMPLETE_EXIT)
            checkpoint = os.path.join(out, "ckpt_0000.npz")
            cell = ExperimentCell(
                parameters=PaperParameters(
                    noise_std=0.1,
                    convergence_periods=1_000,
                    measurement_periods=5,
                ),
                label="cli_lifecycle_test",
                prehistory="nash",
                price_grid="per_value",
            )
            with self.assertRaisesRegex(ValueError, "session_index"):
                Session(cell, 1, 314159).load_checkpoint(
                    checkpoint,
                    expected_training_chunk_size=5,
                )
            with self.assertRaisesRegex(ValueError, "training_chunk_size"):
                Session(cell, 0, 314159).load_checkpoint(
                    checkpoint,
                    expected_training_chunk_size=10,
                )

            # A checkpoint cannot silently cross a scientific-code revision.
            with np.load(checkpoint, allow_pickle=False) as data:
                payload = {name: np.array(data[name], copy=True) for name in data.files}
            payload["scientific_source_fingerprint"] = np.array("not-the-current-code")
            atomic_savez_compressed(checkpoint, **payload)
            with self.assertRaisesRegex(ValueError, "scientific_source_fingerprint"):
                Session(cell, 0, 314159).load_checkpoint(
                    checkpoint,
                    expected_training_chunk_size=5,
                )

    def test_overlapping_owner_is_rejected(self):
        with tempfile.TemporaryDirectory() as out:
            arguments = command(out, convergence=1_000, measurement=5, work=20)
            lock_path = os.path.join(out, "session_0000.lock")
            with cli._exclusive_session_lock(lock_path):
                self.assertEqual(cli.main(arguments), 2)
            self.assertEqual(cli.main(arguments), cli.SAFE_INCOMPLETE_EXIT)

    def test_scheduler_stop_is_observed_after_one_chunk(self):
        with tempfile.TemporaryDirectory() as out:
            arguments = command(out, convergence=1_000, measurement=5, work=20)
            original_train = Session.train

            def train_then_request_stop(session, **kwargs):
                result = original_train(session, **kwargs)
                cli._STOP_REQUESTED = True
                return result

            with mock.patch.object(Session, "train", train_then_request_stop):
                self.assertEqual(cli.main(arguments), cli.SAFE_INCOMPLETE_EXIT)
            with open(os.path.join(out, "progress_0000.json"), encoding="utf-8") as handle:
                progress = json.load(handle)
            self.assertEqual(progress["stop_reason"], "scheduler_signal")
            self.assertEqual(progress["periods_completed"], 5)

    def test_converged_checkpoint_can_resume_measurement_after_crash(self):
        with tempfile.TemporaryDirectory() as out:
            arguments = command(out, convergence=1, measurement=5, work=100)
            with mock.patch.object(
                Session,
                "measure",
                side_effect=RuntimeError("simulated crash before measurement"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                    cli.main(arguments)
            checkpoint = os.path.join(out, "ckpt_0000.npz")
            self.assertTrue(os.path.exists(checkpoint))
            self.assertFalse(os.path.exists(os.path.join(out, "session_0000.npz")))
            with np.load(checkpoint, allow_pickle=False) as data:
                self.assertEqual(str(data["phase"].item()), "converged")

            self.assertEqual(cli.main(arguments), 0)
            self.assertFalse(os.path.exists(checkpoint))
            self.assertTrue(os.path.exists(os.path.join(out, "session_0000.npz")))

    def test_fifty_billion_review_is_enforced_before_resume(self):
        with tempfile.TemporaryDirectory() as out:
            arguments = command(out, convergence=1_000, measurement=5, work=20)
            self.assertEqual(cli.main(arguments), cli.SAFE_INCOMPLETE_EXIT)
            checkpoint = os.path.join(out, "ckpt_0000.npz")
            with np.load(checkpoint, allow_pickle=False) as data:
                payload = {name: np.array(data[name], copy=True) for name in data.files}
            cursor = np.array(payload["cursor"], copy=True)
            cursor[C_T] = cli.DIAGNOSTIC_REVIEW_PERIODS - 5
            payload["cursor"] = cursor
            atomic_savez_compressed(checkpoint, **payload)

            # The ordinary runner clamps this invocation exactly at 50B.
            self.assertEqual(cli.main(arguments), cli.SAFE_INCOMPLETE_EXIT)
            with open(os.path.join(out, "progress_0000.json"), encoding="utf-8") as handle:
                progress = json.load(handle)
            self.assertEqual(progress["periods_completed"], cli.DIAGNOSTIC_REVIEW_PERIODS)
            self.assertEqual(progress["stop_reason"], "diagnostic_review_boundary")

            # A later invocation is blocked until an operator explicitly reviews it.
            self.assertEqual(cli.main(arguments), 2)
            reviewed = arguments + ["--allow-after-diagnostic-review"]
            self.assertEqual(cli.main(reviewed), cli.SAFE_INCOMPLETE_EXIT)

    def test_legacy_censored_result_is_not_treated_as_complete(self):
        with tempfile.TemporaryDirectory() as out:
            arguments = command(out, convergence=1_000, measurement=5, work=20)
            self.assertEqual(cli.main(arguments), cli.SAFE_INCOMPLETE_EXIT)
            os.remove(os.path.join(out, "ckpt_0000.npz"))
            parsed = cli._parser().parse_args(arguments)
            session = Session(cli.build_cell(parsed), 0, 314159)
            manifest = session.manifest()
            manifest["censored"] = True
            result = os.path.join(out, "session_0000.npz")
            atomic_savez_compressed(
                result,
                rows=np.zeros((5, 10)),
                converged_at=-1,
                manifest=json.dumps(manifest),
            )
            self.assertEqual(cli.main(arguments), 2)
            self.assertTrue(os.path.exists(result))       # audit evidence preserved
            with self.assertRaisesRegex(ValueError, "censored/unconverged"):
                aggregate(session.cell, out)

    def test_existing_cell_mismatch_is_rejected_without_touching_checkpoint(self):
        with tempfile.TemporaryDirectory() as out:
            arguments = command(out, convergence=1_000, measurement=5, work=20)
            self.assertEqual(cli.main(arguments), cli.SAFE_INCOMPLETE_EXIT)
            checkpoint = os.path.join(out, "ckpt_0000.npz")
            mismatch = list(arguments)
            mismatch[mismatch.index("--noise-std") + 1] = "100"
            self.assertEqual(cli.main(mismatch), 2)
            self.assertTrue(os.path.exists(checkpoint))


if __name__ == "__main__":
    unittest.main()
