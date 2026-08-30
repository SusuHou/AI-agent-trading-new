"""The recovery helper must preserve old evidence and omit censored results."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import tempfile
import unittest

import numpy as np

import _setup  # noqa: F401
from dgj.config import ExperimentCell, PaperParameters
from dgj.game.session import Session, atomic_savez_compressed
from hpc import aggregate_dir, prepare_recovery


class TestPrepareRecovery(unittest.TestCase):
    def test_converged_is_reused_and_censored_id_is_left_for_replay(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "old")
            target = os.path.join(root, "new")
            os.makedirs(source)
            cell = ExperimentCell(
                parameters=PaperParameters(measurement_periods=5),
                label="low_noise",
                prehistory="nash",
                price_grid="per_value",
            )
            seed = 20260828
            cell_document = {
                "cell": cell.to_dict(),
                "cell_key": cell.key(),
                "experiment_seed": seed,
            }
            with open(os.path.join(source, "cell.json"), "w", encoding="utf-8") as handle:
                json.dump(cell_document, handle)

            rows = np.zeros((5, 10))
            for index, censored in ((0, False), (1, True)):
                manifest = Session(cell, index, seed).manifest()
                manifest["censored"] = censored
                if not censored:
                    manifest["converged_at"] = 123
                    manifest["periods_completed"] = 128
                atomic_savez_compressed(
                    os.path.join(source, f"session_{index:04d}.npz"),
                    rows=rows,
                    converged_at=-1 if censored else 123,
                    manifest=json.dumps(manifest),
                )

            with redirect_stdout(io.StringIO()):
                result = prepare_recovery.main([
                    source,
                    target,
                    "--expect-sessions", "2",
                    "--expect-censored", "1",
                    "--legacy-source-commit", prepare_recovery.AUDITED_LEGACY_COMMIT,
                ])
            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(os.path.join(source, "session_0001.npz")))
            self.assertTrue(os.path.exists(os.path.join(target, "session_0000.npz")))
            self.assertFalse(os.path.exists(os.path.join(target, "session_0001.npz")))
            with open(os.path.join(target, "recovery_array.txt"), encoding="utf-8") as handle:
                self.assertEqual(handle.read().strip(), "1")
            with open(os.path.join(target, "recovery_manifest.json"), encoding="utf-8") as handle:
                receipt = json.load(handle)
            self.assertEqual(receipt["reused_converged_sessions"], 1)
            self.assertEqual(receipt["sessions_requiring_replay"], 1)
            self.assertTrue(receipt["source_preserved"])
            self.assertEqual(receipt["publication_mode"], "independent_copy")
            with open(os.path.join(target, "recovery_selection.json"), encoding="utf-8") as handle:
                selection = json.load(handle)
            self.assertEqual(selection[0]["session_index"], 0)
            self.assertEqual(selection[0]["source_sha256"], selection[0]["copied_sha256"])
            with redirect_stderr(io.StringIO()):
                aggregate_dir.verify_legacy_selection(target, ["session_0000.npz"])
            with open(os.path.join(target, "session_0000.npz"), "ab") as handle:
                handle.write(b"tamper")
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    aggregate_dir.verify_legacy_selection(target, ["session_0000.npz"])

    def test_expected_count_cannot_hide_wrong_index_set(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "old")
            target = os.path.join(root, "new")
            os.makedirs(source)
            cell = ExperimentCell(
                parameters=PaperParameters(measurement_periods=5),
                label="low_noise",
                prehistory="nash",
                price_grid="per_value",
            )
            seed = 20260828
            with open(os.path.join(source, "cell.json"), "w", encoding="utf-8") as handle:
                json.dump({
                    "cell": cell.to_dict(),
                    "cell_key": cell.key(),
                    "experiment_seed": seed,
                }, handle)
            rows = np.zeros((5, 10))
            for index, censored in ((0, False), (2, True)):
                manifest = Session(cell, index, seed).manifest()
                manifest["censored"] = censored
                if not censored:
                    manifest["converged_at"] = 123
                    manifest["periods_completed"] = 128
                atomic_savez_compressed(
                    os.path.join(source, f"session_{index:04d}.npz"),
                    rows=rows,
                    converged_at=-1 if censored else 123,
                    manifest=json.dumps(manifest),
                )

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    prepare_recovery.main([
                        source,
                        target,
                        "--expect-sessions", "2",
                        "--expect-censored", "1",
                    ])
            self.assertFalse(os.path.exists(target))


if __name__ == "__main__":
    unittest.main()
