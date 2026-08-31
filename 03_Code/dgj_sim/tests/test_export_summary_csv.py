"""Step 37A must export accepted session metrics without changing evidence."""

from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np

import _setup  # noqa: F401
from dgj.config import ExperimentCell, PaperParameters
from hpc import export_summary_csv


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".test_work"


@contextmanager
def writable_temp_directory():
    """Avoid Windows tempfile's restrictive sandbox ACL. / 避免 Windows 临时目录 ACL。"""

    TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    root = TEST_TEMP_ROOT / f"export_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        yield str(root)
    finally:
        shutil.rmtree(root)
        try:
            TEST_TEMP_ROOT.rmdir()
        except OSError:
            # Another test may still own a sibling directory. / 其他测试可能仍在使用同级目录。
            pass


def _summary(rows: list[dict]) -> dict:
    deltas = np.asarray([row["delta_c"] for row in rows], dtype=np.float64)
    chi = np.asarray([row["chi_hat"] for row in rows], dtype=np.float64)
    information = np.asarray(
        [row["price_informativeness"] for row in rows], dtype=np.float64
    )
    converged = np.asarray([row["converged_at"] for row in rows], dtype=np.float64)
    liquidity = np.asarray([row["liquidity"] for row in rows], dtype=np.float64)
    mispricing = np.asarray([row["mispricing"] for row in rows], dtype=np.float64)
    current = sum(row["provenance_class"] == "current_schema" for row in rows)
    legacy = len(rows) - current
    mechanisms = sum(row["mechanism"] is not None for row in rows)
    return {
        "sessions": len(rows),
        "delta_c_mean": float(deltas.mean()),
        "delta_c_p01": float(np.percentile(deltas, 1)),
        "delta_c_p99": float(np.percentile(deltas, 99)),
        "chi_hat_mean": float(chi.mean()),
        "chi_nash": 166.6666,
        "chi_cartel": 124.9999,
        "informativeness_mean": float(information.mean()),
        "converged_at_median": float(np.median(converged)),
        "liquidity_mean": float(liquidity.mean()),
        "mispricing_mean": float(mispricing.mean()),
        "censored_sessions": 0,
        "legacy_unversioned_sessions": legacy,
        "current_schema_sessions": current,
        "cohort_provenance": (
            "mixed_legacy_unversioned_and_current_schema"
            if legacy
            else "homogeneous_current_schema"
        ),
        "analysis_identity": {
            "analysis_source_fingerprint": "a" * 64,
            "python_version": "3.11.5",
            "numpy_version": "2.4.2+computecanada",
            "numba_version": "0.65.1+computecanada",
            "numba_enabled": True,
            "platform_system": "Linux",
            "platform_machine": "x86_64",
        },
        "mechanism_shares": (
            {
                name: sum(row["mechanism"] == name for row in rows) / len(rows)
                for name in export_summary_csv.MECHANISM_NAMES
            }
            if mechanisms == len(rows)
            else None
        ),
        "mechanism_sessions": mechanisms,
        "per_session": rows,
    }


def _row(index: int, *, delta: float, provenance: str = "current_schema") -> dict:
    return {
        "file": f"session_{index:04d}.npz",
        "provenance_class": provenance,
        "mechanism": None,
        "liquidity": 10.0 + index,
        "mispricing": 0.2 + index / 100,
        "converged_at": 1_000_000_000 + index,
        "delta_c": delta,
        "profit_gain_vs_nash": 0.5 + index / 10,
        "chi_hat": 140.0 + index,
        "price_informativeness": 7.0 + index / 10,
        "mean_lambda_hat": 0.002 + index / 10_000,
    }


def _cell_key(cell: dict) -> str:
    payload = dict(cell)
    payload.pop("label")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


class ExportFixture:
    def __init__(self, root: str, *, rows: list[dict] | None = None) -> None:
        self.root = Path(root)
        self.summary_path = self.root / "summary.json"
        self.cell_path = self.root / "cell.json"
        self.rows = rows or [_row(0, delta=0.72), _row(1, delta=0.76)]
        cell = ExperimentCell(
            parameters=PaperParameters(noise_std=100.0),
            label="high_noise",
            prehistory="nash",
            price_grid="per_value",
        )
        self.cell = cell
        self.cell_document = {
            "artifact_schema_version": 1,
            "cell": cell.to_dict(),
            "cell_key": cell.key(),
            "experiment_seed": 20260828,
            "training_chunk_size": 1_000_000,
            "scientific_engine_version": 1,
            "scientific_source_fingerprint": "b" * 64,
            "python_version": "3.11.5",
            "numpy_version": "2.4.2+computecanada",
            "numba_version": "0.65.1+computecanada",
            "numba_enabled": True,
            "platform_system": "Linux",
            "platform_machine": "x86_64",
        }
        self.write()

    def write(self, summary: dict | None = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        document = _summary(self.rows) if summary is None else summary
        document.setdefault("summary_schema_version", 1)
        document.setdefault("cell_key", self.cell_document["cell_key"])
        document.setdefault("cell", self.cell_document["cell"])
        document.setdefault("experiment_seed", self.cell_document["experiment_seed"])
        document.setdefault(
            "training_chunk_size", self.cell_document["training_chunk_size"]
        )
        with self.summary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
        with self.cell_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.cell_document, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def export(self) -> dict:
        return export_summary_csv.export_summary(
            self.summary_path,
            expected_sessions=2,
            expected_noise_std=100.0,
        )


class TestExportSummaryCsv(unittest.TestCase):
    def test_sorted_lossless_export_receipt_and_idempotent_rerun(self):
        with writable_temp_directory() as root:
            fixture = ExportFixture(
                root,
                rows=[_row(1, delta=0.76), _row(0, delta=0.72)],
            )
            result = fixture.export()
            self.assertEqual(result["csv_status"], "published")
            self.assertEqual(result["receipt_status"], "published")

            csv_path = Path(result["csv_path"])
            receipt_path = Path(result["receipt_path"])
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["session_index"] for row in rows], ["0", "1"])
            self.assertEqual([row["source_file"] for row in rows], [
                "session_0000.npz",
                "session_0001.npz",
            ])
            self.assertEqual([float(row["delta_c"]) for row in rows], [0.72, 0.76])
            self.assertEqual([row["mechanism"] for row in rows], ["", ""])
            self.assertTrue(all(row["noise_std"] == "100" for row in rows))

            with receipt_path.open(encoding="utf-8") as handle:
                receipt = json.load(handle)
            self.assertEqual(receipt["row_count"], 2)
            self.assertEqual(receipt["columns"], list(export_summary_csv.CSV_COLUMNS))
            self.assertEqual(
                receipt["csv_sha256"],
                hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                receipt["source_summary_sha256"],
                hashlib.sha256(fixture.summary_path.read_bytes()).hexdigest(),
            )

            before_csv = csv_path.read_bytes()
            before_receipt = receipt_path.read_bytes()
            rerun = fixture.export()
            self.assertEqual(rerun["csv_status"], "validated_existing")
            self.assertEqual(rerun["receipt_status"], "validated_existing")
            self.assertEqual(csv_path.read_bytes(), before_csv)
            self.assertEqual(receipt_path.read_bytes(), before_receipt)

    def test_csv_bytes_do_not_depend_on_source_row_order(self):
        with writable_temp_directory() as root:
            first = ExportFixture(
                os.path.join(root, "first"),
                rows=[_row(0, delta=0.72), _row(1, delta=0.76)],
            )
            second = ExportFixture(
                os.path.join(root, "second"),
                rows=[_row(1, delta=0.76), _row(0, delta=0.72)],
            )
            first.export()
            second.export()
            self.assertEqual(
                (first.root / export_summary_csv.DEFAULT_CSV_NAME).read_bytes(),
                (second.root / export_summary_csv.DEFAULT_CSV_NAME).read_bytes(),
            )

    def test_rejects_bad_session_identity_or_schema(self):
        cases = []
        duplicate = [_row(0, delta=0.72), _row(0, delta=0.76)]
        cases.append(("duplicate", duplicate))
        gap = [_row(0, delta=0.72), _row(2, delta=0.76)]
        cases.append(("gap", gap))
        extra = [_row(0, delta=0.72), _row(1, delta=0.76)]
        extra[0]["unexpected"] = 1
        cases.append(("extra field", extra))

        for name, rows in cases:
            with self.subTest(name=name), writable_temp_directory() as root:
                fixture = ExportFixture(root, rows=rows)
                with self.assertRaises(ValueError):
                    fixture.export()
                self.assertFalse((fixture.root / export_summary_csv.DEFAULT_CSV_NAME).exists())

    def test_rejects_nonfinite_bool_and_inconsistent_aggregates(self):
        with writable_temp_directory() as root:
            fixture = ExportFixture(root)
            bad = _summary(fixture.rows)
            bad["per_session"][0]["delta_c"] = float("nan")
            fixture.write(bad)
            with self.assertRaisesRegex(ValueError, "must be finite"):
                fixture.export()

        with writable_temp_directory() as root:
            fixture = ExportFixture(root)
            bad = _summary(fixture.rows)
            bad["per_session"][0]["converged_at"] = True
            fixture.write(bad)
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                fixture.export()

        with writable_temp_directory() as root:
            fixture = ExportFixture(root)
            bad = _summary(fixture.rows)
            bad["delta_c_mean"] += 0.01
            fixture.write(bad)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                fixture.export()

    def test_rejects_provenance_mechanism_or_cell_mismatch(self):
        with writable_temp_directory() as root:
            fixture = ExportFixture(root)
            bad = _summary(fixture.rows)
            bad["current_schema_sessions"] = 1
            bad["legacy_unversioned_sessions"] = 1
            fixture.write(bad)
            with self.assertRaisesRegex(ValueError, "provenance"):
                fixture.export()

        with writable_temp_directory() as root:
            fixture = ExportFixture(root)
            bad = _summary(fixture.rows)
            bad["mechanism_sessions"] = 1
            fixture.write(bad)
            with self.assertRaisesRegex(ValueError, "mechanism_sessions"):
                fixture.export()

        with writable_temp_directory() as root:
            fixture = ExportFixture(root)
            with self.assertRaisesRegex(ValueError, "noise_std"):
                export_summary_csv.export_summary(
                    fixture.summary_path,
                    expected_sessions=2,
                    expected_noise_std=0.1,
                )

    def test_rejects_summary_paired_with_another_cell(self):
        with writable_temp_directory() as root:
            fixture = ExportFixture(root)
            other = ExperimentCell(
                parameters=PaperParameters(noise_std=0.1),
                label="low_noise",
                prehistory="nash",
                price_grid="per_value",
            )
            bad = _summary(fixture.rows)
            bad.update({
                "summary_schema_version": 1,
                "cell_key": other.key(),
                "cell": other.to_dict(),
                "experiment_seed": 20260828,
                "training_chunk_size": 1_000_000,
            })
            fixture.write(bad)
            with self.assertRaisesRegex(ValueError, "different experiments"):
                fixture.export()

    def test_requires_complete_analysis_and_legacy_provenance(self):
        with writable_temp_directory() as root:
            fixture = ExportFixture(root)
            bad = _summary(fixture.rows)
            bad["analysis_identity"].pop("numba_version")
            fixture.write(bad)
            with self.assertRaisesRegex(ValueError, "analysis_identity schema"):
                fixture.export()

        with writable_temp_directory() as root:
            rows = [
                _row(0, delta=0.72, provenance="legacy_unversioned"),
                _row(1, delta=0.76),
            ]
            fixture = ExportFixture(root, rows=rows)
            with self.assertRaisesRegex(ValueError, "recovery provenance"):
                fixture.export()

            good = _summary(rows)
            good.update({
                "legacy_source_commit_assertion": "legacy-commit",
                "legacy_runtime_receipt": "archived-runtime.json",
                "legacy_provenance_status": "selection_hashes_only",
                "core_equivalence_status": "established_for_audited_base_commit",
            })
            fixture.write(good)
            result = fixture.export()
            with open(result["receipt_path"], encoding="utf-8") as handle:
                receipt = json.load(handle)
            self.assertEqual(
                receipt["source_recovery_provenance"]["legacy_source_commit_assertion"],
                "legacy-commit",
            )

    def test_rejects_invalid_parameter_types_extra_parameters_and_choices(self):
        cases = (
            ("integer type", "num_speculators", 2.0, None),
            ("extra parameter", "unexpected_parameter", 1, None),
            ("choice", None, None, ("prehistory", "invented")),
        )
        for name, parameter, value, choice in cases:
            with self.subTest(name=name), writable_temp_directory() as root:
                fixture = ExportFixture(root)
                if parameter is not None:
                    fixture.cell_document["cell"]["parameters"][parameter] = value
                if choice is not None:
                    fixture.cell_document["cell"][choice[0]] = choice[1]
                fixture.cell_document["cell_key"] = _cell_key(
                    fixture.cell_document["cell"]
                )
                fixture.write()
                with self.assertRaises(ValueError):
                    fixture.export()

    def test_conflicting_existing_csv_is_never_overwritten(self):
        with writable_temp_directory() as root:
            fixture = ExportFixture(root)
            result = fixture.export()
            csv_path = Path(result["csv_path"])
            receipt_path = Path(result["receipt_path"])
            original_receipt = receipt_path.read_bytes()
            csv_path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                fixture.export()
            self.assertEqual(csv_path.read_text(encoding="utf-8"), "tampered\n")
            self.assertEqual(receipt_path.read_bytes(), original_receipt)

    def test_identical_orphan_csv_can_repair_missing_receipt(self):
        with writable_temp_directory() as root:
            fixture = ExportFixture(root)
            first = fixture.export()
            receipt_path = Path(first["receipt_path"])
            receipt_path.unlink()
            repaired = fixture.export()
            self.assertEqual(repaired["csv_status"], "validated_existing")
            self.assertEqual(repaired["receipt_status"], "published")
            self.assertTrue(receipt_path.exists())

    def test_conflicting_receipt_stops_before_missing_csv_is_created(self):
        with writable_temp_directory() as root:
            fixture = ExportFixture(root)
            first = fixture.export()
            csv_path = Path(first["csv_path"])
            receipt_path = Path(first["receipt_path"])
            csv_path.unlink()
            receipt_path.write_text("conflicting receipt\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                fixture.export()
            self.assertFalse(csv_path.exists())


if __name__ == "__main__":
    unittest.main()
