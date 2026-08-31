"""Step 37B must describe accepted sessions without changing their evidence."""

from contextlib import contextmanager
from contextlib import redirect_stderr
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import unittest
import uuid

import _setup  # noqa: F401
from dgj.config import ExperimentCell, PaperParameters
from hpc import export_summary_csv, summarize_session_metrics


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".test_work"


@contextmanager
def writable_temp_directory():
    """Use a repo-local temporary directory on restricted Windows hosts."""

    TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    root = TEST_TEMP_ROOT / f"describe_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root)
        try:
            TEST_TEMP_ROOT.rmdir()
        except OSError:
            pass


class Step37AFixture:
    def __init__(
        self,
        root: Path,
        *,
        order: list[int] | None = None,
        mechanisms: list[str] | None = None,
    ) -> None:
        self.root = root
        self.csv_path = root / export_summary_csv.DEFAULT_CSV_NAME
        self.receipt_path = root / export_summary_csv.DEFAULT_RECEIPT_NAME
        self.output_dir = root / "step37b_high_noise"
        self.count = 4
        order = list(range(self.count)) if order is None else order
        mechanisms = [""] * self.count if mechanisms is None else mechanisms
        cell = ExperimentCell(
            parameters=PaperParameters(noise_std=100.0),
            label="high_noise",
            prehistory="nash",
            price_grid="per_value",
        )
        cell_dict = cell.to_dict()
        rows: list[dict[str, object]] = []
        for index in order:
            metric_scale = float(index)
            row: dict[str, object] = {
                "cell_key": cell.key(),
                "label": "high_noise",
                "experiment_seed": 20260828,
                "session_index": index,
                "source_file": f"session_{index:04d}.npz",
                **cell_dict["parameters"],
                "prehistory": cell_dict["prehistory"],
                "price_mapping": cell_dict["price_mapping"],
                "price_grid": cell_dict["price_grid"],
                "training_tie_rule": cell_dict["training_tie_rule"],
                "measurement_tie_rule": cell_dict["measurement_tie_rule"],
                "provenance_class": "current_schema",
                "mechanism": mechanisms[index],
                "converged_at": 100 + index,
                "delta_c": metric_scale,
                "profit_gain_vs_nash": 10.0 + metric_scale,
                "chi_hat": 140.0 + metric_scale,
                "price_informativeness": 7.0 + metric_scale,
                "liquidity": 20.0 + metric_scale,
                "mispricing": 0.2 + metric_scale,
                "mean_lambda_hat": 0.002 + metric_scale / 1000,
            }
            self.assert_complete_row(row)
            rows.append(row)
        self.write(rows, cell.key())

    @staticmethod
    def assert_complete_row(row: dict[str, object]) -> None:
        missing = set(export_summary_csv.CSV_COLUMNS) - set(row)
        extra = set(row) - set(export_summary_csv.CSV_COLUMNS)
        if missing or extra:
            raise AssertionError(f"fixture schema mismatch: missing={missing}, extra={extra}")

    def write(self, rows: list[dict[str, object]], cell_key: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=export_summary_csv.CSV_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        payload = self.csv_path.read_bytes()
        receipt = {
            "export_schema_version": 1,
            "row_count": self.count,
            "columns": list(export_summary_csv.CSV_COLUMNS),
            "session_index_min": 0,
            "session_index_max": self.count - 1,
            "cell_key": cell_key,
            "label": "high_noise",
            "noise_std": 100.0,
            "experiment_seed": 20260828,
            "source_scientific_identity": {
                "scientific_engine_version": 1,
                "scientific_source_fingerprint": "b" * 64,
                "python_version": "3.11.5",
                "numpy_version": "2.4.2+computecanada",
                "numba_version": "0.65.1+computecanada",
                "numba_enabled": True,
                "platform_system": "Linux",
                "platform_machine": "x86_64",
            },
            "cohort_provenance": "homogeneous_current_schema",
            "source_analysis_identity": {
                "analysis_source_fingerprint": "c" * 64,
                "python_version": "3.11.5",
                "numpy_version": "2.4.2+computecanada",
                "numba_version": "0.65.1+computecanada",
                "numba_enabled": True,
                "platform_system": "Linux",
                "platform_machine": "x86_64",
            },
            "source_recovery_provenance": {},
            "source_summary_file": "summary.json",
            "source_summary_sha256": "a" * 64,
            "source_summary_bytes": 12345,
            "source_cell_file": "cell.json",
            "source_cell_sha256": "d" * 64,
            "source_cell_bytes": 1234,
            "csv_file": self.csv_path.name,
            "csv_sha256": hashlib.sha256(payload).hexdigest(),
            "csv_bytes": len(payload),
            "csv_encoding": "UTF-8 without BOM",
            "csv_line_ending": "LF",
            "null_encoding": "empty CSV field",
            "exporter_source_sha256": hashlib.sha256(
                Path(export_summary_csv.__file__).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest(),
            "publication_rule": "immutable_identical_rerun_only",
        }
        self.receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def analyze(self) -> dict:
        return summarize_session_metrics.summarize_session_metrics(
            self.csv_path,
            receipt_path=self.receipt_path,
            output_dir=self.output_dir,
            expected_sessions=self.count,
            expected_noise_std=100.0,
            expected_label="high_noise",
        )


class TestSummarizeSessionMetrics(unittest.TestCase):
    def test_accepts_the_real_step37a_exporter_receipt(self):
        from test_export_summary_csv import ExportFixture, _row

        with writable_temp_directory() as root:
            exported = ExportFixture(
                str(root),
                rows=[_row(0, delta=0.72), _row(1, delta=0.76)],
            ).export()
            result = summarize_session_metrics.summarize_session_metrics(
                Path(exported["csv_path"]),
                receipt_path=Path(exported["receipt_path"]),
                output_dir=root / "step37b",
                expected_sessions=2,
                expected_noise_std=100.0,
                expected_label="high_noise",
            )
            self.assertEqual(result["row_count"], 2)
            self.assertEqual(result["mechanism_evidence"]["status"], "unavailable")

    def test_hand_statistics_ecdf_receipt_and_idempotent_rerun(self):
        with writable_temp_directory() as root:
            fixture = Step37AFixture(root)
            source_csv_before = fixture.csv_path.read_bytes()
            source_receipt_before = fixture.receipt_path.read_bytes()
            result = fixture.analyze()

            self.assertEqual(result["row_count"], 4)
            self.assertEqual(result["ecdf_rows"], 4 * len(summarize_session_metrics.METRICS))
            self.assertEqual(
                result["mechanism_evidence"],
                {
                    "status": "unavailable",
                    "available_sessions": 0,
                    "required_sessions": 4,
                    "shares": None,
                    "reason": (
                        "no per-session mechanism classifications are present in the "
                        "accepted Step 37A export"
                    ),
                },
            )
            self.assertTrue(
                all(file["status"] == "published" for file in result["files"].values())
            )

            with (
                fixture.output_dir / summarize_session_metrics.DEFAULT_STATISTICS_NAME
            ).open(encoding="utf-8", newline="") as handle:
                statistics_rows = {row["metric"]: row for row in csv.DictReader(handle)}
            delta = statistics_rows["delta_c"]
            self.assertEqual(int(delta["n"]), 4)
            self.assertAlmostEqual(float(delta["mean"]), 1.5)
            self.assertAlmostEqual(
                float(delta["standard_deviation"]), math.sqrt(5.0 / 3.0)
            )
            self.assertAlmostEqual(
                float(delta["standard_error"]), math.sqrt(5.0 / 3.0) / 2.0
            )
            self.assertAlmostEqual(float(delta["p01"]), 0.03)
            self.assertAlmostEqual(float(delta["p05"]), 0.15)
            self.assertAlmostEqual(float(delta["p25"]), 0.75)
            self.assertAlmostEqual(float(delta["median"]), 1.5)
            self.assertAlmostEqual(float(delta["p99"]), 2.97)

            with (
                fixture.output_dir / summarize_session_metrics.DEFAULT_ECDF_NAME
            ).open(encoding="utf-8", newline="") as handle:
                ecdf_rows = [
                    row for row in csv.DictReader(handle) if row["metric"] == "delta_c"
                ]
            self.assertEqual(
                [int(row["cumulative_count"]) for row in ecdf_rows], [1, 2, 3, 4]
            )
            self.assertEqual(
                [int(row["session_index"]) for row in ecdf_rows], [0, 1, 2, 3]
            )
            self.assertEqual(
                [float(row["ecdf"]) for row in ecdf_rows], [0.25, 0.5, 0.75, 1.0]
            )

            with (
                fixture.output_dir / summarize_session_metrics.DEFAULT_RECEIPT_NAME
            ).open(encoding="utf-8") as handle:
                receipt = json.load(handle)
            self.assertEqual(receipt["analysis_schema_version"], 1)
            self.assertEqual(receipt["row_count"], 4)
            self.assertEqual(receipt["provenance_counts"]["current_schema"], 4)
            self.assertEqual(receipt["provenance_counts"]["legacy_unversioned"], 0)
            for name, metadata in receipt["outputs"].items():
                payload = (fixture.output_dir / name).read_bytes()
                self.assertEqual(metadata["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(metadata["bytes"], len(payload))

            rerun = fixture.analyze()
            self.assertTrue(
                all(
                    file["status"] == "validated_existing"
                    for file in rerun["files"].values()
                )
            )
            self.assertEqual(fixture.csv_path.read_bytes(), source_csv_before)
            self.assertEqual(fixture.receipt_path.read_bytes(), source_receipt_before)

    def test_data_outputs_do_not_depend_on_input_row_order(self):
        with writable_temp_directory() as root:
            first = Step37AFixture(root / "first", order=[0, 1, 2, 3])
            second = Step37AFixture(root / "second", order=[3, 1, 0, 2])
            first.analyze()
            second.analyze()
            for name in (
                summarize_session_metrics.DEFAULT_STATISTICS_NAME,
                summarize_session_metrics.DEFAULT_ECDF_NAME,
                summarize_session_metrics.DEFAULT_PROVENANCE_NAME,
                summarize_session_metrics.DEFAULT_REPORT_NAME,
            ):
                self.assertEqual(
                    (first.output_dir / name).read_bytes(),
                    (second.output_dir / name).read_bytes(),
                )

    def test_tampered_csv_or_wrong_cell_expectation_is_rejected(self):
        with writable_temp_directory() as root:
            fixture = Step37AFixture(root)
            with fixture.csv_path.open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ValueError, "csv_bytes|csv_sha256"):
                fixture.analyze()

        with writable_temp_directory() as root:
            fixture = Step37AFixture(root)
            with self.assertRaisesRegex(ValueError, "expected 0.1"):
                summarize_session_metrics.summarize_session_metrics(
                    fixture.csv_path,
                    receipt_path=fixture.receipt_path,
                    output_dir=fixture.output_dir,
                    expected_sessions=4,
                    expected_noise_std=0.1,
                    expected_label="high_noise",
                )
            with self.assertRaisesRegex(ValueError, "expected 'low_noise'"):
                summarize_session_metrics.summarize_session_metrics(
                    fixture.csv_path,
                    receipt_path=fixture.receipt_path,
                    output_dir=fixture.output_dir,
                    expected_sessions=4,
                    expected_noise_std=100.0,
                    expected_label="low_noise",
                )

    def test_duplicate_index_and_nonfinite_metric_are_rejected(self):
        with writable_temp_directory() as root:
            fixture = Step37AFixture(root)
            with fixture.csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["session_index"] = "0"
            rows[1]["source_file"] = "session_0000.npz"
            fixture.write(rows, rows[0]["cell_key"])
            with self.assertRaisesRegex(ValueError, "duplicate session_index"):
                fixture.analyze()

        with writable_temp_directory() as root:
            fixture = Step37AFixture(root)
            with fixture.csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[2]["delta_c"] = "nan"
            fixture.write(rows, rows[0]["cell_key"])
            with self.assertRaisesRegex(ValueError, "must be finite"):
                fixture.analyze()

    def test_partial_mechanism_evidence_is_not_converted_to_shares(self):
        with writable_temp_directory() as root:
            fixture = Step37AFixture(
                root,
                mechanisms=["price_trigger", "", "", ""],
            )
            result = fixture.analyze()
            evidence = result["mechanism_evidence"]
            self.assertEqual(evidence["status"], "incomplete_unavailable")
            self.assertEqual(evidence["available_sessions"], 1)
            self.assertIsNone(evidence["shares"])

    def test_complete_mechanism_evidence_has_valid_shares(self):
        with writable_temp_directory() as root:
            fixture = Step37AFixture(
                root,
                mechanisms=[
                    "price_trigger",
                    "price_trigger",
                    "over_pruning",
                    "unclassified",
                ],
            )
            evidence = fixture.analyze()["mechanism_evidence"]
            self.assertEqual(evidence["status"], "available")
            self.assertEqual(evidence["shares"]["price_trigger"], 0.5)
            self.assertEqual(evidence["shares"]["over_pruning"], 0.25)
            self.assertEqual(evidence["shares"]["unclassified"], 0.25)

    def test_tied_values_share_the_true_ecdf_value(self):
        with writable_temp_directory() as root:
            fixture = Step37AFixture(root)
            with fixture.csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[2]["delta_c"] = rows[1]["delta_c"]
            fixture.write(rows, rows[0]["cell_key"])
            fixture.analyze()
            with (
                fixture.output_dir / summarize_session_metrics.DEFAULT_ECDF_NAME
            ).open(encoding="utf-8", newline="") as handle:
                delta = [
                    row for row in csv.DictReader(handle) if row["metric"] == "delta_c"
                ]
            self.assertEqual(
                [int(row["cumulative_count"]) for row in delta], [1, 3, 3, 4]
            )
            self.assertEqual(
                [float(row["ecdf"]) for row in delta], [0.25, 0.75, 0.75, 1.0]
            )

    def test_provenance_mismatch_and_truncated_receipt_are_rejected(self):
        with writable_temp_directory() as root:
            fixture = Step37AFixture(root)
            with fixture.csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["provenance_class"] = "legacy_unversioned"
            fixture.write(rows, rows[0]["cell_key"])
            with self.assertRaisesRegex(ValueError, "cohort_provenance disagrees"):
                fixture.analyze()

        with writable_temp_directory() as root:
            fixture = Step37AFixture(root)
            receipt = json.loads(fixture.receipt_path.read_text(encoding="utf-8"))
            del receipt["source_analysis_identity"]
            fixture.receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                status = summarize_session_metrics.main(
                    [
                        str(fixture.csv_path),
                        "--receipt",
                        str(fixture.receipt_path),
                        "--output-dir",
                        str(fixture.output_dir),
                        "--expected-sessions",
                        "4",
                        "--expected-noise-std",
                        "100",
                        "--expected-label",
                        "high_noise",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("STOP: Step 37A receipt is missing fields", stderr.getvalue())

    def test_identity_schema_nullable_versions_and_legacy_equivalence(self):
        with writable_temp_directory() as root:
            fixture = Step37AFixture(root)
            receipt = json.loads(fixture.receipt_path.read_text(encoding="utf-8"))
            for identity_name in (
                "source_scientific_identity",
                "source_analysis_identity",
            ):
                receipt[identity_name]["numba_version"] = None
                receipt[identity_name]["numba_enabled"] = False
            fixture.receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            self.assertEqual(fixture.analyze()["row_count"], 4)

        with writable_temp_directory() as root:
            fixture = Step37AFixture(root)
            receipt = json.loads(fixture.receipt_path.read_text(encoding="utf-8"))
            del receipt["source_scientific_identity"]["scientific_engine_version"]
            fixture.receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(ValueError, "scientific_engine_version"):
                fixture.analyze()

        with writable_temp_directory() as root:
            fixture = Step37AFixture(root)
            with fixture.csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["provenance_class"] = "legacy_unversioned"
            fixture.write(rows, rows[0]["cell_key"])
            receipt = json.loads(fixture.receipt_path.read_text(encoding="utf-8"))
            receipt["cohort_provenance"] = (
                "mixed_legacy_unversioned_and_current_schema"
            )
            receipt["source_recovery_provenance"] = {
                "legacy_source_commit_assertion": "62e54ff",
                "legacy_runtime_receipt": "archived-runtime.json",
                "legacy_provenance_status": "externally_documented",
                "core_equivalence_status": "not_established_for_asserted_commit",
            }
            fixture.receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(ValueError, "core equivalence is not established"):
                fixture.analyze()

    def test_conflict_preflight_prevents_any_overwrite(self):
        with writable_temp_directory() as root:
            fixture = Step37AFixture(root)
            fixture.analyze()
            receipt_path = (
                fixture.output_dir / summarize_session_metrics.DEFAULT_RECEIPT_NAME
            )
            receipt_before = receipt_path.read_bytes()
            statistics_path = (
                fixture.output_dir / summarize_session_metrics.DEFAULT_STATISTICS_NAME
            )
            statistics_path.write_text("conflict\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                fixture.analyze()
            self.assertEqual(statistics_path.read_text(encoding="utf-8"), "conflict\n")
            self.assertEqual(receipt_path.read_bytes(), receipt_before)


if __name__ == "__main__":
    unittest.main()
