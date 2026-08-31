"""Create a deterministic descriptive report from one Step 37A session CSV.

Step 37B is descriptive only.  It validates the immutable Step 37A export,
computes session-level summary statistics, and publishes exact ECDF data for
later figures.  It does not compare experimental cells or run a hypothesis
test. / Step 37B 只做描述性分析：验证 Step 37A 导出、计算 session 层面的统计量，
并发布供后续绘图使用的精确 ECDF 数据；此处不比较实验组，也不做假设检验。

Example / 用法::

    python hpc/summarize_session_metrics.py \
        "$HIGH_OUT/session_metrics.csv" \
        --receipt "$HIGH_OUT/session_metrics_receipt.json" \
        --expected-sessions 1000 \
        --expected-noise-std 100 \
        --expected-label high_noise \
        --output-dir "$HIGH_OUT/step37b_high_noise"

The outputs are immutable: an identical rerun is a safe no-op; different
existing bytes are never overwritten. / 输出不可变：相同重跑可安全跳过，内容不同
的已有文件绝不会被覆盖。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hpc import export_summary_csv  # noqa: E402


ANALYSIS_SCHEMA_VERSION = 1
DEFAULT_STATISTICS_NAME = "descriptive_statistics.csv"
DEFAULT_ECDF_NAME = "ecdf_data.csv"
DEFAULT_PROVENANCE_NAME = "provenance_counts.csv"
DEFAULT_REPORT_NAME = "descriptive_report.md"
DEFAULT_RECEIPT_NAME = "analysis_receipt.json"

METRICS = (
    "delta_c",
    "profit_gain_vs_nash",
    "chi_hat",
    "price_informativeness",
    "liquidity",
    "mispricing",
    "mean_lambda_hat",
    "converged_at",
)

METRIC_LABELS = {
    "delta_c": "Normalized collusion profitability (Delta C)",
    "profit_gain_vs_nash": "Profit gain relative to Nash",
    "chi_hat": "Estimated trading intensity (chi hat)",
    "price_informativeness": "Price informativeness",
    "liquidity": "Market liquidity",
    "mispricing": "Mispricing",
    "mean_lambda_hat": "Mean estimated price impact (lambda hat)",
    "converged_at": "Training periods at convergence",
}

STATISTICS_COLUMNS = (
    "metric",
    "n",
    "mean",
    "standard_deviation",
    "standard_error",
    "min",
    "p01",
    "p05",
    "p25",
    "median",
    "p75",
    "p95",
    "p99",
    "max",
)
ECDF_COLUMNS = (
    "metric",
    "cumulative_count",
    "n",
    "session_index",
    "value",
    "ecdf",
)
PROVENANCE_COLUMNS = ("provenance_class", "count", "share")

CONSTANT_COLUMNS = (
    "cell_key",
    "label",
    "experiment_seed",
    *export_summary_csv.PARAMETER_COLUMNS,
    *export_summary_csv.CELL_CHOICE_COLUMNS[1:],
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, description: str) -> tuple[dict, bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} {str(path)!r}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain one JSON object")
    return value, payload


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer_object(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_number_object(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256_string(value: object, name: str) -> str:
    result = _required_string(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex string")
    return result


def _validate_identity(
    value: object,
    name: str,
    fingerprint_field: str,
    *,
    require_engine_version: bool,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    runtime_fields = (
        "python_version",
        "numpy_version",
        "numba_version",
        "numba_enabled",
        "platform_system",
        "platform_machine",
    )
    required = (
        *(("scientific_engine_version",) if require_engine_version else ()),
        fingerprint_field,
        *runtime_fields,
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"{name} is missing fields: {missing}")
    extra = sorted(set(value) - set(required))
    if extra:
        raise ValueError(f"{name} has unexpected fields: {extra}")
    if require_engine_version:
        _positive_integer(value["scientific_engine_version"], f"{name}.scientific_engine_version")
    _sha256_string(value[fingerprint_field], f"{name}.{fingerprint_field}")
    _required_string(value["python_version"], f"{name}.python_version")
    for field in ("numpy_version", "numba_version"):
        version = value[field]
        if version is not None:
            _required_string(version, f"{name}.{field}")
    if not isinstance(value["numba_enabled"], bool):
        raise ValueError(f"{name}.numba_enabled must be boolean")
    for field in ("platform_system", "platform_machine"):
        _required_string(value[field], f"{name}.{field}")


def _canonical_integer(text: str, name: str, *, nonnegative: bool = True) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if str(value) != text:
        raise ValueError(f"{name} must use canonical integer text")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _finite_float(text: str, name: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def _normalized_source_sha256() -> str:
    return _sha256_bytes(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))


def _validate_receipt(
    receipt: dict,
    csv_path: Path,
    csv_payload: bytes,
    *,
    expected_sessions: int,
    expected_noise_std: float | None,
    expected_label: str | None,
) -> None:
    required_fields = (
        "export_schema_version",
        "row_count",
        "columns",
        "session_index_min",
        "session_index_max",
        "cell_key",
        "label",
        "noise_std",
        "experiment_seed",
        "source_scientific_identity",
        "cohort_provenance",
        "source_analysis_identity",
        "source_recovery_provenance",
        "source_summary_file",
        "source_summary_sha256",
        "source_summary_bytes",
        "source_cell_file",
        "source_cell_sha256",
        "source_cell_bytes",
        "csv_file",
        "csv_sha256",
        "csv_bytes",
        "csv_encoding",
        "csv_line_ending",
        "null_encoding",
        "exporter_source_sha256",
        "publication_rule",
    )
    missing = [name for name in required_fields if name not in receipt]
    if missing:
        raise ValueError(f"Step 37A receipt is missing fields: {missing}")
    if receipt.get("export_schema_version") != export_summary_csv.EXPORT_SCHEMA_VERSION:
        raise ValueError("Step 37A receipt uses an unsupported export_schema_version")
    if receipt.get("publication_rule") != "immutable_identical_rerun_only":
        raise ValueError("Step 37A receipt has an unexpected publication_rule")
    if receipt.get("row_count") != expected_sessions:
        raise ValueError(
            f"receipt row_count is {receipt.get('row_count')}, expected {expected_sessions}"
        )
    if receipt.get("session_index_min") != 0:
        raise ValueError("receipt session_index_min must be 0")
    if receipt.get("session_index_max") != expected_sessions - 1:
        raise ValueError("receipt session_index_max disagrees with expected sessions")
    if receipt.get("columns") != list(export_summary_csv.CSV_COLUMNS):
        raise ValueError("receipt columns do not match the Step 37A schema")
    cell_key = _required_string(receipt["cell_key"], "receipt cell_key")
    if re.fullmatch(r"[0-9a-f]{16}", cell_key) is None:
        raise ValueError("receipt cell_key must contain 16 lowercase hexadecimal characters")
    _required_string(receipt["label"], "receipt label")
    noise_std = _finite_number_object(receipt["noise_std"], "receipt noise_std")
    _nonnegative_integer_object(receipt["experiment_seed"], "receipt experiment_seed")
    _validate_identity(
        receipt["source_scientific_identity"],
        "receipt source_scientific_identity",
        "scientific_source_fingerprint",
        require_engine_version=True,
    )
    _validate_identity(
        receipt["source_analysis_identity"],
        "receipt source_analysis_identity",
        "analysis_source_fingerprint",
        require_engine_version=False,
    )
    if not isinstance(receipt["source_recovery_provenance"], dict):
        raise ValueError("receipt source_recovery_provenance must be an object")
    if receipt["cohort_provenance"] not in (
        "homogeneous_current_schema",
        "mixed_legacy_unversioned_and_current_schema",
    ):
        raise ValueError("receipt cohort_provenance is invalid")
    for prefix in ("source_summary", "source_cell"):
        file_name = _required_string(receipt[f"{prefix}_file"], f"receipt {prefix}_file")
        if Path(file_name).name != file_name:
            raise ValueError(f"receipt {prefix}_file must be a basename")
        _sha256_string(receipt[f"{prefix}_sha256"], f"receipt {prefix}_sha256")
        _positive_integer(receipt[f"{prefix}_bytes"], f"receipt {prefix}_bytes")
    _sha256_string(receipt["exporter_source_sha256"], "receipt exporter_source_sha256")
    if receipt.get("csv_file") != csv_path.name:
        raise ValueError("receipt csv_file does not match the selected CSV")
    if receipt.get("csv_bytes") != len(csv_payload):
        raise ValueError("receipt csv_bytes does not match the selected CSV")
    if receipt.get("csv_sha256") != _sha256_bytes(csv_payload):
        raise ValueError("receipt csv_sha256 does not match the selected CSV")
    if receipt.get("csv_encoding") != "UTF-8 without BOM":
        raise ValueError("receipt declares an unsupported CSV encoding")
    if receipt.get("csv_line_ending") != "LF":
        raise ValueError("receipt declares an unsupported CSV line ending")
    if receipt.get("null_encoding") != "empty CSV field":
        raise ValueError("receipt declares an unsupported null encoding")
    if expected_noise_std is not None:
        if noise_std != float(expected_noise_std):
            raise ValueError(
                f"receipt noise_std is {noise_std}, expected {expected_noise_std}"
            )
    if expected_label is not None and receipt.get("label") != expected_label:
        raise ValueError(
            f"receipt label is {receipt.get('label')!r}, expected {expected_label!r}"
        )


def _read_and_validate_rows(
    csv_path: Path,
    receipt_path: Path,
    *,
    expected_sessions: int,
    expected_noise_std: float | None,
    expected_label: str | None,
) -> tuple[list[dict], dict, bytes, bytes]:
    try:
        csv_payload = csv_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read session CSV {str(csv_path)!r}: {error}") from error
    if csv_payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError("session CSV must not contain a UTF-8 BOM")
    if b"\r" in csv_payload:
        raise ValueError("session CSV must use LF line endings")
    try:
        csv_text = csv_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("session CSV must be valid UTF-8") from error

    receipt, receipt_payload = _read_json(receipt_path, "Step 37A receipt")
    _validate_receipt(
        receipt,
        csv_path,
        csv_payload,
        expected_sessions=expected_sessions,
        expected_noise_std=expected_noise_std,
        expected_label=expected_label,
    )

    reader = csv.DictReader(io.StringIO(csv_text, newline=""))
    if reader.fieldnames != list(export_summary_csv.CSV_COLUMNS):
        raise ValueError("session CSV columns do not match the Step 37A schema")
    rows = list(reader)
    if len(rows) != expected_sessions:
        raise ValueError(f"session CSV has {len(rows)} rows, expected {expected_sessions}")

    parsed: list[dict] = []
    seen_indices: set[int] = set()
    constants: dict[str, str] | None = None
    for row_number, row in enumerate(rows, start=2):
        if None in row or set(row) != set(export_summary_csv.CSV_COLUMNS):
            raise ValueError(f"CSV row {row_number} has an invalid column count")
        session_index = _canonical_integer(
            row["session_index"], f"CSV row {row_number} session_index"
        )
        if session_index in seen_indices:
            raise ValueError(f"duplicate session_index {session_index}")
        seen_indices.add(session_index)
        expected_file = f"session_{session_index:04d}.npz"
        if row["source_file"] != expected_file:
            raise ValueError(
                f"session {session_index} source_file is {row['source_file']!r}, "
                f"expected {expected_file!r}"
            )

        row_constants = {name: row[name] for name in CONSTANT_COLUMNS}
        if constants is None:
            constants = row_constants
        elif row_constants != constants:
            changed = [name for name in CONSTANT_COLUMNS if row[name] != constants[name]]
            raise ValueError(
                f"session {session_index} changes cell-level columns: {changed}"
            )

        provenance = row["provenance_class"]
        if provenance not in export_summary_csv.PROVENANCE_CLASSES:
            raise ValueError(f"session {session_index} has invalid provenance_class")
        mechanism = row["mechanism"]
        if mechanism and mechanism not in export_summary_csv.MECHANISM_NAMES:
            raise ValueError(f"session {session_index} has invalid mechanism")

        metric_values: dict[str, int | float] = {}
        for metric in METRICS:
            if metric == "converged_at":
                metric_values[metric] = _canonical_integer(
                    row[metric], f"session {session_index} {metric}"
                )
            else:
                metric_values[metric] = _finite_float(
                    row[metric], f"session {session_index} {metric}"
                )
        parsed.append(
            {
                "session_index": session_index,
                "provenance_class": provenance,
                "mechanism": mechanism or None,
                "metrics": metric_values,
            }
        )

    expected_indices = set(range(expected_sessions))
    if seen_indices != expected_indices:
        missing = sorted(expected_indices - seen_indices)
        unexpected = sorted(seen_indices - expected_indices)
        raise ValueError(
            f"session indices are incomplete: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    assert constants is not None
    if constants["cell_key"] != receipt.get("cell_key"):
        raise ValueError("CSV cell_key disagrees with the Step 37A receipt")
    if constants["label"] != receipt.get("label"):
        raise ValueError("CSV label disagrees with the Step 37A receipt")
    experiment_seed = _canonical_integer(
        constants["experiment_seed"], "CSV experiment_seed"
    )
    if experiment_seed != receipt.get("experiment_seed"):
        raise ValueError("CSV experiment_seed disagrees with the Step 37A receipt")
    noise_std = _finite_float(constants["noise_std"], "CSV noise_std")
    if noise_std != float(receipt["noise_std"]):
        raise ValueError("CSV noise_std disagrees with the Step 37A receipt")
    if expected_noise_std is not None and noise_std != float(expected_noise_std):
        raise ValueError(
            f"CSV noise_std is {noise_std}, expected {expected_noise_std}"
        )
    if expected_label is not None and constants["label"] != expected_label:
        raise ValueError(
            f"CSV label is {constants['label']!r}, expected {expected_label!r}"
        )
    for name in export_summary_csv.PARAMETER_COLUMNS:
        _finite_float(constants[name], f"CSV constant {name}")

    legacy_count = sum(
        row["provenance_class"] == "legacy_unversioned" for row in parsed
    )
    expected_cohort_provenance = (
        "mixed_legacy_unversioned_and_current_schema"
        if legacy_count
        else "homogeneous_current_schema"
    )
    if receipt["cohort_provenance"] != expected_cohort_provenance:
        raise ValueError(
            "receipt cohort_provenance disagrees with per-session provenance"
        )
    if legacy_count:
        recovery = receipt["source_recovery_provenance"]
        required_recovery = set(export_summary_csv.RECOVERY_PROVENANCE_FIELDS)
        missing_recovery = sorted(required_recovery - set(recovery))
        if missing_recovery:
            raise ValueError(
                "legacy cohort is missing recovery provenance fields: "
                f"{missing_recovery}"
            )
        for name in (
            "legacy_source_commit_assertion",
            "legacy_runtime_receipt",
            "legacy_provenance_status",
            "core_equivalence_status",
        ):
            _required_string(recovery[name], f"receipt source_recovery_provenance.{name}")
        if recovery["core_equivalence_status"] != (
            "established_for_audited_base_commit"
        ):
            raise ValueError(
                "legacy cohort core equivalence is not established for the "
                "audited base commit"
            )

    parsed.sort(key=lambda item: item["session_index"])
    return parsed, receipt, csv_payload, receipt_payload


def _linear_quantile(sorted_values: list[float], probability: float) -> float:
    """Hyndman-Fan Type 7, NumPy ``method='linear'``. / 线性分位数。"""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0, 1]")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] + weight * (
        sorted_values[upper] - sorted_values[lower]
    )


def _statistics(metric: str, values: list[int | float]) -> dict[str, int | float | str]:
    numeric = [float(value) for value in values]
    ordered = sorted(numeric)
    count = len(numeric)
    if count < 2:
        raise ValueError("Step 37B requires at least two sessions")
    mean = math.fsum(numeric) / count
    standard_deviation = math.sqrt(
        math.fsum((value - mean) ** 2 for value in numeric) / (count - 1)
    )
    return {
        "metric": metric,
        "n": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_deviation / math.sqrt(count),
        "min": ordered[0],
        "p01": _linear_quantile(ordered, 0.01),
        "p05": _linear_quantile(ordered, 0.05),
        "p25": _linear_quantile(ordered, 0.25),
        "median": _linear_quantile(ordered, 0.50),
        "p75": _linear_quantile(ordered, 0.75),
        "p95": _linear_quantile(ordered, 0.95),
        "p99": _linear_quantile(ordered, 0.99),
        "max": ordered[-1],
    }


def _csv_payload(columns: Sequence[str], rows: Sequence[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(_format_value(row[name]) for name in columns)
    return stream.getvalue().encode("utf-8")


def _report_payload(
    *,
    label: str,
    noise_std: float,
    expected_sessions: int,
    statistics_rows: list[dict],
    mechanism_evidence: dict,
    provenance_counts: dict[str, int],
    cohort_provenance: str,
) -> bytes:
    lines = [
        f"# Step 37B descriptive report: {label}",
        "",
        f"- Accepted sessions: {expected_sessions}",
        f"- Noise standard deviation: {_format_value(noise_std)}",
        "- Analysis: descriptive only; no cross-cell test or paper comparison",
        f"- Cohort provenance: `{cohort_provenance}`",
        (
            "- Provenance counts: "
            f"current schema = {provenance_counts['current_schema']}; "
            f"legacy unversioned = {provenance_counts['legacy_unversioned']}"
        ),
        (
            "- Mechanism evidence: unavailable (no per-session mechanism "
            "classifications are present in the accepted Step 37A export)"
            if mechanism_evidence["status"] == "unavailable"
            else f"- Mechanism evidence: {mechanism_evidence['status']}"
        ),
        "",
        "| Metric | Mean | Sample SD | Median | P01 | P99 | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in statistics_rows:
        lines.append(
            "| "
            + METRIC_LABELS[row["metric"]]
            + " | "
            + " | ".join(
                _format_value(row[name])
                for name in (
                    "mean",
                    "standard_deviation",
                    "median",
                    "p01",
                    "p99",
                    "min",
                    "max",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The ECDF file retains every accepted session without trimming, "
            "winsorization, clamping, or log transformation.",
            "Figure rendering is deliberately deferred to the comparison/reporting "
            "layer; `ecdf_data.csv` is the canonical figure evidence.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _publish_immutable(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite different existing file: {path}")
        return "validated_existing"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
            status = "published"
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"concurrent process published different file: {path}")
            status = "validated_existing"
        if path.read_bytes() != payload:
            raise ValueError(f"published file failed byte validation: {path}")
        return status
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _preflight_immutable(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"refusing to overwrite different existing file: {path}")


def summarize_session_metrics(
    csv_path: Path,
    *,
    receipt_path: Path | None = None,
    output_dir: Path | None = None,
    expected_sessions: int = 1000,
    expected_noise_std: float | None = None,
    expected_label: str | None = None,
) -> dict:
    """Validate one Step 37A export and publish Step 37B. / 验证并发布 Step 37B。"""

    expected_sessions = _positive_integer(expected_sessions, "expected_sessions")
    csv_path = csv_path.resolve()
    receipt_path = (
        csv_path.parent / export_summary_csv.DEFAULT_RECEIPT_NAME
        if receipt_path is None
        else receipt_path
    ).resolve()
    output_dir = (csv_path.parent / "step37b_descriptive" if output_dir is None else output_dir).resolve()
    rows, source_receipt, csv_bytes, source_receipt_bytes = _read_and_validate_rows(
        csv_path,
        receipt_path,
        expected_sessions=expected_sessions,
        expected_noise_std=expected_noise_std,
        expected_label=expected_label,
    )

    statistics_rows = [
        _statistics(metric, [row["metrics"][metric] for row in rows])
        for metric in METRICS
    ]
    statistics_payload = _csv_payload(STATISTICS_COLUMNS, statistics_rows)

    ecdf_rows: list[dict] = []
    for metric in METRICS:
        ordered = sorted(
            (
                (row["metrics"][metric], row["session_index"])
                for row in rows
            ),
            key=lambda item: (item[0], item[1]),
        )
        start = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and ordered[end][0] == ordered[start][0]:
                end += 1
            # F_n(x) counts every observation <= x, so tied values share the
            # upper cumulative count. / 并列值共享同一个正确的经验分布值。
            cumulative_count = end
            for value, session_index in ordered[start:end]:
                ecdf_rows.append(
                    {
                        "metric": metric,
                        "cumulative_count": cumulative_count,
                        "n": expected_sessions,
                        "session_index": session_index,
                        "value": value,
                        "ecdf": cumulative_count / expected_sessions,
                    }
                )
            start = end
    ecdf_payload = _csv_payload(ECDF_COLUMNS, ecdf_rows)

    provenance_counts = {
        name: sum(row["provenance_class"] == name for row in rows)
        for name in sorted(export_summary_csv.PROVENANCE_CLASSES)
    }
    provenance_rows = [
        {
            "provenance_class": name,
            "count": provenance_counts[name],
            "share": provenance_counts[name] / expected_sessions,
        }
        for name in sorted(provenance_counts)
    ]
    provenance_payload = _csv_payload(PROVENANCE_COLUMNS, provenance_rows)

    mechanisms = [row["mechanism"] for row in rows if row["mechanism"] is not None]
    if not mechanisms:
        mechanism_evidence = {
            "status": "unavailable",
            "available_sessions": 0,
            "required_sessions": expected_sessions,
            "shares": None,
            "reason": (
                "no per-session mechanism classifications are present in the "
                "accepted Step 37A export"
            ),
        }
    elif len(mechanisms) < expected_sessions:
        mechanism_evidence = {
            "status": "incomplete_unavailable",
            "available_sessions": len(mechanisms),
            "required_sessions": expected_sessions,
            "shares": None,
            "reason": "partial mechanism evidence cannot support cohort shares",
        }
    else:
        mechanism_evidence = {
            "status": "available",
            "available_sessions": expected_sessions,
            "required_sessions": expected_sessions,
            "shares": {
                name: mechanisms.count(name) / expected_sessions
                for name in sorted(export_summary_csv.MECHANISM_NAMES)
            },
            "reason": None,
        }

    label = str(source_receipt["label"])
    noise_std = float(source_receipt["noise_std"])
    report_payload = _report_payload(
        label=label,
        noise_std=noise_std,
        expected_sessions=expected_sessions,
        statistics_rows=statistics_rows,
        mechanism_evidence=mechanism_evidence,
        provenance_counts=provenance_counts,
        cohort_provenance=str(source_receipt["cohort_provenance"]),
    )

    output_payloads = {
        DEFAULT_STATISTICS_NAME: statistics_payload,
        DEFAULT_ECDF_NAME: ecdf_payload,
        DEFAULT_PROVENANCE_NAME: provenance_payload,
        DEFAULT_REPORT_NAME: report_payload,
    }
    output_metadata = {
        name: {"sha256": _sha256_bytes(payload), "bytes": len(payload)}
        for name, payload in output_payloads.items()
    }
    analysis_receipt = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_name": "step37b_descriptive_session_metrics",
        "analysis_scope": "single_cell_descriptive_only",
        "formal_comparison_ready": False,
        "figure_rendering_status": "deferred_to_comparison_reporting_layer",
        "row_count": expected_sessions,
        "cell_key": source_receipt["cell_key"],
        "label": label,
        "noise_std": noise_std,
        "experiment_seed": source_receipt["experiment_seed"],
        "metrics": list(METRICS),
        "statistics_rows": len(statistics_rows),
        "ecdf_rows": len(ecdf_rows),
        "definitions": {
            "standard_deviation": "sample standard deviation; denominator n-1",
            "standard_error": "sample standard deviation divided by sqrt(n)",
            "quantiles": "Hyndman-Fan Type 7 linear interpolation",
            "ecdf": (
                "number of observations less than or equal to value divided by n; "
                "ties share one upper cumulative count"
            ),
            "transformations": "none; no trimming, winsorization, clamping, or logs",
        },
        "mechanism_evidence": mechanism_evidence,
        "provenance_counts": provenance_counts,
        "cohort_provenance": source_receipt.get("cohort_provenance"),
        "source_csv_file": csv_path.name,
        "source_csv_sha256": _sha256_bytes(csv_bytes),
        "source_csv_bytes": len(csv_bytes),
        "source_export_receipt_file": receipt_path.name,
        "source_export_receipt_sha256": _sha256_bytes(source_receipt_bytes),
        "source_export_receipt_bytes": len(source_receipt_bytes),
        "source_summary_sha256": source_receipt.get("source_summary_sha256"),
        "source_exporter_source_sha256": source_receipt["exporter_source_sha256"],
        "current_exporter_dependency_sha256": _sha256_bytes(
            Path(export_summary_csv.__file__).read_bytes().replace(b"\r\n", b"\n")
        ),
        "analysis_source_sha256": _normalized_source_sha256(),
        "analysis_runtime": {
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
        },
        "outputs": output_metadata,
        "publication_rule": "immutable_identical_rerun_only",
    }
    analysis_receipt_payload = (
        json.dumps(
            analysis_receipt,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    all_payloads = {
        **output_payloads,
        DEFAULT_RECEIPT_NAME: analysis_receipt_payload,
    }
    paths = {name: output_dir / name for name in all_payloads}
    for name, payload in all_payloads.items():
        _preflight_immutable(paths[name], payload)
    statuses = {
        name: _publish_immutable(paths[name], payload)
        for name, payload in all_payloads.items()
    }
    return {
        **analysis_receipt,
        "output_dir": str(output_dir),
        "files": {
            name: {"path": str(paths[name]), "status": statuses[name]}
            for name in all_payloads
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create deterministic Step 37B descriptive outputs."
    )
    parser.add_argument("csv", type=Path, help="immutable Step 37A session_metrics.csv")
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Step 37A receipt (default: beside the CSV)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="immutable Step 37B output directory"
    )
    parser.add_argument(
        "--expected-sessions", type=int, default=1000, help="required cohort size"
    )
    parser.add_argument(
        "--expected-noise-std",
        type=float,
        required=True,
        help="intent guard for the selected noise cell",
    )
    parser.add_argument(
        "--expected-label",
        required=True,
        help="intent guard for the selected experiment label",
    )
    arguments = parser.parse_args(argv)
    try:
        result = summarize_session_metrics(
            arguments.csv,
            receipt_path=arguments.receipt,
            output_dir=arguments.output_dir,
            expected_sessions=arguments.expected_sessions,
            expected_noise_std=arguments.expected_noise_std,
            expected_label=arguments.expected_label,
        )
    except (OSError, ValueError) as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
