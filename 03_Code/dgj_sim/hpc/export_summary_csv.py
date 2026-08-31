"""Export one accepted experiment summary to an immutable session-level CSV.

Step 37A is deliberately only a format conversion.  It does not recompute an
economic statistic and it does not decide how low and high noise are compared.
The source ``summary.json`` must already have passed strict aggregation.

/ Step 37A 只转换文件格式，不重新计算经济指标，也不在此决定高低噪声如何比较。
输入的 summary.json 必须先通过严格汇总。

Example / 用法::

    python hpc/export_summary_csv.py "$HIGH_OUT/summary.json" \\
        --expected-sessions 1000 --expected-noise-std 100

Default outputs beside ``summary.json`` / 默认输出:

    session_metrics.csv
    session_metrics_receipt.json

Publication is immutable and restart-safe: an identical existing file is a
safe no-op, while different bytes are never overwritten. / 已有文件若完全一致则
安全跳过；只要内容不同就拒绝覆盖。
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
import re
import sys
import tempfile
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgj.config import ExperimentCell, PaperParameters  # noqa: E402


EXPORT_SCHEMA_VERSION = 1
DEFAULT_CSV_NAME = "session_metrics.csv"
DEFAULT_RECEIPT_NAME = "session_metrics_receipt.json"
SESSION_FILE_PATTERN = re.compile(r"session_(\d{4})\.npz\Z")

PARAMETER_COLUMNS = (
    "num_speculators",
    "value_mean",
    "value_std",
    "noise_std",
    "investor_slope",
    "pricing_error_weight",
    "discount_factor",
    "learning_rate",
    "exploration_decay",
    "num_value_points",
    "num_action_points",
    "num_price_points",
    "grid_widening",
    "market_maker_window",
    "convergence_periods",
    "measurement_periods",
)
INTEGER_PARAMETER_COLUMNS = (
    "num_speculators",
    "num_value_points",
    "num_action_points",
    "num_price_points",
    "market_maker_window",
    "convergence_periods",
    "measurement_periods",
)

CELL_CHOICE_COLUMNS = (
    "label",
    "prehistory",
    "price_mapping",
    "price_grid",
    "training_tie_rule",
    "measurement_tie_rule",
)

SESSION_METRIC_COLUMNS = (
    "provenance_class",
    "mechanism",
    "converged_at",
    "delta_c",
    "profit_gain_vs_nash",
    "chi_hat",
    "price_informativeness",
    "liquidity",
    "mispricing",
    "mean_lambda_hat",
)

EXPECTED_SESSION_KEYS = frozenset(("file", *SESSION_METRIC_COLUMNS))
FINITE_METRIC_COLUMNS = (
    "delta_c",
    "profit_gain_vs_nash",
    "chi_hat",
    "price_informativeness",
    "liquidity",
    "mispricing",
    "mean_lambda_hat",
)
MECHANISM_NAMES = frozenset(("price_trigger", "over_pruning", "unclassified"))
PROVENANCE_CLASSES = frozenset(("current_schema", "legacy_unversioned"))
RECOVERY_PROVENANCE_FIELDS = (
    "legacy_source_commit_assertion",
    "legacy_runtime_receipt",
    "legacy_provenance_status",
    "core_equivalence_status",
)

CSV_COLUMNS = (
    "cell_key",
    "label",
    "experiment_seed",
    "session_index",
    "source_file",
    *PARAMETER_COLUMNS,
    *CELL_CHOICE_COLUMNS[1:],
    *SESSION_METRIC_COLUMNS,
)


def _read_json(path: Path, description: str) -> tuple[dict, bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} {str(path)!r}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain one JSON object")
    return value, payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalized_source_sha256() -> str:
    """Identify the exact exporter implementation. / 记录导出器源码身份。"""

    return _sha256_bytes(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_number(value: object, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return value


def _format_csv_value(value: object) -> str:
    """Use stable, round-trippable scalar text. / 使用稳定、可往返的标量文本。"""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def _validate_cell(
    document: dict,
    *,
    expected_noise_std: float | None,
) -> tuple[dict, dict]:
    if document.get("artifact_schema_version") != 1:
        raise ValueError("cell.json uses an unsupported artifact_schema_version")
    _positive_integer(document.get("training_chunk_size"), "training_chunk_size")
    cell = document.get("cell")
    if not isinstance(cell, dict):
        raise ValueError("cell.json is missing the cell object")
    parameters = cell.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("cell.json is missing cell.parameters")

    missing_parameters = [name for name in PARAMETER_COLUMNS if name not in parameters]
    if missing_parameters:
        raise ValueError(f"cell parameters are missing: {missing_parameters}")
    extra_parameters = sorted(set(parameters) - set(PARAMETER_COLUMNS))
    if extra_parameters:
        raise ValueError(f"cell parameters are unexpected: {extra_parameters}")
    missing_choices = [name for name in CELL_CHOICE_COLUMNS if name not in cell]
    if missing_choices:
        raise ValueError(f"cell choices are missing: {missing_choices}")

    expected_cell_keys = {"parameters", *CELL_CHOICE_COLUMNS}
    if set(cell) != expected_cell_keys:
        raise ValueError(
            "cell object schema mismatch: "
            f"missing={sorted(expected_cell_keys - set(cell))}, "
            f"extra={sorted(set(cell) - expected_cell_keys)}"
        )
    cell_key = document.get("cell_key")
    if not isinstance(cell_key, str) or not re.fullmatch(r"[0-9a-f]{16}", cell_key):
        raise ValueError("cell_key must contain exactly 16 lowercase hexadecimal characters")
    _nonnegative_integer(document.get("experiment_seed"), "experiment_seed")
    key_payload = dict(cell)
    key_payload.pop("label")
    calculated_key = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    if cell_key != calculated_key:
        raise ValueError(f"cell_key mismatch: saved={cell_key}, calculated={calculated_key}")

    identity_fields = (
        "scientific_engine_version",
        "scientific_source_fingerprint",
        "python_version",
        "numpy_version",
        "numba_version",
        "numba_enabled",
        "platform_system",
        "platform_machine",
    )
    missing_identity = [name for name in identity_fields if name not in document]
    if missing_identity:
        raise ValueError(f"cell scientific identity is missing: {missing_identity}")
    _positive_integer(
        document["scientific_engine_version"], "scientific_engine_version"
    )
    fingerprint = document["scientific_source_fingerprint"]
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("scientific_source_fingerprint must be a SHA-256 hex string")
    if not isinstance(document["numba_enabled"], bool):
        raise ValueError("numba_enabled must be boolean")

    for name in PARAMETER_COLUMNS:
        _finite_number(parameters[name], f"cell.parameters.{name}")
    for name in INTEGER_PARAMETER_COLUMNS:
        _positive_integer(parameters[name], f"cell.parameters.{name}")
    try:
        parsed_parameters = PaperParameters(**parameters)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid paper parameters: {error}") from error
    if expected_noise_std is not None and not math.isclose(
        float(parameters["noise_std"]),
        expected_noise_std,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            f"noise_std is {parameters['noise_std']!r}, expected {expected_noise_std!r}"
        )
    for name in CELL_CHOICE_COLUMNS:
        if not isinstance(cell[name], str) or not cell[name]:
            raise ValueError(f"cell.{name} must be a non-empty string")
    allowed_choices = {
        "prehistory": {"nash", "cartel"},
        "price_mapping": {"nearest"},
        "price_grid": {"per_value", "global"},
        "training_tie_rule": {"uniform"},
        "measurement_tie_rule": {"lowest_index"},
    }
    for name, allowed in allowed_choices.items():
        if cell[name] not in allowed:
            raise ValueError(f"cell.{name}={cell[name]!r} is unsupported")
    parsed_cell = ExperimentCell(
        parameters=parsed_parameters,
        label=cell["label"],
        prehistory=cell["prehistory"],
        price_mapping=cell["price_mapping"],
        price_grid=cell["price_grid"],
        training_tie_rule=cell["training_tie_rule"],
        measurement_tie_rule=cell["measurement_tie_rule"],
    )
    if parsed_cell.to_dict() != cell or parsed_cell.key() != cell_key:
        raise ValueError("cell.json does not round-trip through the experiment schema")
    return cell, parameters


def _validate_summary(summary: dict, expected_sessions: int) -> list[dict]:
    required_top_level = (
        "summary_schema_version",
        "cell_key",
        "cell",
        "experiment_seed",
        "training_chunk_size",
        "sessions",
        "delta_c_mean",
        "delta_c_p01",
        "delta_c_p99",
        "chi_hat_mean",
        "chi_nash",
        "chi_cartel",
        "informativeness_mean",
        "converged_at_median",
        "liquidity_mean",
        "mispricing_mean",
        "censored_sessions",
        "legacy_unversioned_sessions",
        "current_schema_sessions",
        "cohort_provenance",
        "analysis_identity",
        "mechanism_shares",
        "mechanism_sessions",
        "per_session",
    )
    missing_top_level = [name for name in required_top_level if name not in summary]
    if missing_top_level:
        raise ValueError(f"summary fields are missing: {missing_top_level}")
    if summary["summary_schema_version"] != 1:
        raise ValueError("summary uses an unsupported summary_schema_version")
    declared_sessions = _positive_integer(summary.get("sessions"), "summary.sessions")
    if declared_sessions != expected_sessions:
        raise ValueError(
            f"summary declares {declared_sessions} sessions, expected {expected_sessions}"
        )
    censored = _nonnegative_integer(
        summary.get("censored_sessions"), "summary.censored_sessions"
    )
    if censored != 0:
        raise ValueError("formal export requires censored_sessions == 0")

    per_session = summary.get("per_session")
    if not isinstance(per_session, list) or len(per_session) != expected_sessions:
        actual = len(per_session) if isinstance(per_session, list) else "not a list"
        raise ValueError(
            f"per_session contains {actual} rows, expected {expected_sessions}"
        )

    validated: list[tuple[int, dict]] = []
    seen: set[int] = set()
    for position, row in enumerate(per_session):
        if not isinstance(row, dict):
            raise ValueError(f"per_session[{position}] must be an object")
        keys = frozenset(row)
        if keys != EXPECTED_SESSION_KEYS:
            missing = sorted(EXPECTED_SESSION_KEYS - keys)
            extra = sorted(keys - EXPECTED_SESSION_KEYS)
            raise ValueError(
                f"per_session[{position}] schema mismatch: missing={missing}, extra={extra}"
            )
        source_file = row["file"]
        match = SESSION_FILE_PATTERN.fullmatch(source_file) if isinstance(source_file, str) else None
        if match is None:
            raise ValueError(f"per_session[{position}].file is not session_NNNN.npz")
        session_index = int(match.group(1))
        if session_index in seen:
            raise ValueError(f"duplicate session index {session_index}")
        seen.add(session_index)

        provenance = row["provenance_class"]
        if provenance not in PROVENANCE_CLASSES:
            raise ValueError(f"session {session_index} has invalid provenance_class")
        mechanism = row["mechanism"]
        if mechanism is not None and mechanism not in MECHANISM_NAMES:
            raise ValueError(f"session {session_index} has invalid mechanism")
        _nonnegative_integer(row["converged_at"], f"session {session_index} converged_at")
        for name in FINITE_METRIC_COLUMNS:
            _finite_number(row[name], f"session {session_index} {name}")
        validated.append((session_index, row))

    expected_indices = set(range(expected_sessions))
    if seen != expected_indices:
        raise ValueError(
            "session index set is not exactly 0.."
            f"{expected_sessions - 1}: missing={sorted(expected_indices - seen)[:10]}, "
            f"unexpected={sorted(seen - expected_indices)[:10]}"
        )
    validated.sort(key=lambda item: item[0])
    result = [dict(row, session_index=index) for index, row in validated]
    _validate_summary_consistency(summary, result, expected_sessions)
    return result


def _close(name: str, saved: object, calculated: float) -> None:
    saved_number = _finite_number(saved, f"summary.{name}")
    if not math.isclose(
        float(saved_number),
        float(calculated),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"summary.{name}={saved_number!r} disagrees with per_session={calculated!r}"
        )


def _validate_summary_consistency(
    summary: dict,
    rows: list[dict],
    expected_sessions: int,
) -> None:
    """Prove that the top-level summary describes these exact rows.

    / 证明顶层汇总值确实对应这些 session 行。
    """

    legacy = _nonnegative_integer(
        summary["legacy_unversioned_sessions"],
        "summary.legacy_unversioned_sessions",
    )
    current = _nonnegative_integer(
        summary["current_schema_sessions"],
        "summary.current_schema_sessions",
    )
    if legacy + current != expected_sessions:
        raise ValueError("top-level provenance counts do not sum to sessions")
    row_legacy = sum(row["provenance_class"] == "legacy_unversioned" for row in rows)
    row_current = sum(row["provenance_class"] == "current_schema" for row in rows)
    if (legacy, current) != (row_legacy, row_current):
        raise ValueError("top-level provenance counts disagree with per_session")
    expected_cohort = (
        "mixed_legacy_unversioned_and_current_schema"
        if legacy
        else "homogeneous_current_schema"
    )
    if summary["cohort_provenance"] != expected_cohort:
        raise ValueError("cohort_provenance disagrees with per-session provenance")
    if not isinstance(summary["analysis_identity"], dict):
        raise ValueError("analysis_identity must be an object")
    analysis_identity = summary["analysis_identity"]
    expected_analysis_keys = {
        "analysis_source_fingerprint",
        "python_version",
        "numpy_version",
        "numba_version",
        "numba_enabled",
        "platform_system",
        "platform_machine",
    }
    if set(analysis_identity) != expected_analysis_keys:
        raise ValueError(
            "analysis_identity schema mismatch: "
            f"missing={sorted(expected_analysis_keys - set(analysis_identity))}, "
            f"extra={sorted(set(analysis_identity) - expected_analysis_keys)}"
        )
    fingerprint = analysis_identity["analysis_source_fingerprint"]
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("analysis_source_fingerprint must be a SHA-256 hex string")
    if not isinstance(analysis_identity["numba_enabled"], bool):
        raise ValueError("analysis_identity.numba_enabled must be boolean")
    for name in ("python_version", "platform_system", "platform_machine"):
        if not isinstance(analysis_identity[name], str) or not analysis_identity[name]:
            raise ValueError(f"analysis_identity.{name} must be a non-empty string")
    for name in ("numpy_version", "numba_version"):
        if analysis_identity[name] is not None and not isinstance(
            analysis_identity[name], str
        ):
            raise ValueError(f"analysis_identity.{name} must be a string or null")

    recovery_fields_present = {
        name for name in RECOVERY_PROVENANCE_FIELDS if name in summary
    }
    if legacy:
        if recovery_fields_present != set(RECOVERY_PROVENANCE_FIELDS):
            raise ValueError("legacy sessions require complete recovery provenance")
        if summary["core_equivalence_status"] != "established_for_audited_base_commit":
            raise ValueError("legacy sessions lack established core equivalence")
    elif recovery_fields_present and recovery_fields_present != set(
        RECOVERY_PROVENANCE_FIELDS
    ):
        raise ValueError("partial recovery provenance is not allowed")

    deltas = np.asarray([row["delta_c"] for row in rows], dtype=np.float64)
    chi = np.asarray([row["chi_hat"] for row in rows], dtype=np.float64)
    informativeness = np.asarray(
        [row["price_informativeness"] for row in rows], dtype=np.float64
    )
    converged = np.asarray([row["converged_at"] for row in rows], dtype=np.float64)
    liquidity = np.asarray([row["liquidity"] for row in rows], dtype=np.float64)
    mispricing = np.asarray([row["mispricing"] for row in rows], dtype=np.float64)
    calculated = {
        "delta_c_mean": float(deltas.mean()),
        "delta_c_p01": float(np.percentile(deltas, 1)),
        "delta_c_p99": float(np.percentile(deltas, 99)),
        "chi_hat_mean": float(chi.mean()),
        "informativeness_mean": float(informativeness.mean()),
        "converged_at_median": float(np.median(converged)),
        "liquidity_mean": float(liquidity.mean()),
        "mispricing_mean": float(mispricing.mean()),
    }
    for name, value in calculated.items():
        _close(name, summary[name], value)
    _finite_number(summary["chi_nash"], "summary.chi_nash")
    _finite_number(summary["chi_cartel"], "summary.chi_cartel")

    mechanism_count = sum(row["mechanism"] in MECHANISM_NAMES for row in rows)
    declared_mechanisms = _nonnegative_integer(
        summary["mechanism_sessions"], "summary.mechanism_sessions"
    )
    if declared_mechanisms != mechanism_count:
        raise ValueError("mechanism_sessions disagrees with per_session")
    shares = summary["mechanism_shares"]
    if mechanism_count == expected_sessions:
        if not isinstance(shares, dict) or set(shares) != set(MECHANISM_NAMES):
            raise ValueError("complete mechanism evidence requires all three shares")
        for name in MECHANISM_NAMES:
            _close(
                f"mechanism_shares.{name}",
                shares[name],
                sum(row["mechanism"] == name for row in rows) / expected_sessions,
            )
    elif shares is not None:
        raise ValueError("partial mechanism evidence requires mechanism_shares == null")


def _build_csv(cell_document: dict, cell: dict, parameters: dict, rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        values = {
            "cell_key": cell_document["cell_key"],
            "label": cell["label"],
            "experiment_seed": cell_document["experiment_seed"],
            "session_index": row["session_index"],
            "source_file": row["file"],
            **parameters,
            **{name: cell[name] for name in CELL_CHOICE_COLUMNS[1:]},
            **{name: row[name] for name in SESSION_METRIC_COLUMNS},
        }
        writer.writerow([_format_csv_value(values[name]) for name in CSV_COLUMNS])
    return stream.getvalue().encode("utf-8")


def _publish_immutable(path: Path, payload: bytes) -> str:
    """Publish once; validate instead of overwriting. / 只发布一次，不覆盖。"""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite different existing file: {path}")
        return "validated_existing"

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        published = False
        try:
            os.link(temporary_path, path)
            published = True
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"concurrent process published different file: {path}")
        if path.read_bytes() != payload:
            raise ValueError(f"published file failed byte validation: {path}")
        return "published" if published else "validated_existing"
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _preflight_immutable(path: Path, payload: bytes) -> str:
    """Check both destinations before publishing either one. / 先检查两个目标再发布。"""

    if not path.exists():
        return "missing"
    if path.read_bytes() != payload:
        raise ValueError(f"refusing to overwrite different existing file: {path}")
    return "validated_existing"


def export_summary(
    summary_path: Path,
    *,
    cell_path: Path | None = None,
    csv_path: Path | None = None,
    receipt_path: Path | None = None,
    expected_sessions: int = 1000,
    expected_noise_std: float | None = None,
) -> dict:
    """Validate, convert, and publish one accepted cell. / 验证、转换并发布一个 cell。"""

    summary_path = summary_path.resolve()
    cell_path = (summary_path.parent / "cell.json" if cell_path is None else cell_path).resolve()
    csv_path = (summary_path.parent / DEFAULT_CSV_NAME if csv_path is None else csv_path).resolve()
    receipt_path = (
        summary_path.parent / DEFAULT_RECEIPT_NAME
        if receipt_path is None
        else receipt_path
    ).resolve()
    if len({summary_path, cell_path, csv_path, receipt_path}) != 4:
        raise ValueError("summary, cell, CSV, and receipt paths must be different")
    _positive_integer(expected_sessions, "expected_sessions")
    if expected_noise_std is not None:
        _finite_number(expected_noise_std, "expected_noise_std")

    summary, summary_payload = _read_json(summary_path, "summary")
    cell_document, cell_payload = _read_json(cell_path, "cell identity")
    cell, parameters = _validate_cell(
        cell_document,
        expected_noise_std=expected_noise_std,
    )
    rows = _validate_summary(summary, expected_sessions)
    bound_identity = {
        "cell_key": cell_document["cell_key"],
        "cell": cell_document["cell"],
        "experiment_seed": cell_document["experiment_seed"],
        "training_chunk_size": cell_document["training_chunk_size"],
    }
    mismatched_binding = [
        name for name, expected in bound_identity.items()
        if summary.get(name) != expected
    ]
    if mismatched_binding:
        raise ValueError(
            "summary.json and cell.json belong to different experiments: "
            f"mismatched {mismatched_binding}"
        )
    csv_payload = _build_csv(cell_document, cell, parameters, rows)
    csv_sha256 = _sha256_bytes(csv_payload)

    receipt = {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "row_count": len(rows),
        "columns": list(CSV_COLUMNS),
        "session_index_min": rows[0]["session_index"],
        "session_index_max": rows[-1]["session_index"],
        "cell_key": cell_document["cell_key"],
        "label": cell["label"],
        "noise_std": parameters["noise_std"],
        "experiment_seed": cell_document["experiment_seed"],
        "source_scientific_identity": {
            name: cell_document[name]
            for name in (
                "scientific_engine_version",
                "scientific_source_fingerprint",
                "python_version",
                "numpy_version",
                "numba_version",
                "numba_enabled",
                "platform_system",
                "platform_machine",
            )
        },
        "cohort_provenance": summary.get("cohort_provenance"),
        "source_analysis_identity": summary.get("analysis_identity"),
        "source_recovery_provenance": {
            name: summary[name]
            for name in RECOVERY_PROVENANCE_FIELDS
            if name in summary
        },
        "source_summary_file": summary_path.name,
        "source_summary_sha256": _sha256_bytes(summary_payload),
        "source_summary_bytes": len(summary_payload),
        "source_cell_file": cell_path.name,
        "source_cell_sha256": _sha256_bytes(cell_payload),
        "source_cell_bytes": len(cell_payload),
        "csv_file": csv_path.name,
        "csv_sha256": csv_sha256,
        "csv_bytes": len(csv_payload),
        "csv_encoding": "UTF-8 without BOM",
        "csv_line_ending": "LF",
        "null_encoding": "empty CSV field",
        "exporter_source_sha256": _normalized_source_sha256(),
        "publication_rule": "immutable_identical_rerun_only",
    }
    receipt_payload = (
        json.dumps(
            receipt,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    # A conflicting receipt must stop before a missing CSV is created, and
    # vice versa. / 任一已有文件冲突时，另一文件也不得先发布。
    _preflight_immutable(csv_path, csv_payload)
    _preflight_immutable(receipt_path, receipt_payload)
    csv_status = _publish_immutable(csv_path, csv_payload)
    receipt_status = _publish_immutable(receipt_path, receipt_payload)
    return {
        **receipt,
        "csv_path": str(csv_path),
        "receipt_path": str(receipt_path),
        "csv_status": csv_status,
        "receipt_status": receipt_status,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export an accepted summary.json to immutable session-level CSV."
    )
    parser.add_argument("summary", type=Path)
    parser.add_argument("--cell", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--expected-sessions", type=int, default=1000)
    parser.add_argument("--expected-noise-std", type=float, default=None)
    arguments = parser.parse_args(argv)
    try:
        result = export_summary(
            arguments.summary,
            cell_path=arguments.cell,
            csv_path=arguments.output,
            receipt_path=arguments.receipt,
            expected_sessions=arguments.expected_sessions,
            expected_noise_std=arguments.expected_noise_std,
        )
    except (OSError, ValueError) as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
