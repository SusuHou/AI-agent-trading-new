"""Prepare a clean recovery cohort from an older capped campaign.

The source directory is never modified. Genuinely converged results are copied
into a new directory; censored session IDs are omitted and written as a Slurm
array list so they can restart with the same deterministic identities.

/ 原目录保持不变。已收敛结果独立复制到新目录；被 cap 截断的 session 留空，并
生成 Slurm array ID 清单，让它们用原来的 seed/session ID 从头重跑。

Example / 示例:
    python hpc/prepare_recovery.py OLD_OUT NEW_OUT \
        --expect-sessions 1000 --expect-censored 178
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dgj.game import protocol  # noqa: E402
from dgj.provenance import scientific_identity  # noqa: E402


SESSION_PATTERN = re.compile(r"session_(\d{4})\.npz\Z")
AUDITED_LEGACY_COMMIT = "9ab452fb6dc54e7ce25a9ec9417e346aa177d366"


def _read_result(path: Path, cell_document: dict) -> tuple[int, bool, int]:
    match = SESSION_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected session filename: {path.name}")
    session_index = int(match.group(1))
    try:
        with np.load(path, allow_pickle=False) as data:
            rows = data["rows"]
            converged_at = int(data["converged_at"].item())
            manifest = json.loads(str(data["manifest"].item()))
            finite = bool(np.isfinite(rows).all())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot validate {path.name}: {error}") from error

    cell = cell_document["cell"]
    parameters = cell["parameters"]
    expected_shape = (
        int(parameters["measurement_periods"]),
        protocol.row_width(int(parameters["num_speculators"])),
    )
    shocks = manifest.get("shocks", {})
    expected_identity = (
        cell_document["cell_key"],
        int(cell_document["experiment_seed"]),
        session_index,
    )
    actual_identity = (
        manifest.get("cell_key"),
        shocks.get("experiment_seed"),
        shocks.get("session_index"),
    )
    if actual_identity != expected_identity:
        raise ValueError(
            f"identity mismatch in {path.name}: found {actual_identity}, "
            f"expected {expected_identity}"
        )
    if manifest.get("cell") != cell or shocks.get("cell_key") != cell_document["cell_key"]:
        raise ValueError(f"full cell identity mismatch in {path.name}")
    if rows.shape != expected_shape or not finite:
        raise ValueError(
            f"invalid measurement rows in {path.name}: {rows.shape}, expected {expected_shape}"
        )
    censored = converged_at < 0 or manifest.get("censored") is True
    if not censored:
        if manifest.get("converged_at") != converged_at:
            raise ValueError(f"converged_at mismatch in {path.name}")
        expected_periods = converged_at + expected_shape[0]
        if manifest.get("periods_completed") != expected_periods:
            raise ValueError(f"period count mismatch in {path.name}")
    return session_index, censored, converged_at


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_result(source: Path, target: Path) -> str:
    """Copy independently and prove source/target bytes match. / 独立复制并核对哈希。"""
    source_hash = _sha256(source)
    shutil.copy2(source, target)
    if _sha256(target) != source_hash:
        raise OSError(f"copied result failed SHA-256 verification: {source.name}")
    return source_hash


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="immutable old capped output directory")
    parser.add_argument("target", type=Path, help="new recovery output directory")
    parser.add_argument("--expect-sessions", type=int, default=None)
    parser.add_argument("--expect-censored", type=int, default=None)
    parser.add_argument("--training-chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--legacy-source-commit",
        default="unknown",
        help="externally recovered commit for the old campaign; not inferred from NPZ files",
    )
    parser.add_argument(
        "--legacy-runtime-receipt",
        default="unknown",
        help="path or identifier of an archived old Python/NumPy/Numba receipt",
    )
    arguments = parser.parse_args(argv)
    if arguments.training_chunk_size < 1:
        parser.error("--training-chunk-size must be positive")
    if (
        arguments.legacy_source_commit != "unknown"
        and re.fullmatch(r"[0-9a-fA-F]{7,40}", arguments.legacy_source_commit) is None
    ):
        parser.error("--legacy-source-commit must be a 7-40 character Git hash")

    source = arguments.source.resolve()
    target = arguments.target.resolve()
    if not source.is_dir():
        parser.error(f"source directory does not exist: {source}")
    if target.exists():
        parser.error(f"target must not already exist: {target}")
    cell_path = source / "cell.json"
    try:
        cell_document = json.loads(cell_path.read_text(encoding="utf-8"))
        for key in ("cell", "cell_key", "experiment_seed"):
            if key not in cell_document:
                raise ValueError(f"cell.json is missing {key}")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"invalid source cell.json: {error}")

    files = sorted(
        path for path in source.iterdir() if SESSION_PATTERN.fullmatch(path.name)
    )
    if arguments.expect_sessions is not None and len(files) != arguments.expect_sessions:
        parser.error(
            f"expected {arguments.expect_sessions} session files, found {len(files)}"
        )

    converged: list[tuple[int, Path, int]] = []
    censored_ids: list[int] = []
    seen: set[int] = set()
    try:
        for path in files:
            session_index, censored, converged_at = _read_result(path, cell_document)
            if session_index in seen:
                raise ValueError(f"duplicate session index {session_index}")
            seen.add(session_index)
            if censored:
                censored_ids.append(session_index)
            else:
                converged.append((session_index, path, converged_at))
    except ValueError as error:
        parser.error(str(error))
    if (
        arguments.expect_censored is not None
        and len(censored_ids) != arguments.expect_censored
    ):
        parser.error(
            f"expected {arguments.expect_censored} censored sessions, found {len(censored_ids)}"
        )
    if arguments.expect_sessions is not None:
        expected_indices = set(range(arguments.expect_sessions))
        if seen != expected_indices:
            missing = sorted(expected_indices - seen)[:10]
            unexpected = sorted(seen - expected_indices)[:10]
            parser.error(
                f"source session index set is wrong; missing={missing}, unexpected={unexpected}"
            )
        if set(censored_ids) != expected_indices.difference(
            index for index, _, _ in converged
        ):
            parser.error("censored IDs are not the exact complement of converged IDs")
    if not censored_ids:
        parser.error("source contains no censored sessions to recover")

    target.mkdir(parents=True)
    audited_equivalence = (
        arguments.legacy_source_commit.lower() == AUDITED_LEGACY_COMMIT
    )
    target_cell_document = {
        **cell_document,
        "artifact_schema_version": 1,
        "training_chunk_size": arguments.training_chunk_size,
        **scientific_identity(),
        "recovery_source_directory": str(source),
        "cohort_provenance": "mixed_legacy_unversioned_and_current_schema",
    }
    (target / "cell.json").write_text(
        json.dumps(target_cell_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selection = []
    for position, (session_index, source_result, converged_at) in enumerate(
        converged,
        start=1,
    ):
        target_result = target / source_result.name
        file_hash = _copy_result(source_result, target_result)
        selection.append({
            "session_index": session_index,
            "filename": source_result.name,
            "source_path": str(source_result),
            "converged_at": converged_at,
            "source_sha256": file_hash,
            "copied_sha256": file_hash,
        })
        if position == 1 or position % 50 == 0 or position == len(converged):
            print(
                f"copy/hash progress / 复制校验进度: {position}/{len(converged)}",
                flush=True,
            )
    (target / "recovery_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    newline_ids = "".join(f"{index}\n" for index in censored_ids)
    array_ids = ",".join(str(index) for index in censored_ids)
    (target / "recovery_session_ids.txt").write_text(newline_ids, encoding="utf-8")
    (target / "recovery_array.txt").write_text(array_ids + "\n", encoding="utf-8")
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source),
        "target_directory": str(target),
        "source_preserved": True,
        "cell_key": cell_document["cell_key"],
        "experiment_seed": cell_document["experiment_seed"],
        "source_session_files": len(files),
        "reused_converged_sessions": len(converged),
        "sessions_requiring_replay": len(censored_ids),
        "recovery_session_ids": censored_ids,
        "publication_mode": "independent_copy",
        "training_chunk_size": arguments.training_chunk_size,
        "legacy_source_commit_assertion": arguments.legacy_source_commit,
        "legacy_runtime_receipt": arguments.legacy_runtime_receipt,
        "legacy_training_chunk_size_assertion": arguments.training_chunk_size,
        "legacy_provenance_status": (
            "externally_documented"
            if arguments.legacy_source_commit != "unknown"
            and arguments.legacy_runtime_receipt != "unknown"
            else "partial_unversioned"
        ),
        "legacy_artifacts_embed_commit_runtime_chunk": False,
        "core_equivalence_status": (
            "established_for_audited_base_commit"
            if audited_equivalence
            else "not_established_for_asserted_commit"
        ),
        "core_equivalence_audit": (
            "Maintainer audit for 9ab452fb6dc54e7ce25a9ec9417e346aa177d366: "
            "legacy/current protocol, shocks, config, environment, player equations, "
            "Session initialization/training/measurement, 1M chunks, and 100k frozen "
            "measurement are equivalent; orchestration/provenance changed"
            if audited_equivalence
            else "No core-equivalence audit is claimed for the asserted legacy commit"
        ),
        "selection_manifest": "recovery_selection.json",
        **scientific_identity(),
        "scientific_cumulative_period_cap": None,
    }
    (target / "recovery_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "recovery_session_ids"}, indent=2))
    print(f"Slurm array IDs / 需重跑的 IDs: {array_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
