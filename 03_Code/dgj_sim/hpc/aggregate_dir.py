"""Aggregate every session_*.npz in a cell directory into summary.json. / 汇总一个实验单元。

    python hpc/aggregate_dir.py outputs/low_noise_per_value --expected-sessions 1000
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dgj.config import ExperimentCell, PaperParameters  # noqa: E402
from dgj.experiments.run_cell import aggregate  # noqa: E402


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_legacy_selection(out_dir: str, session_files: list[str]) -> None:
    """Require and re-hash the selection receipt for every schema-less result."""
    legacy_names = []
    for name in session_files:
        path = os.path.join(out_dir, name)
        try:
            with np.load(path, allow_pickle=False) as data:
                if "result_schema_version" not in data.files:
                    legacy_names.append(name)
        except (OSError, ValueError) as error:
            raise SystemExit(f"STOP: cannot inspect {name}: {error}") from error
    if not legacy_names:
        return

    manifest_path = os.path.join(out_dir, "recovery_manifest.json")
    selection_path = os.path.join(out_dir, "recovery_selection.json")
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            receipt = json.load(handle)
        with open(selection_path, encoding="utf-8") as handle:
            selection = json.load(handle)
        with open(os.path.join(out_dir, "cell.json"), encoding="utf-8") as handle:
            cell_receipt = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            "STOP: schema-less legacy results require valid recovery receipts"
        ) from error
    records = {record.get("filename"): record for record in selection}
    if set(records) != set(legacy_names):
        raise SystemExit(
            "STOP: recovery selection does not exactly match legacy result files"
        )
    if receipt.get("reused_converged_sessions") != len(legacy_names):
        raise SystemExit("STOP: recovery manifest legacy count is inconsistent")
    if receipt.get("core_equivalence_status") != "established_for_audited_base_commit":
        raise SystemExit(
            "STOP: legacy/current core equivalence is not established for this source commit"
        )
    if receipt.get("legacy_training_chunk_size_assertion") != cell_receipt.get(
        "training_chunk_size"
    ):
        raise SystemExit("STOP: legacy/current training chunk-size assertions disagree")
    for position, name in enumerate(sorted(legacy_names), start=1):
        expected_hash = records[name].get("copied_sha256")
        if not expected_hash or _sha256(os.path.join(out_dir, name)) != expected_hash:
            raise SystemExit(f"STOP: legacy selection hash mismatch for {name}")
        if position == 1 or position % 50 == 0 or position == len(legacy_names):
            print(
                f"legacy hash verification / 旧结果哈希校验: {position}/{len(legacy_names)}",
                file=sys.stderr,
                flush=True,
            )
    print(
        "WARNING: legacy files have selection hashes but do not embed code/runtime/chunk "
        f"identity; report this cohort as mixed provenance "
        f"({receipt.get('legacy_provenance_status', 'unknown')}).",
        file=sys.stderr,
    )


def cell_from_dir(out_dir: str) -> ExperimentCell:
    with open(os.path.join(out_dir, "cell.json"), encoding="utf-8") as f:
        d = json.load(f)["cell"]
    return ExperimentCell(parameters=PaperParameters(**d["parameters"]), label=d["label"],
                          prehistory=d["prehistory"], price_mapping=d["price_mapping"],
                          price_grid=d.get("price_grid", "global"),
                          training_tie_rule=d["training_tie_rule"], measurement_tie_rule=d["measurement_tie_rule"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument(
        "--expected-sessions",
        type=int,
        default=None,
        help="refuse aggregation unless this many genuine result files exist",
    )
    arguments = parser.parse_args()
    out_dir = arguments.out_dir
    session_files = [
        name for name in os.listdir(out_dir)
        if name.startswith("session_") and name.endswith(".npz")
    ]
    checkpoint_files = [
        name for name in os.listdir(out_dir)
        if name.startswith("ckpt_") and name.endswith(".npz")
    ]
    if arguments.expected_sessions is not None and len(session_files) != arguments.expected_sessions:
        raise SystemExit(
            f"STOP: found {len(session_files)} results, expected {arguments.expected_sessions}; "
            f"{len(checkpoint_files)} sessions still have checkpoints"
        )
    if arguments.expected_sessions is not None:
        expected_names = {
            f"session_{index:04d}.npz" for index in range(arguments.expected_sessions)
        }
        actual_names = set(session_files)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)[:10]
            unexpected = sorted(actual_names - expected_names)[:10]
            raise SystemExit(
                f"STOP: session index set is wrong; missing={missing}, unexpected={unexpected}"
            )
    if checkpoint_files:
        raise SystemExit(
            f"STOP: {len(checkpoint_files)} sessions are still incomplete; aggregation is forbidden"
        )
    verify_legacy_selection(out_dir, session_files)
    summary = aggregate(cell_from_dir(out_dir), out_dir, progress=True)
    per = summary.pop("per_session", [])
    print(json.dumps(summary, indent=2))
    print(
        f"{len(per)} genuinely converged sessions, 0 censored -> "
        f"{os.path.join(out_dir, 'summary.json')}"
    )


if __name__ == "__main__":
    main()
