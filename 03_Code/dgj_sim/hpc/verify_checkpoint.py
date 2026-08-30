"""Validate one resumable checkpoint against its output cell and progress receipt.

Usage / 用法:
    python hpc/verify_checkpoint.py OUT_DIR SESSION_ID \
        --expect-stop-reason scheduler_signal
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dgj.config import ExperimentCell, PaperParameters  # noqa: E402
from dgj.game.session import Session  # noqa: E402
from dgj.provenance import scientific_identity  # noqa: E402


def _cell_from_document(document: dict) -> ExperimentCell:
    raw = dict(document["cell"])
    parameters = PaperParameters(**raw.pop("parameters"))
    return ExperimentCell(parameters=parameters, **raw)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("session", type=int)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--expect-stop-reason", default=None)
    parser.add_argument("--expect-job-id", default=None)
    arguments = parser.parse_args(argv)

    out = arguments.out.resolve()
    try:
        document = json.loads((out / "cell.json").read_text(encoding="utf-8"))
        cell = _cell_from_document(document)
        seed = int(document["experiment_seed"])
        progress = json.loads(
            (out / f"progress_{arguments.session:04d}.json").read_text(encoding="utf-8")
        )
        session = Session(cell, arguments.session, seed)
        session.load_checkpoint(
            str(out / f"ckpt_{arguments.session:04d}.npz"),
            expected_training_chunk_size=arguments.chunk_size,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(f"checkpoint validation failed: {error}")

    if progress.get("status") != "incomplete":
        parser.error(f"progress status is {progress.get('status')!r}, expected 'incomplete'")
    if (
        arguments.expect_stop_reason is not None
        and progress.get("stop_reason") != arguments.expect_stop_reason
    ):
        parser.error(
            f"stop reason is {progress.get('stop_reason')!r}, "
            f"expected {arguments.expect_stop_reason!r}"
        )
    if int(progress.get("periods_completed", -1)) != session.periods_completed:
        parser.error("progress and checkpoint period counters disagree")
    expected_progress = {
        "cell_key": cell.key(),
        "session_index": arguments.session,
        "experiment_seed": seed,
        "phase": "training",
        "training_chunk_size": arguments.chunk_size,
        **scientific_identity(),
    }
    mismatches = {
        key: (progress.get(key), expected)
        for key, expected in expected_progress.items()
        if progress.get(key) != expected
    }
    if mismatches:
        parser.error(f"progress identity mismatch: {mismatches}")
    if session.phase != "training":
        parser.error(f"checkpoint phase is {session.phase!r}, expected 'training'")
    if (
        arguments.expect_job_id is not None
        and str(progress.get("slurm_job_id")) != str(arguments.expect_job_id)
    ):
        parser.error(
            f"progress came from Slurm job {progress.get('slurm_job_id')!r}, "
            f"expected {arguments.expect_job_id!r}"
        )

    print(
        f"VALID checkpoint / checkpoint 验证通过: session={arguments.session}, "
        f"periods={session.periods_completed:,}, reason={progress.get('stop_reason')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
