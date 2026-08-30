"""List incomplete sessions at the 50B diagnostic review point. / 列出需人工复核的 session。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


PATTERN = re.compile(r"progress_(\d{4})\.json\Z")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--ids-only", action="store_true")
    arguments = parser.parse_args(argv)
    due = []
    for path in sorted(arguments.out.glob("progress_*.json")):
        match = PATTERN.fullmatch(path.name)
        if match is None:
            continue
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read {path}: {error}")
        index = int(match.group(1))
        if receipt.get("session_index") != index:
            parser.error(f"session identity mismatch in {path.name}")
        if (
            receipt.get("status") == "incomplete"
            and receipt.get("diagnostic_review_due") is True
        ):
            due.append((index, int(receipt.get("training_periods_completed", -1))))
    if arguments.ids_only:
        print(",".join(str(index) for index, _ in due))
    elif not due:
        print("No sessions require 50B review. / 没有 session 需要 50B 复核。")
    else:
        print("Sessions requiring documented review / 需要人工复核:")
        for index, periods in due:
            print(f"  session {index:04d}: {periods:,} training periods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
