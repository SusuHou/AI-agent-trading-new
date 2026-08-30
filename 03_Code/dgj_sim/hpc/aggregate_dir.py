"""Aggregate every session_*.npz in a cell directory into summary.json. / 汇总一个实验单元。

    python hpc/aggregate_dir.py outputs/low_noise_per_value
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dgj.config import ExperimentCell, PaperParameters  # noqa: E402
from dgj.experiments.run_cell import aggregate  # noqa: E402


def cell_from_dir(out_dir: str) -> ExperimentCell:
    with open(os.path.join(out_dir, "cell.json"), encoding="utf-8") as f:
        d = json.load(f)["cell"]
    return ExperimentCell(parameters=PaperParameters(**d["parameters"]), label=d["label"],
                          prehistory=d["prehistory"], price_mapping=d["price_mapping"],
                          price_grid=d.get("price_grid", "global"),
                          training_tie_rule=d["training_tie_rule"], measurement_tie_rule=d["measurement_tie_rule"])


def main() -> None:
    out_dir = sys.argv[1]
    summary = aggregate(cell_from_dir(out_dir), out_dir)
    per = summary.pop("per_session", [])
    censored = sum(1 for s in per if s["converged_at"] < 0)
    summary["censored_sessions"] = censored
    print(json.dumps(summary, indent=2))
    print(f"{len(per)} sessions, {censored} censored (cap reached) -> {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
