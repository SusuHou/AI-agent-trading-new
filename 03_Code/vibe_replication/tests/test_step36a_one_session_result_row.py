"""Tests for Step 36A's one-session result pipeline. / Step 36A 测试。"""

from dataclasses import asdict
import csv
import json
import sys
import unittest
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = PROJECT_ROOT / "steps"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from step_36a_one_session_result_row import (
    PRICE_GRID_ENCODING,
    SMOKE_CONVERGENCE_PERIODS,
    SMOKE_MEASUREMENT_PERIODS,
    SMOKE_MODE,
    build_engineering_smoke_row,
    save_result_row,
)


class Step36AResultRowTests(unittest.TestCase):
    """Dependency-free tests runnable with Python itself. / 可直接用 Python 运行的无依赖测试。"""

    def test_one_session_smoke_pipeline_and_saved_files(self) -> None:
        """The full wiring must finish and save exactly one honest row.

        完整连接必须成功结束，并保存恰好一行、标签诚实的结果。
        """

        row = build_engineering_smoke_row(experiment_seed=12345, session_index=2)

        self.assertEqual(row.mode, SMOKE_MODE)
        self.assertFalse(row.research_result)
        self.assertFalse(row.paper_scale)
        self.assertIsNone(row.mechanism_label)
        self.assertEqual(
            row.convergence_periods_required,
            SMOKE_CONVERGENCE_PERIODS,
        )
        self.assertEqual(
            row.training_periods_completed,
            SMOKE_CONVERGENCE_PERIODS,
        )
        self.assertEqual(row.policy_change_events, 0)
        self.assertEqual(
            row.measurement_periods_completed,
            SMOKE_MEASUREMENT_PERIODS,
        )
        self.assertEqual(row.experiment_seed, 12345)
        self.assertEqual(row.session_index, 2)
        self.assertEqual(len(row.config_hash), 64)
        self.assertEqual(row.price_grid_encoding, PRICE_GRID_ENCODING)
        self.assertEqual(len(row.price_grid_sha256), 64)

        # A session number changes the path identity, not the experiment-cell
        # configuration. / session 编号改变随机路径身份，但不改变实验单元配置。
        sibling_row = build_engineering_smoke_row(
            experiment_seed=12345,
            session_index=3,
        )
        self.assertEqual(sibling_row.config_hash, row.config_hash)
        self.assertNotEqual(sibling_row.session_seed, row.session_seed)
        self.assertNotEqual(sibling_row.run_id, row.run_id)

        replay_row = build_engineering_smoke_row(
            experiment_seed=12345,
            session_index=2,
        )
        row_without_clock = asdict(row)
        replay_without_clock = asdict(replay_row)
        row_without_clock.pop("elapsed_seconds")
        replay_without_clock.pop("elapsed_seconds")
        self.assertEqual(replay_without_clock, row_without_clock)

        # Windows can make nested atomic writes unusually slow inside a
        # ``TemporaryDirectory`` on this workspace. Create and verify one exact
        # scratch directory instead. / 本工作区的 Windows 环境中，在
        # ``TemporaryDirectory`` 内再做原子写入可能异常缓慢，因此建立一个明确的
        # scratch 目录，并在 finally 中准确清理它。
        temporary_path = PROJECT_ROOT / "results" / f"step36a_test_{uuid4().hex}"
        temporary_path.mkdir(parents=True)
        try:
            json_path = temporary_path / "result.json"
            csv_path = temporary_path / "result.csv"
            save_result_row(row, json_path=json_path, csv_path=csv_path)

            saved_json = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_json, asdict(row))

            with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
                saved_csv_rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(saved_csv_rows), 1)
            self.assertEqual(saved_csv_rows[0]["run_id"], row.run_id)
            self.assertEqual(saved_csv_rows[0]["mode"], SMOKE_MODE)
            self.assertEqual(saved_csv_rows[0]["research_result"], "False")
        finally:
            for path in (temporary_path / "result.json", temporary_path / "result.csv"):
                if path.exists():
                    path.unlink()
            temporary_path.rmdir()


if __name__ == "__main__":
    unittest.main()
