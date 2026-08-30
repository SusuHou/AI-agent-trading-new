"""Tests for Step 13B's n_x/n_p resolution diagnostic. / 第 13B 步自动测试。"""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = PROJECT_ROOT / "steps"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from step_13b_grid_resolution import build_resolution_table


class TestStep13BGridResolution(unittest.TestCase):
    """Check counts, widths, and the low-noise footnote-25 condition. / 检查点数与宽度。"""

    def test_points_create_one_fewer_intervals(self) -> None:
        parameters = PaperParameters()
        row = build_resolution_table(parameters, parameters.noise_std)[0]
        self.assertEqual(row["number_of_action_points_n_x"], 15)
        self.assertEqual(row["number_of_action_intervals"], 14)
        self.assertEqual(row["number_of_price_points_n_p"], 31)
        self.assertEqual(row["number_of_price_intervals"], 30)

    def test_low_noise_one_action_is_about_one_price_step(self) -> None:
        parameters = PaperParameters()
        rows = build_resolution_table(parameters, parameters.noise_std)
        self.assertEqual(len(rows), parameters.num_value_points)
        for row in rows:
            with self.subTest(value=row["fundamental_value"]):
                ratio = float(row["movement_to_grid_step_ratio"])
                self.assertGreaterEqual(ratio, 0.9)
                self.assertLessEqual(ratio, 1.2)

    def test_high_noise_band_makes_the_same_31_points_coarser(self) -> None:
        parameters = PaperParameters()
        rows = build_resolution_table(parameters, 100.0)
        self.assertTrue(
            all(
                float(row["movement_to_grid_step_ratio"]) < 0.3
                for row in rows
            )
        )


if __name__ == "__main__":
    unittest.main()
