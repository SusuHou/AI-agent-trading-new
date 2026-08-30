"""Regression tests for Step 13's value-specific P(v). / 第 13 步分价值网格测试。"""

from math import isclose
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
from src.step01_value_grid import build_value_grid, discrete_value_std
from step_10_fixed_point_solver import solve_benchmark_fixed_point
from step_13_price_grid import build_price_grids_by_value
from step_13b_grid_resolution import build_resolution_table


class TestStep13PerValuePriceGrids(unittest.TestCase):
    """Prove that production Step 13 uses 10 local rows, not one global row."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.parameters = PaperParameters()
        cls.values = build_value_grid(
            cls.parameters.value_mean,
            cls.parameters.value_std,
            cls.parameters.num_value_points,
        )
        value_std = discrete_value_std(cls.values, cls.parameters.value_mean)
        cls.nash = solve_benchmark_fixed_point(
            "nash",
            cls.parameters.num_speculators,
            cls.parameters.noise_std,
            value_std,
            cls.parameters.investor_slope,
            cls.parameters.pricing_error_weight,
        )
        cls.cartel = solve_benchmark_fixed_point(
            "cartel",
            cls.parameters.num_speculators,
            cls.parameters.noise_std,
            value_std,
            cls.parameters.investor_slope,
            cls.parameters.pricing_error_weight,
        )
        cls.price_grids = build_price_grids_by_value(
            value_grid=cls.values,
            value_mean=cls.parameters.value_mean,
            number_of_speculators=cls.parameters.num_speculators,
            nash_price_impact=cls.nash["price_impact"],
            nash_intensity=cls.nash["intensity"],
            cartel_intensity=cls.cartel["intensity"],
            noise_std=cls.parameters.noise_std,
            grid_widening=cls.parameters.grid_widening,
            number_of_prices=cls.parameters.num_price_points,
        )

    def test_shape_is_ten_by_thirty_one(self) -> None:
        self.assertEqual(len(self.price_grids), 10)
        self.assertTrue(all(len(row) == 31 for row in self.price_grids))

    def test_rows_are_strictly_increasing_and_equally_spaced(self) -> None:
        for value_index, row in enumerate(self.price_grids):
            with self.subTest(value_index=value_index):
                steps = [right - left for left, right in zip(row, row[1:])]
                self.assertTrue(all(step > 0.0 for step in steps))
                self.assertTrue(
                    all(isclose(step, steps[0], abs_tol=1e-12) for step in steps)
                )

    def test_low_noise_numeric_anchors(self) -> None:
        anchors = {
            (0, 0): -0.1244537283,
            (0, 15): 0.0405021332,
            (4, 15): 0.9266975540,
            (5, 15): 1.0733024460,
            (9, 15): 1.9594978668,
            (9, 30): 2.1244537283,
        }
        for (value_index, price_index), expected in anchors.items():
            with self.subTest(value_index=value_index, price_index=price_index):
                self.assertAlmostEqual(
                    self.price_grids[value_index][price_index],
                    expected,
                    places=9,
                )

    def test_paired_rows_are_mirror_images(self) -> None:
        for value_index, row in enumerate(self.price_grids):
            mirror = self.price_grids[-1 - value_index]
            for price_index, price in enumerate(row):
                self.assertAlmostEqual(
                    price + mirror[-1 - price_index],
                    2.0 * self.parameters.value_mean,
                    places=10,
                )

    def test_official_rows_match_step13b_resolution_diagnostic(self) -> None:
        diagnostic_rows = build_resolution_table(
            self.parameters,
            self.parameters.noise_std,
        )
        for value_index, (row, diagnostic) in enumerate(
            zip(self.price_grids, diagnostic_rows, strict=True)
        ):
            with self.subTest(value_index=value_index):
                self.assertAlmostEqual(
                    row[1] - row[0],
                    float(diagnostic["one_price_grid_step"]),
                    places=12,
                )


if __name__ == "__main__":
    unittest.main()
