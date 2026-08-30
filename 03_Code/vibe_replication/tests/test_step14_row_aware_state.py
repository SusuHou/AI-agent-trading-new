"""Regression tests for row-aware price-state mapping. / 按价值选行的状态映射测试。"""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = PROJECT_ROOT / "steps"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from step_14_state_representation import (
    build_state_indexes,
    continuous_price_to_index,
    number_of_price_points,
    number_of_states,
    validate_price_grids_by_value,
)


class TestStep14RowAwareState(unittest.TestCase):
    """Use small hand-checkable rows so a wrong row is immediately visible."""

    def setUp(self) -> None:
        self.values = (0.0, 1.0, 2.0)
        self.price_grids = (
            (0.0, 10.0, 20.0),
            (100.0, 110.0, 120.0),
            (200.0, 210.0, 220.0),
        )

    def test_previous_value_selects_the_price_row(self) -> None:
        self.assertEqual(
            build_state_indexes(
                111.0,
                1.0,
                2.0,
                self.price_grids,
                self.values,
            ),
            (1, 1, 2),
        )
        self.assertEqual(
            build_state_indexes(
                111.0,
                2.0,
                1.0,
                self.price_grids,
                self.values,
            ),
            (0, 2, 1),
        )

    def test_next_state_uses_current_value_as_its_previous_value(self) -> None:
        # s_(t+1)=(p_t,v_t,v_(t+1)); v_t=2 selects the third row.
        self.assertEqual(
            build_state_indexes(
                211.0,
                2.0,
                0.0,
                self.price_grids,
                self.values,
            ),
            (1, 2, 0),
        )

    def test_every_point_round_trips_inside_its_own_row(self) -> None:
        for row in self.price_grids:
            for expected_index, price in enumerate(row):
                self.assertEqual(
                    continuous_price_to_index(price, row),
                    expected_index,
                )

    def test_midpoint_tie_and_clipping_are_row_local(self) -> None:
        row = self.price_grids[1]
        self.assertEqual(continuous_price_to_index(115.0, row), 1)
        self.assertEqual(continuous_price_to_index(115.000001, row), 2)
        self.assertEqual(continuous_price_to_index(99.0, row), 0)
        self.assertEqual(continuous_price_to_index(121.0, row), 2)

    def test_state_count_does_not_gain_an_extra_dimension(self) -> None:
        self.assertEqual(number_of_states(31, 10), 3_100)
        self.assertEqual(number_of_price_points(self.price_grids), 3)

    def test_flat_or_malformed_grids_are_rejected(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            validate_price_grids_by_value([0.0, 1.0, 2.0], 3, 3)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            validate_price_grids_by_value(self.price_grids[:2], 3, 3)
        with self.assertRaises(ValueError):
            validate_price_grids_by_value(
                ((0.0, 10.0, 20.0), (100.0, 90.0, 120.0), (200.0, 210.0, 220.0)),
                3,
                3,
            )


if __name__ == "__main__":
    unittest.main()
