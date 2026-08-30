"""Paper-based checks for Step 1. / 第一步的论文依据测试。"""

import unittest

import numpy as np

from src.parameters import PaperParameters
from src.step01_value_grid import build_value_grid, discrete_value_std


class TestStep01ValueGrid(unittest.TestCase):
    """All paper-based checks for the value grid. / 基本价值网格的全部论文依据检查。"""

    def setUp(self) -> None:
        """Build a fresh baseline grid before each test. / 每个测试前新建基准网格。"""
        self.parameters = PaperParameters()
        self.grid = build_value_grid(
            self.parameters.value_mean,
            self.parameters.value_std,
            self.parameters.num_value_points,
        )

    def test_baseline_parameters_match_paper(self) -> None:
        """Check values listed in Section 4.2. / 检查第 4.2 节参数。"""
        p = self.parameters
        self.assertEqual(p.num_speculators, 2)
        self.assertEqual(p.value_mean, 1.0)
        self.assertEqual(p.value_std, 1.0)
        self.assertEqual(p.noise_std, 0.1)
        self.assertEqual(p.investor_slope, 500.0)
        self.assertEqual(p.pricing_error_weight, 0.1)
        self.assertEqual(p.discount_factor, 0.95)
        self.assertEqual(p.num_value_points, 10)

    def test_grid_has_ten_ordered_points(self) -> None:
        """The paper uses n_v=10 ordered points. / 论文使用 10 个递增网格点。"""
        self.assertEqual(self.grid.size, 10)
        self.assertTrue(np.all(np.diff(self.grid) > 0))

    def test_grid_is_symmetric_around_mean(self) -> None:
        """Normal quantiles are symmetric around v_bar. / 正态分位数关于 v_bar 对称。"""
        deviations = self.grid - 1.0
        np.testing.assert_allclose(deviations, -deviations[::-1], atol=1e-12)
        self.assertAlmostEqual(float(self.grid.mean()), 1.0)

    def test_discrete_std_matches_paper(self) -> None:
        """Footnote 24 reports sigma_v_hat about 0.938. / 脚注 24 报告约为 0.938。"""
        self.assertAlmostEqual(discrete_value_std(self.grid, 1.0), 0.938, delta=5e-4)

    def test_invalid_inputs_are_rejected(self) -> None:
        """Clear errors expose configuration mistakes. / 清晰错误可暴露配置问题。"""
        with self.assertRaises(ValueError):
            build_value_grid(mean=1.0, std=0.0, num_points=10)
        with self.assertRaises(ValueError):
            build_value_grid(mean=1.0, std=1.0, num_points=1)


if __name__ == "__main__":
    unittest.main()
