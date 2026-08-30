"""Step 1: create the paper's fundamental-value grid.
第一步：建立论文中的基本价值网格。

Run / 运行:
    py -3 steps/step_01_fundamental_value_grid.py
"""

from statistics import NormalDist


# The three inputs from the paper. / 论文给出的三个输入。
value_mean = 1.0          # v_bar: long-run mean / 长期平均基本价值
value_std = 1.0           # sigma_v: volatility scale / 基本价值波动尺度
number_of_values = 10     # n_v: number of grid values / 网格点数量

# NormalDist gives us Phi inverse: probability -> normal-distribution value.
# NormalDist 提供 Phi 的反函数：把概率位置转换成正态分布数值。
normal_distribution = NormalDist()

# This list will hold the ten final values. / 这个列表保存最终的 10 个价值。
value_grid = []

for k in range(1, number_of_values + 1):
    # k is only the number of the value currently being calculated.
    # k 只是当前正在计算第几个价值的编号。
    probability = (2 * k - 1) / (2 * number_of_values)

    standard_normal_value = normal_distribution.inv_cdf(probability)
    fundamental_value = value_mean + value_std * standard_normal_value
    value_grid.append(fundamental_value)

    print(
        f"k={k:2d}  probability={probability:.2f}  "
        f"fundamental value={fundamental_value:.6f}"
    )

# Check the two numbers reported by the paper. / 检查论文报告的两个数值。
grid_mean = sum(value_grid) / number_of_values
grid_std = (
    sum((value - value_mean) ** 2 for value in value_grid)
    / number_of_values
) ** 0.5

print()
print(f"Number of values / 价值数量: {len(value_grid)}")
print(f"Grid mean / 网格平均值: {grid_mean:.6f}")
print(f"Discrete standard deviation / 离散标准差: {grid_std:.6f}")

