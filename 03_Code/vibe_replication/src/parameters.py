"""Paper parameters. / 论文参数。

Keeping parameters in one object prevents a value from being silently changed
in one part of the simulation. / 把参数集中在一个对象中，可以避免某个参数在模拟的
不同位置被悄悄改动。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperParameters:
    """Baseline calibration from Section 4.2. / 第 4.2 节的基准校准。"""

    # Economic environment / 经济环境
    num_speculators: int = 2       # I: informed AI speculators / 知情 AI 投机者数量
    value_mean: float = 1.0        # v_bar: mean fundamental / 基本价值均值
    value_std: float = 1.0         # sigma_v: continuous standard deviation / 连续标准差
    noise_std: float = 0.1         # sigma_u: low-noise baseline / 低噪声基准
    investor_slope: float = 500.0  # xi: insensitive-investor demand slope / 需求斜率

    # Preferences / 偏好参数
    pricing_error_weight: float = 0.1  # theta: market-maker weight / 做市商权重
    discount_factor: float = 0.95      # rho: speculator discount factor / 投机者贴现因子

    # Learning hyperparameters / 学习超参数
    learning_rate: float = 0.01        # alpha: Q-learning rate / Q-learning 学习率
    exploration_decay: float = 5e-7   # beta: exploration decay / 探索衰减率

    # Discretization and memory / 离散化与记忆
    num_value_points: int = 10    # n_v / 基本价值网格点数
    num_action_points: int = 15   # n_x / 动作网格点数
    num_price_points: int = 31    # n_p / 价格网格点数
    grid_widening: float = 0.1    # iota / 网格扩展参数
    market_maker_window: int = 10_000  # T_m / 做市商滚动窗口

    def __post_init__(self) -> None:
        """Reject impossible inputs early. / 尽早拒绝不可能的输入。"""
        if self.num_speculators < 1:
            raise ValueError("num_speculators must be at least 1 / 投机者数量至少为 1")
        if self.value_std <= 0 or self.noise_std <= 0:
            raise ValueError("standard deviations must be positive / 标准差必须为正")
        if self.investor_slope < 0:
            raise ValueError("investor_slope xi must be non-negative / 投资者需求斜率 xi 必须非负")
        if self.pricing_error_weight <= 0:
            raise ValueError("pricing_error_weight theta must be positive / 定价误差权重 theta 必须为正")
        if self.num_value_points < 2:
            raise ValueError("num_value_points must be at least 2 / 价值网格至少需要 2 点")
        if not 0 < self.discount_factor < 1:
            raise ValueError("discount_factor must lie between 0 and 1 / 贴现因子必须在 0 和 1 之间")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must lie in (0, 1] / 学习率必须在 (0, 1] 内")
