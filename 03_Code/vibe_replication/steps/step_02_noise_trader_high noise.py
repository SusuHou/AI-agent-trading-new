"""Step 2: create the paper's noise-trader order.
第二步：生成论文中的噪声交易者订单。

Run / 运行:
    py -3 steps/step_02_noise_trader.py
"""

import random
import statistics


# Try the paper's high-noise market. / 尝试论文的高噪声市场。
noise_std = 100.0     # sigma_u: standard deviation / 标准差
random_seed = 42      # makes the random sequence repeatable / 让随机序列可重复

# A Random object is our reproducible random-number generator.
# Random 对象是一个可复现的随机数生成器。
random_generator = random.Random(random_seed)

# Paper rule: u_t follows N(0, sigma_u^2).
# Python's gauss() asks for the mean and STANDARD DEVIATION, not the variance.
# 论文规则：u_t 服从 N(0, sigma_u^2)。
# Python 的 gauss() 输入平均值和标准差，而不是方差。
noise_order = random_generator.gauss(mu=0.0, sigma=noise_std)

print(f"Random seed / 随机种子: {random_seed}")
print(f"Noise standard deviation sigma_u / 噪声标准差: {noise_std}")
print(f"One noise order u_t / 一个噪声订单: {noise_order:.6f}")


# Validation 1: the same seed must reproduce the same first order.
# 验证 1：相同种子必须产生相同的第一个订单。
second_generator = random.Random(random_seed)
repeated_order = second_generator.gauss(mu=0.0, sigma=noise_std)
assert noise_order == repeated_order


# Validation 2: many draws should have mean near 0 and std near sigma_u.
# 验证 2：大量抽样的平均值应接近 0，标准差应接近 sigma_u。
validation_generator = random.Random(random_seed)
sample_size = 100_000
sample = [
    validation_generator.gauss(mu=0.0, sigma=noise_std)
    for _ in range(sample_size)
]

sample_mean = statistics.fmean(sample)
sample_std = statistics.pstdev(sample)

print(f"Validation sample size / 验证样本数: {sample_size}")
print(f"Sample mean (target 0) / 样本均值: {sample_mean:.6f}")
print(f"Sample std (target {noise_std}) / 样本标准差: {sample_std:.6f}")

# Use a 2% tolerance relative to sigma_u, so this validation works for
# both sigma_u=0.1 and sigma_u=100. / 使用相对 sigma_u 的 2% 容差，
# 因此低噪声和高噪声环境都能使用同一个验证方法。
tolerance = 0.02 * noise_std
assert abs(sample_mean) < tolerance
assert abs(sample_std - noise_std) < tolerance

print("Validation passed / 验证通过")
