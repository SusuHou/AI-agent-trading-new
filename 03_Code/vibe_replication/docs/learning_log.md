# Replication Learning Log / 复现学习日志

This file records questions raised while rebuilding the paper, the evidence we
observed, and the conclusion we agreed on. / 本文件记录复现过程中提出的问题、实际
观察到的证据，以及最终确认的结论。

---

## 2026-08-27 — Why did the sample mean grow in the high-noise environment? / 为什么高噪声环境的样本均值也变大了？

### Question / 问题

After changing the noise-trader standard deviation from `sigma_u=0.1` to
`sigma_u=100`, the sample mean increased from `0.000226` to `0.226079`. Did the
noise trader's true mean change? / 把噪声交易者的标准差从 `0.1` 改为 `100` 后，
样本均值从 `0.000226` 变成 `0.226079`。噪声交易者的真实均值是否改变？

### Observed output / 实际输出

| Environment / 环境 | `sigma_u` | Sample mean / 样本均值 | Sample std / 样本标准差 |
|---|---:|---:|---:|
| Low noise / 低噪声 | 0.1 | 0.000226 | 0.100263 |
| High noise / 高噪声 | 100 | 0.226079 | 100.262576 |

Both runs used seed `42` and 100,000 validation draws. / 两次运行都使用随机种子
`42` 和 100,000 次验证抽样。

### Explanation / 解释

The paper's population mean remains:

`E[u_t] = 0`.

The displayed number is a finite-sample mean, so random positive and negative
draws do not cancel exactly. Using the same seed, changing `sigma_u` from `0.1`
to `100` scales the same standardized draws by 1,000. The small leftover sample
mean therefore also scales by about 1,000. / 论文的总体均值仍然是 `0`。屏幕显示的
是有限样本均值，随机的正数与负数不会刚好完全抵消。相同种子下，`sigma_u` 从
`0.1` 增加到 `100`，同一组标准化随机数被放大 1,000 倍，剩余的样本均值也会
大约放大 1,000 倍。

Relative to the noise scale, the two means are the same:

`0.000226 / 0.1 ~= 0.226079 / 100 ~= 0.00226`.

For 100,000 draws, the standard error of the high-noise sample mean is:

`100 / sqrt(100000) ~= 0.316`.

The observed sample mean `0.226079` is smaller than this typical sampling-error
scale, so it is consistent with a true mean of zero. / 高噪声样本均值的标准误约为
`0.316`，观察到的 `0.226079` 小于这一典型抽样误差尺度，因此与真实均值为零
完全一致。

### Conclusion / 结论

- Population mean / 总体理论均值: still `0` / 仍为 `0`.
- Sample mean / 有限样本均值: not exactly zero and scales with `sigma_u` / 不会精确为零，并随 `sigma_u` 成比例放大。
- Validation / 验证结论: passed / 通过。

### Related code / 相关代码

`../steps/step_02_noise_trader_high noise.py`
