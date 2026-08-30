"""Step 31: compute the paper's price-informativeness metric.

第 31 步：计算论文的价格信息效率指标。

Run the short hand-checkable demonstration / 运行简短手算演示:
    py -3 -X utf8 steps/step_31_price_informativeness.py

Run the separate automated tests / 运行独立自动测试:
    py -3 -X utf8 -m unittest discover -s tests \
        -p "test_step31_price_informativeness.py" -v

Paper rule, Online Appendix equation IA.4.5 / 论文规则，在线附录 IA.4.5:

    price informativeness = (I * chi_hat_C)^2 * (sigma_v_hat / sigma_u)^2

This is one scalar for one completed simulation session. It is a
signal-to-noise ratio / 每个完整 simulation session 得到一个标量，它是信噪比:

    informed-flow variance / 知情订单流方差
    ------------------------------------------------
    noise-order variance / 噪声订单方差

Important / 重要:
    - ``chi_hat_C`` comes from Step 30. / chi_hat_C 来自 Step 30。
    - ``sigma_v_hat`` is the standard deviation of the discrete value grid,
      not the continuous calibration ``sigma_v`` and not the realized sample
      standard deviation. / sigma_v_hat 是离散价值网格的标准差，不是连续
      校准 sigma_v，也不是本次随机路径的样本标准差。
    - ``sigma_u`` is the configured noise-trader standard deviation, not the
      realized sample standard deviation. / sigma_u 是设定的噪声交易者标准差，
      不是本次路径实现出来的样本标准差。
    - A signal-to-noise ratio is not bounded by 1. Never clip it.
      / 信噪比不以 1 为上限，绝不能把结果截断到 0 至 1。
"""

from dataclasses import dataclass
from math import fsum, isclose, isfinite
from numbers import Integral
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.step01_value_grid import discrete_value_std
from step_26_reproducible_random_streams import (
    RandomizedMarketSession,
    SessionSeedManifest,
)
from step_28_session_phases import (
    SessionPhase,
    SessionPhaseController,
)
from step_30_trading_intensity import (
    ESTIMATOR_VERSION,
    REGRESSION_SPECIFICATION,
    OnlineTradingIntensityScorer,
)


PRICE_INFORMATIVENESS_FORMULA = (
    "I_C = (I * chi_hat_C)^2 * (sigma_v_hat / sigma_u)^2"
)
PRICE_INFORMATIVENESS_VERSION = "appendix-ia.4.5-v1"


class UndefinedPriceInformativenessError(ArithmeticError):
    """The signal-to-noise ratio is mathematically undefined.

    信噪比在数学上没有定义。
    """


@dataclass(frozen=True)
class PriceInformativenessCalculation:
    """A small immutable calculation without session metadata.

    一份不含 session 元数据的小型不可修改计算结果。
    """

    number_of_agents: int
    average_trading_intensity: float
    aggregate_informed_slope: float
    discrete_value_std: float
    noise_std: float
    standard_deviation_ratio: float
    informed_flow_variance: float
    noise_order_variance: float
    price_informativeness: float


def calculate_price_informativeness(
    number_of_agents: int,
    average_trading_intensity: float,
    discrete_value_standard_deviation: float,
    noise_standard_deviation: float,
) -> PriceInformativenessCalculation:
    """Apply equation IA.4.5 to four explicit inputs.

    对四个明确输入直接应用公式 IA.4.5。

    The separate function makes the economic arithmetic easy to test by hand.
    / 把纯公式单独写成函数，方便人工手算检查。

    All displayed intermediate variances must also fit in an ordinary Python
    float. Extremely large canceling scales are rejected because their audit
    fields could not be stored consistently. / 所有展示出来的中间方差也必须能由
    普通 Python 浮点数保存；即使极端大数在比率中碰巧抵消，只要审计字段无法
    一致保存，就明确拒绝。
    """

    if (
        isinstance(number_of_agents, bool)
        or not isinstance(number_of_agents, Integral)
        or int(number_of_agents) < 1
    ):
        raise ValueError("number_of_agents must be positive. / agent 数量必须为正整数。")
    agent_count = int(number_of_agents)

    try:
        intensity = float(average_trading_intensity)
        value_std_hat = float(discrete_value_standard_deviation)
        noise_std = float(noise_standard_deviation)
    except (TypeError, ValueError) as error:
        raise ValueError("All numerical inputs must be real numbers. / 全部数值输入必须是实数。") from error

    if not isfinite(intensity):
        raise ValueError("average_trading_intensity must be finite. / 平均交易强度必须有限。")
    if not isfinite(value_std_hat) or value_std_hat <= 0.0:
        raise UndefinedPriceInformativenessError(
            "Discrete value std must be positive and finite. "
            "/ 离散价值标准差必须为有限正数。"
        )
    if not isfinite(noise_std) or noise_std <= 0.0:
        raise UndefinedPriceInformativenessError(
            "Noise std must be positive and finite. / 噪声标准差必须为有限正数。"
        )

    aggregate_slope = agent_count * intensity
    std_ratio = value_std_hat / noise_std
    informed_flow_scale = aggregate_slope * value_std_hat
    informed_variance = informed_flow_scale * informed_flow_scale
    noise_variance = noise_std * noise_std
    scaled_signal_to_noise = aggregate_slope * std_ratio
    formula_informativeness = scaled_signal_to_noise * scaled_signal_to_noise
    outputs = (
        aggregate_slope,
        std_ratio,
        informed_flow_scale,
        informed_variance,
        noise_variance,
        scaled_signal_to_noise,
        formula_informativeness,
    )
    if (
        not all(isfinite(number) for number in outputs)
        or noise_variance <= 0.0
        or (
            aggregate_slope != 0.0
            and value_std_hat != 0.0
            and informed_flow_scale == 0.0
        )
        or (informed_flow_scale != 0.0 and informed_variance == 0.0)
        or (value_std_hat != 0.0 and std_ratio == 0.0)
        or (
            aggregate_slope != 0.0
            and std_ratio != 0.0
            and scaled_signal_to_noise == 0.0
        )
        or (
            scaled_signal_to_noise != 0.0
            and formula_informativeness == 0.0
        )
    ):
        raise OverflowError(
            "Price-informativeness calculation overflowed or underflowed. "
            "/ 价格信息效率计算发生溢出或下溢。"
        )

    variance_ratio = informed_variance / noise_variance
    if not isfinite(variance_ratio) or not isclose(
        variance_ratio,
        formula_informativeness,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise OverflowError(
            "The two equivalent informativeness calculations lost numerical "
            "agreement. / 两种等价的信息效率算法失去数值一致性。"
        )

    return PriceInformativenessCalculation(
        number_of_agents=agent_count,
        average_trading_intensity=intensity,
        aggregate_informed_slope=aggregate_slope,
        discrete_value_std=value_std_hat,
        noise_std=noise_std,
        standard_deviation_ratio=std_ratio,
        informed_flow_variance=informed_variance,
        noise_order_variance=noise_variance,
        price_informativeness=variance_ratio,
    )


@dataclass(frozen=True)
class PriceInformativenessReceipt:
    """Auditable Step-31 result for one completed Step-30 session.

    一个已完成 Step-30 session 的可审计 Step-31 结果。
    """

    measurement_periods_scored: int
    first_measurement_index: int
    last_measurement_index: int
    first_global_period_index: int
    last_global_period_index: int
    number_of_agents: int
    slope_by_agent: tuple[float, ...]
    average_trading_intensity: float
    aggregate_informed_slope: float
    discrete_value_std: float
    continuous_value_std_parameter: float
    noise_std: float
    standard_deviation_ratio: float
    informed_flow_variance: float
    noise_order_variance: float
    price_informativeness: float
    value_grid_points: int
    value_grid: tuple[float, ...]
    session_seed_manifest: SessionSeedManifest
    source_step30_estimator_version: str
    source_step30_regression_specification: str
    formula: str
    calculation_version: str
    uses_discrete_value_grid_std: bool
    uses_configured_noise_std: bool


def build_price_informativeness_receipt(
    trading_intensity_scorer: OnlineTradingIntensityScorer,
    controller: SessionPhaseController,
) -> PriceInformativenessReceipt:
    """Validate provenance and convert one Step-30 result into Step 31.

    核对数据来源，再把一份 Step-30 结果转换为 Step 31。

    Step 31 does not replay 100,000 rows. It asks the bound Step-30 scorer to
    finalize through its exact Step-28 controller, then uses that same session's
    fixed grid/noise parameters. / Step 31 不重新播放十万行；它让绑定的 Step-30
    scorer 通过对应的 Step-28 controller 完成，再读取同一 session 的固定网格和噪声参数。
    """

    if not isinstance(trading_intensity_scorer, OnlineTradingIntensityScorer):
        raise TypeError("trading_intensity_scorer has the wrong type. / Step-30 scorer 类型错误。")
    if not isinstance(controller, SessionPhaseController):
        raise TypeError("controller has the wrong type. / controller 类型错误。")
    if controller.phase is not SessionPhase.COMPLETE:
        raise RuntimeError("Step 28 is not complete. / Step 28 尚未完成。")

    # Step 30 itself checks object identity between its scorer and controller.
    # This is stronger than comparing only seed labels. / Step 30 会检查 scorer
    # 与 controller 的对象身份；这比只比较种子标签更严格。
    trading_intensity = trading_intensity_scorer.finalize(controller)
    session = controller.session
    if not isinstance(session, RandomizedMarketSession):
        raise TypeError("controller session has the wrong type. / controller session 类型错误。")
    if session.execution_mode != "complete":
        raise RuntimeError("The bound market session is not complete. / 绑定的市场 session 尚未完成。")
    if trading_intensity.session_seed_manifest != session.streams.manifest:
        raise RuntimeError("Step 30 belongs to another seeded session. / Step 30 属于另一随机种子 session。")

    if session.parameters != trading_intensity.parameter_snapshot:
        raise RuntimeError(
            "The live session parameters differ from the pre-training Step-30 "
            "snapshot. / live session 参数与 Step-30 训练前快照不一致。"
        )
    live_value_grid = tuple(float(value) for value in session.value_grid)
    if live_value_grid != trading_intensity.value_grid_snapshot:
        raise RuntimeError(
            "The live value grid differs from the pre-training Step-30 snapshot. "
            "/ live 价值网格与 Step-30 训练前快照不一致。"
        )

    parameters = trading_intensity.parameter_snapshot
    if trading_intensity.number_of_agents != parameters.num_speculators:
        raise RuntimeError("Step-30 agent count disagrees with the session. / Step-30 agent 数量与 session 不一致。")
    if not isclose(
        trading_intensity.value_mean_parameter,
        parameters.value_mean,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Step-30 value mean disagrees with the session. / Step-30 价值均值与 session 不一致。")
    if (
        not trading_intensity.unrestricted_intercept_estimated
        or not trading_intensity.actual_raw_orders_used
        or trading_intensity.regression_specification
        != REGRESSION_SPECIFICATION
        or trading_intensity.estimator_version != ESTIMATOR_VERSION
    ):
        raise RuntimeError("Step-30 estimator provenance is incompatible. / Step-30 估计器来源不兼容。")

    row_count = trading_intensity.measurement_periods_scored
    if (
        row_count < 2
        or trading_intensity.first_measurement_index != 0
        or trading_intensity.last_measurement_index != row_count - 1
        or trading_intensity.last_global_period_index
        - trading_intensity.first_global_period_index
        + 1
        != row_count
        or session.period_number != trading_intensity.last_global_period_index + 1
    ):
        raise RuntimeError("Step-30 measurement boundaries are inconsistent. / Step-30 测量边界不一致。")

    slopes = trading_intensity.slope_by_agent
    if len(slopes) != parameters.num_speculators or not all(
        isfinite(slope) for slope in slopes
    ):
        raise RuntimeError("Step-30 slopes are incomplete or invalid. / Step-30 斜率不完整或无效。")
    recomputed_average = fsum(slopes) / parameters.num_speculators
    if not isclose(
        trading_intensity.average_trading_intensity,
        recomputed_average,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Step-30 average slope is internally inconsistent. / Step-30 平均斜率内部不一致。")
    aggregate_slope_from_agents = fsum(slopes)
    if not isclose(
        aggregate_slope_from_agents,
        parameters.num_speculators
        * trading_intensity.average_trading_intensity,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Step-30 aggregate slope is inconsistent. / Step-30 总斜率不一致。")

    value_grid = np.asarray(trading_intensity.value_grid_snapshot, dtype=float)
    if (
        value_grid.ndim != 1
        or value_grid.size != parameters.num_value_points
        or not np.all(np.isfinite(value_grid))
    ):
        raise RuntimeError("The session value grid is invalid. / session 价值网格无效。")
    grid_mean = float(np.mean(value_grid))
    if not isclose(
        grid_mean,
        parameters.value_mean,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "The value-grid mean disagrees with v_bar, so its grid standard "
            "deviation is not defined as required by the paper. "
            "/ 价值网格均值与 v_bar 不一致，无法按论文定义网格标准差。"
        )
    value_std_hat = discrete_value_std(value_grid, parameters.value_mean)
    if not isclose(
        value_std_hat,
        trading_intensity.discrete_value_std_snapshot,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError(
            "The stored Step-30 grid standard deviation is inconsistent. "
            "/ Step-30 保存的网格标准差内部不一致。"
        )
    calculation = calculate_price_informativeness(
        parameters.num_speculators,
        trading_intensity.average_trading_intensity,
        value_std_hat,
        parameters.noise_std,
    )

    return PriceInformativenessReceipt(
        measurement_periods_scored=row_count,
        first_measurement_index=trading_intensity.first_measurement_index,
        last_measurement_index=trading_intensity.last_measurement_index,
        first_global_period_index=trading_intensity.first_global_period_index,
        last_global_period_index=trading_intensity.last_global_period_index,
        number_of_agents=calculation.number_of_agents,
        slope_by_agent=tuple(float(slope) for slope in slopes),
        average_trading_intensity=calculation.average_trading_intensity,
        aggregate_informed_slope=calculation.aggregate_informed_slope,
        discrete_value_std=calculation.discrete_value_std,
        continuous_value_std_parameter=parameters.value_std,
        noise_std=calculation.noise_std,
        standard_deviation_ratio=calculation.standard_deviation_ratio,
        informed_flow_variance=calculation.informed_flow_variance,
        noise_order_variance=calculation.noise_order_variance,
        price_informativeness=calculation.price_informativeness,
        value_grid_points=int(value_grid.size),
        value_grid=tuple(float(value) for value in value_grid),
        session_seed_manifest=trading_intensity.session_seed_manifest,
        source_step30_estimator_version=trading_intensity.estimator_version,
        source_step30_regression_specification=(
            trading_intensity.regression_specification
        ),
        formula=PRICE_INFORMATIVENESS_FORMULA,
        calculation_version=PRICE_INFORMATIVENESS_VERSION,
        uses_discrete_value_grid_std=True,
        uses_configured_noise_std=True,
    )


def main() -> None:
    """Run an artificial example with an exact answer of 25.

    运行一个精确答案为 25 的人工例子。
    """

    # From the Step-30 hand example: average chi = (2 + 0.5) / 2 = 1.25.
    # / 来自 Step-30 手算例：平均 chi = (2 + 0.5) / 2 = 1.25。
    calculation = calculate_price_informativeness(
        number_of_agents=2,
        average_trading_intensity=1.25,
        discrete_value_standard_deviation=1.0,
        noise_standard_deviation=0.5,
    )

    # Total informed slope = 2 * 1.25 = 2.5.
    # Signal variance = (2.5 * 1)^2 = 6.25.
    # Noise variance = 0.5^2 = 0.25.
    # Information ratio = 6.25 / 0.25 = 25.
    # / 总知情斜率=2.5；信号方差=6.25；噪声方差=0.25；信噪比=25。
    assert isclose(calculation.aggregate_informed_slope, 2.5, abs_tol=1e-12)
    assert isclose(calculation.informed_flow_variance, 6.25, abs_tol=1e-12)
    assert isclose(calculation.noise_order_variance, 0.25, abs_tol=1e-12)
    assert isclose(calculation.price_informativeness, 25.0, abs_tol=1e-12)

    print("Step 31: price informativeness / 第 31 步：价格信息效率")
    print("Artificial validation inputs / 人工验证输入:")
    print("  I = 2")
    print("  chi_hat_C = 1.25")
    print("  sigma_v_hat = 1.0")
    print("  sigma_u = 0.5")
    print(
        "Aggregate informed slope I*chi_hat_C / 总知情订单斜率: "
        f"{calculation.aggregate_informed_slope:.2f}"
    )
    print(
        "Informed-flow variance / 知情订单流方差: "
        f"{calculation.informed_flow_variance:.2f}"
    )
    print(
        "Noise-order variance / 噪声订单方差: "
        f"{calculation.noise_order_variance:.2f}"
    )
    print(
        "Price informativeness / 价格信息效率: "
        f"{calculation.price_informativeness:.2f}"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
