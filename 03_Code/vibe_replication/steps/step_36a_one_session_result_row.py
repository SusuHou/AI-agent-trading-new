"""Step 36A: turn one completed session into one auditable result row.

步骤 36A：把一个完整 session 变成一行可审查的实验结果。

Run / 运行:
    py -3 -X utf8 steps/step_36a_one_session_result_row.py

Important boundary / 重要边界:
    This file runs an ENGINEERING SMOKE TEST with a deliberately stable,
    synthetic policy.  It checks that Steps 26--33 connect correctly and that
    one result can be saved.  Its numbers are not evidence about the paper.
    / 本文件运行“工程冒烟测试”，故意使用稳定的合成策略。它只检查第 26--33
    步能否正确连接，以及结果能否保存；输出数字不是论文复现证据。

Why start here? / 为什么先做这一步？
    A paper experiment eventually produces thousands of session-level rows.
    Before spending hours or days training Q-learners, we first prove that one
    row has the right provenance, metrics, and output format.
    / 正式实验最终会产生数千行 session 结果。在花费数小时或数天训练
    Q-learner 之前，我们先证明一行结果的来源、指标和保存格式都是正确的。
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import csv
import json
import os
import sys
import tempfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEPS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STEPS_DIRECTORY))


from src.parameters import PaperParameters
from step_25_one_market_period import build_paper_inputs
from step_26_reproducible_random_streams import build_randomized_paper_session
from step_28_session_phases import SessionPhaseController
from step_29_matched_path_collusion_profitability import (
    MatchedPathCollusionScorer,
    build_matched_path_benchmarks,
)
from step_30_trading_intensity import (
    OnlineTradingIntensityScorer,
    build_measurement_sink_fanout,
)
from step_31_price_informativeness import build_price_informativeness_receipt
from step_32_market_liquidity import OnlineMarketLiquidityScorer
from step_33_mispricing import DeferredOnlineMispricingScorer


SCHEMA_VERSION = "step36a-one-session-row-v2"
PRICE_GRID_ENCODING = "value_specific_by_lagged_value_v1"
SMOKE_MODE = "engineering_smoke_not_research_result"
SMOKE_POLICY_SOURCE = "synthetic_stable_action_index_0"
SMOKE_CONVERGENCE_PERIODS = 5
SMOKE_MEASUREMENT_PERIODS = 200
DEFAULT_EXPERIMENT_SEED = 20260829


@dataclass(frozen=True)
class OneSessionResultRow:
    """A small, immutable table row for one completed session.

    一个完整 session 对应的一行小型、不可修改的表格记录。

    ``frozen=True`` is appropriate here because this object is a finished
    historical result, not a learning agent. / 这里使用 ``frozen=True``，因为
    它是已经完成的历史结果，而不是还要继续学习和变化的 agent。
    """

    schema_version: str
    run_id: str
    mode: str
    research_result: bool
    paper_scale: bool
    policy_source: str
    experiment_cell_key: str
    config_hash: str
    price_grid_encoding: str
    price_grid_sha256: str
    experiment_seed: int
    session_index: int
    session_seed: int
    noise_std: float
    investor_slope: float
    convergence_periods_required: int
    training_periods_completed: int
    policy_change_events: int
    measurement_periods_completed: int
    mean_actual_profit_agent_1: float
    mean_actual_profit_agent_2: float
    mean_nash_profit: float
    mean_cartel_profit: float
    delta_c: float
    trading_intensity: float
    price_informativeness: float
    market_liquidity: float
    mispricing: float | None
    mispricing_requires_research_decision: bool
    mechanism_label: str | None
    elapsed_seconds: float


def _canonical_config_hash(config: dict[str, object]) -> str:
    """Hash one sorted JSON configuration. / 对排序后的 JSON 配置生成指纹。"""

    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_engineering_smoke_row(
    *,
    parameters: PaperParameters | None = None,
    experiment_seed: int = DEFAULT_EXPERIMENT_SEED,
    session_index: int = 0,
) -> OneSessionResultRow:
    """Run one tiny end-to-end wiring check and return one result row.

    运行一个很小的端到端连接检查，并返回一行结果。

    The Q-table is intentionally replaced by an unmistakably stable policy:
    action index 0 has a huge Q-value in every state.  Therefore this function
    tests the experiment plumbing, not Q-learning's scientific behavior.
    / Q 表被故意替换成一个明显稳定的策略：每个状态下动作 0 的 Q 值都极大。
    因此，本函数测试的是实验管线，而不是 Q-learning 的科学行为。
    """

    if parameters is None:
        parameters = PaperParameters()
    if not isinstance(parameters, PaperParameters):
        raise TypeError("parameters must be PaperParameters. / parameters 类型错误。")
    if isinstance(session_index, bool) or not isinstance(session_index, int) or session_index < 0:
        raise ValueError("session_index must be a non-negative integer. / session_index 必须是非负整数。")

    (
        value_grid,
        price_grid,
        action_multipliers,
        initial_q_table,
        prehistory,
    ) = build_paper_inputs(parameters)

    # A very large gap guarantees that ordinary smoke-test updates cannot
    # change the greedy action. / 极大的 Q 值差距保证普通冒烟更新不会改变贪心动作。
    stable_q_table = np.zeros_like(initial_q_table, dtype=float)
    stable_q_table[:, 0] = 1_000_000_000.0

    experiment_cell_key = (
        f"figure2a_smoke|sigma_u={parameters.noise_std}"
        f"|xi={parameters.investor_slope}|policy={SMOKE_POLICY_SOURCE}"
    )
    # Session index is deliberately excluded: every independent session in the
    # same experiment cell must share one config hash. / 故意不把 session 编号
    # 放入配置指纹；同一个实验单元中的独立 session 必须共享同一个 config hash。
    config = {
        "schema_version": SCHEMA_VERSION,
        "mode": SMOKE_MODE,
        "policy_source": SMOKE_POLICY_SOURCE,
        "price_grid_encoding": PRICE_GRID_ENCODING,
        "parameters": asdict(parameters),
        "experiment_seed": experiment_seed,
        "convergence_periods_required": SMOKE_CONVERGENCE_PERIODS,
        "measurement_periods_required": SMOKE_MEASUREMENT_PERIODS,
    }
    price_grid_sha256 = sha256(
        json.dumps(
            price_grid,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    config["price_grid_sha256"] = price_grid_sha256
    config_hash = _canonical_config_hash(config)

    session = build_randomized_paper_session(
        parameters=parameters,
        value_grid=value_grid,
        price_grid=price_grid,
        action_multipliers=action_multipliers,
        initial_q_table=stable_q_table,
        prehistory=prehistory,
        experiment_seed=experiment_seed,
        experiment_cell_key=experiment_cell_key,
        session_index=session_index,
    )

    # Each scorer receives the same 200 measurement rows. / 每个 scorer 都接收
    # 完全相同的 200 条测量记录。
    profit_scorer = MatchedPathCollusionScorer(
        session,
        build_matched_path_benchmarks(parameters, value_grid),
    )
    intensity_scorer = OnlineTradingIntensityScorer(session)
    liquidity_scorer = OnlineMarketLiquidityScorer(session)
    mispricing_scorer = DeferredOnlineMispricingScorer(session)
    measurement_sink = build_measurement_sink_fanout(
        profit_scorer.observe,
        intensity_scorer.observe,
        liquidity_scorer.observe,
        mispricing_scorer.observe,
    )

    controller = SessionPhaseController.create_for_fresh_session(
        session,
        convergence_periods_required=SMOKE_CONVERGENCE_PERIODS,
        measurement_periods_required=SMOKE_MEASUREMENT_PERIODS,
        measurement_sink=measurement_sink,
    )

    started = perf_counter()
    phase_receipt = controller.run_until_complete(
        maximum_training_periods=SMOKE_CONVERGENCE_PERIODS,
    )
    elapsed_seconds = perf_counter() - started

    profitability = profit_scorer.finalize(controller)
    intensity = intensity_scorer.finalize(controller)
    informativeness = build_price_informativeness_receipt(
        intensity_scorer,
        controller,
    )
    liquidity = liquidity_scorer.finalize(controller)
    mispricing = mispricing_scorer.finalize(intensity_scorer, controller)

    if len(profitability.mean_actual_profits) != 2:
        raise RuntimeError("Step 36A expects the paper baseline I=2. / Step 36A 要求论文基准 I=2。")

    convergence = phase_receipt.convergence_receipt
    manifest = session.streams.manifest
    run_id = f"smoke-{config_hash[:12]}-session-{session_index}"
    return OneSessionResultRow(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        mode=SMOKE_MODE,
        research_result=False,
        paper_scale=False,
        policy_source=SMOKE_POLICY_SOURCE,
        experiment_cell_key=experiment_cell_key,
        config_hash=config_hash,
        price_grid_encoding=PRICE_GRID_ENCODING,
        price_grid_sha256=price_grid_sha256,
        experiment_seed=manifest.experiment_seed,
        session_index=manifest.session_index,
        session_seed=manifest.session_seed,
        noise_std=parameters.noise_std,
        investor_slope=parameters.investor_slope,
        convergence_periods_required=convergence.required_unchanged_periods,
        training_periods_completed=convergence.training_periods_completed,
        policy_change_events=convergence.policy_change_events,
        measurement_periods_completed=phase_receipt.measurement_periods_completed,
        mean_actual_profit_agent_1=profitability.mean_actual_profits[0],
        mean_actual_profit_agent_2=profitability.mean_actual_profits[1],
        mean_nash_profit=profitability.mean_nash_profit,
        mean_cartel_profit=profitability.mean_cartel_profit,
        delta_c=profitability.delta_c,
        trading_intensity=intensity.average_trading_intensity,
        price_informativeness=informativeness.price_informativeness,
        market_liquidity=liquidity.average_market_liquidity,
        mispricing=mispricing.reported_average_mispricing,
        mispricing_requires_research_decision=(
            mispricing.requires_explicit_research_decision
        ),
        # Steps 35E--F are intentionally not invoked by this synthetic smoke
        # row, so a mechanism label would be scientifically invalid. / 本合成
        # 冒烟记录有意不调用第 35E--F 步，因此不能伪造机制标签。
        mechanism_label=None,
        elapsed_seconds=elapsed_seconds,
    )


def _atomic_text_write(path: Path, text: str) -> None:
    """Write beside the target, then atomically replace it. / 先写临时文件，再原子替换目标。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_result_row(
    row: OneSessionResultRow,
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    """Save the same row as audit-friendly JSON and table-friendly CSV.

    把同一行同时保存为便于审计的 JSON 和便于制表的 CSV。
    """

    if not isinstance(row, OneSessionResultRow):
        raise TypeError("row has the wrong type. / row 类型错误。")
    row_dictionary = asdict(row)
    json_text = json.dumps(
        row_dictionary,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"

    from io import StringIO

    csv_buffer = StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(row_dictionary))
    writer.writeheader()
    writer.writerow(row_dictionary)

    _atomic_text_write(json_path, json_text)
    _atomic_text_write(csv_path, csv_buffer.getvalue())


def main() -> None:
    """Run the smoke pipeline, save one row, and explain the boundary.

    运行冒烟管线、保存一行结果，并说明它的边界。
    """

    output_directory = PROJECT_ROOT / "results" / "step36a_engineering_smoke"
    json_path = output_directory / "one_session_result.json"
    csv_path = output_directory / "one_session_result.csv"

    row = build_engineering_smoke_row()
    save_result_row(row, json_path=json_path, csv_path=csv_path)

    print("Step 36A: one session -> one result row / 第 36A 步：一个 session → 一行结果")
    print(f"Mode / 模式: {row.mode}")
    print(f"Training periods / 训练期数: {row.training_periods_completed}")
    print(f"Measurement periods / 测量期数: {row.measurement_periods_completed}")
    print(f"Delta^C: {row.delta_c:.6f}")
    print(f"Trading intensity chi_hat^C / 交易强度: {row.trading_intensity:.6f}")
    print(f"Price informativeness / 价格信息效率: {row.price_informativeness:.6f}")
    print(f"Market liquidity / 市场流动性: {row.market_liquidity:.6f}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(
        "Boundary / 边界: pipeline validation only; not a paper replication result. "
        "/ 只验证实验管线，不是论文复现结果。"
    )


if __name__ == "__main__":
    main()
