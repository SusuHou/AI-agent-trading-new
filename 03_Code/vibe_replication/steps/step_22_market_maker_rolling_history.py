"""Step 22: store the market maker's rolling history D_t.

步骤 22：保存做市商的滚动历史 D_t。

Run / 运行:
    py -3 -X utf8 steps/step_22_market_maker_rolling_history.py

Paper rule / 论文规则 (main paper, Section 4.1, PDF pages 22-23):

    D_t = {v_(t-tau), p_(t-tau), z_(t-tau), y_(t-tau)}
          for tau = 1, ..., T_m

    baseline T_m = 10,000

Here / 这里:
    v = realized fundamental value / 已实现基本价值
    p = continuous market price / 连续市场价格
    z = information-insensitive investors' order / 信息不敏感投资者订单
    y = informed traders plus noise order flow / 知情交易者加噪声的订单流

Timing is essential / 时间顺序非常重要:
    1. At period t, estimate from the EXISTING D_t (past periods only).
       / 在第 t 期，使用现有 D_t（只含过去时期）估计。
    2. Use current y_t to set current p_t.
       / 使用本期 y_t 决定本期 p_t。
    3. Only after p_t, z_t, and profits exist, append (v_t,p_t,z_t,y_t).
       / 只有在 p_t、z_t 和利润产生后，才加入 (v_t,p_t,z_t,y_t)。
    4. That new row is available in D_(t+1), never in its own pricing regression.
       / 新记录供 D_(t+1) 使用，绝不能参与决定它自己价格的回归。

REPLICATION DECISION A3 — PAPER SILENT / 复现决定 A3——原文缺失:
    Status: BASELINE IMPLEMENTED; SENSITIVITY PENDING
    / 状态：基准实现已完成；敏感性结果待完成

    The paper does not explain how the first T_m observations are generated.
    Our selected baseline is a Nash-consistent synthetic prehistory created
    before t=0. It contains exactly T_m rows generated from the value grid,
    a balanced Gaussian-noise grid, Nash informed orders, the Nash price rule,
    and z=-xi(p-v_bar). The market maker would receive only the resulting rows
    and would still estimate its coefficients by OLS. / 论文没有说明最初 T_m 条
    数据如何产生。建议的基准方案是在 t=0 前生成“与 Nash 一致的合成前历史”：
    使用价值网格、平衡的高斯噪声网格、Nash 知情订单、Nash 定价规则以及
    z=-xi(p-v_bar)，恰好生成 T_m 行。做市商只接收这些记录，仍须通过 OLS
    自己估计系数。

    This is OUR replication choice, not a claimed paper rule. Before final
    results, compare it with at least cartel and expanding-window starts. / 这是
    我们的复现选择，不是论文规则。正式结果前至少要与 cartel 初始化及扩展窗口
    初始化进行敏感性比较。

    This isolated Step-22 container still starts empty by design. The separately
    validated Step 24C initializer now constructs and preloads the selected
    history before first-period pricing. / 这个独立的第 22 步容器仍按设计从空历史
    开始；已经验证的第 24C 步现在会在第一期定价前构造并载入选定前历史。

This step stores data only. Step 23 will run OLS; Step 24 will set adaptive
prices. / 本步骤只保存数据；第 23 步才运行 OLS，第 24 步才进行自适应定价。
"""

from collections import deque
from itertools import islice
from dataclasses import FrozenInstanceError, dataclass
from math import isfinite
from numbers import Real


# These labels keep our judgment call visible in future experiment logs. The
# baseline and a cartel initializer now exist, but final-outcome sensitivity to
# cartel and expanding-window starts is still pending. / 这些标签让未来实验日志继续
# 看见我们的判断。基准和 cartel 初始化器已经存在，但最终结果对 cartel 与扩展窗口
# 初始化的敏感性仍待检验。
INITIAL_HISTORY_DECISION_ID = "A3"
INITIAL_HISTORY_DECISION_STATUS = "BASELINE_IMPLEMENTED_SENSITIVITY_PENDING"
INITIAL_HISTORY_SELECTED_METHOD = "balanced_nash_consistent_synthetic_prehistory"


@dataclass(frozen=True)
class MarketObservation:
    """One immutable historical row (v, p, z, y).

    一条不可修改的历史记录 (v, p, z, y)。

    The long field names deliberately include the paper symbols. This makes it
    harder to accidentally swap z and y during later integration. / 字段名故意
    写出论文符号，降低后续整合时误把 z 与 y 对调的风险。
    """

    fundamental_value_v: float
    market_price_p: float
    insensitive_order_z: float
    informed_and_noise_order_y: float

    def __post_init__(self) -> None:
        """Reject non-numeric, NaN, and infinite history values.

        拒绝非数字、NaN 和无穷大的历史数值。
        """

        named_values = {
            "fundamental_value_v": self.fundamental_value_v,
            "market_price_p": self.market_price_p,
            "insensitive_order_z": self.insensitive_order_z,
            "informed_and_noise_order_y": self.informed_and_noise_order_y,
        }
        for field_name, number in named_values.items():
            if isinstance(number, bool) or not isinstance(number, Real):
                raise TypeError(
                    f"{field_name} must be a real number. / "
                    f"{field_name} 必须是实数。"
                )
            if not isfinite(float(number)):
                raise ValueError(
                    f"{field_name} must be finite. / "
                    f"{field_name} 必须是有限数。"
                )


class MarketMakerHistory:
    """A mutable fixed-capacity rolling window of immutable rows.

    一个可变化、容量固定的滚动窗口；窗口内每条记录不可修改。
    """

    def __init__(self, window_size: int = 10_000) -> None:
        """Create an empty window; do not invent initial paper data.

        建立空窗口；不要虚构论文未说明的初始数据。
        """

        if (
            isinstance(window_size, bool)
            or not isinstance(window_size, int)
            or window_size <= 0
        ):
            raise ValueError(
                "window_size must be a positive integer. / "
                "window_size 必须是正整数。"
            )

        # deque automatically removes the oldest row when maxlen is reached.
        # deque 达到 maxlen 后，会自动移除最旧的一条记录。
        self._rows: deque[MarketObservation] = deque(maxlen=window_size)

    @property
    def window_size(self) -> int:
        """Return T_m without allowing it to be changed later. / 返回不可随意修改的 T_m。"""

        # maxlen is fixed when the deque is constructed. / maxlen 在建立 deque 时固定。
        return int(self._rows.maxlen)

    def __len__(self) -> int:
        """Allow len(history) to report the current row count. / 允许用 len(history) 查看行数。"""

        return len(self._rows)

    @property
    def is_full(self) -> bool:
        """Report whether the window currently contains T_m rows. / 是否已有 T_m 行。"""

        return len(self._rows) == self.window_size

    def oldest(self) -> MarketObservation | None:
        """Return the oldest frozen row without removing it, or None if empty.

        返回最旧但不移除的冻结记录；空窗口返回 None。

        Step 24B needs this small read-only view so it can validate a rolling
        OLS update *before* the deque evicts anything. Access is O(1), whereas
        making a complete snapshot merely to inspect row zero would be O(T_m).
        / 第 24B 步需要这个只读入口，以便在 deque 真正淘汰旧记录之前验证滚动
        OLS 更新。该操作是 O(1)；若只为查看第零行就复制完整快照，则是 O(T_m)。
        """

        return self._rows[0] if self._rows else None

    def append(self, observation: MarketObservation) -> MarketObservation | None:
        """Append one completed-period row and return any evicted oldest row.

        加入一条已经完成的时期记录；若窗口已满，则返回被移除的最旧记录。
        """

        if not isinstance(observation, MarketObservation):
            raise TypeError(
                "History accepts only MarketObservation records. / "
                "历史窗口只接受 MarketObservation 记录。"
            )

        evicted_oldest_row = self._rows[0] if self.is_full else None
        self._rows.append(observation)
        return evicted_oldest_row

    def _rollback_full_window_appends(
        self,
        *,
        appended_rows: tuple[MarketObservation, ...],
        evicted_rows: tuple[MarketObservation, ...],
    ) -> None:
        """Undo a small, already-audited sequence of full-window appends.

        撤销一小段已经核对过的“满窗口追加”。

        This is a private engineering helper for disposable Step-35 paths.
        If a full window ``[a, b, c]`` receives ``x`` and then ``y``, it
        becomes ``[c, x, y]`` and the evicted log is ``[a, b]``. Walking the
        two logs backwards restores ``[a, b, c]`` without copying all T_m
        rows. Work and temporary memory are both O(k), where k is the short
        path length. / 这是供第 35 步一次性短路径使用的私有工程接口。若满窗口
        ``[a, b, c]`` 依次加入 ``x``、``y``，窗口变为 ``[c, x, y]``，淘汰
        日志是 ``[a, b]``。倒序处理两份日志即可恢复 ``[a, b, c]``，无需复制
        全部 T_m 行；时间和临时内存都是 O(k)，k 是短路径长度。

        All suffix checks happen before the first mutation. A wrong or stale
        log therefore fails atomically. / 第一次修改之前会先核对完整后缀，因此
        错误或过期日志会原子地失败，不会只恢复一半。
        """

        if not isinstance(appended_rows, tuple) or not isinstance(
            evicted_rows,
            tuple,
        ):
            raise TypeError(
                "Rollback logs must be tuples. / 回滚日志必须是 tuple。"
            )
        if len(appended_rows) != len(evicted_rows):
            raise ValueError(
                "Appended and evicted logs must have equal length. / "
                "追加与淘汰日志长度必须相同。"
            )
        if len(appended_rows) > self.window_size:
            raise ValueError(
                "A rollback cannot exceed the rolling window. / "
                "一次回滚不能超过滚动窗口长度。"
            )
        if not self.is_full:
            raise RuntimeError(
                "Short-path rollback requires a full rolling window. / "
                "短路径回滚要求滚动窗口已满。"
            )
        if not all(
            isinstance(row, MarketObservation)
            for row in (*appended_rows, *evicted_rows)
        ):
            raise TypeError(
                "Every rollback row must be MarketObservation. / "
                "每条回滚记录都必须是 MarketObservation。"
            )

        # Compare only the k newest rows. ``reversed(deque)`` walks from the
        # right in O(k); it does not materialize the other T_m-k rows. / 只比较
        # 最新的 k 行；reversed(deque) 从右侧 O(k) 遍历，不会复制其余行。
        actual_newest = islice(reversed(self._rows), len(appended_rows))
        expected_newest = reversed(appended_rows)
        if any(
            actual != expected
            for actual, expected in zip(
                actual_newest,
                expected_newest,
                strict=True,
            )
        ):
            raise RuntimeError(
                "The live history suffix differs from the append log. / "
                "实时历史后缀与追加日志不一致。"
            )

        for expected_appended, evicted in zip(
            reversed(appended_rows),
            reversed(evicted_rows),
            strict=True,
        ):
            removed = self._rows.pop()
            if removed != expected_appended:  # Defensive; precheck already passed.
                raise RuntimeError(
                    "History changed during rollback. / 回滚过程中历史被改变。"
                )
            self._rows.appendleft(evicted)

    def snapshot(self) -> tuple[MarketObservation, ...]:
        """Return an outside-safe view ordered oldest to newest.

        返回按“最旧到最新”排列、不会暴露内部 deque 的快照。
        """

        return tuple(self._rows)


def main() -> None:
    """Validate capacity, eviction, timing, and record safety. / 验证容量、淘汰与时间顺序。"""

    paper_window = MarketMakerHistory()
    assert paper_window.window_size == 10_000
    assert len(paper_window) == 0
    assert not paper_window.is_full

    # Use T_m=3 only so the rolling behavior is visible by hand.
    # 仅在手算演示中使用 T_m=3，方便看清滚动过程。
    history = MarketMakerHistory(window_size=3)

    row_1 = MarketObservation(
        fundamental_value_v=0.80,
        market_price_p=1.012,
        insensitive_order_z=-6.0,
        informed_and_noise_order_y=10.0,
    )
    row_2 = MarketObservation(
        fundamental_value_v=0.90,
        market_price_p=0.99,
        insensitive_order_z=5.0,
        informed_and_noise_order_y=-2.0,
    )
    row_3 = MarketObservation(
        fundamental_value_v=1.10,
        market_price_p=1.00,
        insensitive_order_z=0.0,
        informed_and_noise_order_y=3.0,
    )
    current_row_4 = MarketObservation(
        fundamental_value_v=1.20,
        market_price_p=1.02,
        insensitive_order_z=-10.0,
        informed_and_noise_order_y=4.0,
    )

    for old_row in (row_1, row_2, row_3):
        evicted = history.append(old_row)
        assert evicted is None
        assert len(history) <= history.window_size

    before_period_4_pricing = history.snapshot()
    assert isinstance(before_period_4_pricing, tuple)
    assert before_period_4_pricing == (row_1, row_2, row_3)
    assert history.oldest() == row_1
    assert before_period_4_pricing[0].market_price_p == 1.012
    assert current_row_4 not in before_period_4_pricing
    assert history.is_full

    # Period 4 must be priced from the snapshot above. Only after period 4 is
    # complete may its row enter the window for period 5. / 第 4 期必须使用上面的
    # 历史快照定价；只有第 4 期结束后，本期记录才能进入第 5 期的窗口。
    evicted = history.append(current_row_4)
    after_period_4_completion = history.snapshot()

    assert evicted == row_1
    assert after_period_4_completion == (row_2, row_3, current_row_4)
    assert len(history) == 3
    assert history.is_full
    assert row_1 not in after_period_4_completion
    assert current_row_4 in after_period_4_completion
    assert current_row_4.market_price_p == 1.02
    assert history.oldest() == row_2

    # Keep the four future OLS columns aligned after eviction. Step 23 will
    # regress z on p and v on y. / 淘汰旧记录后，四个未来 OLS 数据列仍须逐行
    # 对齐。第 23 步将回归 z 对 p，以及 v 对 y。
    prices_p = tuple(row.market_price_p for row in after_period_4_completion)
    insensitive_orders_z = tuple(
        row.insensitive_order_z for row in after_period_4_completion
    )
    informed_and_noise_orders_y = tuple(
        row.informed_and_noise_order_y for row in after_period_4_completion
    )
    fundamental_values_v = tuple(
        row.fundamental_value_v for row in after_period_4_completion
    )
    assert prices_p == (0.99, 1.00, 1.02)
    assert insensitive_orders_z == (5.0, 0.0, -10.0)
    assert informed_and_noise_orders_y == (-2.0, 3.0, 4.0)
    assert fundamental_values_v == (0.90, 1.10, 1.20)

    # Frozen rows cannot be rewritten after the event. / 事件发生后，冻结记录不能改写。
    try:
        setattr(row_2, "market_price_p", 999.0)
    except FrozenInstanceError:
        row_is_frozen = True
    else:
        row_is_frozen = False
    assert row_is_frozen
    assert row_2.market_price_p == 0.99

    # Bad input must fail without corrupting existing history. / 错误输入必须报错，且不能破坏历史。
    history_before_bad_input = history.snapshot()
    try:
        history.append((1.0, 1.0, 0.0, 0.0))  # type: ignore[arg-type]
    except TypeError:
        wrong_row_type_was_rejected = True
    else:
        wrong_row_type_was_rejected = False
    assert wrong_row_type_was_rejected
    assert history.snapshot() == history_before_bad_input

    try:
        MarketObservation(
            fundamental_value_v=float("nan"),
            market_price_p=1.0,
            insensitive_order_z=0.0,
            informed_and_noise_order_y=0.0,
        )
    except ValueError:
        nonfinite_row_was_rejected = True
    else:
        nonfinite_row_was_rejected = False
    assert nonfinite_row_was_rejected

    for bad_window_size in (0, -1, 1.5, True):
        try:
            MarketMakerHistory(window_size=bad_window_size)  # type: ignore[arg-type]
        except ValueError:
            bad_window_was_rejected = True
        else:
            bad_window_was_rejected = False
        assert bad_window_was_rejected

    print("Step 22: Market-maker rolling history / 步骤 22：做市商滚动历史")
    print(f"Paper baseline T_m / 论文基准 T_m: {paper_window.window_size:,}")
    print(f"Toy window size / 玩具窗口大小: {history.window_size}")
    print("\nD_4 before period-4 pricing / 第 4 期定价前的 D_4:")
    for observation in before_period_4_pricing:
        print(f"  {observation}")
    print("Current period 4 is absent. / 当前第 4 期记录尚未进入历史。")
    print("\nAfter period 4 completes / 第 4 期结束后:")
    print(f"  Evicted oldest row / 被移除的最旧记录: {evicted}")
    for observation in after_period_4_completion:
        print(f"  {observation}")
    print("Current period 4 is now available to D_5. / 第 4 期记录现在可供 D_5 使用。")
    print("Every row is frozen; the rolling container remains mutable. / 每条记录冻结，窗口仍可滚动。")
    print(
        "Initialization decision / 初始化决定: "
        f"{INITIAL_HISTORY_DECISION_ID} = {INITIAL_HISTORY_DECISION_STATUS}"
    )
    print(
        "Selected method / 选定方法: "
        f"{INITIAL_HISTORY_SELECTED_METHOD}"
    )
    print(
        "This is our implemented replication choice, not a paper rule. / "
        "这是我们已实现的复现选择，不是论文规则。"
    )
    print("No OLS regression or adaptive price was calculated. / 尚未计算 OLS 或自适应价格。")
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
