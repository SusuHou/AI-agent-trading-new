"""Step 24B: make rolling market-maker OLS fast, safe, and auditable.

步骤 24B：让做市商的滚动 OLS 更快、更安全，而且仍然容易核对。

Run / 运行:
    py -3 -X utf8 steps/step_24b_fast_rolling_ols.py

Why this extra step exists / 为什么要增加这一小步:
    Step 23 deliberately recalculates OLS from every row because that version
    is easiest to read and verify. With the paper's T_m=10,000, scanning all
    10,000 rows in every period would be wasteful. This class keeps centered
    sufficient statistics and updates them when one row enters and (after the
    window fills) one row leaves. Ordinary updates therefore take constant
    work, O(1). / 第 23 步故意每次扫描全部记录，因为它最容易阅读和验证。但论文
    使用 T_m=10,000；若每期重新扫描一万行，会浪费大量计算。本类保存“中心化充分
    统计量”，每当一行进入、另一行离开时更新它们，因此普通更新只需 O(1) 的固定工作量。

Numerical engineering choice / 数值工程选择:
    We do NOT use raw variance as sum(x^2)-sum(x)^2/n. Those two large terms
    can nearly cancel, especially because prices are close to one. Instead we
    use Welford-style centered add/remove formulas. Every resynchronize_every
    successful appends, we rebuild the statistics from the frozen history with
    math.fsum. This periodic rebuild limits accumulated floating-point drift.
    It does not change the paper's equal-weight, unregularized OLS estimator.
    / 我们不使用 sum(x^2)-sum(x)^2/n 计算方差，因为两个大数相减可能造成严重
    浮点抵消，尤其价格都接近 1。这里使用 Welford 风格的中心化加入/移除公式；每
    成功追加 resynchronize_every 次，就用 math.fsum 从冻结历史重新计算，以限制
    长期浮点漂移。它不会改变论文的等权、无正则化 OLS。

Timing remains unchanged / 时间顺序不变:
    Estimate from past D_t -> price current y_t -> append the completed current
    row. This container never sees the current row before its price exists.
    / 先用过去 D_t 估计 -> 给本期 y_t 定价 -> 本期结束后才追加记录。本容器不会
    让本期记录参与决定自己的价格。
"""

from collections import deque
from dataclasses import dataclass
from math import fsum, isclose, isfinite

import numpy as np

from step_22_market_maker_rolling_history import (
    MarketMakerHistory,
    MarketObservation,
)
from step_23_market_maker_ols import (
    MarketMakerOLSEstimates,
    fit_market_maker_regressions,
)
from step_24_adaptive_market_maker_price import (
    calculate_adaptive_price_quote,
)


# This tolerance audits that a saved exact accumulator is economically and
# numerically consistent with its rows. We still restore the saved values
# themselves, so the future path retains its exact floating-point state. / 该容差
# 只用于核对“保存的累加器与历史行数值一致”；真正恢复的仍是保存值本身，因此未来
# 路径会保留精确浮点状态。
CHECKPOINT_STATISTICS_RELATIVE_TOLERANCE = 1e-8
CHECKPOINT_STATISTICS_ABSOLUTE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class CenteredPairStatisticsState:
    """Immutable public checkpoint state for one centered OLS regression.

    一条中心化 OLS 回归的公开、不可修改快照。

    These five numbers are saved in addition to the history rows because
    rebuilding them can change the last floating-point bits. Those tiny
    differences can eventually change a long simulated path. / 除了历史行，
    还必须保存这五个数；若从历史重新计算，浮点数最后几位可能不同，并可能在
    很长的模拟中逐渐改变路径。
    """

    sample_size: int
    mean_x: float
    mean_dependent: float
    sum_squared_x: float
    sum_cross_products: float


@dataclass(frozen=True)
class RollingMarketMakerState:
    """Lossless immutable state used to continue one rolling OLS exactly.

    用来精确续跑滚动 OLS 的无损、不可修改状态。
    """

    rows: tuple[MarketObservation, ...]
    window_size: int
    resynchronize_every: int
    demand_statistics: CenteredPairStatisticsState
    value_statistics: CenteredPairStatisticsState
    updates_since_resynchronization: int
    successful_append_count: int
    resynchronization_count: int


@dataclass(frozen=True)
class RollingMarketMakerAppendTransactionToken:
    """Opaque proof for one bounded, rollback-only append transaction.

    一次有上限、只用于回滚的追加事务之不透明凭证。

    Callers should keep this object and pass it back unchanged; its private
    markers deliberately have no economic meaning. / 调用者只需保存并原样交回；
    其中私有标记没有任何经济含义。
    """

    _owner_marker: object
    _transaction_marker: object
    max_appends: int


@dataclass(frozen=True)
class _CenteredPairStatistics:
    """Internal OLS memory for dependent = intercept + slope*x.

    一条 OLS 回归的内部记忆：因变量 = 截距 + 斜率*x。

    `sum_squared_x` is S_xx and `sum_cross_products` is S_xy. The leading
    underscore means this is an implementation detail, not a paper object the
    experiment should manipulate directly. / `sum_squared_x` 是 S_xx，
    `sum_cross_products` 是 S_xy。类名前的下划线表示它只是内部实现细节，实验
    不应直接修改它。
    """

    sample_size: int = 0
    mean_x: float = 0.0
    mean_dependent: float = 0.0
    sum_squared_x: float = 0.0
    sum_cross_products: float = 0.0

    def _validated(self) -> "_CenteredPairStatistics":
        """Reject an overflowed candidate before history changes. / 历史变化前拒绝溢出。"""

        if self.sample_size < 0:
            raise RuntimeError("Internal sample size became negative. / 内部样本量变成负数。")
        numerical_fields = (
            self.mean_x,
            self.mean_dependent,
            self.sum_squared_x,
            self.sum_cross_products,
        )
        if not all(isfinite(number) for number in numerical_fields):
            raise ValueError(
                "Rolling OLS statistics overflowed. / 滚动 OLS 统计量发生溢出。"
            )
        return self

    def add(self, x_value: float, dependent_value: float) -> "_CenteredPairStatistics":
        """Return new statistics after adding one pair; do not mutate this copy.

        返回加入一对数值后的新统计量；不修改当前副本。
        """

        new_sample_size = self.sample_size + 1
        delta_x = x_value - self.mean_x
        delta_dependent = dependent_value - self.mean_dependent
        new_mean_x = self.mean_x + delta_x / new_sample_size
        new_mean_dependent = (
            self.mean_dependent + delta_dependent / new_sample_size
        )
        candidate = _CenteredPairStatistics(
            sample_size=new_sample_size,
            mean_x=new_mean_x,
            mean_dependent=new_mean_dependent,
            sum_squared_x=(
                self.sum_squared_x + delta_x * (x_value - new_mean_x)
            ),
            sum_cross_products=(
                self.sum_cross_products
                + delta_x * (dependent_value - new_mean_dependent)
            ),
        )
        return candidate._validated()

    def remove(
        self,
        x_value: float,
        dependent_value: float,
    ) -> "_CenteredPairStatistics":
        """Return new statistics after removing one known pair.

        返回移除一对已知数值后的新统计量。
        """

        if self.sample_size <= 0:
            raise RuntimeError("Cannot remove from empty OLS statistics. / 不能从空统计量移除。")
        if self.sample_size == 1:
            return _CenteredPairStatistics()

        new_sample_size = self.sample_size - 1
        # This algebra avoids n*mean, which can overflow sooner. / 这种等价写法
        # 避免计算 n*mean，因此更不容易溢出。
        new_mean_x = self.mean_x - (x_value - self.mean_x) / new_sample_size
        new_mean_dependent = (
            self.mean_dependent
            - (dependent_value - self.mean_dependent) / new_sample_size
        )
        candidate = _CenteredPairStatistics(
            sample_size=new_sample_size,
            mean_x=new_mean_x,
            mean_dependent=new_mean_dependent,
            sum_squared_x=(
                self.sum_squared_x
                - (x_value - self.mean_x) * (x_value - new_mean_x)
            ),
            sum_cross_products=(
                self.sum_cross_products
                - (x_value - self.mean_x)
                * (dependent_value - new_mean_dependent)
            ),
        )
        return candidate._validated()

    def fitted_line(self, explanatory_name: str) -> tuple[float, float]:
        """Return (intercept, slope), or explain why the slope is unidentified.

        返回（截距、斜率）；若斜率无法识别，则给出明确错误。
        """

        if self.sample_size < 2:
            raise ValueError("OLS requires at least two observations. / OLS 至少需要两条观测。")
        if self.sum_squared_x <= 0.0:
            raise ValueError(
                f"OLS cannot identify a slope because {explanatory_name} "
                f"has no variation. / {explanatory_name} 没有变化，OLS 无法识别斜率。"
            )
        slope = self.sum_cross_products / self.sum_squared_x
        intercept = self.mean_dependent - slope * self.mean_x
        if not isfinite(intercept) or not isfinite(slope):
            raise ValueError("OLS coefficients are not finite. / OLS 系数不是有限数。")
        return intercept, slope


@dataclass
class _ActiveRollingAppendTransaction:
    """Private O(1)-start memory plus the O(k) row log. / 私有的 O(1) 起点记忆和 O(k) 行日志。"""

    token: RollingMarketMakerAppendTransactionToken
    starting_demand_statistics: _CenteredPairStatistics
    starting_value_statistics: _CenteredPairStatistics
    starting_updates_since_resynchronization: int
    starting_successful_append_count: int
    starting_resynchronization_count: int
    appended_rows: list[MarketObservation]
    evicted_rows: list[MarketObservation]


def _rebuilt_pair_statistics(
    x_values: tuple[float, ...],
    dependent_values: tuple[float, ...],
) -> _CenteredPairStatistics:
    """Rebuild centered statistics with accurate summation. / 用精确求和重新构建统计量。"""

    if len(x_values) != len(dependent_values):
        raise RuntimeError("Internal OLS columns are misaligned. / 内部 OLS 列没有对齐。")
    sample_size = len(x_values)
    if sample_size == 0:
        return _CenteredPairStatistics()
    mean_x = fsum(x_values) / sample_size
    mean_dependent = fsum(dependent_values) / sample_size
    rebuilt = _CenteredPairStatistics(
        sample_size=sample_size,
        mean_x=mean_x,
        mean_dependent=mean_dependent,
        sum_squared_x=fsum((x_value - mean_x) ** 2 for x_value in x_values),
        sum_cross_products=fsum(
            (x_value - mean_x) * (dependent_value - mean_dependent)
            for x_value, dependent_value in zip(
                x_values,
                dependent_values,
                strict=True,
            )
        ),
    )
    return rebuilt._validated()


def _rebuilt_statistics_from_rows(
    rows: tuple[MarketObservation, ...],
) -> tuple[_CenteredPairStatistics, _CenteredPairStatistics]:
    """Rebuild both paper regressions from one aligned row snapshot.

    从同一份逐行对齐的快照重建论文的两条回归。
    """

    demand_statistics = _rebuilt_pair_statistics(
        tuple(row.market_price_p for row in rows),
        tuple(row.insensitive_order_z for row in rows),
    )
    value_statistics = _rebuilt_pair_statistics(
        tuple(row.informed_and_noise_order_y for row in rows),
        tuple(row.fundamental_value_v for row in rows),
    )
    return demand_statistics, value_statistics


class RollingMarketMakerOLS:
    """Own one rolling history and its matching O(1) OLS statistics.

    同时管理一份滚动历史及与之严格对应的 O(1) OLS 统计量。

    The mutable history is deliberately private. Callers can receive frozen
    rows or an immutable tuple snapshot, but cannot append behind this class's
    back and desynchronize the regressions. / 可变历史故意保持私有。外部只能获得
    冻结记录或不可变 tuple 快照，不能绕开本类追加记录，因而不会让回归与历史失配。
    """

    def __init__(
        self,
        window_size: int = 10_000,
        resynchronize_every: int | None = None,
    ) -> None:
        self._history = MarketMakerHistory(window_size=window_size)
        if resynchronize_every is None:
            resynchronize_every = window_size
        if (
            isinstance(resynchronize_every, bool)
            or not isinstance(resynchronize_every, int)
            or resynchronize_every <= 0
        ):
            raise ValueError(
                "resynchronize_every must be a positive integer. / "
                "resynchronize_every 必须是正整数。"
            )
        self._resynchronize_every = resynchronize_every
        self._demand_statistics = _CenteredPairStatistics()
        self._value_statistics = _CenteredPairStatistics()
        self._updates_since_resynchronization = 0
        self._successful_append_count = 0
        self._resynchronization_count = 0
        # One owner marker prevents a token from another maker being accepted.
        # 独立 owner 标记可防止误用另一个做市商签发的 token。
        self._append_transaction_owner_marker = object()
        self._active_append_transaction: (
            _ActiveRollingAppendTransaction | None
        ) = None

    def __len__(self) -> int:
        """Return the current number of historical rows. / 返回当前历史行数。"""

        return len(self._history)

    @property
    def window_size(self) -> int:
        """Return T_m. / 返回 T_m。"""

        return self._history.window_size

    @property
    def is_full(self) -> bool:
        """Report whether exactly T_m rows are available. / 是否已有完整 T_m 行。"""

        return self._history.is_full

    @property
    def resynchronize_every(self) -> int:
        """Return the engineering resynchronization interval. / 返回工程重同步间隔。"""

        return self._resynchronize_every

    @property
    def successful_append_count(self) -> int:
        """Count only rows that were safely committed. / 只统计安全提交成功的记录。"""

        return self._successful_append_count

    @property
    def resynchronization_count(self) -> int:
        """Report how many exact rebuilds have occurred. / 返回精确重建次数。"""

        return self._resynchronization_count

    @property
    def has_active_append_transaction(self) -> bool:
        """Report whether a reversible short-path transaction is open.

        返回是否有一段可逆短路径事务仍未结束。

        A checkpoint must be taken between complete periods, so Step 36C uses
        this read-only flag to reject a half-open rollback transaction. / checkpoint
        必须保存在完整时期之间，因此第 36C 步用这个只读标记拒绝尚未结束的回滚事务。
        """

        return self._active_append_transaction is not None

    def snapshot(self) -> tuple[MarketObservation, ...]:
        """Return frozen rows ordered oldest to newest. / 返回从旧到新的冻结记录。"""

        return self._history.snapshot()

    def begin_reversible_append_transaction(
        self,
        *,
        max_appends: int,
    ) -> RollingMarketMakerAppendTransactionToken:
        """Begin one short transaction whose successful appends can be undone.

        开始一段可撤销其成功追加操作的短事务。

        Beginning is O(1): it saves the two small OLS accumulators and counters,
        not the T_m history rows. Each later append records only its new row and
        the row it evicted. / 开始操作是 O(1)：只保存两组很小的 OLS 累加器与
        计数器，不复制 T_m 行历史。之后每次追加只记录新行和被淘汰的旧行。

        This facility is deliberately rollback-only. It is intended for many
        disposable IRF paths that all start from the same converged market. / 此
        接口故意只支持回滚，用于大量都从同一收敛市场出发的一次性 IRF 路径。
        """

        if (
            isinstance(max_appends, bool)
            or not isinstance(max_appends, int)
            or max_appends <= 0
        ):
            raise ValueError(
                "max_appends must be a positive integer. / "
                "max_appends 必须是正整数。"
            )
        if max_appends > self.window_size:
            raise ValueError(
                "max_appends cannot exceed T_m. / max_appends 不能超过 T_m。"
            )
        if self._active_append_transaction is not None:
            raise RuntimeError(
                "A reversible append transaction is already active. / "
                "已有可逆追加事务正在运行。"
            )
        if not self.is_full:
            raise RuntimeError(
                "Reversible short paths require a full T_m history. / "
                "可逆短路径要求已有完整 T_m 历史。"
            )

        token = RollingMarketMakerAppendTransactionToken(
            _owner_marker=self._append_transaction_owner_marker,
            _transaction_marker=object(),
            max_appends=max_appends,
        )
        self._active_append_transaction = _ActiveRollingAppendTransaction(
            token=token,
            starting_demand_statistics=self._demand_statistics,
            starting_value_statistics=self._value_statistics,
            starting_updates_since_resynchronization=(
                self._updates_since_resynchronization
            ),
            starting_successful_append_count=self._successful_append_count,
            starting_resynchronization_count=self._resynchronization_count,
            appended_rows=[],
            evicted_rows=[],
        )
        return token

    def reversible_append_count(
        self,
        token: RollingMarketMakerAppendTransactionToken,
    ) -> int:
        """Return how many rows the live transaction has appended.

        返回当前事务已经追加的行数。

        The token check prevents one session from reading another session's
        transaction state. / token 核对可防止一个 session 读取另一个 session
        的事务状态。
        """

        active = self._validated_active_append_transaction(token)
        return len(active.appended_rows)

    def _validated_active_append_transaction(
        self,
        token: RollingMarketMakerAppendTransactionToken,
    ) -> _ActiveRollingAppendTransaction:
        """Return the matching live log or reject a stale/foreign token.

        返回相符的实时日志；拒绝过期或来自其他对象的 token。
        """

        if not isinstance(token, RollingMarketMakerAppendTransactionToken):
            raise TypeError(
                "token has the wrong type. / token 类型错误。"
            )
        active = self._active_append_transaction
        if (
            active is None
            or token is not active.token
            or token._owner_marker is not self._append_transaction_owner_marker
            or token._transaction_marker is not active.token._transaction_marker
        ):
            raise RuntimeError(
                "The append-transaction token is stale or foreign. / "
                "追加事务 token 已过期或来自其他对象。"
            )
        return active

    def rollback_reversible_append_transaction(
        self,
        token: RollingMarketMakerAppendTransactionToken,
    ) -> int:
        """Restore the exact pre-transaction history, OLS memory, and counters.

        精确恢复事务前的历史、OLS 记忆和计数器。

        Returns the number of undone appends. Validation precedes mutation, so
        a stale token or mismatched history cannot cause a partial rollback. /
        返回撤销的追加次数。所有核对先于修改，因此过期 token 或不匹配历史不会
        导致只回滚一半。
        """

        active = self._validated_active_append_transaction(token)
        number_of_appends = len(active.appended_rows)
        if len(active.evicted_rows) != number_of_appends:
            raise RuntimeError(
                "The reversible append log is incomplete. / 可逆追加日志不完整。"
            )
        if (
            self._successful_append_count
            != active.starting_successful_append_count + number_of_appends
        ):
            raise RuntimeError(
                "The append counter differs from the transaction log. / "
                "追加计数器与事务日志不一致。"
            )

        self._history._rollback_full_window_appends(
            appended_rows=tuple(active.appended_rows),
            evicted_rows=tuple(active.evicted_rows),
        )
        self._demand_statistics = active.starting_demand_statistics
        self._value_statistics = active.starting_value_statistics
        self._updates_since_resynchronization = (
            active.starting_updates_since_resynchronization
        )
        self._successful_append_count = (
            active.starting_successful_append_count
        )
        self._resynchronization_count = active.starting_resynchronization_count
        self._active_append_transaction = None
        return number_of_appends

    @staticmethod
    def _public_statistics_state(
        statistics: _CenteredPairStatistics,
    ) -> CenteredPairStatisticsState:
        """Copy private numerical memory into a frozen public record.

        把私有数值记忆复制到公开的冻结记录中。
        """

        return CenteredPairStatisticsState(
            sample_size=statistics.sample_size,
            mean_x=statistics.mean_x,
            mean_dependent=statistics.mean_dependent,
            sum_squared_x=statistics.sum_squared_x,
            sum_cross_products=statistics.sum_cross_products,
        )

    @staticmethod
    def _private_statistics_from_state(
        state: CenteredPairStatisticsState,
    ) -> _CenteredPairStatistics:
        """Validate and restore one private centered-statistics object.

        检查并恢复一组私有中心化统计量。
        """

        if not isinstance(state, CenteredPairStatisticsState):
            raise TypeError(
                "statistics state has the wrong type. / statistics 状态类型错误。"
            )
        if isinstance(state.sample_size, bool) or not isinstance(
            state.sample_size,
            int,
        ):
            raise TypeError("sample_size must be an integer. / sample_size 必须是整数。")
        return _CenteredPairStatistics(
            sample_size=state.sample_size,
            mean_x=float(state.mean_x),
            mean_dependent=float(state.mean_dependent),
            sum_squared_x=float(state.sum_squared_x),
            sum_cross_products=float(state.sum_cross_products),
        )._validated()

    def export_state(self) -> RollingMarketMakerState:
        """Return every causal bit needed for an exact future continuation.

        返回精确续跑未来路径所需的全部因果状态。

        This is intentionally stronger than ``snapshot()``: it includes the
        current OLS accumulators and the countdown to the next numerical
        resynchronization. / 它有意比 ``snapshot()`` 更完整：还包含当前 OLS
        累加器，以及距离下一次数值重同步还有多久。
        """

        return RollingMarketMakerState(
            rows=self.snapshot(),
            window_size=self.window_size,
            resynchronize_every=self.resynchronize_every,
            demand_statistics=self._public_statistics_state(
                self._demand_statistics
            ),
            value_statistics=self._public_statistics_state(
                self._value_statistics
            ),
            updates_since_resynchronization=(
                self._updates_since_resynchronization
            ),
            successful_append_count=self._successful_append_count,
            resynchronization_count=self._resynchronization_count,
        )

    @classmethod
    def from_state(
        cls,
        state: RollingMarketMakerState,
    ) -> "RollingMarketMakerOLS":
        """Build a new independent market maker from a validated exact state.

        从经过检查的精确状态建立一个全新的、相互独立的做市商。

        We restore the saved accumulators rather than silently rebuilding them
        from the rows. / 我们恢复已保存的累加器，而不是悄悄从历史行重算。
        """

        if not isinstance(state, RollingMarketMakerState):
            raise TypeError("state must be RollingMarketMakerState. / state 类型错误。")
        restored = cls(
            window_size=state.window_size,
            resynchronize_every=state.resynchronize_every,
        )
        if len(state.rows) > state.window_size:
            raise ValueError("Saved history exceeds T_m. / 保存的历史超过 T_m。")
        if not all(isinstance(row, MarketObservation) for row in state.rows):
            raise TypeError("Every saved row must be MarketObservation. / 每个历史行类型必须正确。")
        demand = cls._private_statistics_from_state(state.demand_statistics)
        value = cls._private_statistics_from_state(state.value_statistics)
        if demand.sample_size != len(state.rows) or value.sample_size != len(
            state.rows
        ):
            raise ValueError(
                "Saved OLS sample sizes differ from the saved rows. / "
                "保存的 OLS 样本量与历史行数不同。"
            )
        integer_fields = (
            state.updates_since_resynchronization,
            state.successful_append_count,
            state.resynchronization_count,
        )
        if any(isinstance(number, bool) or not isinstance(number, int) for number in integer_fields):
            raise TypeError("Saved counters must be integers. / 保存的计数器必须是整数。")
        if not 0 <= state.updates_since_resynchronization < state.resynchronize_every:
            raise ValueError(
                "The saved resynchronization phase is invalid. / "
                "保存的重同步阶段无效。"
            )
        if state.successful_append_count < len(state.rows):
            raise ValueError(
                "Append count cannot be smaller than saved history. / "
                "追加次数不能小于保存的历史行数。"
            )
        if state.resynchronization_count < 0:
            raise ValueError("Resynchronization count cannot be negative. / 重同步次数不能为负。")
        if state.updates_since_resynchronization > state.successful_append_count:
            raise ValueError(
                "Updates since resynchronization cannot exceed all appends. / "
                "重同步后的更新数不能超过全部追加数。"
            )
        if state.resynchronization_count == 0:
            if (
                state.updates_since_resynchronization
                != state.successful_append_count
                or state.successful_append_count >= state.resynchronize_every
            ):
                raise ValueError(
                    "Counters describe a resynchronization that was not recorded. / "
                    "计数器描述了一次没有记录的重同步。"
                )

        rebuilt_demand, rebuilt_value = _rebuilt_statistics_from_rows(
            state.rows
        )
        for saved, rebuilt, label in (
            (demand, rebuilt_demand, "demand"),
            (value, rebuilt_value, "value"),
        ):
            saved_numbers = (
                saved.mean_x,
                saved.mean_dependent,
                saved.sum_squared_x,
                saved.sum_cross_products,
            )
            rebuilt_numbers = (
                rebuilt.mean_x,
                rebuilt.mean_dependent,
                rebuilt.sum_squared_x,
                rebuilt.sum_cross_products,
            )
            if not all(
                isclose(
                    saved_number,
                    rebuilt_number,
                    rel_tol=CHECKPOINT_STATISTICS_RELATIVE_TOLERANCE,
                    abs_tol=CHECKPOINT_STATISTICS_ABSOLUTE_TOLERANCE,
                )
                for saved_number, rebuilt_number in zip(
                    saved_numbers,
                    rebuilt_numbers,
                    strict=True,
                )
            ):
                raise ValueError(
                    f"Saved {label} OLS statistics disagree with the rows. / "
                    f"保存的 {label} OLS 统计量与历史行不一致。"
                )

        # The owner class may fill its own private history directly. Doing so
        # avoids changing the saved append/rebuild counters. / 本类可以直接填充
        # 自己的私有历史，从而不会误改已保存的追加与重建计数。
        for row in state.rows:
            evicted = restored._history.append(row)
            if evicted is not None:
                raise RuntimeError("Restoration unexpectedly evicted a row. / 恢复时意外淘汰了历史行。")
        restored._demand_statistics = demand
        restored._value_statistics = value
        restored._updates_since_resynchronization = (
            state.updates_since_resynchronization
        )
        restored._successful_append_count = state.successful_append_count
        restored._resynchronization_count = state.resynchronization_count
        return restored

    def append_completed_observation(
        self,
        observation: MarketObservation,
    ) -> MarketObservation | None:
        """Safely add one completed-period row and update both regressions.

        安全加入一条已完成时期记录，并更新两条回归。

        Candidate statistics are calculated before the real history changes.
        Thus a wrong row or numerical overflow cannot partly update the object.
        / 候选统计量在真实历史变化之前算好，所以错误记录或数值溢出不会只更新一半。
        """

        if not isinstance(observation, MarketObservation):
            raise TypeError(
                "Only MarketObservation records may be appended. / "
                "只能追加 MarketObservation 记录。"
            )

        active_transaction = self._active_append_transaction
        if active_transaction is not None:
            if len(active_transaction.appended_rows) >= (
                active_transaction.token.max_appends
            ):
                raise RuntimeError(
                    "The reversible transaction append limit was reached. / "
                    "已达到可逆事务的追加上限。"
                )
            if not self.is_full:
                raise RuntimeError(
                    "The rolling window stopped being full during a short path. / "
                    "短路径运行时滚动窗口不再是满的。"
                )

        oldest_row = self._history.oldest() if self.is_full else None
        candidate_demand = self._demand_statistics
        candidate_value = self._value_statistics
        if oldest_row is not None:
            candidate_demand = candidate_demand.remove(
                oldest_row.market_price_p,
                oldest_row.insensitive_order_z,
            )
            candidate_value = candidate_value.remove(
                oldest_row.informed_and_noise_order_y,
                oldest_row.fundamental_value_v,
            )

        candidate_demand = candidate_demand.add(
            observation.market_price_p,
            observation.insensitive_order_z,
        )
        candidate_value = candidate_value.add(
            observation.informed_and_noise_order_y,
            observation.fundamental_value_v,
        )
        expected_size = min(len(self) + 1, self.window_size)
        if (
            candidate_demand.sample_size != expected_size
            or candidate_value.sample_size != expected_size
        ):
            raise RuntimeError("Rolling OLS size mismatch. / 滚动 OLS 样本量失配。")

        next_updates_since_resynchronization = (
            self._updates_since_resynchronization + 1
        )
        should_resynchronize = (
            next_updates_since_resynchronization >= self.resynchronize_every
        )
        if should_resynchronize:
            # Build the would-be new snapshot before touching the deque. This
            # preserves all-or-nothing behavior even during a rebuild. / 在真正
            # 修改 deque 前构造“更新后的快照”，让重建也保持全有或全无。
            existing_rows = self.snapshot()
            if oldest_row is not None:
                existing_rows = existing_rows[1:]
            candidate_rows = existing_rows + (observation,)
            candidate_demand, candidate_value = _rebuilt_statistics_from_rows(
                candidate_rows
            )

        evicted_row = self._history.append(observation)
        if evicted_row != oldest_row:
            raise RuntimeError("Unexpected rolling-history eviction. / 滚动历史淘汰结果异常。")

        # Commit only after every candidate calculation succeeds. / 所有候选计算
        # 成功后才统一提交。
        self._demand_statistics = candidate_demand
        self._value_statistics = candidate_value
        self._successful_append_count += 1
        if should_resynchronize:
            self._updates_since_resynchronization = 0
            self._resynchronization_count += 1
        else:
            self._updates_since_resynchronization = (
                next_updates_since_resynchronization
            )
        if active_transaction is not None:
            if evicted_row is None:
                raise RuntimeError(
                    "A reversible full-window append did not evict a row. / "
                    "可逆满窗口追加没有淘汰旧行。"
                )
            active_transaction.appended_rows.append(observation)
            active_transaction.evicted_rows.append(evicted_row)
        return evicted_row

    def force_resynchronize(self) -> None:
        """Rebuild both statistics from the current immutable snapshot.

        从当前不可变快照强制重建两组统计量。

        This is an engineering audit/recovery operation, not a different
        economic estimator. / 这是工程核对/恢复操作，不是另一种经济计量方法。
        """

        if self._active_append_transaction is not None:
            raise RuntimeError(
                "Cannot force-resynchronize inside a reversible transaction. / "
                "可逆事务中不能强制重同步。"
            )
        candidate_demand, candidate_value = _rebuilt_statistics_from_rows(
            self.snapshot()
        )
        self._demand_statistics = candidate_demand
        self._value_statistics = candidate_value
        self._updates_since_resynchronization = 0
        self._resynchronization_count += 1

    def _estimates_from_current_statistics(self) -> MarketMakerOLSEstimates:
        """Translate internal lines to the paper-named coefficients.

        把内部回归直线转换为论文命名的系数。
        """

        if self._demand_statistics.sample_size != self._value_statistics.sample_size:
            raise RuntimeError("The two OLS samples are misaligned. / 两条 OLS 样本没有对齐。")
        xi_0_hat, raw_demand_slope = self._demand_statistics.fitted_line("p")
        gamma_0_hat, gamma_1_hat = self._value_statistics.fitted_line("y")
        return MarketMakerOLSEstimates(
            xi_0_hat=xi_0_hat,
            # Paper: z=xi_0-xi_1*p. Ordinary slope is therefore -xi_1.
            # 论文写 z=xi_0-xi_1*p，所以普通斜率等于 -xi_1。
            xi_1_hat=-raw_demand_slope,
            gamma_0_hat=gamma_0_hat,
            gamma_1_hat=gamma_1_hat,
            sample_size=self._demand_statistics.sample_size,
        )

    def estimates(
        self,
        *,
        require_full_window: bool = True,
    ) -> MarketMakerOLSEstimates:
        """Return frozen OLS estimates; paper mode requires a full window.

        返回冻结的 OLS 估计；论文模式默认要求完整窗口。

        `False` exists only for small hand tests. Step 25 must keep the default
        `True`. / `False` 只用于小型手算测试；第 25 步必须保留默认的 `True`。
        """

        if not isinstance(require_full_window, bool):
            raise TypeError("require_full_window must be bool. / require_full_window 必须是布尔值。")
        if require_full_window and not self.is_full:
            raise ValueError(
                "Paper-mode OLS requires a full T_m window. / "
                "论文模式 OLS 需要完整的 T_m 窗口。"
            )
        if len(self) < 2:
            raise ValueError("OLS requires at least two observations. / OLS 至少需要两条观测。")

        try:
            return self._estimates_from_current_statistics()
        except ValueError:
            # A tiny non-positive S_xx may be accumulated floating-point drift.
            # Rebuild once; if the sample truly has no variation, the second
            # attempt still raises instead of inventing a coefficient. / 极小的
            # 非正 S_xx 可能是浮点漂移。先精确重建一次；若数据确实无变化，第二次
            # 仍会报错，不会虚构系数。
            self.force_resynchronize()
            return self._estimates_from_current_statistics()


def main() -> None:
    """Validate every rolling window against readable Steps 23 and 24.

    把每个滚动窗口与可读的第 23、24 步逐一核对。
    """

    toy_window_size = 4
    rolling = RollingMarketMakerOLS(
        window_size=toy_window_size,
        # Deliberately frequent in this test so rebuilds are visible. The paper
        # baseline defaults to every T_m appends. / 测试中故意频繁重建；论文规模
        # 默认每 T_m 次追加重建。
        resynchronize_every=3,
    )
    reference_rows: deque[MarketObservation] = deque(maxlen=toy_window_size)
    maximum_coefficient_difference = 0.0
    maximum_price_difference = 0.0
    full_windows_checked = 0
    evictions_checked = 0

    for time_index in range(20):
        price_p = 0.98 + 0.005 * time_index + 0.001 * (time_index % 2)
        insensitive_z = 500.0 - 500.0 * price_p + 0.1 * ((time_index % 3) - 1)
        order_flow_y = -3.0 + 0.7 * time_index + 0.2 * (time_index % 2)
        value_v = 0.9 + 0.05 * order_flow_y + 0.01 * ((time_index % 4) - 1.5)
        row = MarketObservation(
            fundamental_value_v=value_v,
            market_price_p=price_p,
            insensitive_order_z=insensitive_z,
            informed_and_noise_order_y=order_flow_y,
        )

        expected_eviction = reference_rows[0] if len(reference_rows) == toy_window_size else None
        returned_eviction = rolling.append_completed_observation(row)
        reference_rows.append(row)
        assert returned_eviction == expected_eviction
        if returned_eviction is not None:
            evictions_checked += 1
        assert rolling.snapshot() == tuple(reference_rows)

        if time_index == 0:
            try:
                rolling.estimates()
            except ValueError:
                incomplete_window_was_rejected = True
            else:
                incomplete_window_was_rejected = False
            assert incomplete_window_was_rejected

        if len(rolling) >= 2:
            fast_estimates = rolling.estimates(require_full_window=False)
            readable_estimates = fit_market_maker_regressions(rolling.snapshot())
            fast_coefficients = np.array(
                [
                    fast_estimates.xi_0_hat,
                    fast_estimates.xi_1_hat,
                    fast_estimates.gamma_0_hat,
                    fast_estimates.gamma_1_hat,
                ]
            )
            readable_coefficients = np.array(
                [
                    readable_estimates.xi_0_hat,
                    readable_estimates.xi_1_hat,
                    readable_estimates.gamma_0_hat,
                    readable_estimates.gamma_1_hat,
                ]
            )
            coefficient_difference = float(
                np.max(np.abs(fast_coefficients - readable_coefficients))
            )
            maximum_coefficient_difference = max(
                maximum_coefficient_difference,
                coefficient_difference,
            )
            np.testing.assert_allclose(
                fast_coefficients,
                readable_coefficients,
                rtol=1e-10,
                atol=1e-10,
            )

            if rolling.is_full:
                # Default paper-mode call must now succeed. / 窗口已满后，默认
                # 论文模式调用必须成功。
                assert rolling.estimates() == fast_estimates
                probe_order_flow_y = 1.25
                fast_quote = calculate_adaptive_price_quote(
                    probe_order_flow_y,
                    fast_estimates,
                    pricing_error_weight=0.1,
                )
                readable_quote = calculate_adaptive_price_quote(
                    probe_order_flow_y,
                    readable_estimates,
                    pricing_error_weight=0.1,
                )
                price_difference = abs(
                    fast_quote.continuous_price_p_hat
                    - readable_quote.continuous_price_p_hat
                )
                maximum_price_difference = max(
                    maximum_price_difference,
                    price_difference,
                )
                assert price_difference < 1e-10
                full_windows_checked += 1

    assert rolling.successful_append_count == 20
    assert evictions_checked == 16
    assert full_windows_checked == 17
    assert rolling.resynchronization_count >= 6

    # Exercise hundreds of pure add/remove updates without any periodic
    # rebuild. This protects the removal formulas themselves, not merely the
    # resynchronization path. / 在完全不触发定期重建的情况下执行数百次纯加入/移除，
    # 从而单独保护“移除旧样本”的公式。
    stress_window_size = 17
    stress_rolling = RollingMarketMakerOLS(
        window_size=stress_window_size,
        resynchronize_every=10_000,
    )
    stress_windows_checked = 0
    stress_maximum_coefficient_difference = 0.0
    for time_index in range(500):
        stress_price_p = (
            0.90 + 0.0002 * time_index + 0.003 * ((time_index * 7) % 13)
        )
        stress_order_flow_y = (
            -4.0 + 0.015 * time_index + 0.08 * ((time_index * 5) % 11)
        )
        stress_row = MarketObservation(
            fundamental_value_v=(
                0.95
                + 0.04 * stress_order_flow_y
                + 0.005 * ((time_index % 7) - 3)
            ),
            market_price_p=stress_price_p,
            insensitive_order_z=(
                500.0
                - 500.0 * stress_price_p
                + 0.03 * ((time_index % 5) - 2)
            ),
            informed_and_noise_order_y=stress_order_flow_y,
        )
        stress_rolling.append_completed_observation(stress_row)
        if stress_rolling.is_full:
            stress_fast = stress_rolling.estimates()
            stress_readable = fit_market_maker_regressions(
                stress_rolling.snapshot()
            )
            stress_fast_coefficients = np.array(
                [
                    stress_fast.xi_0_hat,
                    stress_fast.xi_1_hat,
                    stress_fast.gamma_0_hat,
                    stress_fast.gamma_1_hat,
                ]
            )
            stress_readable_coefficients = np.array(
                [
                    stress_readable.xi_0_hat,
                    stress_readable.xi_1_hat,
                    stress_readable.gamma_0_hat,
                    stress_readable.gamma_1_hat,
                ]
            )
            stress_difference = float(
                np.max(
                    np.abs(
                        stress_fast_coefficients
                        - stress_readable_coefficients
                    )
                )
            )
            stress_maximum_coefficient_difference = max(
                stress_maximum_coefficient_difference,
                stress_difference,
            )
            np.testing.assert_allclose(
                stress_fast_coefficients,
                stress_readable_coefficients,
                rtol=1e-10,
                atol=1e-8,
            )
            stress_windows_checked += 1
    assert stress_windows_checked == 484
    assert stress_rolling.resynchronization_count == 0

    # Wrong types and overflowing candidates must change absolutely nothing.
    # 错误类型和溢出候选必须完全不改变对象。
    snapshot_before_bad_append = rolling.snapshot()
    estimates_before_bad_append = rolling.estimates()
    append_count_before_bad_append = rolling.successful_append_count
    resync_count_before_bad_append = rolling.resynchronization_count
    try:
        rolling.append_completed_observation((1.0, 1.0, 0.0, 0.0))  # type: ignore[arg-type]
    except TypeError:
        wrong_type_was_rejected = True
    else:
        wrong_type_was_rejected = False
    assert wrong_type_was_rejected

    enormous_but_finite_row = MarketObservation(
        fundamental_value_v=1e308,
        market_price_p=1e308,
        insensitive_order_z=1e308,
        informed_and_noise_order_y=1e308,
    )
    try:
        rolling.append_completed_observation(enormous_but_finite_row)
    except ValueError:
        overflow_was_rejected = True
    else:
        overflow_was_rejected = False
    assert overflow_was_rejected
    assert rolling.snapshot() == snapshot_before_bad_append
    assert rolling.estimates() == estimates_before_bad_append
    assert rolling.successful_append_count == append_count_before_bad_append
    assert rolling.resynchronization_count == resync_count_before_bad_append

    # Truly constant explanatory variables remain an explicit error after an
    # exact rebuild; we do not clip S_xx or invent a fallback slope. / 解释变量
    # 真的不变时，即使精确重建后也必须报错；不能截断 S_xx 或虚构备用斜率。
    constant_price = RollingMarketMakerOLS(window_size=2, resynchronize_every=100)
    constant_price.append_completed_observation(MarketObservation(0.9, 1.0, 2.0, 3.0))
    constant_price.append_completed_observation(MarketObservation(1.1, 1.0, 4.0, 4.0))
    try:
        constant_price.estimates()
    except ValueError:
        constant_price_was_rejected = True
    else:
        constant_price_was_rejected = False
    assert constant_price_was_rejected

    # Vary p but keep y constant so the second regression's failure path is
    # tested independently. / 让 p 变化但 y 保持不变，独立测试第二条回归的报错路径。
    constant_order_flow = RollingMarketMakerOLS(
        window_size=2,
        resynchronize_every=100,
    )
    constant_order_flow.append_completed_observation(
        MarketObservation(0.9, 0.9, 50.0, 3.0)
    )
    constant_order_flow.append_completed_observation(
        MarketObservation(1.1, 1.1, -50.0, 3.0)
    )
    try:
        constant_order_flow.estimates()
    except ValueError:
        constant_order_flow_was_rejected = True
    else:
        constant_order_flow_was_rejected = False
    assert constant_order_flow_was_rejected

    for invalid_interval in (0, -1, 1.5, True):
        try:
            RollingMarketMakerOLS(
                window_size=4,
                resynchronize_every=invalid_interval,  # type: ignore[arg-type]
            )
        except ValueError:
            invalid_interval_was_rejected = True
        else:
            invalid_interval_was_rejected = False
        assert invalid_interval_was_rejected

    print("Step 24B: Fast rolling OLS parity / 步骤 24B：高效滚动 OLS 一致性")
    print(f"Rows appended / 已追加记录: {rolling.successful_append_count}")
    print(f"Full rolling windows checked / 已核对完整滚动窗口: {full_windows_checked}")
    print(f"FIFO evictions checked / 已核对先进先出淘汰: {evictions_checked}")
    print(f"Exact resynchronizations / 精确重同步次数: {rolling.resynchronization_count}")
    print(
        "Maximum coefficient difference / 最大系数差: "
        f"{maximum_coefficient_difference:.3e}"
    )
    print(
        "Maximum adaptive-price difference / 最大自适应价格差: "
        f"{maximum_price_difference:.3e}"
    )
    print(
        "No-resync stress windows / 未重同步压力测试窗口: "
        f"{stress_windows_checked}"
    )
    print(
        "No-resync maximum coefficient difference / 未重同步最大系数差: "
        f"{stress_maximum_coefficient_difference:.3e}"
    )
    print(
        "Ordinary add/drop updates are O(1); periodic rebuilds are O(T_m). / "
        "普通加入/移除更新为 O(1)；定期重建为 O(T_m)。"
    )
    print(
        "Fast estimates and prices match the readable reference every period. / "
        "高效估计与价格在每个时期都匹配可读基准。"
    )
    print("Validation passed / 验证通过")


if __name__ == "__main__":
    main()
