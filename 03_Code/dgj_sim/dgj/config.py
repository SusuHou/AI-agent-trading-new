"""Paper calibration and the labelled replication choices. / 论文参数与明确标注的复现选择。

One frozen object per experiment cell. Its hash names the output directory and
seeds the random streams, so two runs with the same cell are the same experiment.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json


@dataclass(frozen=True)
class PaperParameters:
    """Baseline calibration, Section 4.2. / 第 4.2 节基准校准。"""

    num_speculators: int = 2           # I
    value_mean: float = 1.0            # v_bar
    value_std: float = 1.0             # sigma_v (continuous)
    noise_std: float = 0.1             # sigma_u  (0.1 low noise, 100 high noise)
    investor_slope: float = 500.0      # xi
    pricing_error_weight: float = 0.1  # theta
    discount_factor: float = 0.95      # rho
    learning_rate: float = 0.01        # alpha
    exploration_decay: float = 5e-7    # beta
    num_value_points: int = 10         # n_v
    num_action_points: int = 15        # n_x
    num_price_points: int = 31         # n_p
    grid_widening: float = 0.1         # iota
    market_maker_window: int = 10_000  # T_m
    convergence_periods: int = 1_000_000   # policy unchanged for this many periods (p.26)
    measurement_periods: int = 100_000     # T after T_c (OA 4.1)

    def __post_init__(self) -> None:
        if self.num_speculators < 1:
            raise ValueError("num_speculators must be >= 1")
        if self.value_std <= 0 or self.noise_std <= 0:
            raise ValueError("standard deviations must be positive")
        if self.investor_slope < 0:
            raise ValueError("xi must be non-negative")
        if self.pricing_error_weight <= 0:
            raise ValueError("theta must be positive")
        if not 0 < self.discount_factor < 1:
            raise ValueError("rho must lie in (0, 1)")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("alpha must lie in (0, 1]")
        if self.exploration_decay <= 0:
            raise ValueError("beta must be positive")
        if self.num_value_points < 2 or self.num_action_points < 2 or self.num_price_points < 2:
            raise ValueError("grids need at least two points")
        if self.num_action_points > 62:
            raise ValueError("policy bit masks support at most 62 actions")
        if self.market_maker_window < 2:
            raise ValueError("T_m must be >= 2")
        if self.convergence_periods < 1 or self.measurement_periods < 1:
            raise ValueError("convergence/measurement periods must be positive")


@dataclass(frozen=True)
class ExperimentCell:
    """Parameters + every choice the paper leaves open. / 参数 + 论文未说明处的每一个选择。"""

    parameters: PaperParameters = PaperParameters()
    label: str = "baseline_low_noise"
    prehistory: str = "nash"           # A3: how D_0 is built ("nash" | "cartel")
    price_mapping: str = "nearest"     # A2: continuous p -> P (nearest, clip, tie -> lower)
    price_grid: str = "per_value"      # A5: P(v_{t-1}) per value (footnote 25) | "global" (steps/ reading)
    training_tie_rule: str = "uniform"  # exact Q ties during exploitation in training
    measurement_tie_rule: str = "lowest_index"  # frozen greedy policy after convergence

    def with_parameters(self, **changes) -> "ExperimentCell":
        return replace(self, parameters=replace(self.parameters, **changes))

    def to_dict(self) -> dict:
        return {
            "parameters": asdict(self.parameters),
            "label": self.label,
            "prehistory": self.prehistory,
            "price_mapping": self.price_mapping,
            "price_grid": self.price_grid,
            "training_tie_rule": self.training_tie_rule,
            "measurement_tie_rule": self.measurement_tie_rule,
        }

    def key(self) -> str:
        """Stable 16-hex identity of the cell (label excluded on purpose)."""
        payload = self.to_dict()
        payload.pop("label")
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def key_uint32(self) -> int:
        """First 32 bits of the key, used as a SeedSequence spawn key."""
        return int(self.key()[:8], 16)
