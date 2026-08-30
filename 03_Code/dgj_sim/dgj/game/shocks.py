"""Independent, reproducible random streams — replication choice A4.

每个 session 拥有自己的一组随机流，互不影响、可复现、与执行顺序无关：

    spawn_key = (experiment_seed, cell_key32, session_index, stream_id)

Streams / 随机流:
    0 initial_state   uniform s_0 over P x V x V
    1 value           v_{t+1} index, uniform over V (equal-probability grid)
    2 noise           u_t ~ N(0, sigma_u^2)
    3+2i mode_i       U[0,1): explore vs exploit for speculator i
    4+2i action_i     large uniform int: exploratory action / tie break for speculator i

Draws are made in CHUNKS in Python and handed to the kernel as arrays, so the
kernel itself contains no RNG and results do not depend on numba/threads.
"""

from dataclasses import dataclass

import numpy as np

from dgj.config import ExperimentCell
from dgj.environment import noise_trader

STREAM_INITIAL_STATE, STREAM_VALUE, STREAM_NOISE = 0, 1, 2
ACTION_DRAW_LIMIT = 1 << 62


@dataclass(frozen=True)
class Shocks:
    """One chunk of pre-drawn randomness. / 一段预先抽好的随机数。"""

    value_index: np.ndarray   # int64[n]
    noise: np.ndarray         # float64[n]
    mode: np.ndarray          # float64[I, n]
    action: np.ndarray        # int64[I, n]

    def __len__(self) -> int:
        return int(self.noise.shape[0])


class ShockStreams:
    """All generators for one session. / 一个 session 的全部随机流。"""

    def __init__(self, experiment_seed: int, cell: ExperimentCell, session_index: int) -> None:
        self.experiment_seed = int(experiment_seed)
        self.cell_key = cell.key()
        self.session_index = int(session_index)
        self.number_of_speculators = cell.parameters.num_speculators
        self.noise_std = cell.parameters.noise_std
        n_streams = 3 + 2 * self.number_of_speculators
        self._generators = [self._generator(k) for k in range(n_streams)]

    def _generator(self, stream_id: int) -> np.random.Generator:
        seq = np.random.SeedSequence(
            entropy=self.experiment_seed,
            spawn_key=(int(self.cell_key[:8], 16), self.session_index, stream_id),
        )
        return np.random.Generator(np.random.PCG64(seq))

    # --- one-off draws -------------------------------------------------
    def initial_state(self, num_prices: int, num_values: int) -> tuple[int, int, int]:
        g = self._generators[STREAM_INITIAL_STATE]
        return int(g.integers(num_prices)), int(g.integers(num_values)), int(g.integers(num_values))

    # --- chunked draws for the kernel -----------------------------------
    def draw(self, n: int, num_values: int) -> Shocks:
        I = self.number_of_speculators
        value_index = self._generators[STREAM_VALUE].integers(num_values, size=n, dtype=np.int64)
        noise = noise_trader.draw(self._generators[STREAM_NOISE], self.noise_std, n)
        mode = np.empty((I, n))
        action = np.empty((I, n), dtype=np.int64)
        for i in range(I):
            mode[i] = self._generators[3 + 2 * i].random(n)
            action[i] = self._generators[4 + 2 * i].integers(ACTION_DRAW_LIMIT, size=n, dtype=np.int64)
        return Shocks(value_index=value_index, noise=noise, mode=mode, action=action)

    # --- checkpoint support ---------------------------------------------
    def state(self) -> list[dict]:
        return [g.bit_generator.state for g in self._generators]

    def restore(self, states: list[dict]) -> None:
        for g, s in zip(self._generators, states, strict=True):
            g.bit_generator.state = s

    def manifest(self) -> dict:
        return {
            "experiment_seed": self.experiment_seed,
            "cell_key": self.cell_key,
            "session_index": self.session_index,
            "streams": 3 + 2 * self.number_of_speculators,
            "rng": "numpy PCG64 via SeedSequence(entropy, spawn_key=(cell32, session, stream))",
        }
