"""How often does the price fall outside a value's price grid? (register item A2)

    python hpc/clip_share.py RUN_DIR [RUN_DIR ...]

For each finished session (session_K.npz, 100,000 measurement rows) the price p_t is mapped to
the state index the traders see next period (grid row of v_t, as in the kernel). Reports the
share of periods at the two end indices (0 and n_p-1: prices clipped or in the outer bins), the
share strictly outside the grid range, and how many distinct price states each value uses.
Read-only. / 只读：统计价格落在网格边缘的比例。
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_dir import cell_from_dir  # noqa: E402
from dgj.game import protocol  # noqa: E402
from dgj.game.session import build_grids  # noqa: E402

for d in sys.argv[1:]:
    cell = cell_from_dir(d)
    g = build_grids(cell)
    P, V = g.price_grid, g.value_grid
    n_p = P.shape[1]
    ends, outside, distinct = [], [], []
    for f in sorted(glob.glob(os.path.join(d, "session_*.npz"))):
        rows = np.load(f, allow_pickle=False)["rows"]
        v, p = rows[:, protocol.COL_V], rows[:, protocol.COL_P]
        k = np.searchsorted(V, v)
        k = np.where((k < len(V)) & np.isclose(V[np.minimum(k, len(V) - 1)], v), k, np.abs(V[None, :] - v[:, None]).argmin(1))
        lo, hi = P[k, 0], P[k, -1]
        mid = (P[k, 1:] + P[k, :-1]) / 2          # nearest-point boundaries
        idx = (p[:, None] > mid).sum(1)
        ends.append(np.mean((idx == 0) | (idx == n_p - 1)))
        outside.append(np.mean((p < lo) | (p > hi)))
        distinct.append(np.mean([len(np.unique(idx[k == j])) for j in range(len(V)) if np.any(k == j)]))
    if not ends:
        print(f"{os.path.basename(d)}: no session files"); continue
    e, o, n = map(np.array, (ends, outside, distinct))
    print(f"{os.path.basename(d)} [{cell.price_grid}] n={len(e)}: share at end states mean {e.mean():.3f} "
          f"(median {np.median(e):.3f}, max {e.max():.3f}); outside grid range mean {o.mean():.3f}; "
          f"distinct price states used per value mean {n.mean():.1f} of {n_p}")
