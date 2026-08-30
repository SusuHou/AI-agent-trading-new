"""Scientific-engine identity used by checkpoints and formal results.

The Git commit alone is not sufficient: a checkout can contain uncommitted
changes, and documentation-only commits do not change the simulated market.
We therefore hash the exact Python source files that define one scientific
trajectory.  A resumed checkpoint must present the same fingerprint.

/ 仅记录 Git commit 不够：代码可能有未提交修改。这里直接对决定模拟路径的
Python 文件计算哈希；续跑 checkpoint 时，哈希必须完全相同。
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib import metadata
import platform
from pathlib import Path


SCIENTIFIC_ENGINE_VERSION = 1

# Paths are relative to the ``dgj`` package.  ``session.py`` is included
# because it constructs grids, initial state, and chunked random continuation.
# Metrics and IRF files are deliberately excluded: they do not change the core
# training/measurement trajectory used by the low/high-noise experiment.
SCIENTIFIC_SOURCE_FILES = (
    "_jit.py",
    "config.py",
    "provenance.py",
    "environment/fundamental.py",
    "environment/insensitive_investors.py",
    "environment/noise_trader.py",
    "experiments/run_session_cli.py",
    "game/protocol.py",
    "game/session.py",
    "game/shocks.py",
    "players/benchmarks.py",
    "players/market_maker/adaptive.py",
    "players/market_maker/prehistory.py",
    "players/market_maker/theoretical.py",
    "players/speculator/action_space.py",
    "players/speculator/policy.py",
    "players/speculator/q_learning.py",
    "players/speculator/state_space.py",
)

ANALYSIS_SOURCE_FILES = (
    "dgj/provenance.py",
    "dgj/metrics/collusion.py",
    "dgj/metrics/market_quality.py",
    "dgj/metrics/trading_policy.py",
    "dgj/experiments/run_cell.py",
    "hpc/aggregate_dir.py",
)


@lru_cache(maxsize=1)
def scientific_source_fingerprint() -> str:
    """Return a stable SHA-256 fingerprint of trajectory-defining source.

    / 返回决定实验路径的源代码 SHA-256 指纹。
    """
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative in SCIENTIFIC_SOURCE_FILES:
        path = package_root / relative
        if not path.is_file():
            raise RuntimeError(f"scientific source file is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        # Git checkouts may use CRLF on Windows and LF on Narval. Normalize
        # newlines so identical source has one cross-platform identity.
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def scientific_identity() -> dict[str, object]:
    """JSON-ready scientific identity. / 可直接写入 JSON 的科学代码身份。"""
    return {
        "scientific_engine_version": SCIENTIFIC_ENGINE_VERSION,
        "scientific_source_fingerprint": scientific_source_fingerprint(),
        **scientific_runtime_identity(),
    }


@lru_cache(maxsize=1)
def analysis_source_fingerprint() -> str:
    """Hash the code that converts saved rows into reported statistics."""
    project_root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for relative in ANALYSIS_SOURCE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise RuntimeError(f"analysis source file is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def analysis_identity() -> dict[str, object]:
    """Identity of the offline metric/aggregation implementation."""
    return {
        "analysis_source_fingerprint": analysis_source_fingerprint(),
        **scientific_runtime_identity(),
    }


@lru_cache(maxsize=1)
def scientific_runtime_identity() -> dict[str, str | bool | None]:
    """Versions that can affect RNG or compiled numerical behavior.

    / 记录可能影响随机数或编译计算结果的软件版本。
    """
    def package_version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    # Import the project's actual execution switch, not merely package
    # metadata: Numba can be installed yet fail to import and trigger fallback.
    from dgj._jit import HAVE_NUMBA

    return {
        "python_version": platform.python_version(),
        "numpy_version": package_version("numpy"),
        "numba_version": package_version("numba"),
        "numba_enabled": bool(HAVE_NUMBA),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }
