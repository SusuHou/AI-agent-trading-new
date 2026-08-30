"""Optional numba: compiled when available, plain Python otherwise.

有 numba 就编译成机器码；没有就退化为纯 Python（结果完全相同，只是慢约 200 倍）。
Install with: py -3 -m pip install numba
"""

try:  # pragma: no cover - depends on environment
    from numba import njit as _numba_njit

    HAVE_NUMBA = True

    def njit(*args, **kwargs):
        """numba.njit with cache=True by default; usable bare or with options."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _numba_njit(cache=True)(args[0])
        kwargs.setdefault("cache", True)
        return _numba_njit(*args, **kwargs)

except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        """Identity decorator so every kernel also runs as ordinary Python."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(function):
            return function

        return wrap
