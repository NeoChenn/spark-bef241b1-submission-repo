from __future__ import annotations

import math

import numpy as np
from scipy import stats as _st

Z95 = 1.959963984540054


def mean_ci(x, conf: float = 0.95) -> tuple[float, float, float]:
    a = np.asarray(list(x), dtype=np.float64)
    a = a[~np.isnan(a)]
    n = a.size
    if n == 0:
        return (math.nan, math.nan, math.nan)
    m = float(a.mean())
    if n == 1:
        return (m, m, m)
    half = float(_st.t.ppf(0.5 + conf / 2, n - 1) * a.std(ddof=1) / math.sqrt(n))
    return (m, m - half, m + half)


def rate_ci(k, n: int | None = None, z: float = Z95) -> tuple[float, float, float]:
    if n is None:
        b = np.asarray(list(k), dtype=bool)
        k, n = int(b.sum()), int(b.size)
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, 0.0 if k == 0 else max(0.0, c - h), 1.0 if k == n else min(1.0, c + h))


def bootstrap_ci(fn, *arrays, n_boot: int = 2000, conf: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    arrs = [np.asarray(a) for a in arrays]
    n = len(arrs[0])
    est = float(fn(*arrs))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = float(fn(*[a[idx] for a in arrs]))
        except ValueError:
            continue
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return (est, math.nan, math.nan)
    lo, hi = np.quantile(vals, [(1 - conf) / 2, 0.5 + conf / 2])
    return (est, float(lo), float(hi))


def episode_seeds(n: int, base: int = 0) -> list[int]:
    ss = np.random.SeedSequence(20261108 + base)
    return [int(s.generate_state(1)[0]) for s in ss.spawn(n)]


def summarize(x, rate: bool = False) -> dict:
    vals = list(x)
    m, lo, hi = rate_ci(vals) if rate else mean_ci(vals)
    n = len(vals) if rate else int(np.sum(~np.isnan(np.asarray(vals, dtype=np.float64)))) if vals else 0
    return {"mean": m, "lo": lo, "hi": hi, "n": n}
