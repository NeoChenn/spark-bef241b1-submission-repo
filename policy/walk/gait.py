from __future__ import annotations

import numpy as np


def gait_period(q: np.ndarray, dt: float, lo_s: float = 0.2, hi_s: float = 1.5) -> int:
    x = q - q.mean(0)
    lo, hi = max(1, int(round(lo_s / dt))), min(len(x) // 2, int(round(hi_s / dt)))
    scores = [float((x[:-k] * x[k:]).sum() / (len(x) - k)) for k in range(lo, hi + 1)]
    return lo + int(np.argmax(scores))


def loop_segment(q: np.ndarray, period: int, cycles: int = 2) -> np.ndarray:
    n = period * cycles
    return q[len(q) - n:]


def best_loop(q: np.ndarray, period: int, cycles: int = 2, slack: int = 3):
    best = (np.inf, 0, period * cycles)
    for n in range(max(2, period * cycles - slack), period * cycles + slack + 1):
        for s in range(0, len(q) - n):
            e = float(np.abs(q[s + n] - q[s]).max())
            if e < best[0]:
                best = (e, s, n)
    return best[1], best[2]
