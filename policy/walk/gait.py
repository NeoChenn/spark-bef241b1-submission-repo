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
