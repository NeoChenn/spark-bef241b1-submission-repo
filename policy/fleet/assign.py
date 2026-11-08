from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

INF = 1e9


def assign(cost: np.ndarray) -> list[tuple[int, int]]:
    cost = np.asarray(cost, np.float64)
    if cost.size == 0:
        return []
    rows, cols = linear_sum_assignment(np.minimum(cost, INF))
    return [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < INF]


def pick_responder(eta: np.ndarray, feasible: np.ndarray):
    eta = np.where(np.asarray(feasible, bool), np.asarray(eta, np.float64), np.inf)
    if not np.isfinite(eta).any():
        return None
    return int(np.argmin(eta))
