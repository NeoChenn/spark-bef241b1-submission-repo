from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GAS_BASE = 0.02
GAS_NOISE = 0.004
LEAK_COEFF = 0.4
DECAY_M = 1.5
LEL_PER_G_S = LEAK_COEFF / 3.5
TURBULENCE = 0.12
MAX_STRETCH = 0.8


def stretch(wind) -> tuple[np.ndarray, float]:
    w = np.asarray(wind if wind is not None else (0.0, 0.0), float)[:2]
    u = float(np.linalg.norm(w))
    if u < 1e-6:
        return np.array([1.0, 0.0]), 0.0
    return w / u, min(MAX_STRETCH, 0.25 * u)


def shape(points_xy: np.ndarray, source_xy: np.ndarray, wind, decay_m: float = DECAY_M) -> np.ndarray:
    w, g = stretch(wind)
    d = np.asarray(points_xy, float) - np.asarray(source_xy, float)
    r = np.linalg.norm(d, axis=-1)
    return np.exp(-(r - g * (d @ w)) / decay_m)


def amplitude(leak: dict) -> float:
    if leak.get("amplitude_lel") is not None:
        return float(leak["amplitude_lel"])
    return LEL_PER_G_S * float(leak.get("rate_g_s", 0.0))


def leaks(state: dict) -> list[dict]:
    return [lk for lk in state.get("gas_leaks") or [] if amplitude(lk) > 0.0]


def true_concentration(state: dict, xy) -> float:
    p = np.asarray(xy, float)[None, :2]
    c = GAS_BASE
    for lk in leaks(state):
        c += amplitude(lk) * float(shape(p, np.asarray(lk["position"], float)[:2], state.get("wind"),
                                         float(lk.get("decay_m", DECAY_M)))[0])
    return c


@dataclass
class GasSensor:
    rng: np.random.Generator
    reads: int = 4
    dwell_s: float = 2.0

    def sample(self, state: dict, xy) -> float:
        mean = true_concentration(state, xy)
        excess = mean - GAS_BASE
        vals = []
        for _ in range(self.reads):
            turb = math.exp(self.rng.normal(0.0, TURBULENCE)) if excess > 0 else 1.0
            vals.append(GAS_BASE + excess * turb + self.rng.normal(0.0, GAS_NOISE))
        return float(np.mean(vals))

    def sigma(self, excess: float) -> float:
        return math.sqrt((GAS_NOISE ** 2 + (TURBULENCE * max(excess, 0.0)) ** 2) / self.reads)
