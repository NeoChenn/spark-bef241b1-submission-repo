"""Simulated gas plume and the robot's gas sensor.

Ported from `gas-leak-detection` flange_robot/env.py: background 0.02 of LEL, sensor noise 0.004, a leak
term that decays as exp(-d / 1.5 m), peak 0.4 of LEL for a full-strength leak. Extended here with wind:
the decay distance is shortened by the downwind component, so the plume stretches downwind and drops off
fast upwind, and with turbulent (multiplicative, log-normal) fluctuation on the leak term.

Leaks live in `World.state["gas_leaks"]`, the fleet coordinator's format (PR #7):
    [{"id": "leak_01", "position": [x, y, z], "rate_g_s": 3.5, "t0": 205.0, "source": "flange near valve_05",
      "isolated": False,
      # optional, this module only:
      "amplitude_lel": 0.4, "decay_m": 1.5, "flow_dir": [1, 0], "isolating_valves": ["valve_03"], "residual": 0.05}]
The plume amplitude is `amplitude_lel`, else rate_g_s * LEL_PER_G_S (3.5 g/s = the gas branch's full strength).
Every leak with a positive rate contributes, isolated or not (an isolated leak keeps its residual rate).
Wind lives in `World.state["wind"]` = [vx, vy] in m/s (default calm).

All values are ILLUSTRATIVE, as on the gas branch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GAS_BASE = 0.02        # background, fraction of LEL (gas-leak-detection env.py)
GAS_NOISE = 0.004      # sensor noise per reading (gas-leak-detection env.py)
LEAK_COEFF = 0.4       # full-strength leak at the source (gas-leak-detection env.py)
DECAY_M = 1.5          # decay distance (gas-leak-detection env.py)
LEL_PER_G_S = LEAK_COEFF / 3.5   # the coordinator demo's 3.5 g/s leak = full strength
TURBULENCE = 0.12      # log-normal sigma on the leak term (new here)
MAX_STRETCH = 0.8      # cap on the downwind stretch factor


def stretch(wind) -> tuple[np.ndarray, float]:
    """Unit downwind direction and stretch factor gamma in [0, MAX_STRETCH) for a wind vector."""
    w = np.asarray(wind if wind is not None else (0.0, 0.0), float)[:2]
    u = float(np.linalg.norm(w))
    if u < 1e-6:
        return np.array([1.0, 0.0]), 0.0
    return w / u, min(MAX_STRETCH, 0.25 * u)


def shape(points_xy: np.ndarray, source_xy: np.ndarray, wind, decay_m: float = DECAY_M) -> np.ndarray:
    """Unit-rate plume shape exp(-(r - gamma * d.w) / L) at points (N, 2) for sources (..., 2).

    Broadcasts: source_xy of shape (M, 1, 2) against points (N, 2) gives (M, N)."""
    w, g = stretch(wind)
    d = np.asarray(points_xy, float) - np.asarray(source_xy, float)
    r = np.linalg.norm(d, axis=-1)
    return np.exp(-(r - g * (d @ w)) / decay_m)


def amplitude(leak: dict) -> float:
    if leak.get("amplitude_lel") is not None:
        return float(leak["amplitude_lel"])
    return LEL_PER_G_S * float(leak.get("rate_g_s", 0.0))


def leaks(state: dict) -> list[dict]:
    """Leaks that currently emit (positive amplitude)."""
    return [lk for lk in state.get("gas_leaks") or [] if amplitude(lk) > 0.0]


def true_concentration(state: dict, xy) -> float:
    """Noise-free mean gas at a point (fraction of LEL), summed over all emitting leaks."""
    p = np.asarray(xy, float)[None, :2]
    c = GAS_BASE
    for lk in leaks(state):
        c += amplitude(lk) * float(shape(p, np.asarray(lk["position"], float)[:2], state.get("wind"),
                                         float(lk.get("decay_m", DECAY_M)))[0])
    return c


@dataclass
class GasSensor:
    """The robot's gas sensor: averages `reads` noisy readings at a point (about 0.5 s each)."""
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
        """Std of one averaged sample: sensor noise plus turbulence on the leak term."""
        return math.sqrt((GAS_NOISE ** 2 + (TURBULENCE * max(excess, 0.0)) ** 2) / self.reads)
