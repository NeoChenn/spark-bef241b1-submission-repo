"""LeakWorld: the leak equivalent of RustState. Pure numpy, no Isaac.

Holds, per flange, how far each bolt has turned and how badly the flange is cracked, and turns that into a
gas field. The orchestrator never reads this directly: it sees gas through ``bridge.gas_reading()`` and the
bolts and cracks only through the camera. Tests and the demo read it as ground truth.

The gas model is ILLUSTRATIVE (same constants as flange_robot/env.py on the gas-leak-detection branch):
leak strength = total bolt looseness / 0.6 rad + crack severity, capped at 1; the reading relaxes toward
that strength with a 0.5 s time constant and falls off with distance (1.5 m length scale).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from policy.leak.geometry import FlangeGeometry

GAS_BASE = 0.02          # background reading, fraction of LEL
LEAK_COEFF = 0.4         # extra reading at the flange for a full-strength leak
LEAK_FULL_RAD = 0.6      # total bolt misalignment giving a full-strength leak
RELAX_TAU_S = 0.5        # gas takes time to build up and to clear
PLUME_LENGTH_M = 1.5


@dataclass
class FlangeState:
    geom: FlangeGeometry
    angles: np.ndarray                       # (n_bolts,) radians each bolt has turned
    crack: float = 0.0                       # 0..1
    level: float = 0.0                       # current leak strength 0..1
    history: list = field(default_factory=list)


class LeakWorld:
    def __init__(self, flanges: list[FlangeGeometry]):
        self.flanges = {f.id: FlangeState(f, np.zeros(f.n_bolts)) for f in flanges}
        self.t = 0.0

    # -- truth and dynamics ---------------------------------------------------
    def target_level(self, fid: str) -> float:
        s = self.flanges[fid]
        return float(np.clip(np.abs(s.angles).sum() / LEAK_FULL_RAD + s.crack, 0.0, 1.0))

    def step(self, dt: float) -> None:
        self.t += dt
        a = 1.0 - math.exp(-dt / RELAX_TAU_S)
        for fid, s in self.flanges.items():
            s.level += a * (self.target_level(fid) - s.level)

    def settle(self) -> None:
        """Jump every leak straight to its steady state (scenario setup)."""
        for fid, s in self.flanges.items():
            s.level = self.target_level(fid)

    def gas_at(self, p) -> float:
        p = np.asarray(p, np.float64)
        g = GAS_BASE
        for s in self.flanges.values():
            d = float(np.linalg.norm(p - s.geom.center))
            g += LEAK_COEFF * s.level * math.exp(-d / PLUME_LENGTH_M)
        return g

    # -- scene changes (what the video hooks call) -----------------------------
    def loosen(self, fid: str, bolt: int, radians: float) -> None:
        self.flanges[fid].angles[bolt] = radians

    def crack(self, fid: str, severity: float = 0.3) -> None:
        self.flanges[fid].crack = float(np.clip(severity, 0.0, 1.0))

    def gas_spike(self, fid: str) -> None:
        """A burst: full-strength leak. It must trip the stop limit from the viewing distance (1.1 m for the
        640x360 wrist camera), which a 0.9 crack does not (0.193 < 0.20)."""
        self.crack(fid, 1.0)

    # -- robot actions -----------------------------------------------------------
    def tighten(self, fid: str, bolt: int) -> float:
        """Re-torque one bolt. Returns how far it had turned. Does not touch a crack."""
        s = self.flanges[fid]
        before = float(s.angles[bolt])
        s.angles[bolt] = 0.0
        s.history.append((self.t, bolt, before))
        return before

    def truth(self, fid: str) -> dict:
        s = self.flanges[fid]
        return {"angles": s.angles.round(4).tolist(), "crack": s.crack, "level": round(s.level, 4),
                "max_angle": float(np.abs(s.angles).max())}
