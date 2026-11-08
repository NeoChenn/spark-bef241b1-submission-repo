from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from policy.leak.geometry import FlangeGeometry

GAS_BASE = 0.02
LEAK_COEFF = 0.4
LEAK_FULL_RAD = 0.6
RELAX_TAU_S = 0.5
PLUME_LENGTH_M = 1.5


@dataclass
class FlangeState:
    geom: FlangeGeometry
    angles: np.ndarray
    crack: float = 0.0
    level: float = 0.0
    history: list = field(default_factory=list)


class LeakWorld:
    def __init__(self, flanges: list[FlangeGeometry]):
        self.flanges = {f.id: FlangeState(f, np.zeros(f.n_bolts)) for f in flanges}
        self.t = 0.0

    def target_level(self, fid: str) -> float:
        s = self.flanges[fid]
        return float(np.clip(np.abs(s.angles).sum() / LEAK_FULL_RAD + s.crack, 0.0, 1.0))

    def step(self, dt: float) -> None:
        self.t += dt
        a = 1.0 - math.exp(-dt / RELAX_TAU_S)
        for fid, s in self.flanges.items():
            s.level += a * (self.target_level(fid) - s.level)

    def settle(self) -> None:
        for fid, s in self.flanges.items():
            s.level = self.target_level(fid)

    def gas_at(self, p) -> float:
        p = np.asarray(p, np.float64)
        g = GAS_BASE
        for s in self.flanges.values():
            d = float(np.linalg.norm(p - s.geom.center))
            g += LEAK_COEFF * s.level * math.exp(-d / PLUME_LENGTH_M)
        return g

    def loosen(self, fid: str, bolt: int, radians: float) -> None:
        self.flanges[fid].angles[bolt] = radians

    def crack(self, fid: str, severity: float = 0.3) -> None:
        self.flanges[fid].crack = float(np.clip(severity, 0.0, 1.0))

    def gas_spike(self, fid: str) -> None:
        self.crack(fid, 1.0)

    def tighten(self, fid: str, bolt: int) -> float:
        s = self.flanges[fid]
        before = float(s.angles[bolt])
        s.angles[bolt] = 0.0
        s.history.append((self.t, bolt, before))
        return before

    def truth(self, fid: str) -> dict:
        s = self.flanges[fid]
        return {"angles": s.angles.round(4).tolist(), "crack": s.crack, "level": round(s.level, 4),
                "max_angle": float(np.abs(s.angles).max())}
