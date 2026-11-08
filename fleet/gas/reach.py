from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from fleet.contract import RIG_META

STANDOFFS = (0.35, 0.5)
LATERAL = (0.0, -0.3, 0.3)
YAWS_DEG = (180, 90, 135, 225, 270, 45, 0, 315)


@lru_cache(maxsize=1)
def model():
    from integration.kinematics import RobotModel
    return RobotModel.from_urdf()


@lru_cache(maxsize=1)
def rig_meta() -> dict:
    try:
        return json.loads(RIG_META.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def patrol_route() -> list[list[float]]:
    return rig_meta().get("patrol_route_m") or []


def deck_below(z: float) -> float:
    decks = sorted((rig_meta().get("deck_heights_m") or {"top_deck": 42.5}).values())
    below = [d for d in decks if d <= z + 1e-6]
    return below[-1] if below else decks[0]


def facing_of(asset: dict, world_assets: dict | None = None) -> np.ndarray:
    p = np.asarray(asset["position_m"], float)
    if asset.get("facing"):
        f = np.asarray(asset["facing"], float)[:2]
        return f / np.linalg.norm(f)
    gauges = [a for a in (world_assets or {}).values() if a.get("type") == "gauge" and a.get("facing")]
    if gauges:
        g = min(gauges, key=lambda a: np.linalg.norm(np.asarray(a["position_m"], float) - p))
        if np.linalg.norm(np.asarray(g["position_m"], float)[:2] - p[:2]) < 4.0:
            f = np.asarray(g["facing"], float)[:2]
            return f / np.linalg.norm(f)
    route = patrol_route()
    if route:
        q = min((np.asarray(r, float)[:2] for r in route), key=lambda r: np.linalg.norm(r - p[:2]))
        d = q - p[:2]
        if np.linalg.norm(d) > 1e-6:
            return d / np.linalg.norm(d)
    return np.array([0.0, -1.0])


@dataclass
class Reach:
    reachable: bool
    stance_xy: Optional[tuple[float, float]] = None
    yaw: Optional[float] = None
    position_error_m: float = math.inf
    q: Optional[dict] = None
    tried: int = 0
    reason: str = ""

    def summary(self) -> dict:
        return {"reachable": self.reachable, "stance_xy": self.stance_xy,
                "yaw_deg": None if self.yaw is None else round(math.degrees(self.yaw)),
                "ik_error_m": None if not math.isfinite(self.position_error_m) else round(self.position_error_m, 4),
                "stances_tried": self.tried, "reason": self.reason}


def check_reach(target_xyz, facing_xy, deck_z: Optional[float] = None) -> Reach:
    from integration.kinematics import TRUNK_HEIGHT, pose
    m = model()
    target = np.asarray(target_xyz, float)
    f = np.asarray(facing_xy, float)[:2]
    f = f / np.linalg.norm(f)
    deck = deck_below(target[2]) if deck_z is None else deck_z
    approach = np.array([-f[0], -f[1], 0.0])
    best, tried = None, 0
    height = target[2] - deck
    for d in STANDOFFS:
        for lat in LATERAL:
            sx = target[0] + d * f[0] - lat * f[1]
            sy = target[1] + d * f[1] + lat * f[0]
            for yd in YAWS_DEG:
                yaw = math.radians(yd)
                r = m.solve_ik(pose(sx, sy, deck + TRUNK_HEIGHT, yaw), m.default_q(), target, approach)
                tried += 1
                if best is None or r.position_error < best[0].position_error:
                    best = (r, (sx, sy), yaw)
                if r.reached:
                    return Reach(True, (round(sx, 3), round(sy, 3)), yaw, r.position_error, r.q, tried,
                                 f"IK converged at {height:.2f} m above the deck")
    r, xy, yaw = best
    return Reach(False, (round(xy[0], 3), round(xy[1], 3)), yaw, r.position_error, None, tried,
                 f"no stance reaches a target {height:.2f} m above the deck (best IK error "
                 f"{r.position_error:.2f} m over {tried} stances)")
