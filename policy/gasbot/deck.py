from __future__ import annotations

import math

import numpy as np

from policy.leak.geometry import FlangeGeometry

MAIN = {"y": 0.0, "z": 1.1, "r": 0.15}
SIDE = {"y": 3.2, "z": 0.8, "r": 0.12}
STUB = 0.45
FLANGES = (("M1", -6.2, "main"), ("M2", -1.8, "main"), ("M3", 1.6, "main"), ("M4", 4.4, "main"),
           ("S1", -4.0, "side"), ("S2", 0.8, "side"))
BOUNDS = ((-7.2, 7.2), (0.75, 2.4))
KEEP_OUT = (((3.2, 2.2), 0.45), ((3.6, 2.4), 0.45), ((5.7, 2.0), 0.7))
MACHINERY = (((5.7, 2.0), 74.0),)
ROUTE = ((-6.8, 1.25), (6.6, 1.25), (2.5, 1.25), (2.3, 2.05), (-6.8, 2.05))
TRUNK_Z = 0.45
STAND = 0.55
TIP_Z = 0.95


def flanges() -> list[FlangeGeometry]:
    out = []
    for fid, x, line in FLANGES:
        p = MAIN if line == "main" else SIDE
        s = 1.0 if line == "main" else -1.0
        out.append(FlangeGeometry(fid, [x, p["y"] + s * (p["r"] + STUB), p["z"]], [0, s, 0], [0, 0, 1]))
    return out


def stand_pose(g: FlangeGeometry) -> tuple[np.ndarray, float]:
    face = math.atan2(-g.normal[1], -g.normal[0])
    return g.center[:2] + STAND * g.normal[:2], face + math.pi / 2


def leak_point(g: FlangeGeometry) -> np.ndarray:
    return g.center + 0.03 * g.normal


def clamp(xy) -> tuple[np.ndarray, bool]:
    p = np.asarray(xy, float).copy()
    q = np.clip(p, [BOUNDS[0][0], BOUNDS[1][0]], [BOUNDS[0][1], BOUNDS[1][1]])
    blocked = bool(np.any(np.abs(q - p) > 1e-9))
    for c, r in KEEP_OUT:
        d = q - np.asarray(c)
        n = float(np.linalg.norm(d))
        if n < r:
            q = np.asarray(c) + (d / n if n > 1e-9 else np.array([0.0, -1.0])) * r
            blocked = True
    return q, blocked


class Route:
    def __init__(self, pts=ROUTE):
        self.pts = [np.asarray(p, float) for p in pts]
        self.i = 1

    def target(self) -> np.ndarray:
        return self.pts[self.i]

    def nearest(self, xy) -> None:
        segs = [(k, self.pts[k - 1], self.pts[k]) for k in range(len(self.pts))]
        best = min(segs, key=lambda s: _seg_dist(xy, s[1], s[2]))
        self.i = best[0]

    def step(self, xy, dist: float) -> tuple[np.ndarray, float, bool]:
        p = np.asarray(xy, float).copy()
        lap = False
        while dist > 1e-9:
            d = self.target() - p
            n = float(np.linalg.norm(d))
            if n <= dist:
                p, dist = self.target().copy(), dist - n
                self.i = (self.i + 1) % len(self.pts)
                lap = lap or self.i == 1
            else:
                p, dist = p + d / n * dist, 0.0
        d = self.target() - p
        return p, math.atan2(d[1], d[0]), lap


def _seg_dist(p, a, b) -> float:
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    t = np.clip((p - a) @ (b - a) / max(float((b - a) @ (b - a)), 1e-9), 0, 1)
    return float(np.linalg.norm(p - a - t * (b - a)))
