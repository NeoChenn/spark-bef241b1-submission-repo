import math

import numpy as np

LIFT = 0.85
NEAR = 0.1


def from_usd(world_matrix, focal: float, aperture: float, w: int, h: int) -> dict:
    M = np.asarray(world_matrix, np.float64)
    R = M[:3, :3].T
    R = R / np.linalg.norm(R, axis=0, keepdims=True)
    T = np.eye(4)
    T[:3, :3] = R @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = M[3, :3]
    f = focal / aperture * w
    K = [[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]]
    return {"K": np.round(K, 3).tolist(), "T": np.round(T, 5).tolist(), "w": int(w), "h": int(h)}


def look_at(eye, target, focal: float = 24.0, aperture: float = 20.955, w: int = 1280, h: int = 720) -> dict:
    from rig.sim.orbit import look_at_rows
    return from_usd(look_at_rows(tuple(eye), tuple(target)), focal, aperture, w, h)


def _kt(cam):
    return np.asarray(cam["K"], np.float64), np.asarray(cam["T"], np.float64)


def project(cam: dict, xyz) -> tuple[float, float, float]:
    K, T = _kt(cam)
    pc = (np.asarray(xyz, np.float64) - T[:3, 3]) @ T[:3, :3]
    z = float(pc[2])
    zs = z if abs(z) > 1e-9 else 1e-9
    return float(K[0, 0] * pc[0] / zs + K[0, 2]), float(K[1, 1] * pc[1] / zs + K[1, 2]), z


def unproject_ground(cam: dict, u: float, v: float, z: float = 0.0) -> list[float]:
    K, T = _kt(cam)
    d = T[:3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    o = T[:3, 3]
    s = (z - o[2]) / d[2]
    return [float(c) for c in o + s * d]


def clamp_edge(x: float, y: float, w: float, h: float, margin: float) -> tuple[float, float, float]:
    cx, cy = w / 2, h / 2
    dx, dy = x - cx, y - cy
    hx, hy = cx - margin, cy - margin
    s = min(hx / abs(dx) if dx else math.inf, hy / abs(dy) if dy else math.inf)
    return cx + dx * s, cy + dy * s, math.atan2(dy, dx)


def anchor(cam: dict, xyz, lift: float = LIFT, margin: float = 24.0, occluded: bool = False) -> dict:
    p = np.asarray(xyz, np.float64) + [0.0, 0.0, lift]
    u, v, z = project(cam, p)
    w, h = cam["w"], cam["h"]
    if z <= NEAR:
        return {"mode": "behind", "x": None, "y": None, "angle": None}
    if margin <= u <= w - margin and margin <= v <= h - margin:
        return {"mode": "occluded" if occluded else "on", "x": u, "y": v, "angle": None}
    x, y, a = clamp_edge(u, v, w, h, margin)
    return {"mode": "edge", "x": x, "y": y, "angle": a}
