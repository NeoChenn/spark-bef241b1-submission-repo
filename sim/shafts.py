from __future__ import annotations

import math

import numpy as np

SEGS, RINGS = 40, 12
DECK_Z, SEA_Z = 42.5, 0.3
PICKS = (
    ("area", 0, DECK_Z, 30.0, 1.0, True),
    ("area", 3, DECK_Z, 20.0, 0.8, True),
    ("area", 4, DECK_Z, 20.0, 0.8, False),
    ("sea", 0, SEA_Z, 24.0, 1.0, False),
    ("sea", 1, SEA_Z, 24.0, 1.0, False),
)
TIERS = {"none": 0, "live": 2, "offline": len(PICKS)}
DOWN = (0.0, 0.0, -1.0)


def specs(area, sea, tier: str) -> list[dict]:
    out = []
    for kind, index, floor, angle, gain, live in PICKS[:TIERS[tier]]:
        if kind == "area":
            _, apex, kelvin, _, aim, _ = area[index]
        else:
            apex, kelvin, _ = sea[index]
            aim = DOWN
        w = np.asarray(aim, float) / np.linalg.norm(aim)
        out.append({"apex": tuple(map(float, apex)), "aim": tuple(map(float, w)), "angle": angle, "gain": gain,
                    "kelvin": float(kelvin), "length": float((apex[2] - floor) / max(-w[2], 1e-3))})
    return out


def basis(aim):
    w = np.asarray(aim, float) / np.linalg.norm(aim)
    up = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(up, w)
    u /= np.linalg.norm(u)
    return u, np.cross(w, u), w


def cone(spec: dict, segs: int = SEGS, rings: int = RINGS):
    u, v, w = basis(spec["aim"])
    ta = math.tan(math.radians(spec["angle"]))
    s = np.linspace(0.0, 1.0, rings)
    a = np.arange(segs) * 2 * math.pi / segs
    ring = np.cos(a)[:, None] * u + np.sin(a)[:, None] * v
    pts = (np.asarray(spec["apex"]) + (s * spec["length"])[:, None, None] * (w + ta * ring[None]))
    ca, sa = math.cos(math.atan(ta)), math.sin(math.atan(ta))
    nrm = np.broadcast_to(ca * ring - sa * w, (rings, segs, 3))
    r, k = np.meshgrid(np.arange(rings - 1), np.arange(segs), indexing="ij")
    a0 = r * segs + k
    a1 = r * segs + (k + 1) % segs
    quads = np.stack((a0, a1, a1 + segs, a0 + segs), -1).reshape(-1)
    return (pts.reshape(-1, 3), nrm.reshape(-1, 3).copy(), np.repeat(s, segs), np.full(len(quads) // 4, 4),
            quads.astype(np.int32))


def inside(eye, spec: dict, margin: float = 1.15) -> bool:
    _, _, w = basis(spec["aim"])
    rel = np.asarray(eye, float) - np.asarray(spec["apex"])
    h = float(rel @ w)
    if h <= 0 or h >= spec["length"]:
        return False
    return float(np.linalg.norm(rel - h * w)) < h * math.tan(math.radians(spec["angle"])) * margin


def smooth(e0: float, e1: float, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def alpha(points, normals, s, eye, spec: dict, t: float = 0.0) -> np.ndarray:
    if inside(eye, spec):
        return np.zeros(len(points))
    view = np.asarray(eye, float) - points
    d = np.linalg.norm(view, axis=-1)
    facing = np.abs(np.einsum("ij,ij->i", normals, view / np.maximum(d, 1e-6)[:, None]))
    along = smooth(0.0, 0.08, s) * (1 - s) ** 1.3 * smooth(1.0, 0.7, s)
    near = smooth(2.0, 8.0, d)
    p = points * 0.35
    motes = 0.82 + 0.18 * np.sin(p[:, 0] + 0.7 * t) * np.sin(p[:, 1] * 1.3 - 0.5 * t) * np.sin(p[:, 2] * 0.9 + 1.1 * t)
    return np.clip(facing ** 1.5 * along * near * motes, 0.0, 1.0)
