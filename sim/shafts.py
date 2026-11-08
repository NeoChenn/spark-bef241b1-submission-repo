from __future__ import annotations

import math

import numpy as np

RINGS, COLS = 12, 9
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


def card(spec: dict, eye, rings: int = RINGS, cols: int = COLS) -> np.ndarray:
    w = np.asarray(spec["aim"], float)
    apex = np.asarray(spec["apex"], float)
    s = np.linspace(0.0, 1.0, rings)
    centre = apex + (s * spec["length"])[:, None] * w
    view = np.asarray(eye, float) - centre
    side = np.cross(w, view)
    n = np.linalg.norm(side, axis=-1, keepdims=True)
    side = np.where(n > 1e-6, side / np.maximum(n, 1e-6), basis(w)[0])
    r = s * spec["length"] * math.tan(math.radians(spec["angle"]))
    u = np.linspace(-1.0, 1.0, cols)
    return (centre[:, None] + (r[:, None] * u[None])[..., None] * side[:, None]).reshape(-1, 3)


def topology(rings: int = RINGS, cols: int = COLS):
    r, c = np.meshgrid(np.arange(rings - 1), np.arange(cols - 1), indexing="ij")
    a = r * cols + c
    quads = np.stack((a, a + 1, a + cols + 1, a + cols), -1).reshape(-1).astype(np.int32)
    v, u = np.meshgrid(np.linspace(0.0, 1.0, rings), np.linspace(0.0, 1.0, cols), indexing="ij")
    return np.full(len(quads) // 4, 4, np.int32), quads, np.stack((u.reshape(-1), v.reshape(-1)), -1)


def inside(eye, spec: dict, margin: float = 1.15) -> bool:
    _, _, w = basis(spec["aim"])
    rel = np.asarray(eye, float) - np.asarray(spec["apex"])
    h = float(rel @ w)
    if h <= 0 or h >= spec["length"]:
        return False
    return float(np.linalg.norm(rel - h * w)) < h * math.tan(math.radians(spec["angle"])) * margin
