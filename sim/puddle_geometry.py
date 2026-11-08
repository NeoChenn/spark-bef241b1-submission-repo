from __future__ import annotations

import math

import numpy as np

RINGS = (0.0, 0.45, 0.72, 0.88, 1.0)


def outline(radius: float, seed: int = 0, n: int = 48) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    r = np.ones(n)
    for k in range(2, 7):
        r += rng.uniform(0.04, 0.22) / (k - 0.8) * np.sin(k * a + rng.uniform(0, 2 * math.pi))
    r = np.clip(r, 0.55, 1.4) * radius
    return np.stack([np.cos(a) * r, np.sin(a) * r], 1)


def rim_weight(f, start: float = 0.72):
    t = np.clip((np.asarray(f, float) - start) / (1.0 - start), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def triangulate(n: int, rings: int) -> np.ndarray:
    tris = [(0, 1 + k, 1 + (k + 1) % n) for k in range(n)]
    for ring in range(1, rings - 1):
        a0, b0 = 1 + (ring - 1) * n, 1 + ring * n
        for k in range(n):
            j = (k + 1) % n
            tris += [(a0 + k, b0 + k, b0 + j), (a0 + k, b0 + j, a0 + j)]
    return np.array(tris, int)


def mesh(center, radius: float, seed: int = 0, n: int = 48, lift: float = 0.004) -> dict:
    edge = outline(radius, seed, n)
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    unit = np.stack([np.cos(a), np.sin(a)], 1)
    cx, cy, cz = (float(v) for v in center)
    pts, uvs, fs = [(cx, cy, cz + lift)], [(0.5, 0.5)], [0.0]
    for f in RINGS[1:]:
        for k in range(n):
            pts.append((cx + edge[k, 0] * f, cy + edge[k, 1] * f, cz + lift))
            uvs.append((0.5 + 0.5 * f * unit[k, 0], 0.5 + 0.5 * f * unit[k, 1]))
            fs.append(f)
    tris = triangulate(n, len(RINGS))
    return {"points": np.array(pts), "uvs": np.array(uvs), "f": np.array(fs), "rim": rim_weight(fs),
            "counts": np.full(len(tris), 3), "indices": tris.reshape(-1), "normals": np.tile((0.0, 0.0, 1.0), (len(pts), 1))}
