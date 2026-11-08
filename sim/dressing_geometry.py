from __future__ import annotations

import math

import numpy as np


def sag(p0, p1, depth: float, n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)[:, None]
    pts = np.asarray(p0, float) * (1 - s) + np.asarray(p1, float) * s
    pts[:, 2] -= depth * 4 * s[:, 0] * (1 - s[:, 0])
    return pts


def spline(ctrl, samples: int) -> np.ndarray:
    ctrl = np.asarray(ctrl, float)
    padded = np.vstack([2 * ctrl[0] - ctrl[1], ctrl, 2 * ctrl[-1] - ctrl[-2]])
    out = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1:i + 3]
        for u in np.linspace(0, 1, samples, endpoint=False):
            out.append(0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2 + (3 * p1 - p0 - 3 * p2 + p3) * u**3))
    out.append(ctrl[-1])
    return np.array(out)


def frames(pts: np.ndarray):
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    seed = np.array([0.0, 0.0, 1.0]) if abs(t[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n = np.zeros_like(t)
    n[0] = np.cross(np.cross(t[0], seed), t[0])
    n[0] /= np.linalg.norm(n[0])
    for i in range(1, len(t)):
        v = n[i - 1] - np.dot(n[i - 1], t[i]) * t[i]
        n[i] = v / np.linalg.norm(v)
    b = np.cross(t, n)
    return t, n, b


def tube(pts: np.ndarray, radius: float, sides: int) -> dict:
    t, n, b = frames(pts)
    angles = np.linspace(0, 2 * math.pi, sides + 1)
    ring = np.cos(angles)[None, :, None] * n[:, None, :] + np.sin(angles)[None, :, None] * b[:, None, :]
    points = (pts[:, None, :] + radius * ring).reshape(-1, 3)
    normals = ring.reshape(-1, 3)
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    uvs = np.stack(np.broadcast_arrays(angles[None, :] / (2 * math.pi), length[:, None] / (2 * math.pi * radius)), -1).reshape(-1, 2)
    row = sides + 1
    i, j = np.meshgrid(np.arange(len(pts) - 1), np.arange(sides), indexing="ij")
    a = i * row + j
    quads = np.stack([a, a + 1, a + row + 1, a + row], -1).reshape(-1)
    return {"points": points, "normals": normals, "uvs": uvs, "counts": np.full(len(quads) // 4, 4), "indices": quads}


def coil(center, inner: float, outer: float, turns: float, lift: float, samples: int = 48) -> np.ndarray:
    count = max(int(turns * samples), 2)
    s = np.linspace(0.0, 1.0, count)
    r = inner + (outer - inner) * s
    a = 2 * math.pi * turns * s
    return np.stack([center[0] + r * np.cos(a), center[1] + r * np.sin(a), np.full(count, center[2] + lift)], 1)


def chain_links(pts: np.ndarray, pitch: float):
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    stations = np.arange(pitch / 2, length[-1] - pitch / 2 + 1e-9, pitch)
    t, n, b = frames(pts)
    links = []
    for k, s in enumerate(stations):
        i = min(np.searchsorted(length, s), len(pts) - 1)
        centre = np.array([np.interp(s, length, pts[:, d]) for d in range(3)])
        side = n[i] if k % 2 == 0 else b[i]
        links.append((centre, np.stack([t[i], side, np.cross(t[i], side)], 1)))
    return links


def link_mesh(length: float, width: float, wire: float, sides: int = 8, steps: int = 24) -> dict:
    straight = max(length - width, 0.0) / 2
    half = width / 2 - wire
    path = []
    for k in range(steps):
        a = 2 * math.pi * k / steps
        x = math.cos(a) * half
        path.append((x + math.copysign(straight, x), math.sin(a) * half, 0.0))
    pts = np.array(path + [path[0]])
    return tube(pts, wire, sides)


def drape(boxes, extent, floor: float, res: int, seed: int = 0, fall: float = 0.35):
    x0, y0, x1, y1 = extent
    xs, ys = np.meshgrid(np.linspace(x0, x1, res), np.linspace(y0, y1, res), indexing="ij")
    zs = np.full(xs.shape, floor, float)
    for (bx0, by0), (bx1, by1), top in boxes:
        dx = np.maximum(np.maximum(bx0 - xs, xs - bx1), 0.0)
        dy = np.maximum(np.maximum(by0 - ys, ys - by1), 0.0)
        d = np.hypot(dx, dy)
        zs = np.maximum(zs, top - (top - floor) * np.clip(d / fall, 0, 1) ** 0.6 + 0.02)
    rng = np.random.default_rng(seed)
    wrinkle = sum(rng.uniform(0.004, 0.012) * np.sin(xs * rng.uniform(6, 14) + ys * rng.uniform(-8, 8) + rng.uniform(0, 6)) for _ in range(5))
    return xs, ys, np.maximum(zs + wrinkle, floor)


def _segment_distance(p, a, b) -> float:
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    u = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + u * ab)))


def blocked(p, route, rects, eyes, corridor: float = 1.2, eye_radius: float = 1.0) -> bool:
    if any(_segment_distance(p, a, b) < corridor for a, b in zip(route, route[1:])):
        return True
    if any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for x0, x1, y0, y1 in rects):
        return True
    return any(math.hypot(p[0] - e[0], p[1] - e[1]) < eye_radius for e in eyes)
