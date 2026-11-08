from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

S_MIN, V_MIN = 0.5, 0.3
HUES = {"red": 0.0, "yellow": 50.0, "green": 120.0, "blue": 225.0, "purple": 290.0}
R0 = 0.033
LYING_ASPECT = 1.4
MAX_LEN, MAX_DIA = 0.35, 0.2
MIN_FILL = 0.5
RING_PX = 4


def hsv(rgb: np.ndarray):
    x = np.asarray(rgb, np.float32)[..., :3] / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    d = np.maximum(mx - mn, 1e-6)
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
    return h, s, mx


def colour_class(h: np.ndarray):
    names = list(HUES)
    c = np.array([HUES[n] for n in names])
    return names, np.abs((h[..., None] - c + 180.0) % 360.0 - 180.0).argmin(-1)


def world_points(us, vs, z, K, T) -> np.ndarray:
    p = np.stack([(us - K[0, 2]) / K[0, 0] * z, (vs - K[1, 2]) / K[1, 1] * z, z, np.ones_like(z)], -1)
    return (p @ T.T)[:, :3]


def plane_depth(us, vs, K, T, z0: float) -> np.ndarray:
    d = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones(len(us))], -1) @ T[:3, :3].T
    return (z0 - T[2, 3]) / np.where(np.abs(d[:, 2]) > 1e-9, d[:, 2], -1e-9)


def _depths(us, vs, depth, K, T, z0):
    if depth is None:
        return us, vs, plane_depth(us.astype(float), vs.astype(float), K, T, z0)
    z = depth[vs, us]
    ok = np.isfinite(z) & (z > 0)
    return us[ok], vs[ok], z[ok]


def _axes(xy: np.ndarray):
    c = xy.mean(0)
    _, e = np.linalg.eigh(np.cov((xy - c).T) + 1e-12 * np.eye(2))
    major, minor = e[:, 1], e[:, 0]
    return c, major, minor, (xy - c) @ major, (xy - c) @ minor


def _one(m, y0, x0, hot, s, depth, K, T, deck_z, name):
    vs, us = np.nonzero(m)
    vs, us = vs + y0, us + x0
    _, _, _, pa, pb = _axes(np.stack([us, vs], -1).astype(float))
    fill = len(us) / max((np.ptp(pa) + 1) * (np.ptp(pb) + 1), 1.0)
    if fill < MIN_FILL:
        return None
    su, sv, z = _depths(us, vs, depth, K, T, deck_z + R0)
    if len(z) < 3:
        return None
    P = world_points(su.astype(float), sv.astype(float), z, K, T)
    c, major, minor, a, b = _axes(P[:, :2])
    px = float(np.median(z)) / K[0, 0]
    L, D = float(np.ptp(a)) + px, float(np.ptp(b)) + px
    top = float(np.median(P[:, 2]))
    floor = None
    if depth is not None:
        h, w = hot.shape
        ys, xs = slice(max(y0 - RING_PX, 0), min(y0 + m.shape[0] + RING_PX, h)), \
            slice(max(x0 - RING_PX, 0), min(x0 + m.shape[1] + RING_PX, w))
        big = np.zeros(hot[ys, xs].shape, bool)
        big[y0 - ys.start:y0 - ys.start + m.shape[0], x0 - xs.start:x0 - xs.start + m.shape[1]] = m
        ring = ndimage.binary_dilation(big, iterations=RING_PX) & ~hot[ys, xs]
        rv, ru = np.nonzero(ring)
        ru, rv, rz = _depths(ru + xs.start, rv + ys.start, depth, K, T, deck_z)
        if len(rz):
            floor = float(np.median(world_points(ru.astype(float), rv.astype(float), rz, K, T)[:, 2]))
    height = max(top - floor, 0.0) if floor is not None else None
    lying = L / max(D, 1e-6) >= LYING_ASPECT or (height is not None and height <= 1.15 * D)
    if lying:
        length, diameter = L, D
        yaw = (math.atan2(major[1], major[0]) + math.pi / 2) % math.pi - math.pi / 2
    else:
        diameter = (L + D) / 2
        length = height if height is not None else L
        yaw = 0.0
    if length > MAX_LEN or diameter > MAX_DIA:
        return None
    xy = c + major * (a.min() + a.max()) / 2 + minor * (b.min() + b.max()) / 2
    if floor is not None:
        cz = (top + floor) / 2
    else:
        cz = deck_z + (diameter if lying else length) / 2
    score = float(np.mean(s[vs, us])) * min(1.0, fill / 0.8)
    return {"xyz": [round(float(xy[0]), 4), round(float(xy[1]), 4), round(cz, 4)], "yaw": round(float(yaw), 4),
            "lying": bool(lying), "diameter": round(diameter, 4), "length": round(length, 4),
            "score": round(min(score, 1.0), 3), "bbox": [int(us.min()), int(vs.min()), int(us.max()), int(vs.max())],
            "colour": name}


def detect_cans(rgb: np.ndarray, depth: np.ndarray | None, K, cam_to_world, min_px: int = 40,
                deck_z: float = 0.0) -> list[dict]:
    K, T = np.asarray(K, float), np.asarray(cam_to_world, float)
    dep = None if depth is None else np.asarray(depth, float)
    h, s, v = hsv(rgb)
    names, cls = colour_class(h)
    hot = (s > S_MIN) & (v > V_MIN)
    out = []
    for k, name in enumerate(names):
        lab, n = ndimage.label(ndimage.binary_fill_holes(hot & (cls == k)))
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            m = lab[sl] == i
            if m.sum() < min_px:
                continue
            det = _one(m, sl[0].start, sl[1].start, hot, s, dep, K, T, deck_z, name)
            if det is not None:
                out.append(det)
    return sorted(out, key=lambda d: -d["score"])


def noisy_pose(true_xyz, true_yaw: float, rng: np.random.Generator, sigma_xy: float = 0.01,
               sigma_yaw: float = 0.08) -> tuple[np.ndarray, float]:
    p = np.array(true_xyz, float)
    p[:2] += rng.normal(0.0, sigma_xy, 2)
    return p, float(true_yaw + rng.normal(0.0, sigma_yaw))
