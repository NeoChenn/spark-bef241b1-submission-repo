from __future__ import annotations

import math

import numpy as np

DECK_Z, SEA_Z = 42.5, 0.3
PICKS = (
    ("deck", 0, DECK_Z, 18.0, 1.0),
    ("deck", 1, DECK_Z, 18.0, 1.0),
    ("deck", 2, DECK_Z, 18.0, 1.0),
    ("deck", 3, DECK_Z, 18.0, 1.0),
    ("area", 3, DECK_Z, 20.0, 0.8),
    ("area", 4, DECK_Z, 20.0, 0.8),
    ("sea", 0, SEA_Z, 24.0, 1.0),
    ("sea", 1, SEA_Z, 24.0, 1.0),
)
KELVIN = {2000.0: 2100.0, 4000.0: 4500.0, 6000.0: 7000.0}
TIERS = {"none": 0, "live": 0, "offline": len(PICKS)}
DOWN = (0.0, 0.0, -1.0)


def specs(area, sea, deck, tier: str) -> list[dict]:
    out = []
    for kind, index, floor, angle, gain in PICKS[:TIERS[tier]]:
        if kind == "area":
            _, apex, kelvin, _, aim, _ = area[index]
        elif kind == "deck":
            apex, target, kelvin, _, _ = deck[index]
            aim = np.subtract(target, apex)
        else:
            apex, kelvin, _ = sea[index]
            aim = DOWN
        w = np.asarray(aim, float) / np.linalg.norm(aim)
        out.append({"apex": tuple(map(float, apex)), "aim": tuple(map(float, w)), "angle": angle, "gain": gain,
                    "kelvin": KELVIN.get(float(kelvin), float(kelvin)), "length": float((apex[2] - floor) / max(-w[2], 1e-3))})
    return out


def basis(aim):
    w = np.asarray(aim, float) / np.linalg.norm(aim)
    up = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(up, w)
    u /= np.linalg.norm(u)
    return u, np.cross(w, u), w


def inside(eye, spec: dict, margin: float = 1.15) -> bool:
    _, _, w = basis(spec["aim"])
    rel = np.asarray(eye, float) - np.asarray(spec["apex"])
    h = float(rel @ w)
    if h <= 0 or h >= spec["length"]:
        return False
    return float(np.linalg.norm(rel - h * w)) < h * math.tan(math.radians(spec["angle"])) * margin


def _ramp(e0: float, e1: float, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def layer(cam: dict, specs, t: float = 0.0, gain: float = 400.0, scale: int = 4, samples: int = 40,
          depth: np.ndarray | None = None) -> np.ndarray:
    from topside_frames import kelvin_rgb

    h, w = max(1, cam["h"] // scale), max(1, cam["w"] // scale)
    K, T = np.asarray(cam["K"], float), np.asarray(cam["T"], float)
    eye = T[:3, 3]
    acc = np.zeros((h, w, 3), np.float32)
    if depth is not None:
        d = np.asarray(depth, np.float32).reshape(depth.shape[0], depth.shape[1])
        d = np.where(np.isfinite(d) & (d > 0), d, np.inf)[: h * scale, : w * scale]
        low = d.reshape(h, scale, w, scale).min(axis=(1, 3))
    s = np.linspace(0.02, 1.0, samples)
    along = _ramp(0.0, 0.15, s) * (1 - s) ** 0.7 * (1 - _ramp(0.8, 1.0, s))
    for sp in specs:
        if inside(eye, sp):
            continue
        ax = np.asarray(sp["aim"], float)
        pts = np.asarray(sp["apex"], float) + (s * sp["length"])[:, None] * ax
        pc = (pts - eye) @ T[:3, :3]
        z = pc[:, 2]
        ok = z > 0.5
        if not ok.any():
            continue
        zs = np.maximum(z, 0.5)
        u = (K[0, 0] * pc[:, 0] / zs + K[0, 2]) / scale
        v = (K[1, 1] * pc[:, 1] / zs + K[1, 2]) / scale
        r = np.maximum(K[0, 0] * s * sp["length"] * math.tan(math.radians(sp["angle"])) / zs / scale, 0.7)
        step = np.hypot(np.gradient(u), np.gradient(v))
        dist = np.linalg.norm(pts - eye, axis=-1)
        motes = 0.85 + 0.15 * np.sin(s * 23.0 - 1.3 * t)
        amp = along * motes * _ramp(2.0, 10.0, z) * np.clip(step / r, 0.0, 1.0) * ok
        rgb = np.asarray(kelvin_rgb(sp["kelvin"]), np.float32) * gain * sp["gain"]
        for ui, vi, ri, ai, di in zip(u, v, r, amp, dist):
            if ai <= 1e-3:
                continue
            x0, x1 = int(max(0, ui - 2.5 * ri)), int(min(w, ui + 2.5 * ri + 1))
            y0, y1 = int(max(0, vi - 2.5 * ri)), int(min(h, vi + 2.5 * ri + 1))
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            g = np.exp(-((xx - ui) ** 2 + (yy - vi) ** 2) / (2 * (0.5 * ri) ** 2)) * ai
            if depth is not None:
                g = g * (low[y0:y1, x0:x1] > di)
            acc[y0:y1, x0:x1] += g[..., None] * rgb
    return acc


def add(rgb: np.ndarray, beams: np.ndarray) -> np.ndarray:
    from PIL import Image

    if not beams.any():
        return rgb
    h, w = rgb.shape[:2]
    up = np.stack([np.asarray(Image.fromarray(beams[..., c]).resize((w, h), Image.BILINEAR)) for c in range(3)], -1)
    return np.clip(rgb[..., :3].astype(np.float32) + up, 0, 255).astype(np.uint8)
