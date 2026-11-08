from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

W, H = 320, 256
HFOV_DEG = 24.0
FOCAL_PX = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))


@dataclass
class FlareTipState:
    nozzle_temps_c: list[float]
    nozzle_pitch_m: float = 0.45
    sky_c: float = 8.0
    flame_c: float = 820.0

    @classmethod
    def from_state(cls, state: dict) -> "FlareTipState":
        f = state.get("flare") or {}
        temps = f.get("nozzle_temps_c")
        if temps is None:
            temps = [560.0] * 8
        return cls([float(t) for t in temps], float(f.get("nozzle_pitch_m", 0.45)),
                   float(f.get("sky_c", 8.0)), float(f.get("flame_c", 820.0)))


def nozzle_pixels(n: int, pitch_m: float, distance_m: float) -> list[tuple[float, float]]:
    s = FOCAL_PX / distance_m
    cx, cy = W / 2, H * 0.62
    return [(cx + (i - (n - 1) / 2) * pitch_m * s, cy) for i in range(n)]


def render(tip: FlareTipState, distance_m: float, wind_toward_cam_x: float = 0.0, seed: int = 0,
           pointing_px: Optional[tuple[float, float]] = None, netd_c: float = 1.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = FOCAL_PX / distance_m
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    off = pointing_px if pointing_px is not None else tuple(rng.uniform(-4, 4, 2))
    img = tip.sky_c + 6.0 * (yy / H) + rng.normal(0, 0.3, (H, W))
    n = len(tip.nozzle_temps_c)
    pts = [(x + off[0], y + off[1]) for x, y in nozzle_pixels(n, tip.nozzle_pitch_m, distance_m)]
    cy = pts[0][1]
    half = (n / 2 + 0.6) * tip.nozzle_pitch_m * s
    x0, x1 = pts[0][0] - 0.5 * tip.nozzle_pitch_m * s - 4, pts[-1][0] + 0.5 * tip.nozzle_pitch_m * s + 4
    body = (xx >= x0) & (xx <= x1) & (yy >= cy + 2) & (yy <= cy + 0.6 * s)
    img[body] = 300.0 + rng.normal(0, 3, body.sum())
    boom = (np.abs(xx - (x0 + x1) / 2 - (yy - cy) * 0.25) <= 0.35 * s) & (yy > cy + 0.6 * s)
    img[boom] = np.maximum(img[boom], 90.0 + 140.0 * np.exp(-(yy[boom] - cy) / (2.5 * s)))
    top = cy - 0.5 * s
    up = np.clip(top - yy, 0, None)
    axis = (x0 + x1) / 2 + wind_toward_cam_x * up * 0.6
    width = half * (1.0 + up / (6 * s))
    plume = np.exp(-((xx - axis) / np.maximum(width, 1)) ** 2) * (up > 0) * np.exp(-up / (10 * s))
    img = np.maximum(img, tip.sky_c + (tip.flame_c - tip.sky_c) * plume * 0.9)
    sigma = max(0.12 * s, 1.2)
    for (px, py), t in zip(pts, tip.nozzle_temps_c):
        spot = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2))
        img = img * (1 - spot) + t * spot
    return (img + rng.normal(0, netd_c, img.shape)).astype(np.float32)


@dataclass
class NozzleReport:
    temps_c: list[float]
    median_c: float
    mad_c: float
    anomalies: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {"temps_c": [round(t, 1) for t in self.temps_c], "median_c": round(self.median_c, 1),
                "mad_c": round(self.mad_c, 1), "anomalies": self.anomalies}


def detect(frame: np.ndarray, n: int, pitch_m: float, distance_m: float, k: float = 5.0,
           min_delta_c: float = 60.0, search_px: int = 8) -> NozzleReport:
    expected = np.array(nozzle_pixels(n, pitch_m, distance_m))
    spacing = pitch_m * FOCAL_PX / distance_m
    search_px = int(min(search_px, max(1, math.floor(0.45 * spacing))))
    best, shift = -np.inf, (0, 0)
    for dy in range(-search_px, search_px + 1):
        for dx in range(-search_px, search_px + 1):
            xs = np.clip(np.round(expected[:, 0] + dx).astype(int), 0, W - 1)
            ys = np.clip(np.round(expected[:, 1] + dy).astype(int), 0, H - 1)
            score = float(np.median(frame[ys, xs]))
            if score > best:
                best, shift = score, (dx, dy)
    temps = []
    for x, y in expected:
        ix, iy = int(round(x + shift[0])), int(round(y + shift[1]))
        block = frame[max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2]
        temps.append(float(np.median(block)))
    t = np.array(temps)
    med = float(np.median(t))
    mad = float(np.median(np.abs(t - med))) * 1.4826 + 1e-6
    anomalies = []
    for i, v in enumerate(t):
        d = float(v - med)
        z = d / mad
        if abs(z) > k and abs(d) > min_delta_c:
            anomalies.append({"nozzle": i + 1, "temp_c": round(float(v), 1), "delta_c": round(d, 1),
                              "z": round(z, 1), "kind": "hot" if d > 0 else "cold"})
    return NozzleReport([float(v) for v in t], med, mad, anomalies)


def false_colour(frame: np.ndarray, n: int, pitch_m: float, distance_m: float, scale: int = 3) -> np.ndarray:
    from PIL import Image, ImageDraw
    lo, hi = np.percentile(frame, 1), np.percentile(frame, 99.8)
    v = np.clip((frame - lo) / max(hi - lo, 1e-6), 0, 1)
    stops = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], np.float32)
    pos = v * (len(stops) - 1)
    i = np.clip(pos.astype(int), 0, len(stops) - 2)
    f = (pos - i)[..., None]
    rgb = (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)
    im = Image.fromarray(rgb).resize((W * scale, H * scale), Image.NEAREST)
    draw = ImageDraw.Draw(im)
    for j, (x, y) in enumerate(nozzle_pixels(n, pitch_m, distance_m)):
        draw.text((x * scale - 3, (y + 14) * scale), str(j + 1), fill=(255, 255, 255))
    return np.asarray(im)
