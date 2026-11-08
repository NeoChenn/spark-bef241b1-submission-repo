from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def heat_shimmer(img: np.ndarray, source, radius: float, t: float, strength: float, height: float = 3.0) -> np.ndarray:
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    cx, cy = source
    x0, x1 = int(max(0, cx - radius)), int(min(w, cx + radius))
    y0, y1 = int(max(0, cy - radius * height)), int(min(h, cy + radius * 0.3))
    if x0 >= x1 or y0 >= y1:
        return img
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    fall = np.clip(1 - np.abs(xx - cx) / radius, 0, 1) * np.clip((cy + radius * 0.3 - yy) / (radius * 0.6), 0, 1)
    fall *= np.clip((yy - (cy - radius * height)) / (radius * height * 0.5), 0, 1)
    dx = strength * fall * (np.sin(yy * 0.21 + t * 11.0) + 0.5 * np.sin(yy * 0.53 - t * 17.0 + xx * 0.11))
    dy = strength * 0.5 * fall * np.sin(xx * 0.17 + t * 9.0)
    sx = np.clip(np.rint(xx + dx), 0, w - 1).astype(np.intp)
    sy = np.clip(np.rint(yy + dy), 0, h - 1).astype(np.intp)
    out = img.copy()
    out[y0:y1, x0:x1] = img[sy, sx]
    return out


class LensDrops:
    def __init__(self, seed: int = 0, rate: float = 6.0, max_drops: int = 48, life: float = 9.0):
        self.rng = np.random.default_rng(seed)
        self.rate, self.max_drops, self.life = rate, max_drops, life
        self.drops = np.zeros((0, 4))

    def step(self, dt: float, rain: float) -> None:
        d = self.drops
        if len(d):
            d[:, 3] += dt
            d[:, 1] += dt * 0.015 * (d[:, 2] > 0.035)
            self.drops = d[(d[:, 3] < self.life) & (d[:, 1] < 1.1)]
        k = int(self.rng.poisson(max(0.0, self.rate * rain * dt)))
        if k:
            new = np.column_stack([self.rng.uniform(0, 1, k), self.rng.uniform(-0.05, 1, k),
                                   self.rng.uniform(0.012, 0.05, k), np.zeros(k)])
            self.drops = np.concatenate([self.drops, new])[-self.max_drops:]

    def apply(self, img: np.ndarray) -> np.ndarray:
        if not len(self.drops):
            return img
        h, w = img.shape[:2]
        m = min(h, w)
        f = img[..., :3].astype(np.float32)
        blur = np.asarray(Image.fromarray(img[..., :3]).filter(ImageFilter.GaussianBlur(m * 0.012)), np.float32)
        out = f.copy()
        for x, y, r, age in self.drops:
            rr = max(2.0, r * m)
            cx, cy = x * w, y * h
            ya, yb, xa, xb = int(max(0, cy - rr)), int(min(h, cy + rr + 1)), int(max(0, cx - rr)), int(min(w, cx + rr + 1))
            if ya >= yb or xa >= xb:
                continue
            yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
            u, v = (xx - cx) / rr, (yy - cy) / rr
            q = u * u + v * v
            fade = min(1.0, (self.life - age) / 2.0)
            a = (np.clip((1 - q) * 3.0, 0, 1) * 0.9 * fade)[..., None]
            sx = np.clip(cx - u * rr * 0.8, 0, w - 1).astype(np.intp)
            sy = np.clip(cy - v * rr * 0.8, 0, h - 1).astype(np.intp)
            lens = blur[sy, sx] * (1.08 - 0.25 * q[..., None]) + 6.0 * np.exp(-((u + 0.35) ** 2 + (v + 0.4) ** 2) * 18)[..., None] * 18
            out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - a) + lens * a
        res = np.clip(out, 0, 255).astype(np.uint8)
        return np.concatenate([res, img[..., 3:]], -1) if img.shape[-1] == 4 else res


def grade(img: np.ndarray, lift=(0.0, 0.0, 0.0), gain=(1.0, 1.0, 1.0), vignette: float = 0.0) -> np.ndarray:
    if not any(lift) and gain == (1.0, 1.0, 1.0) and not vignette:
        return img
    f = img[..., :3].astype(np.float32) / 255.0
    f = np.asarray(lift, np.float32) + f * (1 - np.asarray(lift, np.float32))
    f *= np.asarray(gain, np.float32)
    if vignette:
        h, w = f.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 2
        f *= (1 - vignette * r ** 1.5)[..., None]
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)
