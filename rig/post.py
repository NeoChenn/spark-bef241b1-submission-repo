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
    def __init__(self, seed: int = 0, rate: float = 3.0, max_drops: int = 24, life: float = 9.0):
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
        blur = np.asarray(Image.fromarray(img[..., :3]).filter(ImageFilter.GaussianBlur(m * 0.02)), np.float32)
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
            a = (np.clip(1 - q, 0, 1) ** 0.7 * 0.6 * fade)[..., None]
            sx = np.clip(cx - u * rr * 0.8, 0, w - 1).astype(np.intp)
            sy = np.clip(cy - v * rr * 0.8, 0, h - 1).astype(np.intp)
            lens = blur[sy, sx] * (1.04 - 0.1 * q[..., None]) + 25.0 * np.exp(-((u + 0.3) ** 2 + (v + 0.35) ** 2) * 12)[..., None]
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


def height_fog(img: np.ndarray, depth: np.ndarray, cam: dict, density: float, falloff: float, colour,
               max_dist: float = 3000.0) -> np.ndarray:
    if density <= 0:
        return img
    h, w = depth.shape
    K, T = np.asarray(cam["K"], np.float32), np.asarray(cam["T"], np.float32)
    sy, sx = img.shape[0] / h, img.shape[1] / w
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    rays = np.stack([((u + 0.5) * sx - K[0, 2]) / K[0, 0], ((v + 0.5) * sy - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    dz = (rays @ T[:3, :3].T)[..., 2] / np.linalg.norm(rays, axis=-1)
    d = np.where(np.isfinite(depth) & (depth > 0), np.minimum(depth, max_dist), max_dist)
    z0 = max(float(T[2, 3]), 0.0)
    k = dz * d / falloff
    ratio = np.where(np.abs(k) > 1e-4, (1 - np.exp(-np.clip(k, -30, 30))) / np.where(np.abs(k) > 1e-4, k, 1), 1.0)
    tau = density * d * np.exp(-z0 / falloff) * ratio
    fog = 1 - np.exp(-np.clip(tau, 0, 50))
    if fog.shape != img.shape[:2]:
        fog = np.asarray(Image.fromarray(fog.astype(np.float32)).resize((img.shape[1], img.shape[0]), Image.BILINEAR))
    f = img[..., :3].astype(np.float32)
    out = f + (np.asarray(colour, np.float32) - f) * fog[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def god_rays(img: np.ndarray, source, strength: float, samples: int = 24, decay: float = 0.94, threshold: float = 170.0,
             scale: int = 4, reach: float = 0.75) -> np.ndarray:
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    small = np.asarray(Image.fromarray(img[..., :3]).resize((w // scale, h // scale), Image.BILINEAR), np.float32)
    bright = np.clip(small - threshold, 0, None) * (255.0 / (255.0 - threshold))
    sh, sw = bright.shape[:2]
    cx, cy = source[0] / scale, source[1] / scale
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    acc = np.zeros_like(bright)
    wgt = 1.0
    for i in range(samples):
        s = 1.0 - reach * i / samples
        x = np.clip(np.rint(cx + (xx - cx) * s), 0, sw - 1).astype(np.intp)
        y = np.clip(np.rint(cy + (yy - cy) * s), 0, sh - 1).astype(np.intp)
        acc += bright[y, x] * wgt
        wgt *= decay
    acc *= strength * 3.0 / samples
    up = np.asarray(Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), np.float32)
    out = img[..., :3].astype(np.float32) + up
    return np.clip(out, 0, 255).astype(np.uint8)
