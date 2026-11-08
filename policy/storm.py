from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, uniform_filter1d

TWO_PI = 2 * math.pi
WEATHER = {"calm": (0.0, 0.0, 0.0), "wind": (0.8, 1.0, 0.0), "rain": (0.7, 0.25, 1.0), "fog": (0.6, 0.0, 0.8),
           "storm": (1.0, 1.0, 1.0)}
FOG = np.array([138.0, 144.0, 152.0], np.float32)


def _u(shape, lo, hi, generator, device):
    return lo + (hi - lo) * torch.rand(shape, generator=generator, device=device)


def bump(u: torch.Tensor) -> torch.Tensor:
    inside = (u >= 0) & (u <= 1)
    return torch.where(inside, 0.5 * (1 - torch.cos(TWO_PI * u)), torch.zeros_like(u))


def sample_gusts(n: int, level, horizon_s: float, k: Optional[int] = None, generator=None, device="cpu",
                 peak=(10.0, 30.0), mean=(0.0, 4.0), dur=(1.5, 4.0), every_s: float = 4.0) -> dict:
    k = k or max(1, int(math.ceil(horizon_s / every_s)))
    lv = torch.as_tensor(level, dtype=torch.float32, device=device).expand(n).reshape(n, 1)
    d = _u((n, k), *dur, generator, device)
    centre = _u((n, k), 0.0, horizon_s, generator, device)
    az = _u((n,), 0.0, TWO_PI, generator, device)
    el = _u((n,), -0.25, 0.25, generator, device)
    direction = torch.stack([torch.cos(az) * torch.cos(el), torch.sin(az) * torch.cos(el), torch.sin(el)], -1)
    return {"t0": centre - d / 2, "dur": d, "peak": _u((n, k), *peak, generator, device) * lv,
            "mean": _u((n,), *mean, generator, device) * lv[:, 0], "dir": direction}


def gust_force(g: dict, t: torch.Tensor) -> torch.Tensor:
    u = (t.view(-1, 1) - g["t0"]) / g["dur"]
    return g["mean"] + (g["peak"] * bump(u)).sum(-1)


def sample_deck(n: int, level, generator=None, device="cpu", amp=(0.02, 0.05), freq=(0.1, 0.3)) -> dict:
    lv = torch.as_tensor(level, dtype=torch.float32, device=device).expand(n).reshape(n, 1)
    return {"amp": _u((n, 2), *amp, generator, device) * lv, "freq": _u((n, 2), *freq, generator, device),
            "phase": _u((n, 2), 0.0, TWO_PI, generator, device)}


def deck_offset(d: dict, t: torch.Tensor) -> torch.Tensor:
    return d["amp"] * torch.sin(TWO_PI * d["freq"] * t.view(-1, 1) + d["phase"])


def wet_factor(level: float) -> float:
    return 1.0 - 0.35 * float(np.clip(level, 0.0, 1.0))


class Lens:
    def __init__(self, seed: int = 0, rate: float = 1.5, max_drops: int = 60):
        self.rng = np.random.default_rng(seed)
        self.rate, self.max_drops = rate, max_drops
        self.drops = np.zeros((0, 3))

    def rain(self, dt: float, intensity: float) -> None:
        k = int(self.rng.poisson(max(0.0, self.rate * intensity * dt)))
        if not k:
            return
        new = np.column_stack([self.rng.uniform(0, 1, k), self.rng.uniform(0, 1, k), self.rng.uniform(0.05, 0.13, k)])
        self.drops = np.concatenate([self.drops, new])[-self.max_drops:]

    def wipe(self) -> None:
        self.drops = self.drops[: len(self.drops) // 10]

    def dirt(self) -> float:
        return float(min(1.0, (math.pi * self.drops[:, 2] ** 2).sum() * 0.75)) if len(self.drops) else 0.0

    def _mask(self, h: int, w: int) -> np.ndarray:
        m = min(h, w)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = np.zeros((h, w), np.float32)
        for x, y, r in self.drops:
            rr = r * m
            d = np.hypot(xx - x * w, yy - y * h) / rr
            a = np.maximum(a, np.clip((1.0 - d) * 1.6, 0, 1))
        return a

    def apply(self, img: np.ndarray, visibility_loss: float = 0.0) -> np.ndarray:
        v = float(np.clip(visibility_loss, 0.0, 1.0))
        if v <= 0 and not len(self.drops):
            return img
        f = img[..., :3].astype(np.float32)
        h, w = f.shape[:2]
        if v > 0:
            f = gaussian_filter(f, (2.2 * v, 2.2 * v, 0))
            f = f * (1 - 0.55 * v) + FOG * 0.55 * v
        if len(self.drops):
            s = 0.07 * min(h, w)
            smeared = gaussian_filter(f, (s, s, 0)) * 0.85 + 30.0
            a = (self._mask(h, w) * 0.95)[..., None]
            f = f * (1 - a) + smeared * a
            veil = 0.5 * self.dirt()
            f = f * (1 - veil) + FOG * veil
        out = np.clip(f, 0, 255).astype(np.uint8)
        if img.shape[-1] == 4:
            out = np.concatenate([out, img[..., 3:]], -1)
        return out


def lens_quality(img: np.ndarray) -> float:
    g = np.asarray(img, np.float32)[..., :3].mean(-1)
    b = 0.0
    for axis in (0, 1):
        d = np.abs(np.diff(g, axis=axis))
        db = np.abs(np.diff(uniform_filter1d(g, 9, axis=axis), axis=axis))
        s = d.sum()
        b = max(b, float(np.maximum(0.0, d - db).sum() / s) if s > 0 else 0.0)
    lo, hi = np.percentile(g, (2, 98))
    contrast = min(1.0, (hi - lo) / 128.0)
    return float(b * contrast)


class Weather:
    def __init__(self, seed: int = 0, horizon_s: float = 900.0, lull_every_s: float = 22.0, lull_s: float = 9.0,
                 lull_depth: float = 0.85):
        self.seed, self.horizon = seed, horizon_s
        self.lull_every, self.lull_s, self.lull_depth = lull_every_s, lull_s, lull_depth
        self.level, self.t_on = 0.0, 0.0
        self.gust_scale, self.rain_scale, self.kind = 1.0, 1.0, "storm"
        self._sample()

    def _sample(self):
        gen = torch.Generator().manual_seed(self.seed)
        self.gusts = sample_gusts(1, 1.0, self.horizon, generator=gen, peak=(10.0, 26.0), mean=(1.5, 2.5),
                                  dur=(1.5, 3.5), every_s=8.0)
        self.deck = sample_deck(1, 1.0, generator=gen, amp=(0.02, 0.04), freq=(0.12, 0.25))
        rng = np.random.default_rng(self.seed)
        n = int(self.horizon / self.lull_every) + 1
        self.lulls = np.arange(n) * self.lull_every + rng.uniform(8.0, self.lull_every - self.lull_s, n)

    def set_level(self, level: float, t: float) -> None:
        self.level, self.t_on = float(np.clip(level, 0.0, 1.0)), float(t)

    def set_weather(self, level, t: float, wind_dir: Optional[float] = None, gust: Optional[float] = None) -> float:
        if isinstance(level, str):
            self.kind = level
            lv, self.gust_scale, self.rain_scale = WEATHER[level]
        else:
            self.kind, lv, self.gust_scale, self.rain_scale = "storm", float(level), 1.0, 1.0
        if gust is not None:
            self.gust_scale = float(gust) / 18.0
        if wind_dir is not None:
            a = math.radians(float(wind_dir))
            self.gusts["dir"] = torch.tensor([[math.cos(a), math.sin(a), 0.0]])
        self.set_level(lv, t)
        return self.level

    def _envelope(self, tau: float) -> float:
        u = (tau - self.lulls) / self.lull_s
        dip = np.where((u >= 0) & (u <= 1), np.minimum(1.0, 2.0 * np.sin(math.pi * np.clip(u, 0, 1))), 0.0).max()
        return float(1.0 - self.lull_depth * dip)

    def at(self, t: float) -> dict:
        if self.level <= 0:
            return {"level": 0.0, "gust_n": 0.0, "rain": 0.0, "deck_m": (0.0, 0.0), "gust_dir": (1.0, 0.0, 0.0)}
        tau = (float(t) - self.t_on) % self.horizon
        env = self._envelope(tau)
        tt = torch.tensor([tau])
        gust = float(gust_force(self.gusts, tt)[0]) * self.level * env * self.gust_scale
        deck = deck_offset(self.deck, tt)[0] * self.level
        return {"level": self.level, "gust_n": gust, "rain": min(1.0, self.level * env * self.rain_scale),
                "deck_m": (float(deck[0]), float(deck[1])), "gust_dir": tuple(float(v) for v in self.gusts["dir"][0])}

    def next_lull(self, t: float, threshold: float = 0.35, within_s: float = 120.0) -> Optional[float]:
        for s in np.arange(0.0, within_s, 0.25):
            if self.at(t + s)["rain"] < threshold:
                return float(t + s)
        return None


def scripted_gusts(times, peak: float, dur: float, mean: float = 0.0, direction=(0.0, -1.0, 0.0)) -> dict:
    t = torch.tensor([list(map(float, times))])
    d = torch.tensor([list(map(float, direction))])
    return {"t0": t - dur / 2, "dur": torch.full_like(t, float(dur)), "peak": torch.full_like(t, float(peak)),
            "mean": torch.tensor([float(mean)]), "dir": d / d.norm(dim=-1, keepdim=True)}


class GustGate:
    def __init__(self, limit: float = 12.0, resume: float = 6.0):
        self.limit, self.resume, self.paused = limit, resume, False

    def update(self, gust: float) -> bool:
        self.paused = gust > self.limit if not self.paused else gust >= self.resume
        return self.paused


def storm_grade(rgb: np.ndarray, level: float, t: float, seed: int = 0) -> np.ndarray:
    if level <= 0:
        return rgb
    from PIL import Image, ImageDraw

    f = rgb[..., :3].astype(np.float32)
    grey = f.mean(-1, keepdims=True)
    f = f + (grey - f) * 0.55 * level
    f = f * (1 - 0.45 * level) + np.array([14.0, 22.0, 34.0]) * 0.45 * level
    im = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).convert("RGBA")
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    rng = np.random.default_rng(seed * 100003 + int(round(t * 1000)))
    h, w = rgb.shape[:2]
    for _ in range(int(500 * level)):
        x, y = rng.uniform(-0.2 * w, w), rng.uniform(-0.1 * h, h)
        n = rng.uniform(0.02, 0.05) * h
        d.line([(x, y), (x + 0.35 * n, y + n)], fill=(205, 215, 230, int(rng.uniform(40, 110))), width=1)
    return np.asarray(Image.alpha_composite(im, over).convert("RGB"))
