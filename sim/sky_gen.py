"""Procedural lat-long sky: elevation gradient, soft sun glow and blurry cloud masses. Pure numpy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RGB = tuple[float, float, float]


@dataclass(frozen=True)
class SkySpec:
    horizon: RGB
    mid: RGB
    zenith: RGB
    mid_height: float = 0.35
    horizon_hold: float = 1.0
    sun_azimuth: float = 0.0
    sun_elevation: float = 5.0
    glow_color: RGB = (0.0, 0.0, 0.0)
    glow_width_deg: float = 25.0
    disc: float = 0.0
    disc_deg: float = 1.2
    cloud_color: RGB = (0.0, 0.0, 0.0)
    cloud_amount: float = 0.0
    clouds: int = 18
    seed: int = 3
    width: int = 1024


def _lerp(a, b, t):
    return np.asarray(a)[None, None] * (1 - t[..., None]) + np.asarray(b)[None, None] * t[..., None]


def directions(width: int) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth and elevation (radians) of every pixel; azimuth follows sim.ocean_presets.hdri_azimuth."""
    h = width // 2
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(h) + 0.5) / h
    az = np.radians(360.0 * (1.0 - u))
    el = np.radians(90.0 - 180.0 * v)
    return np.broadcast_to(az[None, :], (h, width)), np.broadcast_to(el[:, None], (h, width))


def angle_between(az: np.ndarray, el: np.ndarray, az0: float, el0: float) -> np.ndarray:
    c = np.sin(el) * np.sin(el0) + np.cos(el) * np.cos(el0) * np.cos(az - az0)
    return np.arccos(np.clip(c, -1.0, 1.0))


def cloud_density(az: np.ndarray, el: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = np.zeros(az.shape)
    for _ in range(count):
        a0 = rng.uniform(0, 2 * np.pi)
        e0 = np.radians(rng.uniform(4, 35))
        sx = np.radians(rng.uniform(18, 45))
        sy = np.radians(rng.uniform(3, 9))
        da = np.angle(np.exp(1j * (az - a0))) * np.cos(el)
        d += rng.uniform(0.5, 1.0) * np.exp(-((da / sx) ** 2 + ((el - e0) / sy) ** 2))
    return np.clip(d, 0, None)


def render_sky(spec: SkySpec) -> np.ndarray:
    az, el = directions(spec.width)
    up = np.clip(el / (np.pi / 2), 0, 1)
    m = spec.mid_height
    low = np.clip(up / m, 0, 1) ** spec.horizon_hold
    high = np.clip((up - m) / (1 - m), 0, 1)
    sky = np.where(
        (up < m)[..., None],
        _lerp(spec.horizon, spec.mid, low * low * (3 - 2 * low)),
        _lerp(spec.mid, spec.zenith, high * high * (3 - 2 * high)),
    )
    sun = angle_between(az, el, np.radians(spec.sun_azimuth), np.radians(spec.sun_elevation))
    glow = np.exp(-sun / np.radians(spec.glow_width_deg)) * np.exp(-np.clip(el, 0, None) / 0.25)
    disc = spec.disc * np.clip((np.radians(spec.disc_deg) - sun) / np.radians(spec.disc_deg * 0.4), 0, 1)
    sky = sky + np.asarray(spec.glow_color) * glow[..., None] + disc[..., None] * np.asarray([1.0, 0.85, 0.7])
    if spec.cloud_amount > 0:
        c = np.clip(cloud_density(az, el, spec.clouds, spec.seed), 0, 1.5) / 1.5
        c = spec.cloud_amount * c * c * (3 - 2 * c) * (el > 0)
        lit = np.asarray(spec.cloud_color)[None, None] * (1 + 2.0 * glow[..., None])
        sky = sky * (1 - c[..., None]) + lit * c[..., None]
    below = el < 0
    sky[below] = np.asarray(spec.horizon) * 0.7
    return sky.astype(np.float32)
