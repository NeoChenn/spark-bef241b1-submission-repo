"""Gerstner ocean: Pierson-Moskowitz wave sets, displacement and a polar sea mesh. Pure numpy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81
EARTH_RADIUS = 6_371_000.0


@dataclass
class Waves:
    dir: np.ndarray
    k: np.ndarray
    omega: np.ndarray
    amp: np.ndarray
    phase: np.ndarray
    q: np.ndarray

    @property
    def wavelength(self) -> np.ndarray:
        return 2 * np.pi / self.k


def pierson_moskowitz(omega: np.ndarray, wind_speed: float) -> np.ndarray:
    alpha, beta = 8.1e-3, 0.74
    w0 = G / wind_speed
    return alpha * G**2 * omega**-5.0 * np.exp(-beta * (w0 / omega) ** 4)


def make_waves(
    wind_speed: float,
    wind_dir_deg: float,
    seed: int = 0,
    bands: int = 28,
    per_band: int = 4,
    min_wavelength: float = 0.1,
    spread: float = 0.5,
    choppiness: float = 0.8,
    swell: tuple[float, float, float] | None = None,
) -> Waves:
    """Sample a wind sea; ``swell`` adds (height m, wavelength m, dir deg) as three near-parallel components."""
    rng = np.random.default_rng(seed)
    w_peak = 0.877 * G / wind_speed
    w_max = np.sqrt(G * 2 * np.pi / min_wavelength)
    edges = np.geomspace(0.6 * w_peak, max(w_max, 0.7 * w_peak * 1.01), bands + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    energy = 2 * pierson_moskowitz(centers, wind_speed) * np.diff(edges)
    lo, hi = np.repeat(edges[:-1], per_band), np.repeat(edges[1:], per_band)
    omega = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    amp2 = np.repeat(energy / per_band, per_band)
    widen = np.clip(np.sqrt(omega / w_peak), 1.0, 3.0)
    theta = np.radians(wind_dir_deg) + rng.normal(0.0, 1.0, omega.size) * spread * widen
    if swell:
        h, length, sdir = swell
        sw = np.full(3, np.sqrt(2 * np.pi * G / length)) * np.array([0.98, 1.0, 1.03])
        omega = np.concatenate([omega, sw])
        amp2 = np.concatenate([amp2, np.full(3, (h / 4) ** 2 * 2 / 3)])
        theta = np.concatenate([theta, np.radians(sdir) + np.array([-0.08, 0.0, 0.1])])
    k = omega**2 / G
    amp = np.sqrt(amp2)
    q = choppiness / (k * amp * omega.size)
    return Waves(
        dir=np.stack([np.cos(theta), np.sin(theta)], 1).astype(np.float32),
        k=k.astype(np.float32),
        omega=omega.astype(np.float32),
        amp=amp.astype(np.float32),
        phase=rng.uniform(0, 2 * np.pi, omega.size).astype(np.float32),
        q=q.astype(np.float32),
    )


def significant_height(w: Waves) -> float:
    return float(4 * np.sqrt(np.sum(w.amp.astype(np.float64) ** 2) / 2))


def lod_weights(w: Waves, spacing: np.ndarray) -> np.ndarray:
    """Per-vertex, per-wave weight: fade waves shorter than ~4 vertex spacings."""
    s = np.clip((w.wavelength[None, :] / spacing[:, None] - 3.0) / 3.0, 0.0, 1.0)
    return (s * s * (3 - 2 * s)).astype(np.float32)


def displace(
    w: Waves,
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    fade: np.ndarray | None = None,
    spacing: np.ndarray | None = None,
) -> np.ndarray:
    theta = (np.outer(x, w.dir[:, 0] * w.k) + np.outer(y, w.dir[:, 1] * w.k)) - (w.omega * t - w.phase)
    a = np.broadcast_to(w.amp, theta.shape)
    if spacing is not None:
        a = a * lod_weights(w, spacing)
    c, s = np.cos(theta), np.sin(theta)
    qa = a * w.q
    out = np.stack([(qa * c) @ w.dir[:, 0], (qa * c) @ w.dir[:, 1], (a * s).sum(1)], 1).astype(np.float32)
    if fade is not None:
        out *= fade[:, None]
    return out


@dataclass
class PolarGrid:
    xy: np.ndarray
    counts: np.ndarray
    indices: np.ndarray
    ring_radius: np.ndarray
    spacing: np.ndarray


def ring_radii(segments: int, inner_radius: float, inner_step: float, outer_radius: float, aspect: float) -> np.ndarray:
    radii = [inner_step]
    growth = 2 * np.pi / segments * aspect
    while radii[-1] < outer_radius:
        r = radii[-1]
        radii.append(r + (inner_step if r < inner_radius else max(inner_step, r * growth)))
    radii[-1] = outer_radius
    return np.array(radii)


def polar_grid(
    segments: int = 512,
    inner_radius: float = 20.0,
    inner_step: float = 0.4,
    outer_radius: float = 40_000.0,
    aspect: float = 2.0,
) -> PolarGrid:
    radii = ring_radii(segments, inner_radius, inner_step, outer_radius, aspect)
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring_xy = np.stack([np.outer(radii, np.cos(a)), np.outer(radii, np.sin(a))], -1).reshape(-1, 2)
    xy = np.concatenate([[[0.0, 0.0]], ring_xy]).astype(np.float64)
    dr = np.diff(np.concatenate([[0.0], radii]))
    ring_spacing = np.maximum(dr, radii * 2 * np.pi / segments)
    spacing = np.concatenate([[inner_step], np.repeat(ring_spacing, segments)])

    j = np.arange(segments)
    jn = (j + 1) % segments
    fan = np.stack([np.zeros(segments, int), 1 + j, 1 + jn], 1).ravel()
    i = np.arange(len(radii) - 1)[:, None]
    v0 = 1 + i * segments + j
    v1 = 1 + i * segments + jn
    quads = np.stack([v0, v0 + segments, v1 + segments, v1], -1).reshape(-1)
    counts = np.concatenate([np.full(segments, 3), np.full(quads.size // 4, 4)])
    return PolarGrid(xy, counts.astype(np.int32), np.concatenate([fan, quads]).astype(np.int32), radii, spacing)


def jacobian(w: Waves, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """Determinant of the horizontal Gerstner map; below ~0.5 the surface is folding into a breaking crest."""
    theta = (np.outer(x, w.dir[:, 0] * w.k) + np.outer(y, w.dir[:, 1] * w.k)) - (w.omega * t - w.phase)
    qs = (w.q * w.k * w.amp) * np.sin(theta)
    jxx = qs @ (w.dir[:, 0] ** 2)
    jyy = qs @ (w.dir[:, 1] ** 2)
    jxy = qs @ (w.dir[:, 0] * w.dir[:, 1])
    return (1 - jxx) * (1 - jyy) - jxy**2


def whitecap_fraction(wind_speed: float) -> float:
    """Monahan & O'Muircheartaigh (1980) whitecap coverage."""
    return 3.84e-6 * wind_speed**3.41


def foam_threshold(w: Waves, coverage: float, seed: int = 0, samples: int = 20_000) -> float:
    """Jacobian level below which the given fraction of the sea surface is covered."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-500, 500, (samples, 2))
    j = jacobian(w, xy[:, 0], xy[:, 1], float(rng.uniform(0, 100)))
    return float(np.quantile(j, coverage))
