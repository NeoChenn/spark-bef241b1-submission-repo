from __future__ import annotations

import math

import numpy as np

SWELL_COUNT = 14


def swell(count: int = SWELL_COUNT, seed: int = 11, longest: float = 150.0, shortest: float = 9.0,
          heading: float = 18.0) -> tuple:
    rng = np.random.default_rng(seed)
    lengths = longest * (shortest / longest) ** (np.arange(count) / (count - 1)) * rng.uniform(0.9, 1.1, count)
    rel = (lengths / longest) ** 0.85 * rng.uniform(0.8, 1.2, count)
    rel *= math.sqrt(1.56 / np.sum(rel**2))
    spread = 22.0 + 30.0 * (1.0 - lengths / longest)
    headings = heading + rng.normal(0.0, 1.0, count) * spread
    steep = rng.uniform(0.45, 0.6, count)
    phases = rng.uniform(0.0, math.tau, count)
    return tuple((float(a), float(b), float(c), float(d), float(e)) for a, b, c, d, e in zip(lengths, rel, headings, steep, phases))


WAVES = swell()
GRAVITY = 9.81
FADE_DISTANCE_M = 1600.0


def make_grid(rings: int = 190, segments: int = 256, inner: float = 6.0, outer: float = 24000.0):
    radii = inner * (outer / inner) ** (np.arange(rings) / (rings - 1))
    angles = np.arange(segments) / segments * math.tau
    grid_r, grid_a = np.meshgrid(radii, angles, indexing="ij")
    points = np.stack((grid_r * np.cos(grid_a), grid_r * np.sin(grid_a)), axis=-1).reshape(-1, 2)
    points = np.concatenate((np.zeros((1, 2)), points))

    ring, step = np.meshgrid(np.arange(rings - 1), np.arange(segments), indexing="ij")
    a = 1 + ring * segments + step
    b = 1 + ring * segments + (step + 1) % segments
    quads = np.stack((a, a + segments, b + segments, b), axis=-1).reshape(-1)
    fan = np.stack(
        (np.zeros(segments, np.int64), 1 + np.arange(segments), 1 + (np.arange(segments) + 1) % segments), axis=-1
    ).reshape(-1)
    indices = np.concatenate((fan, quads)).astype(np.int32)
    counts = np.concatenate((np.full(segments, 3), np.full((rings - 1) * segments, 4))).astype(np.int32)
    return points, np.linalg.norm(points, axis=1), counts, indices


def wave_table(height: float) -> np.ndarray:
    rows = []
    for wavelength, relative, heading, steepness, phase in WAVES:
        k = math.tau / wavelength
        rows.append((math.cos(math.radians(heading)) * k, math.sin(math.radians(heading)) * k, k,
                     height * relative, math.sqrt(GRAVITY * k), phase, 0.8 * min(steepness, 1.0) / (k * len(WAVES))))
    return np.array(rows, np.float32)


def surface(rest: np.ndarray, distance: np.ndarray, time: float, height: float):
    w = wave_table(height)
    fade = np.exp(-distance / FADE_DISTANCE_M).astype(np.float32)[:, None]
    theta = rest.astype(np.float32) @ w[:, :2].T - (w[:, 4] * time - w[:, 5])
    sin, cos = np.sin(theta), np.cos(theta)
    unit = w[:, :2] / w[:, 2:3]
    hc = cos * (w[:, 6] * fade)
    out = np.empty((len(rest), 3), np.float32)
    out[:, :2] = rest + hc @ unit
    out[:, 2] = (sin * fade) @ w[:, 3]
    ac = cos * fade
    sx, sy = ac @ (w[:, 0] * w[:, 3]), ac @ (w[:, 1] * w[:, 3])
    qs = sin * fade * (w[:, 6] * w[:, 2])
    jxx, jyy, jxy = qs @ unit[:, 0] ** 2, qs @ unit[:, 1] ** 2, qs @ (unit[:, 0] * unit[:, 1])
    normals = np.stack((-jxy * sy - sx * (1 - jyy), -jxy * sx - sy * (1 - jxx), (1 - jxx) * (1 - jyy) - jxy**2), axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return out, normals
