from __future__ import annotations

import math

import numpy as np

WAVES = (
    (140.0, 1.00, 18.0, 0.45),
    (86.0, 0.62, -12.0, 0.50),
    (52.0, 0.36, 41.0, 0.55),
    (31.0, 0.20, -33.0, 0.60),
    (19.0, 0.11, 72.0, 0.60),
    (12.0, 0.06, -64.0, 0.55),
)
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


def surface(rest: np.ndarray, distance: np.ndarray, time: float, height: float):
    fade = np.exp(-distance / FADE_DISTANCE_M)
    x, y, z = rest[:, 0].copy(), rest[:, 1].copy(), np.zeros(len(rest))
    slope_x, slope_y, stretch = np.zeros(len(rest)), np.zeros(len(rest)), np.zeros(len(rest))
    for wavelength, relative, heading, steepness in WAVES:
        amplitude = height * relative
        k = math.tau / wavelength
        omega = math.sqrt(GRAVITY * k)
        dx, dy = math.cos(math.radians(heading)), math.sin(math.radians(heading))
        phase = k * (rest[:, 0] * dx + rest[:, 1] * dy) - omega * time
        sin, cos = np.sin(phase), np.cos(phase)
        q = steepness / max(k * amplitude * len(WAVES), 1e-6) if amplitude > 0 else 0.0
        q = min(q, 1.0 / max(k * amplitude * len(WAVES), 1e-6)) * 0.8
        x += q * amplitude * dx * cos * fade
        y += q * amplitude * dy * cos * fade
        z += amplitude * sin * fade
        slope_x += dx * k * amplitude * cos * fade
        slope_y += dy * k * amplitude * cos * fade
        stretch += q * k * amplitude * sin * fade
    normals = np.stack((-slope_x, -slope_y, 1.0 - stretch), axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return np.stack((x, y, z), axis=1), normals
