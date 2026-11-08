from __future__ import annotations

import numpy as np

from photoreal_masks import smoothstep, tileable_noise


def drip_mask(width: int, height: int, seed: int = 0, count: int = 24) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((height, width))
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    for _ in range(count):
        x = rng.uniform(0.1, 0.9) * width
        w = rng.uniform(0.01, 0.06) * width
        start = rng.uniform(0.0, 0.12) * height
        length = rng.uniform(0.35, 1.0) * (height - start)
        along = np.clip((rows - start) / length, 0.0, 1.0)
        wobble = x + 0.015 * width * np.sin(rows / height * rng.uniform(6, 14) + rng.uniform(0, 6.3))
        core = np.exp(-(((cols - wobble) / (w * (1.0 - 0.6 * along))) ** 2))
        fade = (rows >= start) * (1.0 - along) ** 1.3 * rng.uniform(0.4, 1.0)
        out = np.maximum(out, core * fade)
    top = np.exp(-(((rows / height) - 0.04) / 0.05) ** 2) * (0.5 + 0.5 * np.sin(cols / width * 9.0)) * 0.6
    out = np.maximum(out, top)
    edge = smoothstep(0.0, 0.12, cols / width) * smoothstep(0.0, 0.12, 1 - cols / width)
    return (np.clip(out * edge, 0, 1) * 255).round().astype(np.uint8)


def _stamp(field, cx, cy, r, kernel):
    size = field.shape[0]
    n = kernel.shape[0] // 2
    ys = (np.arange(-n, n + 1) + int(cy)) % size
    xs = (np.arange(-n, n + 1) + int(cx)) % size
    block = field[np.ix_(ys, xs)]
    field[np.ix_(ys, xs)] = np.maximum(block, kernel)


def marine_growth(size: int = 2048, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    big = tileable_noise(size, 4.0, seed)
    mid = tileable_noise(size, 18.0, seed + 1)
    fine = tileable_noise(size, 90.0, seed + 2, octaves=2)
    olive = np.array((0.075, 0.085, 0.03))
    brown = np.array((0.11, 0.07, 0.035))
    weed = np.array((0.09, 0.14, 0.04))
    t = smoothstep(0.3, 0.7, big)[..., None]
    algae = olive * (1 - t) + brown * t
    algae = algae * (1 - 0.6 * smoothstep(0.55, 0.8, mid)[..., None]) + weed * 0.6 * smoothstep(0.55, 0.8, mid)[..., None]
    strands = sum(np.roll(fine, k, axis=0) for k in range(0, 48, 3)) / 16.0
    strands = (strands - strands.min()) / max(np.ptp(strands), 1e-9)
    algae = algae * (0.55 + 0.9 * strands[..., None]) * (0.8 + 0.4 * fine[..., None])
    height = 0.25 * mid + 0.1 * fine + 0.25 * strands
    shell = np.zeros((size, size))
    density = 0.75 * smoothstep(0.5, 0.85, tileable_noise(size, 6.0, seed + 3))
    count = int(size * size / 260)
    for _ in range(count):
        cx, cy = rng.uniform(0, size, 2)
        if rng.uniform() > density[int(cy) % size, int(cx) % size]:
            continue
        r = rng.uniform(0.003, 0.007) * size
        n = int(np.ceil(r)) + 1
        g = np.arange(-n, n + 1)
        d = np.hypot(*np.meshgrid(g, g)) / r
        cone = np.sqrt(np.clip(1 - d * d, 0, 1)) - 0.55 * np.exp(-(d / 0.3) ** 2)
        _stamp(shell, cx, cy, r, np.clip(cone, 0, 1))
    covered = smoothstep(0.05, 0.3, shell)[..., None]
    white = np.array((0.24, 0.23, 0.19)) * (0.5 + 0.7 * shell[..., None]) * (0.8 + 0.4 * fine[..., None])
    diff = algae * (1 - covered) + white * covered
    height = height + 1.4 * shell
    gy, gx = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) / 2, (np.roll(height, -1, 1) - np.roll(height, 1, 1)) / 2
    k = size / 256.0
    n = np.stack((-gx * k, gy * k, np.ones_like(height)), -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    rough = 0.32 + 0.2 * fine + 0.35 * covered[..., 0]
    srgb = np.clip(diff, 0, 1) ** (1 / 2.2)
    return {"diff": (srgb * 255).round().astype(np.uint8), "rough": (np.clip(rough, 0, 1) * 255).round().astype(np.uint8),
            "nor_gl": ((n * 0.5 + 0.5) * 255).round().astype(np.uint8)}
