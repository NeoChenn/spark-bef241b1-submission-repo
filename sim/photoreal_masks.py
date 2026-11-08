from __future__ import annotations

import numpy as np

PUDDLE_EDGE = 0.06


def tileable_noise(size: int, frequency: float, seed: int = 0, octaves: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(size) * size
    radius = np.hypot(*np.meshgrid(k, k, indexing="ij"))
    total = np.zeros((size, size))
    for octave in range(octaves):
        f = frequency * 2**octave
        spectrum = np.exp(-((radius / f) ** 2)) * (radius > 0)
        phase = np.fft.fft2(rng.standard_normal((size, size)))
        total += np.real(np.fft.ifft2(phase * spectrum)) * 0.5**octave / max(np.sqrt((spectrum**2).sum()), 1e-9)
    return (total - total.min()) / (total.max() - total.min())


def smoothstep(a: float, b: float, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def puddle_threshold(level: float) -> float:
    return 1.0 - 0.55 * float(np.clip(level, 0.0, 1.0))


def puddle_coverage(field: np.ndarray, level: float) -> float:
    if level <= 0:
        return 0.0
    edge = puddle_threshold(level)
    return float(smoothstep(edge, edge + PUDDLE_EDGE, field).mean())


def masks(size: int = 2048, seed: int = 11) -> np.ndarray:
    macro = tileable_noise(size, 3.0, seed)
    puddle = tileable_noise(size, 2.0, seed + 1, octaves=3)
    grime = tileable_noise(size, 12.0, seed + 2)
    salt = smoothstep(0.45, 0.8, tileable_noise(size, 24.0, seed + 3)) * tileable_noise(size, 4.0, seed + 4)
    salt = salt / max(salt.max(), 1e-9)
    return (np.stack((macro, puddle, grime, salt), axis=-1) * 255).round().astype(np.uint8)


def streaks(size: int = 2048, seed: int = 21, count: int = 260) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size))
    rows = np.arange(size)
    for _ in range(count):
        x = rng.integers(size)
        width = rng.integers(2, max(3, size // 48))
        start = rng.integers(size)
        length = rng.integers(size // 8, size // 2)
        strength = rng.uniform(0.35, 1.0)
        along = (rows - start) % size
        fade = np.clip(1 - along / length, 0.0, 1.0) ** 1.5 * strength
        cols = (x + np.arange(-width, width + 1)) % size
        profile = np.exp(-((np.arange(-width, width + 1) / (width / 2 + 0.5)) ** 2))
        out[:, cols] = np.maximum(out[:, cols], fade[:, None] * profile[None, :])
    jitter = tileable_noise(size, 40.0, seed + 1)
    out = np.clip(out * (0.55 + 0.6 * jitter), 0, 1)
    return (out * 255).round().astype(np.uint8)
