"""Pure-numpy helpers for the demo render: rust grid -> RGBA overlay texture, sleeve mesh around the pipe segment,
rig-deck placement and camera paths. Texture/mesh conventions follow RustState.surface_to_uv: u = theta / 2pi
(theta from the pipe ref axis toward the binormal), v = (z + L/2) / L, grid rows = z, cols = theta."""
from __future__ import annotations

import math

import numpy as np


def sleeve_mesh(origin, axis, ref, binormal, radius: float, length: float, n_theta: int = 96, n_z: int = 24):
    """Open cylinder around the pipe segment. Returns points (V,3), face counts, face indices, uv (V,2)."""
    o, a, e, b = (np.asarray(v, np.float64) for v in (origin, axis, ref, binormal))
    th = np.linspace(0.0, 2 * math.pi, n_theta + 1)
    z = np.linspace(-length / 2, length / 2, n_z + 1)
    Z, T = np.meshgrid(z, th, indexing="ij")
    n = np.cos(T)[..., None] * e + np.sin(T)[..., None] * b
    pts = o + Z[..., None] * a + radius * n
    uv = np.stack([T / (2 * math.pi), (Z + length / 2) / length], -1)
    w = n_theta + 1
    iz, it = np.meshgrid(np.arange(n_z), np.arange(n_theta), indexing="ij")
    k = (iz * w + it).ravel()
    idx = np.stack([k, k + 1, k + w + 1, k + w], -1).ravel()
    return pts.reshape(-1, 3), [4] * (n_z * n_theta), idx.tolist(), uv.reshape(-1, 2)


def grid_to_mask(grid: np.ndarray, out_h: int, out_w: int, flip_v: bool = False) -> np.ndarray:
    """Bilinear upsample of an (H, W) grid sampled at cell centres; columns wrap (theta), rows clamp (z).
    flip_v puts v = 1 at image row 0 (texture-image convention)."""
    g = np.asarray(grid, np.float32)
    H, W = g.shape
    v = (np.arange(out_h) + 0.5) / out_h
    if flip_v:
        v = 1.0 - v
    y = np.clip(v * H - 0.5, 0, H - 1)
    x = (np.arange(out_w) + 0.5) / out_w * W - 0.5
    y0 = np.floor(y).astype(int)
    y1 = np.minimum(y0 + 1, H - 1)
    fy = (y - y0)[:, None].astype(np.float32)
    x0 = np.floor(x).astype(int)
    fx = (x - x0)[None, :].astype(np.float32)
    x0, x1 = x0 % W, (x0 + 1) % W
    top = g[y0][:, x0] * (1 - fx) + g[y0][:, x1] * fx
    bot = g[y1][:, x0] * (1 - fx) + g[y1][:, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def periodic_noise(h: int, w: int, seed: int = 0, octaves: int = 5) -> np.ndarray:
    """Fractal value noise in [0, 1], periodic along columns."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for k in range(octaves):
        gh, gw = 4 * 2 ** k, 8 * 2 ** k
        out += amp * grid_to_mask(rng.random((gh, gw), np.float32), h, w)
        total += amp
        amp *= 0.55
    out /= total
    return (out - out.min()) / max(float(out.max() - out.min()), 1e-6)


def tile_texture(tex: np.ndarray, h: int, w: int, tiles_v: float = 1.0, tiles_u: float = 1.0) -> np.ndarray:
    """Nearest-neighbour resample of `tex` repeated tiles_v x tiles_u times over an (h, w) image."""
    th, tw = tex.shape[:2]
    rows = (np.floor(np.arange(h) * tiles_v * th / h).astype(int)) % th
    cols = (np.floor(np.arange(w) * tiles_u * tw / w).astype(int)) % tw
    return tex[rows][:, cols]


def rust_rgba(level: np.ndarray, tex: np.ndarray, noise: np.ndarray, edge: tuple = (0.12, 0.2),
              breakup: float = 0.6) -> np.ndarray:
    """Upsampled rust level (h, w) in [0, 1] -> RGBA uint8: rust texture colour, alpha = ragged threshold of the
    level perturbed by `noise`, so scraping erodes the patch from its thin edges inward."""
    h, w = level.shape
    if tex.shape[:2] != (h, w):
        tex = tile_texture(tex, h, w)
    s = np.clip(level, 0, 1) * 1.3 - noise * breakup
    t = np.clip((s - edge[0]) / (edge[1] - edge[0]), 0, 1)
    alpha = t * t * (3 - 2 * t)
    shade = (1.1 - 0.35 * np.clip(level, 0, 1))[..., None]
    rgb = np.clip(tex[..., :3].astype(np.float32) * shade, 0, 255)
    return np.concatenate([rgb, alpha[..., None] * 255.0], -1).round().astype(np.uint8)


def rig_translation(meta: dict, env_origin=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Translation to apply to the rig deck so its main pipe lands on the scrub env's pipe (play_scrub.apply_meta)."""
    tf = meta["env_to_world"]
    assert abs(tf["yaw"]) < 1e-9, "only the unrotated env frame is supported"
    return np.asarray(env_origin, np.float64) - np.asarray(tf["translation"], np.float64)


def camera_path(keys, t: float):
    """keys: [(time, eye, lookat), ...] sorted by time -> (eye, lookat) at t, smoothstep-eased between keys."""
    if t <= keys[0][0]:
        return np.asarray(keys[0][1], float), np.asarray(keys[0][2], float)
    for (t0, e0, l0), (t1, e1, l1) in zip(keys, keys[1:]):
        if t <= t1:
            s = (t - t0) / max(t1 - t0, 1e-9)
            s = s * s * (3 - 2 * s)
            return (np.asarray(e0, float) + s * (np.asarray(e1, float) - np.asarray(e0, float)),
                    np.asarray(l0, float) + s * (np.asarray(l1, float) - np.asarray(l0, float)))
    return np.asarray(keys[-1][1], float), np.asarray(keys[-1][2], float)
