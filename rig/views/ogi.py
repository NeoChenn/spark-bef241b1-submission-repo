import numpy as np
from PIL import Image

from rig.views.geom import rays
from sim.plume.colormap import lel_rgb


def gas_column(field, cam: dict, depth=None, stride: int = 4, n: int = 24, max_range: float = 25.0,
               near: float = 0.3) -> np.ndarray:
    o, d = rays(cam, stride)
    norm = np.linalg.norm(d, axis=-1)
    u = d / norm[..., None]
    far = np.full(norm.shape, max_range)
    if depth is not None:
        z = np.asarray(depth, np.float64)[::stride, ::stride][: d.shape[0], : d.shape[1]] * norm
        far = np.where(np.isfinite(z), np.minimum(z, max_range), max_range)
    lo, hi = field.bounds()
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(u) < 1e-9, 1e-9, u)
        a, b = (lo - o) * inv, (hi - o) * inv
    t0 = np.maximum(np.minimum(a, b).max(-1), near)
    t1 = np.minimum(np.maximum(a, b).min(-1), far)
    hit = t1 > t0
    out = np.zeros(norm.shape)
    if not hit.any() or field.fraction.max() / field.lel * 100.0 < 0.5:
        return out
    s0, s1, uu = t0[hit], t1[hit], u[hit]
    k = (np.arange(n) + 0.5) / n
    t = s0[:, None] + (s1 - s0)[:, None] * k
    pts = o + uu[:, None, :] * t[..., None]
    pct = field.lel_percent(pts.reshape(-1, 3)).reshape(len(s0), n)
    out[hit] = pct.sum(1) * (s1 - s0) / n
    return out


def ogi_image(rgb, column, ref: float = 50.0) -> Image.Image:
    p = np.asarray(rgb, np.float32)[..., :3]
    lum = p @ np.array([0.299, 0.587, 0.114], np.float32)
    a, b = np.percentile(lum, (2, 98))
    base = np.clip((lum - a) / max(b - a, 1.0), 0, 1) * 200 + 25
    h, w = lum.shape
    g = np.asarray(Image.fromarray(np.asarray(column, np.float32)).resize((w, h), Image.BILINEAR))
    q = g / ref
    col = lel_rgb(np.clip(q * 100.0, 0, 100)) * 255.0
    alpha = (np.clip(q * 1.5, 0, 0.85) * (g > 1.0))[..., None]
    out = base[..., None] * (1 - alpha) + col * alpha
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))
