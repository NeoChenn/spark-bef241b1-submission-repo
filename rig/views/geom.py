import numpy as np


def _kt(cam):
    return np.asarray(cam["K"], np.float64), np.asarray(cam["T"], np.float64)


def scaled(cam: dict, w: int, h: int) -> dict:
    K, _ = _kt(cam)
    sx, sy = w / cam["w"], h / cam["h"]
    K2 = [[K[0, 0] * sx, 0.0, K[0, 2] * sx], [0.0, K[1, 1] * sy, K[1, 2] * sy], [0.0, 0.0, 1.0]]
    return {**cam, "K": K2, "w": int(w), "h": int(h)}


def rays(cam: dict, stride: int = 1):
    K, T = _kt(cam)
    v, u = np.mgrid[0:cam["h"]:stride, 0:cam["w"]:stride].astype(np.float64) + 0.5 * stride
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    return T[:3, 3].copy(), d @ T[:3, :3].T


def backproject(depth, cam: dict, stride: int = 1, max_range: float = 40.0) -> np.ndarray:
    o, d = rays(cam, stride)
    z = np.asarray(depth, np.float64)[::stride, ::stride][: d.shape[0], : d.shape[1]]
    ok = np.isfinite(z) & (z > 0.05) & (z < max_range)
    return o + d[ok] * z[ok][:, None]
