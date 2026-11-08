from __future__ import annotations

import cv2
import numpy as np


def render_lens(salt: float, size: int = 160, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), (95, 100, 105), np.float32)
    c, r = size // 2, int(size * 0.36)
    cv2.circle(img, (c, c), int(r * 1.2), (60, 62, 66), -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), r, (22, 28, 40), -1, cv2.LINE_AA)
    window = np.zeros((size, size), np.uint8)
    cv2.circle(window, (c, c), r, 255, -1)
    field = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), size * 0.03)
    inside = field[window > 0]
    if salt > 0 and inside.size:
        thr = np.quantile(inside, 1.0 - min(max(salt, 0.0), 1.0))
        crust = (field >= thr) & (window > 0)
        img[crust] = (215, 218, 214) + rng.normal(0, 12, (int(crust.sum()), 3))
    img += rng.normal(0, 4, img.shape)
    return img.clip(0, 255).astype(np.uint8)


def lens_obscuration(rgb: np.ndarray) -> float:
    gray = cv2.GaussianBlur(cv2.cvtColor(np.asarray(rgb)[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY), (5, 5), 0)
    h, w = gray.shape
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=w, param1=80, param2=25,
                               minRadius=int(w * 0.25), maxRadius=int(w * 0.45))
    if circles is not None:
        cx, cy, r = circles[0][0]
    else:
        cx, cy, r = w / 2.0, h / 2.0, w * 0.36
    yy, xx = np.mgrid[0:h, 0:w]
    window = (xx - cx) ** 2 + (yy - cy) ** 2 < (0.95 * r) ** 2
    return float((gray[window] > 140).mean()) if window.any() else 1.0
