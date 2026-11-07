"""Mock wrist-camera renderer for a bolted flange (no Isaac). Used by the CPU mock and to make synthetic
training crops for the crack classifier (like the rust detector's synthetic subset).

Draws, in perspective through a policy.interfaces.CameraModel: the flange face, the pipe bore, each bolt head
turned by its true angle, the red torque stripe on the head and its fixed partner on the flange face, a crack
(dark jagged line, plus a faint vapour haze when severe), and optional dust (haze + blur) or glare.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from policy.leak.geometry import project

STEEL = (150, 153, 160)
BORE = (58, 62, 70)
HEAD = (118, 122, 130)
EDGE = (80, 83, 90)
RED = (205, 35, 35)
CRACK = (22, 22, 26)
BACKGROUND = (38, 44, 52)


def _circle(center, radius, u, v, n=48, phase=0.0):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False) + phase
    return center + radius * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v)


def _quad(p0, p1, half_width, normal):
    e = (p1 - p0) / np.linalg.norm(p1 - p0)
    w = np.cross(normal, e)
    return np.stack([p0 + half_width * w, p1 + half_width * w, p1 - half_width * w, p0 - half_width * w])


def crack_path(geom, severity: float, seed: int = 7) -> np.ndarray:
    """Deterministic jagged crack between bolts 1 and 2, from near the bore toward the rim (world points)."""
    rng = np.random.default_rng(seed)
    phi = 2 * math.pi * 1.5 / geom.n_bolts
    e = math.cos(phi) * geom.ref + math.sin(phi) * geom.binormal
    w = np.cross(geom.normal, e)
    r = np.linspace(geom.pipe_radius * 1.05, geom.outer_radius * 0.93, 9)
    jitter = rng.normal(0.0, 0.006, len(r))
    jitter[0] = 0.0
    return np.stack([geom.center + ri * e + ji * w for ri, ji in zip(r, jitter)])


def render_flange(camera, state, dust: float = 0.0, glare: bool = False, crack_seed: int = 7) -> np.ndarray:
    g = state.geom
    W, H = int(camera.width), int(camera.height)
    img = Image.new("RGB", (W, H), BACKGROUND)
    d = ImageDraw.Draw(img, "RGBA")
    u, v, n = g.ref, g.binormal, g.normal

    def poly(pts):
        uv, z = project(camera, pts)
        if (z <= 0.05).any():
            return None
        return [tuple(map(float, p)) for p in uv]

    def draw(pts, fill, outline=None):
        p = poly(pts)
        if p:
            d.polygon(p, fill=fill, outline=outline)

    draw(_circle(g.center, g.outer_radius, u, v, 96), STEEL, EDGE)
    draw(_circle(g.center, g.pipe_radius, u, v, 64), BORE)

    if state.crack > 0:
        path = crack_path(g, state.crack, crack_seed)
        p = poly(path)
        if p:
            d.line(p, fill=CRACK, width=max(1, int(round(1 + 4 * state.crack))))
            if state.crack >= 0.2:
                tip_uv, tip_z = project(camera, (path[-1] + 0.03 * n)[None])
                tx, ty = float(tip_uv[0, 0]), float(tip_uv[0, 1])
                rpx = 40 * state.crack
                d.ellipse([tx - rpx, ty - rpx, tx + rpx, ty + rpx], fill=(235, 235, 240, int(70 * state.crack)))

    for k in range(g.n_bolts):
        c = g.bolt_center(k)
        face_c = g.center + g.bolt_circle * g.radial(k)
        rh = g.head_radius
        e0 = g.radial(k)
        draw(_quad(face_c + 1.25 * rh * e0, face_c + 2.1 * rh * e0, 0.14 * rh, n), RED)     # stripe on the face
        draw(_circle(c, rh, u, v, 6, phase=g.bolt_phi(k) + float(state.angles[k]) + math.pi / 6), HEAD, EDGE)
        ea = g.radial(k, float(state.angles[k]))
        draw(_quad(c - 0.9 * rh * ea, c + 0.9 * rh * ea, 0.12 * rh, n), RED)                 # stripe on the head

    if dust > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=5.0 * dust))
    a = np.asarray(img, np.float32)
    if glare:
        yy, xx = np.mgrid[0:H, 0:W]
        gl = np.exp(-(((xx - 0.55 * W) / (0.25 * W)) ** 2 + ((yy - 0.45 * H) / (0.2 * H)) ** 2))
        a = a * (1 - 0.6 * gl[..., None]) + 250 * 0.6 * gl[..., None]
    if dust > 0:
        a = a * (1 - 0.8 * dust) + 150.0 * 0.8 * dust    # grime on the lens: haze toward grey
    return np.clip(a, 0, 255).astype(np.uint8)
