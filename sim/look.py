from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageChops

EV_KEYS = ((1.9, -0.2), (8.0, 0.0), (40.0, 0.4), (120.0, 0.8), (330.0, 1.0))
VIEWS = {
    "wide": {"ev": 1.0, "fstop": 0.0},
    "deck_mid": {"ev": 0.0, "fstop": 0.0},
    "repair": {"ev": -0.2, "fstop": 2.0},
    "waterline": {"ev": 0.8, "fstop": 0.0},
    "flare": {"ev": 0.3, "fstop": 0.0},
}
LENS_SCALE = 100.0
BASE_ISO = 100.0
KELVIN = {2000.0: 2100.0, 4000.0: 4500.0, 6000.0: 7000.0}
FIXTURES = {
    "rect": {"gain": 0.6, "cone": 70.0, "softness": 0.6, "focus": 2.0},
    "spot": {"gain": 1.0, "cone": None, "softness": 0.65, "focus": 1.0},
    "bulb": {"gain": 1.0},
}
TONEMAP = {
    "/rtx/post/tonemap/op": 6,
    "/rtx/post/histogram/enabled": False,
}
TIERS = {
    "none": {"rtx": {}, "grade": None},
    "live": {
        "rtx": {**TONEMAP, "/rtx/post/lensFlares/enabled": True, "/rtx/post/lensFlares/physicalSettings": False,
                "/rtx/post/lensFlares/flareScale": 0.08},
        "grade": {"contrast": 0.1, "gamma": (1.2, 0.98, 0.78), "gain": (1.08, 1.0, 0.86), "lift": (0.0, 0.015, 0.025),
                  "vignette": 0.35, "saturation": 1.0, "grain": 0.0},
    },
    "offline": {
        "rtx": {**TONEMAP, "/rtx/post/lensFlares/enabled": True, "/rtx/post/lensFlares/physicalSettings": True,
                "/rtx/post/lensFlares/flareScale": 0.25,
                "/rtx/post/motionblur/enabled": True, "/rtx/post/chromaticAberration/enabled": True,
                "/rtx/post/chromaticAberration/strengthR": -0.006, "/rtx/post/chromaticAberration/strengthG": 0.0,
                "/rtx/post/chromaticAberration/strengthB": 0.006},
        "grade": {"contrast": 0.1, "gamma": (1.2, 0.98, 0.78), "gain": (1.08, 1.0, 0.86), "lift": (0.0, 0.015, 0.025),
                  "vignette": 0.35, "saturation": 0.9, "grain": 0.025},
    },
}


def ev_at(distance: float, keys=EV_KEYS) -> float:
    d = math.log(max(float(distance), 1e-3))
    xs = [math.log(k) for k, _ in keys]
    ys = [v for _, v in keys]
    if d <= xs[0]:
        return ys[0]
    for x0, x1, y0, y1 in zip(xs, xs[1:], ys, ys[1:]):
        if d <= x1:
            return y0 + (y1 - y0) * (d - x0) / (x1 - x0)
    return ys[-1]


def exposure(ev: float) -> dict:
    return {"/rtx/post/tonemap/filmIso": BASE_ISO * 2.0 ** ev}


def view_look(view: str | None, distance: float) -> dict:
    return VIEWS.get(view) or {"ev": ev_at(distance), "fstop": 0.0}


def rtx_settings(tier: str, ev: float = 0.0) -> dict:
    rtx = TIERS[tier]["rtx"]
    return {**rtx, **exposure(ev)} if rtx else {}


def fixture(kind: str, kelvin: float, intensity: float, cone: float | None = None) -> dict:
    f = FIXTURES[kind]
    out = {"kelvin": KELVIN.get(float(kelvin), float(kelvin)), "intensity": float(intensity) * f["gain"]}
    if "softness" in f:
        out.update(cone=f["cone"] if f["cone"] is not None else cone, softness=f["softness"], focus=f["focus"])
    return out


def lens(distance: float, fstop: float) -> dict:
    return {"fStop": float(fstop) * LENS_SCALE, "focusDistance": float(distance)}


def curves(p: dict) -> np.ndarray:
    x = np.linspace(0.0, 1.0, 256)
    s = x * x * (3 - 2 * x)
    base = x + p["contrast"] * (s - x)
    out = []
    for gamma, gain, lift in zip(p["gamma"], p["gain"], p["lift"]):
        c = lift + (1 - lift) * np.clip(gain * base ** gamma, 0, 1)
        out.append(c)
    return np.clip(np.rint(np.stack(out) * 255), 0, 255).astype(np.uint8)


_CACHE: dict = {}


def vignette(h: int, w: int, strength: float) -> np.ndarray:
    key = ("v", h, w, strength)
    if key not in _CACHE:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx + 0.5) / w - 0.5) ** 2 * 2 + ((yy + 0.5) / h - 0.5) ** 2 * 2
        m = np.rint(255 * (1 - strength * np.clip(r2, 0, 1) ** 1.5)).astype(np.uint8)
        _CACHE[key] = Image.fromarray(np.repeat(m[..., None], 3, -1))
    return _CACHE[key]


def _lut(p: dict) -> list:
    key = ("c", tuple(sorted((k, v) for k, v in p.items())))
    if key not in _CACHE:
        _CACHE[key] = curves(p).reshape(-1).tolist()
    return _CACHE[key]


def grade(rgb: np.ndarray, tier: str, t: float = 0.0) -> np.ndarray:
    p = TIERS[tier]["grade"]
    if not p:
        return rgb
    img = Image.fromarray(np.ascontiguousarray(rgb[..., :3])).point(_lut(p))
    if p["vignette"]:
        img = ImageChops.multiply(img, vignette(img.height, img.width, p["vignette"]))
    out = np.asarray(img)
    if p["saturation"] == 1.0 and not p["grain"]:
        return out
    f = out.astype(np.float32)
    lum = f @ np.float32([0.2126, 0.7152, 0.0722])
    f = lum[..., None] + p["saturation"] * (f - lum[..., None])
    if p["grain"]:
        rng = np.random.default_rng(int(t * 1000) & 0xFFFFFFFF)
        n = rng.standard_normal(f.shape[:2], np.float32) * (255 * p["grain"])
        m = np.clip(lum / 255, 0, 1)
        f += (n * (4 * m * (1 - m) + 0.15))[..., None]
    return np.clip(f + 0.5, 0, 255).astype(np.uint8)
