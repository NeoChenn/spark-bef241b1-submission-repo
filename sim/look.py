from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageChops

EV_KEYS = ((1.9, -0.4), (8.0, 0.3), (40.0, 0.4), (120.0, 0.8), (330.0, 1.0))
VIEWS = {
    "wide": {"ev": 1.0, "fstop": 0.0},
    "deck_mid": {"ev": 0.4, "fstop": 0.0},
    "repair": {"ev": -0.4, "fstop": 1.4},
    "waterline": {"ev": 0.8, "fstop": 0.0},
    "flare": {"ev": 0.3, "fstop": 0.0},
}
PRESET_GRADE = {"dusk": {"highlight": (0.06, 0.015, -0.07), "shadow": (-0.035, 0.01, 0.04), "contrast": 0.25}}
PRESET_EV = {"storm": -0.7, "golden": 0.3}
PRESET_EV_KEYS = {"dusk": ((3.0, 0.4), (30.0, 0.0), (150.0, -0.6), (330.0, -0.8))}
ATMOS_EV = {"wide": -0.6, "waterline": -0.3, "flare": -0.3}
ATMOS_EV_KEYS = ((8.0, 0.0), (60.0, -0.3), (250.0, -0.6))
LENS_SCALE = 100.0
BASE_ISO = 100.0
KELVIN = {2000.0: 2100.0, 4000.0: 4500.0, 6000.0: 7000.0}
FIXTURES = {
    "rect": {"gain": 0.6, "cone": 70.0, "softness": 0.6, "focus": 2.0},
    "spot": {"gain": 1.0, "cone": None, "cone_max": 40.0, "softness": 0.5, "focus": 1.0},
    "bulb": {"gain": 1.0},
}
WORK_LIGHT = {"offset": (0.3, 1.0, 1.3), "kelvin": 6500.0, "intensity": 8.0e7, "radius": 0.05, "cone": 24.0,
              "softness": 0.7}
TONEMAP = {
    "/rtx/post/tonemap/op": 6,
    "/rtx/raytracing/fractionalCutoutOpacity": True,
    "/rtx/pathtracing/fractionalCutoutOpacity": True,
    "/rtx/post/histogram/enabled": False,
}
BLOOM_CUTOFF = 6.0
SUN_DISC = {"distance": 2000.0, "radius": 30.0, "color": (1.0, 0.7, 0.42), "gain": 400.0}


def sun_direction(rotate_xyz) -> tuple:
    x, _, z = (math.radians(a) for a in rotate_xyz)
    travel = np.array([-math.sin(z) * math.sin(x), math.cos(z) * math.sin(x), -math.cos(x)])
    return tuple(float(v) for v in -travel)


INSCATTER = {
    "dusk": {"color": (1.0, 0.64, 0.38), "density": 1.0 / 90.0, "height": 30.0, "g": 0.72, "sun": 1.0, "ambient": 0.3,
             "max": 0.5, "sky": 1500.0, "scale": 4, "dark": 0.3},
}


LIVE_HAZE = False


def inscatter(rgb: np.ndarray, depth, cam: dict, preset: str, sun_dir, strength: float = 1.0,
              tier: str = "offline") -> np.ndarray:
    p = INSCATTER.get(preset)
    if not p or depth is None or (tier == "live" and not LIVE_HAZE):
        return rgb
    depth = depth() if callable(depth) else depth
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    s = p["scale"]
    d = np.asarray(depth, np.float32).reshape(depth.shape[0], depth.shape[1])[s // 2::s, s // 2::s]
    d = np.where(np.isfinite(d) & (d > 0), np.minimum(d, p["sky"]), p["sky"])
    hs, ws = d.shape
    K, T = np.asarray(cam["K"], np.float32), np.asarray(cam["T"], np.float32)
    fy = K[1, 1] * hs / h
    fx = K[0, 0] * ws / w
    v, u = np.mgrid[0:hs, 0:ws].astype(np.float32)
    rays = np.stack([(u + 0.5 - ws / 2) / fx, (v + 0.5 - hs / 2) / fy, np.ones_like(u)], -1) @ T[:3, :3].T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    k = rays[..., 2] * d / p["height"]
    safe = np.where(np.abs(k) > 1e-4, k, 1.0)
    ratio = np.where(np.abs(k) > 1e-4, (1 - np.exp(-np.clip(k, -30, 30))) / safe, 1.0)
    tau = p["density"] * d * np.exp(-max(float(T[2, 3]), 0.0) / p["height"]) * ratio
    f = np.clip(1 - np.exp(-np.clip(tau, 0, 50)), 0, p["max"]) * strength
    lum = np.asarray(rgb[s // 2::s, s // 2::s, :3], np.float32).mean(-1)[:hs, :ws] / 255.0
    f *= p.get("dark", 1.0) + (1 - p.get("dark", 1.0)) * np.clip(lum / 0.35, 0, 1)
    g = p["g"]
    cos = rays @ np.asarray(sun_dir, np.float32)
    phase = (1 - g * g) / (1 + g * g - 2 * g * cos) ** 1.5 * (1 - g) ** 3 / (1 - g * g)
    col = np.clip(np.asarray(p["color"], np.float32) * (p["ambient"] + p["sun"] * phase)[..., None], 0, 1)
    fog = Image.fromarray((col * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    mask = Image.fromarray((f * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return np.asarray(Image.composite(fog, Image.fromarray(np.ascontiguousarray(rgb[..., :3])), mask))


def bloom(scale: float) -> dict:
    return {"/rtx/post/lensFlares/enabled": True, "/rtx/post/lensFlares/physicalSettings": False,
            "/rtx/post/lensFlares/flareScale": scale, "/rtx/post/lensFlares/anisoFlareWeight": 0.0,
            "/rtx/post/lensFlares/haloFlareWeight": 0.0, "/rtx/post/lensFlares/isotropicFlareWeight": 0.4,
            "/rtx/post/lensFlares/cutoffPoint": [BLOOM_CUTOFF] * 3, "/rtx/post/lensFlares/cutoffFuzziness": 0.8}


GRADE = {"contrast": 0.3, "toe": 1.15, "lift": (0.0, 0.002, 0.004), "shadow": (-0.03, 0.008, 0.035),
         "highlight": (0.035, 0.01, -0.04), "vignette": 0.3}
TIERS = {
    "none": {"rtx": {}, "grade": None},
    "live": {
        "rtx": {**TONEMAP, **bloom(0.07)},
        "grade": {**GRADE, "saturation": 1.0, "grain": 0.0},
    },
    "offline": {
        "rtx": {**TONEMAP, **bloom(0.1),
                "/rtx/post/motionblur/enabled": True, "/rtx/post/chromaticAberration/enabled": True,
                "/rtx/post/chromaticAberration/strengthR": -0.006, "/rtx/post/chromaticAberration/strengthG": 0.0,
                "/rtx/post/chromaticAberration/strengthB": 0.006},
        "grade": {**GRADE, "saturation": 0.95, "grain": 0.012},
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


def view_look(view: str | None, distance: float, atmos: bool = False, preset: str = "night") -> dict:
    base = dict(VIEWS.get(view) or {"ev": ev_at(distance), "fstop": 0.0})
    base["ev"] += ev_at(distance, PRESET_EV_KEYS[preset]) if preset in PRESET_EV_KEYS else PRESET_EV.get(preset, 0.0)
    if atmos:
        base["ev"] += ATMOS_EV.get(view, 0.0) if view in VIEWS else ev_at(distance, ATMOS_EV_KEYS)
    return base


def rtx_settings(tier: str, ev: float = 0.0) -> dict:
    rtx = TIERS[tier]["rtx"]
    return {**rtx, **exposure(ev)} if rtx else {}


def fixture(kind: str, kelvin: float, intensity: float, cone: float | None = None) -> dict:
    f = FIXTURES[kind]
    out = {"kelvin": KELVIN.get(float(kelvin), float(kelvin)), "intensity": float(intensity) * f["gain"]}
    if "softness" in f:
        c = f["cone"] if f["cone"] is not None else min(cone, f.get("cone_max", cone))
        out.update(cone=c, softness=f["softness"], focus=f["focus"])
    return out


def work_light(target) -> tuple:
    pos = np.add(target, WORK_LIGHT["offset"])
    aim = np.subtract(target, pos)
    return tuple(float(v) for v in pos), tuple(float(v) for v in aim / np.linalg.norm(aim))


def lens(distance: float, fstop: float) -> dict:
    return {"fStop": float(fstop) * LENS_SCALE, "focusDistance": float(distance)}


def curves(p: dict) -> np.ndarray:
    x = np.linspace(0.0, 1.0, 256)
    s = x * x * (3 - 2 * x)
    base = (x + p["contrast"] * (s - x)) ** p["toe"]
    low = 6.75 * base * (1 - base) ** 2
    high = 6.75 * base ** 2 * (1 - base)
    out = [lift + (1 - lift) * np.clip(base + sh * low + hi * high, 0, 1)
           for lift, sh, hi in zip(p["lift"], p["shadow"], p["highlight"])]
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


def grade(rgb: np.ndarray, tier: str, t: float = 0.0, preset: str = "night") -> np.ndarray:
    p = TIERS[tier]["grade"]
    if not p:
        return rgb
    p = {**p, **PRESET_GRADE.get(preset, {})}
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
