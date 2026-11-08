from __future__ import annotations

import math
import os
from pathlib import Path

import photoreal as pr

SIM_DIR = Path(__file__).resolve().parent
TEX = SIM_DIR / "assets" / "third_party" / "textures"
MDL = SIM_DIR / "assets" / "mdl" / "rig_hero.mdl"
LAYER = SIM_DIR / "output" / "hero_mat.usda"
NOISE = SIM_DIR / "assets" / "generated" / "hero_noise.png"
SHADER = "hero_mdl"
SUBS = ("rig_hero_pipe", "rig_hero_rust", "rig_hero_deck", "rig_hero_line", "rig_hero_paint")
PATCH_RINGS, PATCH_SEGMENTS = 8, 48
PIPE_TINT = (0.95, 1.0, 0.92)
PUDDLE_BOOST = 0.25

TARGETS = {
    "/World/RepairZone/Looks/green_metal_rust": "pipe",
    "/World/RepairZone/Looks/rust_coarse_01": "rust",
    "/World/Looks/deck": "deck",
    "/World/Looks/walkway_line": "line",
    "/World/Looks/zone_marking": "line",
    "/World/Looks/frame": "frame",
    "/World/RepairZone/Looks/rusty_painted_metal": "zone_steel",
}
WET_PUDDLES = ("deck", "frame", "zone_steel")
PATCH_GEOM = "/World/Pipes/damage_patch/geom"


def tex(folder: str, name: str) -> Path:
    return TEX / folder / f"{name}.jpg"


def wrap_scale(radius: float, scale: float) -> float:
    circumference = 2.0 * math.pi * radius
    return circumference / max(1, round(circumference / scale))


def patch_st(n_points: int, rings: int = PATCH_RINGS, segments: int = PATCH_SEGMENTS) -> list[tuple[float, float]]:
    if n_points != 1 + rings * segments:
        raise ValueError(f"damage patch has {n_points} points, expected {1 + rings * segments}")
    st = [(0.0, 0.0)]
    for ring in range(1, rings + 1):
        st.extend((ring / rings, step / segments) for step in range(segments))
    return st


def weather(preset: str) -> dict:
    w = pr.weather(pr.rain_for(preset))
    return {"wetness": w["wetness"], "puddles": round(min(w["puddles"] + PUDDLE_BOOST, 1.0) if w["puddles"] else 0.0, 4)}


def warm(rgb) -> bool:
    r, g, b = rgb
    return r > b and r > 0.12 and (r - g) / (r + 1e-3) > 0.07


def recipes(meta: dict) -> dict:
    pipe = meta["world"]["main_pipe"]
    dmg = meta["world"]["damage"]
    geo = {"pipe_y": pipe["y_m"], "pipe_z": pipe["z_m"], "pipe_r": pipe["radius_m"]}
    paint_scale = wrap_scale(pipe["radius_m"], 0.32)
    paint = "green_metal_rust_4k"
    paint_set = {"chip_mask": tex("PaintedMetal010_4k", "diff"), "paint_rough": tex("PaintedMetal010_4k", "rough"),
                 "paint_normal": tex("PaintedMetal010_4k", "nor_gl"), "rust_diff": tex("Rust007_4k", "diff"),
                 "rust_normal": tex("Rust007_4k", "nor_gl"), "noise": NOISE, "streaks": pr.COMMON["streaks"],
                 "grime_tex": tex("SurfaceImperfections011", "diff")}
    return {
        "pipe": ("rig_hero_pipe", {
            "paint_diff": tex(paint, "diff"), "paint_rough": tex(paint, "rough"), "paint_normal": tex(paint, "nor_gl"),
            "chip_mask": tex("PaintedMetal010_4k", "diff"), "rust_diff": tex("Rust007_4k", "diff"),
            "rust_normal": tex("Rust007_4k", "nor_gl"), "noise": NOISE, "streaks": pr.COMMON["streaks"],
            "paint_tint": PIPE_TINT, "paint_scale": paint_scale, "rust_scale": wrap_scale(pipe["radius_m"], 0.45),
            "chip_scale": wrap_scale(pipe["radius_m"], 0.8), **geo, "damage": tuple(dmg["position_m"]),
            "damage_r": dmg["approx_radius_m"] * 0.8,
        }),
        "rust": ("rig_hero_rust", {
            "rust_diff": tex("rust_coarse_01_4k", "diff"), "rust_rough": tex("rust_coarse_01_4k", "rough"),
            "rust_normal": tex("rust_coarse_01_4k", "nor_gl"), "scale_diff": tex("Rust007_4k", "diff"),
            "scale_normal": tex("Rust007_4k", "nor_gl"), "paint_diff": tex(paint, "diff"), "noise": NOISE,
            "paint_tint": PIPE_TINT, "paint_scale": paint_scale, "rust_scale": wrap_scale(pipe["radius_m"], 0.3), **geo,
        }),
        "deck": ("rig_hero_deck", {
            "plate_diff": tex("DiamondPlate008C_4k", "diff"), "plate_rough": tex("DiamondPlate008C_4k", "rough"),
            "plate_normal": tex("DiamondPlate008C_4k", "nor_gl"), "plate_ao": tex("DiamondPlate008C_4k", "ao"),
            "noise": NOISE, "rust_diff": tex("Rust007_4k", "diff"), "grime_tex": tex("SurfaceImperfections011", "diff"),
        }),
        "line": ("rig_hero_line", {"noise": NOISE, "wear_tex": tex("SurfaceImperfections011", "diff")}),
        "frame": ("rig_hero_paint", {**paint_set, "paint_color": (0.19, 0.21, 0.22), "chip_amount": 0.5,
                                     "rust_amount": 0.35}),
        "zone_steel": ("rig_hero_paint", {**paint_set, "paint_color": (0.09, 0.105, 0.11), "chip_amount": 0.8,
                                          "rust_amount": 0.65, "streak_amount": 0.9}),
    }


def spectral(n: int, beta: float, seed: int, lo: float = 1.0):
    import numpy as np

    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n) * n
    f = np.hypot(k[:, None], k[None, :])
    amp = np.where(f >= lo, np.maximum(f, 1.0) ** (-beta / 2.0), 0.0)
    img = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((n, n)))))
    lo_v, hi_v = np.percentile(img, (0.5, 99.5))
    return np.clip((img - lo_v) / (hi_v - lo_v), 0.0, 1.0)


def noise_rgb(n: int = 1024, seed: int = 11):
    import numpy as np

    return np.stack([spectral(n, 2.8, seed), spectral(n, 3.4, seed + 1, lo=2.0), spectral(n, 2.0, seed + 2)], -1)


def attach(stage) -> bool:
    if not LAYER.exists() or os.environ.get("RIG_HERO", "1") == "0":
        return False
    root = stage.GetRootLayer()
    if any(Path(p).name == LAYER.name for p in root.subLayerPaths):
        return True
    target = os.path.relpath(LAYER, Path(root.realPath).parent) if root.realPath else str(LAYER)
    root.subLayerPaths.insert(0, target)
    return True


def apply(stage, preset: str) -> dict:
    from pxr import Sdf, UsdShade

    if not attach(stage):
        return {}
    w = weather(preset)
    for path, kind in TARGETS.items():
        shader = UsdShade.Shader.Get(stage, f"{path}/{SHADER}")
        if not shader:
            continue
        shader.CreateInput("wetness", Sdf.ValueTypeNames.Float).Set(w["wetness"])
        if kind in WET_PUDDLES:
            shader.CreateInput("puddles", Sdf.ValueTypeNames.Float).Set(w["puddles"])
    return w
