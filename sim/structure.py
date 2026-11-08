from __future__ import annotations

import os
from pathlib import Path

import photoreal as pr

LAYER = pr.OUTPUT / "structure.usda"
ROOT = "/World/Structure"
LOOKS = f"{ROOT}/Looks"
OFFLINE = f"{ROOT}/Offline"
GROWTH_TEX = pr.GENERATED / "marine_growth"
DRIP_TEX = pr.GENERATED / "rust_drip.png"
GROWTH_TOP, SPLASH_TOP = 1.7, 6.2
BAND_NAMES = ("growth", "splash", "column")
COLUMN = (0.29, 0.215, 0.07)

RECIPES = {
    "column": {"kind": "paint", "paint_color": COLUMN, "chip_amount": 0.0, "rust_amount": 0.3, "streak_amount": 1.0,
               "salt_amount": 0.3, "macro_scale": 19.0, "scale": 3.7},
    "seam": {"kind": "paint", "paint_color": COLUMN, "chip_amount": 0.9, "rust_amount": 0.62, "streak_amount": 1.0,
             "salt_amount": 0.2, "macro_scale": 4.0, "scale": 1.2},
    "splash": {"kind": "textured", "texture": "rusty_metal_02", "tint": (0.62, 0.45, 0.34), "rust_amount": 0.4,
               "scale": 2.2, "streak_amount": 0.9, "salt_amount": 0.55, "macro_scale": 7.0},
    "growth": {"kind": "textured", "texture": "rusty_metal_02", "tint": (0.75, 1.0, 0.6), "rust_amount": 0.06, "scale": 1.7, "streak_amount": 0.0,
               "salt_amount": 0.05, "macro_scale": 5.0, "normal_strength": 1.4},
    "brace": {"kind": "paint", "paint_color": (0.25, 0.19, 0.065), "chip_amount": 0.35, "rust_amount": 0.48,
              "streak_amount": 1.0, "salt_amount": 0.35, "macro_scale": 5.0, "scale": 1.6},
    "steel": {"kind": "paint", "paint_color": (0.2, 0.215, 0.22), "chip_amount": 0.7, "rust_amount": 0.42,
              "streak_amount": 0.9, "salt_amount": 0.25, "macro_scale": 9.0},
    "plate": {"kind": "paint", "paint_color": (0.16, 0.17, 0.17), "chip_amount": 0.6, "rust_amount": 0.45,
              "streak_amount": 0.6, "salt_amount": 0.2, "macro_scale": 6.0, "scale": 3.0},
    "riser": {"kind": "paint", "paint_color": (0.07, 0.07, 0.075), "chip_amount": 0.5, "rust_amount": 0.35,
              "streak_amount": 0.8, "salt_amount": 0.35, "macro_scale": 6.0},
    "caisson": {"kind": "paint", "paint_color": (0.5, 0.2, 0.05), "chip_amount": 0.6, "rust_amount": 0.4,
                "streak_amount": 0.9, "salt_amount": 0.3, "macro_scale": 6.0},
    "landing": {"kind": "paint", "paint_color": (0.72, 0.46, 0.03), "chip_amount": 0.75, "rust_amount": 0.42,
                "streak_amount": 0.8, "salt_amount": 0.4, "macro_scale": 5.0},
    "rubber": {"kind": "paint", "paint_color": (0.025, 0.025, 0.027), "chip_amount": 0.0, "rust_amount": 0.0,
               "streak_amount": 0.2, "salt_amount": 0.45, "macro_scale": 4.0},
    "anode": {"kind": "paint", "paint_color": (0.42, 0.42, 0.4), "chip_amount": 0.0, "rust_amount": 0.05,
              "streak_amount": 0.0, "salt_amount": 0.7, "macro_scale": 3.0},
    "grating": dict(pr.RECIPES["topside_grating"]),
    "flare_steel": {"kind": "paint", "paint_color": (0.27, 0.28, 0.28), "chip_amount": 0.5, "rust_amount": 0.35,
                    "streak_amount": 0.8, "salt_amount": 0.2, "macro_scale": 6.0},
    "heat": {"kind": "paint", "paint_color": (0.13, 0.1, 0.08), "chip_amount": 0.2, "rust_amount": 0.6,
             "streak_amount": 0.9, "salt_amount": 0.05, "macro_scale": 4.0},
    "soot": {"kind": "paint", "paint_color": (0.028, 0.026, 0.024), "chip_amount": 0.0, "rust_amount": 0.25,
             "streak_amount": 0.6, "salt_amount": 0.0, "macro_scale": 3.0},
    "helideck": {"kind": "paint", "paint_color": (0.035, 0.085, 0.055), "chip_amount": 0.15, "rust_amount": 0.04,
                 "streak_amount": 0.0, "salt_amount": 0.15, "macro_scale": 8.0, "scale": 3.0},
    "mark_white": {"kind": "paint", "paint_color": (0.6, 0.6, 0.58), "chip_amount": 0.4, "rust_amount": 0.02,
                   "streak_amount": 0.0, "salt_amount": 0.1, "macro_scale": 5.0},
    "mark_yellow": {"kind": "paint", "paint_color": (0.6, 0.4, 0.03), "chip_amount": 0.4, "rust_amount": 0.02,
                    "streak_amount": 0.0, "salt_amount": 0.1, "macro_scale": 5.0},
    "net": {"kind": "grating", "texture": "Net004A", "rust_amount": 0.05, "metallic": 0.0, "scale": 1.2},
    "lifeboat": {"kind": "paint", "paint_color": (0.8, 0.13, 0.01), "chip_amount": 0.1, "rust_amount": 0.03,
                 "streak_amount": 0.15, "salt_amount": 0.05, "macro_scale": 4.0},
    "lifeboat_white": {"kind": "paint", "paint_color": (0.58, 0.58, 0.56), "chip_amount": 0.2, "rust_amount": 0.04,
                       "streak_amount": 0.3, "salt_amount": 0.15, "macro_scale": 4.0},
}
WET = {"growth": 1.0, "splash": 0.9, "rubber": 0.8, "anode": 0.9}
DRY = {"helideck": 0.15, "mark_white": 0.15, "mark_yellow": 0.15, "net": 0.2}


def wetness_for(name: str, wet: float) -> float:
    if name in DRY:
        return min(wet, DRY[name])
    return max(wet, WET.get(name, 0.0))


def band_of(z: float) -> int:
    return 0 if z < GROWTH_TOP else (1 if z < SPLASH_TOP else 2)


def attach(stage) -> bool:
    if not LAYER.exists() or os.environ.get("RIG_STRUCTURE", "1") == "0":
        return False
    root = stage.GetRootLayer()
    if any(Path(p).name == LAYER.name for p in root.subLayerPaths):
        return True
    target = os.path.relpath(LAYER, Path(root.realPath).parent) if root.realPath else str(LAYER)
    root.subLayerPaths.insert(0, target)
    return True


def apply(stage, preset: str, quality: str | None = None) -> dict:
    from pxr import Sdf, UsdShade

    if not attach(stage):
        return {}
    w = pr.weather(pr.rain_for(preset))
    q = pr.quality_settings(quality or pr.current_quality())
    looks = stage.GetPrimAtPath(LOOKS)
    if looks.IsValid():
        for prim in looks.GetChildren():
            shader = UsdShade.Shader(stage.GetPrimAtPath(prim.GetPath().AppendChild("rig_mdl")))
            if not shader:
                continue
            shader.CreateInput("wetness", Sdf.ValueTypeNames.Float).Set(wetness_for(prim.GetName(), w["wetness"]))
            if shader.GetSourceAssetSubIdentifier("mdl") == "rig_weathered":
                shader.CreateInput("puddles", Sdf.ValueTypeNames.Float).Set(0.0 if prim.GetName() in DRY else w["puddles"])
                shader.CreateInput("triplanar", Sdf.ValueTypeNames.Bool).Set(q["triplanar"])
    offline = stage.GetPrimAtPath(OFFLINE)
    if offline.IsValid():
        offline.SetActive(q["offline_props"])
    return {**w, **q}
