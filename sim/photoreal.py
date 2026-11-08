from __future__ import annotations

import os
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
OUTPUT = SIM_DIR / "output"
ASSETS = SIM_DIR / "assets" / "third_party"
GENERATED = SIM_DIR / "assets" / "generated"
MDL = SIM_DIR / "assets" / "mdl" / "rig_weathered.mdl"
LAYER = OUTPUT / "photoreal.usda"
ROOTS = ("/World/Looks", "/World/RepairZone/Looks")
KINDS = ("paint", "textured", "tread", "grating", "concrete", "brushed")
PRESET_RAIN = {"night": 0.55, "storm": 1.0, "golden": 0.0}
OFFLINE_SCOPE = "/World/Photoreal/Offline"
WEATHER_SHADERS = ("rig_weathered", "rig_brushed", "rig_decal")

PAINT = "PaintedMetal010_4k"
RUST = "Rust007_4k"


def _paint(color, chip=0.6, rust=0.25, **extra):
    return {"kind": "paint", "paint_color": color, "chip_amount": chip, "rust_amount": rust, **extra}


def _textured(texture, rust=0.15, **extra):
    return {"kind": "textured", "texture": texture, "rust_amount": rust, **extra}


GREY = (0.27, 0.3, 0.32)
WHITE = (0.6, 0.62, 0.61)
YELLOW = (0.72, 0.46, 0.03)
BLUE = (0.07, 0.17, 0.3)
GREEN = (0.08, 0.2, 0.12)
ORANGE = (0.6, 0.2, 0.04)
RED = (0.42, 0.04, 0.03)
PIPE_GREY = (0.33, 0.35, 0.36)
BLACK = (0.035, 0.035, 0.04)

RECIPES = {
    "deck": {"kind": "tread", "texture": "DiamondPlate008C_4k", "rust_amount": 0.18, "metallic": 0.4, "scale": 1.6},
    "helideck": _paint((0.05, 0.16, 0.1), chip=0.25, rust=0.08, streak_amount=0.0),
    "concrete": {"kind": "concrete", "texture": "Concrete042C_4k", "rust_amount": 0.08, "scale": 3.0, "streak_amount": 0.75,
                 "salt_amount": 0.35},
    "splash_zone": {"kind": "concrete", "texture": "Concrete042C_4k", "rust_amount": 0.25, "scale": 2.0,
                    "tint": (0.22, 0.3, 0.22), "streak_amount": 0.4, "salt_amount": 0.5},
    "frame": _paint(GREY, chip=0.7, rust=0.35),
    "support_steel": _paint(GREY, chip=0.7, rust=0.4),
    "flare_steel": _paint(GREY, chip=0.8, rust=0.5),
    "topside_steel": _paint(GREY, chip=0.65, rust=0.3),
    "topside_steel_white": _paint(WHITE, chip=0.55, rust=0.22),
    "topside_steel_rust": _paint((0.32, 0.18, 0.1), chip=1.0, rust=0.85),
    "topside_black": _paint(BLACK, chip=0.3, rust=0.1),
    "rail": _paint(YELLOW, chip=0.75, rust=0.3),
    "crane": _paint(YELLOW, chip=0.6, rust=0.25),
    "topside_yellow": _paint(YELLOW, chip=0.7, rust=0.3),
    "safety_yellow": _paint(YELLOW, chip=0.7, rust=0.3),
    "walkway_line": _paint(YELLOW, chip=0.9, rust=0.1, streak_amount=0.0),
    "derrick": _paint(WHITE, chip=0.6, rust=0.3),
    "tank": _paint(WHITE, chip=0.45, rust=0.18),
    "module_white": _paint(WHITE, chip=0.45, rust=0.15),
    "topside_vessel_white": _paint(WHITE, chip=0.45, rust=0.2),
    "module_blue": _paint(BLUE, chip=0.5, rust=0.2),
    "vessel": _paint(BLUE, chip=0.5, rust=0.22),
    "topside_machine_blue": _paint(BLUE, chip=0.5, rust=0.2),
    "line_water": _paint(BLUE, chip=0.6, rust=0.25),
    "topside_machine_green": _paint(GREEN, chip=0.5, rust=0.2),
    "topside_pipe_green": _paint(GREEN, chip=0.6, rust=0.3),
    "pipe": _paint(ORANGE, chip=0.6, rust=0.3),
    "line_process": _paint(ORANGE, chip=0.6, rust=0.3),
    "pipe_orange": _paint(ORANGE, chip=0.6, rust=0.3),
    "topside_lifeboat": _paint((0.85, 0.3, 0.03), chip=0.3, rust=0.05, streak_amount=0.2),
    "line_gas": _paint(YELLOW, chip=0.6, rust=0.3),
    "topside_pipe_yellow": _paint(YELLOW, chip=0.6, rust=0.3),
    "topside_pipe_red": _paint(RED, chip=0.6, rust=0.3),
    "topside_valve_red": _paint(RED, chip=0.5, rust=0.25),
    "module_grey": _paint(PIPE_GREY, chip=0.5, rust=0.2),
    "topside_pipe_grey": _paint(PIPE_GREY, chip=0.6, rust=0.3),
    "topside_vessel_grey": _paint(PIPE_GREY, chip=0.5, rust=0.25),
    "pipe_grey": _paint(PIPE_GREY, chip=0.6, rust=0.3),
    "topside_galv": {"kind": "brushed", "base_color": (0.5, 0.52, 0.53), "roughness_along": 0.3, "roughness_across": 0.5},
    "topside_insulated": {"kind": "brushed", "base_color": (0.62, 0.64, 0.65), "roughness_along": 0.16,
                          "roughness_across": 0.4, "brush_axis": (0.0, 0.0, 1.0)},
    "topside_grating": {"kind": "grating", "texture": "MetalWalkway010_4k", "rust_amount": 0.25, "metallic": 0.6,
                        "scale": 1.0},
    "metal_grate_rusty": {"kind": "grating", "texture": "MetalWalkway010_4k", "rust_amount": 0.45, "metallic": 0.5,
                          "scale": 1.0},
    "worn_corrugated_iron": _textured("worn_corrugated_iron", rust=0.1, scale=3.0, metallic=0.3),
    "metal_plate": {"kind": "tread", "texture": "DiamondPlate008C_4k", "rust_amount": 0.25, "metallic": 0.4, "scale": 1.6},
    "rusty_metal_02": _textured("rusty_metal_02", rust=0.0, scale=1.5),
    "rusty_painted_metal": _textured("rusty_painted_metal_4k", rust=0.0, scale=2.0),
    "blue_metal_plate": _textured("blue_metal_plate", rust=0.15, scale=2.0, metallic=0.3),
}


def texture_file(folder: str, name: str) -> Path:
    base = ASSETS / "textures" / folder
    for ext in (".jpg", ".png"):
        if (base / f"{name}{ext}").exists():
            return base / f"{name}{ext}"
    return base / f"{name}.jpg"


def _set(folder: str, prefix: str) -> dict:
    return {f"{prefix}_diff": texture_file(folder, "diff"), f"{prefix}_rough": texture_file(folder, "rough"),
            f"{prefix}_normal": texture_file(folder, "nor_gl")}


COMMON = {
    "masks": GENERATED / "rig_masks.png",
    "streaks": GENERATED / "rig_streaks.png",
    "macro_scale": 23.0,
    "puddle_scale": 9.0,
    "wetness": 0.0,
    "puddles": 0.0,
    "triplanar": False,
    "paint_reference": 0.77,
}
OVERRIDES = ("grime_amount", "porosity", "normal_strength", "macro_scale", "streak_amount", "salt_amount")
STREAK_RUST = (0.15, 0.075, 0.035)
STREAK_DARK = (0.06, 0.05, 0.04)


def shader_inputs(recipe: dict) -> tuple[str, dict]:
    kind = recipe["kind"]
    if kind == "brushed":
        return "rig_brushed", {
            "base_color": recipe["base_color"],
            "roughness_along": recipe["roughness_along"],
            "roughness_across": recipe["roughness_across"],
            "brush_axis": recipe.get("brush_axis", (1.0, 0.0, 0.0)),
            "smear": texture_file("Metal049B", "rough"),
            "masks": COMMON["masks"],
            "wetness": 0.0,
        }
    values = {**COMMON, **_set(RUST, "rust"), "rust_scale": 1.5, "rust_amount": recipe.get("rust_amount", 0.2),
              "metallic": recipe.get("metallic", 0.0), "streak_amount": recipe.get("streak_amount", 0.5),
              "salt_amount": recipe.get("salt_amount", 0.15), "streak_color": STREAK_RUST, "use_opacity": False,
              "porosity": 0.35, "grime_amount": 0.35}
    if kind == "paint":
        values.update(_set(PAINT, "paint"), tint_paint=True, paint_color=recipe["paint_color"],
                      chip_amount=recipe.get("chip_amount", 0.6), paint_scale=recipe.get("scale", 2.0))
        values.update({k: recipe[k] for k in OVERRIDES if k in recipe})
        return "rig_weathered", values
    texture = recipe["texture"]
    values.update(_set(texture, "paint"), tint_paint=False, paint_color=recipe.get("tint", (1.0, 1.0, 1.0)),
                  chip_amount=0.0, paint_scale=recipe.get("scale", 2.0), porosity=1.0)
    if kind == "grating":
        values.update(use_opacity=True, opacity_tex=texture_file(texture, "opacity"), opacity_scale=recipe.get("scale", 1.0),
                      streak_amount=0.0)
    if kind == "concrete":
        values.update(streak_color=STREAK_DARK, porosity=1.0, grime_amount=0.5)
    if kind == "tread":
        values.update(streak_amount=0.0, porosity=0.6)
    values.update({k: recipe[k] for k in OVERRIDES if k in recipe})
    return "rig_weathered", values


def material_paths(name: str) -> list[str]:
    return [f"{root}/{name}" for root in ROOTS]


def weather(rain: float) -> dict:
    rain = min(max(float(rain), 0.0), 1.0)
    if rain <= 0.0:
        return {"wetness": 0.0, "puddles": 0.0}
    return {"wetness": round(0.2 + 0.8 * rain, 4), "puddles": round(min(max((rain - 0.15) / 0.85, 0.0), 1.0), 4)}


def rain_for(preset: str) -> float:
    return PRESET_RAIN.get(preset, 0.0)


def quality_settings(quality: str) -> dict:
    offline = quality == "offline"
    return {"triplanar": offline, "offline_props": offline}


def current_quality() -> str:
    return os.environ.get("RIG_QUALITY", "live")


def attach(stage) -> bool:
    if not LAYER.exists() or os.environ.get("RIG_PHOTOREAL", "1") == "0":
        return False
    root = stage.GetRootLayer()
    target = os.path.relpath(LAYER, Path(root.realPath).parent) if root.realPath else str(LAYER)
    if any(Path(p).name == LAYER.name for p in root.subLayerPaths):
        return True
    root.subLayerPaths.insert(0, target)
    return True


def rig_shaders(stage):
    from pxr import UsdShade

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        shader = UsdShade.Shader(prim)
        sub = shader.GetSourceAssetSubIdentifier("mdl")
        if sub in WEATHER_SHADERS:
            yield shader


def apply(stage, preset: str, quality: str | None = None) -> dict:
    from pxr import Sdf

    if not attach(stage):
        return {}
    w = weather(rain_for(preset))
    q = quality_settings(quality or current_quality())
    for shader in rig_shaders(stage):
        sub = shader.GetSourceAssetSubIdentifier("mdl")
        shader.CreateInput("wetness", Sdf.ValueTypeNames.Float).Set(w["wetness"])
        if sub == "rig_weathered":
            shader.CreateInput("puddles", Sdf.ValueTypeNames.Float).Set(w["puddles"])
            shader.CreateInput("triplanar", Sdf.ValueTypeNames.Bool).Set(q["triplanar"])
    offline = stage.GetPrimAtPath(OFFLINE_SCOPE)
    if offline.IsValid():
        offline.SetActive(q["offline_props"])
    return {**w, **q}
