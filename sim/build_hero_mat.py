from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hero_mat as hm
from photoreal_usd import as_overs
from scene_builder import OUTPUT, rel

COLORS = {"paint_tint", "paint_color", "primer_color", "tint"}


def set_inputs(shader, values: dict) -> None:
    for key, value in values.items():
        if isinstance(value, Path):
            attr = shader.CreateInput(key, Sdf.ValueTypeNames.Asset)
            attr.Set(Sdf.AssetPath(rel(value)))
            attr.GetAttr().SetColorSpace("sRGB" if key.endswith("_diff") else "raw")
        elif isinstance(value, (int, float)):
            shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(float(value))
        elif key in COLORS:
            shader.CreateInput(key, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*value))
        else:
            shader.CreateInput(key, Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*value))


def bind(stage, path: str, sub: str, values: dict) -> None:
    prim = stage.OverridePrim(path)
    shader = UsdShade.Shader.Define(stage, f"{path}/{hm.SHADER}")
    shader.SetSourceAsset(Sdf.AssetPath(rel(hm.MDL)), "mdl")
    shader.SetSourceAssetSubIdentifier(sub, "mdl")
    set_inputs(shader, values)
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material = UsdShade.Material(prim)
    material.CreateSurfaceOutput("mdl").ConnectToSource(out)
    material.CreateSurfaceOutput().ConnectToSource(out)


def patch_points() -> int:
    deck = Usd.Stage.Open(str(OUTPUT / "rig_deck.usda"))
    return len(UsdGeom.Mesh(deck.GetPrimAtPath("/World/Pipes/damage_patch")).GetPointsAttr().Get())


def write_noise() -> None:
    Image.fromarray((hm.noise_rgb() * 255).round().astype(np.uint8), "RGB").save(hm.NOISE, optimize=True)


def build(preset: str = "night") -> Path:
    if not hm.NOISE.exists():
        write_noise()
    meta = json.loads((OUTPUT / "rig_merged_meta.json").read_text())
    recipes = hm.recipes(meta)
    w = hm.weather(preset)
    if hm.LAYER.exists():
        hm.LAYER.unlink()
    stage = Usd.Stage.CreateNew(str(hm.LAYER))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for path, kind in hm.TARGETS.items():
        sub, values = recipes[kind]
        extra = {"wetness": w["wetness"], **({"puddles": w["puddles"]} if kind == "deck" else {})}
        bind(stage, path, sub, {**values, **extra})
    geom = stage.OverridePrim(hm.PATCH_GEOM)
    st = UsdGeom.PrimvarsAPI(geom).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in hm.patch_st(patch_points())]))
    as_overs(stage.GetRootLayer())
    stage.GetRootLayer().Save()
    print(f"HERO_WRITTEN {hm.LAYER} materials={len(hm.TARGETS)}")
    return hm.LAYER


if __name__ == "__main__":
    build()
