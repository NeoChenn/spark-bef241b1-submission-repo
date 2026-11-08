from __future__ import annotations

from pathlib import Path

from pxr import Gf, Sdf, UsdShade

import photoreal as pr
from scene_builder import rel

COLOR_INPUTS = {"paint_color", "rust_tint", "streak_color", "salt_color", "base_color", "tint"}
SRGB_INPUTS = {"paint_diff", "rust_diff", "color_tex"}


def mdl_shader(stage, path: str, sub: str) -> UsdShade.Shader:
    shader = UsdShade.Shader.Define(stage, path)
    shader.SetSourceAsset(Sdf.AssetPath(rel(pr.MDL)), "mdl")
    shader.SetSourceAssetSubIdentifier(sub, "mdl")
    return shader


def set_inputs(shader: UsdShade.Shader, values: dict) -> None:
    for key, value in values.items():
        if isinstance(value, Path):
            attr = shader.CreateInput(key, Sdf.ValueTypeNames.Asset)
            attr.Set(Sdf.AssetPath(rel(value)))
            attr.GetAttr().SetColorSpace("sRGB" if key in SRGB_INPUTS else "raw")
        elif isinstance(value, bool):
            shader.CreateInput(key, Sdf.ValueTypeNames.Bool).Set(value)
        elif isinstance(value, (int, float)):
            shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(float(value))
        elif key in COLOR_INPUTS:
            shader.CreateInput(key, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*value))
        else:
            shader.CreateInput(key, Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*value))


def bind_mdl(stage, material_path: str, sub: str, values: dict, define: bool = False) -> UsdShade.Material:
    prim = UsdShade.Material.Define(stage, material_path).GetPrim() if define else stage.OverridePrim(material_path)
    shader = mdl_shader(stage, f"{material_path}/rig_mdl", sub)
    set_inputs(shader, values)
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material = UsdShade.Material(prim)
    material.CreateSurfaceOutput("mdl").ConnectToSource(out)
    material.CreateSurfaceOutput().ConnectToSource(out)
    return material


def as_overs(layer: Sdf.Layer, keep=()) -> None:
    def visit(path):
        spec = layer.GetPrimAtPath(path)
        if spec is None or any(str(path).startswith(k) for k in keep):
            return
        if spec.typeName != "Shader":
            spec.specifier = Sdf.SpecifierOver
        for child in spec.nameChildren:
            visit(child.path)

    for root in layer.rootPrims:
        visit(root.path)
