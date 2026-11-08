from __future__ import annotations

import math
import sys
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdShade

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]

from sim import ocean_fx as fx

SCENES = ("rig_merged.usda", "cinematic.usda")


def scene_legs(stage) -> list[tuple[float, float, float]]:
    legs = []
    for prim in stage.Traverse():
        if "splash_band" in prim.GetName():
            t = prim.GetAttribute("xformOp:translate").Get()
            first = prim.GetAttribute("points").Get()[0]
            legs.append((float(t[0]), float(t[1]), float(math.hypot(first[1], first[2]))))
    return legs


def foam_decals(stage) -> list[str]:
    scope = stage.GetPrimAtPath("/World/Cinematic")
    return [str(c.GetPath()) for c in scope.GetChildren() if c.GetName().startswith("foam_")] if scope else []


def build() -> Path:
    scene = next(fx.OUTPUT / s for s in SCENES if (fx.OUTPUT / s).exists())
    base = Usd.Stage.Open(str(scene))
    legs = scene_legs(base)
    decals = foam_decals(base)
    fx.MDL_FILE.write_text(fx.build_mdl(fx.big_waves(), fx.detail_waves(), legs))
    if fx.LAYER.exists():
        fx.LAYER.unlink()
    stage = Usd.Stage.CreateNew(str(fx.LAYER))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for path in ("/World", "/World/Cinematic"):
        stage.OverridePrim(path)
    UsdGeom.Scope.Define(stage, "/World/OceanFx")
    UsdGeom.Scope.Define(stage, "/World/OceanFx/Looks")
    material = UsdShade.Material.Define(stage, fx.MATERIAL)
    shader = UsdShade.Shader.Define(stage, fx.SHADER)
    shader.SetSourceAsset(Sdf.AssetPath("./" + fx.MDL_FILE.name), "mdl")
    shader.SetSourceAssetSubIdentifier(fx.MODULE, "mdl")
    shader.CreateInput("uv_tile", Sdf.ValueTypeNames.Float).Set(fx.UV_TILE_M)
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(out)
    material.CreateSurfaceOutput().ConnectToSource(out)
    ocean = stage.OverridePrim(fx.OCEAN)
    UsdShade.MaterialBindingAPI.Apply(ocean).Bind(material)
    for path in decals:
        stage.OverridePrim(path).SetActive(False)
    stage.GetRootLayer().GetPrimAtPath("/World").specifier = Sdf.SpecifierOver
    stage.GetRootLayer().Save()
    print(f"OCEAN_FX_WRITTEN {fx.LAYER} legs={len(legs)} decals={len(decals)}")
    return fx.LAYER


if __name__ == "__main__":
    build()
