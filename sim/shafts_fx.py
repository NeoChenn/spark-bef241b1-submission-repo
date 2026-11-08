from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

import look
import shafts
from topside_frames import kelvin_rgb

ROOT = "/World/Cinematic/Shafts"
MDL = Path(__file__).resolve().parent / "assets" / "mdl" / "light_cone.mdl"
PRESENCE = 0.6
GAIN = 6.0


def _material(stage, name: str, tint, gain: float):
    path = f"{ROOT}/Looks/{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/shader")
    sh.SetSourceAsset(Sdf.AssetPath(str(MDL)), "mdl")
    sh.SetSourceAssetSubIdentifier("light_cone", "mdl")
    sh.CreateInput("tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, tint)))
    sh.CreateInput("gain", Sdf.ValueTypeNames.Float).Set(float(gain))
    sh.CreateInput("presence", Sdf.ValueTypeNames.Float).Set(PRESENCE)
    sh.CreateInput("time", Sdf.ValueTypeNames.Float).Set(0.0)
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateSurfaceOutput().ConnectToSource(out)
    return sh, mat


class Shafts:
    def __init__(self, stage, tier: str, area, sea, gain: float = GAIN):
        self.items, self.on = [], True
        specs = shafts.specs(area, sea, tier)
        if not specs:
            return
        UsdGeom.Scope.Define(stage, ROOT)
        counts, idx, st = shafts.topology()
        for i, sp in enumerate(specs):
            m = UsdGeom.Mesh.Define(stage, f"{ROOT}/beam_{i}")
            m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(shafts.card(sp, (0.0, 0.0, 1e4)).astype(np.float32)))
            m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
            m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(idx))
            m.CreateDoubleSidedAttr(True)
            m.CreateSubdivisionSchemeAttr("none")
            for flag in ("doNotCastShadows", "invisibleToSecondaryRays"):
                m.GetPrim().CreateAttribute(f"primvars:{flag}", Sdf.ValueTypeNames.Bool).Set(True)
            UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
                Vt.Vec2fArray.FromNumpy(st.astype(np.float32)))
            tint = kelvin_rgb(look.KELVIN.get(sp["kelvin"], sp["kelvin"]))
            sh, mat = _material(stage, f"beam_{i}", tint, gain * sp["gain"])
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
            self.items.append((m, sp, sh))

    def update(self, eye, t: float) -> None:
        for m, sp, sh in self.items:
            hide = not self.on or shafts.inside(eye, sp)
            UsdGeom.Imageable(m).GetVisibilityAttr().Set("invisible" if hide else "inherited")
            m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(shafts.card(sp, eye).astype(np.float32)))
            sh.GetInput("time").Set(float(t))

    def show(self, on: bool) -> None:
        self.on = on
        for m, *_ in self.items:
            UsdGeom.Imageable(m).GetVisibilityAttr().Set("inherited" if on else "invisible")
