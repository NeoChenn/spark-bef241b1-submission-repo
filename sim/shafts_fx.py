from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

import look
import shafts
from topside_frames import kelvin_rgb

ROOT = "/World/Cinematic/Shafts"
MDL = Path(__file__).resolve().parent / "assets" / "mdl" / "light_cone.mdl"
PRESENCE = 0.3
GAIN = 2.0


def _material(stage, name: str, tint, gain: float, sp: dict):
    path = f"{ROOT}/Looks/{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/shader")
    sh.SetSourceAsset(Sdf.AssetPath(str(MDL)), "mdl")
    sh.SetSourceAssetSubIdentifier("light_cone", "mdl")
    sh.CreateInput("tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, tint)))
    sh.CreateInput("gain", Sdf.ValueTypeNames.Float).Set(float(gain))
    sh.CreateInput("presence", Sdf.ValueTypeNames.Float).Set(PRESENCE)
    sh.CreateInput("apex", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*sp["apex"]))
    sh.CreateInput("axis", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*sp["aim"]))
    sh.CreateInput("length", Sdf.ValueTypeNames.Float).Set(sp["length"])
    for k in ("visible", "time"):
        sh.CreateInput(k, Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("eye", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateSurfaceOutput().ConnectToSource(out)
    return mat


class Shafts:
    def __init__(self, stage, tier: str, area, sea, gain: float = GAIN):
        self.items = []
        specs = shafts.specs(area, sea, tier)
        if not specs:
            return
        UsdGeom.Scope.Define(stage, ROOT)
        for i, sp in enumerate(specs):
            pts, nrm, _, counts, idx = shafts.cone(sp)
            m = UsdGeom.Mesh.Define(stage, f"{ROOT}/cone_{i}")
            m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
            m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nrm.astype(np.float32)))
            m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts.astype(np.int32)))
            m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(idx))
            m.CreateDoubleSidedAttr(True)
            m.CreateSubdivisionSchemeAttr("none")
            for flag in ("doNotCastShadows", "invisibleToSecondaryRays"):
                m.GetPrim().CreateAttribute(f"primvars:{flag}", Sdf.ValueTypeNames.Bool).Set(True)
            tint = kelvin_rgb(look.KELVIN.get(sp["kelvin"], sp["kelvin"]))
            mat = _material(stage, f"cone_{i}", tint, gain * sp["gain"], sp)
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
            self.items.append((m, sp, UsdShade.Shader(stage.GetPrimAtPath(f"{ROOT}/Looks/cone_{i}/shader"))))

    def update(self, eye, t: float) -> None:
        for _, sp, sh in self.items:
            sh.GetInput("eye").Set(Gf.Vec3f(*map(float, eye)))
            sh.GetInput("time").Set(float(t))
            sh.GetInput("visible").Set(0.0 if shafts.inside(eye, sp) else 1.0)

    def show(self, on: bool) -> None:
        for m, *_ in self.items:
            UsdGeom.Imageable(m).GetVisibilityAttr().Set("inherited" if on else "invisible")
