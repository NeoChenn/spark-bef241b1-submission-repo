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


def _material(stage, name: str, tint, gain: float):
    path = f"{ROOT}/Looks/{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/shader")
    sh.SetSourceAsset(Sdf.AssetPath(str(MDL)), "mdl")
    sh.SetSourceAssetSubIdentifier("light_cone", "mdl")
    sh.CreateInput("tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, tint)))
    sh.CreateInput("gain", Sdf.ValueTypeNames.Float).Set(float(gain))
    sh.CreateInput("presence", Sdf.ValueTypeNames.Float).Set(PRESENCE)
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
            pts, nrm, s, counts, idx = shafts.cone(sp)
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
            mat = _material(stage, f"cone_{i}", tint, gain * sp["gain"])
            st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            st.Set(Vt.Vec2fArray.FromNumpy(np.zeros((len(pts), 2), np.float32)))
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
            self.items.append((m, sp, pts, nrm, s, st))

    def update(self, eye, t: float) -> None:
        for _, sp, pts, nrm, s, st in self.items:
            a = shafts.alpha(pts, nrm, s, eye, sp, t)
            st.Set(Vt.Vec2fArray.FromNumpy(np.stack((a, np.zeros_like(a)), -1).astype(np.float32)))

    def show(self, on: bool) -> None:
        for m, *_ in self.items:
            UsdGeom.Imageable(m).GetVisibilityAttr().Set("inherited" if on else "invisible")
