from __future__ import annotations

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

import look
import shafts
from topside_frames import kelvin_rgb

ROOT = "/World/Cinematic/Shafts"
OPACITY = 0.05
GAIN = 20.0


def _material(stage):
    path = f"{ROOT}/Looks/cone"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(OPACITY)
    rd = UsdShade.Shader.Define(stage, f"{path}/glow")
    rd.CreateIdAttr("UsdPrimvarReader_float3")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        rd.CreateOutput("result", Sdf.ValueTypeNames.Float3))
    mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return mat


class Shafts:
    def __init__(self, stage, tier: str, area, sea, gain: float = GAIN):
        self.items, self.stage = [], stage
        specs = shafts.specs(area, sea, tier)
        if not specs:
            return
        UsdGeom.Scope.Define(stage, ROOT)
        mat = _material(stage)
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
            pv = UsdGeom.PrimvarsAPI(m)
            color = np.asarray(kelvin_rgb(look.KELVIN.get(sp["kelvin"], sp["kelvin"])), np.float32) * sp["gain"]
            glow = pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
            glow.Set(Vt.Vec3fArray.FromNumpy(np.zeros((len(pts), 3), np.float32)))
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
            self.items.append((m, sp, pts, nrm, s, glow, color))
        self.gain = gain

    def update(self, eye, t: float) -> None:
        k = self.gain / OPACITY
        for _, sp, pts, nrm, s, glow, color in self.items:
            a = shafts.alpha(pts, nrm, s, eye, sp, t)[:, None] * color[None] * k
            glow.Set(Vt.Vec3fArray.FromNumpy(a.astype(np.float32)))

    def show(self, on: bool) -> None:
        for m, *_ in self.items:
            UsdGeom.Imageable(m).GetVisibilityAttr().Set("inherited" if on else "invisible")
