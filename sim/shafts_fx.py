from __future__ import annotations

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

import look
import shafts
from topside_frames import kelvin_rgb

ROOT = "/World/Cinematic/Shafts"
OPACITY = 0.3
GAIN = 20.0


def _material(stage):
    path = f"{ROOT}/Looks/cone"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    for name, kind, out in (("glow", "UsdPrimvarReader_float3", "emissiveColor"),
                            ("fade", "UsdPrimvarReader_float", "opacity")):
        rd = UsdShade.Shader.Define(stage, f"{path}/{name}")
        rd.CreateIdAttr(kind)
        var = "displayColor" if name == "glow" else "displayOpacity"
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(var)
        vt = Sdf.ValueTypeNames.Float3 if name == "glow" else Sdf.ValueTypeNames.Float
        target = Sdf.ValueTypeNames.Color3f if name == "glow" else Sdf.ValueTypeNames.Float
        sh.CreateInput(out, target).ConnectToSource(rd.CreateOutput("result", vt))
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
            color = np.asarray(kelvin_rgb(look.KELVIN.get(sp["kelvin"], sp["kelvin"])), np.float32) * gain * sp["gain"]
            pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex).Set(
                Vt.Vec3fArray.FromNumpy(np.tile(color, (len(pts), 1))))
            fade = pv.CreatePrimvar("displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
            fade.Set(Vt.FloatArray.FromNumpy(np.zeros(len(pts), np.float32)))
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
            self.items.append((m, sp, pts, nrm, s, fade))

    def update(self, eye, t: float) -> None:
        for _, sp, pts, nrm, s, fade in self.items:
            fade.Set(Vt.FloatArray.FromNumpy((shafts.alpha(pts, nrm, s, eye, sp, t) * OPACITY).astype(np.float32)))

    def show(self, on: bool) -> None:
        for m, *_ in self.items:
            UsdGeom.Imageable(m).GetVisibilityAttr().Set("inherited" if on else "invisible")
