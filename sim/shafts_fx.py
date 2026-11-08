from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

import look
import shafts
from topside_frames import kelvin_rgb

ROOT = "/World/Cinematic/Shafts"
PRESENCE = 0.5
GAIN = 6.0


def _texture() -> Path:
    path = Path(tempfile.gettempdir()) / "shaft_ramp.png"
    if not path.exists():
        ramp = np.full((4, 256, 4), 255, np.uint8)
        ramp[..., 3] = np.rint(np.linspace(0.0, 255.0 * PRESENCE, 256)).astype(np.uint8)
        Image.fromarray(ramp).save(path)
    return path


def _material(stage):
    path = f"{ROOT}/Looks/cone"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.001)
    glow = UsdShade.Shader.Define(stage, f"{path}/glow")
    glow.CreateIdAttr("UsdPrimvarReader_float3")
    glow.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        glow.CreateOutput("result", Sdf.ValueTypeNames.Float3))
    st = UsdShade.Shader.Define(stage, f"{path}/uv")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{path}/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(_texture()))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.CreateOutput("result", Sdf.ValueTypeNames.Float2))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(tex.CreateOutput("a", Sdf.ValueTypeNames.Float))
    mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return mat


class Shafts:
    def __init__(self, stage, tier: str, area, sea, gain: float = GAIN):
        self.items, self.gain = [], gain
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
            st = pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            st.Set(Vt.Vec2fArray.FromNumpy(np.full((len(pts), 2), 0.5, np.float32)))
            color = np.asarray(kelvin_rgb(look.KELVIN.get(sp["kelvin"], sp["kelvin"])), np.float32) * sp["gain"] * gain
            pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant).Set(
                Vt.Vec3fArray.FromNumpy(color[None]))
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
            self.items.append((m, sp, pts, nrm, s, st))
        self.on = True

    def update(self, eye, t: float) -> None:
        for m, sp, pts, nrm, s, st in self.items:
            hide = shafts.inside(eye, sp)
            UsdGeom.Imageable(m).GetVisibilityAttr().Set("invisible" if hide or not self.on else "inherited")
            a = shafts.alpha(pts, nrm, s, eye, sp, t)
            st.Set(Vt.Vec2fArray.FromNumpy(np.stack((a, np.full_like(a, 0.5)), -1).astype(np.float32)))

    def show(self, on: bool) -> None:
        self.on = on
        for m, *_ in self.items:
            UsdGeom.Imageable(m).GetVisibilityAttr().Set("inherited" if on else "invisible")
