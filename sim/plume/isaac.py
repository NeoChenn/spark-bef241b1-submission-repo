import math

import numpy as np

from sim.plume.colormap import VISIBLE_PCT, lel_alpha, lel_rgb

ROOT = "/World/Plume"


def _material(stage, path: str, emissive: float):
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    reads = {}
    for name, typ, out in (("displayColor", "UsdPrimvarReader_float3", Sdf.ValueTypeNames.Float3),
                           ("displayOpacity", "UsdPrimvarReader_float", Sdf.ValueTypeNames.Float)):
        r = UsdShade.Shader.Define(stage, f"{path}/{name}")
        r.CreateIdAttr(typ)
        r.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(name)
        reads[name] = r.CreateOutput("result", out)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(reads["displayColor"])
    if emissive > 0:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(reads["displayColor"])
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(reads["displayOpacity"])
    sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
    sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def cloud(field, max_points: int = 60000, floor: float = VISIBLE_PCT, seed: int = 0):
    pct = field.fraction / field.lel * 100.0
    flat = pct.ravel()
    idx = np.flatnonzero(flat >= floor)
    if len(idx) > max_points:
        idx = idx[np.argpartition(flat[idx], -max_points)[-max_points:]]
    ijk = np.stack(np.unravel_index(idx, pct.shape), 1)
    off = np.modf(np.abs(np.sin((idx[:, None] + seed) * np.array([12.9898, 78.233, 37.719])) * 43758.5453))[0] - 0.5
    pts = field.origin + (ijk + 0.5 + 0.6 * off) * field.dx
    p = flat[idx]
    return pts.astype(np.float32), lel_rgb(p).astype(np.float32), lel_alpha(p).astype(np.float32), p


class PlumeVisual:
    def __init__(self, stage, plume, path: str = ROOT + "/Cloud", max_points: int = 60000, size: float = 1.7,
                 emissive: float = 1.0, opacity: float = 1.0):
        from pxr import UsdGeom, UsdShade

        self.stage, self.plume, self.max_points, self.opacity = stage, plume, max_points, opacity
        self.width = size * plume.dx
        self.prim = UsdGeom.Points.Define(stage, path)
        api = UsdGeom.PrimvarsAPI(self.prim)
        from pxr import Sdf

        self.color = api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        self.alpha = api.CreatePrimvar("displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
        mat = _material(stage, ROOT + "/Looks/gas", emissive)
        UsdShade.MaterialBindingAPI.Apply(self.prim.GetPrim()).Bind(mat)
        self.count = 0
        self.update()

    def update(self) -> int:
        from pxr import Vt

        pts, rgb, a, _ = cloud(self.plume.field(), self.max_points)
        if not len(pts):
            pts, rgb, a = np.full((1, 3), -1e4, np.float32), np.zeros((1, 3), np.float32), np.zeros(1, np.float32)
        self.prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))
        self.prim.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(np.full(len(pts), self.width, np.float32)))
        self.color.Set(Vt.Vec3fArray.FromNumpy(rgb))
        self.alpha.Set(Vt.FloatArray.FromNumpy(np.clip(a * self.opacity, 0, 1).astype(np.float32)))
        self.count = len(pts)
        return self.count


def analysis_pose(plume, side: float = 1.0, distance: float = 7.5, height: float = 3.2, ahead: float = 2.5):
    w = plume.wind.at(plume.t)
    h = math.atan2(w[1], w[0]) if np.linalg.norm(w[:2]) > 1e-6 else math.radians(plume.wind.heading_deg)
    d = np.array([math.cos(h), math.sin(h), 0.0])
    perp = np.array([-d[1], d[0], 0.0]) * side
    s = plume.source
    target = s + d * ahead + np.array([0.0, 0.0, 0.2])
    eye = s + perp * distance + d * (ahead * 0.6) + np.array([0.0, 0.0, height])
    return tuple(eye), tuple(target)


def make_camera(stage, eye, target, path: str = ROOT + "/AnalysisCamera", focal: float = 18.0):
    from pxr import Gf, UsdGeom

    from rig.sim.orbit import look_at_rows

    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * 9 / 16)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5000.0))
    op = UsdGeom.Xformable(cam).AddTransformOp()
    op.Set(Gf.Matrix4d(look_at_rows(tuple(map(float, eye)), tuple(map(float, target)))))
    return path, op
