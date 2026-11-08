from __future__ import annotations

import os

import numpy as np

from policy import night_look

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CINE = os.path.join(ROOT, "sim/output/cinematic.usda")
PIPE_GREY = (0.16, 0.17, 0.175)
BAND = (0.62, 0.36, 0.02)
STEEL = (0.09, 0.09, 0.095)


def frame(axis):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    e = np.cross(a, [0.0, 0.0, 1.0])
    e = e / np.linalg.norm(e) if np.linalg.norm(e) > 1e-6 else np.array([0.0, 1.0, 0.0])
    return a, e, np.cross(a, e)


def ring_mesh(stage, path, o, axis, r_in, r_out, width, material, n=64):
    from pxr import UsdGeom, UsdShade, Vt

    a, e, b = frame(axis)
    ph = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rad = np.cos(ph)[:, None] * e + np.sin(ph)[:, None] * b
    o = np.asarray(o, float)
    rings = [o + s * width / 2 * a + r * rad for s, r in ((-1, r_in), (-1, r_out), (1, r_out), (1, r_in))]
    pts = np.concatenate(rings)
    j = np.arange(n)
    quads = [np.stack([k * n + j, k * n + (j + 1) % n, ((k + 1) % 4) * n + (j + 1) % n, ((k + 1) % 4) * n + j], -1)
             for k in range(4)]
    quads = np.concatenate(quads)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * len(quads)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(quads.reshape(-1).astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def bolt(stage, path, centre, axis, radius, length, material):
    from pxr import Gf, UsdGeom, UsdShade

    a, e, b = frame(axis)
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(length)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = e, b, a, centre
    UsdGeom.Xformable(cyl).AddTransformOp().Set(Gf.Matrix4d(*m.T.reshape(-1)))
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(material)


def flange(stage, path, o, axis, r, material, bolts=8):
    a, e, b = frame(axis)
    ring_mesh(stage, f"{path}/hub", o, axis, r - 0.002, r + 0.02, 0.09, material)
    for s in (-1, 1):
        ring_mesh(stage, f"{path}/face_{'ab'[s > 0]}", np.asarray(o) + s * 0.016 * a, axis, r + 0.01, r + 0.075,
                  0.028, material)
    for k in range(bolts):
        ph = 2 * np.pi * (k + 0.5) / bolts
        c = np.asarray(o) + (r + 0.052) * (np.cos(ph) * e + np.sin(ph) * b)
        bolt(stage, f"{path}/bolt_{k}", c, axis, 0.009, 0.075, material)


def render_settings(z_shift: float) -> dict:
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(CINE)
    out = dict(layer.customLayerData.get("renderSettings", {})) if layer else {}
    if "rtx:fog:fogStartHeight" in out:
        out["rtx:fog:fogStartHeight"] = float(out["rtx:fog:fogStartHeight"]) + z_shift
    return out


def apply_settings(values: dict) -> None:
    import carb.settings

    s = carb.settings.get_settings()
    for key, v in values.items():
        path = "/" + key.replace(":", "/")
        if isinstance(v, bool):
            s.set_bool(path, v)
        elif isinstance(v, (int, float)):
            s.set_float(path, float(v))
        else:
            s.set_float_array(path, [float(c) for c in v])


def apply(stage, pipe_o, pipe_a, pipe_r, work, z_shift, robot_root=None, robot_color="graphite",
          backdrop="/World/Backdrop", lamp_gain=1.0, rig_gain=1.0, flanges=(-0.62, 0.62), bands=(-0.44, 0.44)):
    from pxr import Usd, UsdGeom, UsdLux, UsdShade

    settings = render_settings(z_shift)
    apply_settings(settings)
    if rig_gain != 1.0:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(backdrop + "/Cinematic")):
            if prim.IsA(UsdLux.SphereLight) or prim.IsA(UsdLux.RectLight):
                a = prim.GetAttribute("inputs:intensity")
                a.Set(a.Get() * rig_gain)
    grey = night_look._material(stage, "/World/Looks/rig_pipe_grey", PIPE_GREY, 0.55, 0.35, 0.15)
    band = night_look._material(stage, "/World/Looks/rig_pipe_band", BAND, 0.6, 0.0, 0.1)
    steel = night_look._material(stage, "/World/Looks/rig_flange", STEEL, 0.45, 0.8)
    n = 0
    for root in (backdrop + "/Pipes/main_pipe", backdrop + "/RepairZone/Pipes/main_pipe"):
        prim = stage.GetPrimAtPath(root)
        if not prim.IsValid():
            continue
        for p in Usd.PrimRange(prim):
            if p.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(grey, UsdShade.Tokens.strongerThanDescendants)
                n += 1
    o, a = np.asarray(pipe_o, float), np.asarray(pipe_a, float)
    UsdGeom.Scope.Define(stage, "/World/RigLook")
    for k, z in enumerate(flanges):
        flange(stage, f"/World/RigLook/flange_{k}", o + z * a, a, pipe_r, steel)
    for k, z in enumerate(bands):
        ring_mesh(stage, f"/World/RigLook/band_{k}", o + z * a, a, pipe_r + 0.0015, pipe_r + 0.0025, 0.07, band)
    wx, wy, wz = (float(v) for v in work)
    base = "/World/RigLook/lamps"
    UsdGeom.Scope.Define(stage, base)
    night_look._lamp(stage, f"{base}/work_spot", (wx + 1.2, wy + 1.8, wz + 1.6), night_look.LED, 0.08,
                     1.6e5 * lamp_gain, aim=work, cone=26.0)
    night_look._lamp(stage, f"{base}/deck_flood", (wx - 2.0, wy + 2.5, wz + 2.6), night_look.AMBER, 0.3,
                     2.0e4 * lamp_gain)
    night_look._lamp(stage, f"{base}/rim", (wx - 1.0, wy - 1.6, wz + 1.0), night_look.LED, 0.12, 2.0e4 * lamp_gain)
    painted = night_look.paint_robot(stage, robot_root, robot_color) if robot_root else {}
    return {"pipe_prims": n, "settings": sorted(settings), "robot": painted.get("body", 0)}
