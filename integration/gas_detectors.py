from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LEAK_META = REPO / "sim/output/leak_flanges_meta.json"
ROOT = "/World/GasDetectors"
STATES = {
    "ok": ((0.05, 1.0, 0.2), 30.0),
    "warn": ((1.0, 0.55, 0.02), 40.0),
    "alarm": ((1.0, 0.04, 0.02), 60.0),
    "fault": ((0.35, 0.36, 0.38), 0.0),
}
HOUSING = (0.85, 0.62, 0.03)
BRACKET = (0.32, 0.33, 0.34)
HEAD = (0.12, 0.12, 0.13)
MOUNT = (-0.24, 0.22, 0.10)
BOX = (0.10, 0.08, 0.13)
HEAD_R, HEAD_H = 0.022, 0.055


@dataclass
class Detector:
    id: str
    flange: str
    path: str
    frame: np.ndarray
    sample: np.ndarray
    state: str = "ok"


def flange_frame(center, normal) -> np.ndarray:
    x = np.asarray(normal, float)
    x = np.array([x[0], x[1], 0.0]) / np.linalg.norm(x[:2])
    z = np.array([0.0, 0.0, 1.0])
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, np.cross(z, x), z, np.asarray(center, float)
    return m


def layout(meta: dict, translation=(0.0, 0.0, 0.0), root: str = ROOT, mount=MOUNT) -> list[Detector]:
    out = []
    for f in meta["flanges"]:
        m = flange_frame(np.asarray(f["center_m"], float) + np.asarray(translation, float), f["normal"])
        r = float(f.get("pipe_radius_m", 0.08))
        local = np.array([mount[0], mount[1], r + mount[2], 1.0])
        frame = m.copy()
        frame[:3, 3] = (m @ local)[:3]
        tip = frame @ np.array([0.0, 0.0, -BOX[2] / 2 - HEAD_H, 1.0])
        out.append(Detector(f"GD-{f['id']}", f["id"], f"{root}/GD_{f['id']}", frame, tip[:3]))
    return out


def load_layout(meta_path=LEAK_META, translation=(0.0, 0.0, 0.0), root: str = ROOT) -> list[Detector]:
    return layout(json.loads(Path(meta_path).read_text()), translation, root)


class GasDetectors:

    def __init__(self, stage, meta_path=LEAK_META, translation=(0.0, 0.0, 0.0), root: str = ROOT, glow: bool = True):
        from pxr import Gf, Sdf, UsdGeom

        self.stage, self.glow = stage, glow
        self.detectors = {d.id: d for d in load_layout(meta_path, translation, root)}
        UsdGeom.Scope.Define(stage, root)
        self.mats = {k: self._material(f"{root}/Looks/led_{k}", (0.02, 0.02, 0.02), 0.3, tuple(c * e for c in rgb))
                     for k, (rgb, e) in STATES.items()}
        housing = self._material(f"{root}/Looks/housing", HOUSING, 0.5)
        bracket = self._material(f"{root}/Looks/bracket", BRACKET, 0.45, metal=0.8)
        head = self._material(f"{root}/Looks/head", HEAD, 0.7)
        for d in self.detectors.values():
            x = UsdGeom.Xform.Define(stage, d.path)
            x.AddTransformOp().Set(Gf.Matrix4d(*d.frame.T.reshape(-1)))
            prim = x.GetPrim()
            prim.CreateAttribute("rig:detectorId", Sdf.ValueTypeNames.String).Set(d.id)
            prim.CreateAttribute("rig:flangeId", Sdf.ValueTypeNames.String).Set(d.flange)
            prim.CreateAttribute("rig:samplePoint", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*map(float, d.sample)))
            prim.CreateAttribute("rig:state", Sdf.ValueTypeNames.String).Set(d.state)
            w, dp, h = BOX
            self._cube(f"{d.path}/housing", (0, 0, 0), BOX, housing)
            self._cube(f"{d.path}/cover", (w / 2 + 0.002, 0, 0.01), (0.004, dp * 0.8, h * 0.7), housing)
            self._cube(f"{d.path}/gland", (0, 0, h / 2 + 0.012), (0.03, 0.03, 0.024), bracket)
            self._cyl(f"{d.path}/head", (0, 0, -h / 2 - HEAD_H / 2), HEAD_R, HEAD_H, head)
            self._cyl(f"{d.path}/led", (0, 0, -h / 2 - 0.006), HEAD_R + 0.006, 0.008, self.mats["ok"])
            self._cyl(f"{d.path}/window", (w / 2 + 0.0045, 0, 0.03), 0.009, 0.003, self.mats["ok"], axis="X")
            reach = -MOUNT[1]
            self._cube(f"{d.path}/arm", (0, reach / 2, h / 2 + 0.03), (0.025, abs(reach) + 0.025, 0.012), bracket)
            self._cube(f"{d.path}/post", (0, reach, (h / 2 + 0.03 - MOUNT[2] - 0.02) / 2),
                       (0.025, 0.025, h / 2 + 0.03 + MOUNT[2] + 0.02), bracket)
            if glow:
                from pxr import UsdLux

                light = UsdLux.SphereLight.Define(stage, f"{d.path}/glow")
                light.CreateRadiusAttr(0.01)
                light.CreateIntensityAttr(0.0)
                UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(0, 0, -h / 2 - 0.02))
        for k in self.detectors:
            self.set_detector_state(k, "ok")

    def _material(self, path, color, rough, emissive=None, metal=0.0):
        from pxr import Gf, Sdf, UsdShade

        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, path + "/surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        if emissive is not None:
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
        mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        return mat

    def _bind(self, prim, mat):
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    def _cube(self, path, pos, size, mat):
        from pxr import Gf, UsdGeom

        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(c)
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
        self._bind(c.GetPrim(), mat)

    def _cyl(self, path, pos, radius, height, mat, axis="Z"):
        from pxr import Gf, UsdGeom

        c = UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateAxisAttr(axis)
        c.CreateRadiusAttr(radius)
        c.CreateHeightAttr(height)
        UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(*pos))
        self._bind(c.GetPrim(), mat)

    def positions(self) -> dict[str, np.ndarray]:
        return {k: d.sample.copy() for k, d in self.detectors.items()}

    def state(self, det_id: str) -> str:
        return self.detectors[self._key(det_id)].state

    def _key(self, det_id: str) -> str:
        return det_id if det_id in self.detectors else f"GD-{det_id}"

    def set_detector_state(self, det_id: str, state: str) -> None:
        from pxr import Gf, UsdLux

        if state not in STATES:
            raise ValueError(f"unknown detector state {state!r}; expected one of {sorted(STATES)}")
        d = self.detectors[self._key(det_id)]
        d.state = state
        prim = self.stage.GetPrimAtPath(d.path)
        prim.GetAttribute("rig:state").Set(state)
        for part in ("led", "window"):
            self._bind(self.stage.GetPrimAtPath(f"{d.path}/{part}"), self.mats[state])
        if self.glow:
            rgb, e = STATES[state]
            light = UsdLux.SphereLight(self.stage.GetPrimAtPath(f"{d.path}/glow"))
            light.GetColorAttr().Set(Gf.Vec3f(*rgb))
            light.GetIntensityAttr().Set(25.0 * e)
