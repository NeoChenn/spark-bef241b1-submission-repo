from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

SIM = Path(__file__).resolve().parent
REPO = SIM.parent
sys.path[:0] = [str(REPO), str(SIM)]

from integration.kinematics import ARM_JOINTS, LIFT_JOINT, RobotModel, pose
from rig.sim.workzone import ZONE_TRANSLATION, merged_meta, to_world
from scene_builder import OUTPUT, SceneBuilder

SCENE = OUTPUT / "rig_merged.usda"
META = OUTPUT / "rig_merged_meta.json"
DECK = "./rig_deck.usda"
ROBOT = "./robot/go2_ur3_lift/go2_ur3_lift.usda"
ZONE = "/World/RepairZone"
WIDE_TARGET = (4.0, -4.0, 30.0)


def look(stage, path, eye, target, focal, clip=(0.05, 60000.0)):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(float(focal))
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateClippingRangeAttr(Gf.Vec2f(*clip))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(0, 0, 1))
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(view.GetInverse())
    return path


def ref(stage, path, source, kind="Scope"):
    prim = stage.DefinePrim(path, kind)
    prim.GetReferences().AddReference(DECK, Sdf.Path(source))
    return prim


def translated(stage, path, t):
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*t))
    return xf.GetPrim()


def add_zone(stage, deck_stage, t):
    translated(stage, ZONE, t)
    for group in ("Structure", "Pipes", "Props"):
        ref(stage, f"{ZONE}/{group}", f"/World/{group}")
    moved = {}
    for child in deck_stage.GetPrimAtPath("/World/Pipes").GetChildren():
        name = child.GetName()
        for key in ("main_pipe", "damage_patch"):
            if name.startswith(key):
                stage.OverridePrim(f"{ZONE}/Pipes/{name}").SetActive(False)
                moved[key] = str(child.GetPath())
    UsdGeom.Scope.Define(stage, "/World/Pipes")
    for key, source in moved.items():
        translated(stage, f"/World/Pipes/{key}", t)
        ref(stage, f"/World/Pipes/{key}/geom", source, "Mesh")
    b = SceneBuilder(stage)
    paint = b.plain("zone_marking", (0.9, 0.62, 0.04), 0.5, 0.0)
    x0, x1 = (-2.9, 2.9)
    y0, y1 = (-2.5, 2.5)
    w = 0.1
    for name, c, s in (("mark", ((x0 + x1) / 2, y0, 0.003), (x1 - x0, w, 0.006)),
                       ("mark", ((x0 + x1) / 2, y1, 0.003), (x1 - x0, w, 0.006)),
                       ("mark", (x0, 0, 0.003), (w, y1 - y0, 0.006)),
                       ("mark", (x1, 0, 0.003), (w, y1 - y0, 0.006))):
        b.box(f"{ZONE}/Marking", name, c, s, paint, collide=None)
    return moved


def add_looks(stage, deck_stage):
    for child in deck_stage.GetPrimAtPath("/World/Looks").GetChildren():
        path = f"/World/Looks/{child.GetName()}"
        if not stage.GetPrimAtPath(path).IsValid():
            ref(stage, path, str(child.GetPath()), "Material")


def write_pipe_attrs(prim, w):
    p = w["main_pipe"]
    for name, kind, value in (
        ("rig:axis", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*p["axis"])),
        ("rig:ref", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*p["ref"])),
        ("rig:radius", Sdf.ValueTypeNames.Float, float(p["radius_m"])),
        ("rig:segmentOrigin", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*w["segment_origin_m"])),
        ("rig:segmentLength", Sdf.ValueTypeNames.Float, float(w["segment_length_m"])),
    ):
        prim.CreateAttribute(name, kind, custom=True).Set(value)


def robot_q(model, base, damage, normal):
    q = model.default_q()
    target = np.asarray(damage) + 0.15 * np.asarray(normal)
    best = None
    for lift in (0.42, 0.3, 0.45, 0.2):
        seed = dict(q, **{LIFT_JOINT: lift})
        r = model.solve_ik(base, seed, target, -np.asarray(normal), joints=ARM_JOINTS, iterations=150)
        if best is None or r.position_error < best.position_error:
            best = r
    return best.q, best.position_error


def add_robot(stage, w):
    model = RobotModel.from_urdf()
    spawn = w["robot_spawn"]["position_m"]
    base = pose(*spawn, w["robot_spawn"]["yaw_rad"])
    q, err = robot_q(model, base, w["damage"]["position_m"], w["damage"]["surface_normal"])
    prim = stage.DefinePrim("/World/Robots/scrubber_dog", "Xform")
    prim.GetReferences().AddReference(ROBOT)
    prim.GetVariantSets().AddVariantSet("Physics").SetVariantSelection("none")
    root = UsdGeom.Xformable(prim)
    root.ClearXformOpOrder()
    root.AddTransformOp().Set(Gf.Matrix4d(*base.T.reshape(-1)))
    by_child = {j.child: j for j in model.joints.values()}
    seen = set()
    for p in Usd.PrimRange(prim):
        name, parent = p.GetName(), p.GetParent().GetName()
        if name in seen:
            continue
        if name == model.root and parent == "Geometry":
            local = np.eye(4)
        elif name in by_child and by_child[name].parent == parent:
            local = by_child[name].local(q.get(by_child[name].name, 0.0))
        else:
            continue
        seen.add(name)
        xf = UsdGeom.Xformable(p)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Gf.Matrix4d(*local.T.reshape(-1)))
    return {"prim": "/World/Robots/scrubber_dog", "usd": ROBOT, "links_posed": len(seen), "brush_ik_error_m": round(err, 4),
            "q": {k: round(float(v), 4) for k, v in q.items()}}


def add_cameras(stage, w, t):
    d = np.asarray(w["damage"]["position_m"])
    n = np.asarray(w["damage"]["surface_normal"])
    sx = w["segment_origin_m"][0]
    return {
        "repair_closeup": look(stage, "/World/Cameras/repair_closeup", d + 0.85 * n + [0.15, 0, 0], d, 35),
        "hero": look(stage, "/World/Cameras/hero", to_world((sx - t[0] + 2.6, 2.9, 2.15), t),
                     to_world((sx - t[0] - 0.1, 0.25, 0.95), t), 24),
        "deck_mid": look(stage, "/World/Cameras/deck_mid", to_world((6.5, 9.0, 4.5), t), to_world((-0.5, 0.0, 1.0), t), 22),
        "zone_wide": look(stage, "/World/Cameras/zone_wide", (95.0, 75.0, 75.0), to_world((0.0, 0.0, 1.0), t), 30),
        "rig_wide": look(stage, "/World/Cameras/rig_wide", (270.0, -200.0, 110.0), WIDE_TARGET, 28),
    }


def build(t=ZONE_TRANSLATION, robot=True):
    deck_meta = json.loads((OUTPUT / "rig_deck_meta.json").read_text())
    deck_stage = Usd.Stage.Open(str(OUTPUT / "rig_deck.usda"))
    cine = Sdf.Layer.FindOrOpen(str(OUTPUT / "cinematic.usda"))
    if SCENE.exists():
        SCENE.unlink()
    stage = Usd.Stage.CreateNew(str(SCENE))
    layer = stage.GetRootLayer()
    layer.subLayerPaths.append("./cinematic.usda")
    layer.customLayerData = dict(cine.customLayerData)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    add_looks(stage, deck_stage)
    add_zone(stage, deck_stage, t)
    meta = merged_meta(deck_meta, t)
    w = meta["world"]
    write_pipe_attrs(stage.GetPrimAtPath("/World/Pipes/main_pipe"), w)
    cameras = add_cameras(stage, w, t)
    extra = {
        "scene": "sim/output/rig_merged.usda",
        "base_scene": "sim/output/cinematic.usda",
        "prims": {"main_pipe": "/World/Pipes/main_pipe", "damage_patch": "/World/Pipes/damage_patch",
                  "zone": ZONE, "pipe_scope": "/World/Pipes"},
        "zoom": {"wide_target": list(WIDE_TARGET), "close_target": w["damage"]["position_m"]},
    }
    if robot:
        extra["robot"] = add_robot(stage, w)
    layer.Save()
    meta = merged_meta(deck_meta, t, cameras=cameras, extra=extra)
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"SCENE_WRITTEN {SCENE}")
    print(f"META_WRITTEN {META}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-robot", action="store_true")
    a = ap.parse_args()
    build(robot=not a.no_robot)
