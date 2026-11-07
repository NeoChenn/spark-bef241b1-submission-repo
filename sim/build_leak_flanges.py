"""Add bolted flanges with torque stripes beside the rig's valves, for the gas-leak job.

Writes, without touching full_rig.usda:
  sim/output/leak_flanges.usda        /World/Leak/<flange>: spool, flange face, bore, 8 bolts, stripes, crack
  sim/output/full_rig_leak.usda       root layer = leak_flanges.usda over full_rig.usda (open this one)
  sim/output/leak_flanges_meta.json   flange geometry for perception and the bridge, plus prim paths

Each flange is built in its own frame (face normal = local +X, bolt 0 straight up = local +Z) under an Xform
placed beside a valve and turned to face the nearest patrol waypoint. Each bolt is an Xform with
``xformOp:rotateX:phase`` (fixed) and ``xformOp:rotateX:loosen`` (degrees; the bridge sets it from LeakWorld),
so the red stripe on the head turns away from its partner mark on the face, exactly as the perception expects
(policy/leak/geometry.py, policy/leak/render.py). The crack is hidden until the bridge shows it.

Runs with plain USD (``pip install usd-core``) or Isaac's python:
    python sim/build_leak_flanges.py                  # uses sim/output/full_rig_meta.json
    C:\\isaacsim\\python.bat sim\\build_leak_flanges.py

PLACEMENT IS A GUESS from the valve positions and patrol route in full_rig_meta.json. Check it visually in
Isaac Sim (open full_rig_leak.usda) and adjust --offset or edit the meta before relying on it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
ROOT = SIM_DIR.parent
for p in (str(SIM_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

from policy.leak.geometry import FlangeGeometry  # noqa: E402
from policy.leak.render import crack_path  # noqa: E402
from scene_builder import OUTPUT, SceneBuilder  # noqa: E402

ROOT_PRIM = "/World/Leak"
STEEL = (0.59, 0.6, 0.63)
HEAD = (0.46, 0.48, 0.51)
BORE = (0.23, 0.24, 0.27)
RED = (0.8, 0.12, 0.12)
CRACK = (0.08, 0.08, 0.09)


def placements(meta: dict, offset: float) -> list[dict]:
    """One flange per valve: offset toward the nearest patrol waypoint, facing it (horizontal)."""
    route = np.asarray(meta.get("patrol_route_m", []), float)
    out = []
    for p in meta.get("inspection_points", []):
        if p.get("type") != "valve":
            continue
        v = np.asarray(p["position_m"], float)
        if len(route):
            w = route[int(np.argmin(np.linalg.norm(route[:, :2] - v[:2], axis=1)))]
            d = np.array([w[0] - v[0], w[1] - v[1], 0.0])
        else:
            d = np.array([1.0, 0.0, 0.0])
        d /= np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else 1.0
        out.append({"valve_id": p["id"], "center": v + offset * d, "normal": d})
    return out


def _xform(stage, path, translate=None, rotate_z=None, phase_x=None, loosen=False):
    x = UsdGeom.Xform.Define(stage, path)
    if translate is not None:
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, translate)))
    if rotate_z is not None:
        x.AddRotateZOp().Set(float(rotate_z))
    if phase_x is not None:
        x.AddRotateXOp(opSuffix="phase").Set(float(phase_x))
    if loosen:
        x.AddRotateXOp(opSuffix="loosen").Set(0.0)
    return x


def build_flange(b: SceneBuilder, stage, fid: str, geom: FlangeGeometry, mats: dict) -> dict:
    yaw = math.degrees(math.atan2(geom.normal[1], geom.normal[0]))
    root = f"{ROOT_PRIM}/{fid}"
    rx = _xform(stage, root, translate=geom.center, rotate_z=yaw)
    rx.GetPrim().CreateAttribute("rig:flangeId", Sdf.ValueTypeNames.String).Set(fid)
    rx.GetPrim().CreateAttribute("rig:crackSeverity", Sdf.ValueTypeNames.Float).Set(0.0)

    b.tube(root, "spool", (-0.5, 0, 0), (0, 0, 0), geom.pipe_radius, mats["steel"])
    b.tube(root, "flange", (-0.04, 0, 0), (0, 0, 0), geom.outer_radius, mats["steel"], segments=64)
    b.disc(root, "face", (0, 0, 0), 1, geom.outer_radius, mats["steel"])
    b.disc(root, "bore", (0.0005, 0, 0), 1, geom.pipe_radius, mats["bore"])

    rh, bolts = geom.head_radius, []
    for k in range(geom.n_bolts):
        phi = geom.bolt_phi(k)
        pos = (0.0, -geom.bolt_circle * math.sin(phi), geom.bolt_circle * math.cos(phi))   # local radial(k)
        bp = f"{root}/bolt_{k}"
        bx = _xform(stage, bp, translate=pos, phase_x=math.degrees(phi), loosen=True)
        bx.GetPrim().CreateAttribute("rig:boltIndex", Sdf.ValueTypeNames.Int).Set(k)
        b.tube(bp, "head", (0, 0, 0), (geom.head_height, 0, 0), rh, mats["head"], segments=6)
        b.box(bp, "stripe", (geom.head_height + 0.0004, 0, 0), (0.0008, 0.24 * rh, 1.8 * rh), mats["red"], collide=None)
        mp = f"{root}/mark_{k}"
        _xform(stage, mp, translate=pos, phase_x=math.degrees(phi))
        b.box(mp, "stripe", (0.0004, 0, 1.675 * rh), (0.0008, 0.28 * rh, 0.85 * rh), mats["red"], collide=None)
        bolts.append(bp)

    local = FlangeGeometry("local", [0, 0, 0], [1, 0, 0], [0, 0, 1], geom.pipe_radius, geom.bolt_circle,
                           geom.head_radius, geom.outer_radius, geom.n_bolts)
    cp = f"{root}/crack"
    cx = _xform(stage, cp)
    pts = crack_path(local, 0.5)
    for i in range(len(pts) - 1):
        a, c = pts[i], pts[i + 1]
        mid, seg = (a + c) / 2, c - a
        theta = math.degrees(math.atan2(-seg[1], seg[2]))     # rotate local +Z onto the segment (in the Y-Z plane)
        sp = f"{cp}/seg_{i}"
        _xform(stage, sp, translate=(0.0006, mid[1], mid[2]), phase_x=theta)
        b.box(sp, "line", (0, 0, 0), (0.0008, 0.004, float(np.linalg.norm(seg))), mats["crack"], collide=None)
    cx.MakeInvisible()
    return {"prim": root, "bolt_prims": bolts, "crack_prim": cp, "loosen_attr": "xformOp:rotateX:loosen"}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=str(OUTPUT / "full_rig_meta.json"))
    ap.add_argument("--out", default=str(OUTPUT))
    ap.add_argument("--offset", type=float, default=0.5, help="metres from each valve toward the walkway")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path(a.meta).read_text())

    layer_path = out / "leak_flanges.usda"
    stage = Usd.Stage.CreateNew(str(layer_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, ROOT_PRIM)
    b = SceneBuilder(stage)
    mats = {"steel": b.plain("LeakSteel", STEEL, 0.35, 0.9), "head": b.plain("LeakBoltHead", HEAD, 0.4, 0.9),
            "bore": b.plain("LeakBore", BORE, 0.8), "red": b.plain("TorqueStripe", RED, 0.6),
            "crack": b.plain("LeakCrack", CRACK, 0.9)}

    flanges = []
    for i, p in enumerate(placements(meta, a.offset)):
        fid = f"F{i + 1}"
        geom = FlangeGeometry(fid, p["center"], p["normal"], [0, 0, 1])
        info = build_flange(b, stage, fid, geom, mats)
        flanges.append({**geom.to_meta(), "valve_id": p["valve_id"], **info})
    stage.GetRootLayer().Save()

    root = Sdf.Layer.CreateNew(str(out / "full_rig_leak.usda"))
    root.subLayerPaths.append("leak_flanges.usda")
    if (out / "full_rig.usda").exists():
        root.subLayerPaths.append("full_rig.usda")
    root.Save()

    result = {"flanges": flanges, "units": "m", "up": "Z",
              "note": "Placement is a guess from valve positions; check it in Isaac Sim. Bolt prims carry "
                      "xformOp:rotateX:loosen (degrees); crack prims start invisible."}
    (out / "leak_flanges_meta.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {len(flanges)} flanges -> {layer_path}, full_rig_leak.usda, leak_flanges_meta.json")
    return result


if __name__ == "__main__":
    main()
