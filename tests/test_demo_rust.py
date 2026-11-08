import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from rig.mission.live import MissionLive
from rig.mission.scenarios import DemoRust
from rig.sim.gait import ARM, GaitLoop
from rig.sim.taskmotion import Clip, scrub_base

ROOT = Path(__file__).resolve().parents[1]
META = json.loads((ROOT / "sim/output/rig_merged_meta.json").read_text())["world"]


def run(times):
    m = MissionLive(obs=False)
    clock = [0.0]
    m.scen.clock = lambda: clock[0]
    m.event("demo", {"id": "rust"})
    sim = [e["name"] for e in m.take_sim()]
    out = []
    for t in times:
        clock[0] = t
        m.advance(0.25)
        out.append(next(r for r in m.for_sim()["robots"] if r["id"] == "R1"))
    return m, sim, out


def test_demo_geometry_matches_the_rig_meta():
    dmg, pipe = META["damage"], META["main_pipe"]
    assert DemoRust.patch == pytest.approx(tuple(dmg["position_m"]), abs=1e-4)
    assert DemoRust.axis == pytest.approx((pipe["y_m"], pipe["z_m"], pipe["radius_m"]))


def test_rust_demo_walks_in_scrubs_clean_and_parks():
    m, sim, (r0, r1, r2, r3, r4, r5) = run([0.0, 3.0, 5.5, 8.0, 12.0, 20.0])
    assert m.scen.cur.id == "demo_rust" and m.scen.selected == "R1"
    assert sim == ["set_weather", "camera", "select", "views"]
    s = m.scen.cur
    assert r0["pose"]["x"] == pytest.approx(s.start[0], abs=0.01) and r0["rust"] == 1.0
    assert r1["state"] == "TRANSIT" and s.start[0] < r1["pose"]["x"] < s.stop[0]
    assert r2["job_state"] == "DEPLOY" and r2["pose"]["x"] == pytest.approx(s.stop[0], abs=0.01)
    assert r3["job_state"] == "SCRUB" and r3["task"].startswith("rust") and 0.5 < r3["rust"] < 1.0
    assert r4["job_state"] == "SCRUB" and r4["rust"] < r3["rust"]
    assert r5["state"] == "IDLE" and r5["rust"] == 0.0 and s.done()
    m.event("demo", {"id": "reset"})
    assert m.scen.cur.id == "flyby" and not m.override


def test_scrub_base_puts_the_recorded_brush_on_the_patch():
    pxr = pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    from rig.sim.joints import RobotJoints

    g = GaitLoop.load()
    m = MissionLive(obs=False)
    m.event("demo", {"id": "rust"})
    s = m.scen.cur
    trunk = s.z + g.zmean
    st = Usd.Stage.CreateInMemory()
    x = UsdGeom.Xform.Define(st, "/R")
    x.AddTranslateOp().Set(Gf.Vec3d(s.stop[0], s.stop[1], trunk))
    x.AddRotateZOp().Set(math.degrees(s.yaw))
    body = UsdGeom.Xform.Define(st, "/R/body")
    body.GetPrim().GetReferences().AddReference(str(ROOT / "sim/output/robot/go2_ur3_lift/go2_ur3_lift.usda"))
    joints = RobotJoints(st, "/R/body")
    tip = next(p for p in Usd.PrimRange(st.GetPrimAtPath("/R")) if p.GetName() == "brush_tip")
    clip = Clip.load("scrub")
    patch, (ay, az, r) = np.asarray(s.patch), s.axis
    pts = []
    for q in clip.q[int(1.5 * clip.hz):]:
        joints.set(ARM, q)
        pts.append(np.asarray(UsdGeom.Xformable(tip).ComputeLocalToWorldTransform(0).Transform(Gf.Vec3d(-0.025, 0, 0))))
    pts = np.asarray(pts)
    radial = np.hypot(pts[:, 1] - ay, pts[:, 2] - az)
    assert np.median(radial) == pytest.approx(r + 0.025, abs=0.02)
    on = np.linalg.norm((pts - patch)[:, :3], axis=1) < META["damage"]["approx_radius_m"]
    assert on.mean() > 0.9
    assert np.linalg.norm(pts.mean(0) - patch) < 0.08


def test_scrub_base_mirrors_for_a_patch_on_the_far_side():
    x, y, yaw = scrub_base((0.0, -0.1, 1.2), 0.0, 1.1, 0.15, 0.32)
    assert yaw == pytest.approx(math.pi) and y < 0 and x < 0
    x, y, yaw = scrub_base((0.0, 0.1, 1.2), 0.0, 1.1, 0.15, 0.32)
    assert yaw == 0.0 and y > 0 and x > 0


def test_greebles_drop_the_brown_grit_decals():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    sys.path.insert(0, str(ROOT / "sim"))
    import greebles

    if not greebles.LAYER.exists():
        pytest.skip("no greebles layer")
    st = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(st, "/World/RepairZone")
    greebles.apply(st)
    clutter = st.GetPrimAtPath(f"{greebles.ZONE_ROOT}/Clutter")
    grit = [p for p in clutter.GetAllChildren() if p.GetName().startswith("grit_")]
    assert grit and not any(p.IsActive() for p in grit)
    assert any(p.IsActive() for p in clutter.GetChildren())
