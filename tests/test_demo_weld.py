import json
import math
from pathlib import Path

import numpy as np
import pytest

from rig.mission.live import MissionLive
from rig.sim.gait import ARM
from rig.sim.weld import SPARKS, TRAIL, WeldClip, bead_state, export, sparks, to_world

ROOT = Path(__file__).resolve().parents[1]
PIPE = json.loads((ROOT / "sim/output/rig_merged_meta.json").read_text())["world"]["main_pipe"]


def run(times):
    m = MissionLive(obs=False)
    clock = [0.0]
    m.scen.clock = lambda: clock[0]
    m.event("demo", {"id": "weld"})
    sim = m.take_sim()
    out = []
    for t in times:
        clock[0] = t
        m.advance(0.25)
        out.append(next(r for r in m.for_sim()["robots"] if r["id"] == "R1"))
    return m, sim, out


def test_weld_demo_walks_in_welds_cools_and_parks():
    m, sim, (r0, r1, r2, r3, r4, r5) = run([0.0, 2.0, 5.2, 10.0, 18.0, 24.0])
    s = m.scen.cur
    assert s.id == "demo_weld" and m.scen.selected == "R1"
    assert [e["name"] for e in sim] == ["set_weather", "camera", "select", "views"]
    assert sim[1]["args"]["mode"] == "zoom" and 3.0 <= sim[1]["args"]["lens"]["dist"] <= 5.0
    assert sim[3]["args"] == {"robot": "R1", "on": True, "kind": "fpv"}
    assert not r0["weld"]["arm"] and r0["weld"]["at"] == [16.9, -5.25, 42.5, 0.0] and r0["rust"] == 1.0
    assert r1["state"] == "TRANSIT" and s.start[0] < r1["pose"]["x"] < s.stop[0]
    assert math.dist(s.start, s.stop) == pytest.approx(2.0)
    assert r2["job_state"] == "SEAM_SCAN" and r2["weld"]["arm"]
    assert r3["job_state"] == "WELD" and r3["task"] == "weld:demo" and r3["weld"]["c"] == pytest.approx(6.0)
    assert r4["job_state"] == "INSPECT_WELD"
    assert r5["state"] == "IDLE" and not r5["weld"]["arm"] and s.done()
    m.event("demo", {"id": "weld"})
    assert m.scen.cur.id == "demo_weld" and m.scen.cur.i == 0
    m.event("demo", {"id": "reset"})
    assert m.scen.cur.id == "flyby" and not m.override


def test_weld_clip_is_the_seam_tracker_plan_on_the_main_pipe(tmp_path):
    cl = WeldClip.load()
    fresh = export(tmp_path / "clip.json")
    assert np.allclose(fresh["q"], cl.q, atol=1e-4) and np.allclose(fresh["bead"], cl.bead, atol=1e-4)
    assert cl.metrics["tracker_rmse_mm"] < 1.0 < cl.metrics["open_loop_rmse_mm"]
    at = (16.9, -5.25, PIPE["z_m"] - 1.1, 0.0)
    o = to_world(cl.origin, at)
    assert o[1] == pytest.approx(PIPE["y_m"]) and o[2] == pytest.approx(PIPE["z_m"]) and cl.radius == PIPE["radius_m"]
    bead = to_world(cl.bead, at)
    radial = np.hypot(bead[:, 1] - PIPE["y_m"], bead[:, 2] - PIPE["z_m"])
    assert np.all(np.abs(radial - PIPE["radius_m"]) < 0.002)
    assert 10.0 < cl.off - cl.on < 11.0 and cl.bead_t[0] == pytest.approx(cl.on, abs=1e-4) and cl.bead_t[-1] == pytest.approx(cl.off, abs=1e-4)


def test_bead_cools_behind_the_torch_and_sparks_fall_only_while_the_arc_is_on():
    cl = WeldClip.load()
    shown, bins = bead_state(cl, (cl.on + cl.off) / 2)
    assert 0.3 < shown.mean() < 0.7 and bins[0] == 5 and bins[shown][-1] == 0
    assert np.all(np.diff(bins[shown]) <= 0)
    assert bead_state(cl, cl.off + 4.0)[1].min() == 5
    assert not sparks(cl, cl.on - 0.1)[2].any()
    far, low = [], []
    for c in np.arange(2.0, 10.0, 0.1):
        p, ages, alive = sparks(cl, c, 0.005)
        assert alive.mean() > 0.9 and len(p) == SPARKS * TRAIL and set(ages.tolist()) <= {0, 1, 2}
        far.append(np.linalg.norm(p[alive] - cl.tip_at(c), axis=1).max())
        low.append(p[alive][:, 2].min())
    assert 0.5 < np.median(far) < 3.0 and min(low) == pytest.approx(0.005)
    assert not sparks(cl, cl.off + 1.3)[2].any()


def test_clip_joints_put_the_torch_tip_on_the_planned_tip():
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    from rig.sim.joints import RobotJoints

    cl = WeldClip.load()
    st = Usd.Stage.CreateInMemory()
    x = UsdGeom.Xform.Define(st, "/R")
    x.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.32))
    body = UsdGeom.Xform.Define(st, "/R/body")
    body.GetPrim().GetReferences().AddReference(str(ROOT / "sim/output/robot/go2_ur3_lift/go2_ur3_lift.usda"))
    joints = RobotJoints(st, "/R/body")
    tip = next(p for p in Usd.PrimRange(st.GetPrimAtPath("/R")) if p.GetName() == "brush_tip")
    for c in np.linspace(0.0, cl.duration, 9):
        joints.set(ARM, cl.arm(c))
        w = np.asarray(UsdGeom.Xformable(tip).ComputeLocalToWorldTransform(0).ExtractTranslation())
        assert np.linalg.norm(w - cl.tip_at(c)) < 0.003
