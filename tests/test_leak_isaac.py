"""Isaac-side pieces of the leak job, tested on CPU with plain USD (usd-core). Nothing here runs Isaac Sim."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from integration.leak_bridge import LeakBridgeMixin, apply_leak_state, bolt_waypoint, load_flanges  # noqa: E402
from policy.leak.world import LeakWorld  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "sim" / "output" / "full_rig_meta.json"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import sys
    sys.path.insert(0, str(ROOT / "sim"))
    from build_leak_flanges import main
    out = tmp_path_factory.mktemp("leak")
    main(["--out", str(out), "--meta", str(META)])
    stage = Usd.Stage.Open(str(out / "leak_flanges.usda"))
    geoms, prims = load_flanges(str(out / "leak_flanges_meta.json"))
    return out, stage, geoms, prims


def test_one_flange_per_valve_with_eight_bolts(built):
    out, stage, geoms, prims = built
    valves = [p for p in json.loads(META.read_text())["inspection_points"] if p["type"] == "valve"]
    assert len(geoms) == len(valves) == 6
    for g in geoms:
        assert len(prims[g.id]["bolt_prims"]) == 8
        assert all(stage.GetPrimAtPath(p).IsValid() for p in prims[g.id]["bolt_prims"])


def test_usd_bolt_poses_match_the_geometry_the_camera_code_uses(built):
    out, stage, geoms, prims = built
    cache = UsdGeom.XformCache()
    for g in geoms:
        for k, path in enumerate(prims[g.id]["bolt_prims"]):
            M = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            top = np.array(M.Transform(Gf.Vec3d(g.head_height, 0, 0)))
            stripe = np.array(M.TransformDir(Gf.Vec3d(0, 0, 1)))
            assert np.linalg.norm(top - g.bolt_center(k)) < 1e-3
            assert np.linalg.norm(stripe / np.linalg.norm(stripe) - g.radial(k)) < 1e-3


def test_loosening_turns_the_stripe_the_same_way_as_the_renderer(built):
    out, stage, geoms, prims = built
    g = geoms[0]
    world = LeakWorld(geoms)
    world.loosen(g.id, 3, 0.2)
    apply_leak_state(stage, world, prims)
    path = prims[g.id]["bolt_prims"][3]
    assert stage.GetPrimAtPath(path).GetAttribute("xformOp:rotateX:loosen").Get() == pytest.approx(math.degrees(0.2))
    M = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    stripe = np.array(M.TransformDir(Gf.Vec3d(0, 0, 1)))
    assert np.linalg.norm(stripe / np.linalg.norm(stripe) - g.radial(3, 0.2)) < 1e-3
    world.loosen(g.id, 3, 0.0)
    apply_leak_state(stage, world, prims)


def test_crack_is_hidden_until_the_world_has_one(built):
    out, stage, geoms, prims = built
    g = geoms[1]
    world = LeakWorld(geoms)
    vis = lambda: UsdGeom.Imageable(stage.GetPrimAtPath(prims[g.id]["crack_prim"])).ComputeVisibility()
    apply_leak_state(stage, world, prims)
    assert vis() == UsdGeom.Tokens.invisible
    world.crack(g.id, 0.4)
    apply_leak_state(stage, world, prims)
    assert vis() == UsdGeom.Tokens.inherited
    assert stage.GetPrimAtPath(prims[g.id]["prim"]).GetAttribute("rig:crackSeverity").Get() == pytest.approx(0.4)


def test_root_layer_puts_the_flanges_over_the_rig(built):
    out, *_ = built
    layer = pxr.Sdf.Layer.FindOrOpen(str(out / "full_rig_leak.usda"))
    assert layer.subLayerPaths[0] == "leak_flanges.usda"


# ---------------------------------------------------------------- bridge mixin, with a fake robot
class FakeArm:
    def __init__(self):
        self.calls = []

    def move_to(self, wps):
        self.calls.append(wps)
        return [True] * len(wps)


class Host(LeakBridgeMixin):
    def __init__(self, stage, world, prims, start):
        self.stage, self.leak_world, self.leak_prims = stage, world, prims
        self.start = np.asarray(start, float)
        self.pos = self.start.copy()
        self.arm = FakeArm()
        self.cam = None

    def robot_position(self):
        return self.pos

    def move_base_to(self, xyz, facing):
        d = np.asarray(xyz, float) - self.pos
        n = np.linalg.norm(d)
        if n < 0.05:
            return True
        self.pos = self.pos + d / n * min(0.2, n)
        return False

    def set_camera_pose(self, eye, target):
        self.cam = (np.asarray(eye), np.asarray(target))

    def advance(self, dt):
        self.leak_world.step(dt)

    def work_position(self):
        return self.start


def host(built, start_offset=4.0):
    out, stage, geoms, prims = built
    world = LeakWorld(geoms)
    g = geoms[0]
    return Host(stage, world, prims, g.center + start_offset * g.normal), world, g


def test_approach_walks_to_the_viewing_pose_and_points_the_camera(built):
    h, world, g = host(built)
    assert h.nearest_flange() == g.id
    assert h.approach_flange(g.id)
    assert np.linalg.norm(h.pos - (g.center + h.approach_distance * g.normal)) < 0.06
    assert np.allclose(h.cam[1], g.center)
    assert h.request_flange_view(1) and not h.request_flange_view(5)


def test_approach_stops_early_if_gas_reaches_the_stop_limit_on_the_way(built):
    h, world, g = host(built)
    world.gas_spike(g.id)
    world.settle()
    assert not h.approach_flange(g.id)
    # It notices on the step that crosses into the stop zone (~1.2 m out for a burst), so it never ends up
    # closer than the viewing pose, and the job never starts inspecting.
    assert np.linalg.norm(h.pos - g.center) >= h.approach_distance - 0.06


def test_torque_puts_the_socket_on_the_bolt_and_updates_the_scene(built):
    h, world, g = host(built)
    world.loosen(g.id, 2, 0.15)
    apply_leak_state(h.stage, world, h.leak_prims)
    assert h.torque_bolt(g.id, 2)
    approach, on_head = h.arm.calls[0]
    assert np.allclose(on_head.position, g.bolt_center(2)) and np.allclose(on_head.approach, -g.normal)
    assert world.flanges[g.id].angles[2] == 0.0
    attr = h.stage.GetPrimAtPath(h.leak_prims[g.id]["bolt_prims"][2]).GetAttribute("xformOp:rotateX:loosen")
    assert attr.Get() == pytest.approx(0.0)


def test_bolt_waypoint_standoff_backs_off_along_the_normal():
    from policy.leak.geometry import FlangeGeometry
    g = FlangeGeometry("F", [0, 0, 1], [1, 0, 0], [0, 0, 1])
    w = bolt_waypoint(g, 0, standoff=0.05)
    assert np.allclose(w.position, g.bolt_center(0) + 0.05 * g.normal)
