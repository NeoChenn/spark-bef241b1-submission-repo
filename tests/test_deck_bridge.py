"""Isaac-side pieces of the deck sweep and the leak bridge, tested on CPU with plain USD (usd-core). No Isaac Sim."""
import math
from pathlib import Path

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from detection.rig_vision import floor_range  # noqa: E402
from integration.deck_bridge import (DeckBridgeMixin, deck_z_from_meta, depth_to_ground_range,  # noqa: E402
                                     pixel_to_deck, spawn_debris_can, usd_camera_model, world_to_base)
from integration.leak_bridge import LeakBridgeMixin, load_leak_world  # noqa: E402
from integration.rig_bridge_example import RigBridgeExample  # noqa: E402
from policy.deck.scene import camera_model as mock_deck_camera  # noqa: E402
from policy.leak.geometry import look_at, project  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W, H = 640, 360


def usd_camera(stage, path, eye, target, focal=35.0, aperture=20.955):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(aperture)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    cam.AddTransformOp().Set(view.GetInverse())
    return cam


def flat_deck_depth(camera, deck_z=0.0):
    """Distance-to-image-plane image of an infinite flat deck (inf above the horizon), as Isaac would render it."""
    K, T = np.asarray(camera.K), np.asarray(camera.T_world_cam)
    v, u = np.mgrid[:camera.height, :camera.width].astype(float)
    rc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    dz = rc @ T[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dz[..., 2] < -1e-9, (deck_z - T[2, 3]) / dz[..., 2], np.inf)
    return t.astype(np.float32)


# ---------------------------------------------------------------- camera model from a USD camera
def test_usd_camera_matches_look_at_and_projects_known_points():
    stage = Usd.Stage.CreateInMemory()
    eye, target = (1.0, -2.0, 1.5), (3.0, 0.5, 0.2)
    usd_camera(stage, "/World/cam", eye, target)
    cam = usd_camera_model(stage, "/World/cam", W, H)
    assert np.allclose(cam.T_world_cam, look_at(eye, target), atol=1e-6)
    assert cam.K[0, 0] == pytest.approx(35.0 / 20.955 * W)
    uv, z = project(cam, np.array([target]))
    assert np.allclose(uv[0], [W / 2, H / 2], atol=1e-6) and z[0] > 0
    # a point straight above the target shows above the image centre (y down); one to the camera's right, right
    fwd = np.subtract(target, eye) / np.linalg.norm(np.subtract(target, eye))
    right = np.cross(fwd, [0, 0, 1])
    uv, _ = project(cam, np.array([np.add(target, [0, 0, 0.3]), np.add(target, 0.3 * right)]))
    assert uv[0, 1] < H / 2 and uv[1, 0] > W / 2


def test_camera_on_a_moving_robot_follows_its_parent():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Robot").AddTranslateOp().Set(Gf.Vec3d(5, 0, 0))
    usd_camera(stage, "/World/Robot/cam", (0, 0, 0.6), (2, 0, 0))
    cam = usd_camera_model(stage, "/World/Robot/cam", W, H)
    assert np.allclose(cam.T_world_cam[:3, 3], [5, 0, 0.6])
    assert np.allclose(cam.T_world_cam, look_at((5, 0, 0.6), (7, 0, 0)), atol=1e-6)


# ---------------------------------------------------------------- depth -> ground distance
def test_depth_of_a_flat_deck_gives_floor_range():
    cam = mock_deck_camera()
    rng = depth_to_ground_range(flat_deck_depth(cam), cam)
    ref = floor_range(cam, H, W)
    fin = np.isfinite(ref)
    assert fin.sum() > 0.5 * fin.size
    assert np.array_equal(np.isfinite(rng), fin)
    near = fin & (ref < 30)
    assert np.abs(rng[near] - ref[near]).max() < 0.01


def test_ground_range_works_at_deck_height_and_handles_bad_depth():
    cam = mock_deck_camera()
    T = cam.T_world_cam.copy()
    T[2, 3] += 42.5                                     # the full rig's top deck
    cam.T_world_cam = T
    depth = flat_deck_depth(cam, deck_z=42.5)
    depth[300, 10], depth[300, 11], depth[300, 12] = np.nan, 0.0, -1.0
    rng = depth_to_ground_range(depth[..., None], cam)  # (H, W, 1) as some annotators return
    assert np.isinf(rng[300, 10:13]).all() and np.isinf(rng[0]).all()
    ref = floor_range(mock_deck_camera(), H, W)
    assert abs(rng[300, 320] - ref[300, 320]) < 0.01


def test_pixel_to_deck_round_trip():
    cam = mock_deck_camera()
    for p in ([2.0, 0.3, 0.0], [4.5, -0.8, 0.0], [1.2, 0.0, 0.0]):
        uv, _ = project(cam, np.array([p]))
        q = pixel_to_deck(cam, *uv[0], deck_z=0.0)
        assert np.linalg.norm(q - p) < 1e-6
    assert pixel_to_deck(cam, W / 2, 10, deck_z=0.0) is None        # above the horizon


def test_world_to_base():
    p = world_to_base([1.0, 2.0, 0.0], [1.0, 1.0, 0.3], math.pi / 2)
    assert np.allclose(p, [1.0, 0.0, -0.3])


def test_deck_z_from_meta():
    assert deck_z_from_meta(str(ROOT / "sim" / "output" / "full_rig_meta.json")) == pytest.approx(42.5)


# ---------------------------------------------------------------- debris prims
def test_spawn_debris_can_lying_on_the_deck():
    stage = Usd.Stage.CreateInMemory()
    path = spawn_debris_can(stage, "/World/Debris/can0", (1.0, 2.0, 42.5), yaw=math.pi / 4, physics=True)
    prim = stage.GetPrimAtPath(path)
    body = UsdGeom.Cylinder(stage.GetPrimAtPath(path + "/body"))
    assert body.GetAxisAttr().Get() == UsdGeom.Tokens.x
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
    assert box.GetMin()[2] == pytest.approx(42.5, abs=1e-6)          # resting on the deck
    assert box.GetMax()[2] == pytest.approx(42.5 + 0.066, abs=1e-6)
    mat = pxr.UsdShade.MaterialBindingAPI(body.GetPrim()).ComputeBoundMaterial()[0]
    assert mat and mat.GetPath().name == "CanRed"
    assert prim.HasAPI(pxr.UsdPhysics.RigidBodyAPI)


# ---------------------------------------------------------------- the composed example bridge
class FakeArm:
    def __init__(self):
        self.calls = []

    def move_to(self, wps):
        self.calls.append(wps)
        return [True] * len(wps)


def deck_rig(deck_z=42.5, frames=None):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Robot")
    usd_camera(stage, "/World/Robot/body_cam", (0.3, 0, 0.6), (3.0, 0, 0))
    UsdGeom.Camera.Define(stage, "/World/wrist_cam")
    picks = []

    def pick(xy, yaw):
        picks.append((np.asarray(xy), yaw))
        return True

    class Bridge(RigBridgeExample):
        def grab_camera(self, path, depth=False):         # no renderer on CPU: a counter image + a flat deck
            self.n_grabs = getattr(self, "n_grabs", 0) + 1
            out = {"rgb": np.full((H, W, 3), self.n_grabs % 256, np.uint8)}
            if depth:
                out["depth"] = flat_deck_depth(self.camera_model(path), self.deck_z)
            return out

    b = Bridge(stage, "/World/Robot", step_fn=lambda dt: None, arm=FakeArm(), deck_z=deck_z,
               deck_camera_path="/World/Robot/body_cam", body_camera_path="/World/Robot/body_cam",
               wrist_camera_path="/World/wrist_cam", pick_skill=pick)
    b.set_base_pose([0.0, 0.0, deck_z], 0.0)
    return b, picks


def test_pick_debris_projects_walks_and_hides_the_picked_can():
    b, picks = deck_rig()
    path = b.spawn_debris("can0", [1.6, 0.4])
    img, cam = b.deck_frame()
    uv, _ = project(cam, np.array([[1.6, 0.4, 42.5]]))
    assert b.pick_debris(*uv[0])
    xy, yaw = picks[0]
    assert np.allclose(xy, [b.pick_standoff, 0.0], atol=0.06)        # walked so the can is in reach, dead ahead
    assert UsdGeom.Imageable(b.stage.GetPrimAtPath(path)).ComputeVisibility() == UsdGeom.Tokens.invisible
    assert path not in b.debris_prims
    assert not b.pick_debris(W / 2, 5)                                 # sky: not on the deck


def test_pick_skill_must_be_plugged_in():
    b, _ = deck_rig()
    del b.pick_skill
    b.deck_frame()
    cam = b.camera_model(b.deck_camera_path)
    uv, _ = project(cam, np.array([[0.28, 0.0, 42.5]]))
    with pytest.raises(NotImplementedError, match="pick policy"):
        b.pick_debris(*uv[0])


def test_body_range_comes_from_the_same_frame():
    b, _ = deck_rig()
    rgb, cam = b.body_frame()
    rng = b.body_range()
    assert rgb.shape == (H, W, 3) and rng.shape == (H, W) and b.n_grabs == 1   # depth of the same frame
    lowered = type(cam)(cam.K, cam.T_world_cam.copy(), W, H)
    lowered.T_world_cam[2, 3] -= 42.5                   # floor_range assumes the deck at z = 0
    ref = floor_range(lowered, H, W)
    assert np.array_equal(np.isfinite(rng), np.isfinite(ref))
    near = np.isfinite(ref) & (ref < 30)
    assert np.abs(rng[near] - ref[near]).max() < 0.01


def test_deck_views_move_the_body_and_references_are_per_stop_and_view(tmp_path):
    b, _ = deck_rig()
    b.deck_frame()
    ref0 = b.deck_reference()                            # first visit: the current frame is the reference
    assert np.array_equal(ref0, b._deck()["last"]["deck"][1])
    assert b.request_deck_view(1)
    assert np.allclose(b.robot_position()[:2], [0.0, 0.35], atol=0.06)   # stepped left (yaw 0: +y)
    b.deck_frame()
    ref1 = b.deck_reference()
    assert not np.array_equal(ref0, ref1)
    assert np.array_equal(b.deck_reference(), ref1)     # stored, not re-captured
    assert b.request_deck_view(0) and np.allclose(b.robot_position()[:2], [0, 0], atol=0.06)
    assert not b.request_deck_view(7)
    b.save_references(tmp_path / "refs")
    c, _ = deck_rig()
    assert c.load_references(tmp_path / "refs") == 2
    c.deck_frame()
    assert np.array_equal(c.deck_reference(), ref0)
    c.patrol_stop = 3
    c.deck_frame()
    assert not np.array_equal(c.deck_reference(), ref0)


# ---------------------------------------------------------------- leak methods over the example bridge
@pytest.fixture(scope="module")
def leak_scene(tmp_path_factory):
    import sys
    sys.path.insert(0, str(ROOT / "sim"))
    from build_leak_flanges import main
    out = tmp_path_factory.mktemp("leak")
    main(["--out", str(out), "--meta", str(ROOT / "sim" / "output" / "full_rig_meta.json")])
    return out


def leak_rig(leak_scene, start_offset=3.0):
    stage = Usd.Stage.Open(str(leak_scene / "leak_flanges.usda"))
    UsdGeom.Xform.Define(stage, "/World/Robot")
    UsdGeom.Camera.Define(stage, "/World/wrist_cam")
    world, prims = load_leak_world(str(leak_scene / "leak_flanges_meta.json"))
    g = world.flanges["F1"].geom
    steps = []
    b = RigBridgeExample(stage, "/World/Robot", step_fn=steps.append, leak_world=world, leak_prims=prims,
                         arm=FakeArm(), deck_z=42.5, wrist_camera_path="/World/wrist_cam")
    start = g.center + start_offset * g.normal
    b.set_base_pose([start[0], start[1], 42.5], 0.0)
    b._work = b.robot_position().copy()
    return b, world, g, steps


def test_leak_methods_over_the_example_bridge(leak_scene):
    b, world, g, steps = leak_rig(leak_scene)
    assert isinstance(b, LeakBridgeMixin) and isinstance(b, DeckBridgeMixin)
    assert b.nearest_flange() == "F1" and len(b.flanges()) == 6 and b.flange("F1") is g
    clean = b.gas_reading()
    world.loosen("F1", 2, 0.3)
    b.advance(5.0)                                       # gas builds up in sim time, and the stage follows
    assert b.gas_reading() > clean and steps == [5.0]
    attr = b.stage.GetPrimAtPath(b.leak_prims["F1"]["bolt_prims"][2]).GetAttribute("xformOp:rotateX:loosen")
    assert attr.Get() == pytest.approx(math.degrees(0.3))
    assert b.approach_flange("F1")
    d = b.robot_position()[:2] - g.center[:2]
    assert abs(np.linalg.norm(d) - b.approach_distance) < 0.15
    wrist = usd_camera_model(b.stage, "/World/wrist_cam", W, H)
    uv, z = project(wrist, np.array([g.center]))
    assert np.allclose(uv[0], [W / 2, H / 2], atol=1e-3) and z[0] == pytest.approx(b.approach_distance, abs=1e-6)
    assert b.torque_bolt("F1", 2) and world.flanges["F1"].angles[2] == 0.0 and attr.Get() == pytest.approx(0.0)
    near = np.linalg.norm(b.robot_position()[:2] - g.center[:2])
    b.retreat(3.0)
    assert np.linalg.norm(b.robot_position()[:2] - g.center[:2]) == pytest.approx(near + 3.0, abs=0.15)
    b.return_to_work()
    assert np.allclose(b.robot_position(), b.work_position(), atol=0.06)


def test_gas_override_and_failed_sensor(leak_scene):
    b, world, g, _ = leak_rig(leak_scene)
    b.gas_override = float("nan")
    assert math.isnan(b.gas_reading())                  # the job reads NaN as a failed sensor -> retreat
    b.gas_override = 0.5
    assert b.gas_reading() == 0.5
    b.gas_override = None
    world.crack("F1", 1.0)
    world.settle()
    low = b.gas_reading()
    b.gas_sensor_offset = tuple(-2.0 * g.normal)        # sensor 2 m nearer the flange
    assert b.gas_reading() > low
