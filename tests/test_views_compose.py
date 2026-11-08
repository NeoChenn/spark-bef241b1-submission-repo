import numpy as np

from rig.camera import look_at
from rig.plume import Field
from rig.views.bev import Bev
from rig.views.compose import ROLES, role_of, robot_views
from rig.views.schedule import Schedule


def frame():
    rgb = np.full((48, 64, 3), 90, np.uint8)
    rgb[16:32, 16:32] = (150, 70, 35)
    depth = np.full((48, 64), 6.0, np.float32)
    depth[:10] = np.inf
    return rgb, depth, look_at((0.0, 0.0, 20.7), (10.0, 0.0, 20.0), focal=10.0, w=64, h=48)


def test_roles_by_index():
    assert role_of(0) == "rust" and role_of(1) == "gas" and role_of(2) == "inspection"
    assert role_of(5) in ROLES


def test_rust_views_have_boxes_and_three_insets():
    rgb, depth, cam = frame()
    v = robot_views("R1", "rust", rgb, depth, cam, (0.0, 0.0, 20.0, 0.0), path=[(0, 0, 20), (5, 0, 20)], bev=Bev(), seq=3)
    assert set(v) == {"fpv", "policy", "bev", "info"}
    assert v["policy"]["seq"] == 3 and "rust" in v["policy"]["caption"]
    assert v["info"]["role"] == "rust" and v["info"]["rust_boxes"] >= 1
    assert all(len(v[k]["jpg"]) < 25_000 for k in ("fpv", "policy", "bev"))


def test_gas_views_report_peak_column():
    rgb, depth, cam = frame()
    c = np.zeros((20, 20, 10), np.float32)
    c[8:12, 8:12, 2:6] = 0.04
    f = Field(c, (1.0, -5.0, 19.0), 0.5)
    v = robot_views("R2", "gas", rgb, np.full((48, 64), 30.0, np.float32), cam, (0.0, 0.0, 20.0, 0.0), field=f, bev=Bev())
    assert "OGI" in v["policy"]["caption"]
    assert v["info"]["gas_column"] > 0


def test_gas_views_without_field_and_inspection_with_force():
    rgb, depth, cam = frame()
    v = robot_views("R2", "gas", rgb, depth, cam, (0.0, 0.0, 20.0, 0.0), bev=Bev())
    assert v["info"]["gas_column"] == 0
    w = robot_views("R3", "inspection", rgb, depth, cam, (0.0, 0.0, 20.0, 0.0), bev=Bev(), force=[5, 20, 30])
    assert "depth" in w["policy"]["caption"]


def test_schedule_round_robin_at_rate():
    s = Schedule(rate=2.0)
    ids = ["R1", "R2", "R3"]
    got = [s.due(ids, t) for t in (0.0, 0.01, 0.02, 0.03)]
    assert got == ["R1", "R2", "R3", None]
    assert s.due(ids, 0.51) == "R1"
    assert s.due(ids, 0.52) == "R2"
    assert s.due([], 5.0) is None


def test_agc_brightens_dark_frames_and_keeps_ratios():
    from rig.views.draw import agc
    dark = np.zeros((48, 64, 3), np.uint8)
    dark[..., 0], dark[..., 1], dark[..., 2] = 30, 15, 8
    dark[0, 0] = (40, 20, 10)
    out = agc(dark)
    assert out.mean() > 2 * dark.mean()
    assert out[5, 5, 0] > out[5, 5, 1] > out[5, 5, 2]


def test_targets_projected_into_fpv():
    rgb, depth, cam = frame()
    v = robot_views("R1", "rust", rgb, depth, cam, (0.0, 0.0, 20.0, 0.0), bev=Bev(),
                    targets=[((5.0, 0.0, 20.2), "damage"), ((-5.0, 0.0, 20.0), "behind")])
    assert v["info"]["targets"] == ["damage"]


def test_agc_gain_is_capped_on_near_black_frames():
    from rig.views.draw import agc
    black = np.full((48, 64, 3), 3, np.uint8)
    assert agc(black).max() < 40


def test_demand_one_robot_keepalive_and_rate():
    from rig.views.schedule import Demand
    d = Demand(rate=4.0, keepalive=45.0, warmup=0.5)
    assert d.active is None and not d.due(0.0)
    assert d.request("R2", True, 0.0) == "start"
    assert not d.due(0.2) and d.due(0.6)
    d.captured(0.6)
    assert not d.due(0.7) and d.due(0.86)
    assert d.request("R2", True, 30.0) is None
    assert not d.expire(70.0) and d.active == "R2"
    assert d.request("R1", True, 71.0) == "switch" and d.active == "R1"
    assert not d.due(71.2)
    assert d.expire(117.0) and d.active is None
    d.request("R3", True, 120.0)
    assert d.request("R1", False, 121.0) is None and d.active == "R3"
    assert d.request(None, False, 122.0) == "stop" and d.active is None
