import numpy as np

from rig.camera import look_at
from rig.plume import Field
from rig.views.bev import Bev
from rig.views.compose import ROLES, role_of, robot_views
from rig.views.schedule import Schedule


def frame():
    rgb = np.full((48, 64, 3), 90, np.uint8)
    rgb[16:32, 24:40] = (150, 70, 35)
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
