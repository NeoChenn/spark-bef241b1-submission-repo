import json
import math
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from integration import make_fixer as mf
from integration.kinematics import TRUNK_HEIGHT, RobotModel
from policy.leak.geometry import FlangeGeometry

REPO = pathlib.Path(__file__).resolve().parents[1]
URDF = REPO / "sim/assets/go2_ur3/go2_ur3_fixer.urdf"
FLANGES = REPO / "sim/output/leak_flanges_meta.json"
DECK_Z = 42.5


@pytest.fixture(scope="module")
def robot():
    return ET.parse(URDF).getroot()


@pytest.fixture(scope="module")
def model():
    return RobotModel.from_urdf(URDF)


@pytest.fixture(scope="module")
def f2():
    meta = json.loads(FLANGES.read_text())
    return FlangeGeometry.from_meta(next(f for f in meta["flanges"] if f["id"] == "F2"))


def test_generator_is_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(mf, "DST", tmp_path / "fixer.urdf")
    assert mf.build().read_text() == URDF.read_text()


def test_tree_valid_and_named(robot):
    assert robot.get("name") == "go2_ur3_fixer"
    names = [l.get("name") for l in robot.findall("link")]
    assert len(names) == len(set(names))
    children = [j.find("child").get("link") for j in robot.findall("joint")]
    assert len(children) == len(set(children))
    assert [n for n in names if n not in children] == ["base"]
    assert not {"container", "container_drop", "scrubber_link", "brush_tip"} & set(names)


def test_tool_chain(model):
    chain = [j.name for j in model.chain(mf.TOOL_LINK)]
    assert chain[-4:] == ["ur3_gripper_mount", "gripper_grasp_joint", mf.SOCKET_JOINT, "socket_tip_joint"]
    assert model.joints[mf.SOCKET_JOINT].kind == "continuous"
    assert np.allclose(model.joints[mf.SOCKET_JOINT].axis, [1, 0, 0])
    for j in mf.FINGER_JOINTS:
        assert model.joints[j].lower <= mf.FINGER_CLOSED_ON_SOCKET <= model.joints[j].upper


def test_identity_teal_and_label(robot):
    mats = {m.get("name"): m.find("color").get("rgba") for m in robot.findall("material") if m.find("color") is not None}
    assert mats["fixer_teal"] == mf.TEAL
    rgb = [round(float(v) * 255) for v in mf.TEAL.split()[:3]]
    assert rgb == [0x2E, 0xC4, 0xB6]
    plate = next(l for l in robot.findall("link") if l.get("name") == "unit2_label")
    used = [v.find("material").get("name") for v in plate.findall("visual")]
    assert used.count("fixer_teal") == 2 and used.count("label_white") >= 20


def test_lidar_scan_plane_height(model):
    T = model.tool_pose(np.eye(4), {}, "patrol_lidar")
    assert model.chain("patrol_lidar")[0].parent == "base"
    assert 0.25 <= TRUNK_HEIGHT + T[2, 3] <= 0.35


@pytest.mark.parametrize("side_on", [False, True])
def test_socket_reaches_every_f2_bolt_from_0p6m(model, f2, side_on):
    assert f2.center[2] - DECK_Z == pytest.approx(1.1)
    base = mf.stand_pose(f2.center, f2.normal, 0.6, DECK_Z + TRUNK_HEIGHT, side_on)
    assert np.linalg.norm(base[:2, 3] - f2.center[:2]) == pytest.approx(0.6)
    for k in range(f2.n_bolts):
        r = mf.reach(model, base, f2.bolt_center(k), -f2.normal, side_on=side_on)
        assert r.position_error < 0.003 and r.axis_error < 0.01, (k, r.position_error, r.axis_error)
        T = model.tool_pose(base, r.q, mf.TOOL_LINK)
        assert T[:3, 0] @ -f2.normal > math.cos(0.01)
