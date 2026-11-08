import pathlib
import xml.etree.ElementTree as ET

import pytest

URDF = pathlib.Path(__file__).resolve().parents[1] / "sim/assets/go2_ur3/go2_ur3_gripper.urdf"


@pytest.fixture(scope="module")
def robot():
    return ET.parse(URDF).getroot()


def links(robot):
    return [l.get("name") for l in robot.findall("link")]


def joints(robot):
    return {j.get("name"): j for j in robot.findall("joint")}


def test_name(robot):
    assert robot.get("name") == "go2_ur3_gripper"


def test_brush_removed(robot):
    names = set(links(robot))
    assert not names & {"scrubber_link", "brush_tip"}
    assert not set(joints(robot)) & {"ur3_scrubber_mount", "scrubber_tip_joint"}


def test_gripper_and_container_present(robot):
    names = set(links(robot))
    for l in ("gripper_base", "gripper_finger_left", "gripper_finger_right", "grasp_frame", "container", "container_drop"):
        assert l in names
    j = joints(robot)
    assert j["ur3_gripper_mount"].find("parent").get("link") == "ur3_ee_link"
    assert j["base_container_mount"].find("parent").get("link") == "base"
    assert j["gripper_finger_right_joint"].find("mimic").get("joint") == "gripper_finger_left_joint"


def test_opening(robot):
    j = joints(robot)
    total = 0.0
    for side in ("left", "right"):
        jt = j[f"gripper_finger_{side}_joint"]
        assert jt.get("type") == "prismatic"
        lim = jt.find("limit")
        assert float(lim.get("lower")) == 0.0
        total += float(lim.get("upper"))
    assert total == pytest.approx(0.12)


def test_tree_valid(robot):
    names = links(robot)
    assert len(names) == len(set(names))
    children = set()
    for j in robot.findall("joint"):
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        assert p in names and c in names
        assert c not in children
        children.add(c)
    roots = [n for n in names if n not in children]
    assert roots == ["base"]
