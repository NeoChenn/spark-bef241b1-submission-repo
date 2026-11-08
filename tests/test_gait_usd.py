import math

import pytest
from pxr import Gf, Usd, UsdGeom

from rig.render.usdsync import StageCapture
from rig.sim.gait import ARM, LEGS, PARK, STAND
from rig.sim.joints import JOINTS, RobotJoints

ROBOT = "sim/output/robot/go2_ur3_lift/go2_ur3_lift.usda"
BODY = "/World/R1/body"


@pytest.fixture
def stage():
    st = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(st, "/World/R1")
    body = UsdGeom.Xform.Define(st, BODY)
    body.GetPrim().GetReferences().AddReference(str(__import__("pathlib").Path(ROBOT).resolve()))
    body.GetPrim().GetVariantSets().GetVariantSet("Physics").SetVariantSelection("none")
    return st


def world(st, rel):
    m = UsdGeom.Xformable(st.GetPrimAtPath(f"{BODY}/{rel}")).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return m.ExtractTranslation()


def test_every_joint_binds(stage):
    j = RobotJoints(stage, BODY)
    assert set(j.ops) == set(LEGS + ARM) == set(JOINTS)


def test_stand_pose_puts_feet_under_hips(stage):
    j = RobotJoints(stage, BODY)
    hip = world(stage, JOINTS["FL_thigh_joint"][0])
    assert j.set(LEGS, [STAND[n] for n in LEGS]) == 12
    foot = world(stage, JOINTS["FL_calf_joint"][0] + "/FL_foot")
    assert abs(foot[0] - hip[0]) < 0.03
    assert 0.28 < hip[2] - foot[2] < 0.33
    knee = world(stage, JOINTS["FL_calf_joint"][0])
    assert knee[0] < hip[0] - 0.1


def test_lift_raises_arm_and_quantised_repeats_are_free(stage):
    j = RobotJoints(stage, BODY)
    before = world(stage, JOINTS["ur3_shoulder_pan_joint"][0])
    assert j.set(["lift_joint"], [0.3]) == 1
    after = world(stage, JOINTS["ur3_shoulder_pan_joint"][0])
    assert after[2] - before[2] == pytest.approx(0.3, abs=1e-4)
    assert j.set(["lift_joint"], [0.3002]) == 0


def test_capture_streams_joint_attrs(stage):
    j = RobotJoints(stage, BODY)
    cap = StageCapture(stage)
    j.set(LEGS + ARM, [STAND[n] for n in LEGS] + [PARK[n] for n in ARM])
    attrs = cap.collect()["attrs"]
    names = {(p.rsplit("/", 1)[-1], n) for p, n, _, _ in attrs}
    assert ("FL_thigh", "xformOp:orient") in names and ("lift_carriage", "xformOp:translate") in names
    assert len(attrs) == 19
    j.set(LEGS, [STAND[n] for n in LEGS])
    assert cap.collect()["attrs"] == []
