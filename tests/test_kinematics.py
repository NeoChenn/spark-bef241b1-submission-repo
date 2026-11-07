import math

import numpy as np
import pytest

from integration.kinematics import ARM_JOINTS, RobotModel, pose, rpy_matrix, spring_force
from policy.interfaces import PipeGeometry


@pytest.fixture(scope="module")
def model():
    return RobotModel.from_urdf()


@pytest.fixture
def pipe():
    # rig_deck_meta.json seed 7: main pipe along +x at y=0, z=1.1, r=0.15, theta from +y toward +z
    return PipeGeometry(origin=[-0.4933, 0, 1.1], axis=[1, 0, 0], ref=[0, 1, 0], radius=0.15, length=0.8,
                        n_z=40, n_theta=64)


DAMAGE_THETA = math.atan2(0.7836, 0.6212)


def test_rpy_matches_urdf_convention():
    R = rpy_matrix(0.3, -0.2, 1.1)
    Rx = np.array([[1, 0, 0], [0, math.cos(0.3), -math.sin(0.3)], [0, math.sin(0.3), math.cos(0.3)]])
    Ry = np.array([[math.cos(-0.2), 0, math.sin(-0.2)], [0, 1, 0], [-math.sin(-0.2), 0, math.cos(-0.2)]])
    Rz = np.array([[math.cos(1.1), -math.sin(1.1), 0], [math.sin(1.1), math.cos(1.1), 0], [0, 0, 1]])
    assert np.allclose(R, Rz @ Ry @ Rx)


def test_chain_has_lift_and_six_arm_joints(model):
    movable = [j.name for j in model.chain() if j.kind != "fixed"]
    assert movable == ["lift_joint", *ARM_JOINTS]


def test_training_start_pose_is_8cm_from_damage(model, pipe):
    """ScrubLiftEnvCfg says its FK-solved start puts brush_tip 8 cm off the damage, trunk 0.5 m from the axis."""
    surf, _ = pipe.surface_point(DAMAGE_THETA, 0.0)
    tip = model.tool_pose(pose(pipe.origin[0], 0.5, 0.32), model.default_q())[:3, 3]
    assert np.linalg.norm(tip - surf) == pytest.approx(0.08, abs=0.005)


def test_standing_feet_touch_deck(model):
    poses = model.link_poses(pose(0, 0, 0.32), model.default_q())
    feet = [T[2, 3] for name, T in poses.items() if name.endswith("_foot")]
    assert len(feet) == 4
    assert all(0.0 <= z < 0.04 for z in feet)


def test_ik_reaches_damage_centre_normal_to_surface(model, pipe):
    base = pose(pipe.origin[0], 0.5, 0.32)
    p, n = pipe.surface_point(DAMAGE_THETA, 0.0)
    result = model.solve_ik(base, model.default_q(), p, -n)
    assert result.reached
    T = model.tool_pose(base, result.q)
    assert np.linalg.norm(T[:3, 3] - p) < 0.002
    assert T[:3, 0] @ -n > 0.999


def test_ik_respects_joint_limits(model, pipe):
    base = pose(pipe.origin[0], 0.5, 0.32)
    p, n = pipe.surface_point(DAMAGE_THETA, 0.1)
    q = model.solve_ik(base, model.default_q(), p, -n).q
    for name in ARM_JOINTS:
        j = model.joints[name]
        assert j.lower - 1e-9 <= q[name] <= j.upper + 1e-9


def test_spring_force_matches_training_stiffness(pipe):
    p, n = pipe.surface_point(DAMAGE_THETA, 0.0)
    assert spring_force(p + 0.01 * n, pipe) == 0.0
    assert spring_force(p - 0.01 * n, pipe) == pytest.approx(25.0, rel=1e-6)


def test_weld_bead_raises_force(pipe):
    p, n = pipe.surface_point(DAMAGE_THETA, 0.0)
    tip = p - 0.004 * n
    plain = spring_force(tip, pipe)
    bead = spring_force(tip, pipe, bumps=[(DAMAGE_THETA, 0.0, 0.015, 0.03)])
    assert bead > 40.0 > plain
