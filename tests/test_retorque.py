import json
import math
import pathlib

import numpy as np
import pytest

from policy.leak.geometry import FlangeGeometry
from policy.skills.retorque import (DECK_Z, PHASES, RetorqueConfig, RetorqueSkill, Unreachable, plan_retorque,
                                    work_pose)

REPO = pathlib.Path(__file__).resolve().parents[1]
META = json.loads((REPO / "sim/output/leak_flanges_meta.json").read_text())


def flange(fid):
    return FlangeGeometry.from_meta(next(f for f in META["flanges"] if f["id"] == fid))


@pytest.fixture(scope="module")
def skill():
    return RetorqueSkill()


@pytest.fixture(scope="module")
def f5():
    return flange("F5")


@pytest.fixture(scope="module")
def base(f5):
    return work_pose(f5)


@pytest.fixture(scope="module")
def plans(skill, f5, base):
    return skill.plan_bolts(f5, range(f5.n_bolts), base)


def test_f5_work_pose_faces_the_flange_from_0p6m(f5, base):
    assert f5.center[2] - DECK_Z == pytest.approx(0.9)
    assert np.linalg.norm(base[:2, 3] - f5.center[:2]) == pytest.approx(0.6)
    assert base[:3, 0] @ -f5.normal == pytest.approx(1.0)


def test_every_f5_bolt_is_engaged(skill, f5, base, plans):
    assert [p.bolt for p in plans] == list(range(f5.n_bolts))
    for p in plans:
        tip = skill.tip_pose(base, p.end_of("engage"))[:3, 3]
        assert np.linalg.norm(tip - (f5.bolt_center(p.bolt) - 0.008 * f5.normal)) < 0.003


@pytest.mark.parametrize("fid", ["F2", "F5"])
def test_ik_reaches_every_bolt_head(skill, fid):
    g = flange(fid)
    b = work_pose(g)
    for k in range(g.n_bolts):
        r = skill.solve(b, g.bolt_center(k), -g.normal)
        assert r.position_error < 0.003 and r.axis_error < math.radians(3), (fid, k)


def test_phases_run_in_order(plans):
    for p in plans:
        seen = [ph for i, ph in enumerate(p.phase) if i == 0 or p.phase[i - 1] != ph]
        assert seen == list(PHASES)


def test_socket_axis_aligned_within_3_degrees(skill, base, plans):
    for p in plans:
        for i in np.flatnonzero([ph in ("engage", "torque", "disengage") for ph in p.phase]):
            x = skill.tip_pose(base, dict(zip(p.joint_names, p.q[i])))[:3, 0]
            assert math.degrees(math.acos(min(1.0, x @ p.bolt_axis))) < 3.0


def test_torque_spins_only_the_socket(plans):
    for p in plans:
        a, b = p.phase_span("torque")
        qa, qb = p.at(a), p.at(b)
        assert qb["socket_drive_joint"] - qa["socket_drive_joint"] == pytest.approx(math.radians(30), abs=0.03)
        assert all(abs(qb[n] - qa[n]) < 1e-9 for n in p.joint_names if n != "socket_drive_joint")


def test_joint_velocities_bounded_and_rest_at_ends(skill, plans):
    cfg = skill.config
    limit = np.array([cfg.max_lift_speed if n == "lift_joint" else cfg.max_joint_speed for n in plans[0].joint_names])
    for p in plans:
        dt = np.diff(p.t)
        assert (dt > 0).all()
        v = np.abs(np.diff(p.q, axis=0)) / dt[:, None]
        assert (v <= limit * 1.02).all()
        assert v[0].max() < 0.05 and v[-1].max() < 0.05
        assert np.abs(np.diff(v, axis=0)).max() < 0.2


def test_tool_never_penetrates_the_flange(skill, f5, base, plans):
    for p in plans:
        for q in p.q[::2]:
            pts = skill.tool_points(base, dict(zip(p.joint_names, q)))
            assert ((pts - f5.center) @ f5.normal).min() > 0.0


def test_returns_home(skill, plans):
    home = skill._vec(None)
    for p in plans:
        end = p.q[-1].copy()
        end[p.joint_names.index("socket_drive_joint")] = 0.0
        assert np.allclose(end, home, atol=1e-6)


def test_verify_pose_sniffs_in_front_of_the_flange(skill, f5, base):
    v = skill.verify_pose(f5, base)
    assert np.linalg.norm(v.tip - (f5.center + 0.05 * f5.normal)) < 0.003
    assert ((skill.tool_points(base, v.q) - f5.center) @ f5.normal).min() > 0.0
    vb = skill.verify_pose(f5, base, bolt=4)
    assert np.linalg.norm(vb.tip - vb.target) < 0.003


def test_out_of_reach_raises(skill, f5):
    with pytest.raises(Unreachable):
        skill.plan(f5, 0, work_pose(f5, standoff=2.0))


def test_plan_from_centre_axis_and_bolt_circle(f5, base):
    p = plan_retorque(f5.center, f5.normal, f5.bolt_circle, 4, base, flange_id="F5", torque_angle=math.radians(20))
    assert p.flange_id == "F5" and p.torque_angle == pytest.approx(math.radians(20))
    assert np.allclose(p.bolt_center, f5.bolt_center(4))
    assert isinstance(RetorqueConfig().lift, float)


def test_tool_keeps_clear_of_the_g_f5_detector(skill, base, plans):
    head = np.array([20.153, -10.396, 43.46])
    for p in plans:
        for q in p.q[::5]:
            assert np.linalg.norm(skill.tool_points(base, dict(zip(p.joint_names, q))) - head, axis=1).min() > 0.1
