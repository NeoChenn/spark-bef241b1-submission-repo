import math

import numpy as np
import pytest

from integration.retorque_motion import JOINTS, STOW, RetorqueMotion, flange_from_meta, main

FPS = 30.0
ARM = [j for j in JOINTS if j.startswith("ur3_")]


@pytest.fixture(scope="module")
def m():
    return RetorqueMotion(flange_from_meta("F5"), 3, t0=24.0)


@pytest.fixture(scope="module")
def frames(m):
    ts = m.t0 - 1.0 + np.arange(int((m.duration + 2.0) * FPS)) / FPS
    return ts, np.array([m.vector(t) for t in ts])


def test_f5_bolt_3_from_the_work_pose(m):
    assert np.allclose(m.head, [20.507, -10.584, 43.308], atol=2e-3)
    s = m.summary()
    assert np.allclose(s["base"], [21.0124, -10.2697, -2.7397], atol=1e-3)
    assert list(s["spans"]) == ["approach", "align", "engage", "torque", "disengage", "retract", "home"]
    assert 9.0 <= m.duration <= 14.0


def test_joints_are_continuous_frame_to_frame(frames):
    _, V = frames
    step = np.abs(np.diff(V, axis=0)).max(0)
    lim = {j: 0.06 for j in JOINTS}
    lim["socket_drive_joint"] = 0.08
    lim["lift_joint"] = 0.02
    for j, s in zip(JOINTS, step):
        assert s <= lim[j], (j, s)
    acc = np.abs(np.diff(V[:, 1:7], 2, axis=0)).max()
    assert acc < 0.01


def test_no_ik_branch_switch_and_elbow_stays_up(m, frames):
    _, V = frames
    for j in ("ur3_elbow_joint", "ur3_wrist_2_joint"):
        col = V[:, JOINTS.index(j)]
        assert np.all(col > 0.3) or np.all(col < -0.3), j
    assert all(m.elbow_up(v) for v in V[::3])
    for j in ARM:
        col = V[:, JOINTS.index(j)]
        assert np.all(np.abs(col) < math.pi - 0.2), j


def test_socket_is_coaxial_with_bolt_3_and_slides_on(m):
    assert m.coaxial_error_deg() < 2.0
    a, b = m.spans["engage"]
    for t in np.linspace(m.t0 + m.spans["align"][0], m.t0 + m.spans["torque"][1], 40):
        T = m.tool_pose(float(t))
        ang = math.degrees(math.acos(float(np.clip(T[:3, 0] @ m.axis, -1, 1))))
        assert ang < 2.0
        off = T[:3, 3] - m.head
        assert np.linalg.norm(off - (off @ m.axis) * m.axis) < 0.002
    tip = m.tool_pose(m.t0 + b)[:3, 3]
    depth = float((tip - m.head) @ m.axis)
    assert depth == pytest.approx(0.008, abs=0.0015)


def test_socket_never_enters_the_flange_outside_engage(m, frames):
    ts, V = frames
    for t, v in zip(ts, V):
        d = float((m.tool_pose(v)[:3, 3] - m.flange.center) @ m.flange.normal)
        assert d > 0.0
        if m.phase(t) in ("approach", "home", "stow", "done"):
            assert d > 0.05


def test_visible_turn_in_steps(m, frames):
    ts, V = frames
    k = JOINTS.index("socket_drive_joint")
    total = math.degrees(V[-1, k] - V[0, k])
    assert 90.0 <= total <= 120.0 + 1e-6
    moving = np.abs(np.diff(V[:, k])) > 1e-4
    starts = int(np.sum(moving[1:] & ~moving[:-1]) + int(moving[0]))
    assert 2 <= starts <= 3
    a, b = m.spans["torque"]
    assert np.all(~moving[(ts[1:] < m.t0 + a - 0.05) | (ts[1:] > m.t0 + b + 0.05)])


def test_rests_in_the_stow_pose_before_and_after(m):
    for t in (m.t0 - 1.0, m.t_end + 1.0):
        q = m.at(t)
        for j in ARM + ["lift_joint"]:
            assert q[j] == pytest.approx(STOW[j], abs=1e-6)
    assert m.phase(m.t0 - 0.5) == "stow" and m.phase(m.t_end + 0.1) == "done"


def test_gas_scale_is_one_until_the_torque_completes_then_decays(m):
    ts = np.linspace(m.t0 - 2.0, m.t_end + 10.0, 800)
    g = np.array([m.gas_scale(t) for t in ts])
    assert np.all(np.diff(g) <= 1e-12)
    assert np.all(g[ts <= m.t_fix] == 1.0)
    assert m.t_fix == pytest.approx(m.t0 + m.spans["torque"][1] - m.cfg.hold_s)
    assert m.gas_scale(m.t_fix + 2.0) == pytest.approx(math.exp(-1.0))
    assert m.gas_scale(m.t_fix + 10.0) < 0.01
    assert m.phase(m.t_fix - 0.05) == "torque"


def test_start_q_makes_the_approach_continuous():
    m0 = RetorqueMotion(flange_from_meta("F5"), 3)
    start = dict(zip(JOINTS, map(float, m0.pre_q)))
    m1 = RetorqueMotion(flange_from_meta("F5"), 3, start_q=start)
    assert np.allclose(m1.vector(0.0), m0.pre_q)


def test_cli_writes_rows(tmp_path):
    out = tmp_path / "rt.json"
    data = main(["--t0", "24", "--fps", "10", "--out", str(out)])
    assert out.exists() and data["rows"][0]["t"] == 24.0 and data["rows"][0]["gas_scale"] == 1.0
    assert set(data["rows"][0]["q"]) == set(JOINTS)
