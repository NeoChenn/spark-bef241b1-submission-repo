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
    assert depth == pytest.approx(min(0.015, m.flange.head_height - 0.002), abs=0.0015)


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


def test_fit_to_the_film_recording(tmp_path):
    import json

    from integration.retorque_motion import from_timeline

    rows = [{"type": "tick", "t": t, "robots": {"unit2": {"state": st}}}
            for t, st in ((40.5, "SNIFF"), (44.8, "INSPECT"), (46.8, "RETORQUE"), (51.8, "VERIFY"), (54.7, "RETURN"))]
    rows.insert(4, {"type": "torque_done", "robot": "unit2", "flange": "F5", "bolt": 3, "t": 50.2})
    tl = tmp_path / "timeline.jsonl"
    tl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    m = from_timeline(tl)
    assert m.bolt == 3 and m.flange.id == "F5"
    assert m.t0 + m.spans["engage"][1] == pytest.approx(46.8)
    assert m.t_fix == pytest.approx(50.2)
    assert m.t_end <= 54.7 and m.t0 >= 40.5
    assert m.gas_scale(50.19) == 1.0 and m.gas_scale(50.3) < 1.0
    assert m.phase(46.7) == "engage" and m.phase(48.0) == "torque" and m.phase(54.0) == "home"
    k = JOINTS.index("socket_drive_joint")
    assert math.degrees(m.vector(50.2)[k] - m.vector(46.8)[k]) == pytest.approx(120.0)
    assert m.vector(50.2)[k] == pytest.approx(m.vector(50.3)[k])


@pytest.mark.parametrize("after", [(54.7, "STANDBY"), (114.8, "RETURN")])
def test_standby_after_verify_keeps_the_home_phase_short(tmp_path, after):
    import json

    from integration.retorque_motion import from_timeline, timeline_marks

    seq = [(40.5, "SNIFF"), (46.8, "RETORQUE"), (51.8, "VERIFY"), after, (114.8, "RETURN")]
    rows = [{"type": "tick", "t": t, "robots": {"unit2": {"state": st}}} for t, st in seq]
    rows.insert(3, {"type": "torque_done", "robot": "unit2", "flange": "F5", "bolt": 3, "t": 50.2})
    tl = tmp_path / "timeline.jsonl"
    tl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert timeline_marks(tl)["leave"] == after[0]
    m = from_timeline(tl)
    assert m.t0 + m.spans["torque"][0] == pytest.approx(46.8) and m.t_fix == pytest.approx(50.2)
    assert m.t_end <= 54.7 and m.spans["home"][1] - m.spans["home"][0] <= m.cfg.home_s + 1e-9
    for t in (55.0, 80.0, 114.0):
        assert m.phase(t) == "done"
        assert all(m.at(t)[j] == pytest.approx(STOW[j], abs=1e-6) for j in ARM)


def test_every_link_clears_the_flange_pipe_and_heads(m):
    from integration.retorque_motion import CLEAR_M
    d, t, what = m.min_clearance(30.0)
    assert d >= CLEAR_M, (d, t, what)


def test_checker_sees_a_link_driven_through_the_flange(m):
    from integration.retorque_motion import Cylinder
    v = m.engage_q.copy()
    P = m.model.link_poses(m.base, m._full(v))
    w = P["ur3_wrist_1_link"][:3, 3]
    bad = RetorqueMotion(m.flange, 3, t0=0.0, obstacles=[Cylinder("block", w - [0, 0, 0.05], w + [0, 0, 0.05], 0.03)])
    assert bad.clearance(v)[0] < 0.0
