import math
from pathlib import Path

import numpy as np
import pytest

from policy.gasbot import fx
from policy.pickcan.video import intrinsics

SCENE = Path(__file__).resolve().parents[1] / "sim/output/rig_merged.usda"


def _cam(W=320, H=180):
    eye, at = np.array([23.2, -12.0, 44.9]), np.array([19.8, -9.6, 43.2])
    z = (at - eye) / np.linalg.norm(at - eye)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    T[:3, 3] = eye
    return intrinsics(22.0, 20.955, W, H), T


def _run(seed, secs=1.5, on=True):
    v = fx.Vapour(seed=seed)
    for _ in range(int(secs / 0.075)):
        v.step(0.075, fx.SRC, (0.6, 0.8), 2.0, on)
    return v


def test_vapour_deterministic():
    a, b, c = _run(3), _run(3), _run(4)
    assert len(a) == len(b) and np.array_equal(a.pos, b.pos)
    assert len(a) != len(c) or not np.array_equal(a.pos, c.pos)


def test_vapour_emits_only_when_on_and_clears():
    assert len(_run(0, on=False)) == 0
    v = _run(0)
    assert len(v) > 20
    for _ in range(20):
        v.step(0.075, fx.SRC, (0.6, 0.8), 2.0, False)
    assert len(v) == 0


def test_vapour_draw_and_depth():
    K, T = _cam()
    v = _run(1, 2.0)
    img = np.full((180, 320, 3), 60, np.uint8)
    far = np.full((180, 320), 100.0, np.float32)
    out = v.draw(img, K, T, far)
    from policy.pickcan.layers import project

    uv, _ = project(np.asarray(fx.SRC) + [0, 0, 0.2], K, T)
    x, y = int(uv[0]), int(uv[1])
    assert np.abs(out.astype(int) - img).sum() > 0
    assert np.abs(out[max(y - 25, 0):y + 25, max(x - 25, 0):x + 25].astype(int) - 60).max() > 3
    near = np.full((180, 320), 0.5, np.float32)
    assert np.array_equal(v.draw(img, K, T, near), img)


def test_valve_arm_reaches_wheel():
    from integration.kinematics import RobotModel, pose
    from rig.sim.gait import ARM, STAND, GaitLoop

    m = RobotModel.from_urdf()
    x, y, zp, yaw = fx.valve_stop()
    g = GaitLoop.load()
    base = pose(x, y, zp + (g.zmean if g else 0.36), yaw)

    def ee(e):
        return m.tool_pose(base, {**STAND, **dict(zip(ARM, fx.valve_arm(e)))}, "ur3_ee_link")

    for e in (fx.REACH_S, fx.REACH_S + 1.0, fx.REACH_S + fx.TURN_S + 2.0):
        Te = ee(e)
        tip = Te[:3, 3] + Te[:3, 0] * fx.KEY["pin"][1]
        assert np.linalg.norm(tip[:2] - np.asarray(fx.WHEEL[:2])) < 0.02
        assert abs(Te[:3, 3][2] - (fx.WHEEL[2] + 0.12)) < 0.02
        assert Te[2, 0] < -0.999
    a0 = math.atan2(ee(fx.REACH_S)[1, 1], ee(fx.REACH_S)[0, 1])
    e1 = fx.REACH_S + 0.7
    a1 = math.atan2(ee(e1)[1, 1], ee(e1)[0, 1])
    d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    assert abs(d - fx.wheel_angle(e1)) < 0.01
    assert np.allclose(fx.valve_arm(0.0), fx.PARK_Q)


def test_wheel_angle_monotonic():
    a = [fx.wheel_angle(e) for e in np.linspace(0, fx.REACH_S + fx.TURN_S + 1, 200)]
    assert a[0] == 0.0 and np.all(np.diff(a) <= 1e-12) and abs(a[-1] + fx.SPIN) < 1e-9


@pytest.mark.skipif(not SCENE.exists(), reason="scene not built")
def test_handwheel_moves_one_instance():
    pytest.importorskip("pxr")
    import sys

    from pxr import Usd, UsdGeom

    sys.path.insert(0, str(SCENE.parents[1]))
    import greebles

    st = Usd.Stage.Open(str(SCENE))
    if not greebles.apply(st):
        pytest.skip("greebles layer missing")
    w = fx.Handwheel(st)
    pi = UsdGeom.PointInstancer(st.GetPrimAtPath(w.path))
    before = np.array(pi.ComputeInstanceTransformsAtTime(Usd.TimeCode.Default(), Usd.TimeCode.Default()))
    w.set(0.9)
    after = np.array(pi.ComputeInstanceTransformsAtTime(Usd.TimeCode.Default(), Usd.TimeCode.Default()))
    moved = np.flatnonzero(np.abs(after - before).reshape(len(before), -1).max(1) > 1e-4)
    assert list(moved) == [w.index]
    M = np.array(UsdGeom.Xformable(pi.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    p = M[:3, :3] @ after[w.index][3, :3] + M[:3, 3]
    assert np.linalg.norm(p - np.asarray(fx.WHEEL)) < 0.01
