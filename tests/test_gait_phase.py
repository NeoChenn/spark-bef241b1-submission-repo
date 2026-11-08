import math

import numpy as np
import pytest

from rig.sim.gait import (LEGS, STAND, GaitLoop, Walker, fold_cycle, gait_from_raw, gait_weight, phase_rate,
                          wrap)


def synth_loop(period=0.4, hz=50.0):
    n = round(period * hz)
    ph = np.arange(n) / n
    q = np.array([[STAND[j] + 0.3 * math.sin(2 * math.pi * (p + 0.5 * (k % 2))) for k, j in enumerate(LEGS)]
                  for p in ph])
    z = 0.33 + 0.01 * np.sin(4 * math.pi * ph)
    return GaitLoop(period, q, z, speed=0.8)


def test_fold_cycle_recovers_period_and_shape():
    hz, period = 50.0, 0.42
    t = np.arange(0, 6, 1 / hz)
    rng = np.random.default_rng(0)
    q = np.stack([np.sin(2 * math.pi * t / period + k) for k in range(3)], 1) + 0.01 * rng.standard_normal((len(t), 3))
    p, loop = fold_cycle(q, hz, 0.25, 1.0)
    assert abs(p - period) < 0.01
    assert loop.shape == (round(p * hz), 3)
    ph = np.arange(len(loop)) / len(loop)
    ref = np.sin(2 * math.pi * ph[:, None] + np.arange(3)[None, :])
    err = min(np.abs(np.roll(loop, s, 0) - ref).max() for s in range(len(loop)))
    assert err < 0.1


def test_gait_from_raw_skips_warmup_and_measures_speed():
    hz, period = 50.0, 0.4
    rows = []
    for k in range(400):
        t = k / hz
        rows.append({"t": t, "q": [STAND[j] + 0.2 * math.sin(2 * math.pi * t / period) for j in LEGS],
                     "z": 0.33, "vx": 0.0 if t < 2 else 0.78})
    g = gait_from_raw({"hz": hz, "joints": LEGS, "rows": rows, "source": "x"}, skip_s=2.0)
    assert abs(g["period"] - period) < 0.01 and abs(g["speed"] - 0.78) < 1e-6
    assert len(g["q"]) == round(g["period"] * hz) and len(g["q"][0]) == 12 and g["source"] == "x"
    loop = GaitLoop.from_json(g)
    assert loop.q.shape == (len(g["q"]), 12)


def test_loop_wraps_and_interpolates():
    lp = synth_loop()
    q0, _ = lp.at(0.0)
    q1, _ = lp.at(1.0)
    assert np.allclose(q0, q1)
    n = len(lp.q)
    qm, _ = lp.at(0.5 / n)
    assert np.allclose(qm, (lp.q[0] + lp.q[1]) / 2)
    assert np.allclose(lp.at(-0.25)[0], lp.at(0.75)[0])


def test_weight_and_rate():
    assert gait_weight(0.0) == 0.0 and gait_weight(0.5) == 1.0
    ws = [gait_weight(v) for v in np.linspace(0, 0.5, 30)]
    assert all(b >= a for a, b in zip(ws, ws[1:]))
    r = [phase_rate(v, 0.8, 0.4) for v in (0.2, 0.8, 1.0, 3.0)]
    assert r[0] < r[1] < r[2] < r[3] and r[1] == pytest.approx(2.5) and r[3] == pytest.approx(1.4 / 0.4)


def drive(w, speed, t0, t1, dt=1 / 24, every=0.5, y=0.0, x0=0.0):
    out, t, nxt = [], t0, t0
    while t < t1:
        if t >= nxt:
            w.target((x0 + speed * (nxt - t0), y, 0.0, 0.0), t)
            nxt += every
        out.append(w.step(t))
        t += dt
    return out


def test_standing_robot_holds_stand_pose():
    w = Walker(synth_loop(), (1.0, 2.0, 3.0, 0.5), 0.0)
    for k in range(48):
        w.target((1.0, 2.0, 3.0, 0.5), k / 24)
        pose, q, bob = w.step(k / 24)
    assert pose == pytest.approx((1.0, 2.0, 3.0, 0.5))
    assert np.allclose(q, [STAND[j] for j in LEGS]) and bob == 0.0


def test_walking_cycles_legs_at_route_speed_and_stops_smoothly():
    lp = synth_loop()
    w = Walker(lp, (0.0, 0.0, 0.0, 0.0), 0.0)
    walk = drive(w, 0.8, 0.0, 4.0)
    assert w.speed == pytest.approx(0.8, rel=0.15)
    qs = np.array([q for _, q, _ in walk[48:]])
    assert np.ptp(qs[:, 1]) > 0.3
    xs = [p[0] for p, _, _ in walk]
    steps = np.diff(xs)
    assert steps.min() >= -1e-9 and steps[30:].max() < 0.8 / 24 * 2.5
    stop_x = xs[-1]
    hold = []
    t = 4.0
    for k in range(72):
        if k % 12 == 0:
            w.target((0.8 * 4.0, 0.0, 0.0, 0.0), t)
        hold.append(w.step(t))
        t += 1 / 24
    qh = np.array([q for _, q, _ in hold])
    assert np.abs(np.diff(qh, axis=0)).max() < 0.3
    assert np.allclose(qh[-1], [STAND[j] for j in LEGS], atol=1e-3)
    assert hold[-1][0][0] == pytest.approx(3.2) and hold[0][0][0] >= stop_x - 1e-9


def test_teleport_snaps():
    w = Walker(synth_loop(), (0.0, 0.0, 0.0, 0.0), 0.0)
    w.step(0.0)
    w.target((50.0, 0.0, 10.0, 0.0), 0.5)
    pose, _, _ = w.step(0.5)
    assert pose[0] == 50.0 and pose[2] == 10.0 and w.speed == 0.0


def test_yaw_takes_short_way():
    w = Walker(synth_loop(), (0.0, 0.0, 0.0, 3.0), 0.0)
    w.step(0.0)
    w.target((0.0, 0.0, 0.0, -3.0), 0.0)
    yaws = [w.step(k / 24)[0][3] for k in range(1, 48)]
    assert all(abs(wrap(y)) > 2.9 for y in yaws)
    assert abs(wrap(yaws[-1] + 3.0)) < 0.02
