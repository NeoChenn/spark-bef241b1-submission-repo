import math

import numpy as np
import pytest

from rig.sim.gait import ARM, LEGS, PARK, STAND, GaitLoop, Quantiser, Walker
from rig.sim.taskmotion import Clip, TaskMotion, task_kind, weld_sweep

HZ = 24.0
BUDGET = 3 * (len(LEGS) + 2) + len(ARM)


def loop():
    n = 20
    ph = np.arange(n) / n
    q = np.array([[STAND[j] + 0.3 * math.sin(2 * math.pi * (p + 0.5 * (k % 2))) for k, j in enumerate(LEGS)] for p in ph])
    return GaitLoop(0.4, q, 0.33 + 0.005 * np.sin(4 * math.pi * ph), 0.8)


def clip(n=90, phases=None):
    q = [[PARK[j] + 0.4 * math.sin(k / 10 + i) for i, j in enumerate(ARM)] for k in range(n)]
    return Clip(30.0, ARM, q, phases, "rec")


def run(robots, seconds, selected=None, entry=None):
    lp = loop()
    ws = {rid: Walker(lp, (0.0, 3.0 * i, 0.0, 0.0), 0.0) for i, rid in enumerate(robots)}
    qs = {rid: (Quantiser(), Quantiser(0.05)) for rid in robots}
    arm = TaskMotion({"scrub": clip()})
    counts = []
    for k in range(int(seconds * HZ)):
        t = k / HZ
        n = 0
        for i, (rid, speed) in enumerate(robots.items()):
            if k % 12 == 0:
                ws[rid].target((speed * t, 3.0 * i, 0.0, 0.0), t)
            (x, y, z, yaw), q, bob = ws[rid].step(t)
            qj, qy = qs[rid]
            n += qj.changed("pos", (x, y, z + bob)) + qy.changed("yaw", math.degrees(yaw))
            n += sum(qj.changed(j, v) for j, v in zip(LEGS, q))
            if rid == selected:
                n += sum(qj.changed(j, v) for j, v in zip(ARM, arm.step(entry, t)))
        counts.append(n)
    return counts


def test_three_walking_robots_stay_within_budget():
    c = run({"R1": 0.8, "R2": 1.0, "R3": 0.45}, 10.0, "R1", {"state": "WORK", "task": "rust:F1", "job_state": "SCRUB"})
    assert max(c[48:]) <= BUDGET
    assert np.mean(c[48:]) > 30


def test_standing_robots_cost_nothing_after_settling():
    c = run({"R1": 0.0, "R2": 0.0, "R3": 0.0}, 4.0)
    assert sum(c[24:]) == 0


def test_task_kind_mapping():
    assert task_kind({"state": "WORK", "task": "weld:S1", "job_state": "INSPECT"}) == "weld"
    assert task_kind({"state": "WORK", "task": "pick_can:C2", "job_state": "GRASP"}) == "pick_can"
    assert task_kind({"state": "WORK", "task": "rust:F3", "job_state": "SOAK"}) == "scrub"
    assert task_kind({"state": "WORK", "task": "gas:F1", "job_state": "LEAK_APPROACH"}) is None
    assert task_kind({"state": "MOVING", "task": "rust:F3", "job_state": None}) is None
    assert task_kind({"state": "IDLE"}) is None


def test_clip_pingpong_is_continuous_and_spans_phases():
    c = clip(60, ["APPROACH"] * 20 + ["DESCEND"] * 10 + ["LIFT"] * 10 + ["CARRY"] * 10 + ["RELEASE"] * 10)
    xs = np.array([c.pingpong(k / 24, 0.5) for k in range(24 * 12)])
    assert np.abs(np.diff(xs, axis=0)).max() < 0.05
    assert c.span(("DESCEND",)) == pytest.approx((20 / 30, 29 / 30))
    assert c.span(("CARRY", "LOWER", "RELEASE")) == pytest.approx((40 / 30, 59 / 30))
    assert c.span(("NOPE",)) == (0.0, 0.0)


def test_task_motion_blends_in_and_back_to_park():
    tm = TaskMotion({"scrub": clip()})
    park = np.array([PARK[j] for j in ARM])
    assert np.allclose(tm.step(None, 0.0), park)
    work = {"state": "WORK", "task": "rust:F1", "job_state": "SCRUB"}
    outs = [tm.step(work, 1.0 + k / 24) for k in range(72)]
    assert tm.source == "rec"
    assert np.abs(np.diff(np.array(outs), axis=0)).max() < 0.1
    assert not np.allclose(outs[-1], park)
    back = [tm.step(None, 4.0 + k / 24) for k in range(48)]
    assert np.abs(np.diff(np.array(back), axis=0)).max() < 0.15
    assert np.allclose(back[-1], park) and tm.source is None


def test_weld_sweep_weaves_near_work_pose():
    qs = np.array([weld_sweep(k / 24) for k in range(240)])
    assert np.ptp(qs[:, ARM.index("ur3_wrist_3_joint")]) > 0.3
    assert np.ptp(qs[:, ARM.index("lift_joint")]) == 0.0
    tm = TaskMotion({})
    tm.step({"state": "WORK", "task": "weld:S1", "job_state": "INSPECT"}, 0.0)
    assert tm.source.startswith("weld")


def test_pick_can_plays_phase_segment_then_holds():
    c = clip(60, ["APPROACH"] * 20 + ["DESCEND"] * 10 + ["LIFT"] * 10 + ["CARRY"] * 10 + ["RELEASE"] * 10)
    tm = TaskMotion({"pick_can": c})
    r = {"state": "WORK", "task": "pick_can:C1", "job_state": "GRASP"}
    tm.step(r, 0.0)
    for k in range(60):
        out = tm.step(r, k / 24)
    assert np.allclose(out, c.frame(29.0))


def test_trim_jump_cuts_at_reset():
    from rig.sim.taskmotion import export_clip, trim_jump

    rows = [{"t": k / 30, "q": [0.01 * k, 0.0], "phase": "A"} for k in range(10)] + [{"t": 0.4, "q": [3.0, 0.0], "phase": "B"}]
    assert len(trim_jump(rows)) == 10
    out = export_clip({"hz": 30.0, "joints": ["a", "b"], "rows": rows, "source": "s"}, every=2)
    assert out["hz"] == 15.0 and len(out["rows"]) == 5 and out["rows"][0]["phase"] == "A"
