import math

import numpy as np
import pytest

from rig.hud import check_event
from rig.mission.live import MissionLive
from rig.sim.herocam import fixed_shot
from rig.sim.taskmotion import Rollout


def run_demo(times):
    m = MissionLive(obs=False)
    clock = [0.0]
    m.scen.clock = lambda: clock[0]
    m.event("demo", {"id": "can"})
    out = []
    for t in times:
        clock[0] = t
        m.advance(0.25)
        s = m.for_sim()
        r3 = next(r for r in s["robots"] if r["id"] == "R3")
        out.append((r3, s["hazards"]["obstacles"], m.scen.cur.i))
    return m, out


def test_demo_event_is_allowed():
    assert check_event({"name": "demo", "args": {"id": "can"}})["name"] == "demo"


def test_can_demo_timeline_is_scripted():
    m, out = run_demo([0.0, 1.0, 4.0, 7.0, 12.0, 30.0])
    assert m.scen.cur.id == "demo_can" and m.scen.selected == "R3"
    (r0, c0, _), (_, c1, _), (r2, _, _), (r3, c3, _), (_, _, i4), (r5, c5, i5) = out
    assert r0["pose"]["x"] == 1.4 and not c0
    assert len(c1) == 1 and c1[0]["phase"] == "ground" and c1[0]["radius"] == 0.0
    assert r2["state"] == "TRANSIT" and r2["path"] == [[r2["pose"]["x"], r2["pose"]["y"], r2["pose"]["z"]]]
    assert r3["job_state"] == "ROLLOUT" and r3["rollout"][0] < r3["rollout"][1] and c3[0]["phase"] == "held"
    assert c5[0]["phase"] == "tote" and r5["pose"]["x"] == 4.1 and r5["state"] == "IDLE" and i5 == 4 and i4 >= 2
    ro = Rollout.load()
    can = np.asarray(c1[0]["xyz"][:2]) - [3.5, -4.6]
    assert np.allclose(can, ro.can[40][:2], atol=1e-3)


def test_demo_pins_other_robots_and_resets():
    m, out = run_demo([0.0, 5.0])
    others = {rid: o["xy"].copy() for rid, o in m.override.items() if rid != "R3"}
    assert set(others) == {"R1", "R2"}
    m.event("demo", {"id": "reset"})
    assert m.scen.cur.id == "flyby" and not m.override
    assert not m.fleet.f.hz.obstacles
    with pytest.raises(ValueError):
        m.event("demo", {"id": "nope"})


def test_demo_start_queues_camera_select_and_head_cam():
    m = MissionLive(obs=False)
    m.event("demo", {"id": "can"})
    names = [e["name"] for e in m.take_sim()]
    assert names == ["set_weather", "camera", "select", "views"]


def test_rollout_can_starts_on_deck_and_ends_in_tote():
    ro = Rollout.load()
    assert ro.can[0][2] == pytest.approx(ro.radius, abs=0.005)
    end = ro.can[-1]
    assert abs(end[0] + 0.27) < 0.05 and abs(end[1]) < 0.05 and end[2] > ro.base_z
    assert ro.first("DESCEND") < ro.first("LIFT") < ro.first("DONE")


def test_fixed_shot_lens_overrides():
    eye, look, focal = fixed_shot((0.0, 0.0, 42.5), "zoom", None, {"dist": 5.0, "elev": 30.0, "focal": 20.0, "az": [90.0]})
    assert focal == 20.0 and math.dist(eye, look) == pytest.approx(5.0)
    assert eye[1] > 4.0
    eye, _, _ = fixed_shot((0.0, 0.0, 42.5), "zoom", None, {"eye": [1, 2, 3]})
    assert eye == (1.0, 2.0, 3.0)


def test_gas_demo_searches_isolates_and_stops_the_vapour():
    m = MissionLive(obs=False)
    clock = [0.0]
    m.scen.clock = lambda: clock[0]
    m.event("demo", {"id": "gas"})
    first = m.take_sim()
    assert [e["name"] for e in first] == ["set_weather", "camera", "select", "views", "hide_props"]
    assert "/World/Greebles/Zone/Clutter/tarp" in first[4]["args"]["paths"]
    assert first[3]["args"] == {"robot": "R2", "on": True, "kind": "fpv"}
    assert m.for_sim()["leak"]["on"]
    cams, xs, states, w = [], [], set(), 0.0
    while m.scen.cur.i < 4 and w < 30:
        w += 0.25
        clock[0] = w
        m.advance(0.25)
        cams += [e for e in m.take_sim() if e["name"] == "camera"]
        r2 = next(r for r in m.for_sim()["robots"] if r["id"] == "R2")
        xs.append((r2["pose"]["x"], r2["pose"]["y"]))
        states.add(r2.get("job_state"))
    s = m.scen.cur
    assert s.id == "demo_gas" and 15 < w < 22 and not m.leak.on and not m.for_sim()["leak"]["on"]
    assert {"LEAK_APPROACH", "VALVE_TURN"} <= states and not m.port.extra
    a = np.asarray(xs)
    assert a[:, 0].min() >= 17.2 and a[:, 0].max() <= 21.05 and -10.3 < a[:, 1].min() and a[:, 1].max() < -9.5
    assert len(cams) == 1 and cams[0]["args"]["lens"]["dist"] < first[1]["args"]["lens"]["dist"]
    assert [x["kind"] for x in m.scen.alerts] == ["gas_alarm"]
    m.event("demo", {"id": "reset"})
    assert "F5" in m.leak.flanges and "F7" not in m.leak.flanges
    assert {"name": "hide_props", "args": {"paths": []}} in m.take_sim()


def test_valve_turn_reaches_and_spins_the_wrist():
    from rig.sim.gait import ARM
    from rig.sim.taskmotion import TaskMotion

    t = TaskMotion.load()
    r = {"state": "WORK", "task": "gas:demo", "job_state": "VALVE_TURN"}
    a, b = t.step(r, 0.0), t.step(r, 2.0)
    q = t.step(r, 3.0)
    w = ARM.index("ur3_wrist_3_joint")
    assert q[w] - b[w] > 1.0 and t.source == "valve handwheel turn (scripted)" and not np.allclose(a, b)
