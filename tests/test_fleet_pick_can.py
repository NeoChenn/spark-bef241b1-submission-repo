import json
import math

import numpy as np
import pytest

from policy.fleet.fleet import Fleet
from policy.fleet.hazards import OBSTACLES
from policy.fleet.jobs import JobLibrary, JobTrace
from policy.fleet.live import FleetLive
from policy.fleet.rigmap import load_rig_map
from policy.fleet.scenario import CAPS, build_judges
from rig.hud import FLEET, SCENARIO, apply_event
from tests.test_fleet_sim import by_id, run_until, stub_jobs

GRIP = ("arm", "gripper")


def make(robots, work_s=10.0, jobs=None):
    rm = load_rig_map()
    f = Fleet(rm, jobs or stub_jobs(work_s))
    for rid, poi, caps in robots:
        f.add_robot(rid, rm.poi_node[poi] if isinstance(poi, str) else poi, 1.0, caps=caps)
    return f, rm


def pick(f):
    return next(t for t in f.tasks.values() if t.kind == "pick_can")


def walking(f, rid="R1"):
    f.add_task("leak_check", "F5")
    for _ in range(3):
        f.step()
    return by_id(f, rid)


def test_can_is_a_small_blocking_obstacle():
    assert OBSTACLES["can"] == ("block", 0.6)


def test_drop_can_dispatches_gripper_robot_and_clears_on_pickup():
    f, rm = make([("R1", "dock", GRIP)])
    walking(f)
    f.event("drop_can", at="R1")
    ob = next(iter(f.hz.obstacles.values()))
    t = pick(f)
    assert ob["kind"] == "can" and ob["task"] == t.id and t.poi == ob["id"] and ob["colour"] == "red"
    assert len(rm.closed) == 1 and not rm.trapped(t.node)
    assert np.linalg.norm(rm.pos[t.node][:2] - np.array(ob["xyz"][:2])) == pytest.approx(0.9, abs=0.05)
    f.step()
    assert t.robot == "R1" and any(e["kind"] == "dispatch" and e["robot"] == "R1" for e in f.event_log)
    snap = f.snapshot()
    js = next(x for x in snap["tasks"] if x["kind"] == "pick_can")
    assert js["xyz"] == pytest.approx(ob["xyz"], abs=0.01) and snap["hazards"]["obstacles"][0]["task"] == t.id
    json.dumps(snap)
    assert run_until(f, lambda: t.status == "done", 600)
    assert not f.hz.obstacles and not rm.closed
    assert any(e["kind"] == "clear" and ob["id"] in e["message"] for e in f.event_log)


def test_nearest_robot_with_gripper_is_sent_and_others_reroute():
    f, rm = make([("R1", "dock", ("arm",)), ("R2", "gauge_03", GRIP), ("R3", "gauge_05", GRIP)])
    walking(f)
    f.event("drop_can", at="R1")
    t = pick(f)
    eta, _ = f._plan_matrix([by_id(f, "R2"), by_id(f, "R3")], [t])
    f.step()
    assert t.robot == ("R2" if eta[0, 0] <= eta[1, 0] else "R3")
    assert by_id(f, "R1").task is None or by_id(f, "R1").task.kind != "pick_can"
    assert not any(rm.is_closed(lg.u, lg.v) for r in f.robots for lg in r.legs[len(f._kept(r)):])


def test_without_gripper_can_stays_and_task_waits():
    f, rm = make([("R1", "dock", ("arm",))])
    walking(f)
    f.event("drop_can", at="R1")
    for _ in range(30):
        f.step()
    assert pick(f).status == "pending" and f.hz.obstacles


def test_clear_cancels_pick_job():
    f, rm = make([("R1", "dock", GRIP)])
    walking(f)
    f.event("drop_can", at="R1")
    f.step()
    f.event("clear", id="all")
    t = pick(f)
    assert t.status == "cancelled" and by_id(f, "R1").task is not t
    assert run_until(f, lambda: all(x.status in ("done", "cancelled") for x in f.tasks.values()), 900)


def test_failed_pickup_escalates_and_can_stays():
    def run(kind, scenario):
        out = "ESCALATE_TO_HUMAN" if kind == "pick_can" else "DONE"
        return JobTrace(kind, scenario, 5.0, out, [(0.0, "WORK", "stub"), (5.0, out, "grasp failed")])
    f, rm = make([("R1", "dock", GRIP)], jobs=JobLibrary(run))
    walking(f)
    f.event("drop_can", at="R1")
    t = pick(f)
    assert run_until(f, lambda: t.status == "escalated", 600)
    assert f.hz.obstacles and rm.closed


def test_drop_can_on_walkway_id_and_yaw_arg():
    f, rm = make([("R1", "dock", GRIP)])
    f.event("drop_can", at="w3", yaw=0.3, colour="yellow")
    ob = next(iter(f.hz.obstacles.values()))
    assert ob["yaw"] == pytest.approx(0.3) and ob["colour"] == "yellow" and ob["at"] == "w3"
    assert math.isfinite(rm.route(rm.dock, pick(f).node)[0])


def test_drop_can_without_at_is_seeded_random_walkway_and_records_size():
    spots = []
    for _ in range(2):
        f, rm = make([("R1", "dock", GRIP)])
        f.event("drop_can", size="large", seed=5)
        ob = next(iter(f.hz.obstacles.values()))
        spots.append(ob["xyz"])
        assert ob["size"] == "large" and ob["at"] is None and pick(f).status == "pending"
    assert spots[0] == spots[1]
    f, rm = make([("R1", "dock", GRIP)])
    f.event("drop_can")
    assert next(iter(f.hz.obstacles.values()))["size"] == "small"
    with pytest.raises(ValueError):
        f.event("drop_can", size="huge")


def test_gas_preempts_can_pickup_but_can_does_not_preempt_gas():
    f, rm = make([("R1", "dock", GRIP)], work_s=200.0)
    walking(f)
    f.event("drop_can", at="R1")
    t = pick(f)
    assert run_until(f, lambda: t.status == "active", 600)
    f.schedule(f.t, "gas_alarm", flange="F2", scenario="loose")
    f.step()
    r = by_id(f, "R1")
    assert r.task.kind == "gas" and t.status == "pending"
    f.step()
    assert r.task.kind == "gas"


def test_judges_scenario_gives_grippers_and_live_event_works():
    f = build_judges(disturb=False)
    assert {r.id for r in f.robots if "gripper" in r.caps} == {rid for rid, c in CAPS.items() if "gripper" in c}
    assert any("gripper" in r.caps for r in f.robots)
    live = FleetLive("judges", speed=20.0, jobs=stub_jobs(15.0))
    live.advance(0.5)
    out = live.event("drop_can", {"at": "R1"})
    assert out["log"][0]["kind"] == "drop_can"
    live.advance(0.2)
    s = live.state()
    assert s["hazards"]["obstacles"][0]["kind"] == "can"
    assert any(x["kind"] == "pick_can" and x["robot"] for x in s["tasks"])


def test_hud_whitelists_drop_can_and_routes_to_fleet():
    assert "drop_can" in FLEET and "drop_can" in SCENARIO
    calls = []
    fl = type("F", (), {"event": lambda self, n, a: calls.append((n, a))})()
    apply_event(object(), {"name": "drop_can", "args": {"at": "R1"}}, fleet=fl)
    assert calls == [("drop_can", {"at": "R1"})]
