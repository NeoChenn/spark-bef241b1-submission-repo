import pytest

from fleet.contract import JobStatus, Robot, RobotKind, World
from fleet.coordinator import Coordinator
from fleet.rigplanner import RigPlanner
from policy.control.fleet_adapter import FleetAdapter
from policy.fleet.rigmap import DECKS, DOCK_XY

Q = RobotKind.QUADRUPED
DOCK = (DOCK_XY[0], DOCK_XY[1], DECKS["top_deck"])
ZONE = (18.75, -5.75, 42.5)
SCRUBBER = {"arm", "spray", "brush", "camera", "gas_sensor"}
FIXER = {"arm", "gripper", "gas_sensor", "camera"}


def fleet(*extra):
    world = World.from_rig_meta()
    world.state["flanges"] = {}
    robots = [Robot("U1", Q, set(SCRUBBER), ZONE, battery=0.9), Robot("U2", Q, set(FIXER), DOCK, battery=0.95)]
    robots += [Robot(rid, Q, set(caps), xyz, battery=soc) for rid, caps, xyz, soc in extra]
    c = Coordinator(robots, world, {}, RigPlanner(), docks={r.id: DOCK for r in robots})
    return FleetAdapter(c)


def flange(a, fid):
    return tuple(a.c.planner.rm.flanges[fid])


def test_rust_goes_to_the_scrubber_and_leaks_to_the_fixer():
    a = fleet()
    r = a.assign("rust_repair", "main_pipe", ZONE, why="rust on the main pipe")
    assert r["ok"] and r["robot"] == "U1"
    leak = a.assign("flange_leak_check", "F5", flange(a, "F5"), why="G-F5 alarm")
    assert leak["ok"] and leak["robot"] == "U2"
    c = {x["robot"]: x for x in leak["candidates"]}
    assert "gripper" in c["U1"]["why"] and c["U1"]["eta_s"] < c["U2"]["eta_s"]
    v = a.assign("verify_sensor", "G-F1", flange(a, "F1"), why="G-F1 stuck")
    assert v["ok"] and v["robot"] == "U2" and a.c.jobs[v["job_id"]].needs >= {"gripper", "gas_sensor"}
    a.c.tick(1.0)
    assert a.c.assigned_to.get(r["job_id"]) == "U1"


def test_nearest_capable_fixer_with_battery_wins():
    a = fleet(("U3", FIXER, (18.0, -8.0, 42.5), 0.9), ("U4", FIXER, (19.0, -9.0, 42.5), 0.16))
    leak = a.assign("flange_leak_check", "F5", flange(a, "F5"), why="G-F5 alarm")
    c = {x["robot"]: x for x in leak["candidates"]}
    assert leak["robot"] == "U3" and not c["U4"]["ok"] and c["U4"]["why"].startswith("battery")
    assert c["U3"]["eta_s"] < c["U2"]["eta_s"]


def test_no_capable_robot_is_refused_not_given_to_the_scrubber():
    a = fleet()
    a.c.fail_robot("U2")
    r = a.assign("flange_leak_check", "F2", flange(a, "F2"), why="alarm")
    assert not r["ok"] and "no capable robot" in r["refused"]
    assert not r["candidates"][0]["ok"]
    assert "cannot take" in a.assign("flange_leak_check", "F2", flange(a, "F2"), robot="U1")["refused"]


def test_sensor_ticket_goes_to_the_human_inbox_and_argon_cannot_approve_it():
    a = fleet()
    t = a.ticket("G-F1", "G-F1 stuck high while the fixer reads clean", flange(a, "F1"), {"fixed": 31.0, "robot": 1.2})
    jid = t["job_id"]
    assert t["ok"] and t["status"] == "WAITING_APPROVAL" and t["inbox"]["needs_operator"]
    assert jid in a.c.approvals and a.c.status[jid] == JobStatus.WAITING_APPROVAL
    assert not a.c.approve(jid, by="argon")
    assert a.api.call("approve", {"job_id": jid})["refused"]
    assert a.c.approve(jid, by="operator")


def test_spark_rule_hold_pauses_rust_without_cancelling_it():
    a = fleet()
    r = a.assign("rust_repair", "main_pipe", ZONE, why="rust")
    a.c.tick(1.0)
    h = a.hold("U1", "spark rule: wire brushing near a gas alarm")
    assert h["ok"] and a.c.rt["U1"].state == "HELD" and a.c.status[r["job_id"]] in (JobStatus.ASSIGNED, JobStatus.RUNNING)
    for _ in range(5):
        a.c.tick(1.0)
    assert a.c.rt["U1"].job == r["job_id"]
    assert a.resume("U1", "zone clear")["ok"] and a.c.rt["U1"].state != "HELD"


def test_fixer_on_a_leak_cannot_be_held_by_argon():
    a = fleet()
    leak = a.assign("flange_leak_check", "F5", flange(a, "F5"), why="alarm")
    a.c.tick(1.0)
    assert "gas response" in a.hold("U2", "x")["refused"]
    assert a.c.rt["U2"].job == leak["job_id"]


@pytest.mark.parametrize("name,args,why", [
    ("assign", {"kind": "move", "target": "U1", "position": ZONE, "params": {"cmd_vel": [1, 0, 0]}}, "motors"),
    ("set_joint_torque", {}, "motors"),
    ("queue", {"kind": "esd_reset", "target": "E", "position": ZONE, "why": "x"}, "ESD"),
    ("queue", {"kind": "rust_repair", "target": "p", "position": ZONE, "why": "x", "params": {"force_n": 60}}, "40 N"),
    ("queue", {"kind": "rust_repair", "target": "p", "position": ZONE, "why": "x",
               "params": {"person_radius_m": 1.0}}, "person pause"),
    ("approve", {"job_id": "x"}, "operator only"),
])
def test_envelope_refusals_through_the_adapter(name, args, why):
    a = fleet()
    r = a.call(name, args)
    assert not r["ok"] and why in r["refused"]
