import pytest

from fleet.contract import JobStatus, Level
from fleet.sim import new_job
from policy.control.fleet_port import HOLD_TTL_S, FleetPort
from policy.control.rigsensors import RigSensors
from policy.control.snapshot import SensorHealth
from tests.argon_world import Q, coordinator

SNIFF = {"camera", "gas_sensor"}


@pytest.fixture
def port():
    c = coordinator([("R1", Q, SNIFF, (-12.0, 20.0, 42.5), 1.0), ("R2", Q, SNIFF, (20.0, -8.0, 42.5), 0.2)])
    return FleetPort(c, RigSensors().specs)


def act(kind, robot="", target="", **kw):
    a = {"kind": kind, "robot": robot, "target": target, "confidence": 0.9, "rationale": "because", "source": "cloud",
         "rule": "", "purpose": "", "auto": True}
    a.update(kw)
    a["id"] = f"{a['source']}:{kind}:{robot}:{target}"
    return a


def refused(res) -> str:
    assert not res["ok"]
    return res["refused"]


def tick(port, n=1):
    for _ in range(n):
        port.c.tick(1.0)


def test_verify_action_queues_a_pinned_capable_job_on_the_coordinator(port):
    h = SensorHealth("G-F1", "gas", "STUCK", "N")
    port.apply([act("dispatch", "R1", "G-F1", purpose="verify_sensor")], {"G-F1": h})
    job = port.open_job("verify_sensor", "G-F1")
    j = port.c.jobs[job["id"]]
    assert j.params["robot"] == "R1" and j.needs == {"gas_sensor"} and j.priority == 2
    assert j.params["requested_by"] == "argon"
    port.apply([act("dispatch", "R1", "G-F1", purpose="verify_sensor")], {"G-F1": h})
    assert sum(x.kind == "verify_sensor" for x in port.c.jobs.values()) == 1
    for _ in range(300):
        tick(port)
        if port.c.status[j.id] == JobStatus.DONE:
            break
    assert port.c.status[j.id] == JobStatus.DONE and port.c.results[j.id].robot_id == "R1"
    assert sum(e["event"] == "fault_dispatch" for e in port.timeline) == 1


def test_argon_cannot_approve_its_own_supv_job(port):
    jid = port.queue("sensor_maintenance", "G-F1", "SUPV", 2, params={"needs": ["crew"]}, by="argon", why="flat head")
    assert "cannot approve its own SUPV job" in refused(port.request("approve_request", {"job_id": jid,
                                                                                         "why": "looks fine"}))
    assert not port.c.approve(jid, by="argon")
    assert port.job(jid)["status"] == "WAITING_APPROVAL"
    assert port.c.approve(jid, by="operator") and port.job(jid)["status"] == "QUEUED"


def test_argon_cannot_reset_an_esd(port):
    assert "ESD" in refused(port.request("queue_job", {"kind": "esd_reset", "target": "E", "level": "AUTO",
                                                       "priority": 1, "why": "clear it"}))
    port.esd = {"active": True, "zone": "E"}
    port.c.hold("R1")
    assert "ESD" in refused(port.request("resume_robot", {"robot": "R1", "why": "carry on"}))


def test_argon_cannot_override_the_force_limit(port):
    assert "40 N" in refused(port.request("queue_job", {"kind": "rust_repair", "target": "P-1", "level": "SUPV",
                                                        "priority": 3, "params": {"force_n": 55}, "why": "stubborn"}))
    assert "40 N" in refused(port.request("queue_job", {"kind": "force_override", "target": "R1", "level": "AUTO",
                                                        "priority": 3, "why": "x"}))


def test_argon_cannot_shrink_or_clear_the_person_pause(port):
    assert "3 m" in refused(port.request("queue_job", {"kind": "gauge_read", "target": "gauge_01", "level": "AUTO",
                                                       "priority": 3, "params": {"person_radius_m": 1.5},
                                                       "why": "tight"}))
    rt = port.robot("R1")
    port.c.world.people = [tuple(rt.robot.position)]
    tick(port)
    assert rt.state == "PAUSED_PERSON" and "R1" in port.fleet_state()["person_pauses"]
    assert "person" in refused(port.request("resume_robot", {"robot": "R1", "why": "carry on"}))
    port.apply([act("resume", "R1")], {})
    assert rt.state == "PAUSED_PERSON"


def test_argon_never_commands_motors(port):
    assert "never commands motors" in refused(port.request("set_joint_velocity", {"joint": 3, "why": "x"}))
    assert "never commands motors" in refused(port.request("queue_job", {
        "kind": "move", "target": "R1", "level": "AUTO", "priority": 3, "params": {"cmd_vel": [1, 0, 0]},
        "why": "x"}))
    res = port.apply([act("drive_motor", "R1", "R1")], {})
    assert "never commands motors" in refused(res[0])
    assert not port.c.jobs


def test_hold_resume_and_low_battery_refusal(port):
    res = port.apply([act("hold", "R1")], {})
    assert res[0]["ok"] and port.robot("R1").state == "HELD"
    res = port.apply([act("hold", "R2")], {})
    assert "return level" in refused(res[0]) and port.robot("R2").state != "HELD"
    res = port.apply([act("resume", "R1")], {})
    assert res[0]["ok"] and port.robot("R1").state == "IDLE"


def test_argon_cannot_hold_the_gas_responder(port):
    c = port.c
    c.submit(new_job(c, "flange_leak_check", "F1", tuple(c.planner.rm.flanges["F1"]), Level.AUTO, priority=1,
                     params={"gas_response": True}))
    tick(port)
    rid = next(rid for rid, rt in c.rt.items() if rt.job)
    res = port.apply([act("hold", rid)], {})
    assert "gas response" in refused(res[0]) and port.robot(rid).state != "HELD"


def test_argon_hold_expires_unless_renewed(port):
    port.apply([act("hold", "R1")], {})
    for _ in range(int(HOLD_TTL_S) + 2):
        tick(port)
        port.apply([], {})
    assert port.robot("R1").state != "HELD"
    assert any(e["event"] == "hold_expired" for e in port.timeline)


def test_reprioritise_and_cancel_reach_the_coordinator(port):
    jid = port.queue("gauge_read", "gauge_01", "AUTO", 3, by="argon", why="round")
    assert port.request("reprioritise", {"job_id": jid, "priority": 2, "why": "deadline"})["ok"]
    assert port.c.jobs[jid].priority == 2
    assert port.request("cancel_job", {"job_id": jid, "why": "not needed"})["ok"]
    assert port.job(jid)["status"] == "CANCELLED"
    pj = port.queue("gauge_read", "gauge_02", "AUTO", 1, by="argon", why="urgent")
    assert "safety job" in refused(port.request("cancel_job", {"job_id": pj, "why": "x"}))
