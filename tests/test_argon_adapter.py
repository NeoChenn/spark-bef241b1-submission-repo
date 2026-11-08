import pytest

from policy.control.fleet_port import HOLD_TTL_S, FleetPort
from policy.control.rigsensors import RigSensors
from policy.control.snapshot import SensorHealth
from policy.fleet.fleet import Fleet
from policy.fleet.jobs import JobLibrary, JobTrace
from policy.fleet.rigmap import load_rig_map
from policy.control.brain import register

SNIFF = frozenset({"camera", "gas_sniffer"})


@pytest.fixture
def port(tmp_path):
    rm = load_rig_map()
    rig = RigSensors()
    register(rm, rig)
    f = Fleet(rm, JobLibrary(lambda k, s: JobTrace(k, s, 10.0, "DONE", [(0.0, "WORK", "stub")])))
    f.add_robot("R1", rm.poi_node["gauge_03"], 1.0, sensors=SNIFF)
    f.add_robot("R2", rm.poi_node["gauge_06"], 0.2, sensors=SNIFF)
    return FleetPort(f, rig.specs, timeline_path=tmp_path / "timeline.jsonl")


def act(kind, robot="", target="", **kw):
    a = {"kind": kind, "robot": robot, "target": target, "confidence": 0.9, "rationale": "because", "source": "cloud",
         "rule": "", "purpose": "", "auto": True}
    a.update(kw)
    a["id"] = f"{a['source']}:{kind}:{robot}:{target}"
    return a


def refused(res) -> str:
    assert not res["ok"]
    return res["refused"]


def test_verify_action_queues_a_pinned_capable_job_on_the_fleet_planner(port):
    h = SensorHealth("G-F1", "gas", "STUCK", "N")
    port.apply([act("dispatch", "R1", "G-F1", purpose="verify_sensor")], {"G-F1": h})
    job = port.open_job("verify_sensor", "G-F1")
    t = port.f.tasks[job["task"]]
    assert t.pin == "R1" and t.needs == frozenset({"gas_sniffer"}) and job["priority"] == 2
    port.apply([act("dispatch", "R1", "G-F1", purpose="verify_sensor")], {"G-F1": h})
    assert sum(j["kind"] == "verify_sensor" for j in port.jobs.values()) == 1
    for _ in range(300):
        port.f.step()
        if t.status == "done":
            break
    assert t.status == "done" and t.robot is None and port.job(job["id"])["status"] == "DONE"
    assert (port.path.read_text().count("fault_dispatch")) == 1


def test_argon_cannot_approve_its_own_supv_job(port):
    jid = port.queue("sensor_maintenance", "G-F1", "SUPV", 2, by="argon", why="head is flat")
    assert "cannot approve its own SUPV job" in refused(port.request("approve_request", {"job_id": jid,
                                                                                         "why": "looks fine"}))
    assert port.job(jid)["status"] == "WAITING_APPROVAL"
    assert port.approve_request(jid, "operator", "ok") and port.job(jid)["status"] == "HANDED_TO_HUMAN"
    assert any(e["event"] == "agent_refused" for e in port.timeline)


def test_argon_cannot_reset_an_esd(port):
    assert "ESD" in refused(port.request("queue_job", {"kind": "esd_reset", "target": "E", "level": "AUTO",
                                                       "priority": 1, "why": "clear it"}))
    jid = port.queue("esd_hold", "E", "SUPV", 1, by="operator", why="ESD")
    assert "ESD" in refused(port.request("approve_request", {"job_id": jid, "why": "reset"}))
    port.esd = {"active": True, "zone": "E"}
    port.f.hold("R1")
    assert "ESD" in refused(port.request("resume_robot", {"robot": "R1", "why": "carry on"}))


def test_argon_cannot_override_the_force_limit(port):
    assert "40 N" in refused(port.request("queue_job", {"kind": "rust_repair", "target": "P-1", "level": "SUPV",
                                                        "priority": 3, "params": {"force_n": 55}, "why": "stubborn"}))
    assert "40 N" in refused(port.request("queue_job", {"kind": "force_override", "target": "R1", "level": "AUTO",
                                                        "priority": 3, "why": "x"}))


def test_argon_cannot_shrink_or_clear_the_person_pause(port):
    assert "3 m" in refused(port.request("queue_job", {"kind": "inspect", "target": "gauge_01", "level": "AUTO",
                                                       "priority": 3, "params": {"person_radius_m": 1.5},
                                                       "why": "tight"}))
    r1 = port.robot("R1")
    port.f.schedule(port.f.t, "person", xyz=r1.pos.tolist(), radius=3.0, duration=60.0)
    for _ in range(3):
        port.f.step()
    assert r1.paused and "R1" in port.fleet_state()["person_pauses"]
    assert "person" in refused(port.request("resume_robot", {"robot": "R1", "why": "carry on"}))
    out = port.apply([act("resume", "R1")], {})
    assert out == [] or not out[0]["ok"]
    assert r1.paused


def test_argon_never_commands_motors(port):
    assert "never commands motors" in refused(port.request("set_joint_velocity", {"joint": 3, "why": "x"}))
    assert "never commands motors" in refused(port.request("queue_job", {
        "kind": "move", "target": "R1", "level": "AUTO", "priority": 3, "params": {"cmd_vel": [1, 0, 0]},
        "why": "x"}))
    res = port.apply([act("drive_motor", "R1", "R1")], {})
    assert "never commands motors" in refused(res[0])
    assert not port.jobs


def test_hold_resume_and_low_battery_refusal(port):
    res = port.apply([act("hold", "R1")], {})
    assert res[0]["ok"] and port.robot("R1").held
    res = port.apply([act("hold", "R2")], {})
    assert "return level" in refused(res[0]) and not port.robot("R2").held
    res = port.apply([act("resume", "R1")], {})
    assert res[0]["ok"] and not port.robot("R1").held


def test_argon_cannot_hold_the_gas_responder(port):
    port.f.schedule(port.f.t, "gas_alarm", flange="F5", scenario="loose")
    port.f.step()
    r = next(x for x in port.f.robots if x.task is not None and x.task.kind == "gas")
    res = port.apply([act("hold", r.id)], {})
    assert "gas response" in refused(res[0]) and not r.held


def test_argon_hold_expires_unless_renewed(port):
    port.apply([act("hold", "R1")], {})
    for _ in range(int(HOLD_TTL_S) + 2):
        port.f.step()
        port.apply([], {})
    assert not port.robot("R1").held
    assert any(e["event"] == "hold_expired" for e in port.timeline)


def test_reprioritise_and_cancel_reach_the_planner(port):
    jid = port.queue("inspect", "gauge_01", "AUTO", 3, by="argon", why="round")
    t = port.f.tasks[port.jobs[jid]["task"]]
    assert port.request("reprioritise", {"job_id": jid, "priority": 1, "why": "deadline"})["ok"]
    assert t.priority_boost > 0
    assert "safety job" in refused(port.request("cancel_job", {"job_id": jid, "why": "x"}))
    assert port.reprioritise(jid, 2, "operator", "relax")
    assert port.request("cancel_job", {"job_id": jid, "why": "not needed"})["ok"]
    assert t.status == "cancelled" and port.job(jid)["status"] == "CANCELLED"
