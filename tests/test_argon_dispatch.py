import json
from datetime import date

import numpy as np
import pytest

from policy.control.backend import Reply
from policy.control.brain import ArgonBrain, compare, register
from policy.control.centre import ControlCentre
from policy.control.fleet_port import FleetPort
from policy.control.rigsensors import RigSensors
from policy.control.safety import arbitrate, edge_rules, hazard_first, verify_fallback
from policy.control.schema import Action, SchemaError, validate
from policy.control.snapshot import GasSensor, Robot, SensorHealth, SensorSnapshot
from policy.fleet.fleet import Fleet
from policy.fleet.jobs import JobLibrary, JobTrace
from policy.fleet.rigmap import load_rig_map
from policy.fleet.scenario import nearest_node

DAY = date(2026, 11, 8)
SNIFF = ("camera", "gas_sniffer")


def stub_jobs(work_s=10.0):
    return JobLibrary(lambda kind, scenario: JobTrace(kind, scenario, work_s, "DONE",
                                                      [(0.0, "WORK", "stub"), (work_s, "DONE", "end")]))


def world(robots, backend=None, **kw):
    rm = load_rig_map()
    rig = RigSensors(seed=3)
    register(rm, rig)
    f = Fleet(rm, stub_jobs())
    for rid, xyz, soc, sensors in robots:
        f.add_robot(rid, nearest_node(rm, xyz), soc, sensors=sensors)
    port = FleetPort(f, rig.specs)
    centre = ControlCentre(backend, log_path=None, today=lambda: DAY, timeline=port, **kw)
    return ArgonBrain(f, rig, centre, port, operator_s=None)


def run(brain, n):
    for _ in range(int(n)):
        brain.step()


ROBOTS = [("NOSNIFF", [-20.0, 21.0, 42.5], 1.0, ("camera",)),
          ("LOW", [-17.0, 21.0, 42.5], 0.152, SNIFF),
          ("MID", [-21.0, 14.0, 31.25], 1.0, SNIFF),
          ("TOP", [2.0, 20.0, 42.5], 1.0, SNIFF)]


@pytest.fixture
def stuck():
    b = world(ROBOTS)
    b.rig.inject("G-F1", "stuck", 0.0, value=0.0)
    run(b, 18)
    return b


def test_candidates_rank_by_rig_graph_with_capability_and_battery():
    b = world(ROBOTS)
    run(b, 1)
    cands = b.candidates("G-F1", {})
    c = {x["robot"]: x for x in cands}
    assert c["NOSNIFF"]["why"] == "no gas_sniffer" and not c["NOSNIFF"]["ok"]
    assert not c["LOW"]["ok"]
    assert c["MID"]["ok"] and c["TOP"]["ok"]
    pos = {r.id: r.pos for r in b.f.robots}
    head = np.asarray(b.rig.specs["G-F1"].pos)
    assert np.linalg.norm(pos["MID"] - head) < np.linalg.norm(pos["TOP"] - head)
    assert c["TOP"]["eta_s"] < c["MID"]["eta_s"]
    assert cands[0]["robot"] == "TOP"


def test_battery_round_trip_rule():
    b = world([("LOW", [-17.0, 21.0, 42.5], 0.152, SNIFF), ("FAR", [20.0, -4.0, 42.5], 1.0, SNIFF)])
    run(b, 1)
    c = {x["robot"]: x for x in b.candidates("G-F1", {})}
    assert not c["LOW"]["ok"] and c["LOW"]["why"].startswith("battery") and c["LOW"]["need_wh"] > c["LOW"]["spare_wh"]
    assert c["FAR"]["ok"]


def test_edge_mode_dispatches_nearest_capable_robot_and_raises_maintenance(stuck):
    run(stuck, 1)
    job = next(j for j in stuck.port.jobs.values() if j["kind"] == "verify_sensor")
    assert job["params"]["robot"] == "TOP" and job["target"] == "G-F1"
    task = stuck.f.tasks[job["task"]]
    assert task.pin == "TOP" and task.needs == frozenset({"gas_sniffer"})
    ev = next(e for e in stuck.port.timeline if e["event"] == "fault_dispatch")
    assert ev["detail"]["source"] == "edge" and ev["robot"] == "TOP"
    run(stuck, 200)
    events = [e["event"] for e in stuck.port.timeline]
    assert "verify_reading" in events and "maintenance_raised" in events
    m = next(j for j in stuck.port.jobs.values() if j["kind"] == "sensor_maintenance")
    assert m["status"] == "WAITING_APPROVAL" and m["requested_by"] == "argon"
    assert next(h for h in stuck.snap.health if h.id == "G-F1").job == m["id"]


def test_agreeing_robot_clears_the_fault():
    b = world([("TOP", [2.0, 20.0, 42.5], 1.0, SNIFF)])
    b.rig.process_step("gauge_03", 0.0, 1.2)
    run(b, 60)
    assert any(e["event"] == "fault_dispatch" and e["detail"]["sensor"] == "gauge_03" for e in b.port.timeline)
    run(b, 200)
    ev = {e["event"]: e for e in b.port.timeline}
    assert ev["verify_reading"]["detail"]["agree"] and "fault_cleared" in ev
    assert next(h for h in b.snap.health if h.id == "gauge_03").status == "OK"


def test_compare_rules():
    flat = [0.0] * 15
    moving = list(1.6 + 0.3 * np.sin(np.arange(15) / 3))
    assert not compare("gas", moving, flat)["agree"]
    assert not compare("gas", moving, [])["agree"]
    assert not compare("pressure", [12.0] * 5, [13.0] * 5)["agree"]
    assert compare("pressure", [12.0, 12.01, 12.02], [12.01, 12.0, 12.02])["agree"]


def fault_snap(status="STUCK", hazard=False, lel=2.0, peer=2.0):
    cands = [{"robot": "R2", "sensor": "gas_sniffer", "eta_s": 40.0, "ok": True, "why": ""},
             {"robot": "R1", "sensor": "gas_sniffer", "eta_s": 10.0, "ok": False, "why": "no gas_sniffer"},
             {"robot": "R3", "sensor": "gas_sniffer", "eta_s": 60.0, "ok": True, "why": ""}]
    return SensorSnapshot(
        t=10.0, robots=[Robot("R1", zone="N"), Robot("R2", zone="E"), Robot("R3", zone="dock")],
        gas=[GasSensor("G-A", zone="E", lel=peer), GasSensor("G-B", zone="E", lel=lel)],
        health=[SensorHealth("G-B", "gas", status, "E", 5.0, {"peer_max": peer}, hazard, cands),
                SensorHealth("G-A", "gas", "OK", "E")])


def verify(robot="R2", target="G-B", **kw):
    return {"kind": "dispatch", "robot": robot, "target": target, "purpose": "verify_sensor", "confidence": 0.9,
            "rationale": "flat", **kw}


def test_schema_accepts_verify_dispatch_for_ok_candidate():
    plan = validate({"situation": "x", "actions": [verify()], "anomalies": []}, fault_snap())
    a = plan.actions[0]
    assert a.purpose == "verify_sensor" and a.target == "G-B" and a.label == "Send R2 to verify sensor G-B"


@pytest.mark.parametrize("item,why", [
    (verify(target="G-Z"), "unknown sensor"),
    (verify(target="G-A"), "healthy"),
    (verify(robot="R1"), "cannot verify"),
    (verify(robot="R9"), "not a candidate"),
    (verify(kind="hold"), "dispatch purpose"),
    (verify(purpose="drive_to"), "purpose"),
])
def test_schema_rejects_bad_verify(item, why):
    plan = validate({"situation": "x", "actions": [item], "anomalies": []}, fault_snap())
    assert not plan.actions and why in plan.rejected[0]["why"]


def test_hazard_first_treats_faulty_head_beside_gas_as_gas():
    snap = fault_snap("DEAD", hazard=True, lel=0.0, peer=55.0)
    first = hazard_first(snap)
    assert next(g for g in snap.gas if g.id == "G-B").lel == 55.0
    assert first[0].kind == "escalate" and first[0].rule == "hazard_first"
    edge = edge_rules(snap) + first
    assert any(a.kind == "evacuate" and a.target == "E" for a in edge)
    assert not [a for a in verify_fallback(snap, edge) if a.kind == "dispatch"]


def test_hazard_first_vetoes_verify_into_live_gas_below_evacuation():
    snap = fault_snap("DEAD", hazard=True, lel=0.0, peer=30.0)
    edge = edge_rules(snap) + hazard_first(snap)
    assert not any(a.kind == "evacuate" for a in edge)
    cloud = validate({"situation": "x", "actions": [verify()], "anomalies": []}, snap).actions
    final, vetoed = arbitrate(edge, cloud, snap)
    assert not any(a.purpose == "verify_sensor" for a in final)
    assert "hazard first" in vetoed[0]["why"]
    assert not [a for a in verify_fallback(snap, edge) if a.kind == "dispatch"]


def test_dead_gas_head_missing_from_snapshot_is_reinserted_for_gas_rules():
    snap = fault_snap("DEAD", hazard=True, lel=0.0, peer=60.0)
    snap.gas = [g for g in snap.gas if g.id != "G-B"]
    hazard_first(snap)
    assert next(g for g in snap.gas if g.id == "G-B").lel == 60.0


def test_fallback_picks_nearest_ok_candidate_not_nearest_overall():
    snap = fault_snap()
    a = verify_fallback(snap, [])
    assert len(a) == 1 and a[0].robot == "R2" and a[0].purpose == "verify_sensor" and a[0].source == "edge"


def test_fallback_skips_robots_locked_by_edge_rules_and_escalates_when_none_left():
    snap = fault_snap()
    lock = [Action("pause", r, "", 1.0, "person", source="edge", rule="person") for r in ("R2", "R3")]
    a = verify_fallback(snap, lock)
    assert a[0].kind == "escalate" and "no capable robot" in a[0].rationale


class Quiet:
    live, available, name = True, True, "gemini-4-argon"

    def __init__(self, text=None, error=None):
        self.text, self.error, self.n = text, error, 0

    def call(self, prompt, snap):
        self.n += 1
        if self.error:
            raise self.error
        return Reply(self.text or json.dumps({"situation": "ok", "actions": [], "anomalies": []}), 1000, 50, 0, 0,
                     900.0)


def edge_dispatch(h):
    return [a for a in h["actions"] if a["purpose"] == "verify_sensor" and a["source"] == "edge"]


def test_online_argon_gets_a_grace_window_then_edge_takes_over():
    c = ControlCentre(Quiet(), log_path=None, today=lambda: DAY, interval_s=100.0)
    assert not edge_dispatch(c.tick(fault_snap(), now=0.0))
    assert not edge_dispatch(c.tick(fault_snap(), now=20.0))
    assert edge_dispatch(c.tick(fault_snap(), now=26.0))[0]["robot"] == "R2"


def test_argon_verify_covers_the_fault_so_edge_stays_quiet():
    text = json.dumps({"situation": "x", "actions": [verify(robot="R3")], "anomalies": []})
    c = ControlCentre(Quiet(text), log_path=None, today=lambda: DAY)
    h = c.tick(fault_snap(), now=0.0)
    h = c.tick(fault_snap(), now=40.0)
    assert not edge_dispatch(h)
    assert any(a["purpose"] == "verify_sensor" and a["robot"] == "R3" and a["source"] == "cloud" for a in h["actions"])


@pytest.mark.parametrize("setup", ["cut", "offline", "capped", "none"])
def test_edge_dispatches_at_once_when_argon_is_cut_offline_or_capped(setup):
    backend = None if setup == "none" else Quiet(error=TimeoutError("x")) if setup == "offline" else Quiet()
    c = ControlCentre(backend, log_path=None, today=lambda: DAY, cap_usd_day=0.0 if setup == "capped" else 5.0)
    if setup == "cut":
        c.cut_link(True)
    h = c.tick(fault_snap(), now=0.0)
    assert edge_dispatch(h)[0]["robot"] == "R2"
