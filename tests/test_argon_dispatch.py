import json

import numpy as np
import pytest

from policy.control.backend import Reply
from policy.control.brain import compare
from policy.control.centre import ControlCentre
from policy.control.safety import arbitrate, edge_rules, hazard_first, verify_fallback
from policy.control.schema import Action, validate
from policy.control.snapshot import GasSensor, Robot, SensorHealth, SensorSnapshot
from tests.argon_world import DAY, A, C, Q, events, run, world

SNIFF = {"camera", "gas_sensor"}
ROBOTS = [("NOSNIFF", Q, {"camera"}, (-20.0, 21.0, 42.5), 1.0),
          ("LOW", Q, SNIFF, (-17.0, 21.0, 42.5), 0.152),
          ("MID", Q, SNIFF, (-21.0, 14.0, 31.25), 1.0),
          ("TOP", Q, SNIFF, (2.0, 20.0, 42.5), 1.0)]


def test_candidates_rank_by_rig_graph_with_capability_and_battery():
    b = world(ROBOTS)
    cands = b.candidates("G-F1")
    c = {x["robot"]: x for x in cands}
    assert c["NOSNIFF"]["why"] == "no gas_sensor" and not c["NOSNIFF"]["ok"]
    assert not c["LOW"]["ok"] and c["LOW"]["why"].startswith("battery") and c["LOW"]["need_wh"] > c["LOW"]["spare_wh"]
    assert c["MID"]["ok"] and c["TOP"]["ok"]
    pos = {rid: np.asarray(rt.robot.position) for rid, rt in b.c.rt.items()}
    head = np.asarray(b.rig.specs["G-F1"].pos)
    assert np.linalg.norm(pos["MID"] - head) < np.linalg.norm(pos["TOP"] - head)
    assert c["TOP"]["eta_s"] < c["MID"]["eta_s"]
    assert cands[0]["robot"] == "TOP"


def test_mixed_fleet_sends_the_unit_that_can_reach_the_head():
    b = world([("Q2", Q, {"camera", "thermal", "acoustic"}, (-20.0, 10.0, 42.5), 0.9),
               ("A1", A, {"camera", "thermal"}, (25.0, -25.0, 47.0), 0.9),
               ("C1", C, {"camera", "acoustic"}, (-3.0, -6.5, 42.5), 0.9)])
    flare = {x["robot"]: x for x in b.candidates("T-FLARE")}
    assert flare["A1"]["ok"] and not flare["Q2"]["ok"] and "cannot reach" in flare["Q2"]["why"]
    shell = {x["robot"]: x for x in b.candidates("A-V101")}
    assert shell["C1"]["ok"] and "cannot reach" in shell["Q2"]["why"] and "no acoustic" in shell["A1"]["why"]
    b.rig.inject("T-FLARE", "stuck", 0.0)
    b.rig.inject("A-V101", "dead", 0.0)
    assert run(b, 60, until=lambda: len(events(b, "fault_dispatch")) == 2)
    sent = {e["detail"]["sensor"]: e["robot"] for e in events(b, "fault_dispatch")}
    assert sent == {"T-FLARE": "A1", "A-V101": "C1"}
    assert run(b, 600, until=lambda: len(events(b, "maintenance_raised")) == 2)


@pytest.fixture
def stuck():
    b = world(ROBOTS)
    b.rig.inject("G-F1", "stuck", 0.0, value=0.0)
    run(b, 18)
    return b


def test_edge_mode_dispatches_nearest_capable_robot_and_raises_maintenance(stuck):
    run(stuck, 3)
    ev = events(stuck, "fault_dispatch")[0]
    assert ev["detail"]["source"] == "edge" and ev["robot"] == "TOP" and ev["detail"]["sensor"] == "G-F1"
    job = stuck.c.jobs[ev["job"]]
    assert job.params["robot"] == "TOP" and job.needs == {"gas_sensor"} and job.target == "G-F1"
    assert run(stuck, 300, until=lambda: events(stuck, "maintenance_raised"))
    assert not events(stuck, "verify_reading")[0]["detail"]["agree"]
    m = events(stuck, "maintenance_raised")[0]["job"]
    assert stuck.port.job(m)["status"] == "WAITING_APPROVAL" and stuck.port.job(m)["requested_by"] == "argon"
    run(stuck, 1)
    assert next(h for h in stuck.snap.health if h.id == "G-F1").job == m


def test_approved_maintenance_goes_to_the_crew_and_the_head_comes_back(stuck):
    stuck.operator_s, stuck.crew_s = 5.0, 20.0
    assert run(stuck, 400, until=lambda: events(stuck, "sensor_repaired"))
    assert events(stuck, "approved")[0]["detail"]["by"] == "operator"
    run(stuck, 30)
    assert next(h for h in stuck.snap.health if h.id == "G-F1").status == "OK"


def test_agreeing_robot_clears_the_fault():
    b = world([("TOP", Q, {"camera"}, (-10.0, 20.0, 42.5), 1.0)])
    b.rig.process_step("gauge_03", 0.0, 1.2)
    assert run(b, 300, until=lambda: events(b, "fault_cleared"))
    assert events(b, "verify_reading", sensor="gauge_03")[0]["detail"]["agree"]
    run(b, 2)
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
