import numpy as np

from policy.fleet.fleet import GAS_ZONE_M, HOT_WORK_M, Fleet
from policy.fleet.jobs import SETUP_S, JobLibrary, run_orchestrator
from policy.fleet.live import QUICK, FleetLive
from policy.fleet.rigmap import load_rig_map
from policy.fleet.scenario import JUDGES_WELD
from policy.fleet.tasks import KIND_BASE
from tests.test_fleet_sim import stub_jobs

WELD_XYZ = [10.0, -8.0, 42.5]


def test_weld_job_trace_from_library():
    tr = JobLibrary(run_orchestrator).get("weld", "crack")
    assert tr.kind == "weld" and tr.outcome == "DONE" and tr.total_s == SETUP_S["weld"] + tr.duration_s
    assert tr.state_at(0.0)[0] == "SETUP" and tr.state_at(SETUP_S["weld"] + 1)[0] == "INSPECT"
    assert run_orchestrator("weld", "gas").outcome == "ESCALATE_TO_HUMAN"
    assert QUICK["weld"][-1][1] == "DONE" and KIND_BASE["rust"] < KIND_BASE["weld"] < KIND_BASE["crack"]


def test_judges_has_a_weld_job_and_live_accepts_weld_requests():
    live = FleetLive("judges", speed=20.0, jobs=stub_jobs(15.0))
    assert any(t.kind == "weld" and t.poi == JUDGES_WELD[0] for t in live.f.tasks.values())
    out = live.event("weld_job", {"poi": "weld_extra", "xyz": WELD_XYZ, "scenario": "crack"})
    assert out["log"][0]["kind"] == "weld_job"
    assert "weld:weld_extra" in live.f.tasks


def test_hot_work_is_blocked_inside_wider_zone_around_gas():
    rm = load_rig_map()
    f = Fleet(rm, stub_jobs(30.0))
    f.add_robot("R1", rm.poi_node.get("F5", 0), 0.9)
    w = f.add_task("weld", "weld_near_f5", WELD_XYZ, scenario="crack")
    f.add_robot("R2", w.node, 0.9)
    d = float(np.linalg.norm(rm.pos[w.node][:2] - np.asarray(rm.flanges["F5"])[:2]))
    assert GAS_ZONE_M < d < HOT_WORK_M
    f.schedule(0.0, "gas_alarm", flange="F5", scenario="loose")
    for _ in range(40):
        f.step()
    assert f.tasks[w.id].status == "pending" and f.tasks[w.id].robot is None
    assert f.tasks["gas:F5"].robot == "R1"


def test_running_weld_is_suspended_when_gas_alarm_starts():
    rm = load_rig_map()
    f = Fleet(rm, stub_jobs(200.0))
    w = f.add_task("weld", "weld_near_f5", WELD_XYZ, scenario="crack")
    f.add_robot("R1", w.node, 0.9)
    f.add_robot("R2", rm.poi_node["F5"], 0.9)
    for _ in range(5):
        f.step()
    assert f.tasks[w.id].status == "active"
    f.schedule(f.t, "gas_alarm", flange="F5", scenario="loose")
    for _ in range(3):
        f.step()
    assert f.tasks[w.id].status == "pending"
    assert any(e["kind"] == "suspend" and "hot work" in e["message"] for e in f.event_log)
    assert f.tasks["gas:F5"].robot == "R2"
