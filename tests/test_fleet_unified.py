import json
import math
from pathlib import Path

import pytest

from fleet.contract import Job, JobResult, JobStatus, Level, Robot, RobotKind, World
from fleet.coordinator import GAS_ZONE_M, POWER, RESERVE, Coordinator, Leg, Route
from fleet.jobs.flange import FlangeLeakCheck, FlangeRetorque
from fleet.rigplanner import RigPlanner
from fleet.state import STATE_V1, snapshot
from policy.fleet.jobs import JobTrace
from policy.fleet.rigmap import DECKS

Q, C, A = RobotKind.QUADRUPED, RobotKind.CRAWLER, RobotKind.AERIAL
TOP, MID, LOW = DECKS["top_deck"], DECKS["middle_deck"], DECKS["lower_deck"]
SAMPLE = Path(__file__).resolve().parent.parent / "runs_sample/fleet"


class Exec:
    def __init__(self, kind, level=Level.AUTO, needs=("camera",), kinds=(Q,), duration=10.0, phases=()):
        self.kind, self.level, self.needs, self.robot_kinds = kind, level, set(needs), set(kinds)
        self.duration, self.phases, self.calls = duration, list(phases), []

    def run(self, job, robot, world):
        self.calls.append((job.id, robot.id))
        return JobResult(job.id, robot.id, JobStatus.DONE, world.t, world.t + self.duration, phases=self.phases)


def job(jid, pos, kind="look", target=None, **kw):
    return Job(jid, kind, target or jid, tuple(pos), kw.pop("level", Level.AUTO), **kw)


def run(c, n):
    for _ in range(int(n)):
        c.tick(1.0)


def events(c, name):
    return [e for e in c.timeline if e["event"] == name]


@pytest.fixture(scope="module")
def planner():
    return RigPlanner()


def test_rig_planner_routes_across_decks_on_stairs(planner):
    r = Robot("q", Q, set(), (-12.0, 9.4, TOP))
    route = planner.route(r, (20.0, 6.0, LOW + 1.0), 0.8)
    assert not route.blocked
    for a, b in zip(route.legs, route.legs[1:]):
        assert math.dist(a.end, b.start) < 1e-6
    stairs = [l for l in route.legs if l.kind == "stairs"]
    assert stairs and all(l.segment and l.segment.startswith("stairs") and l.speed < 1.0 for l in stairs)
    assert route.legs[-1].end[2] == pytest.approx(LOW)
    walk = Leg((0, 0, 0), (10, 0, 0))
    assert Route(stairs[:1]).energy_wh(POWER[Q], 1.0) / stairs[0].length > Route([walk]).energy_wh(POWER[Q], 1.0) / 10
    assert route.time(1.0) > route.length


def test_coordinator_walks_a_robot_down_to_the_lower_deck(planner):
    q = Robot("q1", Q, {"camera"}, (-12.0, 9.4, TOP))
    c = Coordinator([q], World(), {"look": Exec("look", duration=5)}, planner=planner)
    c.submit(job("low", (20.0, 6.0, LOW + 1.0)))
    run(c, 600)
    assert c.status["low"] == JobStatus.DONE
    assert c.rt["q1"].dist_m > 50 and c.rt["q1"].used_wh > 0
    assert any(abs(e["detail"].get("route_m", 0)) > 50 for e in events(c, "assigned"))


def test_hungarian_beats_sequential_greedy():
    robots = [Robot("r1", Q, {"camera"}, (0, 0, 0)), Robot("r2", Q, {"camera"}, (6, 0, 0))]
    c = Coordinator(robots, World(), {"look": Exec("look")})
    c.submit(job("j1", (5, 0, 0), params={"standoff": 0.0}))
    c.submit(job("j2", (12, 0, 0), params={"standoff": 0.0}))
    c.tick()
    assert c.assigned_to == {"j1": "r1", "j2": "r2"}


def test_priority_stays_strict_inside_the_assignment():
    c = Coordinator([Robot("r1", Q, {"camera"}, (0, 0, 0))], World(), {"look": Exec("look")})
    c.submit(job("near", (2, 0, 0), priority=5))
    c.submit(job("far", (200, 0, 0), priority=1))
    c.tick()
    assert c.assigned_to == {"far": "r1"}


def test_battery_feasibility_counts_the_walk_back_to_the_dock():
    c = Coordinator([Robot("r1", Q, {"camera"}, (0, 0, 0), battery=0.2)], World(), {"look": Exec("look")})
    rt = c.rt["r1"]
    far = job("far", (300, 0, 0), est_duration_s=60, params={"standoff": 0.0})
    p = POWER[Q]
    one_way = (300 * p.move_w + 60 * p.work_w) / 3600 / p.capacity_wh
    assert 0.2 - one_way >= RESERVE
    assert c.cost(far, rt) is None
    assert c.cost(job("near", (20, 0, 0), est_duration_s=60), rt) is not None


def test_dock_pads_queue_the_third_robot():
    robots = [Robot(f"q{i}", Q, {"camera"}, (0, 0, 0), battery=0.1) for i in range(3)]
    c = Coordinator(robots, World(), {}, docks={r.id: (0, 0, 0) for r in robots}, dock_pads=2)
    run(c, 3)
    states = sorted(rt.state for rt in c.rt.values())
    assert states == ["CHARGING", "CHARGING", "DOCK_QUEUE"] and events(c, "dock_queue")
    run(c, 2000)
    assert all(rt.state == "IDLE" for rt in c.rt.values()) and all(r.battery > 0.85 for r in robots)
    assert sum(rt.charged_wh for rt in c.rt.values()) > 0


def test_gas_zone_holds_routine_work_until_isolated():
    world = World()
    q1 = Robot("q1", Q, {"camera"}, (0, 0, 0))
    q2 = Robot("q2", Q, {"gas_sensor"}, (30, 0, 0))
    c = Coordinator([q1, q2], world, {"look": Exec("look", duration=50), "gas_survey": Exec("gas_survey", needs=("gas_sensor",))})
    c.submit(job("routine", (3, 0, 0)))
    run(c, 3)
    assert c.status["routine"] in (JobStatus.ASSIGNED, JobStatus.RUNNING)
    world.state["gas_leaks"] = [{"id": "L1", "position": (4.0, 0.0, 0.0), "reported": True}]
    c.submit(job("survey", (4.0, 0, 0), kind="gas_survey", priority=1))
    run(c, 1)
    assert c.status["routine"] == JobStatus.QUEUED and events(c, "gas_zone_hold")
    assert "gas zone" in events(c, "priority_change")[-1]["detail"]["explanation"]
    run(c, 60)
    assert c.status["routine"] == JobStatus.QUEUED and "routine" not in c.assigned_to
    assert c.status["survey"] == JobStatus.DONE
    world.state["gas_leaks"][0]["isolated"] = True
    run(c, 120)
    assert c.status["routine"] == JobStatus.DONE
    assert GAS_ZONE_M == 8.0


def test_robot_failure_requeues_and_another_robot_finishes():
    robots = [Robot("q1", Q, {"camera"}, (0, 0, 0)), Robot("q2", Q, {"camera"}, (20, 0, 0))]
    c = Coordinator(robots, World(), {"look": Exec("look", duration=5)})
    c.submit(job("j", (5, 0, 0)))
    run(c, 2)
    assert c.assigned_to["j"] == "q1"
    assert c.fail_robot("q1", "battery fault") and not c.fail_robot("q1")
    held = robots[0].position
    run(c, 60)
    assert c.status["j"] == JobStatus.DONE and c.results["j"].robot_id == "q2"
    assert c.rt["q1"].state == "FAILED" and robots[0].position == held
    assert events(c, "robot_failure")[0]["detail"]["reason"] == "battery fault"


def test_person_on_the_route_triggers_a_replan(planner):
    q = Robot("q1", Q, {"camera"}, (-21.0, -6.0, MID))
    world = World()
    c = Coordinator([q], world, {"look": Exec("look", duration=5)}, planner=RigPlanner(planner.rm))
    c.submit(job("j", (22.0, 14.0, MID), params={"standoff": 0.0}))
    run(c, 3)
    legs = c.rt["q1"].remaining
    mid = legs[len(legs) // 2]
    world.people = [mid.end]
    run(c, 1)
    assert events(c, "replanned")
    blocked = c.planner.blocked
    assert blocked and not any(l.node in blocked for l in c.rt["q1"].remaining[1:])
    world.people = []
    run(c, 400)
    assert c.status["j"] == JobStatus.DONE


def test_snapshot_is_fleet_state_v1_and_readable_by_captions(planner):
    from rig.captions import fleet_robots
    robots = [Robot("q1", Q, {"camera"}, (-12.0, 9.4, TOP)), Robot("a1", A, {"camera"}, (0, 0, 50.0))]
    ph = [(0.0, "SPRAY", "spraying"), (3.0, "SOAK", "soaking")]
    c = Coordinator(robots, World(), {"look": Exec("look", kinds=(Q,), duration=30, phases=ph)}, planner=planner)
    c.submit(job("j", (-5.0, 9.4, TOP), params={"standoff": 0.5}))
    c.submit(job("s", (0, 0, TOP), kind="weld", level=Level.SUPV))
    for _ in range(200):
        c.tick()
        if c.rt["q1"].state == "WORKING" and c.t - c.rt["q1"].work_t0 > 5:
            break
    snap = snapshot(c)
    sample = json.loads((SAMPLE / "fleet_state.json").read_text())
    assert snap["schema"] == sample["schema"] == "fleet_state.v1"
    assert set(sample) <= set(snap)
    assert set(sample["robots"][0]) <= set(snap["robots"][0])
    assert set(sample["tasks"][0]) <= set(snap["tasks"][0])
    assert set(sample["metrics"]) <= set(snap["metrics"])
    q1 = snap["robots"][0]
    assert q1["state"] == "WORK" and q1["job_state"] == "SOAK" and q1["deck"] == "top_deck"
    assert snap["robots"][1]["deck"] == "air"
    assert snap["approvals"][0]["job"] == "s"
    assert set(STATE_V1.values()) | {"DOCKED"} >= {r["state"] for r in snap["robots"]}
    rs = fleet_robots(snap)
    assert rs[0]["state"] == "SOAK" and json.dumps(snap)
    m = json.loads((SAMPLE / "fleet_map.json").read_text())
    assert set(m) == set(planner.to_json())


def stub_library(outcome, states):
    class Lib:
        def get(self, kind, scenario):
            segs = [(float(i), s, f"{s.lower()} step") for i, s in enumerate(states)]
            return JobTrace(kind, scenario, float(len(states) - 1), outcome, segs,
                            [{"kind": "x", "message": "m"}], len(segs))
    return Lib()


@pytest.mark.parametrize("states,outcome,status,finding,follow", [
    (["LEAK_INSPECT", "LEAK_INSPECT", "DONE"], "DONE", JobStatus.DONE, None, None),
    (["LEAK_INSPECT", "RETORQUE", "LEAK_VERIFY", "DONE"], "DONE", JobStatus.FINDING, "flange_leak", Level.SUPV),
    (["LEAK_INSPECT", "ESCALATE_TO_HUMAN"], "ESCALATE_TO_HUMAN", JobStatus.FINDING, "flange_crack", Level.HUMAN),
])
def test_flange_leak_check_maps_the_orchestrator_trace(states, outcome, status, finding, follow):
    ex = FlangeLeakCheck(stub_library(outcome, states))
    j = Job("fl", "flange_leak_check", "F3", (0, 0, 0), Level.AUTO)
    r = ex.run(j, Robot("q", Q, {"camera", "gas_sensor"}), World(t=5.0))
    assert r.status == status and r.phases[0][1] == "SETUP"
    assert all(st in ("SETUP", "LEAK_INSPECT") for _, st, _ in r.phases)
    if finding:
        assert r.findings[0].kind == finding and r.findings[0].follow_up[0].level == follow
    else:
        assert not r.findings


def test_flange_retorque_is_supv_and_clears_the_flange():
    world = World(state={"flanges": {"F3": "loose"}})
    ex = FlangeRetorque(stub_library("DONE", ["LEAK_INSPECT", "RETORQUE", "LEAK_VERIFY", "DONE"]))
    assert ex.level == Level.SUPV and "arm" in ex.needs
    r = ex.run(Job("t", "flange_retorque", "F3", (0, 0, 0), Level.SUPV), Robot("q", Q, {"arm"}), world)
    assert r.status == JobStatus.DONE and world.state["flanges"]["F3"] == "none"
    assert [p[1] for p in r.phases][1:] == ["RETORQUE", "LEAK_VERIFY", "DONE"]
    bad = FlangeRetorque(stub_library("ESCALATE_TO_HUMAN", ["LEAK_INSPECT", "RETORQUE", "ESCALATE_TO_HUMAN"]))
    r = bad.run(Job("t2", "flange_retorque", "F1", (0, 0, 0), Level.SUPV), Robot("q", Q, {"arm"}), World())
    assert r.status == JobStatus.HANDED_TO_HUMAN and r.findings[0].follow_up[0].kind == "flange_replace"


def test_real_orchestrator_loose_flange_check_spawns_supv_retorque():
    pytest.importorskip("torch")
    r = FlangeLeakCheck().run(Job("fl", "flange_leak_check", "F3", (0, 0, 0), Level.AUTO, params={"leak": "loose"}),
                              Robot("q", Q, {"camera", "gas_sensor"}), World())
    assert r.status == JobStatus.FINDING and r.findings[0].follow_up[0].kind == "flange_retorque"


def test_rig_demo_shift_writes_v1_states_and_runs_the_flange_chain(tmp_path):
    pytest.importorskip("torch")
    from fleet import demo
    sim, loaded, _ = demo.build(offline=True, timeline_path=tmp_path / "t.jsonl", states_path=tmp_path / "s.jsonl")
    assert "flange" in loaded
    sim.run(600)
    c = sim.c
    torque = [j for j, job in c.jobs.items() if job.kind == "flange_retorque"]
    assert torque and c.status[torque[0]] == JobStatus.DONE
    assert [e["event"] for e in c.timeline if e["job"] == torque[0]][:3] == ["job_queued", "approval_requested", "approved"]
    assert events(c, "gas_zone_hold")
    lines = (tmp_path / "s.jsonl").read_text().splitlines()
    assert len(lines) == 600 and json.loads(lines[-1])["schema"] == "fleet_state.v1"
    assert all(set(json.loads(x)["robots"][0]) >= {"id", "pose", "state", "battery", "path"} for x in lines[::100])


def test_legacy_walkway_planner_demo_still_runs(tmp_path):
    pytest.importorskip("torch")
    from fleet import demo
    sim, _, _ = demo.build(offline=True, timeline_path=tmp_path / "t.jsonl", planner="walkway")
    sim.run(120)
    assert (tmp_path / "t.jsonl").exists() and sim.c.counts().get("DONE", 0) >= 1


def test_risk_priority_orders_gas_crack_rust_flange_inspect():
    from fleet.risk import risk_priority
    order = [risk_priority("gas"), risk_priority("crack"), risk_priority("rust", 0.9, 1.0),
             risk_priority("leak_check"), risk_priority("inspect")]
    assert order == sorted(order) and order[0] == 1 and order[-1] == 5
    assert risk_priority("rust", 0.2, 0.2) > risk_priority("rust", 0.9, 1.0)


def test_deadline_pulls_routine_work_forward():
    c = Coordinator([Robot("r1", Q, {"camera"}, (0, 0, 0))], World(), {"look": Exec("look")})
    a = job("a", (5, 0, 0), priority=4)
    b = job("b", (5, 1, 0), priority=5, params={"deadline": 1000.0})
    c.submit(a)
    c.submit(b)
    assert c.effective_priority(b) == 5.0
    c.world.t = 1000.0
    assert c.effective_priority(b) == 3.0 and [j.id for j in c.queued()] == ["b", "a"]


def test_job_cut_off_by_a_failed_robot_goes_to_the_crew(planner):
    rm = planner.rm
    gauge = next(p for p in rm.inspection if p["type"] == "gauge")
    node = rm.poi_node[gauge["id"]]
    robots = [Robot("q1", Q, {"camera"}, (-12.0, 9.4, TOP)), Robot("q2", Q, {"camera"}, (-12.0, 9.4, TOP))]
    c = Coordinator(robots, World(), {"look": Exec("look", duration=500)}, planner=RigPlanner(rm))
    c.submit(job("g", gauge["position_m"]))
    for _ in range(300):
        c.tick()
        if c.status["g"] == JobStatus.RUNNING:
            break
    rid = c.assigned_to["g"]
    c.rt[rid].robot.position = tuple(float(v) for v in rm.pos[node])
    c.fail_robot(rid)
    run(c, 200)
    crew = events(c, "handed_to_crew")
    assert c.status["g"] == JobStatus.HANDED_TO_HUMAN and "no safe route" in crew[0]["detail"]["reason"]


def test_rigshift_runs_the_policy_fleet_scenario_on_the_unified_coordinator(tmp_path):
    pytest.importorskip("torch")
    from fleet import rigshift
    sim, loaded, _ = rigshift.build(3, True, tmp_path / "t.jsonl", tmp_path / "s.jsonl")
    rigshift.run(sim, 2400)
    c = sim.c
    names = [e["event"] for e in c.timeline]
    for ev in ("gas_leak_injected", "gas_alarm", "robot_failure", "crack_reported", "new_rust", "replanned",
               "approval_requested", "approved", "priority_change"):
        assert ev in names, ev
    assert not c.approvals and rigshift.finished(sim)
    m = snapshot(c)["metrics"]
    assert m["gas_alarms"] == 2 and m["gas_response_mean_s"] is not None and m["corridor_conflicts"] == 0
    assert m["robot_failures"] == 1 and m["open_tasks"] == 0
    assert "flange_replace" in {c.jobs[j].kind for j in c.crew}
    decks = {json.loads(x)["robots"][0]["deck"] for x in (tmp_path / "s.jsonl").read_text().splitlines()}
    assert {"top_deck", "lower_deck"} <= decks
