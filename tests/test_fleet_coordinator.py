import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fleet.contract import Finding, Job, JobResult, JobStatus, Level, Robot, RobotKind, World
from fleet.coordinator import PERSON_RADIUS_M, Coordinator
from fleet.jobs.rust_repair import RustRepairJob
from fleet.sim import FleetSim, WalkwayPlanner, Walkways, patrol_route

Q, C, A, R = RobotKind.QUADRUPED, RobotKind.CRAWLER, RobotKind.AERIAL, RobotKind.ROV


class Exec:
    def __init__(self, kind, level=Level.AUTO, needs=("camera",), kinds=(Q,), status=JobStatus.DONE,
                 findings=None, duration=10.0):
        self.kind, self.level, self.needs, self.robot_kinds = kind, level, set(needs), set(kinds)
        self.status, self.findings, self.duration, self.calls = status, findings or (lambda job: []), duration, []

    def run(self, job, robot, world):
        self.calls.append((job.id, robot.id))
        return JobResult(job.id, robot.id, self.status, world.t, world.t + self.duration, self.findings(job))


def job(jid, kind="look", target="a", pos=(0, 0, 0), level=Level.AUTO, **kw):
    return Job(jid, kind, target, pos, level, **kw)


def run(c, seconds):
    for _ in range(int(seconds)):
        c.tick(1.0)


def events(c, name):
    return [e for e in c.timeline if e["event"] == name]


# ---------------------------------------------------------------- assignment
def test_nearest_capable_robot_wins_the_auction():
    robots = [Robot("q1", Q, {"camera"}, (0, 0, 0)), Robot("q2", Q, {"camera"}, (50, 0, 0)),
              Robot("c1", C, {"camera"}, (10, 0, 0)), Robot("q3", Q, {"thermal"}, (10, 0, 0))]
    c = Coordinator(robots, World(), {"look": Exec("look")})
    c.submit(job("j1", pos=(40, 0, 0)))
    c.tick()
    assert c.assigned_to["j1"] == "q2"                     # nearest Q with a camera (c1 is the wrong kind)
    bids = events(c, "assigned")[0]["detail"]["bids"]
    assert set(bids) == {"q1", "q2"}


def test_battery_penalty_and_reserve():
    robots = [Robot("q1", Q, {"camera"}, (0, 0, 0), battery=0.4), Robot("q2", Q, {"camera"}, (5, 0, 0), battery=1.0)]
    c = Coordinator(robots, World(), {"look": Exec("look")})
    c.submit(job("j1", pos=(1, 0, 0)))
    c.tick()
    assert c.assigned_to["j1"] == "q2"                     # 4 m further but 72 s less battery penalty
    low = Robot("q3", Q, {"camera"}, (0, 0, 0), battery=0.21)
    c2 = Coordinator([low], World(), {"look": Exec("look")})
    c2.submit(job("j2", pos=(200, 0, 0), est_duration_s=600))
    c2.tick()
    assert "j2" not in c2.assigned_to                      # would drop under the reserve


def test_priority_order_and_urgent_preemption():
    ex = Exec("look", duration=100)
    c = Coordinator([Robot("q1", Q, {"camera"}, (0, 0, 0))], World(), {"look": ex})
    c.submit(job("routine", target="a", pos=(5, 0, 0), priority=4))
    c.tick()
    assert c.assigned_to["routine"] == "q1"
    c.submit(job("urgent", target="b", pos=(-5, 0, 0), priority=1))
    c.tick()
    assert c.assigned_to.get("urgent") == "q1"
    assert c.status["routine"] == JobStatus.QUEUED
    pc = events(c, "priority_change")
    assert pc and "urgent" in pc[0]["job"] and pc[0]["detail"]["used_gemini"] is False
    run(c, 300)
    assert c.status["urgent"] == JobStatus.DONE and c.status["routine"] == JobStatus.DONE


def test_explainer_failure_falls_back():
    def broken(ctx):
        raise RuntimeError("no network")
    c = Coordinator([Robot("q1", Q, {"camera"})], World(), {"look": Exec("look", duration=100)}, explainer=broken)
    c.submit(job("r", target="a", priority=5))
    c.tick()
    c.submit(job("u", target="b", priority=1))
    c.tick()
    e = events(c, "priority_change")[0]
    assert e["detail"]["explanation"] and not c.used_gemini


# ---------------------------------------------------------------- deconfliction
def test_one_robot_per_asset_and_zone():
    robots = [Robot("q1", Q, {"camera"}, (0, 0, 0)), Robot("q2", Q, {"camera"}, (1, 0, 0))]
    c = Coordinator(robots, World(), {"look": Exec("look", duration=30), "other": Exec("other", duration=30)})
    c.submit(job("j1", target="valve_01", pos=(3, 0, 0)))
    c.submit(job("j2", kind="other", target="valve_01", pos=(3, 0, 0)))
    c.submit(job("j3", target="x", pos=(3, 0, 0), params={"zone": "leak"}))
    c.submit(job("j4", kind="other", target="y", pos=(3, 0, 0), params={"zone": "leak"}))
    for _ in range(120):
        c.tick()
        active = [j for j, s in c.status.items() if s in (JobStatus.ASSIGNED, JobStatus.RUNNING)]
        assert not ({"j1", "j2"} <= set(active)) and not ({"j3", "j4"} <= set(active))
    assert all(c.status[j] == JobStatus.DONE for j in ("j1", "j2", "j3", "j4"))


def test_duplicate_job_is_skipped():
    c = Coordinator([], World(), {"look": Exec("look")})
    c.submit(job("j1", target="g"))
    c.submit(job("j2", target="g"))
    assert "j2" not in c.jobs and events(c, "duplicate_skipped")


def test_no_two_quadrupeds_on_one_walkway_segment():
    w = Walkways(patrol_route())
    # both start at the west end and head east along the same corridor
    q1 = Robot("q1", Q, {"camera"}, (-21.0, 20.5, 42.5), speed_mps=1.0)
    q2 = Robot("q2", Q, {"camera"}, (-20.0, 20.5, 42.5), speed_mps=1.0)
    c = Coordinator([q1, q2], World(), {"look": Exec("look", duration=5)}, planner=WalkwayPlanner(w))
    c.submit(job("a", target="ga", pos=(20.0, -9.0, 42.5), params={"standoff": 0.0}))
    c.submit(job("b", target="gb", pos=(22.0, -9.0, 42.5), params={"standoff": 0.0}))
    for _ in range(200):
        c.tick()
        held = [rt.holding for rt in c.rt.values() if rt.holding]
        assert len(held) == len(set(held))
    assert events(c, "segment_wait")
    assert c.status["a"] == JobStatus.DONE and c.status["b"] == JobStatus.DONE


# ---------------------------------------------------------------- levels
def test_supv_waits_for_approval_then_runs():
    ex = Exec("isolate", level=Level.SUPV, needs=("arm",))
    c = Coordinator([Robot("q1", Q, {"arm"})], World(), {"isolate": ex})
    assert c.submit(job("s1", kind="isolate", level=Level.SUPV)) == JobStatus.WAITING_APPROVAL
    run(c, 5)
    assert c.approvals == ["s1"] and not ex.calls
    assert c.approve("s1") and not c.approve("s1")
    run(c, 20)
    assert c.status["s1"] == JobStatus.DONE and ex.calls
    assert [e["event"] for e in c.timeline if e["job"] == "s1"][:3] == ["job_queued", "approval_requested", "approved"]


def test_reject_and_auto_approve():
    ex = Exec("isolate", level=Level.SUPV, needs=("arm",))
    c = Coordinator([Robot("q1", Q, {"arm"})], World(), {"isolate": ex})
    c.submit(job("s1", kind="isolate", level=Level.SUPV))
    assert c.reject("s1", "not now")
    run(c, 5)
    assert c.status["s1"] == JobStatus.FAILED and not ex.calls
    c2 = Coordinator([Robot("q1", Q, {"arm"})], World(), {"isolate": ex}, auto_approve=True)
    assert c2.submit(job("s2", kind="isolate", level=Level.SUPV)) == JobStatus.QUEUED
    assert events(c2, "approved")[0]["detail"]["by"] == "auto"


def test_assist_and_human_go_to_crew():
    ex = Exec("prep", level=Level.ASSIST)
    c = Coordinator([Robot("q1", Q, {"camera"})], World(), {"prep": ex})
    assert c.submit(job("h1", kind="weld", level=Level.HUMAN)) == JobStatus.HANDED_TO_HUMAN
    c.submit(job("p1", kind="prep", level=Level.ASSIST))
    run(c, 20)
    assert ex.calls and c.status["p1"] == JobStatus.HANDED_TO_HUMAN
    assert c.crew == ["h1", "p1"]


def test_finding_follow_up_jobs_are_queued():
    def leak(j):
        iso = Job("iso", "isolate", "valve_05", (2, 0, 0), Level.SUPV, needs={"arm"}, priority=1)
        return [Finding("gas_leak", "critical", "leak", (1, 2, 3), follow_up=[iso])]
    ex = {"survey": Exec("survey", needs=("gas_sensor",), status=JobStatus.FINDING, findings=leak),
          "isolate": Exec("isolate", level=Level.SUPV, needs=("arm",))}
    c = Coordinator([Robot("q1", Q, {"gas_sensor", "arm"})], World(), ex)
    c.submit(job("s", kind="survey"))
    run(c, 20)
    assert c.status["s"] == JobStatus.FINDING
    assert c.jobs["iso"].parent == "s" and c.status["iso"] == JobStatus.WAITING_APPROVAL
    f = events(c, "finding")[0]
    assert f["detail"]["position"] == [1, 2, 3] and f["detail"]["kind"] == "gas_leak"


def test_missing_executor_routes_to_crew_and_crash_is_failed():
    class Boom(Exec):
        def run(self, job, robot, world):
            raise ValueError("bad module")
    c = Coordinator([Robot("q1", Q, {"camera"})], World(), {"boom": Boom("boom")})
    c.submit(job("x", kind="boom", target="a"))
    c.submit(job("y", kind="nobody", target="b"))
    run(c, 200)
    assert c.status["x"] == JobStatus.FAILED and c.status["y"] == JobStatus.HANDED_TO_HUMAN


# ---------------------------------------------------------------- safety
def test_person_within_radius_pauses_robot():
    world = World()
    q = Robot("q1", Q, {"camera"}, (0, 0, 0), speed_mps=1.0)
    c = Coordinator([q], world, {"look": Exec("look", duration=5)})
    c.submit(job("j", pos=(20, 0, 0), params={"standoff": 0.0}))
    run(c, 5)
    world.people = [(q.position[0] + PERSON_RADIUS_M - 0.5, 0, 0)]
    c.tick()
    held = q.position
    run(c, 10)
    assert c.rt["q1"].state == "PAUSED_PERSON" and q.position == held
    world.people = [(q.position[0] + PERSON_RADIUS_M + 0.5, 5, 0)]
    run(c, 40)
    assert c.status["j"] == JobStatus.DONE
    assert events(c, "person_pause") and events(c, "person_clear")


def test_low_battery_docks_and_requeues():
    q = Robot("q1", Q, {"camera"}, (0, 0, 0), battery=0.95)
    c = Coordinator([q], World(), {"look": Exec("look", duration=500)}, docks={"q1": (0, 0, 0)})
    c.submit(job("j", pos=(5, 0, 0), est_duration_s=10))
    run(c, 10)
    q.battery = 0.15
    run(c, 1)
    assert c.status["j"] == JobStatus.QUEUED and c.rt["q1"].state == "DOCKING"
    run(c, 20)
    assert c.rt["q1"].state in ("CHARGING", "IDLE") and events(c, "docked")


# ---------------------------------------------------------------- sim and timeline
def test_walkway_graph_and_kind_motion():
    w = Walkways(patrol_route())
    assert any(n[:2] == (9.0, -4.5) for n in w.nodes)
    assert all(len(w.adj[k]) >= 1 for k in w.adj)
    plan = WalkwayPlanner(w)
    route = plan.route(Robot("q", Q, set(), (-21.0, 20.5, 42.5)), (33.0, -10.67, 43.86))
    for a, b in zip(route.legs, route.legs[1:]):
        assert math.dist(a.end, b.start) < 1e-6
    assert all(l.start[2] == 42.5 and l.end[2] == 42.5 for l in route.legs)
    on = [l for l in route.legs if l.segment]
    assert on and all(l.start[0] == l.end[0] or l.start[1] == l.end[1] for l in on)   # axis-aligned walkways
    air = plan.route(Robot("a", A, set(), (0, 0, 45)), (10, 10, 43))
    assert max(l.end[2] for l in air.legs) >= 60.0 and air.legs[-1].end[2] == 47
    sub = plan.route(Robot("r", R, set(), (0, 0, -10)), (30, 0, 5))
    assert all(l.end[2] < 0 for l in sub.legs)


def test_timeline_file_format(tmp_path):
    world = World()
    robots = [Robot("q1", Q, {"camera"}, (0, 0, 42.5)), Robot("a1", A, {"camera"}, (0, 5, 45), speed_mps=3)]
    c = Coordinator(robots, world, {"look": Exec("look", kinds=(Q, A), duration=5)})
    sim = FleetSim(c, timeline_path=tmp_path / "t.jsonl")
    c.submit(job("j1", pos=(5, 0, 42.5)))
    c.submit(job("j2", target="b", pos=(0, 20, 45)))
    sim.at(3, "leak", lambda s: s.c.emit("gas_leak_injected", None, None, position=[1.0, 2.0, 3.0]))
    sim.run(30)
    lines = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    robot_lines = [l for l in lines if "robot" in l and "event" not in l]
    event_lines = [l for l in lines if "event" in l]
    assert len(robot_lines) == 30 * 2
    for l in robot_lines:
        assert list(l) == ["t", "robot", "kind", "position", "yaw", "state", "job"]
        assert len(l["position"]) == 3 and l["kind"] in ("Q", "A")
    for l in event_lines:
        assert list(l) == ["t", "event", "job", "robot", "detail"] and isinstance(l["detail"], dict)
    assert sorted({l["t"] for l in robot_lines}) == [float(t) for t in range(30)]
    leak = [l for l in event_lines if l["event"] == "gas_leak_injected"][0]
    assert leak["detail"]["position"] == [1.0, 2.0, 3.0] and leak["t"] == 3


# ---------------------------------------------------------------- rust repair executor
def _result(**kw):
    base = {"outcome": "DONE", "label": "success", "unsafe": False, "true_rust_left": 0.01, "sim_time_s": 70.0,
            "final_reason": "ok", "trace": []}
    base.update(kw)
    return base


@pytest.mark.parametrize("res,status,kind", [
    (_result(), JobStatus.DONE, "corrosion_repaired"),
    (_result(outcome="ESCALATE_TO_HUMAN", label="escalate_rust_remains", final_reason="rust remains"),
     JobStatus.HANDED_TO_HUMAN, "corrosion_needs_crew"),
    (_result(label="painted_over_rust", unsafe=True, true_rust_left=0.12), JobStatus.FAILED, "unsafe_repair_outcome"),
])
def test_rust_repair_outcome_mapping(res, status, kind):
    ex = RustRepairJob(runner=lambda sc: res)
    r = ex.run(Job("rr", "rust_repair", "pipe", (0, 0, 0), Level.SUPV), Robot("q1", Q, {"arm", "spray"}), World(t=10))
    assert r.status == status and r.findings[0].kind == kind
    assert r.finished_t - r.started_t == 70.0
    if status == JobStatus.FAILED:
        assert r.findings[0].severity == "critical" and "UNSAFE" in r.findings[0].summary
        assert r.findings[0].follow_up[0].level == Level.HUMAN


def test_rust_repair_real_cpu_mock():
    pytest.importorskip("torch")
    ex = RustRepairJob()
    assert ex.level == Level.SUPV and {"arm", "spray"} <= ex.needs
    r = ex.run(Job("rr", "rust_repair", "pipe", (0, 0, 0), Level.SUPV, params={"scenario": {"seed": 1}}),
               Robot("q1", Q, {"arm", "spray"}), World())
    assert r.status in (JobStatus.DONE, JobStatus.HANDED_TO_HUMAN, JobStatus.FAILED)
    assert r.findings and "label" in r.findings[0].evidence


def test_demo_shift_runs(tmp_path):
    pytest.importorskip("torch")
    from fleet import demo
    sim, loaded, stubbed = demo.build(offline=True, timeline_path=tmp_path / "t.jsonl")
    sim.run(900)
    c = sim.c
    names = [e["event"] for e in c.timeline]
    assert "gas_leak_injected" in names and "priority_change" in names and "approved" in names
    iso = [j for j in c.jobs.values() if j.kind == "valve_isolate"]
    assert iso and [e for e in c.timeline if e["job"] == iso[0].id and e["event"] == "approval_requested"]
    assert c.status[[j for j in c.jobs if j.startswith("rust_repair")][0]] != JobStatus.QUEUED
    assert (tmp_path / "t.jsonl").exists()
