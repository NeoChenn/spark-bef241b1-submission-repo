import json

import pytest

from fleet.api import TOOLS, FleetAPI
from fleet.contract import Job, JobResult, JobStatus, Level, Robot, RobotKind, World
from fleet.coordinator import Coordinator

Q = RobotKind.QUADRUPED


class Exec:
    def __init__(self, kind, level=Level.AUTO, needs=("camera",), duration=10.0):
        self.kind, self.level, self.needs, self.robot_kinds = kind, level, set(needs), {Q}
        self.duration, self.calls = duration, []

    def run(self, job, robot, world):
        self.calls.append(job.id)
        return JobResult(job.id, robot.id, JobStatus.DONE, world.t, world.t + self.duration)


def world():
    return World(assets={"gauge_01": {"id": "gauge_01", "type": "gauge", "position_m": [5.0, 0.0, 0.0]},
                         "pipe_A": {"id": "pipe_A", "type": "pipe", "position_m": [8.0, 0.0, 0.0]}})


def make(auto_approve=False, robots=None):
    ex = {"gauge_read": Exec("gauge_read"), "rust_repair": Exec("rust_repair", Level.SUPV, ("arm",), 30.0),
          "walk_round": Exec("walk_round", duration=200.0)}
    robots = robots or [Robot("q1", Q, {"camera", "arm"}, (0, 0, 0))]
    c = Coordinator(robots, world(), ex, auto_approve=auto_approve)
    return c, FleetAPI(c), ex


def run(c, n):
    for _ in range(n):
        c.tick()


def refused(out):
    return out["ok"] is False and out["refused"]


def test_tools_are_the_five_agent_calls():
    assert [t["name"] for t in TOOLS] == ["queue", "cancel", "approve_request", "reprioritise", "fleet_state"]
    json.dumps(TOOLS)


def test_queue_auto_job_by_asset_and_logs_the_call():
    c, api, ex = make()
    out = api.call("queue", {"kind": "gauge_read", "target": "gauge_01", "priority": 2, "why": "operator asked"})
    assert out["ok"] and out["status"] == "QUEUED" and not out["needs_operator"]
    j = c.jobs[out["job_id"]]
    assert j.position == (5.0, 0.0, 0.0) and j.params["requested_by"] == "gemini" and j.priority == 2
    assert [e for e in c.timeline if e["event"] == "agent_call"][0]["detail"]["ok"] is True
    run(c, 30)
    assert c.status[j.id] == JobStatus.DONE
    again = api.queue("gauge_read", "gauge_01", why="again")
    assert again["ok"] and again["job_id"] != j.id and not again["duplicate"]
    dup = api.queue("gauge_read", "gauge_01", why="twice")
    assert dup["ok"] and dup["duplicate"] and dup["job_id"] == again["job_id"]


def test_queued_supv_job_waits_for_the_operator_even_with_auto_approve():
    c, api, ex = make(auto_approve=True)
    out = api.queue("rust_repair", "pipe_A", priority=3, why="rust at pipe A, don't paint without me")
    assert out["status"] == "WAITING_APPROVAL" and out["needs_operator"]
    run(c, 20)
    assert not ex["rust_repair"].calls and c.approvals == [out["job_id"]]
    assert c.requests[out["job_id"]][-1]["note"].startswith("rust at pipe A")
    assert not c.approve(out["job_id"], by="gemini") and c.status[out["job_id"]] == JobStatus.WAITING_APPROVAL
    assert [e for e in c.timeline if e["event"] == "approval_refused"]
    assert c.approve(out["job_id"], by="operator")
    run(c, 60)
    assert c.status[out["job_id"]] == JobStatus.DONE


@pytest.mark.parametrize("args,why", [
    ({"kind": "rust_repair", "target": "pipe_A", "level": "AUTO"}, "cannot lower"),
    ({"kind": "esd_reset", "target": "pipe_A"}, "safety envelope"),
    ({"kind": "reset_esd_zone2", "target": "pipe_A"}, "safety envelope"),
    ({"kind": "rust_repair", "target": "pipe_A", "params": {"force_limit_n": 60}}, "40 N"),
    ({"kind": "rust_repair", "target": "pipe_A", "params": {"person_radius_m": 1.0}}, "3 m"),
    ({"kind": "rust_repair", "target": "pipe_A", "params": {"auto_approve": True}}, "cannot be set"),
    ({"kind": "gauge_read", "target": "gauge_01", "priority": 0}, "priority"),
    ({"kind": "gauge_read", "target": "gauge_01", "priority": "1"}, "priority"),
    ({"kind": "gauge_read", "target": "nowhere"}, "unknown target"),
    ({"kind": "gauge_read", "target": "x", "level": "MAYBE", "position": [0, 0, 0]}, "unknown level"),
])
def test_queue_refusals(args, why):
    c, api, _ = make()
    n = len(c.jobs)
    out = api.call("queue", {"why": "test", **args})
    assert refused(out) and why in out["refused"] and len(c.jobs) == n
    assert [e for e in c.timeline if e["event"] == "agent_refused"]


def test_queue_may_raise_the_level_to_supv():
    c, api, ex = make()
    out = api.queue("gauge_read", "gauge_01", level="SUPV", why="check with me first")
    assert out["status"] == "WAITING_APPROVAL"


def test_cancel_queued_moving_and_refusals():
    c, api, ex = make(robots=[Robot("q1", Q, {"camera", "arm"}, (0, 0, 0)), Robot("q2", Q, {"camera"}, (90, 0, 0))])
    a = api.queue("walk_round", "x", position=[60, 0, 0], why="t")["job_id"]
    b = api.queue("gauge_read", "gauge_01", why="t")["job_id"]
    run(c, 2)
    mover = c.assigned_to[a]
    assert c.rt[mover].state == "MOVING"
    assert api.cancel(a, "not needed")["status"] == "CANCELLED"
    assert c.rt[mover].state == "IDLE" and a not in c.assigned_to
    assert refused(api.cancel(a)) and refused(api.cancel("nope"))
    run(c, 40)
    assert refused(api.cancel(b))
    s = api.queue("rust_repair", "pipe_A", why="t")["job_id"]
    c.approve(s)
    for _ in range(60):
        c.tick()
        if c.status[s] == JobStatus.RUNNING:
            break
    out = api.cancel(s, "stop")
    assert refused(out) and "half done" in out["refused"]
    q = api.queue("gauge_read", "gauge_01", why="t")["job_id"]
    assert api.cancel(q)["ok"] and c.status[q] == JobStatus.CANCELLED


def test_approve_request_holds_an_auto_job_and_never_approves():
    c, api, ex = make()
    jid = api.queue("gauge_read", "gauge_01", why="t")["job_id"]
    out = api.approve_request(jid, "gauge 01 is next to a hot-work permit, operator should confirm")
    assert out["ok"] and out["status"] == "WAITING_APPROVAL" and out["needs_operator"]
    assert c.jobs[jid].level == Level.SUPV and c.approvals == [jid]
    run(c, 30)
    assert not ex["gauge_read"].calls
    again = api.approve_request(jid, "still waiting")
    assert again["ok"] and c.status[jid] == JobStatus.WAITING_APPROVAL and len(c.requests[jid]) == 2
    assert refused(api.call("approve", {"job_id": jid}))
    assert c.status[jid] == JobStatus.WAITING_APPROVAL
    assert c.approve(jid, by="operator")
    run(c, 30)
    assert c.status[jid] == JobStatus.DONE
    assert refused(api.approve_request(jid)) and refused(api.approve_request("nope"))


def test_approve_request_refused_while_running():
    c, api, ex = make()
    jid = api.queue("walk_round", "x", position=[1, 0, 0], why="t")["job_id"]
    for _ in range(30):
        c.tick()
        if c.status[jid] == JobStatus.RUNNING:
            break
    out = api.approve_request(jid, "late")
    assert refused(out) and "running" in out["refused"]


def test_reprioritise_changes_order_and_explains():
    c, api, ex = make()
    slow = api.queue("walk_round", "x", position=[3, 0, 0], priority=4, why="t")["job_id"]
    run(c, 2)
    late = api.queue("gauge_read", "gauge_01", priority=5, why="t")["job_id"]
    run(c, 2)
    assert c.status[late] == JobStatus.QUEUED
    out = api.reprioritise(late, 1, "gauge 01 reads high on the DCS, read it now")
    assert out["ok"] and c.jobs[late].priority == 1
    pc = [e for e in c.timeline if e["event"] == "priority_change" and e["job"] == late][0]
    assert pc["detail"]["explanation"].startswith("gauge 01") and pc["detail"]["used_gemini"] is True
    run(c, 1)
    assert c.assigned_to.get(late) == "q1" and c.status[slow] == JobStatus.QUEUED
    for bad in (0, 6, 2.5, True, "1"):
        assert refused(api.reprioritise(slow, bad))
    run(c, 60)
    assert refused(api.reprioritise(late, 3)) and refused(api.reprioritise("nope", 3))


def test_fleet_state_is_read_only():
    c, api, ex = make()
    api.queue("rust_repair", "pipe_A", why="t")
    n, before = len(c.timeline), json.dumps(api.fleet_state(), sort_keys=True)
    out = api.call("fleet_state")
    assert out["ok"] and len(c.timeline) == n and json.dumps(out, sort_keys=True) == before
    assert out["robots"][0]["id"] == "q1" and out["approvals"] and out["jobs"][0]["status"] == "WAITING_APPROVAL"
    json.dumps(out)


def test_unknown_tools_and_bad_arguments_are_refused():
    c, api, ex = make()
    for name in ("approve", "reset_esd", "set_force_limit", "disable_person_pause"):
        assert refused(api.call(name, {}))
    assert refused(api.call("queue", {"target": "gauge_01"}))
    assert refused(api.call("reprioritise", {"job_id": "x", "priority": 2, "colour": "red"}))
