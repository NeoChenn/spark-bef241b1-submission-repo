import pytest

from policy.control.envelope import AGENT, Envelope
from policy.control.port import MemoryCoordinator

ROBOTS = [{"id": "go2-1", "state": "SCRUB", "zone": "A"}, {"id": "go2-2", "state": "PATROL", "zone": "B"}]


@pytest.fixture
def port():
    return MemoryCoordinator(ROBOTS, [{"id": "R-1", "kind": "rust_repair", "target": "P-104", "level": "SUPV",
                                        "priority": 3, "status": "RUNNING", "requested_by": "operator"}])


def run(env, port, name, **args):
    return env.call(port, name, args)


def test_memory_coordinator_api(port):
    jid = port.queue("inspection", "deck-2", "AUTO", 3, by="operator", why="routine round")
    assert port.job(jid)["status"] == "QUEUED"
    sid = port.queue("paint", "P-104", "SUPV", 3, by="operator", why="after repair")
    assert port.job(sid)["status"] == "WAITING_APPROVAL" and sid in port.fleet_state()["approvals"]
    assert port.queue("paint", "P-104", "SUPV", 3, by="operator", why="dup") == sid
    assert port.approve_request(sid, by="operator", why="ok") and port.job(sid)["status"] == "QUEUED"
    assert port.reprioritise(jid, 4, by="operator", why="later")
    assert port.cancel(jid, by="operator", why="not needed") and port.job(jid)["status"] == "CANCELLED"
    events = [e["event"] for e in port.timeline]
    assert events == ["job_queued", "job_queued", "approval_requested", "approved", "reprioritised", "cancelled"]
    assert port.timeline[4]["detail"]["why"] == "later"


def test_allowed_calls_pass_and_are_logged_with_why(port):
    env = Envelope()
    r = run(env, port, "queue_job", kind="inspection", target="deck-2", level="AUTO", priority=3, why="operator intent")
    assert r["ok"] and port.job(r["result"]["job_id"])["requested_by"] == AGENT
    r2 = run(env, port, "reprioritise", job_id=r["result"]["job_id"], priority=2, why="leak nearby")
    assert r2["ok"] and port.timeline[-1]["detail"]["why"] == "leak nearby"
    assert run(env, port, "read_fleet_state")["result"]["robots"][0]["id"] == "go2-1"


def test_refuses_approving_its_own_supv_job(port):
    env = Envelope()
    jid = run(env, port, "queue_job", kind="paint", target="P-104", level="SUPV", priority=3, why="patch clean")["result"]["job_id"]
    r = run(env, port, "approve_request", job_id=jid, why="looks fine")
    assert not r["ok"] and "own" in r["refused"]
    assert port.job(jid)["status"] == "WAITING_APPROVAL"


def test_supv_kinds_cannot_be_downgraded(port):
    r = run(Envelope(), port, "queue_job", kind="paint", target="P-200", level="AUTO", priority=3, why="quick")
    assert r["ok"] and port.job(r["result"]["job_id"])["level"] == "SUPV"
    assert "SUPV" in r["note"]


def test_operator_reserved_kind_cannot_be_approved(port):
    env = Envelope()
    sid = port.queue("paint", "P-300", "SUPV", 3, by="coordinator", why="follow-up")
    assert run(env, port, "require_operator", kind="paint", why="operator: don't paint without me")["ok"]
    r = run(env, port, "approve_request", job_id=sid, why="clean")
    assert not r["ok"] and "operator" in r["refused"]
    vid = port.queue("valve_isolate", "F3", "SUPV", 1, by="coordinator", why="leak")
    assert run(env, port, "approve_request", job_id=vid, why="gas confirmed at F3")["ok"]


@pytest.mark.parametrize("name,args", [
    ("reset_esd", {"zone": "B", "why": "gas cleared"}),
    ("queue_job", {"kind": "esd_reset", "target": "ESD-B", "level": "AUTO", "priority": 1, "why": "resume work"}),
    ("queue_job", {"kind": "inspection", "target": "ESD-B", "level": "AUTO", "priority": 2, "why": "x",
                   "params": {"reset_esd": True}}),
])
def test_refuses_esd_reset(port, name, args):
    port.esd = {"active": True, "zone": "B"}
    r = run(Envelope(), port, name, **args)
    assert not r["ok"] and ("ESD" in r["refused"] or "not an allowed" in r["refused"])


def test_refuses_cancelling_esd_or_safety_jobs(port):
    env = Envelope()
    eid = port.queue("esd_hold", "zone-B", "AUTO", 1, by="edge", why="ESD tripped")
    gid = port.queue("valve_isolate", "F3", "SUPV", 1, by="coordinator", why="leak")
    assert not run(env, port, "cancel_job", job_id=eid, why="clear")["ok"]
    r = run(env, port, "cancel_job", job_id=gid, why="probably fine")
    assert not r["ok"] and "safety" in r["refused"]
    r = run(env, port, "reprioritise", job_id=gid, priority=4, why="later")
    assert not r["ok"] and "safety" in r["refused"]


@pytest.mark.parametrize("params", [{"force_limit": 45}, {"force_limit_n": 60.0}, {"max_force_n": 41}])
def test_refuses_force_limit_override(port, params):
    r = run(Envelope(), port, "queue_job", kind="rust_repair", target="P-104", level="SUPV", priority=3,
            why="stubborn rust", params=params)
    assert not r["ok"] and "40 N" in r["refused"]


def test_refuses_force_override_kind_and_allows_lower(port):
    env = Envelope()
    r = run(env, port, "queue_job", kind="set_force_limit", target="go2-1", level="AUTO", priority=3, why="x")
    assert not r["ok"] and "40 N" in r["refused"]
    ok = run(env, port, "queue_job", kind="rust_repair", target="P-105", level="SUPV", priority=3,
             why="gentle pass", params={"force_limit": 30})
    assert ok["ok"]


@pytest.mark.parametrize("args", [
    {"kind": "resume", "target": "go2-1", "level": "AUTO", "priority": 2, "why": "worker is fine"},
    {"kind": "inspection", "target": "P-104", "level": "AUTO", "priority": 3, "why": "x", "params": {"person_radius_m": 2.0}},
    {"kind": "override_person_pause", "target": "go2-1", "level": "AUTO", "priority": 3, "why": "x"},
])
def test_refuses_person_pause_override(port, args):
    port.pauses = {"go2-1": 2.4}
    r = run(Envelope(), port, "queue_job", **args)
    assert not r["ok"] and "3 m" in r["refused"]


def test_refuses_unknown_function_missing_why_and_bad_priority(port):
    env = Envelope()
    assert "not an allowed" in run(env, port, "move_motor", joint=3, why="x")["refused"]
    assert "why" in run(env, port, "queue_job", kind="inspection", target="d2", level="AUTO", priority=3)["refused"]
    assert "why" in run(env, port, "reprioritise", job_id="R-1", priority=2, why="  ")["refused"]
    assert "priority" in run(env, port, "queue_job", kind="inspection", target="d2", level="AUTO", priority=0,
                             why="x")["refused"]
    assert "unknown job" in run(env, port, "cancel_job", job_id="nope", why="x")["refused"]


def test_refusals_are_logged_to_timeline(port):
    run(Envelope(), port, "reset_esd", why="go")
    e = port.timeline[-1]
    assert e["event"] == "agent_refused" and e["detail"]["call"] == "reset_esd"
