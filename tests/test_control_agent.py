import json
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace

import pytest

from policy.control.agent import (AGENT_SYSTEM, TOOLS, ArgonAgentBackend, OrchestrationAgent, ReplayAgentBackend,
                                  agent_body)
from policy.control.envelope import FUNCTIONS
from policy.control.port import ContractAdapter, MemoryCoordinator

ROBOTS = [{"id": "go2-1", "state": "SCRUB", "zone": "A"}, {"id": "go2-2", "state": "PATROL", "zone": "B"},
          {"id": "go2-3", "state": "DOCKED", "zone": "dock"}]
INTENT = "inspect deck 2, isolate any leak, don't paint without me"


def port():
    return MemoryCoordinator(ROBOTS, [{"id": "R-1", "kind": "rust_repair", "target": "P-104", "level": "SUPV",
                                        "priority": 3, "status": "RUNNING", "requested_by": "operator"}])


def test_tools_match_whitelist_and_require_why():
    assert [t["name"] for t in TOOLS] == list(FUNCTIONS)
    for t in TOOLS:
        assert t["type"] == "function" and t["description"]
        if t["name"] != "read_fleet_state":
            assert "why" in t["parameters"]["required"]
    assert "motor" in AGENT_SYSTEM.lower() and "esd" in AGENT_SYSTEM.lower()


def test_agent_body_interactions_function_calling():
    body = agent_body([{"type": "user_input", "content": "hi"}], "gemini-4-argon", "low")
    assert body["tools"] == TOOLS and body["store"] is False
    assert body["input"][0]["type"] == "user_input"
    assert body["generation_config"]["thinking_level"] == "low"
    assert "temperature" not in json.dumps(body)


def scripted(*replies):
    sent = []

    def transport(url, body, headers):
        sent.append(body)
        return replies[len(sent) - 1]
    return transport, sent


def fc(cid, name, **args):
    return {"type": "function_call", "id": cid, "name": name, "arguments": args}


def test_agent_loop_executes_calls_through_envelope_and_logs(tmp_path):
    p = port()
    t1 = {"status": "requires_action", "usage": {"total_input_tokens": 1200, "total_output_tokens": 90,
                                                  "total_thought_tokens": 300},
          "steps": [{"type": "thought", "signature": "sig"},
                    fc("c1", "queue_job", kind="inspection", target="deck-2", level="AUTO", priority=3,
                       why="operator asked for a deck 2 round"),
                    fc("c2", "reset_esd", zone="B", why="gas looks clear")]}
    t2 = {"status": "completed", "usage": {"total_input_tokens": 1500, "total_output_tokens": 40},
          "steps": [{"type": "model_output", "content": [{"type": "text", "text": "Queued the deck 2 round."}]}]}
    transport, sent = scripted(t1, t2)
    agent = OrchestrationAgent(p, ArgonAgentBackend(api_key="k", transport=transport), log_path=tmp_path / "a.jsonl")
    rec = agent.handle(intent="inspect deck 2", now=1.0)
    assert rec["status"] == "ok" and rec["text"] == "Queued the deck 2 round."
    assert [(c["name"], c["ok"]) for c in rec["calls"]] == [("queue_job", True), ("reset_esd", False)]
    second = sent[1]["input"]
    assert second[0]["type"] == "user_input" and second[1]["type"] == "thought"
    results = [x for x in second if x["type"] == "function_result"]
    assert [r["call_id"] for r in results] == ["c1", "c2"]
    assert "refused" in results[1]["result"][0]["text"]
    ev = [e for e in p.timeline if e["event"] == "gemini_call"]
    assert len(ev) == 1 and ev[0]["detail"]["tokens_in"] == 2700 and ev[0]["detail"]["refused"] == 1
    assert ev[0]["detail"]["decision"] == ["queue_job", "reset_esd"]
    line = json.loads((tmp_path / "a.jsonl").read_text().splitlines()[0])
    assert line["cost_usd"] > 0 and line["turns"] == 2


def test_agent_stops_after_max_turns(tmp_path):
    loop = {"steps": [fc("c", "read_fleet_state")], "usage": {}}
    transport, sent = scripted(*[loop] * 10)
    agent = OrchestrationAgent(port(), ArgonAgentBackend(api_key="k", transport=transport), log_path=None, max_turns=3)
    rec = agent.handle(intent="x", now=0.0)
    assert len(sent) == 3 and rec["turns"] == 3


def test_agent_error_is_offline_and_changes_nothing(tmp_path):
    def boom(url, body, headers):
        raise TimeoutError("no reply")
    p = port()
    before = len(p.jobs)
    rec = OrchestrationAgent(p, ArgonAgentBackend(api_key="k", transport=boom), log_path=None).handle(intent="x", now=0)
    assert rec["status"] == "offline" and "no reply" in rec["error"] and len(p.jobs) == before


def test_replay_turns_operator_intent_into_jobs():
    p = port()
    agent = OrchestrationAgent(p, ReplayAgentBackend(), log_path=None)
    rec = agent.handle(intent=INTENT, now=0.0)
    names = [(c["name"], c["args"].get("kind"), c["ok"]) for c in rec["calls"]]
    assert ("queue_job", "inspection", True) in names and ("queue_job", "gas_survey", True) in names
    assert ("require_operator", "paint", True) in names
    assert "paint" in agent.envelope.reserved
    assert all(c["args"].get("why") for c in rec["calls"])
    assert rec["text"]


def test_replay_replans_on_findings_with_why_and_envelope_refusal():
    p = port()
    agent = OrchestrationAgent(p, ReplayAgentBackend(), log_path=None)
    agent.handle(intent=INTENT, now=0.0)
    rec = agent.handle(findings=[{"kind": "gas_leak", "severity": "critical", "target": "F3", "zone": "B",
                                  "summary": "gas and ultrasonic rising at F3"}], now=10.0)
    calls = {(c["name"], c["args"].get("kind")) for c in rec["calls"]}
    assert ("queue_job", "valve_isolate") in calls and ("reprioritise", None) in calls
    rep = [e for e in p.timeline if e["event"] == "reprioritised"]
    assert rep and all(e["detail"]["why"] for e in rep)
    iso = next(j for j in p.jobs.values() if j["kind"] == "valve_isolate")
    assert iso["level"] == "SUPV" and iso["priority"] == 1
    rec = agent.handle(findings=[{"kind": "rust_verified_clean", "severity": "info", "target": "P-104", "zone": "A",
                                  "summary": "detector and Gemini-ER agree the patch is clean"}], now=20.0)
    paint = next(j for j in p.jobs.values() if j["kind"] == "paint")
    assert paint["status"] == "WAITING_APPROVAL"
    refused = [c for c in rec["calls"] if not c["ok"]]
    assert refused and refused[0]["name"] == "approve_request"


def test_cut_link_keeps_last_plan_and_skips_calls(tmp_path):
    p = port()
    transport, sent = scripted()
    agent = OrchestrationAgent(p, ArgonAgentBackend(api_key="k", transport=transport), log_path=None)
    agent.cut_link(True)
    rec = agent.handle(intent="inspect deck 2", now=1.0)
    assert rec["status"] == "cut" and sent == [] and agent.summary()["link"] == "CUT"
    assert p.timeline[-1]["event"] == "link_cut"
    assert p.queue("inspection", "deck-2", by="operator", why="manual") in p.jobs
    agent.cut_link(False)
    assert p.timeline[-1]["event"] == "link_restored" and agent.summary()["link"] == "UP"


class Level(str, Enum):
    AUTO = "AUTO"
    SUPV = "SUPV"
    HUMAN = "HUMAN"


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    RUNNING = "RUNNING"
    FAILED = "FAILED"


@dataclass
class Job:
    id: str
    kind: str
    target: str
    position: tuple
    level: Level
    priority: int = 3
    params: dict = field(default_factory=dict)


class FakeCoordinator:
    def __init__(self):
        self.jobs, self.status, self.approvals, self.timeline, self.n = {}, {}, [], [], 0
        self.rt = {"Q1": SimpleNamespace(state="IDLE", job=None, robot=SimpleNamespace(battery=0.8, position=(1, 2, 3)))}
        self.world = SimpleNamespace(t=5.0, state={}, assets={"F3": {"position_m": [4, 5, 6]}})
        self.t = 5.0

    def emit(self, event, job=None, robot=None, **detail):
        e = {"t": self.t, "event": event, "job": job, "robot": robot, "detail": detail}
        self.timeline.append(e)
        return e

    def new_id(self, kind):
        self.n += 1
        return f"{kind}-{self.n:03d}"

    def submit(self, job):
        self.jobs[job.id] = job
        if job.level == Level.SUPV:
            self.status[job.id] = JobStatus.WAITING_APPROVAL
            self.approvals.append(job.id)
        else:
            self.status[job.id] = JobStatus.QUEUED
        return self.status[job.id]

    def approve(self, job_id, by="operator"):
        if self.status.get(job_id) != JobStatus.WAITING_APPROVAL:
            return False
        self.approvals.remove(job_id)
        self.status[job_id] = JobStatus.QUEUED
        return True

    def reject(self, job_id, reason=""):
        if self.status.get(job_id) != JobStatus.WAITING_APPROVAL:
            return False
        self.approvals.remove(job_id)
        self.status[job_id] = JobStatus.FAILED
        return True


def test_contract_adapter_wraps_fleet_coordinator():
    c = FakeCoordinator()
    a = ContractAdapter(c, SimpleNamespace(Job=Job, Level=Level, JobStatus=JobStatus))
    jid = a.queue("valve_isolate", "F3", "SUPV", 1, by="gemini", why="leak at F3")
    assert c.jobs[jid].position == (4, 5, 6) and a.job(jid)["requested_by"] == "gemini"
    st = a.fleet_state()
    assert st["approvals"] == [jid] and st["robots"][0]["id"] == "Q1"
    assert a.reprioritise(jid, 2, "gemini", "x") and c.jobs[jid].priority == 2
    assert a.cancel(jid, "operator", "duplicate") and a.job(jid)["status"] == "FAILED"
    q = a.queue("inspection", "deck-2", "AUTO", 3, by="gemini", why="round")
    assert a.cancel(q, "gemini", "superseded") and c.timeline[-1]["event"] == "cancelled"
    from policy.control.envelope import Envelope
    s = a.queue("paint", "P-1", "SUPV", 3, by="gemini", why="clean")
    assert not Envelope().call(a, "approve_request", {"job_id": s, "why": "ok"})["ok"]
