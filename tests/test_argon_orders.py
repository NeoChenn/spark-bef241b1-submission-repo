import threading
import time
from datetime import date

from policy.control.agent import OrchestrationAgent, ReplayAgentBackend
from policy.control.centre import ControlCentre
from policy.control.port import LockedPort, MemoryCoordinator
from policy.control.snapshot import SensorSnapshot
from rig.hud import apply_event

DAY = date(2026, 11, 8)


class SlowPlanner(ReplayAgentBackend):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    def call(self, history):
        self.gate.wait(5.0)
        return super().call(history)


def centre_with_agent(gate):
    port = MemoryCoordinator([{"id": "go2-1", "zone": "A"}])
    agent = OrchestrationAgent(port, SlowPlanner(gate), log_path=None, today=lambda: DAY)
    return ControlCentre(None, log_path=None, today=lambda: DAY, agent=agent), port


def test_order_from_the_hud_reaches_the_planner_without_blocking_the_stream():
    gate = threading.Event()
    c, port = centre_with_agent(gate)
    t0 = time.perf_counter()
    apply_event(None, {"name": "intent", "args": {"text": "Inspect deck 2, don't paint without me."}}, c)
    for k in range(5):
        h = c.tick(SensorSnapshot(t=float(k)), now=float(k))
    assert time.perf_counter() - t0 < 1.0
    assert h["agent"]["status"] == "thinking" and h["agent"]["intent"].startswith("Inspect deck 2")
    gate.set()
    c.orders.shutdown(wait=True)
    s = c.tick(SensorSnapshot(t=9.0), now=9.0)["agent"]
    assert s["status"] == "ok" and s["text"]
    assert any(j["kind"] == "inspection" for j in port.jobs.values())
    assert "paint" in s["reserved"]


def test_agent_port_is_locked_with_the_tick():
    c, _ = centre_with_agent(threading.Event())
    assert isinstance(c.agent.port, LockedPort) and c.agent.port.lock is c.lock


def test_shift_takes_an_order_without_stalling_the_fleet(tmp_path):
    from policy.control.shift import run
    res = run("replay", tmp_path, t_end=60.0, orders=[(10.0, "Inspect deck 2, don't paint without me.")])
    tl = res["brain"].port.timeline
    call = next(e for e in tl if e["event"] == "gemini_call")
    assert call["detail"]["status"] == "ok" and "require_operator" in call["detail"]["decision"]
    assert res["brain"].f.t >= 60.0
    assert "paint" in res["brain"].centre.agent.envelope.reserved


def test_the_planner_speaks_as_argon_and_cannot_approve_argons_maintenance(tmp_path):
    from policy.control.shift import build
    brain = build("replay", tmp_path)
    jid = brain.port.queue("sensor_maintenance", "G-F1", "SUPV", 2, by="argon", why="flat head")
    out = brain.centre.agent.envelope.call(brain.centre.agent.port, "approve_request", {"job_id": jid, "why": "ok"})
    assert not out["ok"] and "its own SUPV job" in out["refused"]
