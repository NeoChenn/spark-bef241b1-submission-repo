import math

from policy.control.envelope import Envelope
from policy.control.port import MemoryCoordinator
from policy.control.verifier import VerifiedEnvelope, delta, vetoed


class FakeFork:
    def __init__(self, harm):
        self.harm, self.applied = harm, []
        self.port = MemoryCoordinator([{"id": "Q1"}])

    def apply(self, fn):
        out = fn(self.port)
        self.applied.append(out["name"])

    def advance(self, h):
        bad = any(self.harm(self.port.jobs[j]) for j in self.port.jobs)
        return {"unsafe": 1 if bad == "unsafe" else 0, "isolate_s": 300.0 + (60.0 if bad else 0.0)}


def make(harm):
    forks = []

    def fork():
        forks.append(FakeFork(harm))
        return forks[-1]
    return VerifiedEnvelope(Envelope(), fork), forks


def q(kind, prio=3, target="G-1"):
    return {"kind": kind, "target": target, "level": "AUTO", "priority": prio, "why": "w"}


def test_good_plan_applied():
    v, forks = make(lambda j: False)
    port = MemoryCoordinator([{"id": "Q1"}])
    out = v.call(port, "queue_job", q("inspection"))
    assert out["ok"] and len(port.jobs) == 1 and v.vetoes == [] and v.checked == 1
    assert len(forks) == 2 and forks[1].applied == ["queue_job"] and forks[0].applied == []


def test_harmful_plan_vetoed_and_logged():
    v, _ = make(lambda j: j["priority"] == 1 and j["kind"] == "inspection")
    port = MemoryCoordinator([{"id": "Q1"}])
    out = v.call(port, "queue_job", q("inspection", 1))
    assert not out["ok"] and out["refused"].startswith("plan vetoed by simulation")
    assert port.jobs == {} and v.vetoes[0]["delta"]["isolate_s"] == 60.0
    assert port.timeline[-1]["event"] == "plan_vetoed"


def test_passthrough_and_inner_refusal_skip_rollout():
    v, forks = make(lambda j: True)
    port = MemoryCoordinator([{"id": "Q1"}])
    assert v.call(port, "read_fleet_state", {})["ok"]
    assert v.call(port, "require_operator", {"kind": "paint", "why": "w"})["ok"]
    out = v.call(port, "queue_job", {**q("inspection"), "why": ""})
    assert not out["ok"] and "why" in out["refused"]
    assert forks == [] and "paint" in v.reserved


def test_delta_rules():
    assert vetoed(delta({"unsafe": 0, "isolate_s": 100}, {"unsafe": 1, "isolate_s": 90})).startswith("+1 unsafe")
    assert vetoed(delta({"unsafe": 0, "isolate_s": 100}, {"unsafe": 0, "isolate_s": 105})) is None
    assert "never isolated" in vetoed(delta({"isolate_s": 100}, {"isolate_s": None}))
    assert vetoed(delta({"isolate_s": None}, {"isolate_s": 80})) is None
    assert delta({}, {})["isolate_s"] is None and math.isinf(delta({"isolate_s": 1}, {})["isolate_s"])
