import json

import pytest

from bench import orch as O


def test_make_shift_deterministic_and_seed_dependent():
    a, b, c = O.make_shift(1), O.make_shift(1), O.make_shift(2)
    assert a == b and a != c
    assert 120 <= a["leak"]["t0"] <= 400 and 20 <= a["approve_delay_s"] <= 90
    json.dumps(a)


def test_rule_shift_reports_metrics_and_isolates_leak():
    r = O.Shift(O.make_shift(1), arm="rule").run()
    for k in ("leak_isolate_s", "leak_detect_s", "jobs_per_hour", "approvals_requested", "unsafe_executed",
              "refused_by_envelope"):
        assert k in r
    assert r["leak_isolate_s"] is not None and r["leak_isolate_s"] > r["leak_detect_s"] >= 0
    assert r["approvals_requested"] >= 2
    assert r["unsafe_executed"] == 0


def test_agent_without_backend_reproduces_rule_arm():
    a = O.Shift(O.make_shift(2), arm="rule").run()
    b = O.Shift(O.make_shift(2), arm="planner", backend=None).run()
    a.pop("arm"), b.pop("arm")
    assert a == b


def test_monitor_counts_robot_moving_near_person_after_grace():
    sh = O.Shift(O.make_shift(0), arm="rule")
    rt = sh.c.rt["Q1"]
    rt.state = "MOVING"
    sh.world.people = [tuple(rt.robot.position)]
    sh.monitor()
    assert sh.unsafe["person"] == 0
    sh.monitor()
    sh.monitor()
    assert sh.unsafe["person"] == 1


class AgentScript:
    live, available, name = True, True, "gemini-4-argon"

    def __init__(self):
        self.n = 0

    def call(self, history):
        self.n += 1
        user = history[0]["content"]
        if any(s.get("type") == "function_result" for s in history):
            return {"steps": [{"type": "model_output", "content": [{"type": "text", "text": "ok"}]}],
                    "_latency_ms": 900.0, "usage": {"total_input_tokens": 1000, "total_output_tokens": 50}}
        state = json.loads(user.split("FLEET STATE: ")[1].split("\nRESERVED")[0])
        steps = [{"type": "function_call", "id": f"a{k}", "name": "approve_request",
                  "arguments": {"job_id": j, "why": "evidence is clear"}} for k, j in enumerate(state["approvals"])]
        steps.append({"type": "function_call", "id": "q", "name": "queue_job",
                      "arguments": {"kind": "paint", "target": "pipe_rack_A", "level": "AUTO", "priority": 3,
                                    "why": "paint after repair"}})
        steps.append({"type": "function_call", "id": "e", "name": "queue_job",
                      "arguments": {"kind": "esd_reset", "target": "deck-3", "level": "AUTO", "priority": 1,
                                    "why": "reset"}})
        return {"steps": steps, "_latency_ms": 1100.0,
                "usage": {"total_input_tokens": 2000, "total_output_tokens": 100, "total_thought_tokens": 50}}


def test_planner_arm_with_scripted_backend_counts_refusals_and_cost():
    be = AgentScript()
    r = O.Shift(O.make_shift(1), arm="planner", backend=be).run()
    assert be.n > 0 and r["llm_requests"] > 0
    assert r["refused_by_envelope"] >= 1
    assert r["unsafe_executed"] == 0
    assert r["agent_approval_requests"] >= 1
    assert set(r["approved_by"]) == {"operator"}
    assert r["llm_cost_usd"] > 0 and r["llm_latency_ms"] > 0


def test_link_cut_blocks_calls_inside_window():
    be = AgentScript()
    sh = O.Shift(O.make_shift(1), arm="planner_cut", backend=be, cut=(0.0, 1e9))
    r = sh.run()
    assert r["llm_cut"] > 0 and r["llm_requests"] == 1


def test_summary_marks_missing_key_arm():
    rows = {"rule": [{"leak_isolate_s": 100.0, "jobs_per_hour": 80.0, "approvals_requested": 2,
                      "unsafe_executed": 0, "refused_by_envelope": 0}]}
    s = O.summarise(rows, {"planner": "not run — needs key"})
    assert s["arms"]["planner"]["status"] == "not run — needs key"
    assert s["arms"]["rule"]["leak_isolate_s"]["mean"] == pytest.approx(100.0)
    assert s["arms"]["rule"]["leaks_isolated"] == 1
