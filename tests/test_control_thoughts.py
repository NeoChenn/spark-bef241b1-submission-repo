import json
from datetime import date
from pathlib import Path

import pytest

from policy.control.agent import ArgonAgentBackend, OrchestrationAgent, ReplayAgentBackend, agent_body
from policy.control.backend import (THOUGHT_CHARS, ArgonBackend, Reply, Thoughts, clip, parse_sse, summary_texts)
from policy.control.centre import ControlCentre
from policy.control.limiter import cost_usd
from policy.control.port import MemoryCoordinator
from policy.control.prompt import request_body
from policy.control.snapshot import GasSensor, Robot, SensorSnapshot

FIX = Path(__file__).with_name("fixtures")
DAY = date(2026, 11, 8)
ROBOTS = [{"id": "R1", "state": "IDLE", "zone": "A"}, {"id": "R2", "state": "PATROL", "zone": "B"}]
FILTERED = ["Keep approvals with the operator because the operator explicitly wants to be consulted on painting.",
            "Regarding the leak at F3: it's critical, meaning I need to prioritize it as safety.",
            "Keep approvals with the operator for painting jobs. Queue the valve isolation job for F3."]


def high():
    return json.loads((FIX / "argon_reply_high.json").read_text())


def sse_lines():
    return (FIX / "argon_stream_medium.sse").read_text().splitlines()


def port():
    return MemoryCoordinator(ROBOTS, [{"id": "J-3", "kind": "inspection", "target": "deck-2", "level": "AUTO",
                                        "priority": 3, "status": "QUEUED", "requested_by": "operator"}])


def snap(lel=2.0):
    return SensorSnapshot(t=1.0, robots=[Robot("R1", state="SCRUB", zone="A"), Robot("R2", state="PATROL", zone="B")],
                          gas=[GasSensor("G-F3", zone="B", lel=lel, acoustic_db=40.0)],
                          jobs=[{"id": "L-F3", "kind": "leak", "zone": "B"}])


def fixed(*replies):
    replies = list(replies)
    return lambda url, body, headers: replies.pop(0)


def done(text="Queued."):
    return {"status": "completed", "usage": {"total_input_tokens": 1300, "total_output_tokens": 20},
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}]}


def test_clip_cuts_at_sentence_and_word():
    assert clip("  Short   one. ") == "Short one."
    first = "The first sentence runs long enough to be worth keeping on its own as a thought line."
    assert clip(first + " " + "word " * 80) == first
    out = clip("x" * 10 + " " + "y " * 200)
    assert len(out) <= THOUGHT_CHARS and out.endswith("…")
    assert len(clip("z" * 500)) <= THOUGHT_CHARS


def test_summary_texts_from_recorded_high_reply():
    texts = summary_texts(high()["steps"])
    assert texts[0] == "Analysis and Action Plan for Fleet Coordination"
    assert len(texts) == 5
    assert all(0 < len(t) <= THOUGHT_CHARS and "**" not in t and "`" not in t for t in texts)
    assert texts[1].startswith("Okay, let's break this down.")


def test_parse_sse_rebuilds_recorded_stream():
    seen = []
    reply = parse_sse(sse_lines(), seen.append)
    assert seen == []
    assert reply["status"] == "requires_action" and reply["usage"]["total_thought_tokens"] == 55
    types = [s["type"] for s in reply["steps"]]
    assert types == ["thought", "function_call", "function_call"]
    assert reply["steps"][0]["signature"] == "REDACTED"
    q = reply["steps"][2]
    assert q["name"] == "queue_job" and q["arguments"]["kind"] == "valve_isolate" and q["arguments"]["priority"] == 1
    assert q["id"].startswith("call_")


def test_parse_sse_streams_summaries_and_text():
    ev = [{"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
          {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_summary",
                                                              "content": {"type": "text", "text": "**Plan**\n\nIsolate F3 first."}}},
          {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_summary",
                                                              "content": {"type": "text", "text": "Then the deck 2 round."}}},
          {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": "s"}},
          {"event_type": "step.stop", "index": 0},
          {"event_type": "step.start", "index": 1, "step": {"type": "model_output"}},
          {"event_type": "step.delta", "index": 1, "delta": {"type": "text", "text": "{\"a\":"}},
          {"event_type": "step.delta", "index": 1, "delta": {"type": "text", "text": "1}"}},
          {"event_type": "step.stop", "index": 1},
          {"event_type": "interaction.completed", "interaction": {"status": "completed", "usage": {"total_thought_tokens": 9}}}]
    lines = [x for e in ev for x in (f"event: {e['event_type']}", "data: " + json.dumps(e), "")] + ["data: [DONE]"]
    seen = []
    reply = parse_sse(lines, seen.append)
    assert seen == ["Plan", "Isolate F3 first.", "Then the deck 2 round."]
    assert reply["steps"][0]["signature"] == "s" and len(reply["steps"][0]["summary"]) == 2
    assert reply["steps"][1]["content"] == [{"type": "text", "text": "{\"a\":1}"}]
    assert reply["usage"]["total_thought_tokens"] == 9


def test_parse_sse_raises_on_error_event():
    with pytest.raises(RuntimeError):
        parse_sse(["event: error", 'data: {"event_type":"error","error":{"message":"quota"}}'], None)


def test_request_bodies_ask_for_thought_summaries():
    p = {"system": "s", "text": "t", "images": []}
    assert request_body(p, "m")["generation_config"]["thinking_summaries"] == "auto"
    assert agent_body([], "m")["generation_config"]["thinking_summaries"] == "auto"


def test_thoughts_keep_six_newest_last_and_mode():
    th = Thoughts()
    assert th.mode() == "decisions"
    for k in range(9):
        th.add(f"step {k}.", kind="decision", t=float(k))
    v = th.view()
    assert [x["text"] for x in v] == [f"step {k}." for k in range(3, 9)]
    th.add("Gas at F3 is rising.", t=10.0)
    assert th.mode() == "summaries" and th.view()[-1] == {"t": 10.0, "text": "Gas at F3 is rising.", "kind": "summary"}


def test_agent_summaries_tagged_with_calls_and_cost_counts_thoughts():
    p = port()
    th = Thoughts()
    agent = OrchestrationAgent(p, ArgonAgentBackend(api_key="k", transport=fixed(high(), done())), log_path=None,
                               today=lambda: DAY)
    agent.thoughts = th
    rec = agent.handle(intent="inspect deck 2 and isolate any leak; don't paint without me", now=3.0)
    assert rec["status"] == "ok" and rec["tokens_thought"] == 1043
    assert rec["cost_usd"] == pytest.approx(cost_usd(1078 + 1300, 120, 1043, DAY), abs=1e-6)
    v = th.view()
    assert th.mode() == "summaries" and [x["text"] for x in v] == FILTERED
    assert all(x["kind"] == "summary" and x["call"] == "call_1313667481786457689" for x in v)
    calls = agent.summary()["calls"]
    assert [c["id"] for c in calls] == ["call_1313667481786457689", "call_1313667481786454362"]
    assert all(c["why"] for c in calls)


def test_agent_without_summaries_records_decisions_and_refusals():
    p = port()
    th = Thoughts()
    t1 = {"status": "requires_action", "usage": {"total_input_tokens": 900, "total_output_tokens": 50},
          "steps": [{"type": "thought", "signature": "x"},
                    {"type": "function_call", "id": "c1", "name": "queue_job",
                     "arguments": {"kind": "gas_survey", "target": "deck-2", "level": "AUTO", "priority": 2,
                                   "why": "Survey before the visual round."}},
                    {"type": "function_call", "id": "c2", "name": "cancel_job",
                     "arguments": {"job_id": "J-3", "why": ""}}]}
    agent = OrchestrationAgent(p, ArgonAgentBackend(api_key="k", transport=fixed(t1, done())), log_path=None)
    agent.thoughts = th
    agent.handle(intent="inspect deck 2", now=1.0)
    v = th.view()
    assert th.mode() == "decisions"
    assert [x["call"] for x in v] == ["c1", "c2"]
    assert v[0]["text"] == "Queue gas survey · deck 2 · P2 — Survey before the visual round."
    assert "refused" in v[1]["text"]
    r = agent.refused()
    assert len(r) == 1 and r[0]["label"] == "Cancel inspection round · deck 2" and r[0]["why"] == "every call needs a one-line why"
    assert isinstance(r[0]["t"], float)


def test_streamed_thoughts_are_not_added_twice():
    th = Thoughts()
    replies = [{**high(), "_streamed": True}, done()]
    backend = ArgonAgentBackend(api_key="k")

    def transport(url, body, headers):
        if replies[0].get("_streamed"):
            for t in summary_texts(high()["steps"]):
                backend.on_thought(t)
        return replies.pop(0)
    backend.transport = transport
    agent = OrchestrationAgent(port(), backend, log_path=None)
    agent.thoughts = th
    agent.handle(intent="inspect deck 2", now=1.0)
    v = th.view()
    assert [x["text"] for x in v] == FILTERED and all(x.get("call") == "call_1313667481786457689" for x in v)


def test_agent_backend_streams_by_default(monkeypatch):
    monkeypatch.delenv("GEMINI_STREAM", raising=False)
    monkeypatch.delenv("GEMINI_AGENT_THINKING", raising=False)
    b = ArgonAgentBackend(api_key="k")
    assert b.transport == b._stream and b.thinking == "high"
    monkeypatch.setenv("GEMINI_STREAM", "0")
    assert ArgonAgentBackend(api_key="k").transport.__name__ == "_post"


class Summarising:
    live, available, name = True, True, "gemini-4-argon"

    def __init__(self, thoughts=()):
        self.thoughts = tuple(thoughts)

    def call(self, prompt, snap):
        text = json.dumps({"situation": "Leak suspected at F3.", "actions": [
            {"kind": "dispatch", "robot": "R2", "target": "L-F3", "confidence": 0.9, "rationale": "R2 is nearest."}],
            "anomalies": []})
        return Reply(text, 2000, 300, 500, 0, 900.0, "gemini-4-argon", thoughts=self.thoughts)


def test_centre_header_carries_summaries_with_call_ids():
    c = ControlCentre(Summarising(["Gas and dB rise together.", "Send R2."]), log_path=None, today=lambda: DAY)
    h = c.tick(snap(), now=0.0)
    assert h["thoughts_mode"] == "summaries"
    assert [x["text"] for x in h["thoughts"]] == ["Gas and dB rise together.", "Send R2."]
    assert {x["call"] for x in h["thoughts"]} == {"cloud:dispatch:R2:L-F3"}
    assert h["refused"] == []


def test_centre_decisions_once_per_plan_and_vetoes_refused():
    c = ControlCentre(Summarising(), log_path=None, today=lambda: DAY, interval_s=1.0)
    h = c.tick(snap(), now=0.0)
    assert h["thoughts_mode"] == "decisions"
    assert [x["text"] for x in h["thoughts"]] == ["Dispatch R2 to L-F3 — R2 is nearest."]
    h = c.tick(snap(lel=60.0), now=2.0)
    assert len(h["thoughts"]) == 1
    assert [r["label"] for r in h["refused"]] == ["Dispatch R2 to L-F3"]
    assert "gas" in h["refused"][0]["why"]
    t0 = h["refused"][0]["t"]
    assert c.tick(snap(lel=60.0), now=3.0)["refused"][0]["t"] == t0


def test_centre_shares_thoughts_with_agent():
    agent = OrchestrationAgent(port(), ReplayAgentBackend(), log_path=None)
    c = ControlCentre(None, log_path=None, agent=agent)
    assert agent.thoughts is c.thoughts
    agent.handle(intent="inspect deck 2", now=1.0)
    h = c.tick(snap(), now=0.0)
    assert h["thoughts_mode"] == "decisions" and h["thoughts"][0]["text"].startswith("Queue inspection round · deck 2")
    assert h["agent"]["calls"][0]["id"] == "r1"


def test_argon_backend_reply_carries_summaries():
    b = ArgonBackend(api_key="k", transport=fixed({**high(), "steps": high()["steps"][:1] + [
        {"type": "model_output", "content": [{"type": "text", "text": "{}"}]}]}))
    r = b.call({"system": "s", "text": "t", "images": []}, snap())
    assert r.text == "{}" and r.tokens_thought == 1043 and len(r.thoughts) == 5 and not r.streamed
