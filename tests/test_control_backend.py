import json
from datetime import date

import pytest

from policy.control.backend import (API_URL, ArgonBackend, ReplayBackend, parse_interaction, situation_key)
from policy.control.limiter import Budget, RateLimiter, cost_usd, price
from policy.control.prompt import SYSTEM, build, prompt_hash, request_body
from policy.control.safety import edge_rules
from policy.control.schema import RESPONSE_SCHEMA, validate
from policy.control.snapshot import GasSensor, Robot, SensorSnapshot, Verdict


def fleet(**kw):
    s = SensorSnapshot(
        t=5.0,
        robots=[Robot("go2-1", state="SCRUB", zone="A", force_n=20.0, rust_pct=40.0),
                Robot("go2-2", state="PATROL", zone="B"), Robot("go2-3", state="DOCKED", zone="dock", battery=0.6)],
        gas=[GasSensor("G-F3", zone="B", lel=3.0, acoustic_db=40.0)],
        jobs=[{"id": "R-1", "kind": "rust", "zone": "A"}, {"id": "L-F3", "kind": "leak", "zone": "B"}],
        images=["aGVsbG8="])
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_price_switches_after_intro():
    assert price(date(2026, 11, 30)) == (2.0, 10.0)
    assert price(date(2026, 12, 1)) == (4.0, 20.0)
    assert cost_usd(1_000_000, 0, 0, date(2026, 11, 8)) == pytest.approx(2.0)
    assert cost_usd(1000, 300, 700, date(2026, 11, 8)) == pytest.approx(0.002 + 0.01)
    assert cost_usd(1000, 0, 0, date(2026, 11, 8), cached=1000) == pytest.approx(0.0001)


def test_budget_caps_per_day():
    b = Budget(cap_usd_day=0.05)
    d = date(2026, 11, 8)
    assert b.allows(d)
    b.spend(0.03, d)
    b.spend(0.03, d)
    assert not b.allows(d)
    assert b.spent(d) == pytest.approx(0.06)
    assert b.allows(date(2026, 11, 9))


def test_rate_limiter_interval_alarm_and_burst():
    rl = RateLimiter(interval_s=6.0, alarm_gap_s=1.5, per_minute=3)
    assert rl.due(0.0)
    rl.mark(0.0)
    assert not rl.due(3.0)
    assert rl.due(3.0, alarm=True)
    rl.mark(3.0)
    assert not rl.due(4.0, alarm=True)
    assert rl.due(9.0)
    rl.mark(9.0)
    assert not rl.due(30.0, alarm=True)
    assert rl.due(61.0)


def test_prompt_contains_ids_rules_and_no_image_bytes():
    s = fleet(robots=[Robot("go2-1", state="BACK_OFF", zone="A", force_n=65.0)])
    p = build(s, edge_rules(s))
    assert "go2-1" in p["text"] and "L-F3" in p["text"] and "back_off" in p["text"]
    assert "aGVsbG8=" not in p["text"]
    assert p["images"] == ["aGVsbG8="]
    assert "advisory" in SYSTEM.lower()
    assert prompt_hash(p) == prompt_hash(build(s, edge_rules(s)))
    assert len(prompt_hash(p)) == 12
    assert prompt_hash(p) != prompt_hash(build(fleet()))


def test_request_body_interactions_format():
    p = build(fleet())
    body = request_body(p, "gemini-4-argon", "low")
    assert body["model"] == "gemini-4-argon"
    assert body["generation_config"]["thinking_level"] == "low"
    assert body["response_format"] == {"type": "text", "mime_type": "application/json", "schema": RESPONSE_SCHEMA}
    assert body["system_instruction"] == SYSTEM
    parts = body["input"]
    assert parts[0]["type"] == "text"
    assert parts[1] == {"type": "image", "data": "aGVsbG8=", "mime_type": "image/jpeg"}
    assert "temperature" not in json.dumps(body)


def test_parse_interaction_steps_and_usage():
    reply = {"status": "completed", "usage": {"total_input_tokens": 1500, "total_output_tokens": 300,
                                              "total_thought_tokens": 450, "total_cached_tokens": 0},
             "steps": [{"type": "thought", "signature": "x"},
                       {"type": "model_output", "content": [{"type": "text", "text": '{"situation": "ok",'}]},
                       {"type": "model_output", "content": [{"type": "text", "text": ' "actions": [], "anomalies": []}'}]}]}
    text, usage = parse_interaction(reply)
    assert json.loads(text)["situation"] == "ok"
    assert usage == {"in": 1500, "out": 300, "thought": 450, "cached": 0}


def test_argon_backend_posts_and_parses():
    seen = {}

    def transport(url, body, headers):
        seen.update(url=url, body=body, headers=headers)
        return {"usage": {"total_input_tokens": 10, "total_output_tokens": 5},
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": '{"situation": "x", "actions": [], "anomalies": []}'}]}]}

    b = ArgonBackend(api_key="k", transport=transport)
    r = b.call(build(fleet()), fleet())
    assert seen["url"] == API_URL and seen["headers"]["x-goog-api-key"] == "k"
    assert seen["body"]["model"] == "gemini-4-argon"
    assert r.tokens_in == 10 and r.tokens_out == 5 and r.model == "gemini-4-argon"
    assert b.live and b.available


def test_argon_backend_model_env(monkeypatch):
    monkeypatch.setenv("GEMINI_CONTROL_MODEL", "gemini-4-argon-test")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-robotics-er-2-preview")
    assert ArgonBackend(api_key="k").model == "gemini-4-argon-test"


@pytest.mark.parametrize("mutate,key", [
    (lambda s: setattr(s.robots[0], "state", "PATROL"), "nominal"),
    (lambda s: setattr(s.robots[0], "force_n", 65.0), "force_spike"),
    (lambda s: setattr(s.gas[0], "lel", 55.0), "evacuate"),
    (lambda s: (setattr(s.gas[0], "lel", 14.0), setattr(s.gas[0], "lel_trend", 20.0), setattr(s.gas[0], "db_trend", 30.0)),
     "leak_suspect"),
    (lambda s: setattr(s.robots[0], "person_m", 2.4), "person"),
    (lambda s: (setattr(s, "verdicts", [Verdict("go2-1", "rust", 0.82, "residue")]), setattr(s.robots[0], "rust_pct", 2.2)),
     "verifier_dispute"),
    (lambda s: setattr(s, "verdicts", [Verdict("go2-1", "rust", 0.82, "residue")]), "repair"),
    (lambda s: setattr(s.robots[0], "state", "DONE"), "done"),
    (lambda s: (setattr(s.gas[0], "lel", 9.0), setattr(s.gas[0], "lel_trend", -12.0)), "recovering"),
    (lambda s: setattr(s, "storm_level", 0.7), "storm"),
])
def test_situation_key(mutate, key):
    s = fleet()
    mutate(s)
    assert situation_key(s) == key


def test_replay_backend_is_deterministic_and_valid():
    b = ReplayBackend()
    assert not b.live and b.available
    for key in b.responses:
        s = fleet()
        r1 = b.call(build(s), s, key=key)
        r2 = b.call(build(s), s, key=key)
        assert r1.text == r2.text and r1.latency_ms == r2.latency_ms
        assert 600 <= r1.latency_ms <= 4000
        assert r1.tokens_in > 300 and r1.tokens_out > 50
        plan = validate(json.loads(r1.text), s)
        assert plan.situation, key


def test_replay_fuses_concurrent_situations():
    s = fleet()
    s.robots[0].force_n = 65.0
    s.gas[0].lel, s.gas[0].lel_trend, s.gas[0].db_trend = 16.0, 20.0, 30.0
    r = ReplayBackend().call(build(s), s)
    assert r.key == "force_spike+leak_suspect"
    plan = validate(json.loads(r.text), s)
    assert len(plan.situation) == 2
    assert {a.kind for a in plan.anomalies} == {"surface_damage", "probable_leak"}


def test_mock_fleet_follows_validated_actions():
    from policy.control.mock import FleetScenario
    f = FleetScenario()
    assert f.at(26.0).max_lel() > 50 and f.at(2.0).max_lel() < 5
    assert f.uplink_down(33.0) and not f.uplink_down(20.0)
    f.apply([{"kind": "dispatch", "robot": "go2-2", "target": "L-F3", "auto": False}], 20.0)
    assert f.r2.state == "PATROL"
    f.apply([{"kind": "dispatch", "robot": "go2-2", "target": "L-F3", "auto": True}], 20.0)
    assert f.r2.state == "LEAK_APPROACH" and f.r2.task == "L-F3"
    f.apply([{"kind": "retreat", "robot": "go2-2", "target": "upwind", "auto": True}], 25.0)
    assert f.r2.state == "GAS_RETREAT" and f.r2.zone == "A"
