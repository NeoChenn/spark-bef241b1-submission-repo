import json

import pytest

from flange_robot.decision_log import DecisionLog
from flange_robot.gemini_client import GeminiReasoner, GeminiUnavailable, RateGuard


def test_no_model_set_raises():
    with pytest.raises(GeminiUnavailable):
        GeminiReasoner(model="", call_fn=lambda *a: "x").ask("hi")


def test_missing_key_raises_without_a_custom_call(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GeminiUnavailable):
        GeminiReasoner(model="m").ask("hi")


def test_success_is_logged(tmp_path):
    log = DecisionLog(str(tmp_path / "d.jsonl"))
    out = GeminiReasoner(model="m", log=log, call_fn=lambda m, p, c: "answer").ask("question")
    assert out == "answer"
    event = json.loads((tmp_path / "d.jsonl").read_text().splitlines()[0])
    assert event["kind"] == "genai_call" and event["output"] == "answer" and event["model"] == "m"


def test_retries_on_429_then_succeeds():
    calls = {"n": 0}
    sleeps = []

    def flaky(model, prompt, cfg):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 rate limited")
        return "ok"

    r = GeminiReasoner(model="m", call_fn=flaky, sleep=sleeps.append)
    assert r.ask("q") == "ok"
    assert calls["n"] == 3 and sleeps == [1, 2]


def test_non_retryable_error_fails_fast():
    calls = {"n": 0}

    def bad(model, prompt, cfg):
        calls["n"] += 1
        raise ValueError("bad request")

    with pytest.raises(GeminiUnavailable):
        GeminiReasoner(model="m", call_fn=bad, sleep=lambda s: None).ask("q")
    assert calls["n"] == 1


def test_rate_guard_blocks_then_recovers():
    now = {"t": 0.0}
    g = RateGuard(2, clock=lambda: now["t"])
    assert g.allow() and g.allow() and not g.allow()
    now["t"] = 61.0
    assert g.allow()


def test_reasoner_enforces_rate_guard():
    r = GeminiReasoner(model="m", max_calls_per_minute=1, call_fn=lambda m, p, c: "ok")
    assert r.ask("a") == "ok"
    with pytest.raises(GeminiUnavailable):
        r.ask("b")
