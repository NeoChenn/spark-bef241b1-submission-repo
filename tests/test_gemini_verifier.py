"""Gemini second opinion: reply parsing, offline fallback and the two orchestrator hooks (no network)."""
import json

import numpy as np

from policy.demo_mock import build
from policy.gemini_verifier import GeminiVerdict, GeminiVerifier, _extract_text, parse_verdict
from policy.orchestrator import S

IMAGE = np.full((240, 320, 3), 90, np.uint8)


def reply(payload: dict) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}


def test_parse_scales_points_from_yx_1000():
    v = parse_verdict('```json\n{"verdict": "rust", "confidence": 0.9, "points": [{"point": [500, 250], '
                      '"label": "rust"}], "reason": "orange flaking"}\n```', width=320, height=240)
    assert v.rust and v.confidence == 0.9
    assert v.points == [(80.0, 120.0)]


def test_unknown_verdict_becomes_cannot_tell():
    assert parse_verdict('{"verdict": "maybe", "confidence": 2, "reason": ""}', 10, 10).verdict == "cannot_tell"


def test_extract_text_handles_interactions_shape():
    assert _extract_text({"output_text": '{"verdict": "clean"}'}) == '{"verdict": "clean"}'
    assert _extract_text({"outputs": [{"text": "x"}]}) == "x"


def test_request_has_image_and_schema():
    sent = {}

    def transport(url, body):
        sent.update(url=url, body=body)
        return reply({"verdict": "clean", "confidence": 0.8, "reason": "painted"})

    v = GeminiVerifier(api_key="test", model="m", transport=transport).verify(IMAGE, "clean")
    assert v.clean and v.model == "m" and v.latency_ms >= 0
    assert sent["url"].endswith("/models/m:generateContent")
    parts = sent["body"]["contents"][0]["parts"]
    assert parts[0]["inlineData"]["mimeType"] == "image/jpeg" and "paint" in parts[1]["text"]
    assert sent["body"]["generationConfig"]["responseMimeType"] == "application/json"


def test_no_key_or_network_error_returns_none():
    assert GeminiVerifier(api_key="", transport=lambda u, b: {}).verify(IMAGE) is None

    def broken(url, body):
        raise OSError("network down")

    v = GeminiVerifier(api_key="test", transport=broken)
    assert v.verify(IMAGE) is None and "network down" in v.last_error


class StubAdvisor:
    def __init__(self, answers):
        self.answers, self.questions, self.last_error = dict(answers), [], None

    def verify(self, image, question="rust"):
        self.questions.append(question)
        return self.answers.get(question)

    def confident(self, v):
        return v is not None and v.verdict != "cannot_tell" and v.confidence >= 0.6


def run(orch, limit=400):
    for _ in range(limit):
        if orch.state in (S.DONE, S.ESCALATE_TO_HUMAN):
            break
        orch.step()
    return orch


def test_gemini_veto_sends_robot_back_to_scrub(tmp_path):
    orch, *_ = build(str(tmp_path / "d.jsonl"), glare=False, verbose=False)
    orch.advisor = StubAdvisor({"clean": GeminiVerdict("rust", 0.9, "orange patch at top edge")})
    run(orch)
    vetoes = [r for r in orch.records if r.reason.startswith("Gemini veto")]
    assert 1 <= len(vetoes) <= 2
    assert all(r.to == "SCRUB" and r.evidence["gemini"]["verdict"] == "rust" for r in vetoes)
    assert "gemini: rust" in orch.overlay_text()[-1]


def test_gemini_clean_confirmation_lets_paint_go_ahead(tmp_path):
    orch, *_ = build(str(tmp_path / "d.jsonl"), glare=False, verbose=False)
    orch.advisor = StubAdvisor({"clean": GeminiVerdict("clean", 0.95, "painted, no rust")})
    run(orch)
    assert orch.state == S.DONE
    assert "clean" in orch.advisor.questions


def test_unavailable_gemini_changes_nothing(tmp_path):
    base, *_ = build(str(tmp_path / "a.jsonl"), glare=False, verbose=False)
    with_none, *_ = build(str(tmp_path / "b.jsonl"), glare=False, verbose=False)
    with_none.advisor = StubAdvisor({})
    run(base), run(with_none)
    assert [r.to for r in base.records] == [r.to for r in with_none.records]
    assert len(with_none.overlay_text()) == 4
