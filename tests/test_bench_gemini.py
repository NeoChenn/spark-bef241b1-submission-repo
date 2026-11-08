import pytest

from bench import gemini
from bench.scrub_gemini import describe, run_episode
from bench.scrub_sim import KinematicScrub, ScrubCfg, make_scenario


def test_resolve_model_exact_prefix_missing():
    av = ["gemini-4-argon-preview-11", "gemini-robotics-er-2-preview", "gemini-4-argon"]
    assert gemini.resolve_model("gemini-4-argon", av) == "gemini-4-argon"
    assert gemini.resolve_model("gemini-4-argon", av[:2]) == "gemini-4-argon-preview-11"
    assert gemini.resolve_model("gemini-9", av) is None


def test_cost_needs_prices(monkeypatch):
    monkeypatch.delenv("GEMINI_PRICE_IN_PER_MTOK", raising=False)
    monkeypatch.delenv("GEMINI_PRICE_OUT_PER_MTOK", raising=False)
    assert gemini.cost_usd(1000, 1000) is None
    assert gemini.cost_usd(1_000_000, 0, "gemini-4-argon") > 0
    monkeypatch.setenv("GEMINI_PRICE_IN_PER_MTOK", "2")
    monkeypatch.setenv("GEMINI_PRICE_OUT_PER_MTOK", "10")
    assert gemini.cost_usd(1_000_000, 100_000) == pytest.approx(3.0)


def test_generate_parses_json_and_usage():
    def post(url, body):
        assert url.endswith("models/m:generateContent")
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        return {"candidates": [{"content": {"parts": [{"text": '{"d_theta": 0.1, "d_z": 0.0, "press": 0.5}'}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3, "thoughtsTokenCount": 2}}
    r = gemini.generate("m", "k", [{"text": "x"}], {}, post=post)
    assert r["json"]["press"] == 0.5 and r["tokens_in"] == 10 and r["tokens_out"] == 5


def test_describe_state_is_json_ready():
    sim = KinematicScrub([make_scenario(3)])
    d = describe(sim, 0)
    assert d["rusty_cells"] > 0 and len(d["nearest_rust_cells"]) <= 12 and d["rust_centroid"] is not None


def test_controller_loop_holds_and_logs_calls():
    cfg = ScrubCfg(episode_s=1.0)
    n = {"k": 0}

    def post(url, body):
        n["k"] += 1
        return {"candidates": [{"content": {"parts": [{"text": '{"d_theta": 0.0, "d_z": 0.02, "press": 0.6}'}]}}],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10}}
    ep = run_episode(make_scenario(5, cfg), "m", "k", cfg, post=post)
    assert ep["calls"] == n["k"] >= 1 and ep["errors"] == 0 and ep["tokens_in"] == 100 * n["k"]


def test_no_key_reports_not_run(monkeypatch):
    from bench import scrub
    monkeypatch.setattr(gemini, "api_key", lambda: None)
    r = scrub.run_gemini([], ScrubCfg(), 5, gemini.ARGON)
    assert r["status"] == gemini.NEEDS_KEY


def test_generate_sets_thinking_level():
    seen = {}

    def post(url, body):
        seen.update(body["generationConfig"])
        return {"candidates": [{"content": {"parts": [{"text": '{"press": 0.1}'}]}}]}
    gemini.generate("m", "k", [{"text": "x"}], {}, post=post, thinking="low")
    assert seen["thinkingConfig"] == {"thinkingLevel": "low"}


def test_decision_steps_paused_vs_realtime():
    from bench.scrub_gemini import decision_steps
    dt = 1 / 30
    assert decision_steps("paused", 9.0, dt, 0.5) == 15
    assert decision_steps("realtime", 4.0, dt, 0.5) == 120
    assert decision_steps("realtime", 0.0, dt, 0.5) == 1


def test_paused_mode_calls_every_hold_and_logs_wall():
    cfg = ScrubCfg(episode_s=1.0)
    n = {"k": 0}

    def post(url, body):
        n["k"] += 1
        assert "paused" in body["contents"][0]["parts"][0]["text"]
        return {"candidates": [{"content": {"parts": [{"text": '{"d_theta": 0.0, "d_z": 0.02, "press": 0.6}'}]}}],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10}}
    ep = run_episode(make_scenario(5, cfg), "m", "k", cfg, post=post, mode="paused", hold_s=0.25)
    assert ep["mode"] == "paused" and ep["calls"] == n["k"] >= 3 and ep["wall_s"] >= 0


def test_within_budget_and_spend_cap():
    from bench.scrub_vs_gemini import CostCap, Spend, within_budget
    eps = [{"success": True, "time_to_clean_s": 5.0}, {"success": True, "time_to_clean_s": 9.0},
           {"success": False, "time_to_clean_s": None}, {"success": True, "time_to_clean_s": 7.0}]
    w = within_budget(eps, 7.0)
    assert w["mean"] == 0.5 and w["n"] == 4 and w["lo"] < 0.5 < w["hi"]
    s = Spend(1.0)
    s.add(0.6)
    s.add(None)
    with pytest.raises(CostCap):
        s.add(0.5)


def test_latency_note_ratio():
    from bench.scrub_vs_gemini import LABELS, latency_note
    arms = {LABELS["ppo"]: {"decision_latency_ms": {"median": 0.01}},
            LABELS["paused"]: {"decision_latency_ms": {"median": 8000.0}}}
    assert "800,000x" in latency_note(arms) and "8.0 s" in latency_note(arms)
