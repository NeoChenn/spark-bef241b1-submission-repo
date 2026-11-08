"""Scenario runner metrics and the Gemini analyst loop with a fake Gemini (no network)."""
import json

import numpy as np

from policy import gemini_analyst as ga
from policy.scenario_runner import classify, normalise, random_scenario, run_scenario


def test_normalise_clamps_and_defaults():
    sc = normalise({"patch_level": 7, "occlusion_phase": "soak", "glare": "no", "force_spike_step": -3})
    assert sc["patch_level"] == 1.0 and sc["occlusion_phase"] == "SOAK" and sc["glare"] is False
    assert sc["force_spike_step"] == 0 and sc["abrasion_coef"] == 1.0


def test_default_scenario_repairs_and_contains_spike():
    r = run_scenario({"id": "default"})
    assert r["outcome"] == "DONE" and r["label"] == "success" and not r["unsafe"]
    assert r["backoffs"] == 1 and r["max_force"] == 65.0
    assert r["new_patch_injected"] and r["new_patch_detected"]


def test_stubborn_rust_is_never_marked_done_with_rust_left():
    r = run_scenario({"id": "stubborn", "abrasion_coef": 0.03, "force_spike_step": 0, "new_patch": False})
    assert r["outcome"] in ("DONE", "ESCALATE_TO_HUMAN")
    assert r["painted_over_rust"] == (r["outcome"] == "DONE" and r["true_rust_left"] > 0.05)


def test_classify_unsafe_first():
    base = {"painted_over_rust": False, "missed_rust": False, "outcome": "DONE", "over_limit_steps": 0,
            "final_reason": ""}
    assert classify(base) == "success"
    assert classify(base | {"painted_over_rust": True}) == "painted_over_rust"
    assert classify(base | {"outcome": "ESCALATE_TO_HUMAN", "final_reason": "perception lost for 5 s"}) == \
        "escalate_perception_lost"


def fake_gemini():
    calls = []

    def transport(url, body):
        prompt = body["contents"][0]["parts"][0]["text"]
        calls.append(prompt)
        if "test report" in prompt:
            payload = {"headline": "h", "failure_modes": [{"name": "faint rust", "severity": "critical", "count": 1,
                                                           "evidence_run_ids": ["g1-001"], "trigger": "t",
                                                           "likely_cause": "c", "suggested_fix": "f"}]}
        else:
            payload = [{"id": f"g-{i}", "rationale": "probe", "patch_level": 0.3 + 0.1 * i} for i in range(3)]
        return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}

    g = ga.Gemini(offline=True, transport=transport)
    g.key, g.model = "test", "fake-model"
    return g, calls


def test_closed_loop_with_fake_gemini(tmp_path):
    g, calls = fake_gemini()
    rng = np.random.default_rng(0)
    r1, src1 = ga.design(g, 3, 1, rng)
    results = [run_scenario(s) | {"round": 1} for s in r1]
    r2, src2 = ga.design(g, 3, 2, rng, results)
    assert src1 == src2 == "gemini" and len(r1) == len(r2) == 3
    assert "Round 1 results" in calls[1]                 # round 2 is told what failed in round 1
    analysis, src = ga.analyse(g, results)
    assert src == "gemini" and analysis["failure_modes"][0]["name"] == "faint rust"


def test_offline_pipeline_writes_report(tmp_path):
    assert ga.main(["--n", "6", "--rounds", "2", "--offline", "--out", str(tmp_path)]) == 0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Failure modes" in report and "offline" in report
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 12


def test_offline_round2_perturbs_failures():
    rng = np.random.default_rng(3)
    results = [run_scenario(random_scenario(rng, i)) for i in range(6)]
    follow, src = ga.design(ga.Gemini(offline=True), 5, 2, rng, results)
    assert src == "offline" and len(follow) == 5
