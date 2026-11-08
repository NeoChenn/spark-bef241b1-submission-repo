import math

import numpy as np
import pytest

from bench.detect import (NEEDS_KEY, binary_metrics, gemini_score, latency_stats, list_images, not_run,
                          stratified_subset)


def test_gemini_score_mapping():
    assert gemini_score("rust", 0.9) == pytest.approx(0.9)
    assert gemini_score("clean", 0.9) == pytest.approx(0.1)
    assert gemini_score("cannot_tell", 0.8) == 0.5
    assert gemini_score(None, 0.0) is None


def test_binary_metrics_perfect_and_ci():
    y = np.array([0, 1] * 30)
    s = np.where(y == 1, 0.9, 0.1)
    m = binary_metrics(y, s, 0.5)
    assert m["f1"]["mean"] == 1.0 and m["auroc"]["mean"] == 1.0
    assert m["confusion"] == {"tn": 30, "fp": 0, "fn": 0, "tp": 30}
    assert m["n"] == 60 and m["n_pos"] == 30


def test_binary_metrics_threshold_inclusive_and_ci_brackets():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    m = binary_metrics(y, s, 0.5)
    for k in ("f1", "auroc", "precision", "recall"):
        assert m[k]["lo"] <= m[k]["mean"] <= m[k]["hi"]
    edge = binary_metrics(np.array([1, 0]), np.array([0.5, 0.49]), 0.5)
    assert edge["confusion"]["tp"] == 1 and edge["confusion"]["fp"] == 0


def test_binary_metrics_single_class_auroc_nan():
    m = binary_metrics(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.2]), 0.5)
    assert math.isnan(m["auroc"]["mean"])
    assert m["recall"]["mean"] == pytest.approx(2 / 3)


def test_latency_stats():
    d = latency_stats([1.0, 2.0, 3.0, 4.0, 100.0])
    assert d["median_ms"] == 3.0 and d["n"] == 5 and 4.0 < d["p90_ms"] <= 100.0


def test_list_images(tmp_path):
    for c in ("rust", "clean"):
        (tmp_path / c).mkdir()
    for n in ("a.jpg", "b.png"):
        (tmp_path / "rust" / n).write_bytes(b"x")
    (tmp_path / "clean" / "c.jpg").write_bytes(b"x")
    (tmp_path / "clean" / "notes.txt").write_text("x")
    items = list_images(tmp_path)
    assert [(p.name, y) for p, y in items] == [("c.jpg", 0), ("a.jpg", 1), ("b.png", 1)]


def test_stratified_subset_deterministic_balanced():
    labels = [0] * 300 + [1] * 100
    idx = stratified_subset(labels, 40, seed=3)
    assert idx == stratified_subset(labels, 40, seed=3)
    assert len(idx) == 40 and len(set(idx)) == 40
    assert sum(labels[i] for i in idx) == 10
    assert stratified_subset(labels, 1000, seed=0) == list(range(400))


def test_run_gemini_arm_wiring_with_stub_transport():
    from bench.detect import run_gemini_arm

    img = np.zeros((8, 8, 3), np.uint8)
    items = [(img, 1, "a"), (img, 0, "b"), (img, 1, "c"), (img, 0, "d")]
    calls = []

    def transport(url, body):
        calls.append(url)
        return {"candidates": [{"content": {"parts": [
            {"text": '{"verdict": "rust", "confidence": 0.8, "reason": "r"}'}]}}]}

    arm = run_gemini_arm("gemini-test", {"d": items}, 10, workers=2, api_key="k", transport=transport)
    r = arm["results"]["d"]
    assert len(calls) == 4 and all("gemini-test" in u for u in calls)
    assert r["n_errors"] == 0 and r["confusion"]["tp"] == 2 and r["confusion"]["fp"] == 2
    assert arm["runs_on_robot"] is False and r["latency"]["n"] == 4


def test_merge_gemini_replaces_not_run_arms_in_place_of_old():
    from bench.detect import merge_gemini

    doc = {"arms": [{"name": "Colour stand-in"}, {"name": "Gemini m", "status": "not run"}]}
    out = merge_gemini(doc, [{"name": "Gemini m", "results": {}}])
    assert [a["name"] for a in out["arms"]] == ["Colour stand-in", "Gemini m"] and "status" not in out["arms"][1]


def test_not_run_marker():
    d = not_run("gemini-4-argon")
    assert d["status"] == NEEDS_KEY and d["model"] == "gemini-4-argon" and "results" not in d
