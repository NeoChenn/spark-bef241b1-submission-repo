import math

import numpy as np
import pytest

from infra.edge.stats import (agreement, binary_metrics, latency_stats, loop_rate, parse_smi_power, power_stats,
                              summarize_power_window)


def test_latency_stats_percentiles_and_rate():
    ms = list(range(1, 101))
    s = latency_stats(ms, batch=1)
    assert s["n"] == 100
    assert s["p50_ms"] == pytest.approx(50.5)
    assert s["p99_ms"] == pytest.approx(99.01)
    assert s["mean_ms"] == pytest.approx(50.5)
    assert s["min_ms"] == 1 and s["max_ms"] == 100
    assert s["hz_p50"] == pytest.approx(1000 / 50.5)
    assert s["throughput_per_s"] == pytest.approx(1000 / 50.5)


def test_latency_stats_batch_throughput():
    s = latency_stats([2.0] * 10, batch=64)
    assert s["throughput_per_s"] == pytest.approx(32000.0)
    assert s["hz_p50"] == pytest.approx(500.0)


def test_latency_stats_empty_raises():
    with pytest.raises(ValueError):
        latency_stats([])


def test_parse_smi_power_skips_bad_lines():
    lines = ["39.08", " 41.5 W", "[N/A]", "", "abc", "12"]
    assert parse_smi_power(lines) == [39.08, 41.5, 12.0]


def test_power_stats():
    p = power_stats([10.0, 20.0, 30.0])
    assert p["mean_w"] == pytest.approx(20.0)
    assert p["max_w"] == 30.0
    assert p["n"] == 3
    assert power_stats([]) == {"n": 0, "mean_w": None, "p50_w": None, "max_w": None}


def test_summarize_power_window_delta_and_energy():
    s = summarize_power_window(load=[50.0, 52.0], base=[40.0, 42.0], mean_ms=2.0, batch=1)
    assert s["load_mean_w"] == pytest.approx(51.0)
    assert s["base_mean_w"] == pytest.approx(41.0)
    assert s["delta_w"] == pytest.approx(10.0)
    assert s["mj_per_inference_total"] == pytest.approx(102.0)
    assert s["mj_per_inference_delta"] == pytest.approx(20.0)


def test_summarize_power_window_per_item_for_batches():
    s = summarize_power_window(load=[50.0], base=[40.0], mean_ms=2.0, batch=4)
    assert s["mj_per_inference_delta"] == pytest.approx(5.0)


def test_summarize_power_window_without_samples():
    s = summarize_power_window(load=[], base=[40.0], mean_ms=2.0)
    assert s["delta_w"] is None and s["mj_per_inference_total"] is None


def test_agreement():
    ref = np.array([-2.0, -0.1, 0.5, 3.0])
    got = np.array([-1.9, 0.1, 0.4, 3.2])
    a = agreement(ref, got, thr_logit=0.0)
    assert a["decision_agreement"] == pytest.approx(0.75)
    assert a["max_abs_logit_diff"] == pytest.approx(0.2)
    assert a["mean_abs_logit_diff"] == pytest.approx(0.15)


def test_binary_metrics():
    z = np.array([2.0, 1.0, -1.0, -2.0, 0.5])
    y = np.array([1, 0, 0, 0, 1])
    m = binary_metrics(z, y, thr_logit=0.0)
    assert m["tp"] == 2 and m["fp"] == 1 and m["tn"] == 2 and m["fn"] == 0
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["f1"] == pytest.approx(0.8)


def test_loop_rate():
    assert loop_rate([1.0, 1.5]) == pytest.approx(400.0)
    assert math.isinf(loop_rate([]))


def _fake_run(det_ms, pol_ms):
    def e(ms, w):
        return {"latency": {"p50_ms": ms, "p99_ms": 2 * ms}, "power": {"load_mean_w": w, "delta_w": 1.5}}
    return {"background": {"gpu_util_mean": 90.0, "gpu_power_mean_w": 40.0},
            "models": {"rust_detector": {"trt": {"fp16": e(det_ms, 45.0)}},
                       "leak_crack": {"trt": {"fp16": e(5.0, 46.0)}},
                       "scrub_policy": {"trt": {"fp16": e(pol_ms, 44.0)}}}}


def test_headline_and_summary():
    from infra.edge.report import headline, summary

    h = headline({"contended": _fake_run(2.0, 0.5)}, "contended")
    assert h["loop_hz_p50"] == pytest.approx(400.0)
    assert h["loop_hz_p99"] == pytest.approx(200.0)
    assert h["detector_gpu_w"] == 45.0
    s = summary(h)
    assert "400 Hz" in s and "Gemini" in s
