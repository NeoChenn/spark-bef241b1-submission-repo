import json

import numpy as np
import pytest

from policy.force_tools import (clamp_force_action, crossing, force_stats, intervals, resample_rows,
                                saturation_point, setpoint_action, telemetry_frame)


def test_clamp_keeps_setpoint_at_or_below_cap():
    assert clamp_force_action(1.0, 38.0) == pytest.approx(0.65)
    assert clamp_force_action(0.2, 38.0) == pytest.approx(0.2)
    assert 25 + 20 * clamp_force_action(0.9, 38.0 * 0.7) == pytest.approx(26.6)


def test_clamp_vectorised():
    a = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(clamp_force_action(a, 30.0), [-1.0, 0.0, 0.25])


def test_setpoint_action_inverts_mapping():
    assert setpoint_action(5.0) == pytest.approx(-1.0)
    assert setpoint_action(45.0) == pytest.approx(1.0)
    assert setpoint_action(25.0) == pytest.approx(0.0)


def test_intervals_from_flags():
    t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    f = [False, True, True, False, True, True]
    assert intervals(t, f) == [(0.1, 0.3), (0.4, 0.5)]
    assert intervals(t, [False] * 6) == []


def test_crossing_interpolates():
    assert crossing([10, 20, 30, 40], [20, 30, 38, 46], 40.0) == pytest.approx(32.5)
    assert crossing([10, 20], [1, 2], 40.0) is None


def test_saturation_point_reaches_fraction_of_max():
    x = np.array([5, 10, 15, 20, 25, 30])
    y = np.array([0.1, 0.5, 0.8, 0.95, 1.0, 1.0])
    assert saturation_point(x, y, 0.95) == pytest.approx(20.0)
    assert saturation_point(x, y, 0.9) == pytest.approx(15 + 5 * (0.9 - 0.8) / 0.15)


def test_resample_rows_shape_and_endpoints():
    g = np.tile(np.linspace(0, 1, 32)[:, None], (1, 64))
    out = resample_rows(g, 40)
    assert out.shape == (40, 64)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[-1, 0] == pytest.approx(1.0)
    assert np.all(np.diff(out[:, 3]) >= 0)


def test_force_stats():
    f = np.array([0.0, 10.0, 30.0, 45.0, 0.0])
    s = force_stats(f, 0.1, limit=40.0)
    assert s["peak_n"] == pytest.approx(45.0)
    assert s["frac_over_limit"] == pytest.approx(0.2)
    assert s["frac_contact_over_limit"] == pytest.approx(1 / 3)
    assert s["mean_contact_n"] == pytest.approx(85 / 3)
    assert s["time_over_limit_s"] == pytest.approx(0.1)


def test_telemetry_frame_is_json_with_40x64_grid():
    g = np.random.rand(32, 64)
    fr = telemetry_frame(1.234, "SCRUB", 31.27, 0.4, 0.91, [{"to": "SCRUB"}], g, force_model_n=30.0)
    assert fr["t"] == 1.234 and fr["state"] == "SCRUB"
    assert fr["force_n"] == 31.27
    assert len(fr["rust_grid"]) == 40 and len(fr["rust_grid"][0]) == 64
    assert max(max(r) for r in fr["rust_grid"]) <= 1.0
    assert fr["force_model_n"] == 30.0
    json.dumps(fr)


def test_colour_classifier_hue_threshold_rejects_olive_paint():
    from detection.localize import colour_classifier, fake_colour_classifier

    olive = np.tile(np.array([51, 47, 15], np.uint8), (2, 8, 8, 1))
    rust = np.tile(np.array([133, 54, 10], np.uint8), (2, 8, 8, 1))
    strict = colour_classifier(0.3)
    assert (strict(olive) < 0).all() and (strict(rust) > 0).all()
    assert (fake_colour_classifier(rust) > 0).all()


def test_injected_patch_lands_clear_of_existing_rust_and_reachable():
    import torch

    from integration.lab_bridge import LabWorld
    from policy.envs.rust_state import RustState

    s = RustState(1, 32, 64, 0.6)
    s.radius[:] = 0.15
    s.reachable[:] = False
    s.reachable[0, 4:28, 5:30] = True
    w = LabWorld(s)
    s.rust = w.blob(1.2, 0.1, 0.05, 0.85)
    th, z = w.inject_patch(1.2, 0.13, size=0.035)
    gap = (((th - 1.2) * 0.15) ** 2 + (z - 0.1) ** 2) ** 0.5
    assert gap > 0.16 and abs(z) < 0.2
    i, j = s.H * (z + 0.3) / 0.6, th / (2 * 3.14159265) * s.W
    assert 4 <= i < 28 and 5 <= j < 30
    assert float(s.rust.sum()) > float(w.blob(1.2, 0.1, 0.05, 0.85).sum())
