import numpy as np
import pytest

from policy.weld.seam import Seam, SeamConfig, heat_input


def test_seam_geometry_and_crack():
    s = Seam(seed=1)
    c = s.cfg
    assert s.center.shape == (c.nx,)
    assert np.abs(s.center).max() <= c.wander_mm + c.misalign_mm + 1e-6
    assert np.abs(s.center).max() > 1.5
    assert s.crack.shape[1] == 2 and len(s.crack) >= 3
    assert s.crack_mask.any()
    x0, x1 = s.crack_extent()
    assert 0 <= x0 < x1 <= c.length_mm
    ys = s.crack[:, 1] - np.interp(s.crack[:, 0], s.xs, s.center)
    assert np.abs(ys).max() < c.groove_half_mm
    assert not Seam(seed=1, crack=False).crack_mask.any()


def test_height_profile_is_groove_centred_on_seam():
    s = Seam(seed=2)
    x = 100.0
    ys = np.linspace(-15, 15, 121)
    h = s.height(x, ys)
    assert ys[np.argmin(h)] == pytest.approx(s.center_at(x), abs=0.6)
    assert h.min() < -s.cfg.groove_depth_mm * 0.8


def test_heat_input_formula():
    assert heat_input(160.0, 22.0, 2.8, eta=0.8) == pytest.approx(0.8 * 22 * 160 / 2.8 / 1000)


def test_grind_removes_crack_and_deposit_covers_band():
    s = Seam(seed=3)
    x0, x1 = s.crack_extent()
    for x in np.arange(x0 - 5, x1 + 5, 1.0):
        s.grind(x, s.center_at(x), 8.0)
    assert not s.open_crack().any()
    hi = 1.0
    for x in np.arange(x0 - 5, x1 + 5, 1.0):
        s.deposit(x, s.center_at(x), 13.0, hi, 1.0)
    m = s.metrics(x0 - 3, x1 + 3)
    assert m["coverage"] > 0.98
    assert m["defect_rate"] < 0.05
    assert m["crack_left"] == 0.0


def test_offset_bead_leaves_defects_and_bad_heat_counts():
    s = Seam(seed=3)
    x0, x1 = s.crack_extent()
    for x in np.arange(x0, x1, 1.0):
        s.deposit(x, s.center_at(x) + 6.0, 13.0, 1.0, 1.0)
    assert s.metrics(x0, x1)["coverage"] < 0.8
    t = Seam(seed=3)
    for x in np.arange(x0, x1, 1.0):
        t.deposit(x, t.center_at(x), 13.0, 2.5, 1.0)
    assert t.metrics(x0, x1)["defect_rate"] > 0.9


def test_temperature_rises_with_heat_and_cools():
    s = Seam(seed=4)
    t0 = s.temp_c
    for x in range(60):
        s.deposit(float(x), s.center_at(x), 12.0, 1.2, 0.4)
    assert s.temp_c > t0 + 50
    hot = s.temp_c
    s.cool(30.0)
    assert s.temp_c < hot
    assert SeamConfig().max_interpass_c > 150
