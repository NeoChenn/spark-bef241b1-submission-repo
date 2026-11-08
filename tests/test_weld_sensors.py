import numpy as np
import pytest

from policy.weld.seam import Seam
from policy.weld.sensors import SensorConfig, eddy, groove_center, laser_profile, telemetry, thermal


def test_laser_profile_finds_groove_centre_through_noise():
    s = Seam(seed=5)
    rng = np.random.default_rng(0)
    errs = []
    for x in np.linspace(20, 300, 40):
        ys, z = laser_profile(s, x, 0.0, rng)
        assert ys.shape == z.shape == (SensorConfig().laser_n,)
        c = groove_center(ys, z)
        errs.append(c - s.center_at(x))
    assert np.sqrt(np.mean(np.square(errs))) < 0.5


def test_groove_centre_none_when_bead_fills_groove():
    s = Seam(seed=5)
    for x in range(90, 111):
        s.deposit(float(x), s.center_at(x), 14.0, 1.0, 0.3)
    ys, z = laser_profile(s, 100.0, 0.0, np.random.default_rng(1))
    assert groove_center(ys, z) is None


def test_eddy_sees_open_crack_only():
    s = Seam(seed=6)
    rng = np.random.default_rng(0)
    on = [eddy(s, x, float(np.interp(x, s.crack[:, 0], s.crack[:, 1])), rng) for x in s.crack[1:-1, 0]]
    off = [eddy(s, x, s.center_at(x), rng) for x in np.linspace(5, s.crack_extent()[0] - 10, 20)]
    assert np.mean(on) > 0.6 and np.mean(off) < 0.2
    for x in np.arange(0, 320, 1.0):
        s.grind(x, s.center_at(x), 10.0)
    after = [eddy(s, x, float(np.interp(x, s.crack[:, 0], s.crack[:, 1])), rng) for x in s.crack[1:-1, 0]]
    assert np.mean(after) < 0.2


def test_telemetry_and_thermal_are_noisy_but_close():
    rng = np.random.default_rng(0)
    t = telemetry(160.0, 22.0, 3.0, rng)
    assert set(t) == {"current_a", "voltage_v", "speed_mm_s"}
    assert t["current_a"] == pytest.approx(160.0, abs=8) and t["voltage_v"] == pytest.approx(22.0, abs=1.5)
    s = Seam(seed=0)
    s.temp_c = 120.0
    assert thermal(s, rng) == pytest.approx(120.0, abs=15)
