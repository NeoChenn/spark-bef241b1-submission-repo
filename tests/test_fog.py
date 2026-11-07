import numpy as np
import pytest

from sim.fog import optical_depth, transmittance


def numeric(zc, zp, length, rho, h, n=20000):
    s = (np.arange(n) + 0.5) / n
    z = np.maximum(zp + (zc - zp) * s, 0)
    return rho * length * np.exp(-z / h).mean()


@pytest.mark.parametrize("zc,zp,length", [(62.0, 0.0, 230.0), (2.5, 8.0, 40.0), (30.0, 30.0, 500.0), (5.0, 0.0, 80.0)])
def test_matches_numeric_integral(zc, zp, length):
    assert optical_depth(zc, zp, length, 0.02, 5.0) == pytest.approx(numeric(zc, zp, length, 0.02, 5.0), rel=1e-3)


def test_fog_settles_low_and_grows_with_distance():
    near_low = transmittance(2.0, 0.0, 50.0, 0.02, 5.0)
    far_low = transmittance(2.0, 0.0, 2000.0, 0.02, 5.0)
    high = transmittance(60.0, 40.0, 50.0, 0.02, 5.0)
    assert far_low < near_low < high
    assert high > 0.99


def test_haze_adds_distance_falloff():
    assert transmittance(60.0, 60.0, 3000.0, 0.0, 5.0, haze_distance=3000.0) == pytest.approx(np.exp(-1.0))
