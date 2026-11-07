import numpy as np
import pytest

from sim.hdri import horizon_color
from sim.sky_gen import SkySpec, angle_between, directions, render_sky

SPEC = SkySpec(horizon=(0.02, 0.03, 0.08), mid=(0.06, 0.04, 0.12), zenith=(0.03, 0.02, 0.07), width=256)


def test_gradient_runs_navy_horizon_to_purple():
    sky = render_sky(SPEC)
    assert sky.shape == (128, 256, 3)
    assert horizon_color(sky, 3.0) == pytest.approx(SPEC.horizon, abs=0.01)
    mid_row = sky[int(128 * (0.5 - 0.35 / 2))].mean(0)
    assert mid_row == pytest.approx(SPEC.mid, abs=0.01)
    assert np.allclose(sky[-1], np.asarray(SPEC.horizon) * 0.7)


def test_glow_brightest_toward_sun():
    spec = SkySpec(**{**SPEC.__dict__, "sun_azimuth": 90.0, "sun_elevation": 3.0, "glow_color": (0.3, 0.2, 0.1)})
    sky = render_sky(spec)
    az, el = directions(spec.width)
    row = sky[60].sum(1)
    assert abs(np.degrees(az[60, row.argmax()]) - 90.0) < 3.0


def test_clouds_are_soft_and_only_above_horizon():
    spec = SkySpec(**{**SPEC.__dict__, "cloud_amount": 0.8, "cloud_color": (0.1, 0.09, 0.12)})
    sky = render_sky(spec)
    plain = render_sky(SPEC)
    assert np.abs(sky - plain).max() > 0.01
    assert np.allclose(sky[64:], plain[64:])
    assert np.abs(np.diff(sky[:64], axis=1)).max() < 0.01


def test_angle_between_zero_at_same_direction():
    assert angle_between(np.array(1.0), np.array(0.2), 1.0, 0.2) == pytest.approx(0.0, abs=1e-6)


def test_sun_disc_is_small_and_bright():
    spec = SkySpec(**{**SPEC.__dict__, "sun_azimuth": 90.0, "sun_elevation": 10.0, "disc": 5.0, "disc_deg": 3.0})
    sky = render_sky(spec)
    az, el = directions(spec.width)
    ang = np.degrees(angle_between(az, el, np.radians(90.0), np.radians(10.0)))
    assert sky[ang < 1.0].max(axis=-1).min() > 4.0
    assert sky[ang > 4.0].max() < 1.0


def test_fog_compensation_round_trips():
    from sim.sky_gen import compensate_for_fog

    sky = render_sky(SPEC)
    fog = (0.02, 0.025, 0.06)
    pre = compensate_for_fog(sky, fog, 0.6, floor=0.0)
    assert np.allclose(pre * 0.4 + 0.6 * np.asarray(fog), sky, atol=1e-5)
