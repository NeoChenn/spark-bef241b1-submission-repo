import numpy as np
import pytest

from sim.ocean_waves import G, displace, make_waves, polar_grid, significant_height


def test_dispersion_is_deep_water():
    w = make_waves(wind_speed=10.0, wind_dir_deg=30.0, seed=1)
    assert np.allclose(w.omega**2, G * w.k, rtol=1e-5)


def test_significant_height_matches_pierson_moskowitz():
    for u in (6.0, 10.0, 14.0):
        w = make_waves(wind_speed=u, wind_dir_deg=0.0, seed=2, min_wavelength=0.5)
        assert significant_height(w) == pytest.approx(0.21 * u * u / G, rel=0.2)


def test_steepness_never_folds_more_than_limit():
    w = make_waves(wind_speed=14.0, wind_dir_deg=0.0, seed=3, choppiness=1.0)
    assert np.sum(w.q * w.k * w.amp) <= 1.0 + 1e-6


def test_waves_travel_mostly_downwind():
    w = make_waves(wind_speed=10.0, wind_dir_deg=90.0, seed=4)
    energy = w.amp**2
    assert np.sum(energy * w.dir[:, 1]) > 0.6 * np.sum(energy)


def test_displace_mean_level_and_motion():
    w = make_waves(wind_speed=8.0, wind_dir_deg=0.0, seed=5)
    xs, ys = np.meshgrid(np.linspace(-200, 200, 120), np.linspace(-200, 200, 120))
    x, y = xs.ravel().astype(np.float32), ys.ravel().astype(np.float32)
    d0 = displace(w, x, y, 0.0)
    d1 = displace(w, x, y, 1.0)
    assert d0.shape == (x.size, 3)
    assert abs(d0[:, 2].mean()) < 0.1
    assert np.abs(d1 - d0).max() > 0.05


def test_displace_fade_flattens_far_points():
    w = make_waves(wind_speed=8.0, wind_dir_deg=0.0, seed=5)
    x = np.array([0.0, 5000.0], np.float32)
    d = displace(w, x, np.zeros_like(x), 3.0, fade=np.array([1.0, 0.0], np.float32))
    assert np.allclose(d[1], 0.0)


def test_polar_grid_topology():
    g = polar_grid(segments=64, inner_radius=10.0, inner_step=1.0, outer_radius=1000.0, aspect=2.0)
    n = g.xy.shape[0]
    assert g.counts.sum() == g.indices.size
    assert g.indices.max() == n - 1 and g.indices.min() == 0
    r = np.hypot(g.xy[:, 0], g.xy[:, 1])
    assert r.max() == pytest.approx(1000.0, rel=0.05)
    assert np.all(np.diff(g.ring_radius) > 0)
    assert np.all(g.spacing > 0)


def test_mdl_bakes_every_wave_and_leg():
    from sim.ocean_mdl import build_mdl

    w = make_waves(wind_speed=8.0, wind_dir_deg=0.0, seed=1)
    src = build_mdl(w, legs=[(1.0, 2.0, 0.8), (3.0, 4.0, 0.8)])
    assert f"WAVE_COUNT = {w.k.size};" in src
    assert src.count("float4(") == 2 * w.k.size
    assert "LEG_COUNT = 2;" in src and "float3(3, 4, 0.8)" in src
    assert "LEG_COUNT = 0;" in build_mdl(w)
    assert "@" not in build_mdl(w)
