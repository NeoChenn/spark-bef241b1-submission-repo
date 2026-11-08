import math

import numpy as np

from sim import ocean_fx as fx
from sim.ocean import WAVES, surface


def test_big_waves_match_geometry_jacobian():
    big = fx.big_waves()
    rest = np.array([[3.0, -7.0], [40.0, 12.0]])
    eps, t = 1e-3, 4.2
    dist = np.linalg.norm(rest, axis=1)
    pts, _ = surface(rest, dist, t, 1.5)
    px, _ = surface(rest + [eps, 0], dist, t, 1.5)
    py, _ = surface(rest + [0, eps], dist, t, 1.5)
    jxx, jyx = (px[:, 0] - pts[:, 0]) / eps, (px[:, 1] - pts[:, 1]) / eps
    jxy, jyy = (py[:, 0] - pts[:, 0]) / eps, (py[:, 1] - pts[:, 1]) / eps
    flat = fx.detail_waves()
    flat.amp[:] = 0
    np.testing.assert_allclose(fx.jacobian(big, flat, 1.0, rest, t), jxx * jyy - jxy * jyx, atol=2e-3)
    assert len(big[0]) == len(WAVES)


def test_detail_waves_are_short_and_bounded():
    w = fx.detail_waves(max_wavelength=10.0, choppiness=0.7)
    assert w.wavelength.max() <= 10.0 and w.wavelength.min() < 0.1
    assert math.isclose(float(np.sum(w.q * w.k * w.amp)), 0.7, rel_tol=1e-4)


def test_storm_gets_more_whitecaps():
    big, detail = fx.big_waves(), fx.detail_waves()
    night = fx.foam_threshold(big, detail, fx.FX["night"]["chop"], fx.FX["night"]["coverage"])
    storm = fx.foam_threshold(big, detail, fx.FX["storm"]["chop"], fx.FX["storm"]["coverage"])
    assert storm < 1.0 and night < storm
    rng = np.random.default_rng(3)
    xy = rng.uniform(-150, 150, (8000, 2))
    frac = np.mean(fx.jacobian(big, detail, fx.FX["storm"]["chop"], xy, 17.0) < storm)
    assert 0.05 < frac < 0.14


def test_mdl_bakes_waves_legs_and_no_placeholders():
    big, detail = fx.big_waves(), fx.detail_waves()
    src = fx.build_mdl(big, detail, [(1.0, 2.0, 5.0), (3.0, 4.0, 5.0)])
    assert f"WAVE_COUNT = {detail.k.size};" in src and f"BIG_COUNT = {len(WAVES)};" in src
    assert src.count("float4(") - fx.build_mdl(big, detail, []).count("float4(") == 0
    assert src.split("const int LEG_COUNT")[0].count("float4(") == 2 * (detail.k.size + len(WAVES))
    assert "LEG_COUNT = 2;" in src and "float3(3, 4, 5)" in src
    assert "@" not in src and "@" not in fx.build_mdl(big, detail, [])
    assert "export material OceanFx(" in src


def test_preset_inputs_and_pixel_angle():
    night = fx.preset_inputs("night", 1.5, 0.6)
    assert night["height"] == 1.5 and night["foam_threshold"] == 0.6
    assert fx.preset_inputs("unknown", 1.0, 0.5)["chop"] == fx.FX["night"]["chop"]
    assert math.isclose(fx.pixel_angle(20.955, 18.0, 1280.0), 20.955 / 18.0 / 1280.0)


def test_parse_overrides():
    assert fx.parse_overrides("chop=2, slope_gain=1.5,bad") == {"chop": 2.0, "slope_gain": 1.5}
    assert fx.parse_overrides("") == {}


def test_live_tier_halves_waves():
    assert fx.wave_stride("offline") == 1 and fx.wave_stride("live") == 3


def test_moved_threshold():
    assert fx.moved(None, (0.0, 0.0, 0.0), 1.0)
    assert not fx.moved((0.0, 0.0, 0.0), (0.5, -0.5, 0.2), 1.0)
    assert fx.moved((0.0, 0.0, 0.0), (0.0, 1.5, 0.0), 1.0)


def test_storm_crests_stay_below_braces():
    from sim.ocean import make_grid

    rest, distance, _, _ = make_grid()
    near = distance < 80
    crest = max(surface(rest[near], distance[near], t, 2.5)[0][:, 2].max() for t in np.arange(0, 120, 1.0))
    assert crest < 7.0
