import math

import numpy as np
import pytest

from policy.gasbot.infotaxis import Infotaxis, ParticleFilter, plume_ppm
from policy.gasbot.plume import RHO_CH4_G_M3, GaussianPlume, Wind
from policy.gasbot.sensors import LEL_PPM


def still_wind(direction=0.0, speed=3.0):
    return Wind(speed, direction, meander_deg=0.0, gust=0.0)


def test_vectorised_plume_matches_gaussian_plume():
    w = np.array([math.cos(0.4), math.sin(0.4)])
    pts = np.array([[x, y, z] for x in (-1.0, 0.5, 2.0, 4.0) for y in (-0.5, 0.0, 0.7) for z in (0.45, 0.95)])
    srcs = np.array([[0.0, 0.0, 1.0], [1.0, 0.3, 0.8]])
    qs = np.array([2.0, 7.0])
    got = plume_ppm(srcs, qs, pts, w, 3.0, meander=0.21)
    assert got.shape == (len(pts), 2)
    for k in range(2):
        ref = GaussianPlume(srcs[k], qs[k], still_wind()).concentration(pts, wind_unit=w, u=3.0, q=qs[k],
                                                                         meander=0.21) / RHO_CH4_G_M3 * 1e6
        assert np.allclose(got[:, k], ref, rtol=1e-9, atol=1e-9)


def test_resampling_restores_uniform_weights_around_survivors():
    pf = ParticleFilter((-5, 5), (-1, 4), n=500, seed=1)
    pf.w[:] = 1e-12
    pf.w[7] = 1.0
    pf.w /= pf.w.sum()
    keep = pf.p[7].copy()
    assert pf.ess() < 2
    pf.resample()
    assert np.allclose(pf.w, 1 / 500) and pf.ess() == pytest.approx(500)
    assert np.median(np.linalg.norm(pf.p[:, :2] - keep[:2], axis=1)) < 0.2
    assert np.all(pf.p >= pf.lo - 1e-9) and np.all(pf.p <= pf.hi + 1e-9)


def test_hit_likelihood_favours_upwind_centreline_sources():
    pf = ParticleFilter((-5, 5), (-1, 4), n=3, seed=0)
    pf.p[:] = [[0.0, 1.5, 0.95, math.log(3.0)], [0.0, 3.5, 0.95, math.log(3.0)], [4.0, 1.5, 0.95, math.log(3.0)]]
    pf.u = 3.0
    w = np.array([1.0, 0.0])
    ppm = plume_ppm(pf.p[:1, :3], [3.0], [[2.0, 1.5, 0.95]], w, 3.0)[0, 0]
    ll = pf.loglik([([2.0, 1.5, 0.95], ppm / LEL_PPM)], w)
    assert ll[0] > ll[1] + 2 and ll[0] > ll[2] + 2
    ll = pf.loglik([([2.0, 1.5, 0.95], 0.0)], w)
    assert ll[0] < ll[1] - 1 and ll[0] < ll[2] - 1
    pf.observe([([2.0, 1.5, 0.95], ppm / LEL_PPM)], w, 3.0)
    assert pf.n_obs == 1 and pf.n_hits == 1 and pf.w.sum() == pytest.approx(1.0)


def test_filter_converges_on_a_toy_plume_and_shrinks_its_ellipse():
    src = np.array([1.0, 1.5, 1.0])
    pl = GaussianPlume(src, 3.0, still_wind(), puff=0.0)
    pl.step(10.0)
    pf = ParticleFilter((-6, 6), (-1, 4), seed=3)
    a0, h0 = pf.ellipse_area(), pf.entropy()
    w = np.array([1.0, 0.0])
    for x in np.linspace(5.0, 1.6, 12):
        for y in np.linspace(0.3, 2.7, 7):
            for z in (0.45, 0.95):
                ppm = float(pl.sample([[x, y, z]])[0])
                pf.observe([([x, y, z], ppm / LEL_PPM)], w, 3.0)
    mean, sigma, _ = pf.estimate()
    assert np.linalg.norm(mean - src[:2]) < 0.5 and sigma < 0.6
    assert pf.ellipse_area() < 0.1 * a0 and pf.entropy() < h0 - 2.0
    assert 1.0 < pf.flow() < 9.0 and pf.n_hits > 0


def test_bearing_update_points_the_cloud():
    pf = ParticleFilter((-6, 6), (-1, 4), seed=4)
    for _ in range(6):
        pf.bearing([4.0, 1.5], math.pi, 0.1)
    mean, _, _ = pf.estimate()
    assert mean[0] < 3.0 and abs(mean[1] - 1.5) < 0.6 and pf.n_bear == 6


def test_information_gain_prefers_the_discriminating_spot():
    pf = ParticleFilter((-6, 6), (-1, 4), n=400, seed=5)
    pf.p[:200, :3] = [0.0, 0.6, 0.95]
    pf.p[200:, :3] = [0.0, 2.4, 0.95]
    pf.p[:, 3] = math.log(3.0)
    w = np.array([1.0, 0.0])
    mi = pf.mutual_info(np.array([[1.5, 0.6], [-4.0, 1.5], [1.5, 1.5]]), w)
    assert mi[0] > 0.3 and mi[0] > 10 * mi[1]
    pf.p[200:, :3] = [0.0, 0.6, 0.95]
    assert pf.mutual_info(np.array([[1.5, 0.6]]), w)[0] < 0.05


def test_infotaxis_moves_to_where_it_learns_most_then_homes_in():
    pf = ParticleFilter((-7.2, 7.2), (0.75, 2.4), n=400, seed=6)
    pf.p[:200, :3] = [0.0, 0.8, 0.95]
    pf.p[200:, :3] = [0.0, 2.3, 0.95]
    pf.p[:, 3] = math.log(3.0)
    pf.u = 3.0
    it = Infotaxis(pf)
    w = np.array([1.0, 0.0])
    v = it.command(0.0, [4.0, 1.5], 0.0, 0.0, [0.5, 0], w)
    assert v[0] < 0 and np.linalg.norm(v) == pytest.approx(it.cfg.speed, rel=1e-6)
    pf.p[:, :2] = [-3.0, 1.5]
    v = it.command(0.5, [1.0, 1.5], 0.0, 0.0, [0.5, 0], w)
    assert v[0] < -0.5 * it.cfg.speed and it.mode == "HOMING"


def test_gas_bot_infotaxis_mode_finds_and_fixes_a_loose_flange():
    from policy.gasbot.mock import build, episode
    r = episode(1003, "loose", max_t=600, search="infotaxis")
    assert r["detected"] and r["flange_ok"] and r["t_fixed"] is not None and r["loc_err"] < 1.5
    orch, bridge, _ = build(1003, "loose", search="infotaxis")
    assert orch.gcfg.search == "infotaxis"
    orch0, _, _ = build(1003, "loose")
    assert orch0.gcfg.search == "cast_surge"


def test_hud_header_carries_pf_belief_only_in_infotaxis_mode():
    from policy.gasbot.mock import build
    from rig.hud import HudFeed
    for search, has in (("infotaxis", True), (None, False)):
        orch, bridge, _ = build(1003, "loose", search=search)
        feed = HudFeed(orch)
        while orch.seeker is None:
            orch.step()
        orch.step()
        g = feed.header()["gas"]
        assert ("belief" in g) == has
        if has:
            assert g["belief"]["n"] == 2000 and g["belief"]["sigma_m"] > 0 and g["mode"] in ("INFOTAXIS", "HOMING")
