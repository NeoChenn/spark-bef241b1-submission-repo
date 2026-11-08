import math

import numpy as np
import pytest

from policy.gasbot import deck
from policy.gasbot.locate import CastSurge, SourceEstimator, acoustic_flow, flow_estimate
from policy.gasbot.plume import RHO_CH4_G_M3, GaussianPlume, PlumeField, Wind
from policy.gasbot.sensors import (LEL_PPM, AcousticSensor, H2SCell, PointGasSensor, column_density, lag,
                                   ogi_colour)


def still_wind(direction=0.0, speed=3.0):
    return Wind(speed, direction, meander_deg=0.0, gust=0.0)


def test_plume_is_downwind_and_scales_with_flow():
    p = GaussianPlume([0, 0, 1.0], 2.0, still_wind(0.0), puff=0.0)
    p.step(5.0)
    down, up, side = p.sample([[3, 0, 1.0], [-3, 0, 1.0], [0, 3, 1.0]])
    assert down > 100 * max(up, side) and up < 3 and side < 3
    p.set_flow(4.0)
    p.step(5.0)
    assert 1.3 < (p.sample([[3, 0, 1.0]])[0] - 2) / (down - 2) < 2.0
    assert p.flow_rate() == 4.0


def test_plume_change_travels_with_the_wind():
    p = GaussianPlume([0, 0, 1.0], 2.0, still_wind(0.0, 2.0), puff=0.0)
    for _ in range(50):
        p.step(0.1)
    p.set_flow(0.0)
    for _ in range(6):
        p.step(0.1)
    near, far = p.sample([[0.5, 0, 1.0], [6.0, 0, 1.0]])
    assert near < 5 and far > 50


def test_field_sums_sources_and_reports_leaks():
    f = PlumeField(still_wind(), {"A": [0, 0, 1], "B": [0, 5, 1]})
    f.set_flow("A", 1.0)
    f.step(1.0)
    assert f.flow_rate() == 1.0 and len(f.leaks()) == 1
    a, b = f.sample([[2, 0, 1], [2, 5, 1]])
    assert a > 100 and b == pytest.approx(f.background)


def test_first_order_lag_reaches_90_percent_at_t90():
    y = 0.0
    for _ in range(60):
        y = lag(y, 100.0, 0.1, 6.0)
    assert y == pytest.approx(90.0, abs=0.5)


def test_point_sensor_saturates_and_flags():
    s = PointGasSensor("ir", noise_ppm=0, drift_ppm_rt_s=0, cross={})
    for _ in range(300):
        s.update(2 * LEL_PPM, 0.1)
    assert s.saturated and s.lel == pytest.approx(1.0)


def test_cross_sensitivity_and_h2s_poisoning():
    ir = PointGasSensor("ir", noise_ppm=0, drift_ppm_rt_s=0)
    cat = PointGasSensor("cat", "catalytic", noise_ppm=0, drift_ppm_rt_s=0, cross={"ethane": 0.8})
    assert ir.sensitivity > 1.0
    s0 = cat.sensitivity
    for _ in range(3000):
        cat.update(1000.0, 1.0, h2s_ppm=50.0)
    assert cat.sensitivity < 0.8 * s0


def test_h2s_cell_is_slow():
    c = H2SCell(noise_ppm=0)
    for _ in range(100):
        c.update(20.0, 0.1)
    assert 5 < c.reading < 15


def test_acoustic_level_bearing_and_quiet_deck():
    a = AcousticSensor(noise_db=0, multipath=0.0, seed=1)
    near = a.measure([0, 0], [([2.0, 0.0], 5.0)])
    far = a.measure([0, 0], [([12.0, 0.0], 5.0)])
    assert near.db > far.db and near.snr_db > 6
    assert abs(near.bearing) < math.radians(20)
    quiet = a.measure([0, 0], [])
    assert quiet.bearing is None and quiet.snr_db < 0


def test_known_machinery_is_not_excess_sound():
    a = AcousticSensor(noise_db=0, machinery=(((1.0, 0.0), 80.0),))
    r = a.measure([0, 0], [])
    assert r.db > 70 and r.snr_db < 0


def test_acoustic_flow_inverts_the_level_model():
    a = AcousticSensor(noise_db=0)
    r = a.measure([0, 0], [([3.0, 0.0], 4.0)])
    assert acoustic_flow(r.db, r.background_db, 3.0, a.l1m_db, a.absorb_db_m) == pytest.approx(4.0, rel=0.02)


def test_ogi_column_density_and_colour():
    f = PlumeField(still_wind(), {"A": [0, 0, 1.0]})
    f.set_flow("A", 3.0)
    f.step(2.0)
    o = np.array([[2.0, -4.0, 1.0], [-2.0, -4.0, 1.0]])
    col = column_density(f, o, np.array([[0, 1.0, 0], [0, 1.0, 0]]))
    assert col[0] > 300 > col[1]
    rgb, mask = ogi_colour(col, 300.0, 1e5)
    assert mask.tolist() == [True, False] and rgb.shape == (2, 3)


def test_estimator_converges_from_hits_and_bearings():
    src = np.array([1.0, 1.5])
    est = SourceEstimator((-6, 6), (-1, 4))
    w = np.array([1.0, 0.0])
    rng = np.random.default_rng(0)
    for x in np.linspace(5, 1.5, 15):
        for y in (0.6, 1.5, 2.4):
            hit = abs(y - src[1]) < 0.3 + 0.4 * (x - src[0])
            est.observe([x, y], w, bool(hit))
        d = src - [x, 2.2]
        est.bearing([x, 2.2], math.atan2(d[1], d[0]) + 0.1 * rng.normal(), 0.15)
    mean, sigma, _ = est.estimate()
    assert np.linalg.norm(mean - src) < 0.5 and sigma < 1.0


def test_flow_estimate_recovers_flow_at_the_true_source():
    pl = GaussianPlume([0, 0, 1.0], 2.5, still_wind(0.0, 3.0), puff=0.0)
    pl.step(5.0)
    pts = np.array([[x, y, 0.9] for x in (1.0, 2.0, 3.0) for y in (-0.3, 0.0, 0.3)])
    ppm = pl.sample(pts)
    q = flow_estimate(list(zip(pts, ppm)), [0, 0], [1.0, 0.0], 3.0, 1.0, meander=0.0)
    assert q == pytest.approx(2.5, rel=0.05)


def test_cast_and_surge():
    cs = CastSurge()
    w = np.array([1.0, 0.0])
    v = cs.command(0.0, [0, 0], 0.05, 0.05, [0.5, 0], w)
    assert cs.mode == "SURGE" and v[0] < -0.5 * cs.cfg.surge_speed
    ys = []
    p = np.zeros(2)
    for k in range(1, 40):
        v = cs.command(k * 0.5, p, 0.0, 0.0, [0.5, 0], w)
        p = p + v * 0.5
        ys.append(p[1])
    assert cs.mode in ("CAST", "RELOCATE") and max(ys) > 0.8 and min(ys) < -0.8


def test_deck_clamp_and_route():
    q, blocked = deck.clamp([0.0, 10.0])
    assert blocked and q[1] == deck.BOUNDS[1][1]
    q, blocked = deck.clamp([5.7, 2.0])
    assert blocked and np.linalg.norm(q - [5.7, 2.0]) >= 0.85 - 1e-9
    r = deck.Route()
    p = np.asarray(deck.ROUTE[0], float)
    total = sum(np.linalg.norm(np.subtract(deck.ROUTE[k], deck.ROUTE[k - 1])) for k in range(len(deck.ROUTE)))
    p, _, lap = r.step(p, total)
    assert lap and np.allclose(p, deck.ROUTE[0])
    assert all(deck.BOUNDS[1][0] <= deck.stand_pose(g)[0][1] <= deck.BOUNDS[1][1] for g in deck.flanges())


def test_gas_bot_finds_and_fixes_a_loose_flange():
    from policy.gasbot.mock import episode
    r = episode(1003, "loose", max_t=600)
    assert r["detected"] and r["flange_ok"] and r["t_fixed"] is not None
    assert r["loc_err"] < 1.5 and "repaired" in r["alerts"]


def test_no_leak_no_alarm():
    from policy.gasbot.mock import episode
    r = episode(5001, "none", max_t=400)
    assert not r["detected"] and r["final"] == "DONE" and not r["alerts"]


def test_too_much_gas_retreats_upwind_and_alerts():
    from policy.gasbot.mock import build
    from policy.orchestrator import S
    orch, bridge, truth = build(3, "loose")
    orch.step()
    start = bridge.xy.copy()
    bridge.suite.trunk.reading = 0.3 * LEL_PPM
    orch.step()
    assert orch.state == S.GAS_RETREAT
    orch.step()
    assert orch.state == S.ESCALATE_TO_HUMAN and orch.alerts[-1]["kind"] == "gas_limit"
    assert (bridge.xy - start) @ -bridge.wind.unit(mean=True) > 1.0


def test_crack_is_escalated_not_repaired():
    from policy.gasbot.mock import episode
    r = episode(9001, "crack", max_t=600)
    assert r["final"] == "ESCALATE_TO_HUMAN" and "crack_or_burst" in r["alerts"] and r["t_fixed"] is None


def test_hud_header_carries_gas_fields():
    from policy.gasbot.mock import build
    from rig.hud import HudFeed
    orch, bridge, truth = build(1003, "loose")
    feed = HudFeed(orch)
    for _ in range(30):
        orch.step()
    g = feed.header()["gas"]
    assert {"sensors", "lel_max", "alarm", "source", "acoustic", "flow_gs", "h2s_ppm", "wind", "robot"} <= set(g)
    assert [s["id"] for s in g["sensors"]] == ["trunk_ir", "tip_cat"]


def test_solver_field_adapts_a_rate_scaled_plume():
    from policy.gasbot.plume import SolverField

    class Rel:
        mass_flow = 0.004

    class Fake:
        release, source, rate = Rel(), np.array([1.0, 0.5, 1.0]), 1.0

        def set_rate(self, r):
            self.rate = r

        def step(self, dt):
            pass

        def sample(self, pts):
            return np.full(len(pts), 100.0 * self.rate)

    f = SolverField(Fake(), "M3", still_wind())
    assert f.leaks() == [] and f.sample([[0, 0, 0]])[0] == pytest.approx(f.background)
    f.set_flow("M3", 2.0)
    f.set_flow("M1", 9.0)
    f.step(0.1)
    assert f.plume.rate == pytest.approx(0.5) and f.flow_rate() == 2.0 and len(f.leaks()) == 1
