import math

import numpy as np
import pytest

pytest.importorskip("warp")

from sim.plume.release import orifice
from sim.plume.solver import Plume, Wind, domain_origin, leak_direction

REL = orifice(3.0, 20.0)


def open_field(k=0.1, u=1.0, shape=(64, 32, 32), dx=0.25, **kw):
    src = np.array([4.0, 4.0, 4.0])
    return Plume((0, 0, 0), shape, dx, src, REL, Wind(u, 0.0), diffusivity=k, buoyancy=False, jet=False,
                 static_flow=True, **kw), src


@pytest.fixture(scope="module")
def steady():
    p, src = open_field()
    for _ in range(220):
        p.step(0.1)
    return p, src


def test_source_rate_before_outflow():
    p, _ = open_field(u=0.5)
    for _ in range(20):
        s = p.step(0.1)
    assert s["mass_kg"] == pytest.approx(REL.mass_flow * 2.0, rel=1e-3)
    assert p.flow_rate()["kg_s"] == pytest.approx(REL.mass_flow, rel=1e-4)


def test_mass_conserved_in_closed_box_with_flow_and_obstacle():
    shape = (24, 20, 16)
    solid = np.zeros(shape, bool)
    solid[[0, -1]] = solid[:, [0, -1]] = solid[:, :, [0, -1]] = True
    solid[12, 4:16, 1:10] = True
    p = Plume((0, 0, 0), shape, 0.2, (1.0, 2.0, 1.0), REL, Wind(3.0, 0.0), jet_dir=(1, 0, 0), solid=solid)
    for _ in range(10):
        p.step(0.05)
    p.set_rate(0.0)
    m0 = p.stats()["mass_kg"]
    for _ in range(60):
        s = p.step(0.05)
    assert m0 > 0
    assert s["mass_kg"] == pytest.approx(m0, rel=1e-4)
    f = p.field()
    assert f.fraction[solid].max() == 0.0
    u = p.vel[0].numpy()
    assert np.abs(u[12:14, 4:16, 1:10]).max() == 0.0


def test_steady_centreline_matches_point_source_solution(steady):
    p, src = steady
    xs = np.arange(2.0, 9.0, 1.0)
    pts = np.stack([src[0] + xs, np.full_like(xs, src[1]), np.full_like(xs, src[2])], 1)
    ana = REL.volume_flow / (4 * math.pi * 0.1 * xs) * 1e6
    assert np.allclose(p.sample(pts), ana, rtol=0.1)


def test_crosswind_spread_matches_gaussian_sigma(steady):
    p, src = steady
    x = 6.0
    ys = np.linspace(-3, 3, 241)
    c = p.sample(np.stack([np.full_like(ys, src[0] + x), src[1] + ys, np.full_like(ys, src[2])], 1))
    sigma = math.sqrt(np.sum(c * ys ** 2) / np.sum(c))
    assert sigma == pytest.approx(math.sqrt(2 * 0.1 * x / 1.0), rel=0.15)


def test_methane_rises_without_wind():
    shape = (24, 24, 32)
    src = (2.4, 2.4, 1.0)
    runs = {}
    for b in (False, True):
        p = Plume((0, 0, 0), shape, 0.2, src, REL, Wind(0.0, 0.0), buoyancy=b, jet=False, diffusivity=0.05)
        for _ in range(80):
            p.step(0.05)
        f = p.field().fraction
        z = (np.arange(shape[2]) + 0.5) * 0.2
        runs[b] = float((f.sum((0, 1)) * z).sum() / f.sum())
    assert runs[True] > runs[False] + 0.15


def test_flange_helpers():
    w = Wind(5.0, 90.0)
    d = leak_direction((1.0, 0.0, 0.0), w)
    assert d[1] > 0.9 and d[2] > 0 and abs(d[0]) < 1e-9
    o = domain_origin(np.array([10.0, 0.0, 43.0]), (100, 80, 40), 0.2, w)
    assert o[0] == pytest.approx(0.0) and o[1] == pytest.approx(-4.0)
    s = Wind(5.0, 0.0, storm_level=1.0)
    speeds = [np.linalg.norm(s.at(t)) for t in np.linspace(0, 20, 50)]
    assert max(speeds) > 6 and min(speeds) < 4
    assert np.linalg.norm(Wind(5.0, 0.0).at(3.0)) == pytest.approx(5.0)


def test_set_leak_changes_release_and_rate():
    p, _ = open_field(shape=(16, 8, 8))
    p.meta.update(hole_mm=3.0, bar_g=20.0)
    out = p.set_leak(hole_mm=6.0)
    assert out["kg_s"] == pytest.approx(4 * REL.mass_flow, rel=1e-4)
    assert p.set_leak(rate=0.0)["kg_s"] == 0.0
    s0 = p.step(0.1)["mass_kg"]
    assert s0 == 0.0
