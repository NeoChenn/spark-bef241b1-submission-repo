import math

import numpy as np
import pytest

from sim.plume.colormap import colorbar, lel_alpha, lel_rgb
from sim.plume.release import P_ATM, Gas, critical_ratio, orifice


def test_methane_choked_3mm_20bar():
    r = orifice(3.0, 20.0)
    assert r.choked
    g = r.gas
    a = 0.62 * math.pi * 0.003 ** 2 / 4
    p0 = 20e5 + P_ATM
    expect = a * p0 * math.sqrt(g.gamma * g.molar_mass / (8.314462618 * 288.15) * (2 / (g.gamma + 1)) ** ((g.gamma + 1) / (g.gamma - 1)))
    assert r.mass_flow == pytest.approx(expect, rel=1e-9)
    assert r.mass_flow == pytest.approx(0.0159, rel=0.01)
    assert r.sm3_h == pytest.approx(84.5, rel=0.01)
    assert r.nm3_h < r.sm3_h
    assert 380 < r.exit_velocity < 440
    assert r.momentum > r.mass_flow * r.exit_velocity


def test_choke_boundary_is_continuous():
    g = Gas.mix()
    p_crit = P_ATM / critical_ratio(g.gamma)
    bar = (p_crit - P_ATM) / 1e5
    lo, hi = orifice(5.0, bar * 0.999), orifice(5.0, bar * 1.001)
    assert not lo.choked and hi.choked
    assert lo.mass_flow == pytest.approx(hi.mass_flow, rel=5e-3)


def test_subsonic_low_pressure_and_scaling():
    r = orifice(4.0, 0.3)
    assert not r.choked
    assert r.exit_velocity < 340
    assert orifice(0.0, 10).mass_flow == 0.0
    assert orifice(6.0, 20).mass_flow == pytest.approx(4 * orifice(3.0, 20).mass_flow, rel=1e-9)
    flows = [orifice(3.0, b).mass_flow for b in (0.2, 0.5, 1, 5, 20, 80)]
    assert np.all(np.diff(flows) > 0)


def test_h2s_mixture_is_heavier_and_le_chatelier_lel():
    pure, sour = Gas.mix(0.0), Gas.mix(0.1)
    assert sour.molar_mass > pure.molar_mass
    assert sour.buoyancy < pure.buoyancy
    assert pure.buoyancy == pytest.approx(1 - 16.043 / 28.9647, rel=1e-3)
    assert sour.lel == pytest.approx(1 / (0.9 / 0.05 + 0.1 / 0.04), rel=1e-9)
    assert orifice(3, 20, h2s=0.1).mass_flow > orifice(3, 20).mass_flow


def test_colormap_ends_and_alpha():
    assert np.allclose(lel_rgb(0) * 255, [38, 84, 214])
    assert np.allclose(lel_rgb(150) * 255, [235, 45, 40])
    assert lel_alpha(0.5) == 0
    assert 0 < lel_alpha(10) < lel_alpha(60) <= 0.55
    bar = colorbar(8, 50)
    assert bar.shape == (50, 8, 3) and bar[0, 0, 0] > bar[-1, 0, 0]
