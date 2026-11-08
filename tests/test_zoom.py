import math

import numpy as np
import pytest

from rig.sim.zoom import ZoomPath, ease, interp, spherical

TARGET = (18.26, -5.65, 43.72)
CENTER = (4.0, -4.0, 30.0)


def path(**kw):
    return ZoomPath(wide_target=CENTER, close_target=TARGET, **kw)


def dist(view):
    return float(np.linalg.norm(np.subtract(view[0], view[1])))


def test_ease_endpoints_and_flat_ends():
    assert ease(0.0) == 0.0 and ease(1.0) == 1.0 and ease(0.5) == pytest.approx(0.5)
    assert ease(-1.0) == 0.0 and ease(2.0) == 1.0
    assert ease(0.01) < 0.001 and 1 - ease(0.99) < 0.001


def test_interp_clamps():
    assert interp(0.5, (0, 1), (10, 20)) == 15
    assert interp(-1, (0, 1), (10, 20)) == 10
    assert interp(5, (0, 1), (10, 20)) == 20


def test_spherical_distance_and_direction():
    assert np.allclose(spherical((1.0, 2.0, 3.0), 10.0, 90.0, 0.0), (1.0, 12.0, 3.0))
    assert np.allclose(spherical((0.0, 0.0, 0.0), 2.0, 0.0, 30.0), (math.sqrt(3), 0.0, 1.0))


def test_wide_start_and_close_end():
    p = path()
    view = p.at(0.0)
    assert np.allclose(view[1], CENTER)
    assert dist(view) == pytest.approx(p.wide_dist)
    view = p.at(p.orbit_s + p.descend_s)
    assert np.allclose(view[1], TARGET)
    assert dist(view) == pytest.approx(p.close_dist)
    assert view[2] == pytest.approx(p.close_focal)


def test_descent_takes_descend_seconds_and_is_monotone():
    p = path()
    d = [dist(p.at(t)) for t in np.linspace(p.orbit_s, p.orbit_s + p.descend_s, 200)]
    assert all(b <= a + 1e-9 for a, b in zip(d, d[1:]))
    assert d[0] == pytest.approx(p.wide_dist) and d[-1] == pytest.approx(p.close_dist)


def test_path_is_continuous():
    p = path()
    ts = np.arange(0.0, p.period * 2, 1 / 60)
    eyes = np.array([p.at(t)[0] for t in ts])
    steps = np.linalg.norm(np.diff(eyes, axis=0), axis=1)
    d = np.array([dist(p.at(t)) for t in ts[1:]])
    assert np.all(steps < 0.2 * d + 0.05)


def test_loops_with_period():
    p = path()
    assert p.period == p.orbit_s + 2 * p.descend_s + p.hold_s
    a, b = p.at(1.3), p.at(1.3 + p.period)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_hold_forever_stays_at_close():
    p = path(hold_s=math.inf)
    view = p.at(1000.0)
    assert np.allclose(view[1], TARGET)
    assert dist(view) == pytest.approx(p.close_dist)


def test_close_approach_comes_in_low():
    p = path()
    for t in np.linspace(p.orbit_s, p.orbit_s + p.descend_s, 100):
        view = p.at(t)
        if dist(view) < 8.0:
            assert view[0][2] - TARGET[2] < 2.2
