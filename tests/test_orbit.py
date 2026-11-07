import math

import pytest

from rig.sim.orbit import Orbit, look_at_rows, orbit_eye, view_target

HERO = (
    (0.6826043253839595, 0.7307881601169449, 0.0, 0.0),
    (-0.20814756554229405, 0.19442355023181312, 0.9585791955149425, 0.0),
    (0.7005183266167461, -0.654330305081576, 0.2848261328002154, 0.0),
    (9.5, -8.5, 4.6, 1.0),
)


def close(a, b, tol=1e-6):
    return all(abs(x - y) < tol for x, y in zip(a, b, strict=True))


def test_look_at_rows_axes_and_eye():
    rows = look_at_rows((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert close(rows[0][:3], (0, 1, 0))
    assert close(rows[1][:3], (0, 0, 1))
    assert close(rows[2][:3], (1, 0, 0))
    assert close(rows[3], (1, 0, 0, 1))
    assert [r[3] for r in rows[:3]] == [0.0, 0.0, 0.0]


def test_look_at_rows_reproduces_hero_camera():
    target = view_target(HERO, z=1.0)
    rows = look_at_rows(HERO[3][:3], target)
    for got, want in zip(rows, HERO, strict=True):
        assert close(got, want, 1e-6)


def test_view_target_hits_plane_along_view_ray():
    target = view_target(HERO, z=1.0)
    assert target[2] == pytest.approx(1.0)
    fwd = [-c for c in HERO[2][:3]]
    s = (target[0] - 9.5) / fwd[0]
    assert target[1] == pytest.approx(-8.5 + s * fwd[1])


def test_orbit_from_eye_starts_at_eye_and_loops():
    eye, target = (3.0, 4.0, 2.0), (0.0, 0.0, 1.0)
    orbit = Orbit.from_eye(eye, target, period=20.0)
    assert orbit.radius == pytest.approx(5.0)
    assert close(orbit_eye(orbit, 0.0), eye)
    assert close(orbit_eye(orbit, 20.0), eye)
    assert close(orbit_eye(orbit, 10.0), (-3.0, -4.0, 2.0))


def test_orbit_quarter_turn_is_counter_clockwise():
    orbit = Orbit.from_eye((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), period=4.0)
    assert close(orbit_eye(orbit, 1.0), (math.cos(math.pi / 2), 1.0, 0.0))


def test_orbit_overrides_radius_and_height():
    orbit = Orbit.from_eye((3.0, 4.0, 2.0), (0.0, 0.0, 1.0), period=10.0, radius=10.0, z=7.0)
    assert close(orbit_eye(orbit, 0.0), (6.0, 8.0, 7.0))
