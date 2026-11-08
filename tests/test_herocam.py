import math

import pytest

from rig.sim.herocam import DECK_AT, HeroCam, deck_of, fixed_shot


def down_angle(eye, look):
    d = [l - e for e, l in zip(eye, look)]
    return math.degrees(math.atan2(-d[2], math.hypot(d[0], d[1])))


def test_deck_of():
    assert deck_of(42.5) == 42.5
    assert deck_of(31.3) == 31.25
    assert deck_of(5.0) == 20.0


def test_default_is_flyby():
    h = HeroCam()
    assert h.meta()["mode"] == "flyby" and h.at(0.0) is None


@pytest.mark.parametrize("sid", sorted(DECK_AT))
def test_scenario_starts_in_fixed_deck_view(sid):
    h = HeroCam()
    assert h.event("scenario", {"id": sid})
    eye, look, focal = h.at(0.0)
    at = DECK_AT[sid]
    assert math.dist(look[:2], at[:2]) < 1e-6
    assert 15.0 <= math.dist(eye, look) <= 25.0
    assert 25.0 <= down_angle(eye, look) <= 35.0
    assert h.at(5.0) == (eye, look, focal)


@pytest.mark.parametrize("view,at", [("rust", (18.75, -5.75, 43.6)), ("can", (11.0, -6.0, 42.5)),
                                     ("flange", (17.0, -9.0, 42.5)), ("weld", (10.0, -8.0, 42.5))])
def test_zoom_looks_at_target_close(view, at):
    h = HeroCam()
    h.event("scenario", {"id": "repair"})
    assert h.event("camera", {"mode": "zoom", "at": list(at), "view": view})
    eye, look, _ = h.at(0.0)
    assert math.dist(look[:2], at[:2]) < 1e-6 and 2.0 <= math.dist(eye, look) <= 5.0
    assert h.meta() == {"mode": "zoom", "robot": None, "view": view}


def test_zoom_without_target_is_ignored_and_select_never_moves():
    h = HeroCam()
    h.event("scenario", {"id": "gas"})
    shot = h.at(0.0)
    assert not h.event("camera", {"mode": "zoom", "view": "rust"})
    assert not h.event("select", {"robot": "R2"})
    assert h.at(1.0) == shot


def test_rig_and_deck_toggle_and_obstacle():
    h = HeroCam()
    h.event("scenario", {"id": "patrol"})
    assert h.event("camera", {"mode": "rig"}) and h.at(0.0) is None
    assert h.event("camera", {"mode": "deck"}) and h.at(0.0) is not None
    assert h.event("camera", {"mode": "obstacle", "at": [5.0, 1.0, 42.5]})
    eye, look, _ = h.at(0.0)
    assert math.dist(look[:2], (5.0, 1.0)) < 1e-6
    assert h.at(60.0) == (eye, look, _)


def test_blocked_view_turns_or_pulls_in():
    look_at = (0.0, 0.0, 42.5)
    first = fixed_shot(look_at, "deck")
    side = fixed_shot(look_at, "deck", lambda a, b: 3.0 if math.dist(b, first[0]) < 1e-6 else None)
    assert math.dist(side[0], first[0]) > 1.0 and abs(math.dist(side[0], side[1]) - 20.0) < 1e-6
    inside = fixed_shot(look_at, "deck", lambda a, b: 6.0)
    assert abs(math.dist(inside[0], inside[1]) - 5.5) < 1e-6


def test_lower_deck_stays_under_ceiling():
    eye, _, _ = fixed_shot((0.0, 0.0, 31.25), "deck")
    assert eye[2] <= 42.5 - 1.0 + 1e-6


def test_first_blocker_combines_and_caches_mesh_per_cell():
    from rig.sim.herocam import first_hit, mesh_ray

    assert first_hit(lambda a, b: None, lambda a, b: 3.0)((0, 0, 0), (9, 0, 0)) == 3.0
    assert first_hit(lambda a, b: 2.0, lambda a, b: 3.0)((0, 0, 0), (9, 0, 0)) == 2.0
    assert first_hit(None, lambda a, b: None)((0, 0, 0), (9, 0, 0)) is None
    built = []

    class Caster:
        def cast(self, o, dirs, max_t):
            return [4.0 if dirs[0][0] > 0 else -1.0]

    def build(center):
        built.append(tuple(center))
        return Caster()

    ray = mesh_ray(build, cell=20.0)
    assert ray((0, 0, 30), (10, 0, 30)) == 4.0
    assert ray((1, 1, 30), (-10, 0, 30)) is None
    assert ray((2, 0, 30), (2, 0, 30)) is None
    assert len(built) == 1
    ray((80, 0, 30), (90, 0, 30))
    assert len(built) == 2
