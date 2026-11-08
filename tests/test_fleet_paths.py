import math

import numpy as np
import pytest

from integration.fleet_nav import ARRIVED, REPORT, PROFILES, FleetNav, drive, nav, run_leg, run_route, trot_q
from integration.fleet_paths import (DOCK, F5_WORK, G_F5, HOLD, RETREAT, ROUTES, SCRUB, DeckMap, leg, legs,
                                     safe_pose)
from patrol import features


def _pose_err(r, goal):
    x, y, yaw = goal
    return math.hypot(r.xy[-1, 0] - x, r.xy[-1, 1] - y), abs((r.yaw[-1] - yaw + math.pi) % (2 * math.pi) - math.pi)


def test_every_leg_is_clear_of_the_deck_and_gently_cornered():
    deck = DeckMap.rig_merged()
    for name, lg in legs().items():
        p = lg.path
        h = np.arctan2(*np.diff(p, axis=0).T[::-1])
        assert all(deck.clearance(x, y, a) > 0.04 for (x, y), a in zip(p[:-1], h)), name
        turn = np.abs((np.diff(h) + math.pi) % (2 * math.pi) - math.pi)
        assert turn.max() < 0.45, name


def test_dock_is_eight_to_twelve_metres_of_walkway_from_the_f5_work_pose():
    assert 8.0 <= leg("unit2.dispatch").length <= 12.0
    assert np.allclose(leg("unit2.dispatch").goal, F5_WORK[:2])
    assert np.allclose(leg("unit2.home").goal, DOCK["unit2"][:2])


def test_unit2_reaches_f5_and_returns_to_the_dock_without_collision():
    rs = run_route("unit2", ROUTES["unit2"])
    assert all(r.arrived and r.collided is None for r in rs)
    d, a = _pose_err(rs[0], F5_WORK)
    assert d < 0.06 and a < 0.12
    assert min(math.hypot(x - 31.198, y + 10.857) for x, y in rs[1].xy) < 2.0
    d, a = _pose_err(rs[2], DOCK["unit2"])
    assert d < 0.06 and a < 0.12


def test_unit1_steps_back_more_than_six_metres_from_g_f5_and_returns():
    rs = run_route("unit1", ROUTES["unit1"])
    assert all(r.arrived and r.collided is None for r in rs)
    assert math.dist(rs[0].xy[-1], G_F5[:2]) > 6.0
    assert _pose_err(rs[1], SCRUB)[0] < 0.06


def test_a_push_mid_walk_is_recovered():
    hooks = [(2.0, lambda base, deck, n: base.push(0.0, 0.7))]
    r = run_leg("unit2", "unit2.dispatch", hooks=hooks)
    kinds = [e["type"] for e in r.events]
    assert "off_path" in kinds and "recovered" in kinds[kinds.index("off_path"):]
    assert r.arrived and r.collided is None


def test_an_object_on_the_walkway_stops_the_robot_and_is_reported():
    reports = []
    hooks = [(0.5, lambda base, deck, n: deck.add_box("crate", 23.85, 24.15, -9.83, -9.53, 0.0, 0.4)),
             (18.0, lambda base, deck, n: deck.remove("crate"))]
    r = run_leg("unit2", "unit2.dispatch", hooks=hooks, on_event=reports.append)
    rep = [e for e in reports if e["type"] == "object_on_walkway"]
    assert len(rep) == 1 and abs(rep[0]["xy"][0] - 24.0) < 0.3
    stopped = r.xy[[i for i, s in enumerate(r.states) if s == REPORT]]
    assert len(stopped) > 20 and np.ptp(stopped[:, 0]) < 0.25
    assert stopped[:, 0].min() - 24.15 >= 0.9
    assert "cleared" in [e["type"] for e in reports]
    assert r.arrived and r.collided is None


def test_an_object_left_in_place_holds_until_argon_resumes():
    deck = DeckMap.rig_merged()
    deck.add_box("crate", 23.85, 24.15, -9.83, -9.53, 0.0, 0.4)
    r = run_leg("unit2", "unit2.dispatch", deck=deck, t_max=20.0)
    assert r.states[-1] == REPORT and not r.arrived and r.collided is None


def test_safe_pose_keeps_the_callers_shape():
    p3 = safe_pose("unit1", "hold", (18.2567, -5.25, 42.82))
    assert p3.shape == (3,) and p3[2] == 42.82 and np.allclose(p3[:2], HOLD[:2])
    p4 = safe_pose("unit1", "retreat", (18.2567, -5.25, 42.82, 0.0))
    assert p4.shape == (4,) and np.allclose(p4[[0, 1, 3]], RETREAT)
    assert math.dist(RETREAT[:2], G_F5[:2]) > math.dist(HOLD[:2], G_F5[:2]) > 6.0
    assert RETREAT[1] > HOLD[1] > SCRUB[1]


def test_drive_wraps_the_patrol_controller_per_unit():
    deck = DeckMap.rig_merged()
    nav("unit2", deck=deck)
    angles = PROFILES["unit2"].angles()
    x, y, yaw = DOCK["unit2"]
    vx, wz = drive("unit2", (deck.raycast(x, y, yaw, angles, 0.3), angles), (x, y, yaw), 0.0, 0.0, "unit2.dispatch")
    assert 0.0 < vx <= PROFILES["unit2"].v_max and abs(wz) <= PROFILES["unit2"].w_max


def test_lidar_raycast_matches_the_interface_angle_convention():
    deck = DeckMap.rig_merged(extra=[("box", 30.9, 31.1, -9.7, -9.6, 0.0, 0.5)])
    angles = PROFILES["unit2"].angles()
    ahead = features.pool_sectors_np(deck.raycast(30.0, -9.7, 0.0, angles, 0.3), angles)
    left = features.pool_sectors_np(deck.raycast(30.0, -9.7, -math.pi / 2, angles, 0.3), angles)
    assert ahead.argmin() == 18 and abs(ahead[18] - 0.9) < 0.02
    assert left.argmin() == 27


def test_trot_is_still_at_rest_and_swings_when_walking():
    assert trot_q(1.0, 0.0)["FL_thigh_joint"] == pytest.approx(0.8)
    qs = [trot_q(g, 0.7) for g in np.linspace(0, 0.36, 12)]
    assert np.ptp([q["FL_thigh_joint"] for q in qs]) > 0.3
    assert qs[3]["FL_thigh_joint"] - 0.8 == pytest.approx(-(qs[3]["FR_thigh_joint"] - 0.8), abs=1e-9)


def test_nav_walker_drives_a_walker_goal_on_the_deck_map():
    from integration.fleet_nav import NavWalker
    w = NavWalker("unit2", DOCK["unit2"][:2], DOCK["unit2"][2])
    w.go(F5_WORK[:2], F5_WORK[2])
    steps = 0
    while not w.step(0.25) and steps < 200:
        steps += 1
    assert w.state == ARRIVED and math.dist(w.xy, F5_WORK[:2]) < 0.06
    assert abs((w.yaw - F5_WORK[2] + math.pi) % (2 * math.pi) - math.pi) < 0.12
    assert not any(e["type"] == "collision" for e in w.events)
    w.go(DOCK["unit2"][:2])
    while not w.step(0.25) and steps < 400:
        steps += 1
    assert math.dist(w.xy, DOCK["unit2"][:2]) < 0.06


def test_f5_goal_is_the_retorque_work_pose():
    from integration.fleet_paths import F5, F5_NORMAL
    x = F5[0] + 0.6 * F5_NORMAL[0]
    y = F5[1] + 0.6 * F5_NORMAL[1]
    assert np.allclose(F5_WORK, (x, y, math.atan2(-F5_NORMAL[1], -F5_NORMAL[0])))


@pytest.mark.parametrize("beams", [72, 360])
def test_static_map_never_reads_as_an_object_on_a_straight_dock_to_f5_drive(beams):
    from integration.fleet_nav import BaseSim, _NAVS
    from integration.fleet_paths import polyline
    _NAVS.pop("unit2", None)
    deck = DeckMap.rig_merged()
    angles = np.linspace(-math.pi, math.pi, beams, endpoint=False)
    base = BaseSim(*DOCK["unit2"])
    path = polyline(DOCK["unit2"], F5_WORK)
    t = 0.0
    while t < 25.0 and (t == 0.0 or nav("unit2").state != ARRIVED):
        scan = (deck.raycast(base.x, base.y, base.yaw, angles, 0.3), angles)
        vx, wz = drive("unit2", scan, base.pose, base.v, base.w, path, t=t)
        base.step(vx, wz)
        assert deck.clearance(*base.pose) > 0.0
        t += 0.1
    kinds = [e["type"] for e in nav("unit2").events]
    assert "object_on_walkway" not in kinds and "blocked" not in kinds
    assert nav("unit2").state == ARRIVED and math.dist(base.pose[:2], F5_WORK[:2]) < 0.06
