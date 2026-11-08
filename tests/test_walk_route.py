import math

import numpy as np
import pytest

from policy.walk.route import (WaypointFollower, falls_per_km, polyline_length, resample, route_to_goal,
                               simulate, stance_from_meta, wrap)

SQUARE = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0)]


def test_wrap_maps_to_pi_range():
    assert wrap(3 * math.pi) == pytest.approx(math.pi) or wrap(3 * math.pi) == pytest.approx(-math.pi)
    assert wrap(-0.5) == pytest.approx(-0.5)
    assert wrap(2 * math.pi + 0.25) == pytest.approx(0.25)


def test_polyline_length():
    assert polyline_length(SQUARE) == pytest.approx(7.0)
    assert polyline_length([(1, 1)]) == 0.0


def test_resample_spacing_and_endpoints():
    pts = resample(SQUARE, 0.5)
    assert np.allclose(pts[0], SQUARE[0]) and np.allclose(pts[-1], SQUARE[-1])
    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert len(pts) == 15
    assert np.all(gaps <= 0.5 + 1e-9) and np.all(gaps > 0.25)


def test_resample_drops_3d_and_duplicates():
    pts = resample([(0, 0, 42.5), (0, 0, 42.5), (2, 0, 42.5)], 1.0)
    assert pts.shape == (3, 2)


def test_route_to_goal_cuts_at_closest_point():
    path = route_to_goal(SQUARE, (2.0, -0.6))
    assert np.allclose(path[0], (0, 0))
    assert np.allclose(path[-2], (2.0, 0.0))
    assert np.allclose(path[-1], (2.0, -0.6))
    assert len(path) == 3


def test_route_to_goal_takes_first_pass_on_revisited_segment():
    route = [(0, 0), (10, 0), (10, 2), (0, 2), (0, 0), (10, 0)]
    path = route_to_goal(route, (5.0, -0.5))
    assert polyline_length(path) == pytest.approx(5.0 + 0.5)


def test_stance_from_meta_world():
    meta = {"world": {"robot_spawn": {"position_m": [18.2, -5.25, 42.82], "yaw_rad": 0.0}}}
    assert stance_from_meta(meta) == pytest.approx((18.2, -5.25, 0.0))


def test_stance_from_meta_local_deck():
    meta = {"main_pipe": {"y_m": 0.0}, "damage": {"position_m": [-0.49, 0.095, 1.22]}}
    x, y, yaw = stance_from_meta(meta, standoff=0.5)
    assert (x, y, yaw) == pytest.approx((-0.49, 0.5, 0.0))


def test_follower_heads_toward_lookahead():
    f = WaypointFollower(SQUARE, goal_yaw=math.pi / 2, v_max=1.0)
    vx, vy, wz, done = f.command(0.0, 0.0, 0.0)
    assert vx > 0.5 and abs(wz) < 0.1 and not done
    vx, vy, wz, done = f.command(0.0, 0.0, math.pi / 2)
    assert wz < -0.5 and vx < 0.3


def test_follower_respects_limits():
    f = WaypointFollower(SQUARE, goal_yaw=0.0, v_max=0.8, w_max=0.7)
    for pose in [(0, 0, 0), (1, 0.5, 2.0), (3.9, 2.0, -2.5)]:
        vx, vy, wz, _ = f.command(*pose)
        assert math.hypot(vx, vy) <= 0.8 + 1e-9 and abs(wz) <= 0.7 + 1e-9


def test_follower_progress_is_monotonic():
    f = WaypointFollower([(0, 0), (10, 0), (10, 2), (0, 2)], goal_yaw=0.0)
    f.command(9.0, 0.0, 0.0)
    s = f.progress
    f.command(1.0, 2.0, 0.0)
    assert f.progress >= s
    f2 = WaypointFollower([(0, 0), (10, 0), (10, 2), (0, 2)], goal_yaw=0.0)
    f2.command(1.0, 0.3, 0.0)
    assert f2.progress < 2.0


def test_follower_final_approach_is_holonomic_and_stops():
    f = WaypointFollower(SQUARE, goal_yaw=0.0)
    f.progress = polyline_length(SQUARE)
    vx, vy, wz, done = f.command(4.0, 2.8, 0.0)
    assert vy > 0 and abs(vx) < 0.05 and not done
    _, _, _, done = f.command(4.0, 3.0, 0.0)
    assert done


def test_simulated_unicycle_reaches_goal():
    route = [(-21.0, 20.5), (-10.0, 20.5), (-10.0, 11.0), (-1.0, 11.0), (-1.0, -4.5), (24.0, -4.5)]
    path = route_to_goal(route, (18.26, -5.25))
    f = WaypointFollower(path, goal_yaw=0.0, v_max=1.0)
    traj, t = simulate(f, (-21.0, 20.5, 0.0), dt=0.02, t_max=200.0)
    x, y, yaw = traj[-1]
    assert f.done
    assert math.hypot(x - 18.26, y + 5.25) < 0.05 and abs(wrap(yaw)) < 0.05
    assert t < 1.4 * polyline_length(path) / 1.0
    lat = [np.min(np.linalg.norm(resample(path, 0.05) - p[:2], axis=1)) for p in traj]
    assert max(lat) < 0.6


def test_falls_per_km():
    assert falls_per_km(3, 1500.0) == pytest.approx(2.0)
    assert falls_per_km(0, 0.0) == 0.0


def test_final_approach_keeps_minimum_speed():
    f = WaypointFollower(SQUARE, goal_yaw=0.0, pos_tol=0.03, v_min=0.15)
    f.progress = polyline_length(SQUARE)
    vx, vy, _, done = f.command(4.0, 2.94, 0.0)
    assert not done and math.hypot(vx, vy) == pytest.approx(0.15)


def test_follower_reengages_when_pushed_off_stance():
    f = WaypointFollower(SQUARE, goal_yaw=0.0, pos_tol=0.05, yaw_tol=0.05, hold_tol=0.08, hold_yaw=0.1)
    f.progress = polyline_length(SQUARE)
    assert f.command(4.0, 3.0, 0.0)[3]
    assert f.command(4.0, 2.95, 0.05)[3]
    vx, vy, wz, done = f.command(4.0, 2.85, 0.0)
    assert not done and vy > 0
    _, _, wz, done = f.command(4.0, 3.0, 0.3)
    assert not done and wz < 0
