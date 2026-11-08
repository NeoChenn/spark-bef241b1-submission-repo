import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy.night_look import aim_matrix, grade, lamps, safety_lights, srgb_to_linear


def test_aim_matrix_points_minus_z_at_target():
    m = aim_matrix((1.0, 2.0, 3.0), (0.0, 0.0, 1.0))
    r, t = m[:3, :3], m[3, :3]
    assert np.allclose(t, (1, 2, 3)) and np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.allclose(-r[2], np.array([-1, -2, -2]) / 3.0)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_aim_matrix_straight_down():
    m = aim_matrix((0, 0, 5), (0, 0, 0))
    assert np.allclose(-m[2, :3], (0, 0, -1))


def test_srgb_to_linear_coral():
    assert np.allclose(srgb_to_linear([217 / 255, 119 / 255, 87 / 255]), (0.693, 0.184, 0.0953), atol=2e-3)


def test_grade_vignette_darkens_corners_and_lifts_black():
    img = np.full((90, 160, 3), 128, np.uint8)
    out = grade(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out[45, 80].mean() > out[0, 0].mean()
    black = grade(np.zeros((10, 10, 3), np.uint8), vignette=0.0)
    assert tuple(black[5, 5]) == (2, 7, 9)


def test_lamps_track_work_point_and_safety_on_far_rails():
    ls = lamps((-0.5, 0.0, 1.2))
    assert {l["name"] for l in ls} >= {"flood_back", "flood_key", "led_rim"}
    assert all(l["pos"][2] > 1.5 for l in ls)
    s = safety_lights()
    assert {c for c, _ in s} >= {"red", "green", "amber"}
    assert all(abs(p[0]) > 7 or abs(p[1]) > 4 for _, p in s)
