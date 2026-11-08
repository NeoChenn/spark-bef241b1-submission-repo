import math

import numpy as np
import pytest

from policy.pickcan.detect import detect_cans, noisy_pose

W, H, F = 640, 480, 600.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
CAM_Z = 1.2
R = 0.033
GREY = (95, 98, 100)
RED = (191, 20, 15)
YELLOW = (242, 191, 25)
BLUE = (25, 76, 217)


def down_camera(x=0.0, y=0.0, z=CAM_Z):
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    T[:3, 3] = (x, y, z)
    return T


def scene():
    rgb = np.empty((H, W, 3), np.uint8)
    rgb[:] = GREY
    rng = np.random.default_rng(0)
    rgb = np.clip(rgb.astype(int) + rng.integers(-6, 7, rgb.shape), 0, 255).astype(np.uint8)
    return rgb, np.full((H, W), CAM_Z, np.float32)


def pix(T, p):
    q = np.linalg.inv(T) @ np.array([*p, 1.0])
    return K[0, 0] * q[0] / q[2] + K[0, 2], K[1, 1] * q[1] / q[2] + K[1, 2]


def lay_can(rgb, depth, T, cx, cy, yaw, length=0.12, diameter=2 * R, colour=RED):
    vs, us = np.mgrid[0:H, 0:W]
    z = CAM_Z - diameter
    x = T[0, 3] + (us - K[0, 2]) * z / F
    y = T[1, 3] - (vs - K[1, 2]) * z / F
    dx, dy = x - cx, y - cy
    a = dx * math.cos(yaw) + dy * math.sin(yaw)
    b = -dx * math.sin(yaw) + dy * math.cos(yaw)
    m = (np.abs(a) <= length / 2) & (np.abs(b) <= diameter / 2)
    rgb[m] = colour
    depth[m] = z
    return m


def stand_can(rgb, depth, T, cx, cy, height=0.12, diameter=2 * R, colour=YELLOW):
    vs, us = np.mgrid[0:H, 0:W]
    z = CAM_Z - height
    x = T[0, 3] + (us - K[0, 2]) * z / F
    y = T[1, 3] - (vs - K[1, 2]) * z / F
    m = np.hypot(x - cx, y - cy) <= diameter / 2
    rgb[m] = colour
    depth[m] = z
    return m


def yaw_err(a, b):
    d = (a - b) % math.pi
    return min(d, math.pi - d)


@pytest.mark.parametrize("yaw", [0.0, 0.5, 1.2, -0.9, 2.6])
def test_lying_red_can_pose_from_depth(yaw):
    rgb, depth = scene()
    T = down_camera(0.1, -0.05)
    lay_can(rgb, depth, T, 0.25, 0.1, yaw)
    out = detect_cans(rgb, depth, K, T)
    assert len(out) == 1
    c = out[0]
    assert np.linalg.norm(np.array(c["xyz"]) - (0.25, 0.1, R)) < 0.02
    assert yaw_err(c["yaw"], yaw) < 0.1
    assert c["lying"] and c["colour"] == "red"
    assert c["length"] == pytest.approx(0.12, abs=0.015) and c["diameter"] == pytest.approx(2 * R, abs=0.01)
    assert 0.5 < c["score"] <= 1.0
    x0, y0, x1, y1 = c["bbox"]
    u, v = pix(T, (0.25, 0.1, 2 * R))
    assert x0 <= u <= x1 and y0 <= v <= y1


def test_upright_can_and_several_colours():
    rgb, depth = scene()
    T = down_camera()
    stand_can(rgb, depth, T, -0.2, 0.15)
    lay_can(rgb, depth, T, 0.2, -0.1, 0.3, colour=BLUE)
    out = sorted(detect_cans(rgb, depth, K, T), key=lambda c: c["xyz"][0])
    assert [c["colour"] for c in out] == ["yellow", "blue"]
    up, lie = out
    assert not up["lying"] and lie["lying"]
    assert np.linalg.norm(np.array(up["xyz"]) - (-0.2, 0.15, 0.06)) < 0.02
    assert up["length"] == pytest.approx(0.12, abs=0.02)
    assert np.linalg.norm(np.array(lie["xyz"]) - (0.2, -0.1, R)) < 0.02


def test_without_depth_uses_deck_plane():
    rgb, depth = scene()
    T = down_camera()
    lay_can(rgb, depth, T, 0.0, 0.2, 0.7)
    c = detect_cans(rgb, None, K, T)[0]
    assert np.linalg.norm(np.array(c["xyz"][:2]) - (0.0, 0.2)) < 0.02 and yaw_err(c["yaw"], 0.7) < 0.1 and c["lying"]


def test_ignores_grey_deck_small_specks_and_long_stripes():
    rgb, depth = scene()
    rgb[100:104, 100:104] = RED
    rgb[300:310, 20:620] = YELLOW
    assert detect_cans(rgb, depth, K, down_camera()) == []


def test_noisy_pose_is_seeded_and_small():
    rng = np.random.default_rng(3)
    xyz, yaw = noisy_pose([1.0, 2.0, 0.03], 0.4, rng)
    again = noisy_pose([1.0, 2.0, 0.03], 0.4, np.random.default_rng(3))
    assert np.allclose(xyz, again[0]) and yaw == again[1]
    assert xyz[2] == 0.03 and np.linalg.norm(np.array(xyz[:2]) - (1.0, 2.0)) < 0.05 and abs(yaw - 0.4) < 0.4
    errs = [noisy_pose([0, 0, 0], 0.0, rng)[0][0] for _ in range(2000)]
    assert np.std(errs) == pytest.approx(0.01, rel=0.1)
