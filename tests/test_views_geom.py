import numpy as np

from rig.camera import look_at, project
from rig.views.geom import backproject, rays, scaled


def cam():
    return look_at((0.0, 0.0, 1.0), (10.0, 0.0, 1.0), focal=10.0, w=64, h=48)


def test_rays_centre_points_forward():
    o, d = rays(cam())
    assert d.shape == (48, 64, 3)
    c = d[24, 32] / np.linalg.norm(d[24, 32])
    assert np.allclose(o, (0, 0, 1))
    assert c[0] > 0.99


def test_backproject_roundtrip_matches_project():
    c = cam()
    depth = np.full((48, 64), 5.0, np.float32)
    depth[:4] = np.inf
    pts = backproject(depth, c)
    assert len(pts) == 44 * 64
    assert np.allclose(pts[:, 0], 5.0)
    u, v, z = project(c, pts[100])
    assert abs(z - 5.0) < 1e-6
    assert 0 <= u < 64 and 0 <= v < 48


def test_backproject_stride_and_range():
    depth = np.full((48, 64), 50.0, np.float32)
    assert len(backproject(depth, cam(), stride=4, max_range=30.0)) == 0
    depth[:] = 3.0
    assert len(backproject(depth, cam(), stride=4)) == 12 * 16


def test_scaled_header_keeps_pose():
    c = cam()
    s = scaled(c, 16, 12)
    assert s["w"] == 16 and s["h"] == 12
    assert np.allclose(s["T"], c["T"])
    assert abs(s["K"][0][0] - c["K"][0][0] / 4) < 1e-6
