import numpy as np

from rig.camera import look_at
from rig.views.raycast import cast_numpy, fan, depth_from_hits, ray_grid


def test_fan_triangulates_quads_and_tris():
    tris = fan([4, 3], [0, 1, 2, 3, 4, 5, 6])
    assert tris.tolist() == [[0, 1, 2], [0, 2, 3], [4, 5, 6]]


def wall():
    v = np.array([[5.0, -3, -3], [5.0, 3, -3], [5.0, 3, 3], [5.0, -3, 3]])
    return v, fan([4], [0, 1, 2, 3])


def test_numpy_cast_hits_wall_and_depth_is_z():
    cam = look_at((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), focal=10.0, w=32, h=24)
    o, d = ray_grid(cam, 32, 24)
    v, t = wall()
    hit = cast_numpy(o, d.reshape(-1, 3), v, t, 50.0)
    depth = depth_from_hits(hit, d, cam)
    assert depth.shape == (24, 32)
    assert np.allclose(depth[12, 16], 5.0, atol=1e-3)
    assert 0.3 < np.isfinite(depth).mean() < 0.6


def test_numpy_cast_miss_is_inf():
    cam = look_at((0.0, 0.0, 0.0), (-10.0, 0.0, 0.0), focal=10.0, w=8, h=6)
    o, d = ray_grid(cam, 8, 6)
    v, t = wall()
    assert np.isinf(depth_from_hits(cast_numpy(o, d.reshape(-1, 3), v, t, 50.0), d, cam)).all()
