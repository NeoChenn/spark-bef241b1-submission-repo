import numpy as np
import pytest

from sim.ocean_legs import cluster_circles, waterline_points


def cylinder(cx: float, cy: float, r: float, z0: float, z1: float, n: int = 24):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
    pts = np.concatenate([np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z1)]])
    j = np.arange(n)
    quads = np.stack([j, (j + 1) % n, (j + 1) % n + n, j + n], 1).ravel()
    return pts, np.full(n, 4), quads


def merged(*parts):
    pts, counts, idx, off = [], [], [], 0
    for p, c, i in parts:
        pts.append(p), counts.append(c), idx.append(i + off)
        off += len(p)
    return np.concatenate(pts), np.concatenate(counts), np.concatenate(idx)


def test_finds_each_leg_of_one_merged_mesh():
    pts, counts, idx = merged(cylinder(-20, -10, 3.0, -50, 20), cylinder(25, 15, 2.0, -50, 20), cylinder(0, 0, 5, 10, 20))
    legs = sorted(cluster_circles(waterline_points(pts, counts, idx, 0.0)))
    assert len(legs) == 2
    assert legs[0] == pytest.approx((-20, -10, 3.0), abs=0.05)
    assert legs[1] == pytest.approx((25, 15, 2.0), abs=0.05)


def test_no_crossing_no_legs():
    pts, counts, idx = cylinder(0, 0, 1, 5, 10)
    assert cluster_circles(waterline_points(pts, counts, idx, 0.0)) == []
