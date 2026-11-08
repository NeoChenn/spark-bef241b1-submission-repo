import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from patrol.detect import DetectCfg, PathRef, Tracker, detect_frame  # noqa: E402
from patrol.sim2d import PatrolSim, SimCfg  # noqa: E402

BEAMS = 360
ANG = -math.pi + (np.arange(BEAMS) + 0.5) * (2 * math.pi / BEAMS)


def corridor(objects=(), half_width=0.6):
    """Straight corridor along +x from (0, 0), robot at x=2 facing +x. objects: (x, y, r)."""
    sim = PatrolSim(SimCfg(num_envs=1, seed=0, n_obstacles=max(len(objects), 1)))
    M = sim.M
    x = torch.arange(M, dtype=torch.float32) * sim.cfg.ds
    sim.P[0] = torch.stack([x, torch.zeros(M)], -1)
    sim.h[0] = 0.0
    sim.nrm[0] = torch.tensor([0.0, 1.0])
    sim.wl[0] = half_width
    sim.wr[0] = half_width
    sim.obs_on[0] = False
    for k, (ox, oy, r) in enumerate(objects):
        sim.obs_c[0, k] = torch.tensor([ox, oy])
        sim.obs_r[0, k] = r
        sim.obs_on[0, k] = True
    sim.pos[0] = torch.tensor([2.0, 0.0])
    sim.yaw[0] = 0.0
    sim.idx[0] = 20
    ranges = sim.lidar(torch.tensor(ANG, dtype=torch.float32), noise=False)[0].numpy()
    return ranges, (2.0, 0.0, 0.0), PathRef(sim.P[0].numpy()), 20


def test_object_on_path_detected_with_position_and_pickable():
    ranges, pose, path, ri = corridor([(3.5, 0.05, 0.04)])
    dets = detect_frame(ranges, ANG, pose, path, ri, DetectCfg())
    assert len(dets) == 1
    d = dets[0]
    assert np.linalg.norm(d.xy - np.array([3.5, 0.05])) < 0.05
    assert d.pickable
    assert abs(d.ahead - 1.5) < 0.1


def test_large_object_not_pickable():
    ranges, pose, path, ri = corridor([(3.0, -0.1, 0.15)])
    dets = detect_frame(ranges, ANG, pose, path, ri, DetectCfg())
    assert len(dets) == 1 and not dets[0].pickable
    assert np.linalg.norm(dets[0].xy - np.array([3.0, -0.1])) < 0.08


def test_walls_alone_no_detections():
    ranges, pose, path, ri = corridor([], half_width=0.45)
    assert detect_frame(ranges, ANG, pose, path, ri, DetectCfg()) == []


def test_object_outside_lane_ignored():
    ranges, pose, path, ri = corridor([(3.0, 0.5, 0.04)], half_width=0.8)
    assert detect_frame(ranges, ANG, pose, path, ri, DetectCfg()) == []


def test_object_behind_ignored():
    ranges, pose, path, ri = corridor([(1.2, 0.0, 0.08)])
    assert detect_frame(ranges, ANG, pose, path, ri, DetectCfg()) == []


def test_tracker_confirm():
    ranges, pose, path, ri = corridor([(3.5, 0.0, 0.05)])
    tr = Tracker(DetectCfg(confirm=2))
    assert tr.update(detect_frame(ranges, ANG, pose, path, ri)) == []
    assert len(tr.update(detect_frame(ranges, ANG, pose, path, ri))) == 1


def test_project_lateral_sign():
    path = PathRef(np.array([[0.0, 0.0], [5.0, 0.0]]))
    _, lat, s = path.project(np.array([[1.0, 0.3], [2.0, -0.2]]))
    assert lat[0] == pytest.approx(0.3, abs=1e-6) and lat[1] == pytest.approx(-0.2, abs=1e-6)
    assert s[0] == pytest.approx(1.0, abs=0.06)
