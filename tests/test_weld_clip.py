import json

import numpy as np
import pytest

from policy.weld.clip import plan, world


@pytest.fixture(scope="module")
def clip():
    return plan(frames=60, weld=(5, 50))


def test_tip_follows_seam(clip):
    seam = np.array(clip["seam_world"])
    x0 = clip["pipe"]["x0"]
    for f in clip["frames"]:
        if not f["arc"]:
            continue
        x, y = f["torch_seam_xy"]
        assert abs(y - f["true_center_y_mm"]) < 3.0
        tip = np.array(f["tip"])
        assert np.linalg.norm(tip - world(x, y, 11.0, x0)) < 3e-3
        assert np.min(np.linalg.norm(seam - tip, axis=1)) < 0.02


def test_ik_and_tracking(clip):
    m = clip["metrics"]
    assert m["ik_max_err_mm_weld"] < 2.0
    assert m["tracker_rmse_mm"] < 0.6 < 3.0 < m["open_loop_rmse_mm"]
    assert m["max_dq_rad_per_frame"] < 0.15


def test_arc_phases_and_bead(clip):
    arcs = [f["arc"] for f in clip["frames"]]
    assert not arcs[0] and arcs[20] and not arcs[-1]
    prog = [f["progress_mm"] for f in clip["frames"]]
    assert np.all(np.diff(prog) >= -1e-9) and prog[-1] > 180
    last = clip["frames"][-1]
    assert len(last["bead_world"]) == len(last["bead_width_m"]) == len(last["bead_age_s"]) <= 300
    assert len(clip["frames"][20]["laser_world"]) == 64


def test_json(clip):
    json.dumps(clip, allow_nan=False)
