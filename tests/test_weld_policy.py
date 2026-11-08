import numpy as np

from policy.weld.env import WeldEnv, bead_width, run_pass
from policy.weld.policy import OpenLoop, SeamTracker
from policy.weld.seam import Seam


def mean_metrics(make, seeds=range(8), **kw):
    out = [run_pass(Seam(seed=s), make(), *Seam(seed=s).crack_extent(), seed=s, **kw)["metrics"] for s in seeds]
    return {k: float(np.mean([m[k] for m in out])) for k in out[0]}


def test_env_step_moves_torch_and_lays_bead():
    s = Seam(seed=0)
    env = WeldEnv(s, 50.0, 80.0, seed=0)
    obs = env.obs()
    assert {"x", "y", "profile_y", "profile_z", "center_est", "eddy", "tele", "temp_c"} <= set(obs)
    for _ in range(40):
        obs, done = env.step(0.0, 3.0, 160.0)
        if done:
            break
    assert env.x > 60 and s.bead.any()
    assert 8 < bead_width(160.0, 3.0) < 16


def test_seam_tracker_beats_open_loop_and_is_perception_driven():
    t = mean_metrics(SeamTracker)
    o = mean_metrics(OpenLoop)
    blind = mean_metrics(SeamTracker, blind=True)
    assert t["track_rmse_mm"] < 0.6 < 1.2 < o["track_rmse_mm"]
    assert t["coverage"] > 0.97 and o["coverage"] < t["coverage"] - 0.05
    assert t["defect_rate"] < 0.1 and o["defect_rate"] > 0.3
    assert blind["track_rmse_mm"] > 1.2 and blind["defect_rate"] > 0.3


def test_tracker_holds_heat_input_in_band_against_voltage_drift():
    m = mean_metrics(SeamTracker)
    o = mean_metrics(OpenLoop)
    assert m["heat_out_of_band"] < 0.05
    assert o["heat_out_of_band"] > m["heat_out_of_band"]
    assert m["temp_max_c"] < 250
