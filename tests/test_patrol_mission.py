import math

import numpy as np
import pytest
import torch

from patrol import features
from patrol.controller import PatrolController, cmd_to_action
from patrol.mission import MissionCfg, PatrolMission
from patrol.sim2d import PatrolSim, SimCfg


def expected_obs(sim, k):
    lat, head = sim.path_errors()
    sec = features.pool_sectors_torch(sim.lidar(noise=False), sim.beam_angles)
    o = features.build_obs_torch(sec, lat, head, sim.lookahead_robot(), sim.v, sim.w, sim.last_act, sim.dist_to_goal())
    return o[k].numpy()


def controller_obs(sim, k, ctl):
    r = sim.lidar(noise=False)[k].numpy()
    pose = (float(sim.pos[k, 0]), float(sim.pos[k, 1]), float(sim.yaw[k]))
    ctl.last_act = sim.last_act[k].numpy().copy()
    obs, _ = ctl.features(r, sim.beam_angles.numpy(), pose, float(sim.v[k]), float(sim.w[k]))
    return obs


def test_controller_matches_sim_observation():
    torch.manual_seed(0)
    sim = PatrolSim(SimCfg(num_envs=4, seed=0))
    for _ in range(15):
        sim.step(torch.rand(4, 2) * 2 - 1, auto_reset=False)
    for k in range(4):
        ctl = PatrolController("pure_pursuit", sim.P[k].numpy(), ds=sim.cfg.ds)
        assert ctl.M == sim.M
        ctl.idx = int(sim.idx[k])
        got = controller_obs(sim, k, ctl)
        assert ctl.idx == int(sim.idx[k])
        # controller heading = central difference, sim heading = incoming segment: <= 2e-3 after scaling
        np.testing.assert_allclose(got, expected_obs(sim, k), atol=2e-3)


def test_reverse_stays_in_sync_with_sim():
    torch.manual_seed(0)
    sim = PatrolSim(SimCfg(num_envs=2, seed=3))
    for _ in range(20):
        sim.step(torch.rand(2, 2) * 2 - 1, auto_reset=False)
    ctls = []
    for k in range(2):
        c = PatrolController("pure_pursuit", sim.P[k].numpy(), ds=sim.cfg.ds)
        c.idx = int(sim.idx[k])
        ctls.append(c)
    sim.reverse(torch.arange(2))
    for k, c in enumerate(ctls):
        c.reverse(pos=sim.pos[k].numpy())
        assert c.idx == int(sim.idx[k])
        np.testing.assert_allclose(c.P, sim.P[k].numpy(), atol=1e-5)
        got, exp = controller_obs(sim, k, c), expected_obs(sim, k)
        # sim.reverse keeps the old per-point heading (+pi), i.e. the heading one point off; bounded by max_curv*ds
        assert abs(got[37] - exp[37]) <= sim.cfg.max_curv * sim.cfg.ds / math.pi + 1e-4
        assert abs(got[36] - exp[36]) < 1e-2  # normals differ by the same one-point heading offset (< 5 mm)
        got[36:38] = exp[36:38]
        np.testing.assert_allclose(got, exp, atol=1e-4)


def test_onnx_policy_runs(tmp_path):
    from patrol.ppo import ActorCritic, export

    export(ActorCritic(), tmp_path)
    sim = PatrolSim(SimCfg(num_envs=1, seed=0))
    ctl = PatrolController(str(tmp_path / "policy.onnx"), sim.P[0].numpy())
    r = sim.lidar()[0].numpy()
    vx, wz, obs = ctl.step(r, sim.beam_angles.numpy(), (float(sim.pos[0, 0]), float(sim.pos[0, 1]), float(sim.yaw[0])), 0.0, 0.0)
    assert obs.shape == (features.OBS_DIM,)
    assert math.isfinite(vx) and math.isfinite(wz)
    assert features.V_MIN <= vx <= features.V_MAX and abs(wz) <= features.W_MAX


def straight_world(obstacles):
    """20 m straight corridor along +x, walls 0.8 m each side, no noise or pushes."""
    sim = PatrolSim(SimCfg(num_envs=1, seed=0, start_frac=0.0, p_push=0.0, lidar_noise=(0.0, 0.0),
                           lidar_dropout=0.0, max_steps=10**9))
    M = sim.M
    s = torch.arange(M) * sim.cfg.ds
    sim.P[0] = torch.stack([s, torch.zeros(M)], -1)
    sim.h[0] = 0.0
    sim.nrm[0] = torch.tensor([0.0, 1.0])
    sim.wl[0] = 0.8
    sim.wr[0] = 0.8
    sim.obs_on[0] = False
    for k, (xy, r) in enumerate(obstacles):
        sim.obs_c[0, k] = torch.tensor(xy)
        sim.obs_r[0, k] = r
        sim.obs_on[0, k] = True
    sim.pos[0] = torch.tensor([0.0, 0.0])
    sim.yaw[0] = 0.0
    sim.v[0] = sim.w[0] = 0.0
    sim.idx[0] = 0
    sim.alpha[0] = 0.6
    return sim


def drive(sim, mission, max_steps, stop_on=None):
    beams = -math.pi + (torch.arange(360) + 0.5) * (2 * math.pi / 360)
    events = []
    for _ in range(max_steps):
        r = sim.lidar(beams=beams)[0].numpy()
        n = mission.reversed_count
        cmd = mission.step(0.0, r, beams.numpy(), (float(sim.pos[0, 0]), float(sim.pos[0, 1]), float(sim.yaw[0])),
                           float(sim.v[0]), float(sim.w[0]))
        if mission.reversed_count != n:
            sim.reverse(torch.tensor([0]))
        if cmd.event:
            events.append(cmd.event)
        if cmd.event == "picked":
            d = (sim.obs_c[0] - torch.from_numpy(cmd.arm_target_xy).float()).norm(dim=-1)
            k = int(torch.where(sim.obs_on[0], d, torch.full_like(d, 1e9)).argmin())
            if float(d[k]) < 0.2:
                sim.remove_obstacle(0, k)
        _, _, _, info = sim.step(torch.from_numpy(cmd_to_action(cmd.vx, cmd.wz))[None], auto_reset=False)
        assert not bool(info["collision"][0]), f"collision in {mission.state}, events {events}"
        if stop_on and cmd.event == stop_on:
            break
    return events


def test_mission_picks_object_and_dumps_at_end():
    sim = straight_world([((8.0, 0.05), 0.04)])
    mission = PatrolMission(PatrolController("pure_pursuit", sim.P[0].numpy()), MissionCfg())
    events = drive(sim, mission, 1500, stop_on="dumped")
    assert "picked" in events and "arrived" in events and events[-1] == "dumped"
    assert mission.picked == 1 and mission.items == 0 and mission.legs == 1
    assert not bool(sim.obs_on[0, 0])
    assert len(mission.overlay_text()) == 4


@pytest.mark.parametrize("radius,pickable", [(0.05, True), (0.08, False)])
def test_size_check_ignores_wall_but_rejects_wide_objects(radius, pickable):
    # object 4 cm from the left wall, robot stopped 0.45 m away at the pick standoff
    sim = straight_world([((8.0, 0.42 - 0.04 - radius), radius)])
    sim.wl[0] = 0.42
    obj = sim.obs_c[0, 0].numpy()
    sim.pos[0] = torch.tensor([float(obj[0]) - 0.45, float(obj[1]) - 0.1])
    sim.yaw[0] = math.atan2(0.1, 0.45)
    beams = -math.pi + (torch.arange(360) + 0.5) * (2 * math.pi / 360)
    mission = PatrolMission(PatrolController("pure_pursuit", sim.P[0].numpy()), MissionCfg())
    mission._scan = (sim.lidar(beams=beams)[0].numpy(), beams.numpy(),
                     (float(sim.pos[0, 0]), float(sim.pos[0, 1]), float(sim.yaw[0])))
    width = mission._extent_near(obj, 0.18)
    assert abs(width - 2 * radius) < 0.025
    assert (width <= mission.cfg.detect.pick_max_width) == pickable



def test_big_object_is_reported_then_robot_turns_back_to_start_bin():
    sim = straight_world([((6.0, 0.05), 0.15)])
    mission = PatrolMission(PatrolController("pure_pursuit", sim.P[0].numpy()), MissionCfg())
    events = drive(sim, mission, 1200, stop_on="arrived")  # drive() fails on any collision
    assert events[:2] == ["reported", "turn_back"], events
    assert events[-1] == "arrived" and "blocked" not in events and mission.picked == 0
    rep = mission.reports[0]
    assert rep["width_m"] > mission.cfg.detect.pick_max_width and abs(rep["s_m"] - 6.0) < 0.3
    assert float(sim.pos[0, 0]) < 1.0  # back at the start bin, never went past the object


def test_reported_object_that_gets_cleared_lets_the_robot_continue():
    sim = straight_world([((6.0, 0.05), 0.15)])
    mission = PatrolMission(PatrolController("pure_pursuit", sim.P[0].numpy()), MissionCfg(report_wait=30.0))
    drive(sim, mission, 400, stop_on="reported")
    sim.remove_obstacle(0, 0)
    events = drive(sim, mission, 600, stop_on="arrived")
    assert events[0] == "cleared" and events[-1] == "arrived" and mission.legs == 1
