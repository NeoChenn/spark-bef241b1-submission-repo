import math

import numpy as np
import pytest
import torch

from patrol import features
from patrol.sim2d import PatrolSim, SimCfg


def small_sim(**kw):
    base = dict(num_envs=16, seed=0)
    base.update(kw)
    return PatrolSim(SimCfg(**base))


def test_obs_shape_and_finite():
    sim = small_sim()
    o = sim.observe()
    assert o.shape == (16, features.OBS_DIM)
    assert torch.isfinite(o).all()
    o, r, d, _ = sim.step(torch.zeros(16, 2))
    assert o.shape == (16, features.OBS_DIM) and torch.isfinite(o).all() and torch.isfinite(r).all()


@pytest.mark.parametrize("beams", [360, 72, 37])
def test_pool_sectors_np_torch_agree(beams):
    rng = np.random.default_rng(beams)
    angles = -math.pi + (np.arange(beams) + 0.5) * (2 * math.pi / beams)
    for _ in range(5):
        r = rng.uniform(0.05, 6.0, beams)
        r[rng.random(beams) < 0.1] = np.inf
        a = features.pool_sectors_np(r, angles)
        t = features.pool_sectors_torch(torch.tensor(r)[None], torch.tensor(angles))[0].numpy()
        assert a.shape == (features.SECTORS,)
        np.testing.assert_allclose(a, t, atol=1e-9)
        assert a.max() <= features.LIDAR_RANGE


def test_pool_sectors_partial_fov():
    angles = np.linspace(-math.radians(135), math.radians(135), 271)
    r = np.full(271, 1.0)
    a = features.pool_sectors_np(r, angles)
    t = features.pool_sectors_torch(torch.tensor(r)[None], torch.tensor(angles))[0].numpy()
    np.testing.assert_allclose(a, t)
    assert (a == features.LIDAR_RANGE).any() and (a == 1.0).any()


def test_build_obs_np_torch_agree():
    rng = np.random.default_rng(0)
    for _ in range(10):
        sec = rng.uniform(0, 4, features.SECTORS)
        lat, head = rng.normal(0, 0.3), rng.uniform(-6, 6)
        look = rng.normal(0, 1, (len(features.LOOKAHEAD), 2))
        v, w = rng.uniform(-0.3, 1), rng.uniform(-1.5, 1.5)
        la = rng.uniform(-1, 1, 2)
        dg = rng.uniform(0, 20)
        a = features.build_obs_np(sec, lat, head, look, v, w, la, dg)
        T = lambda x: torch.tensor(np.atleast_1d(x), dtype=torch.float64)
        t = features.build_obs_torch(T(sec)[None], T(lat), T(head), T(look)[None], T(v), T(w), T(la)[None], T(dg))[0]
        assert a.shape == (features.OBS_DIM,)
        np.testing.assert_allclose(a, t.numpy(), atol=1e-5)


def test_action_to_cmd_agree_and_bounded():
    rng = np.random.default_rng(1)
    acts = rng.uniform(-2, 2, (200, 2))
    vt, wt = features.action_to_cmd_torch(torch.tensor(acts))
    for k, a in enumerate(acts):
        vx, wz = features.action_to_cmd_np(a)
        assert abs(vx - float(vt[k])) < 1e-9 and abs(wz - float(wt[k])) < 1e-9
        assert features.V_MIN - 1e-9 <= vx <= features.V_MAX + 1e-9
        assert abs(wz) <= features.W_MAX + 1e-9
    assert features.action_to_cmd_np([1, 1]) == (features.V_MAX, features.W_MAX)
    assert features.action_to_cmd_np([-1, -1]) == (features.V_MIN, -features.W_MAX)


def test_walls_respect_min_half_width():
    sim = small_sim(num_envs=64)
    m = sim.cfg.min_half_width
    assert (sim.wl >= m - 1e-6).all() and (sim.wr >= m - 1e-6).all()


def test_obstacles_leave_gap():
    sim = small_sim(num_envs=64, p_obstacle=1.0)
    c = sim.cfg
    for n in range(sim.N):
        for k in range(c.n_obstacles):
            oc, r = sim.obs_c[n, k], float(sim.obs_r[n, k])
            i = int(((sim.P[n] - oc) ** 2).sum(-1).argmin())
            off = float(((oc - sim.P[n, i]) * sim.nrm[n, i]).sum())
            gap_l = float(sim.wl[n, i]) - off - r
            gap_r = float(sim.wr[n, i]) + off - r
            assert max(gap_l, gap_r) >= c.min_gap - 0.05, (n, k, gap_l, gap_r)


def test_reverse_twice_restores():
    sim = small_sim()
    P, wl, wr, h = sim.P.clone(), sim.wl.clone(), sim.wr.clone(), sim.h.clone()
    ids = torch.arange(sim.N)
    sim.reverse(ids)
    assert torch.allclose(sim.P[:, 0], P[:, -1])
    assert torch.allclose(sim.wl, wr.flip(1)) and torch.allclose(sim.wr, wl.flip(1))
    sim.reverse(ids)
    assert torch.allclose(sim.P, P) and torch.allclose(sim.wl, wl) and torch.allclose(sim.wr, wr)
    dh = torch.remainder(sim.h - h + math.pi, 2 * math.pi) - math.pi
    assert dh.abs().max() < 1e-4


def test_reverse_keeps_left_wall_on_left():
    sim = small_sim(p_obstacle=0.0)
    sim.reverse(torch.arange(sim.N))
    i = sim.M // 2
    left = sim.P[:, i] + sim.nrm[:, i] * sim.wl[:, i, None]
    fwd = torch.stack([sim.h[:, i].cos(), sim.h[:, i].sin()], -1)
    rel = left - sim.P[:, i]
    cross = fwd[:, 0] * rel[:, 1] - fwd[:, 1] * rel[:, 0]
    assert (cross > 0).all()


def test_drive_into_wall_collides_and_resets():
    sim = small_sim(num_envs=8, p_obstacle=0.0, p_push=0.0, lidar_noise=(0.0, 0.0), start_frac=0.0)
    lat, _ = sim.path_errors()
    sim.yaw = sim._at(sim.h, sim.idx) + math.pi / 2
    hit = torch.zeros(sim.N, dtype=torch.bool)
    for _ in range(60):
        act = torch.tensor([[1.0, 0.0]]).repeat(sim.N, 1)
        act[hit] = 0.0
        _, r, d, info = sim.step(act)
        new = info["collision"] & ~hit
        if new.any():
            assert d[new].all()
            assert (r[new] <= sim.cfg.r_collision + 1.0).all()
            assert (sim.steps[new] == 0).all()
        hit |= info["collision"]
        if hit.all():
            break
    assert hit.all()


def test_lidar_sees_walls_within_corridor():
    sim = small_sim(p_obstacle=0.0, lidar_noise=(0.0, 0.0), lidar_dropout=0.0)
    r = sim.lidar(noise=False)
    side = sim.beam_angles.abs().sub(math.pi / 2).abs() < 0.05
    clear = sim.clearance() + sim.cfg.robot_r
    assert (r[:, side].amin(-1) < features.LIDAR_RANGE).all()
    # clearance() uses the lateral offset at the nearest path point, which overestimates on bends.
    assert (r.amin(-1) >= clear - 0.1).all()


def test_onnx_export_roundtrip(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from patrol import ppo

    torch.manual_seed(0)
    ac = ppo.ActorCritic()
    ac.norm.update(torch.randn(256, features.OBS_DIM) * 3 + 1)
    ppo.export(ac, tmp_path)
    assert (tmp_path / "policy.onnx").exists() and (tmp_path / "policy_meta.json").exists()
    x = torch.randn(7, features.OBS_DIM)
    with torch.no_grad():
        ref = ppo.Deploy(ac).eval()(x).numpy()
    sess = ort.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    out = sess.run(["action"], {"obs": x.numpy().astype(np.float32)})[0]
    assert out.shape == (7, 2)
    np.testing.assert_allclose(out, ref, atol=1e-5)
    assert np.abs(out).max() <= 1.0 + 1e-6
