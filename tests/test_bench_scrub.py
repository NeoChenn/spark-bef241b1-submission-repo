import math

import numpy as np
import torch

from bench.scrub_sim import ArmFK, KinematicScrub, ScrubCfg, make_scenario
from bench.stats import episode_seeds
from integration.kinematics import ARM_JOINTS, RobotModel


def test_scenario_is_deterministic_per_seed():
    a, b = make_scenario(123), make_scenario(123)
    c = make_scenario(124)
    assert torch.equal(a["rust"], b["rust"]) and a["q0"] == b["q0"] and a["event_step"] == b["event_step"]
    assert not torch.equal(a["rust"], c["rust"])
    assert a["rust"].sum() > 1.0
    assert a["event_step"] > 0


def test_scenario_rust_within_reach_and_window():
    s = make_scenario(7)
    assert not (s["rust"] > 0)[~s["reachable"]].any()
    assert (s["sprayed"] | (s["rust"] <= 0.02)).all()


def test_fk_matches_urdf_model():
    m = RobotModel.from_urdf()
    q = {**m.default_q(), "ur3_shoulder_pan_joint": -1.9, "ur3_elbow_joint": -1.1, "lift_joint": 0.41}
    base = np.eye(4)
    base[2, 3] = 0.32
    T_ref = m.tool_pose(base, q)
    fk = ArmFK()
    qt = torch.tensor([[q[j] for j in ARM_JOINTS]])
    T, axes, pts = fk.forward(torch.tensor(base).float()[None], qt, torch.tensor([0.41]))
    assert np.allclose(T[0].numpy(), T_ref, atol=1e-5)
    assert axes.shape == (1, 6, 3) and pts.shape == (1, 6, 3)


def test_obs_layout_matches_policy():
    sim = KinematicScrub([make_scenario(s) for s in episode_seeds(3)])
    o = sim.obs()
    assert o.shape == (3, 327) and torch.isfinite(o).all()


def test_batched_equals_single():
    seeds = episode_seeds(3)
    sc = [make_scenario(s) for s in seeds]
    batch = KinematicScrub(sc)
    single = KinematicScrub([sc[1]])
    g = torch.Generator().manual_seed(0)
    for _ in range(40):
        a = torch.rand(3, 4, generator=g) * 2 - 1
        batch.step(a)
        single.step(a[1:2])
    assert torch.allclose(batch.rust.rust[1], single.rust.rust[0], atol=1e-5)
    assert torch.allclose(batch.obs()[1], single.obs()[0], atol=1e-4)


def test_pressing_on_rust_with_target_interface_removes_rust_and_respects_clamp():
    sim = KinematicScrub([make_scenario(s) for s in episode_seeds(4)])
    f0 = sim.rust.fraction_remaining().clone()
    for k in range(200):
        uv = sim.rust.surface_to_uv(sim.theta, sim.axial)
        near, _ = sim.rust.nearest_rust(uv)
        th = sim.theta + near[:, 0] / sim.pipe_r
        z = sim.axial + near[:, 1] + 0.04 * (1 if (k // 6) % 2 else -1)
        sim.step(sim.native_from_target(th, z, torch.ones(4)))
    assert (sim.rust.fraction_remaining() < f0).all()
    assert (sim.f_set <= 38.0 + 1e-6).all()


def test_episode_ends_and_stats():
    c = ScrubCfg(episode_s=1.0)
    sim = KinematicScrub([make_scenario(1, c)], c)
    for _ in range(c.max_steps + 5):
        sim.step(torch.zeros(1, 4))
    st = sim.episode_stats()[0]
    assert sim.done.all() and st["episode_s"] <= 1.0 + 1e-6
    assert set(st) >= {"success", "time_to_clean_s", "peak_force_N", "over_40N", "removed_pct_per_s"}
