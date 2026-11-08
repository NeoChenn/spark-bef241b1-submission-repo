"""PickState (litter pickup physics, pure torch) and the scripted baseline."""
import math

import torch

from policy.envs.pick_scripted import rollout, scripted_action
from policy.envs.pick_state import G_MAX, LIFT_H, OBS_DIM, REACH, TOOL_LEN, PickParams, PickState


def one(**kw):
    s = PickState(1, PickParams(est_bias=(0, 0), est_yaw_bias=(0, 0), est_noise=0.0), seed=0)
    s.obj[:] = torch.tensor([[0.28, 0.0, 0.0]])
    s.obj_yaw[:] = 0.0
    s.w[:], s.h[:], s.l[:] = 0.06, 0.06, 0.15
    for k, v in kw.items():
        getattr(s, k)[:] = torch.as_tensor(v)
    return s


def drive(s, a, n):
    for _ in range(n):
        s.step(torch.as_tensor([a], dtype=torch.float32))


def test_observation_size_and_reset_ranges():
    s = PickState(256, seed=1)
    o = s.observe(torch.zeros(256, 5), torch.zeros(256))
    assert o.shape == (256, OBS_DIM) and torch.isfinite(o).all()
    assert (s.trunk_h >= 0.17).all() and (s.trunk_h <= 0.21).all()
    assert (s.w <= G_MAX - 0.03).all()                       # every object fits in the open jaws with margin


def test_reach_shell_and_deck_are_respected():
    s = PickState(64, seed=2)
    drive_all = torch.zeros(64, 5)
    drive_all[:, 2] = -1                                       # push straight down for a long time
    drive_all[:, 0] = 1                                        # and far forward
    for _ in range(200):
        s.step(drive_all)
    assert (s.tcp[:, 2] >= 0.002 - 1e-6).all()
    wrist = s.tcp + torch.tensor([0, 0, TOOL_LEN])
    assert ((wrist - s.shoulder).norm(dim=-1) <= REACH[1] + 1e-4).all()


def test_closing_around_a_centred_aligned_object_grasps_it():
    s = one(tcp=[[0.28, 0.0, 0.03]], tgt=[[0.28, 0.0, 0.03]], yaw=0.0, yaw_tgt=0.0, grip=G_MAX)
    drive(s, [0, 0, 0, 0, -1], 30)
    assert s.attached.item() and abs(s.grip.item() - 0.06) < 1e-6
    drive(s, [0, 0, 1, 0, -1], 30)                             # lift
    assert s.success.item() and s.obj[0, 2] >= LIFT_H


def test_misaligned_jaws_do_not_grasp():
    s = one(tcp=[[0.28, 0.0, 0.03]], tgt=[[0.28, 0.0, 0.03]], yaw=math.pi / 2, yaw_tgt=math.pi / 2, grip=G_MAX)
    drive(s, [0, 0, 0, 0, -1], 30)
    assert not s.attached.item()


def test_off_centre_does_not_grasp():
    s = one(tcp=[[0.28, 0.025, 0.03]], tgt=[[0.28, 0.025, 0.03]], yaw=0.0, yaw_tgt=0.0, grip=G_MAX)
    drive(s, [0, 0, 0, 0, -1], 30)
    assert not s.attached.item()


def test_coming_down_with_closed_jaws_knocks_the_object():
    s = one(tcp=[[0.28, 0.0, 0.12]], tgt=[[0.28, 0.0, 0.12]], yaw=0.0, yaw_tgt=0.0, grip=0.0)
    before = s.obj[0, :2].clone()
    for _ in range(20):
        drive(s, [0, 0, -1, 0, -1], 1)
        if s.knocks.item() >= 1:
            break
    assert s.knocks.item() == 1 and (s.obj[0, :2] - before).norm() > 0.01
    assert s.tcp[0, 2] >= s.h.item() - 1e-4                   # at the knock it rests on the object, not through it


def test_opening_while_holding_drops_it():
    s = one(tcp=[[0.28, 0.0, 0.03]], tgt=[[0.28, 0.0, 0.03]], yaw=0.0, yaw_tgt=0.0, grip=G_MAX)
    drive(s, [0, 0, 0, 0, -1], 30)
    drive(s, [0, 0, 1, 0, -1], 10)
    drive(s, [0, 0, 0, 0, 1], 2)
    assert not s.attached.item() and s.drops.item() == 1 and s.obj[0, 2] == 0.0


def test_scripted_pick_works_with_a_perfect_estimate():
    s = PickState(256, PickParams(est_bias=(0, 0), est_yaw_bias=(0, 0)), seed=3)
    r = rollout(s, lambda st, o: scripted_action(st))
    assert r["success"].float().mean() > 0.9


def test_camera_error_hurts_the_scripted_pick():
    perfect = rollout(PickState(256, PickParams(est_bias=(0, 0), est_yaw_bias=(0, 0)), seed=4),
                      lambda st, o: scripted_action(st))["success"].float().mean()
    noisy = rollout(PickState(256, PickParams(est_bias=(0.01, 0.02), est_yaw_bias=(0.1, 0.2)), seed=4),
                    lambda st, o: scripted_action(st))["success"].float().mean()
    assert noisy < perfect - 0.1


# ------------------------------------------------------------------ rewards (pick_cpu_env)
def test_finishing_always_pays_more_than_farming_the_shaping():
    """Success ends the episode; it must beat hovering at the grasp point (or holding just below the lift height)
    for all the remaining steps, or PPO learns not to finish (it did)."""
    from policy.envs.pick_cpu_env import EPISODE_STEPS, MAX_STEP_REWARD, pick_reward
    s = one(tcp=[[0.28, 0.0, 0.03]], tgt=[[0.28, 0.0, 0.03]], yaw=0.0, yaw_tgt=0.0, grip=G_MAX)
    ev = {k: torch.zeros(1, dtype=torch.bool) for k in ("grasped", "knocked", "dropped")}
    a = torch.zeros(1, 5)
    for t in (0, 100, 230):
        remaining = torch.tensor([EPISODE_STEPS - 1 - t])
        farm = pick_reward(s, ev, a, a, torch.zeros(1, dtype=torch.bool), remaining).item()
        assert farm <= MAX_STEP_REWARD
        s.success[:] = True
        finish = pick_reward(s, ev, a, a, torch.zeros(1, dtype=torch.bool), remaining).item()
        s.success[:] = False
        assert finish > farm * (remaining.item() + 1)


def test_cpu_env_steps_and_resets():
    from policy.envs.pick_cpu_env import PickCpuEnv
    env = PickCpuEnv(16, seed=0)
    obs, _ = env.reset()
    assert obs["policy"].shape == (16, OBS_DIM)
    for _ in range(EPISODE_STEPS_CHECK):
        obs, rew, done, extras = env.step(torch.zeros(16, 5))
    assert done.any() and "time_outs" in extras and torch.isfinite(rew).all()


EPISODE_STEPS_CHECK = 240
