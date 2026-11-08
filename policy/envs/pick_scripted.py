"""A hand-written pick, used as a baseline for the RL policy and to test the physics in pick_state.py.

hover above the camera's grasp point -> line the jaws up with the object's width axis -> open -> lower to half the
object's height -> close -> lift. It trusts the camera estimate completely, so it misses when the estimate is off by
more than the grasp tolerance; the RL policy sees the same estimate and learns to cope.
"""
from __future__ import annotations

import math

import torch

from policy.envs.pick_state import PickState


def scripted_action(s: PickState) -> torch.Tensor:
    est = s.est_grasp_point()
    err = est - s.tcp
    dyaw = torch.remainder(s.est_yaw() - s.yaw + math.pi / 2, math.pi) - math.pi / 2
    if not hasattr(s, "_committed") or s._committed.shape[0] != s.n:
        s._committed = torch.zeros(s.n, dtype=torch.bool, device=s.dev)
    holding = s.attached
    xy_ok = err[:, :2].norm(dim=-1) < 0.006
    aligned = dyaw.abs() < 0.08
    open_enough = s.grip > s.w + 0.015
    at_depth = (s.tcp[:, 2] - est[:, 2]).abs() < 0.006
    # once the jaws start closing at depth, stay there until it is held (or the jaws are shut on nothing)
    s._committed = (s._committed | (at_depth & xy_ok & aligned & open_enough)) & ~holding & (s.grip > 0.002)
    ready = xy_ok & aligned & open_enough
    hover = est[:, 2] + 0.08
    z_goal = torch.where(holding, torch.full_like(hover, 0.30),
                         torch.where(s._committed | ready, est[:, 2], hover))
    a = torch.zeros(s.n, 5, device=s.dev)
    move_xy = ~holding & ~s._committed
    a[:, :2] = torch.where(move_xy[:, None], err[:, :2] / s.p.step_m, torch.zeros_like(err[:, :2]))
    a[:, 2] = (z_goal - s.tcp[:, 2]) / s.p.step_m
    a[:, 3] = torch.where(move_xy, dyaw / s.p.step_yaw, torch.zeros_like(dyaw))
    a[:, 4] = torch.where(holding | s._committed, -torch.ones_like(dyaw), torch.ones_like(dyaw))
    return a.clamp(-1, 1)


def rollout(s: PickState, policy, steps: int = 240) -> dict:
    """Run ``policy(state, obs) -> action`` until success or ``steps``; returns per-env outcome tensors."""
    last = torch.zeros(s.n, 5, device=s.dev)
    done = torch.zeros(s.n, dtype=torch.bool, device=s.dev)
    t_success = torch.full((s.n,), -1, device=s.dev)
    for t in range(steps):
        obs = s.observe(last, torch.full((s.n,), t / steps, device=s.dev))
        a = policy(s, obs)
        a = torch.where(done[:, None], torch.zeros_like(a), a)
        s.step(a)
        last = a
        newly = s.success & ~done
        t_success[newly] = t
        done |= s.success
        if done.all():
            break
    return {"success": done, "t_success": t_success, "knocks": s.knocks.clone(), "drops": s.drops.clone()}
