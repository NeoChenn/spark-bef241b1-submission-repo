"""Batched litter-on-deck pick model. PURE TORCH (no Isaac imports), like policy/envs/rust_state.py, so it is
unit-testable on CPU and shared by the CPU trainer (policy/envs/pick_cpu_env.py) and the Isaac Lab env
(policy/envs/pick_env.py).

Robot: the patrol team's ``sim/assets/go2_ur3/go2_ur3_gripper.urdf`` (branch patrol-nav): Go2 + lift + UR3 +
parallel-jaw gripper (fingers 0-0.06 m each, 0.12 m max opening), TCP = ``grasp_frame`` (ur3_ee_link + 0.07 m;
0.15 m from the wrist). This skill replaces the scripted grasp in their PICK state (patrol/mission.py).

Frame (per env, metres, world z up): the deck is z = 0. The Go2 trunk is at (0, 0, trunk_h), facing +x. The UR3
shoulder is at SHOULDER_XY, trunk_h + SHOULDER_DZ (lift mount +0.064, carriage +0.1 with the lift fully down,
shoulder +0.1519). Reach: standing (trunk 0.32 m) the shoulder is 0.64 m up and the UR3 (0.5 m) cannot get the
gripper to the deck, so for a pick the Go2 crouches to trunk_h 0.17-0.21 m (set at reset, NOT a policy action;
like the scrub skill's lift height) and stops with the object 0.20-0.32 m ahead of the trunk.

As with the scrub skill's brush, contact is analytic: the grasp is modelled, not simulated.
  * opening g in [0, G_MAX]; jaws close across the tool's local y axis (rotated by the tool yaw).
  * reach: the wrist point (TCP + TOOL_LEN up) must stay 0.12-0.50 m from the shoulder (the scrub skill's
    shoulder-distance proxy for UR3 reachability); the TCP cannot go below the deck.

Objects (litter): an oriented box on the deck, width w (across, what the jaws close on), length l (along), height h.
  lying drink can: w = h = 2r (r 0.026-0.04), l 0.09-0.20;  flat packaging: w 0.04-0.08, h 0.02-0.06, l 0.06-0.15.

Events each step:
  * grasp: the jaws close onto the object (opening reaches w) with the TCP centred across it (|c| < GRASP_C), inside
    its length, at 15-85 % of its height, and the jaw axis within GRASP_YAW of the object's width axis -> attached.
  * knock: the fingers come down onto the object instead of around it -> the object is pushed 1.5 cm aside and turns
    a little (a penalty, and the policy must re-aim).
  * drop: the gripper opens while holding -> the object falls back onto the deck.
  * success: holding the object with its bottom at least LIFT_H above the deck.

The policy only sees the object through a camera estimate (``est_*``): a fixed per-episode bias (DR) plus small
per-step noise, as it would get from the debris detector's pick point and blob orientation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

SHOULDER_XY = (-0.01, 0.0)
SHOULDER_DZ = 0.064 + 0.1 + 0.1519   # go2_ur3_gripper.urdf, lift_joint at 0 (lowest)
TOOL_LEN = 0.15            # wrist point to TCP (gripper + flange), m
REACH = (0.12, 0.50)       # wrist point distance from the shoulder, m
G_MAX = 0.12               # max jaw opening, m (a 2F-140-class gripper opens 0.14 m)
GRIP_SPEED = 0.30          # m/s jaw speed
LIFT_H = 0.15              # success: object bottom this high, m
GRASP_C = 0.012            # max off-centre across the object for a grasp, m
GRASP_YAW = 0.35           # rad
OBS_DIM = 23
ACT_DIM = 5


@dataclass
class PickParams:
    dt: float = 1 / 30
    step_m: float = 0.01               # max TCP target move per step (scrub: 1 cm / step at 30 Hz)
    step_yaw: float = 0.08             # rad per step
    track: float = 0.5                 # first-order tracking of the target (IK + joint control lag)
    trunk_h: tuple = (0.17, 0.21)      # Go2 crouched for the pick
    obj_x: tuple = (0.20, 0.32)        # object ahead of the trunk, m
    obj_y: tuple = (-0.10, 0.10)
    can_frac: float = 0.6
    est_bias: tuple = (0.0, 0.015)     # camera position bias std range (per episode), m
    est_yaw_bias: tuple = (0.0, 0.15)  # rad
    est_noise: float = 0.002           # per-step position noise, m
    hover_start: float = 0.0           # training curriculum: share of episodes that start hovering over the object


def wrap_pi(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def jaw_yaw_error(tool_yaw: torch.Tensor, obj_yaw: torch.Tensor) -> torch.Tensor:
    """|angle| between the jaw closing axis and the object's width axis; jaws are symmetric, so modulo pi."""
    d = torch.remainder(tool_yaw - obj_yaw + math.pi / 2, math.pi) - math.pi / 2
    return d.abs()


class PickState:
    def __init__(self, num_envs: int, params: PickParams | None = None, device="cpu", seed: int | None = None):
        self.n, self.p, self.dev = num_envs, params or PickParams(), torch.device(device)
        self.g = torch.Generator(device=self.dev)
        if seed is not None:
            self.g.manual_seed(seed)
        z = lambda *s: torch.zeros(num_envs, *s, device=self.dev)  # noqa: E731
        self.trunk_h, self.shoulder = z(), z(3)
        self.tcp, self.tgt, self.yaw, self.yaw_tgt, self.grip, self.grip_cmd = z(3), z(3), z(), z(), z(), z()
        self.obj, self.obj_yaw, self.w, self.l, self.h = z(3), z(), z(), z(), z()
        self.bias, self.yaw_bias = z(2), z()
        self.attached = torch.zeros(num_envs, dtype=torch.bool, device=self.dev)
        self.success = torch.zeros_like(self.attached)
        self.ever_grasped = torch.zeros_like(self.attached)
        self.knocks, self.drops = z(), z()
        self.reset(torch.arange(num_envs, device=self.dev))

    # ------------------------------------------------------------------ sampling
    def _u(self, n, lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=self.g, device=self.dev)

    def reset(self, ids: torch.Tensor) -> None:
        p, n = self.p, len(ids)
        if n == 0:
            return
        self.trunk_h[ids] = self._u(n, *p.trunk_h)
        self.shoulder[ids] = torch.stack([torch.full((n,), SHOULDER_XY[0], device=self.dev),
                                          torch.full((n,), SHOULDER_XY[1], device=self.dev),
                                          self.trunk_h[ids] + SHOULDER_DZ], -1)
        can = torch.rand(n, generator=self.g, device=self.dev) < p.can_frac
        r = self._u(n, 0.026, 0.04)
        self.w[ids] = torch.where(can, 2 * r, self._u(n, 0.04, 0.08))
        self.h[ids] = torch.where(can, 2 * r, self._u(n, 0.02, 0.06))
        self.l[ids] = torch.where(can, self._u(n, 0.09, 0.20), self._u(n, 0.06, 0.15))
        self.obj[ids] = torch.stack([self._u(n, *p.obj_x), self._u(n, *p.obj_y), torch.zeros(n, device=self.dev)], -1)
        self.obj_yaw[ids] = self._u(n, -math.pi, math.pi)
        sb = self._u(n, *p.est_bias)
        self.bias[ids] = torch.randn(n, 2, generator=self.g, device=self.dev) * sb[:, None]
        self.yaw_bias[ids] = torch.randn(n, generator=self.g, device=self.dev) * self._u(n, *p.est_yaw_bias)
        start = torch.stack([self._u(n, 0.18, 0.30), self._u(n, -0.10, 0.10), self._u(n, 0.22, 0.32)], -1)
        yaw0 = self._u(n, -0.5, 0.5)
        if p.hover_start > 0:
            # start a share of episodes with the open gripper 3-8 cm above the object, roughly lined up, so PPO
            # sees grasps early (reset-distribution curriculum); evaluation uses hover_start = 0
            hov = torch.rand(n, generator=self.g, device=self.dev) < p.hover_start
            above = torch.stack([self.obj[ids, 0] + self._u(n, -0.015, 0.015), self.obj[ids, 1] + self._u(n, -0.015, 0.015),
                                 0.5 * self.h[ids] + self._u(n, 0.03, 0.08)], -1)
            start = torch.where(hov[:, None], above, start)
            yaw0 = torch.where(hov, self.obj_yaw[ids] + self._u(n, -0.2, 0.2), yaw0)
        self.tcp[ids] = start
        self.tgt[ids] = start
        self.yaw[ids] = wrap_pi(yaw0)
        self.yaw_tgt[ids] = self.yaw[ids]
        self.grip[ids] = self._u(n, 0.0, G_MAX)
        self.grip_cmd[ids] = self.grip[ids]
        self.attached[ids] = False
        self.success[ids] = False
        self.ever_grasped[ids] = False
        self.knocks[ids] = 0
        self.drops[ids] = 0

    # ------------------------------------------------------------------ geometry
    def to_obj(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(along, across) coordinates of world xy points in the object's frame."""
        d = xy - self.obj[:, :2]
        c, s = torch.cos(self.obj_yaw), torch.sin(self.obj_yaw)
        return d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c

    def grasp_point(self) -> torch.Tensor:
        """Where the TCP should be for a grasp: object centre, half its height."""
        return torch.stack([self.obj[:, 0], self.obj[:, 1], self.obj[:, 2] + 0.5 * self.h], -1)

    def est_grasp_point(self) -> torch.Tensor:
        gp = self.grasp_point()
        noise = torch.randn(self.n, 2, generator=self.g, device=self.dev) * self.p.est_noise
        return torch.cat([gp[:, :2] + self.bias + noise, gp[:, 2:]], -1)

    def est_yaw(self) -> torch.Tensor:
        return self.obj_yaw + self.yaw_bias

    def clamp_reach(self, tcp: torch.Tensor) -> torch.Tensor:
        """Keep the TCP above the deck and the wrist point inside the UR3's reach shell."""
        tcp = tcp.clone()
        tcp[:, 2] = tcp[:, 2].clamp(min=0.002)
        wrist = tcp + torch.tensor([0.0, 0.0, TOOL_LEN], device=self.dev)
        d = wrist - self.shoulder
        r = d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        rc = r.clamp(REACH[0], REACH[1])
        wrist = self.shoulder + d / r * rc
        tcp = wrist - torch.tensor([0.0, 0.0, TOOL_LEN], device=self.dev)
        tcp[:, 2] = tcp[:, 2].clamp(min=0.002)
        return tcp

    # ------------------------------------------------------------------ step
    def command(self, a: torch.Tensor) -> torch.Tensor:
        """Turn an action into targets: TCP target, yaw target, gripper goal. Returns the 'open' command (N,) bool.

        a: (N, 5) in [-1, 1]: dx, dy, dz of the TCP target, d(yaw), gripper (> 0 open, <= 0 close)."""
        p = self.p
        a = a.clamp(-1, 1)
        self.tgt = self.clamp_reach(self.tgt + a[:, :3] * p.step_m)
        self.yaw_tgt = wrap_pi(self.yaw_tgt + a[:, 3] * p.step_yaw)
        open_cmd = a[:, 4] > 0
        self.grip_cmd = torch.where(open_cmd, torch.full_like(self.grip, G_MAX), torch.zeros_like(self.grip))
        return open_cmd

    def step(self, a: torch.Tensor) -> dict:
        """CPU step: command, a first-order model of the arm tracking its target, then the contact events.
        The Isaac env replaces the tracking model with the real arm and calls ``events`` with the measured TCP."""
        p = self.p
        open_cmd = self.command(a)
        prev_tcp, prev_yaw, prev_grip = self.tcp.clone(), self.yaw.clone(), self.grip.clone()
        self.tcp = self.clamp_reach(self.tcp + p.track * (self.tgt - self.tcp))
        self.yaw = wrap_pi(self.yaw + p.track * wrap_pi(self.yaw_tgt - self.yaw))
        self.grip = self.grip + (self.grip_cmd - self.grip).clamp(-GRIP_SPEED * p.dt, GRIP_SPEED * p.dt)
        return self.events(open_cmd, prev_tcp, prev_yaw, prev_grip)

    def events(self, open_cmd, prev_tcp, prev_yaw, prev_grip) -> dict:
        """Grasp / knock / drop / carry / success, given where the TCP, yaw and jaws are now (self.tcp, self.yaw,
        self.grip) and were one step ago."""
        ev = {"grasped": torch.zeros_like(self.attached), "knocked": torch.zeros_like(self.attached),
              "dropped": torch.zeros_like(self.attached)}
        free = ~self.attached
        along, across = self.to_obj(self.tcp[:, :2])
        inside_len = along.abs() < 0.5 * self.l - 0.01
        z_ok = (self.tcp[:, 2] > 0.15 * self.h) & (self.tcp[:, 2] < 0.85 * self.h)
        yaw_ok = jaw_yaw_error(self.yaw, self.obj_yaw) < GRASP_YAW
        centred = across.abs() < GRASP_C
        # grasp: jaws closing reach the object's width this step
        closing_onto = free & (prev_grip > self.w) & (self.grip <= self.w + 1e-4) & ~open_cmd
        grasp = closing_onto & inside_len & z_ok & yaw_ok & centred
        self.attached |= grasp
        self.grip = torch.where(self.attached, self.w, self.grip)          # jaws rest on the held object
        ev["grasped"] = grasp
        # knock: fingers come down onto the object rather than around it
        straddle = (self.grip > self.w + 0.004) & (across.abs() < 0.5 * (self.grip - self.w)) & yaw_ok
        over = (along.abs() < 0.5 * self.l) & (across.abs() < 0.5 * self.w + 0.5 * self.grip + 0.01)
        moving_down = self.tcp[:, 2] < prev_tcp[:, 2] - 1e-4
        knock = free & ~grasp & over & ~straddle & (self.tcp[:, 2] < self.h - 0.004) & moving_down
        if knock.any():
            k = knock.nonzero().squeeze(-1)
            side = torch.sign(across[k] + 1e-6)
            c, s = torch.cos(self.obj_yaw[k]), torch.sin(self.obj_yaw[k])
            push = 0.015 * side[:, None] * torch.stack([s, -c], -1)        # away from the fingers, across the object
            self.obj[k, :2] = self.obj[k, :2] + push
            self.obj_yaw[k] = wrap_pi(self.obj_yaw[k] + 0.3 * (torch.rand(len(k), generator=self.g, device=self.dev) - 0.5))
            self.knocks[k] += 1
            # the TCP cannot pass through the object: it rests on top of it
            self.tcp[k, 2] = torch.maximum(self.tcp[k, 2], self.h[k])
            self.tgt[k, 2] = torch.maximum(self.tgt[k, 2], self.h[k])
        ev["knocked"] = knock
        # drop: opening while holding
        drop = self.attached & open_cmd
        if drop.any():
            self.attached &= ~drop
            self.obj[drop, 2] = 0.0
            self.drops[drop] += 1
        ev["dropped"] = drop
        # carried object follows the TCP
        held = self.attached
        if held.any():
            self.obj[held, 0] = self.tcp[held, 0]
            self.obj[held, 1] = self.tcp[held, 1]
            self.obj[held, 2] = (self.tcp[held, 2] - 0.5 * self.h[held]).clamp(min=0.0)
            self.obj_yaw[held] = wrap_pi(self.obj_yaw[held] + wrap_pi(self.yaw[held] - prev_yaw[held]))
        self.ever_grasped |= self.attached
        self.success = self.attached & (self.obj[:, 2] >= LIFT_H)
        return ev

    # ------------------------------------------------------------------ observation
    def observe(self, last_action: torch.Tensor, time_frac: torch.Tensor) -> torch.Tensor:
        est = self.est_grasp_point()
        dyaw = self.est_yaw() - self.yaw
        return torch.cat([
            self.tcp - self.shoulder,                                  # 3
            self.tcp[:, 2:3],                                          # 1
            torch.sin(self.yaw)[:, None], torch.cos(self.yaw)[:, None],  # 2
            (self.grip / G_MAX)[:, None],                              # 1
            self.attached.float()[:, None],                            # 1
            est - self.tcp,                                            # 3
            torch.sin(2 * dyaw)[:, None], torch.cos(2 * dyaw)[:, None],  # 2 (jaws are symmetric: angle x 2)
            torch.stack([self.w, self.h, self.l], -1),                 # 3
            self.trunk_h[:, None],                                     # 1
            last_action,                                               # 5
            time_frac[:, None],                                        # 1
        ], -1)
