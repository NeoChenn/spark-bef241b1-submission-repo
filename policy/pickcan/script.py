from __future__ import annotations

import math

import torch

from policy.pickcan.core import TOTE_CENTER

ABOVE, DESCEND, LIFT, CARRY, LOWER, RELEASE, DONE = range(7)
NAMES = ("APPROACH", "DESCEND", "LIFT", "CARRY", "LOWER", "RELEASE", "DONE")
HOVER = 0.10
CARRY_Z = 0.72
SHOULDER = (-0.01, 0.0)
ARC_R = 0.30
ARC_STEP = 0.6
ARC_SWITCH = 0.6
LIFT_UP = 0.0
RELEASE_Z = 0.64
LIFT_CLEAR = 0.25


class ScriptedPicker:
    def __init__(self, n: int, device, dt: float = 1 / 30, scale: float = 0.012, descend_timeout: float = 2.5,
                 press: float = 0.004, slow: float = 0.35, lift_up: float = LIFT_UP):
        self.n, self.dev, self.dt, self.scale = n, device, dt, scale
        self.timeout, self.press, self.slow, self.lift_up = descend_timeout, press, slow, lift_up
        self.phase = torch.zeros(n, dtype=torch.long, device=device)
        self.t_phase = torch.zeros(n, device=device)
        self.hold = torch.zeros(n, 3, device=device)
        self.tries = torch.zeros(n, device=device)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.phase[ids] = ABOVE
        self.t_phase[ids] = 0.0
        self.tries[ids] = 0.0

    def _go(self, mask, phase):
        self.phase[mask] = phase
        self.t_phase[mask] = 0.0

    def act(self, tip, p_tgt, grasp, sealed, lift=None) -> torch.Tensor:
        dev = self.dev
        lift = torch.zeros(tip.shape[0], device=dev) if lift is None else lift
        self.t_phase += self.dt
        ph = self.phase
        above = grasp + torch.tensor([0.0, 0.0, HOVER], device=dev)
        carry_z = CARRY_Z + lift
        tote = torch.stack([torch.full_like(lift, TOTE_CENTER[0]), torch.full_like(lift, TOTE_CENTER[1]), carry_z], -1)
        dist = lambda a, b: (a - b).norm(dim=-1)

        m = (ph == ABOVE) & (dist(tip, above) < 0.015)
        self._go(m, DESCEND)
        m = (ph == DESCEND) & sealed
        self.hold[m] = tip[m]
        self._go(m, LIFT)
        m = (ph == DESCEND) & ~sealed & (self.t_phase > self.timeout)
        self.tries[m] += 1
        self._go(m, ABOVE)
        m = (ph == LIFT) & sealed & (tip[:, 2] > self.hold[:, 2] + LIFT_CLEAR - 0.02)
        self._go(m, CARRY)
        m = ((ph == LIFT) | (ph == CARRY) | (ph == LOWER)) & ~sealed
        self._go(m, ABOVE)
        m = (ph == CARRY) & (dist(tip[:, :2], tote[:, :2]) < 0.02)
        self._go(m, LOWER)
        m = (ph == LOWER) & (lift < 0.01) & (tip[:, 2] < RELEASE_Z + lift + 0.015)
        self._go(m, RELEASE)
        m = (ph == RELEASE) & (self.t_phase > 0.5)
        self._go(m, DONE)
        ph = self.phase

        th = torch.atan2(tip[:, 1] - SHOULDER[1], tip[:, 0] - SHOULDER[0])
        dth = torch.remainder(math.pi - th, 2 * math.pi)
        a_next = th + dth.clamp(0.0, ARC_STEP)
        arc = torch.stack([SHOULDER[0] + ARC_R * torch.cos(a_next), SHOULDER[1] + ARC_R * torch.sin(a_next), carry_z], -1)
        carry = torch.where(((dth > ARC_SWITCH) & (dth < 2 * math.pi - 0.2))[:, None], arc, tote)

        target = above.clone()
        target = torch.where((ph == DESCEND)[:, None], grasp - torch.tensor([0.0, 0.0, self.press], device=dev), target)
        lift_t = self.hold + torch.tensor([0.0, 0.0, LIFT_CLEAR], device=dev)
        target = torch.where((ph == LIFT)[:, None], lift_t, target)
        target = torch.where((ph == CARRY)[:, None], carry, target)
        drop = torch.cat([tote[:, :2], (RELEASE_Z + lift)[:, None]], -1)
        target = torch.where(((ph == LOWER) | (ph == RELEASE) | (ph == DONE))[:, None], drop, target)
        gain = torch.where((ph == DESCEND) | (ph == LOWER), self.slow, 1.0)
        a = ((target - p_tgt) / self.scale).clamp(-1.0, 1.0) * gain[:, None]
        suction = torch.where((ph == DESCEND) | (ph == LIFT) | (ph == CARRY) | (ph == LOWER), 1.0, -1.0)
        up = (ph == LIFT) | (ph == CARRY)
        lift_a = torch.where(up, ((self.lift_up - lift) / 0.004).clamp(-1.0, 1.0), torch.full_like(lift, -1.0))
        return torch.cat([a, suction[:, None], lift_a[:, None]], -1)
