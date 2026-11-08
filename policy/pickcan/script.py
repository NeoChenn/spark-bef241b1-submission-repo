from __future__ import annotations

import torch

from policy.pickcan.core import TOTE_CENTER

ABOVE, DESCEND, LIFT, CARRY, RELEASE, DONE = range(6)
NAMES = ("APPROACH", "DESCEND", "LIFT", "CARRY", "RELEASE", "DONE")
HOVER = 0.10
CARRY_Z = 0.72
READY = (0.1, 0.2, 0.6)


class ScriptedPicker:
    def __init__(self, n: int, device, dt: float = 1 / 30, scale: float = 0.012, descend_timeout: float = 2.5,
                 press: float = 0.004, slow: float = 0.35):
        self.n, self.dev, self.dt, self.scale = n, device, dt, scale
        self.timeout, self.press, self.slow = descend_timeout, press, slow
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

    def act(self, tip: torch.Tensor, p_tgt: torch.Tensor, grasp: torch.Tensor, sealed: torch.Tensor) -> torch.Tensor:
        dev = self.dev
        self.t_phase += self.dt
        ph = self.phase
        above = grasp + torch.tensor([0.0, 0.0, HOVER], device=dev)
        tote = torch.tensor([TOTE_CENTER[0], TOTE_CENTER[1], CARRY_Z], device=dev).expand_as(tip)
        ready = torch.tensor(READY, device=dev).expand_as(tip)
        dist = lambda a, b: (a - b).norm(dim=-1)

        m = (ph == ABOVE) & (dist(tip, above) < 0.015)
        self._go(m, DESCEND)
        m = (ph == DESCEND) & sealed
        self.hold[m] = tip[m]
        self._go(m, LIFT)
        m = (ph == DESCEND) & ~sealed & (self.t_phase > self.timeout)
        self.tries[m] += 1
        self._go(m, ABOVE)
        m = (ph == LIFT) & sealed & (tip[:, 2] > CARRY_Z - 0.04)
        self._go(m, CARRY)
        m = ((ph == LIFT) | (ph == CARRY)) & ~sealed
        self._go(m, ABOVE)
        m = (ph == CARRY) & (dist(tip[:, :2], tote[:, :2]) < 0.02)
        self._go(m, RELEASE)
        m = (ph == RELEASE) & (self.t_phase > 0.5)
        self._go(m, DONE)
        ph = self.phase

        target = above.clone()
        desc = grasp - torch.tensor([0.0, 0.0, self.press], device=dev)
        target = torch.where((ph == DESCEND)[:, None], desc, target)
        lift_t = torch.cat([self.hold[:, :2], torch.full_like(self.hold[:, :1], CARRY_Z)], -1)
        target = torch.where((ph == LIFT)[:, None], lift_t, target)
        target = torch.where((ph == CARRY)[:, None], tote, target)
        target = torch.where((ph == RELEASE)[:, None], tote, target)
        target = torch.where((ph == DONE)[:, None], ready, target)
        gain = torch.where(ph == DESCEND, self.slow, 1.0)
        a = ((target - p_tgt) / self.scale).clamp(-1.0, 1.0) * gain[:, None]
        suction = torch.where((ph == DESCEND) | (ph == LIFT) | (ph == CARRY), 1.0, -1.0)
        lift = torch.full_like(suction, -1.0)
        return torch.cat([a, suction[:, None], lift[:, None]], -1)
