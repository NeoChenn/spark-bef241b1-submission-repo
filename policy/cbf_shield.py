from __future__ import annotations

import torch


def rls_step(k, P, s, F, lam=0.98, s_min=5e-4, p_max=1e8, k_min=300.0):
    on = (s > s_min) & (F > 0.5)
    g = P * s / (lam + P * s * s)
    k_new = torch.where(on, (k + g * (F - k * s)).clamp_min(k_min), k)
    P_new = torch.where(on, ((1.0 - g * s) * P / lam).clamp(max=p_max), P)
    return k_new, P_new


def cbf_bound(F, s, k, w, alpha=1.0, f_max=40.0, gamma=0.5):
    f_lim = f_max - (1.0 - gamma) * (f_max - F)
    return s + (f_lim / k.clamp_min(1.0) - s - w) / alpha


class CBFShield:
    def __init__(self, n, k0=2500.0, f_max=40.0, gamma=0.5, margin=2.0, alpha=1.0, lam=0.98, p0=1e7,
                 retract=0.05, trend=1.0, device="cpu"):
        self.n, self.k0, self.p0, self.trend = n, k0, p0, trend
        self.f_max, self.gamma, self.margin, self.alpha, self.lam, self.retract = f_max, gamma, margin, alpha, lam, retract
        z = lambda: torch.zeros(n, device=device)
        self.k_hat, self.P = torch.full((n,), k0, device=device), torch.full((n,), p0, device=device)
        self.w_hat, self.w_last, self.s_prev, self.cmd_prev = z(), z(), z(), z()
        self.valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.active = torch.zeros(n, dtype=torch.bool, device=device)
        self.steps, self.active_steps = 0, 0

    def reset(self, mask):
        m = mask.bool()
        self.k_hat[m], self.P[m] = self.k0, self.p0
        self.w_hat[m], self.w_last[m] = 0.0, 0.0
        self.valid[m] = False

    def observe(self, F, s):
        self.k_hat, self.P = rls_step(self.k_hat, self.P, s, F, self.lam)
        w = (s - self.s_prev) - self.alpha * (self.cmd_prev - self.s_prev)
        w = torch.where(self.valid, w, torch.zeros_like(w))
        self.w_hat = w + self.trend * (w - self.w_last) * self.valid
        self.w_last = w

    def filter(self, F, s, cmd):
        smax = cbf_bound(F, s, self.k_hat, self.w_hat, self.alpha, self.f_max - self.margin, self.gamma)
        out = torch.minimum(cmd, smax.clamp_min(s - self.retract))
        self.active = out < cmd
        self.s_prev, self.cmd_prev = s.clone(), out.clone()
        self.valid[:] = True
        self.steps += self.n
        self.active_steps += int(self.active.sum())
        return out

    def predicted_force(self, s, cmd):
        return self.k_hat * (s + self.alpha * (cmd - s) + self.w_hat)

    def duty(self):
        return self.active_steps / max(self.steps, 1)


def attach(base, shield):
    orig = base._pre_physics_step

    def step(actions):
        orig(actions)
        c = base.cfg
        za, ye, xb = base._to_pipe(base.p_cmd)
        rho = torch.sqrt(ye ** 2 + xb ** 2)
        th = torch.atan2(xb, ye)
        surf = base.pipe_r + c.brush_radius
        s = -base.gap
        F = base.force.clone()
        shield.observe(F, s)
        safe = shield.filter(F, s, surf - rho)
        base.p_cmd[:] = base._from_cyl(za, surf - safe, th)

    base._pre_physics_step = step
    return shield
