from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from integration.kinematics import ARM_JOINTS, LIFT_URDF, RobotModel
from policy.envs.rust_state import RustState, random_patches

GRID_H, GRID_W, PATCH_K, COARSE_H, COARSE_W = 32, 64, 9, 8, 16
ARM_INIT = [-2.035, -0.712, -0.963, -0.285, -1.288, 0.044]


@dataclass
class ScrubCfg:
    ctrl_dt: float = 4 / 120
    substeps: int = 4
    episode_s: float = 15.0
    base_height: float = 0.32
    brush_offset: tuple = (-0.025, 0.0, 0.0)
    brush_radius: float = 0.025
    pipe_length: float = 0.6
    radius_range: tuple = (0.12, 0.18)
    height_range: tuple = (1.05, 1.15)
    axis_dist_range: tuple = (0.45, 0.55)
    axial_range: tuple = (-0.1, 0.1)
    yaw_range: tuple = (-0.1745, 0.1745)
    lift_range: tuple = (0.38, 0.45)
    reach_range: tuple = (0.12, 0.62)
    shoulder_offset: tuple = (-0.01, 0.0, 0.3159)
    rust_theta_window: tuple = (0.9006, 0.35)
    theta_limits: tuple = (0.0, 2.0)
    spring_k_range: tuple = (1500.0, 4000.0)
    spring_k_nominal: float = 2500.0
    abrasion_coef_range: tuple = (0.5, 1.5)
    n_patches_range: tuple = (1, 4)
    prob_not_soaked: float = 0.15
    contact_snap: float = 0.01
    action_scale: float = 0.01
    force_set_mid: float = 25.0
    force_set_half: float = 20.0
    force_set_max: float = 38.0
    max_target_gap: float = 0.20
    new_patch_time_range: tuple = (3.0, 10.0)
    prob_new_patch: float = 1.0
    success_fraction: float = 0.05
    init_joint_noise: float = 0.1
    joint_tau: float = 0.05

    @property
    def max_steps(self) -> int:
        return int(math.ceil(self.episode_s / self.ctrl_dt))


def _u(g, lo, hi):
    return lo + (hi - lo) * float(torch.rand((), generator=g))


def make_scenario(seed: int, cfg: ScrubCfg | None = None) -> dict:
    c = cfg or ScrubCfg()
    g = torch.Generator().manual_seed(int(seed) % (2 ** 63))
    r = _u(g, *c.radius_range)
    yaw = _u(g, *c.yaw_range)
    o = (_u(g, *c.axial_range), -_u(g, *c.axis_dist_range), _u(g, *c.height_range))
    lift = _u(g, *c.lift_range)
    spring_k = _u(g, *c.spring_k_range)
    abr = _u(g, *c.abrasion_coef_range)
    soak = _u(g, 0.0, 25.0) if float(torch.rand((), generator=g)) < c.prob_not_soaked else 60.0
    q0 = [a + _u(g, -c.init_joint_noise, c.init_joint_noise) for a in ARM_INIT]
    th0 = c.rust_theta_window[0] + _u(g, -1.0, 1.0) * c.rust_theta_window[1]
    za0 = _u(g, -0.15, 0.15)
    rho0 = r + c.brush_radius + _u(g, 0.03, 0.12)
    ev = int(_u(g, *c.new_patch_time_range) / c.ctrl_dt) if float(torch.rand((), generator=g)) < c.prob_new_patch else -1
    geo = pipe_frame(torch.tensor([o]), torch.tensor([yaw]))
    reach = reachable_mask(c, geo, torch.tensor([r]), torch.tensor([lift]))
    rs = RustState(1, GRID_H, GRID_W, c.pipe_length, theta_window=c.rust_theta_window)
    rs.reset(radius=r, soak_time=soak, reachable=reach, n_patches=c.n_patches_range, abrasion_coef=abr, generator=g)
    patch = random_patches(1, GRID_H, GRID_W, torch.tensor([r]), c.pipe_length, n_min=1, n_max=1,
                           size_range=(0.03, 0.07), generator=g, theta_window=c.rust_theta_window) * reach.float()
    return {"seed": int(seed), "radius": r, "yaw": yaw, "origin": o, "lift": lift, "spring_k": spring_k,
            "abrasion_coef": abr, "soak": soak, "q0": q0, "tgt0": (za0, rho0, th0), "event_step": ev,
            "rust": rs.rust[0].clone(), "sprayed": rs.sprayed[0].clone(), "soak_time": rs.soak_time[0].clone(),
            "reachable": reach[0].clone(), "patch": patch[0].clone()}


def pipe_frame(origin: torch.Tensor, yaw: torch.Tensor) -> dict:
    a = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], -1)
    e = torch.stack([-torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)], -1)
    b = torch.tensor([0.0, 0.0, 1.0]).expand_as(a).clone()
    return {"o": origin.float(), "a": a.float(), "e": e.float(), "b": b}


def reachable_mask(c: ScrubCfg, geo: dict, r: torch.Tensor, lift: torch.Tensor) -> torch.Tensor:
    ox, oy, oz = c.shoulder_offset
    sh = torch.stack([torch.full_like(lift, ox), torch.full_like(lift, oy), c.base_height + oz + lift], -1)
    th = ((torch.arange(GRID_W).float() + 0.5) / GRID_W * 2 * math.pi).view(1, 1, -1)
    z = ((torch.arange(GRID_H).float() + 0.5) / GRID_H * c.pipe_length - c.pipe_length / 2).view(1, -1, 1)
    o, a, e, b = geo["o"], geo["a"], geo["e"], geo["b"]
    nrm = torch.cos(th)[..., None] * e.view(-1, 1, 1, 3) + torch.sin(th)[..., None] * b.view(-1, 1, 1, 3)
    p = o.view(-1, 1, 1, 3) + z[..., None] * a.view(-1, 1, 1, 3) + r.view(-1, 1, 1, 1) * nrm
    d = (p - sh.view(-1, 1, 1, 3)).norm(dim=-1)
    return (d >= c.reach_range[0]) & (d <= c.reach_range[1])


def _rot(axis: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    x, y, z = (float(v) for v in axis)
    c, s = torch.cos(ang), torch.sin(ang)
    t = 1 - c
    return torch.stack([
        torch.stack([t * x * x + c, t * x * y - s * z, t * x * z + s * y], -1),
        torch.stack([t * x * y + s * z, t * y * y + c, t * y * z - s * x], -1),
        torch.stack([t * x * z - s * y, t * y * z + s * x, t * z * z + c], -1)], -2)


class ArmFK:

    def __init__(self, urdf=LIFT_URDF, tip: str = "brush_tip"):
        m = RobotModel.from_urdf(urdf)
        self.chain = m.chain(tip)
        self.lower = torch.tensor([m.joints[j].lower for j in ARM_JOINTS]).float()
        self.upper = torch.tensor([m.joints[j].upper for j in ARM_JOINTS]).float()

    def forward(self, base: torch.Tensor, q: torch.Tensor, lift: torch.Tensor):
        n = q.shape[0]
        T = base.clone()
        axes, pts = [], []
        k = 0
        for j in self.chain:
            T = T @ torch.as_tensor(j.origin, dtype=torch.float32).expand(n, 4, 4)
            if j.kind == "prismatic":
                M = torch.eye(4).repeat(n, 1, 1)
                M[:, :3, 3] = torch.as_tensor(j.axis, dtype=torch.float32) * lift.view(-1, 1)
                T = T @ M
            elif j.kind in ("revolute", "continuous"):
                axes.append(T[:, :3, :3] @ torch.as_tensor(j.axis, dtype=torch.float32))
                pts.append(T[:, :3, 3].clone())
                M = torch.eye(4).repeat(n, 1, 1)
                M[:, :3, :3] = _rot(j.axis, q[:, k])
                T = T @ M
                k += 1
        return T, torch.stack(axes, 1), torch.stack(pts, 1)


def native_from_target(e, theta_t, z_t, press):
    c = e.cfg
    dev = e.pipe_r.device
    press = torch.as_tensor(press, device=dev).float()
    surf = e.pipe_r + c.brush_radius
    rho = torch.where(press > 0.05, surf, surf + 0.03)
    th = torch.as_tensor(theta_t, device=dev).float().clamp(c.theta_limits[0], c.theta_limits[1])
    des = e._from_cyl(torch.as_tensor(z_t, device=dev).float(), rho, th)
    za, ye, xb = e._to_pipe(des)
    tz, ty, tx = e._to_pipe(e.p_tgt)
    a = torch.stack([za - tz, ye - ty, xb - tx], -1) / c.action_scale
    a3 = (5.0 + 40.0 * press.clamp(0, 1) - c.force_set_mid) / c.force_set_half
    return torch.cat([a.clamp(-1, 1), a3.clamp(-1, 1)[:, None]], -1)


class KinematicScrub:

    def __init__(self, scenarios: list[dict], cfg: ScrubCfg | None = None, fk: ArmFK | None = None):
        c = self.cfg = cfg or ScrubCfg()
        self.fk = fk or ArmFK()
        self.scen = scenarios
        N = self.N = len(scenarios)
        f = lambda k: torch.tensor([s[k] for s in scenarios], dtype=torch.float32)
        self.pipe_r, self.spring_k, self.lift = f("radius"), f("spring_k"), f("lift")
        geo = pipe_frame(torch.tensor([s["origin"] for s in scenarios]), f("yaw"))
        self.pipe_o, self.pipe_a, self.pipe_e, self.pipe_b = geo["o"], geo["a"], geo["e"], geo["b"]
        self.rust = RustState(N, GRID_H, GRID_W, c.pipe_length, theta_window=c.rust_theta_window)
        self.rust.radius[:] = self.pipe_r
        self.rust.abrasion_coef[:] = f("abrasion_coef")
        self.rust.rust = torch.stack([s["rust"] for s in scenarios]).float()
        self.rust.sprayed = torch.stack([s["sprayed"] for s in scenarios]).bool()
        self.rust.soak_time = torch.stack([s["soak_time"] for s in scenarios]).float()
        self.rust.reachable = torch.stack([s["reachable"] for s in scenarios]).bool()
        self.rust.initial_total = self.rust.rust.sum(dim=(1, 2)).clamp_min(1e-6)
        self.patches = torch.stack([s["patch"] for s in scenarios]).float()
        self.event_step = torch.tensor([s["event_step"] for s in scenarios], dtype=torch.long)
        self.base = torch.eye(4).repeat(N, 1, 1)
        self.base[:, 2, 3] = c.base_height
        self.boff = torch.tensor(c.brush_offset).float()
        self.q = torch.maximum(torch.minimum(torch.tensor([s["q0"] for s in scenarios]).float(), self.fk.upper),
                               self.fk.lower)
        self.qd = torch.zeros(N, 6)
        tg = torch.tensor([s["tgt0"] for s in scenarios]).float()
        self.p_tgt = self._from_cyl(tg[:, 0], tg[:, 1], tg[:, 2])
        self.p_cmd = self.p_tgt.clone()
        self.f_set = torch.full((N,), c.force_set_mid)
        self.actions = torch.zeros(N, 4)
        z1 = lambda: torch.zeros(N)
        self.force, self.gap, self.theta, self.axial, self.max_force, self.removed = z1(), z1(), z1(), z1(), z1(), z1()
        self.step_i = torch.zeros(N, dtype=torch.long)
        self.event_fired = torch.zeros(N, dtype=torch.bool)
        self.success = torch.zeros(N, dtype=torch.bool)
        self.done = torch.zeros(N, dtype=torch.bool)
        self.clean_step = torch.full((N,), -1, dtype=torch.long)
        self.end_step = torch.full((N,), -1, dtype=torch.long)
        self.final_frac = torch.ones(N)
        self.brush_w, self.brush_v = self._brush()[0], torch.zeros(N, 3)
        self._update_contact(self.brush_w, torch.zeros(N, 3), record=False)

    def _to_pipe(self, p):
        d = p - self.pipe_o
        return (d * self.pipe_a).sum(-1), (d * self.pipe_e).sum(-1), (d * self.pipe_b).sum(-1)

    def _vec_to_pipe(self, v):
        return torch.stack([(v * self.pipe_a).sum(-1), (v * self.pipe_e).sum(-1), (v * self.pipe_b).sum(-1)], -1)

    def _from_cyl(self, za, rho, th):
        n = torch.cos(th)[:, None] * self.pipe_e + torch.sin(th)[:, None] * self.pipe_b
        return self.pipe_o + za[:, None] * self.pipe_a + rho[:, None] * n

    def _brush(self, q=None):
        T, axes, pts = self.fk.forward(self.base, self.q if q is None else q, self.lift)
        p = T[:, :3, 3] + T[:, :3, :3] @ self.boff
        return p, axes, pts

    def brush_theta_z(self):
        return self.theta.clone(), self.axial.clone()

    def native_from_target(self, theta_t, z_t, press):
        return native_from_target(self, theta_t, z_t, press)

    def step(self, actions: torch.Tensor):
        c = self.cfg
        a = torch.as_tensor(actions).float().clamp(-1, 1)
        live = ~self.done
        a = torch.where(live[:, None], a, torch.zeros_like(a))
        self.actions = a
        za, ye, xb = self._to_pipe(self.p_tgt)
        za = za + a[:, 0] * c.action_scale
        ye = ye + a[:, 1] * c.action_scale
        xb = xb + a[:, 2] * c.action_scale
        rho = torch.sqrt(ye ** 2 + xb ** 2).clamp_min(1e-6)
        th = torch.atan2(xb, ye).clamp(c.theta_limits[0], c.theta_limits[1])
        surf = self.pipe_r + c.brush_radius
        rho = torch.minimum(torch.maximum(rho, surf - 0.03), surf + c.max_target_gap)
        half = c.pipe_length / 2 + 0.05
        za = za.clamp(-half, half)
        self.p_tgt = torch.where(live[:, None], self._from_cyl(za, rho, th), self.p_tgt)
        self.f_set = (c.force_set_mid + c.force_set_half * a[:, 3]).clamp(max=c.force_set_max)
        rho_cmd = torch.where(rho < surf + c.contact_snap, surf - self.f_set / c.spring_k_nominal, rho)
        self.p_cmd = self._from_cyl(za, rho_cmd, th)
        sub_dt = c.ctrl_dt / c.substeps
        alpha = 1.0 - math.exp(-sub_dt / c.joint_tau)
        q_start = self.q.clone()
        prev = self.brush_w
        for _ in range(c.substeps):
            p, axes, pts = self._brush()
            Jv = torch.cross(axes, (p[:, None, :] - pts), dim=-1).transpose(1, 2)
            err = self.p_cmd - p
            n = err.norm(dim=-1, keepdim=True)
            err = err * torch.clamp(0.05 / n.clamp_min(1e-6), max=1.0)
            JJt = Jv @ Jv.transpose(1, 2) + 0.05 ** 2 * torch.eye(3)
            dq = (Jv.transpose(1, 2) @ torch.linalg.solve(JJt, err.unsqueeze(-1))).squeeze(-1)
            q_t = torch.maximum(torch.minimum(self.q + dq, self.fk.upper), self.fk.lower)
            prev = p
            self.q = torch.where(live[:, None], self.q + alpha * (q_t - self.q), self.q)
        self.qd = (self.q - q_start) / c.ctrl_dt
        brush, _, _ = self._brush()
        v = (brush - prev) / sub_dt
        self.step_i = self.step_i + live.long()
        self._update_contact(brush, v, record=True, live=live)
        frac = self.rust.fraction_remaining()
        fire = live & (self.event_step >= 0) & ~self.event_fired & (
            (self.step_i >= self.event_step) | (frac < c.success_fraction))
        ids = fire.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            patch = self.patches[ids]
            self.rust.rust[ids] = torch.maximum(self.rust.rust[ids], patch)
            self.rust.spray(self.rust.dilate(patch > self.rust.p.rust_eps, 2), env_ids=ids, soak_time=60.0)
            self.rust.initial_total[ids] = torch.maximum(self.rust.initial_total[ids], self.rust.rust[ids].sum(dim=(1, 2)))
            self.event_fired[ids] = True
            frac = self.rust.fraction_remaining()
        succ = live & (frac < c.success_fraction)
        newly = succ & (self.clean_step < 0)
        self.clean_step[newly] = self.step_i[newly]
        self.success |= succ
        trunc = live & (self.step_i >= c.max_steps - 1)
        ended = succ | trunc
        self.end_step[ended] = self.step_i[ended]
        self.final_frac = torch.where(ended, frac, self.final_frac)
        self.done |= ended
        return self.done.clone()

    def _update_contact(self, brush, v, record: bool, live=None):
        c = self.cfg
        live = torch.ones(self.N, dtype=torch.bool) if live is None else live
        self.brush_w, self.brush_v = brush, v
        za, ye, xb = self._to_pipe(brush)
        rho = torch.sqrt(ye ** 2 + xb ** 2)
        th = torch.atan2(xb, ye)
        self.theta, self.axial = th, za
        pen = (self.pipe_r + c.brush_radius - rho).clamp_min(0.0)
        self.gap = rho - self.pipe_r - c.brush_radius
        on_pipe = za.abs() < c.pipe_length / 2 + 0.5
        self.force = self.spring_k * pen * on_pipe * live
        if not record:
            return
        n = torch.cos(th)[:, None] * self.pipe_e + torch.sin(th)[:, None] * self.pipe_b
        v_t = v - (v * n).sum(-1, keepdim=True) * n
        slide = v_t.norm(dim=-1)
        crad = (0.012 + torch.sqrt(2 * c.brush_radius * pen)).clamp(max=c.brush_radius)
        uv = self.rust.surface_to_uv(th, za)
        before = self.rust.rust.clone()
        self.removed = self.rust.abrade(uv, crad, self.force, slide, c.ctrl_dt)
        self.rust.rust = torch.where(live.view(-1, 1, 1), self.rust.rust, before)
        soak_before = self.rust.soak_time.clone()
        self.rust.step_soak(c.ctrl_dt)
        self.rust.soak_time = torch.where(live.view(-1, 1, 1), self.rust.soak_time, soak_before)
        self.max_force = torch.where(live, torch.maximum(self.max_force, self.force), self.max_force)

    def obs(self) -> torch.Tensor:
        c = self.cfg
        za, ye, xb = self._to_pipe(self.brush_w)
        brush_p = torch.stack([za, ye, xb], -1)
        brush_v = self._vec_to_pipe(self.brush_v)
        tz, ty, tx = self._to_pipe(self.p_tgt)
        tgt_rel = torch.stack([tz, ty, tx], -1) - brush_p
        uv = self.rust.surface_to_uv(self.theta, self.axial)
        near, dist = self.rust.nearest_rust(uv)
        o = torch.cat([
            self.q, self.qd * 0.1, brush_p * 4.0, brush_v, self.gap[:, None] * 10.0,
            torch.sin(self.theta)[:, None], torch.cos(self.theta)[:, None], (self.axial / c.pipe_length)[:, None],
            tgt_rel * 10.0, (self.force / 40.0)[:, None], (self.f_set / 40.0)[:, None], (self.pipe_r * 10.0)[:, None],
            self.rust.local_patch(uv, PATCH_K).flatten(1), self.rust.local_sprayed(uv, PATCH_K).flatten(1),
            self.rust.coarse_map(COARSE_H, COARSE_W).flatten(1), near * 10.0, dist[:, None] * 10.0,
            self.rust.fraction_remaining()[:, None], self.rust.last_unsprayed_contact[:, None], self.actions], -1)
        return torch.nan_to_num(o).clamp(-10.0, 10.0)

    def episode_stats(self) -> list[dict]:
        dt = self.cfg.ctrl_dt
        out = []
        for i in range(self.N):
            end = int(self.end_step[i]) if self.end_step[i] >= 0 else int(self.step_i[i])
            t_end = max(end, 1) * dt
            frac = float(self.final_frac[i]) if self.done[i] else float(self.rust.fraction_remaining()[i])
            out.append({"seed": self.scen[i]["seed"], "success": bool(self.success[i]),
                        "time_to_clean_s": float(self.clean_step[i]) * dt if self.success[i] else None,
                        "peak_force_N": float(self.max_force[i]), "over_40N": bool(self.max_force[i] > 40.0),
                        "fraction_removed": 1.0 - frac, "episode_s": t_end,
                        "removed_pct_per_s": 100.0 * (1.0 - frac) / t_end, "new_patch": bool(self.event_fired[i])})
        return out
