"""Batched 2D top-down patrol sim for training the path-following policy without Isaac Sim.

World per env: a predefined path (start bin at s=0, end bin at s=L), pipe walls on both sides with
random narrowings, up to K round objects lying on or near the path, a 2D lidar, and a unicycle base
whose speed lags the command (stands in for the Go2 locomotion controller tracking (vx, wz)).

Everything is torch, shaped [N, ...], and resets per env.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from patrol import features


@dataclass
class SimCfg:
    num_envs: int = 1024
    dt: float = 0.1
    path_len: float = 20.0
    ds: float = 0.1
    max_curv: float = 0.8
    half_width: tuple = (0.55, 1.1)
    min_half_width: float = 0.42
    n_narrow: int = 3
    narrow_depth: float = 0.35
    n_obstacles: int = 3
    p_obstacle: float = 0.7
    obstacle_r: tuple = (0.04, 0.25)
    min_gap: float = 0.6
    robot_r: float = 0.22
    lidar_beams: int = 72
    lidar_range: float = features.LIDAR_RANGE
    lidar_noise: tuple = (0.0, 0.03)
    lidar_dropout: float = 0.02
    v_max: float = features.V_MAX
    v_min: float = features.V_MIN
    w_max: float = features.W_MAX
    lag: tuple = (0.3, 0.8)
    p_push: float = 0.01
    push_speed: tuple = (0.3, 0.8)
    max_steps: int = 450
    window: int = 50
    wall_stride: int = 2
    start_frac: float = 0.3
    w_progress: float = 3.0
    w_lat: float = 0.4
    w_head: float = 0.05
    w_clear: float = 1.0
    clear_band: float = 0.15
    w_rate: float = 0.05
    w_time: float = 0.02
    r_collision: float = -10.0
    r_goal: float = 10.0
    goal_tol: float = 0.35
    device: str = "cpu"
    seed: int | None = None


class PatrolSim:
    def __init__(self, cfg: SimCfg = SimCfg()):
        self.cfg = c = cfg
        self.dev = torch.device(c.device)
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(c.seed if c.seed is not None else torch.seed() % (2**31))
        self.N = c.num_envs
        self.M = int(round(c.path_len / c.ds)) + 1
        B = c.lidar_beams
        self.beam_angles = -math.pi + (torch.arange(B, device=self.dev) + 0.5) * (2 * math.pi / B)
        self.look_idx = torch.tensor([round(d / c.ds) for d in features.LOOKAHEAD], device=self.dev)
        z = lambda *s: torch.zeros(*s, device=self.dev)
        N, M, K = self.N, self.M, c.n_obstacles
        self.P, self.h, self.nrm = z(N, M, 2), z(N, M), z(N, M, 2)
        self.wl, self.wr = z(N, M), z(N, M)
        self.obs_c, self.obs_r = z(N, K, 2), z(N, K)
        self.obs_on = torch.zeros(N, K, dtype=torch.bool, device=self.dev)
        self.pos, self.yaw, self.v, self.w = z(N, 2), z(N), z(N), z(N)
        self.idx = torch.zeros(N, dtype=torch.long, device=self.dev)
        self.steps = torch.zeros(N, dtype=torch.long, device=self.dev)
        self.last_act, self.push = z(N, 2), z(N, 2)
        self.alpha, self.noise = z(N), z(N)
        self.reset(torch.arange(N, device=self.dev))

    # ------------------------------------------------------------------ random helpers
    def _u(self, n, lo, hi, *extra):
        return lo + (hi - lo) * torch.rand(n, *extra, generator=self.gen, device=self.dev)

    # ------------------------------------------------------------------ world generation
    def _make_world(self, ids):
        c, n, M = self.cfg, len(ids), self.M
        knots = torch.randn(n, 1, M // 10 + 2, generator=self.gen, device=self.dev) * c.max_curv * 0.7
        k = F.interpolate(knots, size=M, mode="linear", align_corners=True)[:, 0]
        k = (k * self._u(n, 0.0, 1.0, 1)).clamp(-c.max_curv, c.max_curv)
        h = self._u(n, -math.pi, math.pi, 1) + torch.cumsum(k * c.ds, 1) - k[:, :1] * c.ds
        step = torch.stack([h.cos(), h.sin()], -1) * c.ds
        P = torch.cumsum(step, 1) - step[:, :1]
        nrm = torch.stack([-h.sin(), h.cos()], -1)
        s = torch.arange(M, device=self.dev) * c.ds
        wl = self._u(n, *c.half_width, 1).expand(n, M).clone()
        wr = self._u(n, *c.half_width, 1).expand(n, M).clone()
        for _ in range(c.n_narrow):
            centre = self._u(n, 0.0, c.path_len, 1)
            bump = self._u(n, 0.0, c.narrow_depth, 1) * torch.exp(-0.5 * ((s - centre) / 0.3) ** 2)
            side = torch.randint(0, 3, (n, 1), generator=self.gen, device=self.dev)
            wl = wl - bump * (side != 1)
            wr = wr - bump * (side != 0)
        inner = 0.9 / k.abs().clamp(min=1e-3)
        wl = torch.minimum(wl, torch.where(k > 0, inner, wl)).clamp(min=c.min_half_width)
        wr = torch.minimum(wr, torch.where(k < 0, inner, wr)).clamp(min=c.min_half_width)

        K = c.n_obstacles
        oi = (self._u(n, 3.0, c.path_len - 2.0, K) / c.ds).long().clamp(0, M - 1)
        owl, owr = wl.gather(1, oi), wr.gather(1, oi)
        r = self._u(n, *c.obstacle_r, K)
        off = self._u(n, -1.0, 1.0, K) * 0.6 * torch.minimum(owl, owr)
        gap_l = owl - off - r
        gap_r = owr + off - r
        need = (c.min_gap - torch.maximum(gap_l, gap_r)).clamp(min=0)
        off = torch.where(gap_l >= gap_r, off - need, off + need)
        on = torch.rand(n, K, generator=self.gen, device=self.dev) < c.p_obstacle
        oc = P.gather(1, oi[..., None].expand(n, K, 2)) + nrm.gather(1, oi[..., None].expand(n, K, 2)) * off[..., None]

        self.P[ids], self.h[ids], self.nrm[ids] = P, h, nrm
        self.wl[ids], self.wr[ids] = wl, wr
        self.obs_c[ids], self.obs_r[ids], self.obs_on[ids] = oc, r, on

    def reset(self, ids):
        if len(ids) == 0:
            return
        c, n = self.cfg, len(ids)
        self._make_world(ids)
        start = (self._u(n, 0.0, c.start_frac) * (self.M - 1) * (torch.rand(n, generator=self.gen, device=self.dev) < 0.5)).long()
        p0 = self.P[ids, start]
        self.pos[ids] = p0 + self.nrm[ids, start] * self._u(n, -0.15, 0.15, 1)
        self.yaw[ids] = self.h[ids, start] + self._u(n, -0.4, 0.4)
        self.v[ids] = 0.0
        self.w[ids] = 0.0
        self.idx[ids] = start
        self.steps[ids] = 0
        self.last_act[ids] = 0.0
        self.push[ids] = 0.0
        self.alpha[ids] = self._u(n, *c.lag)
        self.noise[ids] = self._u(n, *c.lidar_noise)

    def reverse(self, ids):
        """Turn the path around (shuttle back): start and end bins swap, left and right walls swap."""
        self.P[ids] = self.P[ids].flip(1)
        self.h[ids] = (self.h[ids].flip(1) + math.pi + math.pi) % (2 * math.pi) - math.pi
        self.nrm[ids] = -self.nrm[ids].flip(1)
        self.wl[ids], self.wr[ids] = self.wr[ids].flip(1), self.wl[ids].flip(1)
        self.idx[ids] = self.M - 1 - self.idx[ids]
        self._update_idx(window=(-30, 30))

    # ------------------------------------------------------------------ geometry
    def _update_idx(self, window=(-5, 20)):
        offs = torch.arange(window[0], window[1] + 1, device=self.dev)
        cand = (self.idx[:, None] + offs).clamp(0, self.M - 1)
        pts = self.P.gather(1, cand[..., None].expand(-1, -1, 2))
        d2 = ((pts - self.pos[:, None]) ** 2).sum(-1)
        self.idx = cand.gather(1, d2.argmin(1, keepdim=True))[:, 0]

    def _at(self, t, idx):
        return t.gather(1, idx[:, None])[:, 0] if t.dim() == 2 else t.gather(1, idx[:, None, None].expand(-1, 1, 2))[:, 0]

    def path_errors(self):
        p, nrm, h = self._at(self.P, self.idx), self._at(self.nrm, self.idx), self._at(self.h, self.idx)
        lat = ((self.pos - p) * nrm).sum(-1)
        head = torch.remainder(self.yaw - h + math.pi, 2 * math.pi) - math.pi
        return lat, head

    def lookahead_robot(self):
        li = (self.idx[:, None] + self.look_idx).clamp(max=self.M - 1)
        pts = self.P.gather(1, li[..., None].expand(-1, -1, 2)) - self.pos[:, None]
        cy, sy = self.yaw.cos()[:, None], self.yaw.sin()[:, None]
        return torch.stack([cy * pts[..., 0] + sy * pts[..., 1], -sy * pts[..., 0] + cy * pts[..., 1]], -1)

    def dist_to_goal(self):
        return (self.M - 1 - self.idx).float() * self.cfg.ds

    def lidar(self, beams: torch.Tensor | None = None, noise: bool = True):
        """Ranges [N, B] for beam angles (robot frame) `beams` (default: training beams)."""
        c = self.cfg
        ang = self.beam_angles if beams is None else beams
        th = self.yaw[:, None] + ang[None]
        d = torch.stack([th.cos(), th.sin()], -1)[:, :, None]
        p = self.pos[:, None, None]
        wi = (self.idx[:, None] + torch.arange(-c.window, c.window + 1, c.wall_stride, device=self.dev)).clamp(0, self.M - 1)
        segs = []
        for W in (self.P + self.nrm * self.wl[..., None], self.P - self.nrm * self.wr[..., None]):
            pts = W.gather(1, wi[..., None].expand(-1, -1, 2))
            segs.append((pts[:, :-1], pts[:, 1:]))
        a = torch.cat([s[0] for s in segs], 1)[:, None]
        e = torch.cat([s[1] for s in segs], 1)[:, None] - a
        cross = lambda x, y: x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
        den = cross(d, e)
        ap = a - p
        safe = torch.where(den.abs() > 1e-9, den, torch.full_like(den, 1e-9))
        t = cross(ap, e) / safe
        u = cross(ap, d) / safe
        ok = (den.abs() > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
        t_wall = torch.where(ok, t, torch.full_like(t, math.inf)).amin(-1)

        f = p - self.obs_c[:, None]
        bq = (f * d).sum(-1)
        cq = (f * f).sum(-1) - self.obs_r[:, None] ** 2
        disc = bq**2 - cq
        tc = -bq - disc.clamp(min=0).sqrt()
        okc = (disc >= 0) & (tc > 0) & self.obs_on[:, None]
        t_obj = torch.where(okc, tc, torch.full_like(tc, math.inf)).amin(-1)

        r = torch.minimum(t_wall, t_obj).clamp(max=c.lidar_range)
        if noise:
            r = r + torch.randn(r.shape, generator=self.gen, device=self.dev) * self.noise[:, None]
            drop = torch.rand(r.shape, generator=self.gen, device=self.dev) < c.lidar_dropout
            r = torch.where(drop, torch.full_like(r, c.lidar_range), r)
        return r.clamp(0.0, c.lidar_range)

    def obstacle_world(self):
        return self.obs_c, self.obs_r, self.obs_on

    def clearance(self):
        lat, _ = self.path_errors()
        wall = torch.minimum(self._at(self.wl, self.idx) - lat, self._at(self.wr, self.idx) + lat)
        dob = (self.pos[:, None] - self.obs_c).norm(dim=-1) - self.obs_r
        dob = torch.where(self.obs_on, dob, torch.full_like(dob, math.inf)).amin(-1)
        return torch.minimum(wall, dob) - self.cfg.robot_r

    # ------------------------------------------------------------------ RL interface
    def observe(self):
        lat, head = self.path_errors()
        return features.build_obs_torch(
            features.pool_sectors_torch(self.lidar(), self.beam_angles),
            lat, head, self.lookahead_robot(), self.v, self.w, self.last_act, self.dist_to_goal())

    def remove_obstacle(self, env_id: int, k: int):
        """Object picked up: it no longer exists for lidar or collisions."""
        self.obs_on[env_id, k] = False

    def step(self, action: torch.Tensor, auto_reset: bool = True):
        c = self.cfg
        a = action.clamp(-1, 1)
        v_cmd, w_cmd = features.action_to_cmd_torch(a)
        self.v += self.alpha * (v_cmd - self.v)
        self.w += self.alpha * (w_cmd - self.w)
        self.yaw = torch.remainder(self.yaw + self.w * c.dt + math.pi, 2 * math.pi) - math.pi
        fwd = torch.stack([self.yaw.cos(), self.yaw.sin()], -1)
        self.pos = self.pos + fwd * (self.v * c.dt)[:, None] + self.push * c.dt
        self.push *= 0.8
        kick = torch.rand(self.N, generator=self.gen, device=self.dev) < c.p_push
        if kick.any():
            ang = self._u(self.N, -math.pi, math.pi)
            spd = self._u(self.N, *c.push_speed)
            self.push = torch.where(kick[:, None], torch.stack([ang.cos(), ang.sin()], -1) * spd[:, None], self.push)
        old = self.idx.clone()
        self._update_idx()
        self.steps += 1

        lat, head = self.path_errors()
        clear = self.clearance()
        progress = (self.idx - old).float() * c.ds
        rate = ((a - self.last_act) ** 2).sum(-1)
        self.last_act = a
        collision = clear < 0
        goal = ((self.pos - self.P[:, -1]).norm(dim=-1) < c.goal_tol) & ~collision
        timeout = self.steps >= c.max_steps
        rew = (c.w_progress * progress - c.w_lat * lat.abs() - c.w_head * head.abs()
               - c.w_clear * (c.clear_band - clear).clamp(min=0) / c.clear_band
               - c.w_rate * rate - c.w_time
               + c.r_collision * collision + c.r_goal * goal)
        done = collision | goal | timeout
        info = {"collision": collision, "goal": goal, "timeout": timeout & ~collision & ~goal,
                "lat": lat.abs(), "speed": self.v.clone()}
        if auto_reset:
            self.reset(done.nonzero()[:, 0])
        return self.observe(), rew, done, info
