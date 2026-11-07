from __future__ import annotations

import math
from dataclasses import dataclass

import torch

TWO_PI = 2.0 * math.pi


@dataclass
class AbrasionParams:
    k: float = 0.4
    min_force: float = 5.0
    max_force: float = 40.0
    soak_required: float = 30.0
    mult_soaked: float = 4.0
    mult_sprayed: float = 1.5
    max_slide_speed: float = 0.5
    rust_eps: float = 0.02


def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, TWO_PI) - math.pi


def _per_env(x, n: int, device) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32, device=device)
    return t.expand(n).clone() if t.dim() == 0 else t.float()


def random_patches(
    num_envs: int,
    H: int = 32,
    W: int = 64,
    radius: torch.Tensor | float = 0.1,
    length: float = 0.6,
    n_min: int = 1,
    n_max: int = 3,
    size_range: tuple[float, float] = (0.03, 0.09),
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    theta_window: tuple[float, float] | None = None,
) -> torch.Tensor:
    dev = torch.device(device)
    rad = _per_env(radius, num_envs, dev)
    g = generator

    def rnd(*shape):
        return torch.rand(*shape, device=dev, generator=g)

    theta = (torch.arange(W, device=dev).float() + 0.5) / W * TWO_PI
    z = (torch.arange(H, device=dev).float() + 0.5) / H * length - length / 2

    field = torch.zeros(num_envs, H, W, device=dev)
    n_patches = torch.randint(n_min, n_max + 1, (num_envs,), device=dev, generator=g)
    max_lobes = 5
    for p in range(n_max):
        active = (n_patches > p).float().view(-1, 1, 1)
        if theta_window is None:
            ct = rnd(num_envs) * TWO_PI
        else:
            ct = theta_window[0] + (2 * rnd(num_envs) - 1) * theta_window[1]
        cz = (rnd(num_envs) - 0.5) * length * 0.75
        size = size_range[0] + rnd(num_envs) * (size_range[1] - size_range[0])
        blob = torch.zeros(num_envs, H, W, device=dev)
        n_lobes = torch.randint(3, max_lobes + 1, (num_envs,), device=dev, generator=g)
        for l in range(max_lobes):
            la = (n_lobes > l).float().view(-1, 1, 1)
            off_t = (rnd(num_envs) - 0.5) * size / rad
            off_z = (rnd(num_envs) - 0.5) * size
            sx = size * (0.3 + 0.5 * rnd(num_envs))
            sz = size * (0.3 + 0.5 * rnd(num_envs))
            dth = _wrap_angle(theta.view(1, 1, W) - (ct + off_t).view(-1, 1, 1)) * rad.view(-1, 1, 1)
            dz = z.view(1, H, 1) - (cz + off_z).view(-1, 1, 1)
            blob = blob + la * torch.exp(-0.5 * ((dth / sx.view(-1, 1, 1)) ** 2 + (dz / sz.view(-1, 1, 1)) ** 2))
        field = torch.maximum(field, active * blob)
    noise = rnd(num_envs, H, W)
    level = torch.clamp((field + 0.25 * (noise - 0.5) - 0.35) / 0.25, 0.0, 1.0)
    interior = 0.6 + 0.4 * rnd(num_envs, 1, 1)
    level = level * interior
    if theta_window is not None:
        inside = _wrap_angle(theta - theta_window[0]).abs() <= theta_window[1] + 0.3
        level = level * inside.view(1, 1, W).float()
    return level


class RustState:

    def __init__(
        self,
        num_envs: int,
        H: int = 32,
        W: int = 64,
        length: float = 0.6,
        device: str | torch.device = "cpu",
        params: AbrasionParams | None = None,
        theta_window: tuple[float, float] | None = None,
    ):
        self.num_envs, self.H, self.W, self.length = num_envs, H, W, length
        self.theta_window = theta_window
        self.device = torch.device(device)
        self.p = params or AbrasionParams()
        d = self.device
        self.rust = torch.zeros(num_envs, H, W, device=d)
        self.sprayed = torch.zeros(num_envs, H, W, dtype=torch.bool, device=d)
        self.soak_time = torch.zeros(num_envs, H, W, device=d)
        self.reachable = torch.ones(num_envs, H, W, dtype=torch.bool, device=d)
        self.radius = torch.full((num_envs,), 0.1, device=d)
        self.abrasion_coef = torch.ones(num_envs, device=d)
        self.initial_total = torch.ones(num_envs, device=d)
        self.theta_c = (torch.arange(W, device=d).float() + 0.5) / W * TWO_PI
        self.z_c = (torch.arange(H, device=d).float() + 0.5) / H * length - length / 2
        self.last_excess_force = torch.zeros(num_envs, dtype=torch.bool, device=d)
        self.last_unsprayed_contact = torch.zeros(num_envs, device=d)
        self.last_removed = torch.zeros(num_envs, device=d)

    def _ids(self, env_ids):
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def reset(
        self,
        env_ids=None,
        radius: torch.Tensor | float | None = None,
        rust: torch.Tensor | None = None,
        spray_all_rust: bool = True,
        soak_time: torch.Tensor | float = 60.0,
        abrasion_coef: torch.Tensor | float | None = None,
        generator: torch.Generator | None = None,
        reachable: torch.Tensor | None = None,
        n_patches: tuple[int, int] = (1, 3),
    ):
        ids = self._ids(env_ids)
        n = ids.numel()
        if n == 0:
            return
        if radius is not None:
            self.radius[ids] = _per_env(radius, n, self.device)
        if abrasion_coef is not None:
            self.abrasion_coef[ids] = _per_env(abrasion_coef, n, self.device)
        if reachable is not None:
            self.reachable[ids] = reachable.to(self.device).bool()
        if rust is None:
            reach = self.reachable[ids].float()
            rust = random_patches(n, self.H, self.W, self.radius[ids], self.length, device=self.device,
                                  generator=generator, theta_window=self.theta_window,
                                  n_min=n_patches[0], n_max=n_patches[1]) * reach
            empty = rust.sum(dim=(1, 2)) < 1.0
            if empty.any():
                rust[empty] = self._central_blob(ids[empty]) * reach[empty]
        self.rust[ids] = rust.to(self.device).float().clamp(0, 1)
        self.sprayed[ids] = False
        self.soak_time[ids] = 0.0
        if spray_all_rust:
            self.spray(self.dilate(self.rust[ids] > self.p.rust_eps, 2), env_ids=ids, soak_time=soak_time)
        self.initial_total[ids] = self.rust[ids].sum(dim=(1, 2)).clamp_min(1e-6)

    def _central_blob(self, ids, sigma: float = 0.04) -> torch.Tensor:
        th0 = self.theta_window[0] if self.theta_window else 0.0
        uv0 = self.surface_to_uv(torch.full((self.num_envs,), th0, device=self.device),
                                 torch.zeros(self.num_envs, device=self.device))
        darc, dz = self._surface_dist(uv0)
        d2 = (darc ** 2 + dz ** 2)[ids]
        d2r = torch.where(self.reachable[ids], d2, torch.full_like(d2, 1e6)).view(len(ids), -1)
        c = d2r.argmin(1)
        row, col = c // self.W, c % self.W
        uv = torch.stack([(col.float() + 0.5) / self.W, (row.float() + 0.5) / self.H], -1)
        full = torch.zeros(self.num_envs, 2, device=self.device)
        full[ids] = uv
        darc, dz = self._surface_dist(full)
        blob = torch.exp(-0.5 * (darc[ids] ** 2 + dz[ids] ** 2) / sigma ** 2)
        return (blob > 0.3).float() * 0.8

    def dilate(self, mask: torch.Tensor, cells: int = 1) -> torch.Tensor:
        k = 2 * cells + 1
        m = mask.float().unsqueeze(1)
        m = torch.cat([m[..., -cells:], m, m[..., :cells]], dim=-1)
        m = torch.nn.functional.pad(m, (0, 0, cells, cells))
        out = torch.nn.functional.max_pool2d(m, k, stride=1)
        return out.squeeze(1) > 0.5

    def spray(self, mask: torch.Tensor, env_ids=None, soak_time: torch.Tensor | float = 0.0):
        ids = self._ids(env_ids)
        mask = mask.to(self.device).bool()
        new = mask & ~self.sprayed[ids]
        st = torch.as_tensor(soak_time, device=self.device, dtype=torch.float32)
        if st.dim() == 1:
            st = st.view(-1, 1, 1)
        soak = self.soak_time[ids]
        soak = torch.where(new, st.expand_as(soak), soak)
        self.soak_time[ids] = soak
        self.sprayed[ids] = self.sprayed[ids] | mask

    def step_soak(self, dt: float):
        self.soak_time += dt * self.sprayed.float()

    def add_patch(self, env_ids, size_range=(0.03, 0.07), sprayed: bool = True, soak_time: float = 60.0,
                  generator: torch.Generator | None = None) -> torch.Tensor:
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return torch.zeros(0, self.H, self.W, device=self.device)
        patch = random_patches(ids.numel(), self.H, self.W, self.radius[ids], self.length, n_min=1, n_max=1,
                               size_range=size_range, device=self.device, generator=generator,
                               theta_window=self.theta_window) * self.reachable[ids].float()
        self.rust[ids] = torch.maximum(self.rust[ids], patch)
        if sprayed:
            self.spray(self.dilate(patch > self.p.rust_eps, 2), env_ids=ids, soak_time=soak_time)
        self.initial_total[ids] = torch.maximum(self.initial_total[ids], self.rust[ids].sum(dim=(1, 2)))
        return patch

    def uv_to_cell(self, uv: torch.Tensor):
        col = torch.floor(torch.remainder(uv[:, 0], 1.0) * self.W).long() % self.W
        row = torch.floor(uv[:, 1] * self.H).long().clamp(0, self.H - 1)
        return row, col

    def surface_to_uv(self, theta: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.remainder(theta, TWO_PI) / TWO_PI, (z + self.length / 2) / self.length], dim=-1)

    def _surface_dist(self, uv: torch.Tensor):
        th = uv[:, 0] * TWO_PI
        zc = uv[:, 1] * self.length - self.length / 2
        darc = _wrap_angle(self.theta_c.view(1, 1, -1) - th.view(-1, 1, 1)) * self.radius.view(-1, 1, 1)
        dz = self.z_c.view(1, -1, 1) - zc.view(-1, 1, 1)
        shape = (uv.shape[0], self.H, self.W)
        return darc.expand(shape), dz.expand(shape)

    def abrade(self, contact_uv: torch.Tensor, contact_radius, normal_force: torch.Tensor,
               slide_speed: torch.Tensor, dt: float) -> torch.Tensor:
        p = self.p
        R = _per_env(contact_radius, self.num_envs, self.device).view(-1, 1, 1)
        darc, dz = self._surface_dist(contact_uv)
        d2 = (darc ** 2 + dz ** 2) / R ** 2
        w = torch.clamp(1.0 - d2, min=0.0)
        F = normal_force.float()
        active = F >= p.min_force
        speed = slide_speed.float().clamp(0.0, p.max_slide_speed)
        soaked = self.sprayed & (self.soak_time >= p.soak_required)
        mult = torch.where(soaked, torch.full_like(w, p.mult_soaked),
                           torch.where(self.sprayed, torch.full_like(w, p.mult_sprayed), torch.ones_like(w)))
        rate = (p.k * self.abrasion_coef * F * speed * active.float()).view(-1, 1, 1)
        delta = torch.minimum(self.rust, rate * mult * w * dt)
        self.rust = self.rust - delta
        removed = delta.sum(dim=(1, 2))
        wsum = w.sum(dim=(1, 2)).clamp_min(1e-6)
        self.last_unsprayed_contact = (w * (~self.sprayed).float()).sum(dim=(1, 2)) / wsum * active.float()
        self.last_excess_force = F > p.max_force
        self.last_removed = removed
        return removed

    def fraction_remaining(self) -> torch.Tensor:
        return self.rust.sum(dim=(1, 2)) / self.initial_total

    def rusty_cells(self) -> torch.Tensor:
        return (self.rust > self.p.rust_eps).sum(dim=(1, 2))

    def _crop(self, grid: torch.Tensor, uv: torch.Tensor, k: int, pad_value: float = 0.0) -> torch.Tensor:
        row, col = self.uv_to_cell(uv)
        off = torch.arange(k, device=self.device) - k // 2
        rows = row.view(-1, 1) + off.view(1, -1)
        cols = (col.view(-1, 1) + off.view(1, -1)) % self.W
        valid = (rows >= 0) & (rows < self.H)
        rows_c = rows.clamp(0, self.H - 1)
        n = torch.arange(grid.shape[0], device=self.device).view(-1, 1, 1)
        out = grid[n, rows_c.view(-1, k, 1), cols.view(-1, 1, k)].float()
        return torch.where(valid.view(-1, k, 1), out, torch.full_like(out, pad_value))

    def local_patch(self, uv: torch.Tensor, k: int = 9) -> torch.Tensor:
        return self._crop(self.rust, uv, k)

    def local_sprayed(self, uv: torch.Tensor, k: int = 9) -> torch.Tensor:
        soaked = (self.sprayed & (self.soak_time >= self.p.soak_required)).float() * 0.5 + self.sprayed.float() * 0.5
        return self._crop(soaked, uv, k)

    def coarse_map(self, h: int = 8, w: int = 16) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(self.rust.unsqueeze(1), (h, w)).squeeze(1)

    def nearest_rust(self, uv: torch.Tensor):
        darc, dz = self._surface_dist(uv)
        d2 = darc ** 2 + dz ** 2
        rusty = self.rust > self.p.rust_eps
        d2m = torch.where(rusty, d2, torch.full_like(d2, 1e6)).view(self.num_envs, -1)
        idx = d2m.argmin(dim=1)
        has = rusty.view(self.num_envs, -1).any(dim=1)
        a = darc.reshape(self.num_envs, -1).gather(1, idx[:, None]).squeeze(1) * has
        b = dz.reshape(self.num_envs, -1).gather(1, idx[:, None]).squeeze(1) * has
        return torch.stack([a, b], dim=-1), torch.sqrt(a ** 2 + b ** 2)

    def to_texture(self, env_ids=None, overlay: bool = False) -> torch.Tensor:
        ids = self._ids(env_ids)
        r = self.rust[ids].unsqueeze(-1)
        steel = torch.tensor([150.0, 152.0, 158.0], device=self.device)
        rust_light = torch.tensor([176.0, 92.0, 38.0], device=self.device)
        rust_dark = torch.tensor([96.0, 44.0, 18.0], device=self.device)
        rust_col = rust_light + (rust_dark - rust_light) * r
        sprayed = self.sprayed[ids].unsqueeze(-1).float()
        tint = torch.tensor([-10.0, 0.0, 25.0], device=self.device) * sprayed
        if overlay:
            rgb = rust_col + tint
            a = torch.maximum(r * 255.0, sprayed * 40.0)
        else:
            rgb = steel + (rust_col - steel) * r.clamp(0, 1) ** 0.7 + tint
            a = torch.full_like(r, 255.0)
        return torch.cat([rgb, a], dim=-1).round().clamp(0, 255).to(torch.uint8)


def rig_meta_scene(meta_path: str = "sim/output/rig_deck_meta.json", standoff: float = 0.30,
                   base_height: float = 0.55) -> dict:
    import json

    with open(meta_path) as f:
        m = json.load(f)
    pipe, dmg = m["main_pipe"], m["damage"]
    r = float(pipe["radius_m"])
    xd, yd, zd = (float(v) for v in dmg["position_m"])
    n = dmg["surface_normal"]
    side = 1.0 if (yd - pipe["y_m"]) >= 0 else -1.0
    theta = math.atan2(float(n[2]), side * float(n[1]))
    origin = (xd, float(pipe["y_m"]), float(pipe["z_m"]))
    robot_pos = (xd, origin[1] + side * (r + standoff), base_height)
    return {
        "pipe_origin": origin,
        "pipe_axis": (1.0, 0.0, 0.0),
        "pipe_ref": (0.0, side, 0.0),
        "pipe_radius": r,
        "damage_theta": theta,
        "damage_z": 0.0,
        "damage_radius": float(dmg.get("approx_radius_m", 0.1)),
        "robot_pos_world": robot_pos,
        "robot_yaw_world": -side * math.pi / 2,
        "standoff": standoff,
        "env_to_world": {"translation": (xd, robot_pos[1], 0.0), "yaw": 0.0 if side > 0 else math.pi},
        "work_zone": m.get("work_zone_m"),
    }
