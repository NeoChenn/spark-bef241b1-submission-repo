from __future__ import annotations

import math

import torch

GEOM_DIM = 14


def ray_grid(n_side: int, fov: float, device) -> torch.Tensor:
    t = torch.linspace(-math.tan(fov / 2), math.tan(fov / 2), n_side, device=device)
    u, v = torch.meshgrid(t, t, indexing="ij")
    d = torch.stack([u.flatten(), v.flatten(), torch.ones(n_side * n_side, device=device)], -1)
    return d / d.norm(dim=-1, keepdim=True)


def look_basis(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    z = target - eye
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    up = torch.tensor([0.0, 0.0, 1.0], device=eye.device).expand_as(z)
    x = torch.cross(z, up, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], -1)


def raycast_can(eye, R_cam, dirs, can_p, can_R, r, h, max_range: float = 1.5):
    d_w = torch.einsum("nij,kj->nki", R_cam, dirs)
    o = torch.bmm(can_R.transpose(1, 2), (eye - can_p).unsqueeze(-1)).squeeze(-1)[:, None, :].expand_as(d_w)
    d = torch.bmm(d_w, can_R)
    big = torch.full(d.shape[:2], float("inf"), device=d.device)
    a = d[..., 0] ** 2 + d[..., 1] ** 2
    b = 2 * (o[..., 0] * d[..., 0] + o[..., 1] * d[..., 1])
    c = o[..., 0] ** 2 + o[..., 1] ** 2 - (r ** 2)[:, None]
    disc = b * b - 4 * a * c
    ok = (disc > 0) & (a > 1e-9)
    t_side = (-b - disc.clamp_min(0).sqrt()) / (2 * a.clamp_min(1e-9))
    z_side = o[..., 2] + t_side * d[..., 2]
    side_hit = ok & (t_side > 0) & (z_side.abs() <= (h / 2)[:, None])
    t = torch.where(side_hit, t_side, big)
    for sgn in (1.0, -1.0):
        tc = (sgn * (h / 2)[:, None] - o[..., 2]) / torch.where(d[..., 2].abs() > 1e-9, d[..., 2], torch.full_like(d[..., 2], 1e-9))
        x = o[..., 0] + tc * d[..., 0]
        y = o[..., 1] + tc * d[..., 1]
        cap = (tc > 0) & (x * x + y * y <= (r ** 2)[:, None])
        t = torch.where(cap & (tc < t), tc, t)
    hit = torch.isfinite(t) & (t < max_range)
    pts = eye[:, None, :] + t.clamp(max=max_range)[..., None] * d_w
    return pts, hit


def fit_cylinder(pts: torch.Tensor, hit: torch.Tensor, deck_z: float = 0.0):
    w = hit.float()
    n = w.sum(1).clamp_min(1.0)
    mu = (pts * w[..., None]).sum(1) / n[:, None]
    q = (pts - mu[:, None, :]) * w[..., None]
    cov = torch.bmm(q.transpose(1, 2), q) / n[:, None, None]
    ev, evec = torch.linalg.eigh(cov + 1e-9 * torch.eye(3, device=pts.device))
    horiz = evec[:, :, 2].clone()
    horiz[:, 2] = 0.0
    horiz = horiz / horiz.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    top = (pts[..., 2] * w + (1 - w) * -1e3).max(1).values
    height = (top - deck_z).clamp_min(0.0)
    proj = ((pts[..., :2] - mu[:, None, :2]) * horiz[:, None, :2]).sum(-1)
    lo = torch.where(hit, proj, torch.full_like(proj, 1e3)).min(1).values
    hi = torch.where(hit, proj, torch.full_like(proj, -1e3)).max(1).values
    span = (hi - lo).clamp_min(0.0)
    perp = torch.stack([-horiz[:, 1], horiz[:, 0]], -1)
    pp = ((pts[..., :2] - mu[:, None, :2]) * perp[:, None, :]).sum(-1)
    plo = torch.where(hit, pp, torch.full_like(pp, 1e3)).min(1).values
    phi = torch.where(hit, pp, torch.full_like(pp, -1e3)).max(1).values
    width = (phi - plo).clamp_min(0.0)
    lying = (span > 1.25 * width) | (height < 1.1 * width)
    radius = torch.where(lying, height / 2, width / 2).clamp(0.015, 0.08)
    length = torch.where(lying, span, height).clamp(0.05, 0.3)
    topm = hit & (pts[..., 2] > top[:, None] - 0.006)
    tn = topm.float().sum(1).clamp_min(1.0)
    top_pp = (pp * topm).sum(1) / tn
    top_xy = (pts[..., :2] * topm[..., None]).sum(1) / tn[:, None]
    xy_lying = mu[:, :2] + horiz[:, :2] * ((lo + hi) / 2)[:, None] + perp * top_pp[:, None]
    xy = torch.where(lying[:, None], xy_lying, top_xy)
    cz = torch.where(lying, radius, length / 2) + deck_z
    centre = torch.cat([xy, cz[:, None]], -1)
    axis = torch.where(lying[:, None], horiz, torch.tensor([0.0, 0.0, 1.0], device=pts.device).expand_as(horiz))
    grasp = centre + torch.where(lying, radius, length / 2)[:, None] * torch.tensor([0.0, 0.0, 1.0], device=pts.device)
    rho = ((pts - centre[:, None, :]) - ((pts - centre[:, None, :]) * axis[:, None, :]).sum(-1, keepdim=True) * axis[:, None, :]).norm(dim=-1)
    side = (((pts - centre[:, None, :]) * axis[:, None, :]).sum(-1).abs() < length[:, None] / 2 - 0.005) & hit
    resid = ((rho - radius[:, None]).abs() * side).sum(1) / side.float().sum(1).clamp_min(1.0)
    conf = (n / 30.0).clamp(max=1.0) * torch.exp(-resid / 0.01) * (w.sum(1) > 5)
    return {"centre": centre, "axis": axis, "radius": radius, "length": length, "lying": lying, "grasp": grasp,
            "residual": resid, "conf": conf, "n": w.sum(1)}


def geom_obs(fit: dict, tip: torch.Tensor) -> torch.Tensor:
    ax = fit["axis"]
    return torch.cat([(fit["centre"] - tip) * 4.0, (fit["grasp"] - tip) * 4.0,
                      ax[:, :1] ** 2 - ax[:, 1:2] ** 2, 2.0 * ax[:, :1] * ax[:, 1:2], ax[:, 2:].abs(),
                      fit["radius"][:, None] * 20.0, fit["length"][:, None] * 5.0, fit["lying"][:, None].float(),
                      (fit["residual"][:, None] * 100.0).clamp(max=5.0), fit["conf"][:, None]], -1)
