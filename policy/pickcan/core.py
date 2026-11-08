from __future__ import annotations

import math

import torch

G = 9.81
TOTE_CENTER = (-0.27, 0.0)
TOTE_HALF = (0.122, 0.102)
TOTE_FLOOR = 0.32 + 0.062 + 0.008
TOTE_RIM = 0.32 + 0.062 + 0.09


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).view(*q.shape[:-1], 3, 3)


def can_quat(lying: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    h = 0.5 * math.pi / 2
    qz = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
    qy = torch.tensor([math.cos(h), 0.0, math.sin(h), 0.0], device=yaw.device).expand_as(qz)
    w1, x1, y1, z1 = qz.unbind(-1)
    w2, x2, y2, z2 = qy.unbind(-1)
    prod = torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)
    return torch.where(lying[:, None], prod, qz)


def rest_height(lying: torch.Tensor, r: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return torch.where(lying, r, h / 2)


def grasp_point(pos: torch.Tensor, axis: torch.Tensor, r: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    up = axis[:, 2].abs() > 0.7
    return pos + torch.where(up, h / 2, r)[:, None] * torch.tensor([0.0, 0.0, 1.0], device=pos.device)


def surface(tip_local: torch.Tensor, r: torch.Tensor, h: torch.Tensor):
    rho = tip_local[:, :2].norm(dim=-1)
    az = tip_local[:, 2].abs()
    d_side = rho - r
    d_cap = az - h / 2
    on_cap = (rho <= r) & ((d_cap >= d_side) | (az > h / 2))
    on_side = ~on_cap & (az <= h / 2)
    dist = torch.where(on_cap, d_cap, torch.where(on_side, d_side, torch.full_like(rho, 1.0)))
    radial = torch.cat([tip_local[:, :2] / rho.clamp_min(1e-6)[:, None], torch.zeros_like(rho)[:, None]], -1)
    cap_n = torch.zeros_like(tip_local)
    cap_n[:, 2] = torch.sign(tip_local[:, 2] + 1e-9)
    normal = torch.where(on_cap[:, None], cap_n, radial)
    margin = torch.where(on_cap, r - rho, h / 2 - az)
    return dist, normal, margin


def seal_ok(tip_p, tool_axis, can_p, can_R, r, h, max_gap=0.012, min_margin=0.012, min_align=0.9):
    local = torch.bmm(can_R.transpose(1, 2), (tip_p - can_p).unsqueeze(-1)).squeeze(-1)
    dist, n_local, margin = surface(local, r, h)
    n_w = torch.bmm(can_R, n_local.unsqueeze(-1)).squeeze(-1)
    align = -(n_w * tool_axis).sum(-1)
    return (dist > -0.02) & (dist < max_gap) & (margin > min_margin) & (align > min_align)


def rot_err(R_des: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    E = torch.bmm(R_des, R.transpose(1, 2))
    return 0.5 * torch.stack([E[:, 2, 1] - E[:, 1, 2], E[:, 0, 2] - E[:, 2, 0], E[:, 1, 0] - E[:, 0, 1]], -1)


def suction_wrench(tip_p, tip_R, tip_v, tip_w, can_p, can_R, can_v, can_w, rel_p, rel_R, mass, inertia,
                   omega=25.0, zeta=1.0, omega_r=20.0):
    p_des = tip_p + torch.bmm(tip_R, rel_p.unsqueeze(-1)).squeeze(-1)
    R_des = torch.bmm(tip_R, rel_R)
    v_des = tip_v + torch.cross(tip_w, p_des - tip_p, dim=-1)
    m = mass[:, None]
    F = m * (omega ** 2 * (p_des - can_p) + 2 * zeta * omega * (v_des - can_v))
    F[:, 2] += mass * G
    I_w = torch.bmm(can_R * inertia[:, None, :], can_R.transpose(1, 2))
    T = torch.bmm(I_w, (omega_r ** 2 * rot_err(R_des, can_R) + 2 * zeta * omega_r * (tip_w - can_w)).unsqueeze(-1)).squeeze(-1)
    lever = tip_p - can_p
    t_contact = T - torch.cross(lever, F, dim=-1)
    return F, T, t_contact, (p_des - can_p).norm(dim=-1)


def can_inertia(mass, r, h):
    lat = mass * (3 * r ** 2 + h ** 2) / 12
    return torch.stack([lat, lat, mass * r ** 2 / 2], -1)


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def in_tote(can_p: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    dx = (can_p[:, 0] - TOTE_CENTER[0]).abs()
    dy = (can_p[:, 1] - TOTE_CENTER[1]).abs()
    return (dx < TOTE_HALF[0] - margin) & (dy < TOTE_HALF[1] - margin) & (can_p[:, 2] < TOTE_RIM + 0.02) & (
        can_p[:, 2] > TOTE_FLOOR - 0.01)
