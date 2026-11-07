"""Pure-torch force setpoint mapping and over-force penalty for the SCRUB env."""
from __future__ import annotations

import torch


def force_setpoint(a: torch.Tensor, mid: float, half: float, f_max: float | None = None) -> torch.Tensor:
    f = mid + half * a
    return f if f_max is None else f.clamp(max=f_max)


def over_force_penalty(F: torch.Tensor, limit: float, w_lin: float, soft: float | None = None,
                       w_quad: float = 0.0) -> torch.Tensor:
    pen = w_lin * (F - limit).clamp_min(0.0) / 10.0
    if soft is not None and w_quad:
        pen = pen + w_quad * ((F - soft).clamp_min(0.0) / 10.0) ** 2
    return pen
