"""Observation and action spec shared by the 2D trainer and the Isaac-side controller.

obs (53 floats, in this order):
  [0:36]   lidar, min range per 10 deg sector, sector j covers [-pi + 10j deg, -pi + 10(j+1) deg) in the
           robot frame (x forward, y left, CCW), divided by LIDAR_RANGE, 1.0 = nothing within range
  [36]     lateral offset from the path (m, + = robot is left of the path), / 0.5
  [37]     heading error yaw - path_heading (rad, wrapped), / pi
  [38:48]  path points 0.5, 1.0, 1.5, 2.0, 2.5 m ahead along the path, robot frame (x, y) pairs, / 2.5
  [48:50]  measured forward speed (m/s) / V_MAX, yaw rate (rad/s) / W_MAX
  [50:52]  previous action (policy output, in [-1, 1])
  [52]     distance left to the end of the path (m), / 5, clipped to 1

action (2 floats in [-1, 1]) -> velocity command for the Go2 locomotion controller:
  vx = a0 * V_MAX if a0 >= 0 else a0 * -V_MIN      (m/s, body frame)
  wz = a1 * W_MAX                                   (rad/s)
"""
from __future__ import annotations

import math

import numpy as np
import torch

LIDAR_RANGE = 4.0
SECTORS = 36
LOOKAHEAD = (0.5, 1.0, 1.5, 2.0, 2.5)
V_MAX, V_MIN, W_MAX = 1.0, -0.3, 1.5
OBS_DIM = SECTORS + 2 + 2 * len(LOOKAHEAD) + 2 + 2 + 1


def _sector_of(angles):
    return ((np.asarray(angles) + math.pi) % (2 * math.pi) / (2 * math.pi / SECTORS)).astype(int) % SECTORS


def pool_sectors_np(ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Any beam count / FOV -> 36 sector minima (m). Sectors with no beams read LIDAR_RANGE."""
    r = np.nan_to_num(np.asarray(ranges, dtype=np.float64), nan=LIDAR_RANGE, posinf=LIDAR_RANGE)
    r = np.clip(r, 0.0, LIDAR_RANGE)
    out = np.full(SECTORS, LIDAR_RANGE)
    np.minimum.at(out, _sector_of(angles), r)
    return out


def pool_sectors_torch(ranges: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    sec = torch.remainder(angles + math.pi, 2 * math.pi).div(2 * math.pi / SECTORS).long() % SECTORS
    out = torch.full((ranges.shape[0], SECTORS), LIDAR_RANGE, device=ranges.device, dtype=ranges.dtype)
    return out.scatter_reduce(1, sec.expand_as(ranges), ranges.clamp(0, LIDAR_RANGE), reduce="amin")


def build_obs_np(sectors_m, lat, head, look_xy, v, w, last_act, dist_goal) -> np.ndarray:
    head = (head + math.pi) % (2 * math.pi) - math.pi
    return np.concatenate([
        np.asarray(sectors_m, dtype=np.float64) / LIDAR_RANGE,
        [lat / 0.5, head / math.pi],
        np.asarray(look_xy, dtype=np.float64).reshape(-1) / 2.5,
        [v / V_MAX, w / W_MAX],
        np.asarray(last_act, dtype=np.float64).reshape(-1),
        [min(dist_goal / 5.0, 1.0)],
    ]).astype(np.float32)


def build_obs_torch(sectors_m, lat, head, look_xy, v, w, last_act, dist_goal) -> torch.Tensor:
    head = torch.remainder(head + math.pi, 2 * math.pi) - math.pi
    return torch.cat([
        sectors_m / LIDAR_RANGE,
        (lat / 0.5)[:, None], (head / math.pi)[:, None],
        look_xy.reshape(look_xy.shape[0], -1) / 2.5,
        (v / V_MAX)[:, None], (w / W_MAX)[:, None],
        last_act,
        (dist_goal / 5.0).clamp(max=1.0)[:, None],
    ], 1)


def action_to_cmd_torch(a: torch.Tensor):
    a = a.clamp(-1, 1)
    vx = torch.where(a[:, 0] >= 0, a[:, 0] * V_MAX, a[:, 0] * -V_MIN)
    return vx, a[:, 1] * W_MAX


def action_to_cmd_np(a) -> tuple[float, float]:
    a = np.clip(np.asarray(a, dtype=np.float64).reshape(-1), -1, 1)
    vx = a[0] * V_MAX if a[0] >= 0 else a[0] * -V_MIN
    return float(vx), float(a[1] * W_MAX)
