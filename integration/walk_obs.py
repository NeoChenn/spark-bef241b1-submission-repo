from __future__ import annotations

import numpy as np

N_SCAN = 187
WALK_OBS = 3 + 3 + 3 + 3 + 12 + 12 + 12 + N_SCAN + 1
ACTION_SCALE = 0.25
SCAN_OFFSET = 0.5


def scan_value(base_height: float, ground: float = 0.0) -> float:
    return float(np.clip(base_height - ground - SCAN_OFFSET, -1.0, 1.0))


def walk_obs(cmd, lin_vel, ang_vel, gravity, joint_pos_rel, joint_vel, last_action, base_height: float,
             payload: float = 0.0) -> np.ndarray:
    parts = [lin_vel, ang_vel, gravity, cmd, joint_pos_rel, joint_vel, last_action,
             np.full(N_SCAN, scan_value(base_height)), [payload]]
    return np.concatenate([np.asarray(p, np.float32).reshape(-1) for p in parts])


def leg_targets(default, action, scale: float = ACTION_SCALE) -> np.ndarray:
    return np.asarray(default, float) + scale * np.asarray(action, float)


def speed_profile(s: float, dist: float, vmax: float = 0.6, ramp: float = 0.4, vmin: float = 0.12) -> float:
    left = dist - s
    if left <= 1e-6:
        return 0.0
    k = min(1.0, (s + 0.05) / ramp, left / ramp)
    return float(max(vmin, vmax * k)) if left > 0.02 else 0.0
