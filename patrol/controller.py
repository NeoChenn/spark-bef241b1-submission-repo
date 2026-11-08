"""Deploy-side path follower: lidar scan + pose + path -> (vx, wz) for the Go2 locomotion controller.

numpy + onnxruntime only. Features match patrol.sim2d.PatrolSim.observe() (see patrol/features.py).

    ctl = PatrolController("runs/patrol/nav_v1/policy.onnx", path_xy)   # or policy="pure_pursuit"
    vx, wz, obs = ctl.step(ranges, angles, (x, y, yaw), v, w)          # call at 10 Hz
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from patrol import features
from patrol.detect import PathRef


def resample_path(path_xy, ds: float = 0.1) -> np.ndarray:
    p = np.asarray(path_xy, dtype=np.float64)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    n = max(int(round(s[-1] / ds)) + 1, 2)
    si = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(si, s, p[:, 0]), np.interp(si, s, p[:, 1])], 1)


def pure_pursuit_np(look_xy, ranges, angles) -> np.ndarray:
    """numpy port of patrol.ppo.pure_pursuit (repulsion normalised to its 72-beam scan)."""
    lx, ly = look_xy[1]
    curv = 2 * ly / max(lx * lx + ly * ly, 1e-3)
    r = np.clip(np.nan_to_num(np.asarray(ranges, float), nan=features.LIDAR_RANGE, posinf=features.LIDAR_RANGE),
                0, features.LIDAR_RANGE)
    a = np.asarray(angles, float)
    close = np.clip(0.7 - r, 0, None)
    front = (np.abs(a) < math.pi / 2).astype(float)
    repel = -(close * np.sin(a) * front).sum() * 1.5 * 72.0 / max(len(r), 1)
    cone = np.abs(a) < 0.5
    ahead = r[cone].min() if cone.any() else features.LIDAR_RANGE
    v = max(0.8 * float(np.clip((ahead - 0.35) / 1.0, 0, 1)), 0.1)
    w = float(np.clip(curv * v + repel, -features.W_MAX, features.W_MAX))
    return np.array([v / features.V_MAX, w / features.W_MAX], dtype=np.float32)


class PatrolController:
    def __init__(self, policy: str | Callable = "pure_pursuit", path_xy=None, ds: float = 0.1):
        self.ds = ds
        self._sess = None
        self._fn = None
        if callable(policy):
            self._fn = policy
        elif policy != "pure_pursuit":
            import onnxruntime as ort
            self._sess = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
        self.mode = "onnx" if self._sess is not None else ("callable" if self._fn else "pure_pursuit")
        self.last_act = np.zeros(2, dtype=np.float32)
        self.idx: int | None = None
        if path_xy is not None:
            self.set_path(path_xy)

    # ------------------------------------------------------------------ path
    def set_path(self, path_xy, resample: bool = True):
        self.P = resample_path(path_xy, self.ds) if resample else np.asarray(path_xy, dtype=np.float64)
        self._derive()
        self.idx = None

    def _derive(self):
        d = np.diff(self.P, axis=0)
        h = np.arctan2(d[:, 1], d[:, 0])
        self.h = np.concatenate([h[:1], h])  # heading at i = direction arriving at point i (as in the sim)
        self.nrm = np.stack([-np.sin(self.h), np.cos(self.h)], 1)
        self.ref = PathRef(self.P, self.ds)

    def reverse(self, pos=None):
        """Swap start and end (shuttle back). Pass the robot position to re-lock the index like the sim."""
        old = None if self.idx is None else len(self.P) - 1 - self.idx
        self.P = self.P[::-1].copy()
        self._derive()
        self.idx = old
        self.last_act = np.zeros(2, dtype=np.float32)
        if pos is not None and self.idx is not None:
            self.update_idx(pos, window=(-30, 30))

    @property
    def M(self) -> int:
        return len(self.P)

    @property
    def progress_idx(self) -> int:
        return 0 if self.idx is None else int(self.idx)

    @property
    def dist_to_goal(self) -> float:
        return (self.M - 1 - self.progress_idx) * self.ds

    @property
    def dist_from_start(self) -> float:
        return self.progress_idx * self.ds

    def update_idx(self, pos, window=(-5, 20)):
        pos = np.asarray(pos, dtype=np.float64)
        if self.idx is None:
            self.idx = int(((self.P - pos) ** 2).sum(1).argmin())
            return self.idx
        lo, hi = max(self.idx + window[0], 0), min(self.idx + window[1], self.M - 1)
        cand = self.P[lo:hi + 1]
        self.idx = int(lo + ((cand - pos) ** 2).sum(1).argmin())
        return self.idx

    def heading_error(self, yaw: float) -> float:
        i = self.progress_idx
        return (yaw - self.h[i] + math.pi) % (2 * math.pi) - math.pi

    # ------------------------------------------------------------------ features + policy
    def features(self, ranges, angles, pose, v, w, update: bool = True):
        x, y, yaw = (float(q) for q in pose)
        i = self.update_idx((x, y)) if update or self.idx is None else self.idx
        lat = float(((np.array([x, y]) - self.P[i]) * self.nrm[i]).sum())
        head = yaw - self.h[i]
        li = np.minimum(i + np.array([round(d / self.ds) for d in features.LOOKAHEAD]), self.M - 1)
        rel = self.P[li] - np.array([x, y])
        cy, sy = math.cos(yaw), math.sin(yaw)
        look = np.stack([cy * rel[:, 0] + sy * rel[:, 1], -sy * rel[:, 0] + cy * rel[:, 1]], 1)
        sectors = features.pool_sectors_np(ranges, angles)
        obs = features.build_obs_np(sectors, lat, head, look, v, w, self.last_act, (self.M - 1 - i) * self.ds)
        return obs, look

    def step(self, ranges, angles, pose, v, w):
        obs, look = self.features(ranges, angles, pose, v, w)
        if self._sess is not None:
            a = self._sess.run(None, {"obs": obs[None].astype(np.float32)})[0][0]
        elif self._fn is not None:
            a = np.asarray(self._fn(obs), dtype=np.float32).reshape(-1)
        else:
            a = pure_pursuit_np(look, ranges, angles)
        a = np.clip(a, -1, 1).astype(np.float32)
        self.last_act = a
        vx, wz = features.action_to_cmd_np(a)
        return vx, wz, obs


def cmd_to_action(vx: float, wz: float) -> np.ndarray:
    """Inverse of features.action_to_cmd_np (to drive the 2D sim from a velocity command)."""
    a0 = vx / features.V_MAX if vx >= 0 else vx / -features.V_MIN
    return np.clip(np.array([a0, wz / features.W_MAX], dtype=np.float32), -1, 1)
