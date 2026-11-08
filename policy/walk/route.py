from __future__ import annotations

import math

import numpy as np


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _xy(pts) -> np.ndarray:
    p = np.asarray(pts, dtype=float)[:, :2]
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9]
    return p[keep]


def polyline_length(pts) -> float:
    p = _xy(pts)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def resample(pts, step: float) -> np.ndarray:
    p = _xy(pts)
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, math.ceil(np.linalg.norm(b - a) / step - 1e-9))
        out.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return np.array(out)


def route_to_goal(route, goal) -> np.ndarray:
    p = _xy(route)
    g = np.asarray(goal, dtype=float)[:2]
    best, best_i, best_q = math.inf, 0, p[0]
    for i, (a, b) in enumerate(zip(p[:-1], p[1:])):
        d = b - a
        t = float(np.clip(np.dot(g - a, d) / np.dot(d, d), 0.0, 1.0))
        q = a + t * d
        dist = float(np.linalg.norm(g - q))
        if dist < best - 1e-9:
            best, best_i, best_q = dist, i, q
    return _xy(np.vstack([p[:best_i + 1], best_q, g]))


def stance_from_meta(meta: dict, standoff: float = 0.5) -> tuple[float, float, float]:
    spawn = meta.get("world", {}).get("robot_spawn")
    if spawn:
        x, y = spawn["position_m"][:2]
        return float(x), float(y), float(spawn.get("yaw_rad", 0.0))
    return float(meta["damage"]["position_m"][0]), float(meta["main_pipe"]["y_m"]) + standoff, 0.0


def falls_per_km(falls: int, dist_m: float) -> float:
    return falls / (dist_m / 1000.0) if dist_m > 0 else 0.0


class WaypointFollower:
    def __init__(self, path, goal_yaw: float, v_max: float = 1.0, w_max: float = 1.0, lookahead: float = 0.8,
                 k_heading: float = 2.0, r_final: float = 1.0, k_final: float = 1.5, v_final: float = 0.5,
                 pos_tol: float = 0.03, yaw_tol: float = 0.05, k_slow: float = 0.8, v_min: float = 0.0):
        self.pts = resample(path, 0.05)
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.pts, axis=0), axis=1))]
        self.total = float(self.s[-1])
        self.goal = self.pts[-1]
        self.goal_yaw = goal_yaw
        self.v_max, self.w_max, self.lookahead, self.k_heading = v_max, w_max, lookahead, k_heading
        self.r_final, self.k_final, self.v_final, self.k_slow = r_final, k_final, v_final, k_slow
        self.pos_tol, self.yaw_tol, self.v_min = pos_tol, yaw_tol, v_min
        self.progress, self.done = 0.0, False

    def _update_progress(self, p):
        win = (self.s >= self.progress - 0.5) & (self.s <= self.progress + 2.0)
        idx = np.flatnonzero(win)
        k = idx[np.argmin(np.linalg.norm(self.pts[idx] - p, axis=1))]
        self.progress = max(self.progress, float(self.s[k]))

    def command(self, x: float, y: float, yaw: float) -> tuple[float, float, float, bool]:
        p = np.array([x, y])
        e = self.goal - p
        dist = float(np.linalg.norm(e))
        yaw_err = wrap(self.goal_yaw - yaw)
        if dist < self.pos_tol and abs(yaw_err) < self.yaw_tol:
            self.done = True
        if self.done:
            return 0.0, 0.0, 0.0, True
        self._update_progress(p)
        if dist < self.r_final and self.progress > self.total - self.r_final - 0.5:
            c, s = math.cos(yaw), math.sin(yaw)
            bx, by = c * e[0] + s * e[1], -s * e[0] + c * e[1]
            v = np.array([bx, by]) * self.k_final
            n = float(np.linalg.norm(v))
            if n > self.v_final:
                v *= self.v_final / n
            elif 1e-9 < n < self.v_min and dist > self.pos_tol:
                v *= self.v_min / n
            wz = float(np.clip(self.k_heading * yaw_err, -self.w_max, self.w_max))
            return float(v[0]), float(v[1]), wz, False
        k = min(int(np.searchsorted(self.s, self.progress + self.lookahead)), len(self.pts) - 1)
        t = self.pts[k] - p
        err = wrap(math.atan2(t[1], t[0]) - yaw)
        v = min(self.v_max, self.k_slow * (self.total - self.progress) + self.v_final)
        vx = v * max(0.0, math.cos(err)) ** 2
        wz = float(np.clip(self.k_heading * err, -self.w_max, self.w_max))
        return vx, 0.0, wz, False


def simulate(follower: WaypointFollower, pose0, dt: float = 0.02, t_max: float = 300.0):
    x, y, yaw = pose0
    traj, t = [(x, y, yaw)], 0.0
    while t < t_max:
        vx, vy, wz, done = follower.command(x, y, yaw)
        if done:
            break
        c, s = math.cos(yaw), math.sin(yaw)
        x, y, yaw = x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt, yaw + wz * dt
        t += dt
        traj.append((x, y, yaw))
    return traj, t
