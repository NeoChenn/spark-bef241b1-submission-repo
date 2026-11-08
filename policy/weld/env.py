from __future__ import annotations

import numpy as np

from policy.weld.seam import Seam, heat_input
from policy.weld.sensors import DEFAULT, SensorConfig, eddy, groove_center, laser_profile, telemetry, thermal

LEAD_MM = 5.0
MAX_LAT_MM_S = 4.0
V_NOM = 22.0


def bead_width(current_a: float, speed_mm_s: float) -> float:
    return float(np.clip(0.13 * current_a / np.sqrt(max(speed_mm_s, 0.3)), 4.0, 18.0))


def arc_voltage(seed: int, length_mm: float):
    rng = np.random.default_rng(seed + 7919)
    a, ph, ramp = rng.uniform(2.5, 4.5), rng.uniform(0, 2 * np.pi), rng.uniform(3.0, 7.0)
    return lambda x: V_NOM + a * np.sin(2 * np.pi * x / 180.0 + ph) + ramp * x / length_mm


class WeldEnv:
    def __init__(self, seam: Seam, x0: float, x1: float, seed: int = 0, dt: float = 0.25,
                 sensors: SensorConfig = DEFAULT, blind: bool = False, y0: float = 0.0):
        self.seam, self.dt, self.cfg, self.blind = seam, dt, sensors, blind
        self.x0, self.x1 = x0 - LEAD_MM, x1 + LEAD_MM
        self.x, self.y = self.x0, y0
        self.rng = np.random.default_rng(seed)
        self.volt = arc_voltage(seed, seam.cfg.length_mm)
        self.current, self.speed, self.t = 0.0, 0.0, 0.0
        self.width, self.hi = 0.0, 0.0
        self.path: list[tuple[float, float]] = []

    def obs(self) -> dict:
        xl = self.x + self.cfg.lookahead_mm
        ys, z = laser_profile(self.seam, xl, self.y, self.rng, self.cfg)
        if self.blind:
            z = np.full_like(z, np.nan)
        return {"x": self.x, "y": self.y, "x_look": xl, "profile_y": ys, "profile_z": z,
                "center_est": groove_center(ys, z), "eddy": eddy(self.seam, xl, self.y, self.rng, self.cfg),
                "tele": telemetry(self.current, self.volt(self.x) if self.current else 0.0, self.speed, self.rng,
                                  self.cfg),
                "temp_c": thermal(self.seam, self.rng, self.cfg), "t": self.t}

    def step(self, dy: float, speed: float, current: float):
        lim = MAX_LAT_MM_S * self.dt
        self.y += float(np.clip(dy, -lim, lim))
        self.speed, self.current = float(np.clip(speed, 0.5, 8.0)), float(np.clip(current, 80.0, 220.0))
        self.width = bead_width(self.current, self.speed)
        self.hi = heat_input(self.current, self.volt(self.x), self.speed, self.seam.cfg.eta)
        x_new = min(self.x + self.speed * self.dt, self.x1)
        cols = np.arange(np.floor(self.x), np.floor(x_new)) + 0.5
        for cx in cols:
            self.seam.deposit(float(cx), self.y, self.width, self.hi, self.dt / len(cols))
        if not len(cols):
            self.seam.cool(self.dt)
        self.x, self.t = x_new, self.t + self.dt
        self.path.append((self.x, self.y))
        return self.obs(), self.x >= self.x1


def run_pass(seam: Seam, policy, x0: float, x1: float, seed: int = 0, blind: bool = False, on_step=None,
             max_steps: int = 4000) -> dict:
    env = WeldEnv(seam, x0, x1, seed=seed, blind=blind)
    obs, err, temps, his = env.obs(), [], [], []
    for _ in range(max_steps):
        dy, v, i = policy.act(obs)
        obs, done = env.step(dy, v, i)
        if x0 <= env.x <= x1:
            err.append(env.y - seam.center_at(env.x))
        temps.append(seam.temp_c)
        his.append(env.hi)
        if on_step is not None:
            on_step(env, obs)
        if done:
            break
    m = seam.metrics(x0 - 3, x1 + 3)
    err = np.asarray(err or [0.0])
    m.update(track_rmse_mm=float(np.sqrt(np.mean(err ** 2))), track_max_mm=float(np.abs(err).max()),
             temp_max_c=float(max(temps)), heat_std=float(np.std(his)), duration_s=env.t)
    return {"metrics": m, "path": env.path, "env": env}
