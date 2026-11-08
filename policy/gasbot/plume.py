from __future__ import annotations

import math
from collections import deque

import numpy as np

RHO_CH4_G_M3 = 656.0
BACKGROUND_PPM = 2.0


class Wind:
    def __init__(self, speed=3.0, direction=0.0, meander_deg=12.0, tau_s=8.0, gust=0.15, seed=0):
        self.speed0, self.dir0 = float(speed), float(direction)
        self.sigma = math.radians(meander_deg)
        self.tau, self.gust = float(tau_s), float(gust)
        self.rng = np.random.default_rng(seed)
        self.d_dir = 0.0
        self.d_speed = 0.0

    def step(self, dt: float) -> None:
        a = math.exp(-dt / self.tau)
        s = math.sqrt(max(1.0 - a * a, 0.0))
        self.d_dir = a * self.d_dir + s * self.sigma * self.rng.normal()
        self.d_speed = a * self.d_speed + s * self.gust * self.rng.normal()

    @property
    def direction(self) -> float:
        return self.dir0 + self.d_dir

    @property
    def u(self) -> float:
        return max(0.5, self.speed0 * (1.0 + self.d_speed))

    def unit(self, mean: bool = False) -> np.ndarray:
        a = self.dir0 if mean else self.direction
        return np.array([math.cos(a), math.sin(a)])


class GaussianPlume:
    def __init__(self, source, flow_gs: float = 1.0, wind: Wind | None = None, sigma0: float = 0.06,
                 spread: float = 0.16, deck_z: float = 0.0, puff: float = 0.25, seed: int = 0):
        self.source = np.asarray(source, np.float64)
        self.q = float(flow_gs)
        self.wind = wind or Wind(seed=seed)
        self.own_wind = wind is None
        self.sigma0, self.spread, self.deck_z, self.puff = sigma0, spread, deck_z, puff
        self.rng = np.random.default_rng(seed + 1)
        self.t = 0.0
        self.hist: deque = deque([(0.0, self.q)], maxlen=3000)
        self.fluct = 1.0

    def set_flow(self, q: float) -> None:
        self.q = max(0.0, float(q))
        last = self.hist[-1][1]
        if abs(self.q - last) > 0.005 * max(last, 1e-3) or (self.q == 0) != (last == 0):
            self.hist.append((self.t, self.q))

    def flow_rate(self) -> float:
        return self.q

    def step(self, dt: float) -> None:
        self.t += dt
        if self.own_wind:
            self.wind.step(dt)
        self.fluct = math.exp(self.puff * self.rng.normal() - 0.5 * self.puff ** 2)

    def _q_at(self, t: np.ndarray) -> np.ndarray:
        ts = np.array([h[0] for h in self.hist])
        qs = np.array([h[1] for h in self.hist])
        return qs[np.clip(np.searchsorted(ts, t, "right") - 1, 0, len(qs) - 1)]

    def quiet(self) -> bool:
        return self.q == 0 and all(h[1] == 0 for h in self.hist)

    def concentration(self, points, wind_unit=None, u=None, q=None, meander=0.0) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, np.float64))
        w = self.wind.unit() if wind_unit is None else np.asarray(wind_unit, float)
        u = self.wind.u if u is None else float(u)
        rel = p[:, :2] - self.source[:2]
        x = rel @ w
        y = rel @ np.array([-w[1], w[0]])
        xe = np.maximum(x, 0.0) + 0.05
        sig = self.sigma0 + 0.04 * math.sqrt(self.q) + self.spread * xe
        sy = np.sqrt(sig ** 2 + (meander * xe) ** 2)
        qq = self._q_at(self.t - np.maximum(x, 0.0) / u) if q is None else np.full(len(p), float(q))
        zs, z = self.source[2] - self.deck_z, p[:, 2] - self.deck_z
        vert = np.exp(-(z - zs) ** 2 / (2 * sig ** 2)) + np.exp(-(z + zs) ** 2 / (2 * sig ** 2))
        c = qq / (2 * math.pi * u * sig * sy) * np.exp(-y ** 2 / (2 * sy ** 2)) * vert
        return np.where(x < 0, c * np.exp(x / 0.15), c)

    def sample(self, points) -> np.ndarray:
        return BACKGROUND_PPM + self.concentration(points) * self.fluct / RHO_CH4_G_M3 * 1e6


class PlumeField:
    def __init__(self, wind: Wind, sources: dict, seed: int = 0, background_ppm: float = BACKGROUND_PPM):
        self.wind = wind
        self.background = background_ppm
        self.plumes = {k: GaussianPlume(v, 0.0, wind, seed=seed + 7 * i) for i, (k, v) in enumerate(sources.items())}
        self.t = 0.0

    def set_flow(self, key, q: float) -> None:
        self.plumes[key].set_flow(q)

    def step(self, dt: float) -> None:
        self.t += dt
        self.wind.step(dt)
        for p in self.plumes.values():
            p.step(dt)

    def flow_rate(self) -> float:
        return float(sum(p.flow_rate() for p in self.plumes.values()))

    def sample(self, points) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, np.float64))
        out = np.full(len(pts), self.background)
        for p in self.plumes.values():
            if not p.quiet():
                out += p.sample(pts) - BACKGROUND_PPM
        return out

    def leaks(self) -> list[tuple[np.ndarray, float]]:
        return [(p.source, p.flow_rate()) for p in self.plumes.values() if p.flow_rate() > 0]


class SolverField:
    def __init__(self, plume, key, wind: Wind, background_ppm: float = BACKGROUND_PPM):
        self.plume, self.key, self.wind, self.background = plume, key, wind, background_ppm
        self.q1 = float(plume.release.mass_flow) * 1000.0
        self.q = 0.0
        self.t = 0.0
        plume.set_rate(0.0)

    def set_flow(self, key, q: float) -> None:
        if key == self.key:
            self.q = max(0.0, float(q))
            self.plume.set_rate(self.q / self.q1 if self.q1 > 0 else 0.0)

    def step(self, dt: float) -> None:
        self.t += dt
        self.wind.step(dt)
        self.plume.step(dt)

    def flow_rate(self) -> float:
        return self.q

    def sample(self, points) -> np.ndarray:
        return self.background + np.asarray(self.plume.sample(np.atleast_2d(np.asarray(points, float))), float)

    def leaks(self) -> list[tuple[np.ndarray, float]]:
        return [(np.asarray(self.plume.source, float), self.q)] if self.q > 0 else []
