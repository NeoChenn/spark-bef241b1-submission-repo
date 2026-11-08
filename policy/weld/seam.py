from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SeamConfig:
    length_mm: float = 320.0
    half_w_mm: float = 20.0
    dx_mm: float = 1.0
    dy_mm: float = 0.5
    pipe_r_mm: float = 150.0
    wander_mm: float = 3.5
    misalign_mm: float = 1.5
    groove_half_mm: float = 5.0
    groove_depth_mm: float = 2.0
    grind_depth_mm: float = 0.8
    crown_mm: float = 0.6
    crack_w_mm: float = 0.6
    eta: float = 0.8
    hi_lo: float = 0.8
    hi_hi: float = 1.2
    max_interpass_c: float = 250.0
    ambient_c: float = 20.0
    heat_gain: float = 2.0
    tau_s: float = 40.0
    min_cover: float = 0.97

    @property
    def nx(self) -> int:
        return int(round(self.length_mm / self.dx_mm))

    @property
    def ny(self) -> int:
        return int(round(2 * self.half_w_mm / self.dy_mm)) + 1


def heat_input(current_a: float, voltage_v: float, speed_mm_s: float, eta: float = 0.8) -> float:
    return eta * voltage_v * current_a / max(speed_mm_s, 1e-6) / 1000.0


def seg_dist(px, py, a, b):
    d = b - a
    t = np.clip(((px - a[0]) * d[0] + (py - a[1]) * d[1]) / max(float(d @ d), 1e-12), 0.0, 1.0)
    return np.hypot(px - a[0] - t * d[0], py - a[1] - t * d[1])


class Seam:
    def __init__(self, cfg: SeamConfig | None = None, seed: int = 0, crack: bool = True):
        self.cfg = c = cfg or SeamConfig()
        rng = np.random.default_rng(seed)
        self.xs = (np.arange(c.nx) + 0.5) * c.dx_mm
        self.ys = np.linspace(-c.half_w_mm, c.half_w_mm, c.ny)
        amp = rng.uniform(0.6, 1.0) * c.wander_mm
        lam = rng.uniform(150.0, 260.0)
        phi = rng.uniform(0, 2 * np.pi)
        off = rng.uniform(-1, 1) * c.misalign_mm
        self.center = np.clip(off + amp * np.sin(2 * np.pi * self.xs / lam + phi),
                              -c.wander_mm - c.misalign_mm, c.wander_mm + c.misalign_mm)
        self.crack = np.zeros((0, 2))
        self.crack_mask = np.zeros((c.nx, c.ny), bool)
        if crack:
            x0 = rng.uniform(0.25, 0.5) * c.length_mm
            n = int(rng.integers(6, 11))
            px = np.sort(x0 + np.linspace(0, rng.uniform(40, 80), n) + rng.normal(0, 1.0, n))
            py = np.interp(px, self.xs, self.center) + np.clip(rng.normal(0, 1.2, n), -3, 3)
            self.crack = np.stack([px, py], 1)
            gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
            d = np.min([seg_dist(gx, gy, self.crack[i], self.crack[i + 1]) for i in range(n - 1)], axis=0)
            self.crack_mask = d < c.crack_w_mm
        self.ground = np.zeros((c.nx, c.ny), bool)
        self.bead = np.zeros((c.nx, c.ny), bool)
        self.heat = np.full(c.nx, np.nan)
        self.hot = np.zeros(c.nx, bool)
        self.temp_c = c.ambient_c

    def col(self, x: float) -> int:
        return int(np.clip(np.floor(x / self.cfg.dx_mm), 0, self.cfg.nx - 1))

    def rows(self, y: float, width: float) -> np.ndarray:
        return np.abs(self.ys - y) < width / 2

    def center_at(self, x: float) -> float:
        return float(np.interp(x, self.xs, self.center))

    def crack_extent(self) -> tuple[float, float] | None:
        if not len(self.crack):
            return None
        return float(self.crack[:, 0].min()), float(self.crack[:, 0].max())

    def open_crack(self) -> np.ndarray:
        return self.crack_mask & ~self.ground

    def height(self, x: float, ys) -> np.ndarray:
        c, ys = self.cfg, np.asarray(ys, float)
        i = self.col(x)
        h = -c.groove_depth_mm * np.clip(1 - np.abs(ys - self.center_at(x)) / c.groove_half_mm, 0, 1)
        j = np.round((ys + c.half_w_mm) / c.dy_mm).astype(int)
        ok = (j >= 0) & (j < c.ny)
        jj = np.clip(j, 0, c.ny - 1)
        h = h - np.where(ok & self.ground[i, jj], c.grind_depth_mm, 0.0)
        h = h - np.where(ok & self.open_crack()[i, jj], 0.4, 0.0)
        return np.where(ok & self.bead[i, jj], c.crown_mm, h)

    def grind(self, x: float, y: float, width: float) -> None:
        self.ground[self.col(x), self.rows(y, width)] = True

    def deposit(self, x: float, y: float, width: float, hi: float, dt: float) -> None:
        c, i = self.cfg, self.col(x)
        self.bead[i, self.rows(y, width)] = True
        self.heat[i] = hi
        self.hot[i] |= self.temp_c > c.max_interpass_c
        self.temp_c += c.heat_gain * hi
        self.cool(dt)

    def cool(self, dt: float) -> None:
        a = self.cfg.ambient_c
        self.temp_c = a + (self.temp_c - a) * float(np.exp(-dt / self.cfg.tau_s))

    def band(self, i0: int, i1: int) -> np.ndarray:
        return np.abs(self.ys[None, :] - self.center[i0:i1, None]) <= self.cfg.groove_half_mm

    def metrics(self, x0: float, x1: float) -> dict:
        c = self.cfg
        i0, i1 = self.col(x0), self.col(x1) + 1
        band = self.band(i0, i1)
        cov_col = (self.bead[i0:i1] & band).sum(1) / np.maximum(band.sum(1), 1)
        hi = self.heat[i0:i1]
        bad_heat = ~((hi >= c.hi_lo) & (hi <= c.hi_hi))
        bad = (cov_col < c.min_cover) | bad_heat | self.hot[i0:i1]
        cm = self.crack_mask
        left = cm & ~(self.ground & self.bead)
        return {"coverage": float((self.bead[i0:i1] & band).sum() / max(band.sum(), 1)),
                "defect_rate": float(bad.mean()),
                "lack_of_fusion": float((cov_col < c.min_cover).mean()),
                "heat_out_of_band": float(bad_heat.mean()),
                "crack_left": float(left.sum() / cm.sum()) if cm.any() else 0.0,
                "heat_mean": float(np.nanmean(hi)) if np.isfinite(hi).any() else 0.0}
