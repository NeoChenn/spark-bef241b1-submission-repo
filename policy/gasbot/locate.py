from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from policy.gasbot.plume import RHO_CH4_G_M3, GaussianPlume


def _perp(w):
    return np.array([-w[1], w[0]])


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


class SourceEstimator:
    def __init__(self, xlim, ylim, res: float = 0.2, p_max: float = 0.85, p_false: float = 0.01,
                 half_angle_deg: float = 22.0, sigma0: float = 0.25, reach_m: float = 18.0):
        xs = np.arange(xlim[0], xlim[1] + 1e-9, res)
        ys = np.arange(ylim[0], ylim[1] + 1e-9, res)
        X, Y = np.meshgrid(xs, ys)
        self.cells = np.stack([X.ravel(), Y.ravel()], 1)
        self.shape = X.shape
        self.logp = np.zeros(len(self.cells))
        self.p_max, self.p_false = p_max, p_false
        self.k = math.tan(math.radians(half_angle_deg))
        self.sigma0, self.reach = sigma0, reach_m
        self.n_hits = self.n_obs = self.n_bear = 0

    def restrict(self, allowed: np.ndarray) -> None:
        self.logp = np.where(allowed.ravel(), self.logp, -np.inf)

    def p_hit(self, pos, wind_unit) -> np.ndarray:
        w = unit(wind_unit)
        d = np.asarray(pos, float)[:2] - self.cells
        x, y = d @ w, d @ _perp(w)
        sig = self.sigma0 + self.k * np.maximum(x, 0.0)
        p = self.p_max * np.exp(-y ** 2 / (2 * sig ** 2)) * np.exp(-np.maximum(x, 0.0) / self.reach)
        p = np.where(x > -0.3, p, 0.0)
        return self.p_false + (1 - self.p_false) * p

    def observe(self, pos, wind_unit, hit: bool) -> None:
        p = self.p_hit(pos, wind_unit)
        self.logp += np.log(p if hit else 1.0 - 0.6 * p)
        self.n_obs += 1
        self.n_hits += int(hit)
        self._norm()

    def bearing(self, pos, bearing: float, sigma: float) -> None:
        d = self.cells - np.asarray(pos, float)[:2]
        a = np.arctan2(d[:, 1], d[:, 0]) - bearing
        kappa = 1.0 / max(sigma, 1e-3) ** 2
        self.logp += 0.9 * np.log(np.exp(kappa * (np.cos(a) - 1.0)) + 0.05)
        self.n_bear += 1
        self._norm()

    def _norm(self) -> None:
        m = np.max(self.logp)
        if np.isfinite(m):
            self.logp -= m

    def posterior(self) -> np.ndarray:
        p = np.exp(self.logp)
        return p / p.sum()

    def estimate(self) -> tuple[np.ndarray, float, np.ndarray]:
        p = self.posterior()
        mean = p @ self.cells
        d = self.cells - mean
        cov = (d * p[:, None]).T @ d
        sigma = float(math.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0)))
        return mean, sigma, self.cells[int(np.argmax(p))]


def flow_estimate(samples, source, wind_unit, u: float, z_source: float, meander: float = 0.21) -> float | None:
    if not samples:
        return None
    pl = GaussianPlume([source[0], source[1], z_source], 1.0)
    pts = np.array([s[0] for s in samples])
    c = np.maximum(np.array([s[1] for s in samples]) - 2.0, 0.0) * RHO_CH4_G_M3 / 1e6
    q = 1.0
    for _ in range(4):
        pl.set_flow(q)
        m = pl.concentration(pts, wind_unit=wind_unit, u=u, q=1.0, meander=meander)
        den = float(m @ m)
        if den <= 1e-12:
            return None
        q = max(1e-3, float(m @ c) / den)
    return q


def acoustic_flow(db: float, expected_db: float, r: float, l1m_db: float, absorb_db_m: float) -> float | None:
    excess = 10 ** (db / 10) - 10 ** (expected_db / 10)
    if excess <= 0:
        return None
    r = max(r, 0.3)
    return 10 ** ((10 * math.log10(excess) - l1m_db + 20 * math.log10(r) + absorb_db_m * r) / 10)


@dataclass
class SeekConfig:
    hit_lel: float = 0.01
    surge_speed: float = 0.6
    cast_speed: float = 0.5
    lost_s: float = 2.5
    cast_width: float = 1.2
    cast_grow: float = 1.4
    cast_max: float = 5.0
    upwind_drift: float = 0.25
    acoustic_w: float = 0.8
    gradient_w: float = 0.6
    estimate_w: float = 0.5
    trust_sigma: float = 1.5
    relocate_s: float = 14.0


class CastSurge:
    def __init__(self, cfg: SeekConfig | None = None):
        self.cfg = cfg or SeekConfig()
        self.mode = "SURGE"
        self.last_hit_t = None
        self.last_hit_pos = None
        self.cast_dir = 1.0
        self.cast_width = self.cfg.cast_width
        self.cast_origin = None

    def hit(self, lel_trunk: float, lel_tip: float) -> bool:
        return max(lel_trunk, lel_tip) >= self.cfg.hit_lel

    def command(self, t: float, pos, lel_trunk: float, lel_tip: float, tip_rel, wind_unit,
                bearing: float | None = None, snr_db: float = 0.0, estimate=None, blocked: bool = False) -> np.ndarray:
        c = self.cfg
        p = np.asarray(pos, float)[:2]
        up = -unit(wind_unit)
        side = _perp(up)
        if self.hit(lel_trunk, lel_tip):
            self.last_hit_t, self.last_hit_pos = t, p.copy()
            self.mode = "SURGE"
            self.cast_width = c.cast_width
            self.cast_origin = None
            v = up.copy()
            rel = np.asarray(tip_rel, float)[:2]
            if np.linalg.norm(rel) > 0.1:
                g = (lel_tip - lel_trunk) / max(lel_tip, lel_trunk, 1e-6)
                v += c.gradient_w * g * unit(rel - (rel @ up) * up)
            if bearing is not None and snr_db > 3.0:
                v += c.acoustic_w * min(1.0, snr_db / 15.0) * np.array([math.cos(bearing), math.sin(bearing)])
            if estimate is not None and estimate[1] < c.trust_sigma:
                v += c.estimate_w * unit(np.asarray(estimate[0], float)[:2] - p)
            return unit(v) * c.surge_speed
        if self.last_hit_t is None or t - self.last_hit_t < c.lost_s:
            if bearing is not None and snr_db > 6.0:
                return unit(np.array([math.cos(bearing), math.sin(bearing)]) + 0.3 * up) * c.surge_speed
            if self.last_hit_t is None and estimate is not None and estimate[1] < c.trust_sigma:
                return unit(np.asarray(estimate[0], float)[:2] - p) * c.surge_speed
            return up * c.surge_speed if self.last_hit_t is not None else np.zeros(2)
        if t - self.last_hit_t > c.relocate_s:
            trusted = estimate is not None and estimate[1] < c.trust_sigma
            target = np.asarray(estimate[0], float)[:2] if trusted else self.last_hit_pos - 1.0 * up
            if np.linalg.norm(target - p) > 0.4 and not blocked:
                self.mode = "RELOCATE"
                return unit(target - p) * c.surge_speed
            self.last_hit_t = t - c.lost_s
            self.cast_origin = None
            self.cast_width = c.cast_width
        self.mode = "CAST"
        if self.cast_origin is None:
            self.cast_origin = p.copy()
        off = (p - self.cast_origin) @ side
        if off * self.cast_dir >= self.cast_width or blocked:
            self.cast_dir = -self.cast_dir
            self.cast_width = min(self.cast_width * c.cast_grow, c.cast_max)
        v = self.cast_dir * side + c.upwind_drift * up
        if bearing is not None and snr_db > 6.0:
            v += 0.5 * np.array([math.cos(bearing), math.sin(bearing)])
        return unit(v) * c.cast_speed
