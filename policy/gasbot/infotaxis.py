from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from policy.gasbot import deck
from policy.gasbot.locate import unit
from policy.gasbot.plume import RHO_CH4_G_M3
from policy.gasbot.sensors import LEL_PPM


def plume_ppm(src, q, pts, wind_unit, u, meander=0.21, sigma0=0.06, spread=0.16, deck_z=0.0) -> np.ndarray:
    src = np.atleast_2d(np.asarray(src, float))
    pts = np.atleast_2d(np.asarray(pts, float))
    q = np.broadcast_to(np.asarray(q, float), (len(src),))
    w = unit(wind_unit)
    rel = pts[:, None, :2] - src[None, :, :2]
    x = rel @ w
    y = rel @ np.array([-w[1], w[0]])
    xe = np.maximum(x, 0.0) + 0.05
    sig = sigma0 + 0.04 * np.sqrt(q)[None, :] + spread * xe
    sy = np.sqrt(sig ** 2 + (meander * xe) ** 2)
    zs, z = src[None, :, 2] - deck_z, pts[:, None, 2] - deck_z
    vert = np.exp(-(z - zs) ** 2 / (2 * sig ** 2)) + np.exp(-(z + zs) ** 2 / (2 * sig ** 2))
    c = q[None, :] / (2 * math.pi * u * sig * sy) * np.exp(-y ** 2 / (2 * sy ** 2)) * vert
    c = np.where(x < 0, c * np.exp(np.minimum(x, 0.0) / 0.15), c)
    return c / RHO_CH4_G_M3 * 1e6


def _h(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


@dataclass
class PFConfig:
    hit_lel: float = 0.01
    p_false: float = 0.02
    p_max: float = 0.9
    log_sigma: float = 1.2
    temper: float = 0.5
    level_sigma_db: float = 2.5
    level_temper: float = 0.3
    bearing_temper: float = 0.9
    meander: float = 0.21
    z: tuple = (0.3, 1.5)
    q_gs: tuple = (0.5, 250.0)
    jitter: tuple = (0.05, 0.05, 0.03, 0.12)


class ParticleFilter:
    def __init__(self, xlim, ylim, n: int = 2000, cfg: PFConfig | None = None, seed: int = 0):
        self.cfg = c = cfg or PFConfig()
        self.lo = np.array([xlim[0], ylim[0], c.z[0], math.log(c.q_gs[0])])
        self.hi = np.array([xlim[1], ylim[1], c.z[1], math.log(c.q_gs[1])])
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.p = self.lo + self.rng.random((n, 4)) * (self.hi - self.lo)
        self.w = np.full(n, 1.0 / n)
        self.n_hits = self.n_obs = self.n_bear = self.resamples = 0
        self.u = 3.0
        self.thr = c.hit_lel * LEL_PPM

    def prior(self, points, sigma: float = 0.6, frac: float = 0.75) -> None:
        pts = np.atleast_2d(np.asarray(points, float))
        k = int(frac * self.n)
        idx = self.rng.integers(len(pts), size=k)
        self.p[:k, :2] = pts[idx, :2] + self.rng.normal(0.0, sigma, (k, 2))
        if pts.shape[1] > 2:
            self.p[:k, 2] = pts[idx, 2] + self.rng.normal(0.0, 0.15, k)
        self.p = np.clip(self.p, self.lo, self.hi)
        self.w[:] = 1.0 / self.n

    def ess(self) -> float:
        return float(1.0 / np.sum(self.w ** 2))

    def resample(self) -> None:
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), pos), self.n - 1)
        self.p = np.clip(self.p[idx] + self.rng.normal(0.0, 1.0, (self.n, 4)) * self.cfg.jitter, self.lo, self.hi)
        self.w[:] = 1.0 / self.n
        self.resamples += 1

    def _update(self, ll) -> None:
        lw = np.log(np.maximum(self.w, 1e-300)) + ll
        lw -= lw.max()
        w = np.exp(lw)
        self.w = w / w.sum()
        if self.ess() < self.n / 2:
            self.resample()

    def predict(self, pts, wind_unit, u=None) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        ppm = plume_ppm(self.p[:, :3], np.exp(self.p[:, 3]), pts, wind_unit, u or self.u, c.meander)
        return c.p_false + (1 - c.p_false) * c.p_max * (1 - np.exp(-ppm / self.thr)), ppm

    def loglik(self, readings, wind_unit) -> np.ndarray:
        c = self.cfg
        pts = np.array([np.asarray(r[0], float)[:3] for r in readings])
        lel = np.array([float(r[1]) for r in readings])
        ph, ppm = self.predict(pts, wind_unit)
        ll = np.zeros(self.n)
        for k in range(len(lel)):
            if lel[k] >= c.hit_lel:
                d = math.log(lel[k] * LEL_PPM) - np.log(ppm[k] + 0.3 * self.thr)
                ll += np.log(ph[k]) - d ** 2 / (2 * c.log_sigma ** 2)
            else:
                ll += np.log(1 - ph[k])
        return ll

    def observe(self, readings, wind_unit, u: float) -> None:
        self.u = float(u)
        ll = self.loglik(readings, wind_unit)
        self.n_obs += 1
        self.n_hits += int(any(float(r[1]) >= self.cfg.hit_lel for r in readings))
        self._update(self.cfg.temper * ll)

    def bearing(self, pos, bearing: float, sigma: float) -> None:
        d = self.p[:, :2] - np.asarray(pos, float)[:2]
        a = np.arctan2(d[:, 1], d[:, 0]) - bearing
        kappa = 1.0 / max(sigma, 1e-3) ** 2
        self._update(self.cfg.bearing_temper * np.log(np.exp(kappa * (np.cos(a) - 1.0)) + 0.05))
        self.n_bear += 1

    def level(self, pos, db: float, expected_db: float, l1m_db: float, absorb_db_m: float) -> None:
        r = np.maximum(np.linalg.norm(self.p[:, :2] - np.asarray(pos, float)[:2], axis=1), 0.3)
        leak = l1m_db + 10 * self.p[:, 3] / math.log(10) - 20 * np.log10(r) - absorb_db_m * r
        pred = 10 * np.log10(10 ** (expected_db / 10) + 10 ** (leak / 10))
        self._update(-self.cfg.level_temper * (db - pred) ** 2 / (2 * self.cfg.level_sigma_db ** 2))

    def cov(self) -> tuple[np.ndarray, np.ndarray]:
        m = self.w @ self.p[:, :2]
        d = self.p[:, :2] - m
        return m, (d * self.w[:, None]).T @ d

    def estimate(self) -> tuple[np.ndarray, float, np.ndarray]:
        m, cov = self.cov()
        return m, float(math.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0))), self.p[int(np.argmax(self.w)), :2]

    def ellipse_area(self) -> float:
        return float(4 * math.pi * math.sqrt(max(np.linalg.det(self.cov()[1]), 0.0)))

    def entropy(self, res: float = 0.25) -> float:
        ij = np.floor((self.p[:, :2] - self.lo[:2]) / res).astype(int)
        _, inv = np.unique(ij, axis=0, return_inverse=True)
        pk = np.bincount(inv.ravel(), weights=self.w)
        pk = pk[pk > 0]
        return float(-(pk * np.log(pk)).sum())

    def flow(self) -> float:
        return float(math.exp(self.w @ self.p[:, 3]))

    def mutual_info(self, cands, wind_unit) -> np.ndarray:
        cands = np.atleast_2d(np.asarray(cands, float))
        k = len(cands)
        pts = np.vstack([np.c_[cands, np.full(k, deck.TRUNK_Z)], np.c_[cands, np.full(k, deck.TIP_Z)]])
        ph, _ = self.predict(pts, wind_unit)
        p = 1 - (1 - ph[:k]) * (1 - ph[k:])
        return _h(p @ self.w) - _h(p) @ self.w

    def belief(self, k: int = 80) -> dict:
        m, sigma, _ = self.estimate()
        idx = self.rng.choice(self.n, size=min(k, self.n), replace=False, p=self.w) if k else []
        return {"kind": "pf", "n": self.n, "sigma_m": round(sigma, 2), "area_m2": round(self.ellipse_area(), 2),
                "entropy": round(self.entropy(), 2), "flow_gs": round(self.flow(), 2), "ess": round(self.ess()),
                "particles": np.round(self.p[idx, :2], 2).tolist()}


@dataclass
class InfotaxisConfig:
    hit_lel: float = 0.01
    speed: float = 0.6
    radii: tuple = (0.6, 1.2, 2.0)
    n_dirs: int = 12
    home_w: float = 0.15
    min_sigma: float = 0.3


class Infotaxis:
    def __init__(self, pf: ParticleFilter, cfg: InfotaxisConfig | None = None):
        self.pf = pf
        self.cfg = cfg or InfotaxisConfig()
        self.mode = "INFOTAXIS"
        self.target = None

    def hit(self, lel_trunk: float, lel_tip: float) -> bool:
        return max(lel_trunk, lel_tip) >= self.cfg.hit_lel

    def candidates(self, p) -> np.ndarray:
        a = np.arange(self.cfg.n_dirs) * 2 * math.pi / self.cfg.n_dirs
        d = np.stack([np.cos(a), np.sin(a)], 1)
        c = np.vstack([p + r * d for r in self.cfg.radii])
        q = np.array([deck.clamp(x)[0] for x in c])
        return q[np.linalg.norm(q - p, axis=1) > 0.25]

    def command(self, t, pos, lel_trunk, lel_tip, tip_rel, wind_unit, bearing=None, snr_db=0.0, estimate=None,
                blocked=False) -> np.ndarray:
        c = self.cfg
        p = np.asarray(pos, float)[:2]
        cands = self.candidates(p)
        if not len(cands):
            return np.zeros(2)
        mi = self.pf.mutual_info(cands, wind_unit)
        xy = self.pf.p[:, :2]
        d0 = self.pf.w @ np.linalg.norm(xy - p, axis=1)
        dc = np.linalg.norm(xy[None, :, :] - cands[:, None, :], axis=-1) @ self.pf.w
        _, sigma, _ = self.pf.estimate()
        home = c.home_w / max(sigma, c.min_sigma) * (d0 - dc)
        k = int(np.argmax(mi + home))
        self.mode = "HOMING" if home[k] > mi[k] else "INFOTAXIS"
        self.target = cands[k]
        return unit(cands[k] - p) * c.speed
