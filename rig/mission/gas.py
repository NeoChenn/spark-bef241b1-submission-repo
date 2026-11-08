from __future__ import annotations

import math

import numpy as np

from policy.gasbot.locate import CastSurge, SeekConfig, SourceEstimator, unit
from policy.gasbot.plume import BACKGROUND_PPM, RHO_CH4_G_M3, GaussianPlume, Wind
from policy.gasbot.sensors import LEL_PPM, PointGasSensor

FLOW_GS = 16.0
WIND_TOWARD_DEG = 240.0
WIND_MS = 4.0
DETECTOR_M = 1.4
TRUNK_Z = 0.45
TIP_Z = 1.0
TIP_M = 0.5
WINDOW_M = 12.0
GRID_M = 0.25
START_M = 11.0
DOWNWIND_M = 3.0
FOUND_M = 0.9
SEEK_MAX_S = 150.0
WALK_MS = 0.6
SUB_S = 0.25
ALARM_LEL = 0.10
MARGIN_M = 1.0
PF_N = 1500
PF_EVERY_S = 1.0
METHODS = {"cast": "cast-and-surge", "infotaxis": "particle filter + infotaxis"}
FADE_S = 90.0


def wind_from_weather(w: dict | None) -> tuple[float, float]:
    w = w or {}
    level = w.get("level") or "calm"
    toward = WIND_TOWARD_DEG if level == "calm" or w.get("wind_dir") is None else float(w["wind_dir"])
    return toward, WIND_MS + 2.0 * float(w.get("intensity") or 0.0)


class GasLeak:
    def __init__(self, flanges: dict, bounds: dict | None = None, seed: int = 0, field=None):
        self.flanges = {k: np.asarray(v, float) for k, v in flanges.items()}
        self.bounds, self.seed, self.field = bounds, seed, field
        self.toward, self.speed = WIND_TOWARD_DEG, WIND_MS
        self.wind = Wind(self.speed, math.radians(self.toward), seed=seed)
        self.fid = None
        self.plume = None
        self.on = False
        self.isolated = False
        self.t = 0.0
        self.seek = None
        self.robot_id = None
        self.found = None
        self.lel = {}
        self.fading: list = []
        self.queued: list = []
        self.method = "cast"

    def _plume(self, fid: str, flow_gs: float) -> GaussianPlume:
        if fid not in self.flanges:
            raise ValueError(f"unknown flange {fid!r}; expected one of {sorted(self.flanges)}")
        src = self.flanges[fid]
        p = GaussianPlume(src, flow_gs, self.wind, deck_z=src[2] - 1.1, seed=self.seed)
        p.t = self.t
        return p

    def start(self, fid: str, flow_gs: float = FLOW_GS) -> None:
        self._take(fid, self._plume(fid, flow_gs))

    def _take(self, fid: str, plume) -> None:
        if self.plume is not None and not self.plume.quiet():
            self.fading.append(self.plume)
        self.fid, self.on, self.isolated, self.found, self.seek, self.robot_id = fid, True, False, None, None, None
        self.plume = plume

    def add(self, fid: str, flow_gs: float = FLOW_GS) -> str:
        if not self.on:
            self.start(fid, flow_gs)
            return "now"
        self.queued.append((fid, self._plume(fid, flow_gs)))
        return "queued"

    def promote(self) -> str | None:
        if self.on or not self.queued:
            return None
        fid, p = self.queued.pop(0)
        self._take(fid, p)
        return fid

    def others(self) -> list:
        return self.fading + [p for _, p in self.queued]

    def stop(self, why: str = "fixed") -> None:
        self.on = False
        self.isolated = why == "isolated"
        if self.plume is not None:
            self.plume.set_flow(0.0)

    def set_wind(self, toward_deg: float, speed: float) -> None:
        self.toward, self.speed = float(toward_deg), float(speed)
        self.wind.dir0, self.wind.speed0 = math.radians(self.toward), self.speed

    @property
    def wind_unit(self) -> np.ndarray:
        return self.wind.unit(mean=True)

    @property
    def active(self) -> bool:
        return self.plume is not None and not self.plume.quiet()

    def source(self) -> np.ndarray | None:
        return None if self.fid is None else self.flanges[self.fid]

    def ppm(self, points) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, float))
        f = self.field() if self.field is not None else None
        if f is not None and self.on:
            return BACKGROUND_PPM + np.asarray(f.sample(p), float)
        ps = ([self.plume] if self.plume is not None else []) + self.others()
        c = sum((q.concentration(p) * q.fluct for q in ps), np.zeros(len(p)))
        return BACKGROUND_PPM + c / RHO_CH4_G_M3 * 1e6

    def lel_at(self, points) -> np.ndarray:
        return self.ppm(points) / LEL_PPM

    def detectors(self) -> list[dict]:
        off = DETECTOR_M * np.array([math.cos(math.radians(WIND_TOWARD_DEG)), math.sin(math.radians(WIND_TOWARD_DEG)), 0.0])
        ids = sorted(self.flanges)
        pts = [self.flanges[k] + off for k in ids]
        lel = self.lel_at(pts)
        return [{"id": f"GD-{k}", "at": f"Flange {k}", "zone": k, "lel": round(float(v), 4), "ppm": round(float(v) * LEL_PPM)}
                for k, v in zip(ids, lel)]

    def step(self, dt: float) -> None:
        self.t += dt
        self.wind.step(dt)
        if self.plume is not None:
            self.plume.step(dt)
        for q in self.others():
            q.step(dt)
        self.fading = [q for q in self.fading if q.q > 0 or self.t - q.hist[-1][0] < FADE_S]
        if self.seek is not None:
            self.seek.step(self, dt)

    def begin_seek(self, robot_id: str, xyz, deck_z: float) -> "Seek":
        self.robot_id = robot_id
        cls = InfoSeek if self.method == "infotaxis" and INFOTAXIS else Seek
        self.seek = cls(self, np.asarray(xyz, float), deck_z, self.bounds, self.seed)
        return self.seek


class Seek:
    def __init__(self, leak: GasLeak, xyz, deck_z: float, bounds=None, seed: int = 0):
        src = leak.source()
        self.pos = np.asarray(xyz, float)[:2].copy()
        self.deck_z, self.bounds = deck_z, bounds
        self.yaw = 0.0
        self.t = 0.0
        self.cs = CastSurge(SeekConfig())
        self.est = SourceEstimator((src[0] - WINDOW_M, src[0] + WINDOW_M), (src[1] - WINDOW_M, src[1] + WINDOW_M), GRID_M)
        self.trunk = PointGasSensor("IR-T", "ir", t90=2.0, noise_ppm=25.0, seed=seed + 11)
        self.tip = PointGasSensor("CAT-A", "catalytic", t90=3.0, noise_ppm=40.0, seed=seed + 12)
        self.mode = "DOWNWIND"
        self.hits = 0
        self.estimate = None
        self.done = False
        self.err_m = None
        self.path = [self.pos.round(2).tolist()]
        self.lels = [0.0]

    def tip_xy(self) -> np.ndarray:
        return self.pos + TIP_M * np.array([math.cos(self.yaw), math.sin(self.yaw)])

    def _clamp(self, p) -> np.ndarray:
        b = self.bounds
        if not b:
            return p
        return np.array([np.clip(p[0], b["xmin"] + MARGIN_M, b["xmax"] - MARGIN_M),
                         np.clip(p[1], b["ymin"] + MARGIN_M, b["ymax"] - MARGIN_M)])

    def step(self, leak: GasLeak, dt: float) -> None:
        if self.done:
            return
        self.t += dt
        w = leak.wind_unit
        tip = self.tip_xy()
        ppm = leak.ppm([[*self.pos, self.deck_z + TRUNK_Z], [*tip, self.deck_z + TIP_Z]])
        self.trunk.update(float(ppm[0]), dt)
        self.tip.update(float(ppm[1]), dt)
        hit = self.cs.hit(self.trunk.lel, self.tip.lel)
        self.hits += int(hit)
        self.est.observe(self.pos, w, hit)
        m, sig, _ = self.est.estimate()
        self.estimate = (m, sig)
        src = leak.source()[:2]
        if self.hits == 0:
            self.mode = "DOWNWIND"
            v = unit(src + DOWNWIND_M * w - self.pos) * self.cs.cfg.surge_speed
        else:
            v = self.cs.command(self.t, self.pos, self.trunk.lel, self.tip.lel, tip - self.pos, w,
                                estimate=(m, sig) if self.hits > 3 else None)
            self.mode = self.cs.mode
        if np.linalg.norm(v) > 1e-6:
            self.yaw = math.atan2(v[1], v[0])
        self.pos = self._clamp(self.pos + v * dt)
        if len(self.path) < 600 and np.linalg.norm(self.pos - np.asarray(self.path[-1])) > 0.3:
            self.path.append(self.pos.round(2).tolist())
            self.lels.append(round(float(self.trunk.lel), 4))
        close = float(np.linalg.norm(self.pos - src)) < FOUND_M
        sure = sig < 0.6 and float(np.linalg.norm(self.pos - m)) < 0.7 and self.hits > 5
        if close or sure or self.t > SEEK_MAX_S:
            self.done, self.mode = True, "FOUND"
            if close and not sure:
                m, sig = self.pos.copy(), FOUND_M / 2
                self.estimate = (m, sig)
            self.err_m = round(float(np.linalg.norm(m - src)), 2)
            leak.found = {"flange": leak.fid, "t": round(leak.t, 1), "err_m": self.err_m, "seek_s": round(self.t, 1),
                          "x": round(float(m[0]), 2), "y": round(float(m[1]), 2), "sigma_m": round(sig, 2)}


class PFView:
    def __init__(self, grid: SourceEstimator, pf: ParticleFilter):
        self.grid, self.pf = grid, pf
        self.cells, self.shape = grid.cells, grid.shape

    @property
    def n_hits(self) -> int:
        return self.pf.n_hits

    def posterior(self) -> np.ndarray:
        ny, nx = self.shape
        xs, ys = self.cells[:nx, 0], self.cells[::nx, 1]
        res = float(xs[1] - xs[0])
        ix = np.clip(np.round((self.pf.p[:, 0] - xs[0]) / res).astype(int), 0, nx - 1)
        iy = np.clip(np.round((self.pf.p[:, 1] - ys[0]) / res).astype(int), 0, ny - 1)
        p = np.bincount(iy * nx + ix, weights=self.pf.w, minlength=nx * ny)
        return p / max(float(p.sum()), 1e-12)


try:
    from policy.gasbot.infotaxis import Infotaxis, ParticleFilter
    INFOTAXIS = True
except ImportError:
    Infotaxis, ParticleFilter, INFOTAXIS = object, None, False


class Planner(Infotaxis):
    def __init__(self, pf, clamp):
        super().__init__(pf)
        self.clamp = clamp

    def candidates(self, p) -> np.ndarray:
        a = np.arange(self.cfg.n_dirs) * 2 * math.pi / self.cfg.n_dirs
        d = np.stack([np.cos(a), np.sin(a)], 1)
        q = np.array([self.clamp(x) for x in np.vstack([p + r * d for r in self.cfg.radii])])
        return q[np.linalg.norm(q - p, axis=1) > 0.25]


class InfoSeek(Seek):
    def __init__(self, leak: GasLeak, xyz, deck_z: float, bounds=None, seed: int = 0):
        super().__init__(leak, xyz, deck_z, bounds, seed)
        src = leak.source()
        self.pf = ParticleFilter((src[0] - WINDOW_M, src[0] + WINDOW_M), (src[1] - WINDOW_M, src[1] + WINDOW_M), PF_N,
                                 seed=seed)
        near = [v for v in leak.flanges.values() if abs(v[0] - src[0]) < WINDOW_M and abs(v[1] - src[1]) < WINDOW_M]
        if near:
            self.pf.prior(near)
        self.plan = Planner(self.pf, self._clamp)
        self.est = PFView(self.est, self.pf)
        self.mode = "INFOTAXIS"
        self.v = np.zeros(2)
        self.next_obs = 0.0

    def step(self, leak: GasLeak, dt: float) -> None:
        if self.done:
            return
        self.t += dt
        w = leak.wind_unit
        tip = self.tip_xy()
        ppm = leak.ppm([[*self.pos, self.deck_z + TRUNK_Z], [*tip, self.deck_z + TIP_Z]])
        self.trunk.update(float(ppm[0]), dt)
        self.tip.update(float(ppm[1]), dt)
        if self.t >= self.next_obs:
            self.next_obs = self.t + PF_EVERY_S
            self.pf.observe([([*self.pos, TRUNK_Z], self.trunk.lel), ([*tip, TIP_Z], self.tip.lel)], w, leak.speed)
            self.hits = self.pf.n_hits
            self.v = self.plan.command(self.t, self.pos, self.trunk.lel, self.tip.lel, tip - self.pos, w)
            self.mode = self.plan.mode
        m, sig, _ = self.pf.estimate()
        self.estimate = (m, sig)
        if np.linalg.norm(self.v) > 1e-6:
            self.yaw = math.atan2(self.v[1], self.v[0])
        self.pos = self._clamp(self.pos + self.v * dt)
        if len(self.path) < 600 and np.linalg.norm(self.pos - np.asarray(self.path[-1])) > 0.3:
            self.path.append(self.pos.round(2).tolist())
            self.lels.append(round(float(self.trunk.lel), 4))
        src = leak.source()[:2]
        sure = sig < 0.6 and float(np.linalg.norm(self.pos - m)) < 0.7 and self.hits > 3
        if sure or self.t > SEEK_MAX_S:
            self.done, self.mode = True, "FOUND"
            self.err_m = round(float(np.linalg.norm(m - src)), 2)
            leak.found = {"flange": leak.fid, "t": round(leak.t, 1), "err_m": self.err_m, "seek_s": round(self.t, 1),
                          "x": round(float(m[0]), 2), "y": round(float(m[1]), 2), "sigma_m": round(sig, 2),
                          "method": METHODS["infotaxis"]}
