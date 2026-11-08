from __future__ import annotations

import glob
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sim.plume.field import Field
from sim.plume.release import G, M_AIR, P_ATM, R, T_STD, Release, orifice

try:
    import warp as wp
except ImportError:
    sys.path.extend(sorted(glob.glob("/isaac-sim/extscache/omni.warp.core-*")))
    import warp as wp

from sim.plume import kernels as K

REPO = Path(__file__).resolve().parents[2]
FLANGES = REPO / "sim/output/leak_flanges_meta.json"
RIG_USD = REPO / "sim/output/full_rig_leak.usda"
DECK_Z = 42.5
RHO_AIR = P_ATM * M_AIR / (R * T_STD)


@dataclass
class Wind:
    speed: float = 5.0
    heading_deg: float = 90.0
    storm_level: float = 0.0
    gust: float = 0.45

    def at(self, t: float) -> np.ndarray:
        s = self.storm_level
        g = 1.0 + self.gust * s * (0.6 * math.sin(2 * math.pi * 0.23 * t) + 0.4 * math.sin(2 * math.pi * 0.071 * t + 1.3))
        h = math.radians(self.heading_deg + 14.0 * s * math.sin(2 * math.pi * 0.05 * t))
        return self.speed * g * np.array([math.cos(h), math.sin(h), 0.0])

    @property
    def diffusivity(self) -> float:
        return 0.02 + 0.015 * self.speed * (1.0 + self.storm_level)

    def to_json(self, t: float = 0.0) -> dict:
        w = self.at(t)
        return {"speed_m_s": round(float(np.linalg.norm(w)), 2), "heading_deg": round(math.degrees(math.atan2(w[1], w[0])), 1),
                "storm_level": self.storm_level}


def _vec(a) -> "wp.vec3":
    return wp.vec3(*[float(x) for x in a])


class Plume:
    def __init__(self, origin, shape, dx: float, source, release: Release, wind: Wind | None = None,
                 jet_dir=(0.0, 0.0, 1.0), solid=None, diffusivity: float | None = None, buoyancy: bool = True,
                 jet: bool = True, static_flow: bool = False, u_cap: float = 30.0, pressure_iters: int = 40,
                 cfl: float = 0.3, sponge: int = 3, relax_s: float = 0.6, device=None, meta: dict | None = None):
        wp.init()
        self.device = device or wp.get_preferred_device()
        self.origin = np.asarray(origin, np.float64)
        self.shape = tuple(int(s) for s in shape)
        self.dx = float(dx)
        self.source = np.asarray(source, np.float64)
        self.release = release
        self.wind = wind or Wind()
        self.kd = self.wind.diffusivity if diffusivity is None else float(diffusivity)
        self.buoy = G * release.gas.buoyancy if buoyancy else 0.0
        self.jet_on, self.static, self.cap, self.iters, self.cfl = jet, static_flow, u_cap, pressure_iters, cfl
        self.sponge, self.relax_s = float(sponge), relax_s
        self.meta = meta or {}
        self.t = 0.0
        self.steps = 0
        self.rate = 1.0
        self._field = None
        nx, ny, nz = self.shape
        sol = np.zeros(self.shape, np.int32) if solid is None else np.asarray(solid, np.int32)
        self.solid_np = sol.astype(bool)
        dev = self.device
        self.solid = wp.array(sol, dtype=wp.int32, device=dev)
        w0 = self.wind.at(0.0)
        self.vel = [wp.full((nx + 1, ny, nz), float(w0[0]), dtype=float, device=dev),
                    wp.full((nx, ny + 1, nz), float(w0[1]), dtype=float, device=dev),
                    wp.full((nx, ny, nz + 1), float(w0[2]), dtype=float, device=dev)]
        self.tmp = [wp.zeros_like(a) for a in self.vel]
        self.c = wp.zeros(self.shape, dtype=float, device=dev)
        self.c2 = wp.zeros(self.shape, dtype=float, device=dev)
        self.phi = wp.zeros(self.shape, dtype=float, device=dev)
        self.div = wp.zeros(self.shape, dtype=float, device=dev)
        self.red = wp.zeros(5, dtype=float, device=dev)
        cells, weights = self._source_cells(self.source)
        self.src_cells = wp.array(cells, dtype=wp.vec3i, device=dev)
        self.src_w = wp.array(weights, dtype=float, device=dev)
        self.jet_cell = tuple(int(v) for v in cells[int(np.argmax(weights))])
        d = np.asarray(jet_dir, np.float64)
        self.jet_dir = d / (np.linalg.norm(d) or 1.0)
        self._apply_solid(0.0)
        self._project(max(200, self.iters * 4))
        self._speed = self._max_speed()

    @property
    def cell_volume(self) -> float:
        return self.dx ** 3

    @property
    def lel(self) -> float:
        return self.release.gas.lel

    def _source_cells(self, p) -> tuple[np.ndarray, np.ndarray]:
        g = (np.asarray(p) - self.origin) / self.dx - 0.5
        n = np.asarray(self.shape)
        i0 = np.clip(np.floor(g).astype(int), 0, n - 2)
        f = np.clip(g - i0, 0.0, 1.0)
        cells, ws = [], []
        for o in np.ndindex(2, 2, 2):
            idx = i0 + o
            w = np.prod(np.where(o, f, 1.0 - f))
            if w > 1e-6 and not self.solid_np[tuple(idx)]:
                cells.append(idx)
                ws.append(w)
        if not cells:
            free = np.argwhere(~self.solid_np)
            cells = [free[np.argmin(np.linalg.norm(free - (g + 0.5), axis=1))]]
            ws = [1.0]
        ws = np.asarray(ws) / np.sum(ws)
        return np.asarray(cells, np.int32), ws.astype(np.float32)

    def _launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)

    def _apply_solid(self, dt: float, wind=None):
        n = _vec(self.shape)
        wv = _vec(self.wind.at(self.t) if wind is None else wind)
        relax = 1.0 - math.exp(-dt / self.relax_s) if dt > 0 else 0.0
        for a, f in enumerate(self.vel):
            self._launch(K.body_forces, f.shape, [f, self.c, self.solid, a, dt, self.buoy, wv, n, self.sponge, relax, self.cap])

    def _project(self, iters: int):
        self._launch(K.divergence, self.shape, [*self.vel, self.solid, self.div])
        for _ in range(iters):
            for color in (0, 1):
                self._launch(K.sor, self.shape, [self.phi, self.div, self.solid, color, 1.7])
        for a, f in enumerate(self.vel):
            self._launch(K.project, f.shape, [f, self.phi, self.solid, a])

    def _max_speed(self) -> float:
        self.red.zero_()
        for f in self.vel:
            self._launch(K.reduce_speed, f.shape, [f, self.red])
        return float(self.red.numpy()[4])

    def _flow_step(self, dt: float, wind):
        n = _vec(self.shape)
        wv = _vec(wind)
        for a, f in enumerate(self.vel):
            self._launch(K.advect_vel, f.shape, [f, *self.vel, self.tmp[a], a, dt / self.dx, wv, n])
        self.vel, self.tmp = self.tmp, self.vel
        if self.jet_on and self.release.momentum > 0:
            acc = self.jet_dir * self.release.momentum * self.rate / (RHO_AIR * self.cell_volume)
            self._launch(K.add_jet, 1, [*self.vel, wp.vec3i(*self.jet_cell), _vec(acc), dt])
        self._apply_solid(dt, wind)
        self._project(self.iters)
        self._speed = self._max_speed()

    def step(self, dt: float) -> dict:
        wind = self.wind.at(self.t + dt)
        if not self.static:
            self._flow_step(dt, wind)
        nsub = max(1, math.ceil(self._speed * dt / (self.cfl * self.dx)), math.ceil(8.0 * self.kd * dt / self.dx ** 2))
        h = dt / nsub
        q = self.release.volume_flow * self.rate * h / self.cell_volume
        for _ in range(nsub):
            self._launch(K.transport, self.shape, [self.c, self.c2, *self.vel, self.solid, h / self.dx, self.kd * h / self.dx ** 2])
            self.c, self.c2 = self.c2, self.c
            if q > 0:
                self._launch(K.add_source, len(self.src_w), [self.c, self.src_cells, self.src_w, q])
        self.t += dt
        self.steps += 1
        self._field = None
        self.substeps = nsub
        return self.stats()

    def set_rate(self, rate: float) -> None:
        self.rate = float(max(0.0, rate))

    def set_leak(self, hole_mm: float | None = None, bar: float | None = None, rate: float | None = None) -> dict:
        m = self.meta
        if hole_mm is not None or bar is not None:
            m["hole_mm"] = float(m.get("hole_mm", 3.0) if hole_mm is None else hole_mm)
            m["bar_g"] = float(m.get("bar_g", 20.0) if bar is None else bar)
            self.release = orifice(m["hole_mm"], m["bar_g"], h2s=m.get("h2s", 0.0))
        if rate is not None:
            self.set_rate(rate)
        return self.flow_rate()

    def stats(self) -> dict:
        self.red.zero_()
        self._launch(K.reduce_stats, self.shape, [self.c, 0.1 * self.lel, 0.5 * self.lel, self.red])
        s = self.red.numpy()
        v = self.cell_volume
        rho = self.release.gas.density()
        return {"t": round(self.t, 3), "mass_kg": float(s[0] * v * rho), "gas_m3": float(s[0] * v),
                "vol_10lel_m3": float(s[1] * v), "vol_50lel_m3": float(s[2] * v),
                "max_lel_pct": float(s[3] / self.lel * 100.0), "max_speed_m_s": float(self._speed)}

    def flow_rate(self) -> dict:
        out = self.release.to_json()
        for k in ("kg_s", "sm3_h", "nm3_h"):
            out[k] = round(out[k] * self.rate, 6)
        return out

    def field(self) -> Field:
        if self._field is None:
            meta = {**self.meta, "source": self.source.tolist(), "flow": self.flow_rate(), "wind": self.wind.to_json(self.t)}
            self._field = Field(np.clip(self.c.numpy(), 0.0, 1.0), self.origin, self.dx, self.lel, self.t, meta)
        return self._field

    def sample(self, points) -> np.ndarray:
        return self.field().sample(points)

    def sample_ppm(self, x: float, y: float, z: float) -> float:
        return self.field().sample_ppm(x, y, z)

    def velocity(self) -> np.ndarray:
        u, v, w = (a.numpy() for a in self.vel)
        return np.stack([0.5 * (u[1:] + u[:-1]), 0.5 * (v[:, 1:] + v[:, :-1]), 0.5 * (w[:, :, 1:] + w[:, :, :-1])], -1)

    @classmethod
    def for_flange(cls, fid: str = "F5", hole_mm: float = 3.0, bar: float = 20.0, h2s: float = 0.0,
                   wind: Wind | None = None, dx: float = 0.2, shape=(96, 80, 48), obstacles=True, stage=None,
                   jet_dir=None, meta_path=FLANGES, **kw) -> "Plume":
        flange = next(f for f in json.loads(Path(meta_path).read_text())["flanges"] if f["id"] == fid)
        wind = wind or Wind()
        rel = orifice(hole_mm, bar, h2s=h2s)
        c = np.asarray(flange["center_m"], np.float64)
        jet_dir = np.asarray(leak_direction(flange["normal"], wind) if jet_dir is None else jet_dir, np.float64)
        jet_dir = jet_dir / np.linalg.norm(jet_dir)
        source = c + jet_dir * flange.get("outer_radius_m", 0.17)
        origin = domain_origin(source, shape, dx, wind)
        solid = np.zeros(shape, bool)
        solid[:, :, : int((DECK_Z - origin[2]) / dx)] = True
        if isinstance(obstacles, np.ndarray):
            solid |= obstacles.astype(bool)
        elif obstacles:
            from sim.plume.voxelize import obstacles_for

            vox = obstacles_for(stage if stage is not None else RIG_USD, origin, shape, dx)
            solid |= False if vox is None else vox
        meta = {"flange": fid, "hole_mm": hole_mm, "bar_g": bar, "h2s": h2s}
        return cls(origin, shape, dx, source, rel, wind, jet_dir=jet_dir, solid=solid, meta=meta, **kw)


def leak_direction(normal, wind: Wind, rise: float = 0.35) -> np.ndarray:
    n = np.asarray(normal, np.float64)
    r = np.array([-n[1], n[0], 0.0])
    r /= np.linalg.norm(r) or 1.0
    h = math.radians(wind.heading_deg)
    if r[:2] @ np.array([math.cos(h), math.sin(h)]) < 0:
        r = -r
    d = r + np.array([0.0, 0.0, rise])
    return d / np.linalg.norm(d)


def domain_origin(source, shape, dx: float, wind: Wind) -> np.ndarray:
    size = np.asarray(shape, np.float64) * dx
    h = math.radians(wind.heading_deg)
    d = np.array([math.cos(h), math.sin(h)])
    xy = np.asarray(source[:2]) - size[:2] * (0.5 - 0.25 * d)
    return np.array([xy[0], xy[1], DECK_Z - 2 * dx])
