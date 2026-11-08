from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from integration.dual_argon import CLEAR_LEL, ALARM_LEL, FixedDetector, led
from integration.gas_detectors import load_layout
from policy.leak.geometry import FlangeGeometry
from policy.leak.world import GAS_BASE, LEAK_COEFF, LEAK_FULL_RAD, PLUME_LENGTH_M

REPO = Path(__file__).resolve().parent.parent
LEAK_META = REPO / "sim/output/leak_flanges_meta.json"
DECK_Z = 42.5
SNIFF_Z = DECK_Z + 0.6
WORK_STANDOFF_M = 0.85
FLANGES = ("F1", "F2", "F3", "F4", "F5", "F6")


@dataclass
class Event:
    name: str
    t: float
    end: Optional[float] = None
    detail: dict = field(default_factory=dict)

    def active(self, t: float) -> bool:
        return self.t <= t < (math.inf if self.end is None else self.end)

    def to_dict(self) -> dict:
        return {"name": self.name, "t": round(self.t, 2), "end": None if self.end is None else round(self.end, 2),
                **self.detail}


def work_point(geom: FlangeGeometry, standoff: float = WORK_STANDOFF_M) -> np.ndarray:
    p = geom.center + standoff * geom.normal
    return np.array([p[0], p[1], DECK_Z])


class Scenario:
    def __init__(self, t_leak: float = 6.0, t_fixed: float = 34.0, t_stuck: Optional[float] = 46.0,
                 leak_flange: str = "F2", bolt: int = 3, loosen_rad: float = 0.27, stuck_flange: str = "F1",
                 stuck_lel: float = 14.0, rise_tau_s: float = 3.0, decay_tau_s: float = 4.0,
                 wind=(0.0, -1.0, 0.0), wind_stretch: float = 0.3, flanges=FLANGES, meta_path=LEAK_META,
                 duration_s: float = 60.0):
        meta = {f["id"]: f for f in json.loads(Path(meta_path).read_text())["flanges"]}
        self.flanges = {fid: FlangeGeometry.from_meta(meta[fid]) for fid in flanges}
        self.prims = {fid: meta[fid].get("prim", f"/World/Leak/{fid}") for fid in flanges}
        layout = {d.flange: d for d in load_layout(meta_path)}
        self.detectors = [FixedDetector(f"G-{fid}", fid, tuple(float(v) for v in layout[fid].sample.round(4))) for fid in flanges]
        self.detector_prims = {f"G-{fid}": layout[fid].path for fid in flanges}
        self.t_leak, self.t_fixed, self.t_stuck = t_leak, t_fixed, t_stuck
        self.leak_flange, self.bolt, self.loosen_rad = leak_flange, bolt, loosen_rad
        self.stuck_id, self.stuck_lel = f"G-{stuck_flange}", stuck_lel
        self.rise_tau_s, self.decay_tau_s = rise_tau_s, decay_tau_s
        w = np.asarray(wind, float)
        self.wind = w / np.linalg.norm(w) if np.linalg.norm(w) > 0 else w
        self.wind_stretch = wind_stretch
        self.duration_s = duration_s
        self.peak = float(np.clip(loosen_rad / LEAK_FULL_RAD, 0.0, 1.0))
        self.script = self._script()

    def level(self, t: float) -> float:
        if t < self.t_leak:
            return 0.0
        on = self.peak * (1.0 - math.exp(-(min(t, self.t_fixed) - self.t_leak) / self.rise_tau_s))
        return on if t < self.t_fixed else on * math.exp(-(t - self.t_fixed) / self.decay_tau_s)

    def bolt_angle(self, t: float) -> float:
        return self.loosen_rad if self.t_leak <= t < self.t_fixed else 0.0

    def gas_frac(self, point, t: float) -> float:
        r = np.asarray(point, float) - self.flanges[self.leak_flange].center
        d = float(np.linalg.norm(r))
        d_eff = d - self.wind_stretch * float(r @ self.wind)
        return GAS_BASE + LEAK_COEFF * self.level(t) * math.exp(-d_eff / PLUME_LENGTH_M)

    def gas_lel(self, point, t: float) -> float:
        return round(100.0 * self.gas_frac(point, t), 2)

    def sniff(self, xy, t: float, z: float = SNIFF_Z) -> float:
        return self.gas_lel((xy[0], xy[1], z if len(xy) < 3 else xy[2]), t)

    def gas_at(self, t: float):
        return lambda p: self.gas_frac(p, t)

    def detector_readings(self, t: float) -> dict[str, float]:
        g = self.gas_at(t)
        for d in self.detectors:
            d.stuck_at = self.stuck_lel if d.id == self.stuck_id and self.stuck(t) else None
            d.read(g)
        return {d.id: d.last for d in self.detectors}

    def stuck(self, t: float) -> bool:
        return self.t_stuck is not None and t >= self.t_stuck

    def leds(self, t: float) -> dict[str, str]:
        return {k: led(v) for k, v in self.detector_readings(t).items()}

    def detector(self, det_id: str) -> FixedDetector:
        return next(d for d in self.detectors if d.id == det_id)

    def _first(self, start: float, stop: float, cond, dt: float = 0.05) -> Optional[float]:
        t = start
        while t <= stop:
            if cond(t):
                return round(t, 2)
            t += dt
        return None

    def _script(self) -> list[Event]:
        det = f"G-{self.leak_flange}"
        p = np.asarray(self.detector(det).pos, float)
        lel = lambda t: 100.0 * self.gas_frac(p, t)
        amber = self._first(self.t_leak, self.t_fixed, lambda t: lel(t) >= CLEAR_LEL)
        red = self._first(self.t_leak, self.t_fixed, lambda t: lel(t) >= ALARM_LEL)
        clear = self._first(self.t_fixed, self.t_fixed + 120.0, lambda t: lel(t) < CLEAR_LEL)
        bolt = {"flange": self.leak_flange, "bolt": self.bolt, "loosen_rad": self.loosen_rad}
        ev = [Event("leak_start", self.t_leak, self.t_leak + 2.0, bolt),
              Event("leak", self.t_leak, self.t_fixed, bolt)]
        if amber is not None:
            ev.append(Event(f"{det} amber", amber, red, {"detector": det}))
        if red is not None:
            ev.append(Event(f"{det} red", red, self.t_fixed, {"detector": det}))
        ev += [Event("retorque_done", self.t_fixed, self.t_fixed + 2.0, bolt),
               Event("source_off", self.t_fixed, self.t_fixed + 2.0, {"flange": self.leak_flange}),
               Event("gas_decay", self.t_fixed, clear, {"flange": self.leak_flange, "tau_s": self.decay_tau_s})]
        if clear is not None:
            ev.append(Event("gas_clear", clear, clear + 2.0, {"detector": det}))
        if self.t_stuck is not None:
            ev.append(Event(f"{self.stuck_id} stuck", self.t_stuck, None,
                            {"detector": self.stuck_id, "stuck_lel": self.stuck_lel}))
        return sorted(ev, key=lambda e: e.t)

    def event(self, name: str) -> Event:
        return next(e for e in self.script if e.name == name)

    def at(self, t: float) -> list[Event]:
        return [e for e in self.script if e.active(t)]

    def coordinates(self) -> dict:
        return {"frame": "rig_merged world (m, Z up, top deck z=42.5)",
                "flanges": {fid: {"center": g.center.round(4).tolist(), "normal": g.normal.round(4).tolist(),
                                  "work_xyz": work_point(g).round(3).tolist(), "prim": self.prims[fid]}
                            for fid, g in self.flanges.items()},
                "detectors": {d.id: {"flange": d.flange, "sample_xyz": list(d.pos),
                                     "prim": self.detector_prims[d.id]} for d in self.detectors},
                "source": {"flange": self.leak_flange, "bolt": self.bolt,
                           "xyz": self.flanges[self.leak_flange].center.round(4).tolist()},
                "wind": self.wind.round(3).tolist()}

    def to_dict(self) -> dict:
        return {"duration_s": self.duration_s, "script": [e.to_dict() for e in self.script],
                **self.coordinates()}
