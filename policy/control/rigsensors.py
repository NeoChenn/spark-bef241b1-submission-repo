from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from policy.control.snapshot import FixedSensor, GasSensor

REPO = Path(__file__).resolve().parents[2]
RIG_META = REPO / "sim/output/full_rig_meta.json"
FLANGE_META = REPO / "sim/output/leak_flanges_meta.json"
ZONES = {"N": (-11.0, 23.5, 43.0), "E": (25.0, -10.5, 43.0)}
ZONE_R = 14.0
NOISE = {"gas": 0.06, "thermal": 0.15, "acoustic": 0.4, "pressure": 0.01}
PROBE_NOISE = {"gas": 0.08, "thermal": 0.3, "acoustic": 0.6, "pressure": 0.02}
LINE_BAR = {"N": 12.0, "E": 16.0}
LEAK_LEN_M = 4.0
LEAK_RAMP_S = 20.0
LEAK_DECAY_S = 25.0


@dataclass
class Spec:
    id: str
    kind: str
    pos: tuple
    zone: str
    flange: Optional[str] = None


@dataclass
class Fault:
    mode: str
    t0: float
    value: Optional[float] = None
    bias: float = 0.0
    sigma: float = 0.0


@dataclass
class Leak:
    flange: str
    pos: np.ndarray
    t0: float
    peak: float
    t_iso: Optional[float] = None

    def level(self, t: float) -> float:
        if t < self.t0:
            return 0.0
        a = self.peak * min(1.0, (t - self.t0) / LEAK_RAMP_S)
        if self.t_iso is not None and t > self.t_iso:
            a *= math.exp(-(t - self.t_iso) / LEAK_DECAY_S)
        return a


def zone_of(pos) -> str:
    p = np.asarray(pos, float)
    best = min(ZONES, key=lambda z: np.linalg.norm(p[:2] - np.asarray(ZONES[z][:2])))
    c = np.asarray(ZONES[best])
    return best if np.linalg.norm(p[:2] - c[:2]) < ZONE_R and abs(p[2] - c[2]) < 3.0 else ""


def default_specs(rig_meta=RIG_META, flange_meta=FLANGE_META) -> list[Spec]:
    specs = []
    for f in json.loads(Path(flange_meta).read_text())["flanges"]:
        c = np.asarray(f["center_m"], float)
        n = np.asarray(f["normal"], float)
        p = tuple(round(float(v), 3) for v in c + 0.6 * n)
        specs.append(Spec(f"G-{f['id']}", "gas", p, zone_of(p), f["id"]))
    specs.append(Spec("G-E1", "gas", (26.0, -8.5, 43.2), "E"))
    for p in json.loads(Path(rig_meta).read_text())["inspection_points"]:
        if p["type"] == "gauge":
            pos = tuple(float(v) for v in p["position_m"])
            specs.append(Spec(p["id"], "pressure", pos, zone_of(pos)))
    specs += [Spec("T-N1", "thermal", (-15.0, 23.0, 43.4), "N"), Spec("T-E1", "thermal", (21.0, -10.0, 43.4), "E"),
              Spec("A-N1", "acoustic", (-6.0, 23.5, 43.4), "N"), Spec("A-E1", "acoustic", (30.0, -10.0, 43.4), "E")]
    return specs


class RigSensors:
    def __init__(self, specs: Optional[list[Spec]] = None, seed: int = 7):
        self.specs = {s.id: s for s in (specs or default_specs())}
        self.rng = np.random.default_rng(seed)
        self.faults: dict[str, Fault] = {}
        self.leaks: dict[str, Leak] = {}
        self.offsets: dict[str, tuple[float, float]] = {}
        self.last: dict[str, float] = {}
        self.flanges = {s.flange: np.asarray(s.pos) for s in self.specs.values() if s.flange}

    def inject(self, sid: str, mode: str, t0: float, **kw) -> None:
        self.faults[sid] = Fault(mode, float(t0), **kw)

    def repair(self, sid: str) -> None:
        self.faults.pop(sid, None)

    def leak(self, flange: str, t0: float, peak: float = 60.0) -> None:
        self.leaks[flange] = Leak(flange, self.flanges[flange], float(t0), peak)

    def isolate(self, flange: str, t: float) -> None:
        lk = self.leaks.get(flange)
        if lk is not None and lk.t_iso is None:
            lk.t_iso = float(t)

    def process_step(self, sid: str, t0: float, delta: float) -> None:
        self.offsets[sid] = (float(t0), float(delta))

    def leaking(self, t: float) -> list[str]:
        return [f for f, lk in self.leaks.items() if lk.level(t) > 1.0]

    def true_value(self, sid: str, t: float) -> float:
        s = self.specs[sid]
        p = np.asarray(s.pos, float)
        ph = 0.0 if s.zone == "N" else 1.7
        common = math.sin(2 * math.pi * t / 40.0 + ph) + 0.5 * math.sin(2 * math.pi * t / 13.0 + 2 * ph)
        near = sum(lk.level(t) * math.exp(-float(np.linalg.norm(p - lk.pos)) / LEAK_LEN_M)
                   for lk in self.leaks.values())
        if s.kind == "gas":
            v = 1.6 + 0.3 * common + near
        elif s.kind == "thermal":
            v = 38.0 + 1.5 * common - 0.08 * near
        elif s.kind == "acoustic":
            v = 42.0 + 1.5 * common + 0.7 * near
        else:
            v = LINE_BAR.get(s.zone, 10.0) + 0.12 * common
        step = self.offsets.get(sid)
        if step is not None and t >= step[0]:
            v += step[1]
        return v

    def report(self, sid: str, t: float) -> Optional[float]:
        s = self.specs[sid]
        v = self.true_value(sid, t) + float(self.rng.normal(0.0, NOISE[s.kind]))
        f = self.faults.get(sid)
        if f is not None and t >= f.t0:
            if f.mode == "dead":
                return None
            if f.mode == "stuck":
                v = self.last.get(sid, v) if f.value is None else f.value
                if f.value is None:
                    f.value = v
            elif f.mode == "drift":
                v += f.bias
            elif f.mode == "noisy":
                v += float(self.rng.normal(0.0, f.sigma))
            elif f.mode == "oor":
                v = f.value
        self.last[sid] = v
        return round(v, 3)

    def probe(self, sid: str, t: float) -> float:
        s = self.specs[sid]
        return round(self.true_value(sid, t) + float(self.rng.normal(0.0, PROBE_NOISE[s.kind])), 3)

    def read(self, t: float) -> tuple[list[GasSensor], list[FixedSensor]]:
        gas, fixed = [], []
        for sid, s in self.specs.items():
            v = self.report(sid, t)
            if v is None:
                continue
            if s.kind == "gas":
                gas.append(GasSensor(sid, zone=s.zone, lel=v, group=s.zone, pos=s.pos))
            else:
                fixed.append(FixedSensor(sid, s.kind, v, s.zone, s.zone, pos=s.pos))
        return gas, fixed
