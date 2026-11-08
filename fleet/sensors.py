"""Fixed fire-and-gas detectors on the top deck, read from the modelled gas plume.

The rig's gas alarm comes from these detectors, not from a timer. Each second `DetectorNetwork.poll` reads every
gas detector from `fleet.gas.plume` (the same 2-D analytic plume the robots' gas sensors sample) and emits
`detector_alarm` when one crosses its threshold, `detector_clear` when it drops back below half of it. The first
alarm per leak is handed to `on_alarm` (the coordinator then queues the response).

Reading model (modelled, not measured):
    point gas  excess(xy) * (1 - exp(-age / TAU_S)),  age = t - leak_t0 - distance / TRANSPORT_MPS
    open path  the same, averaged over the beam and multiplied by its length (LEL.m)
The plume is the steady analytic field; the onset lag stands in for transport and the detector's T90. No noise.

Positions: point detectors GD-01..GD-08 stand on 1.5 m posts on the walkway side of the two process lines of the
top deck (valves, gauges and pipework in sim/output/full_rig_meta.json `inspection_points`): the north line at
y = 23.4..24.3 (walkway y = 20.5) and the south line at y = -10.7 (walkway y = -9.0). The three F&G optical
detectors are the ones `fg_optics_clean` cleans (fleet/jobs/rounds.py DEFAULT_FG_DETECTORS, same ids and
positions); the two IR flame detectors do not read gas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from fleet.gas.plume import GAS_BASE, amplitude, shape

DECK_Z = 42.5
HEAD_Z = DECK_Z + 1.5                 # point detector head on a 1.5 m post
TRANSPORT_MPS = 0.5                   # gas front speed from the source to a detector (no wind)
TAU_S = 4.0                           # detector response / mixing lag
POINT_ALARM_LEL = 0.10                # 10 % LEL low alarm (readings are fractions of LEL)
OPEN_PATH_ALARM_LELM = 1.0            # 1 LEL.m low alarm for the open-path detector
CLEAR_FRACTION = 0.5


@dataclass(frozen=True)
class Detector:
    id: str
    position: tuple[float, float, float]           # detector head
    type: str                                      # point_gas | open_path_gas | ir_flame
    threshold: Optional[float] = None              # LEL (point) or LEL.m (open path); None: does not read gas
    beam: Optional[tuple[tuple[float, float], tuple[float, float]]] = None   # open path: (xy start, xy end)
    unit: str = "LEL"

    @property
    def base_xyz(self) -> tuple[float, float, float]:
        return (self.position[0], self.position[1], DECK_Z)


def _gd(n: int, x: float, y: float) -> Detector:
    return Detector(f"GD-{n:02d}", (x, y, HEAD_Z), "point_gas", POINT_ALARM_LEL)


DETECTORS: list[Detector] = [
    _gd(1, -21.0, 22.6),          # north line, west end (gauge_01, valve_01)
    _gd(2, -9.0, 22.6),           # north line (gauge_04, valve_02)
    _gd(3, -2.5, 22.6),           # north line, east end (valve_04)
    _gd(4, 11.0, -9.9),           # south line, west
    _gd(5, 15.5, -9.9),           # south line (gauge_05)
    _gd(6, 19.2, -9.9),           # south line by valve_05 and its flanges
    _gd(7, 26.0, -9.9),           # south line (gauge_06)
    _gd(8, 33.5, -9.9),           # south line, east end (valve_06, gauge_07)
    # F&G optical detectors, same ids and positions as fleet/jobs/rounds.py DEFAULT_FG_DETECTORS
    Detector("fg_ir_01", (-14.0, 22.6, 44.6), "ir_flame"),
    Detector("fg_ogd_02", (21.0, -9.9, 44.7), "open_path_gas", OPEN_PATH_ALARM_LELM,
             beam=((18.0, -9.9), (24.0, -9.9)), unit="LEL.m"),
    Detector("fg_ir_03", (30.0, -9.9, 44.7), "ir_flame"),
]
BY_ID = {d.id: d for d in DETECTORS}
GAS_DETECTORS = [d for d in DETECTORS if d.threshold is not None]


def _excess(state: dict, xy, t: float) -> float:
    xy = np.asarray(xy, float)[:2]
    total = 0.0
    for lk in state.get("gas_leaks") or []:
        a = amplitude(lk)
        if a <= 0.0 or lk.get("t0") is None:
            continue
        src = np.asarray(lk["position"], float)[:2]
        age = t - float(lk["t0"]) - float(np.linalg.norm(xy - src)) / TRANSPORT_MPS
        if age <= 0.0:
            continue
        steady = a * float(shape(xy[None], src, state.get("wind"), float(lk.get("decay_m", 1.5)))[0])
        total += steady * (1.0 - math.exp(-age / TAU_S))
    return total


def reading(det: Detector, world_or_state, t: Optional[float] = None) -> Optional[float]:
    """Modelled reading: LEL fraction for a point detector, LEL.m for the open-path one, None for flame."""
    state = getattr(world_or_state, "state", world_or_state)
    if t is None:
        t = float(getattr(world_or_state, "t", 0.0))
    if det.type == "point_gas":
        return GAS_BASE + _excess(state, det.position, t)
    if det.type == "open_path_gas" and det.beam:
        (x0, y0), (x1, y1) = det.beam
        n = 25
        pts = [(x0 + (x1 - x0) * k / (n - 1), y0 + (y1 - y0) * k / (n - 1)) for k in range(n)]
        length = math.dist((x0, y0), (x1, y1))
        return float(np.mean([_excess(state, p, t) for p in pts])) * length
    return None


def nearest(xy, detectors=None) -> Detector:
    return min(detectors or GAS_DETECTORS, key=lambda d: math.dist(d.position[:2], list(xy)[:2]))


@dataclass
class DetectorNetwork:
    """Polls the gas detectors once per sim second and raises / clears latched alarms."""
    detectors: list[Detector] = field(default_factory=lambda: list(GAS_DETECTORS))
    alarmed: dict[str, float] = field(default_factory=dict)           # id -> alarm time (latched)
    first: dict[str, dict] = field(default_factory=dict)              # leak id -> first alarm

    def poll(self, world, emit: Callable[..., dict], on_alarm: Optional[Callable[[Detector, float, dict], None]] = None
             ) -> list[dict]:
        t = float(world.t)
        raised = []
        for d in self.detectors:
            r = reading(d, world, t)
            if r is None:
                continue
            if d.id not in self.alarmed and r >= d.threshold:
                self.alarmed[d.id] = t
                leak = self._leak_for(world, d)
                emit("detector_alarm", None, None, detector=d.id, type=d.type, position=list(d.position),
                     reading=round(r, 3), threshold=d.threshold, unit=d.unit,
                     leak=leak.get("id") if leak else None,
                     summary=f"{d.id} alarm {r:.2f} {d.unit}")
                rec = {"detector": d.id, "t": t, "reading": round(r, 3), "position": list(d.position)}
                raised.append(rec)
                if leak is not None and leak.get("id") not in self.first:
                    self.first[leak.get("id")] = rec
                    leak["reported"] = True                  # FleetSim's own fallback alarm stays quiet
                    leak["detector"] = d.id
                    if on_alarm:
                        on_alarm(d, r, leak)
            elif d.id in self.alarmed and r < CLEAR_FRACTION * d.threshold:
                self.alarmed.pop(d.id)
                emit("detector_clear", None, None, detector=d.id, type=d.type, position=list(d.position),
                     reading=round(r, 3), unit=d.unit, summary=f"{d.id} clear {r:.2f} {d.unit}")
        return raised

    @staticmethod
    def _leak_for(world, d: Detector) -> Optional[dict]:
        live = [lk for lk in world.state.get("gas_leaks") or [] if amplitude(lk) > 0 and lk.get("t0") is not None]
        if not live:
            return None
        return min(live, key=lambda lk: math.dist(lk["position"][:2], d.position[:2]))

    def attach(self, sim, on_alarm: Optional[Callable[[Detector, float, dict], None]] = None, every_s: float = 1.0):
        """Poll from the FleetSim script queue (before the sim's own sensing each step)."""
        def tick(s):
            self.poll(s.world, s.c.emit, on_alarm)
            s.at(s.world.t + every_s, "detectors", tick)
        sim.at(0.0, "detectors", tick)
        return self


def quiet_fallback(leak: dict) -> dict:
    """A leak the detectors must find: FleetSim's timer alarm (`alarm_after_s`) never fires."""
    leak["alarm_after_s"] = 1e9
    return leak
