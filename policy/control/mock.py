from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from policy.control.adapters import from_header, merge
from policy.control.backend import ReplayBackend
from policy.control.centre import ControlCentre
from policy.control.snapshot import GasSensor, Robot, SensorSnapshot

MOCK_DAY = date(2026, 11, 8)
F3_LEL = [(0, 2.0), (13, 2.2), (16, 6), (19, 12), (22, 18), (23.5, 22), (24.5, 56), (28, 58), (31, 40), (36, 14),
          (40, 6), (46, 3), (55, 2)]
F3_DB = [(0, 40.0), (13, 40), (16, 52), (19, 63), (22, 68), (24.5, 84), (28, 83), (31, 70), (36, 52), (40, 45),
         (46, 41), (55, 40)]
A1_LEL = [(0, 1.8), (24, 1.9), (28, 2.7), (34, 2.4), (40, 1.9), (55, 1.8)]
PLUME = [(0, 0.0), (14, 0.05), (19, 0.4), (23, 0.8), (24.5, 3.1), (29, 3.0), (33, 1.2), (38, 0.3), (44, 0.0)]
STORM = [(0, 0.15), (24, 0.25), (30, 0.55), (34, 0.7), (40, 0.62), (48, 0.35), (55, 0.25)]
UPLINK_DOWN = (30.5, 37.0)
JOBS = [{"id": "R-1", "kind": "rust", "zone": "A", "robot": "go2-1", "status": "active"},
        {"id": "L-F3", "kind": "leak_check", "zone": "B", "robot": None, "status": "queued"},
        {"id": "R-2", "kind": "rust_inspect", "zone": "A", "robot": None, "status": "queued"}]


def ramp(t: float, pts) -> float:
    xs, ys = zip(*pts)
    return float(np.interp(t, xs, ys))


class FlakyBackend:
    def __init__(self, inner):
        self.inner, self.down = inner, False
        self.live, self.name = inner.live, inner.name

    @property
    def available(self):
        return True

    def call(self, prompt, snap):
        if self.down:
            raise TimeoutError("uplink lost in storm: no reply in 30 s")
        return self.inner.call(prompt, snap)


class FleetScenario:
    def __init__(self):
        self.r2 = Robot("go2-2", state="PATROL", caption="patrolling the deck B pipe rack", zone="B", battery=0.71,
                        pose=(18.0, 4.0, 1.57), task=None)
        self.r3 = Robot("go2-3", state="DOCKED", caption="charging at the dock", zone="dock", battery=0.64,
                        pose=(2.0, -6.0, 0.0), task=None)
        self.jobs = [dict(j) for j in JOBS]
        self.since = 0.0

    def uplink_down(self, pt: float) -> bool:
        return UPLINK_DOWN[0] <= pt < UPLINK_DOWN[1]

    def at(self, pt: float) -> SensorSnapshot:
        self.r2.battery = round(0.71 - pt * 0.0012, 3)
        self.r3.battery = round(min(1.0, 0.64 + pt * 0.003), 3)
        gas = [GasSensor("G-F3", zone="B", lel=round(ramp(pt, F3_LEL), 1), acoustic_db=round(ramp(pt, F3_DB), 1)),
               GasSensor("G-A1", zone="A", lel=round(ramp(pt, A1_LEL), 1), acoustic_db=39.0)]
        return SensorSnapshot(t=pt, robots=[Robot(**vars(self.r2)), Robot(**vars(self.r3))], gas=gas,
                              plume_kg_h=round(ramp(pt, PLUME), 2), plume_source="F3",
                              storm_level=round(ramp(pt, STORM), 2), wind_deg=250.0,
                              jobs=[dict(j) for j in self.jobs], sources=["fleet_mock"])

    def _job(self, jid, **kw):
        for j in self.jobs:
            if j["id"] == jid:
                j.update(kw)

    def apply(self, actions: list[dict], pt: float) -> None:
        r2, r3 = self.r2, self.r3
        for a in actions:
            if not a.get("auto"):
                continue
            k, rid, tgt = a["kind"], a["robot"], a["target"]
            if k == "retreat" and rid == "go2-2":
                r2.state, r2.caption, r2.zone, r2.task = "GAS_RETREAT", "retreated upwind to deck A", "A", None
                self._job("L-F3", robot=None, status="blocked")
            elif k == "dispatch" and rid == "go2-2" and r2.state == "PATROL":
                r2.state, r2.caption, r2.task = "LEAK_APPROACH", "walking to flange F3 (job L-F3)", tgt
                self._job(tgt, robot="go2-2", status="active")
                self.since = pt
            elif k == "second_view" and rid == "go2-3" and r3.state == "DOCKED":
                r3.state, r3.caption, r3.zone = "SECOND_VIEW", "viewing F3 from deck A, upwind", "A"
            elif k == "hold" and rid == "go2-3" and r3.state == "SECOND_VIEW":
                r3.state, r3.caption, r3.zone = "DOCKED", "back on the dock, charging", "dock"
            elif k == "hold" and rid == "go2-2" and r2.state == "GAS_RETREAT":
                r2.state, r2.caption = "HOLD", "holding upwind until F3 < 5% LEL"
            elif k == "dispatch" and rid == "go2-3" and r3.state == "DOCKED":
                r3.state, r3.caption, r3.zone, r3.task = "LEAK_APPROACH", "heading to F3 to re-check bolts", "B", tgt
                self._job(tgt, robot="go2-3", status="active")
            elif k == "resume" and rid == "go2-2" and r2.state == "HOLD":
                r2.state, r2.caption, r2.zone = "PATROL", "patrolling the deck B pipe rack", "B"
        if r2.state == "LEAK_APPROACH" and pt - self.since > 3.0:
            r2.state, r2.caption = "LEAK_INSPECT", "inspecting F3 bolts for torque stripes"



def attach_control(snaps: list[dict], log_path=None, robot_id: str = "go2-1", tail_s: float = 5.0) -> list[dict]:
    if log_path is not None:
        Path(log_path).unlink(missing_ok=True)
    scenario = FleetScenario()
    flaky = FlakyBackend(ReplayBackend())
    centre = ControlCentre(flaky, log_path=log_path, interval_s=5.0, alarm_gap_s=1.5, per_minute=30,
                           sim_latency=True, today=lambda: MOCK_DAY)
    state = {"prev": None, "person": None}

    def tick(s, pt):
        flaky.down = scenario.uplink_down(pt)
        hud = from_header(s, robot_id, zone="A")
        r1 = hud.robot(robot_id)
        state["person"] = r1.person_m or (state["person"] if r1.state == "PAUSE_PERSON" else None)
        r1.person_m = state["person"]
        snap = merge(hud, scenario.at(pt))
        ctl = centre.tick(snap, now=pt)
        scenario.apply(ctl["actions"], pt)
        ctl["fleet"] = [{"id": r.id, "state": r.state, "caption": r.caption, "battery": r.battery}
                        for r in snap.robots]
        changed = ctl != state["prev"]
        state["prev"] = ctl
        return ctl if changed else None

    for s in snaps:
        ctl = tick(s, s["pt"])
        if ctl:
            s["control"] = ctl
    last = snaps[-1]
    end = last["pt"] + tail_s
    pt = last["pt"]
    while centre.pending is not None and pt < end:
        pt = round(pt + 0.5, 3)
        ctl = tick({**last, "events": []}, pt)
        snaps.append({"t": last["t"], "pt": pt, "state": last["state"], "events": [], "control": ctl or state["prev"]})
    return snaps
