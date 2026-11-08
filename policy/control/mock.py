from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from policy.control.adapters import from_header, merge
from policy.control.agent import OrchestrationAgent, ReplayAgentBackend
from policy.control.backend import ReplayBackend
from policy.control.centre import ControlCentre
from policy.control.port import MemoryCoordinator
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
JOBS = [{"id": "R-1", "kind": "rust_repair", "target": "P-104", "zone": "A", "robot": "go2-1", "level": "SUPV",
         "status": "RUNNING"},
        {"id": "L-F3", "kind": "leak_check", "target": "F3", "zone": "B", "robot": None, "status": "QUEUED"},
        {"id": "R-2", "kind": "rust_inspect", "target": "P-118", "zone": "A", "robot": None, "status": "QUEUED"}]
ZONE_OF = {"deck-2": "B", "F3": "B", "P-104": "A", "P-118": "A"}
INTENT = "Inspect deck 2, isolate any leak, don't paint without me."

# Fixed fire-and-gas detectors bolted around the top deck, in the fleet_map.v1 frame (m, Z up). Same shape as
# fleet/sensors.py so the two can be joined: id, type, position, reading (LEL fraction), threshold, state.
DETECTORS = (("GD-01", (-20.6, 24.4, 43.6)), ("GD-02", (-12.0, 24.2, 43.6)), ("GD-03", (-5.9, 24.6, 43.6)),
             ("GD-04", (-1.5, 24.0, 43.4)), ("GD-05", (-13.6, 8.4, 42.9)), ("GD-06", (1.0, 3.0, 42.9)),
             ("GD-07", (-15.9, 24.0, 43.4)), ("GD-08", (20.5, -11.3, 43.4)), ("GD-09", (31.0, -11.4, 43.4)))
DETECTOR_ALARM = 0.20
# How much of the F3 plume each detector sees (GD-07 sits on F3), and the A1 background for the deck A rack.
F3_SEEN = {"GD-07": 1.0, "GD-01": 0.25, "GD-02": 0.3, "GD-03": 0.08}
DOCK = (-12.0, 9.4)
F3_WORK = (-15.94, 22.32)
ROUTE_F3 = [(-10.0, 11.0), (-10.0, 20.5), (-15.94, 20.5), F3_WORK]
WALK_MS = 1.6


def detector_state(reading: float, threshold: float = DETECTOR_ALARM) -> str:
    return "alarm" if reading >= threshold else "warn" if reading >= threshold / 2 else "ok"


def detectors(readings: dict, t: float = 0.0, dispatched: dict | None = None) -> list[dict]:
    """The fixed-detector list for state["detectors"]; readings are LEL fractions keyed by id, others idle."""
    out = []
    for k, (did, pos) in enumerate(DETECTORS):
        r = round(float(readings.get(did, 0.012 + 0.004 * np.sin(0.37 * t + 1.7 * k))), 3)
        d = {"id": did, "type": "point_gas", "position": list(pos), "reading": r, "unit": "LEL",
             "threshold": DETECTOR_ALARM, "state": detector_state(r)}
        if dispatched and dispatched.get(did):
            d["dispatched"] = dispatched[did]
        out.append(d)
    return out


class Walker:
    """Moves a mock dog along waypoints at walking speed so the map shows it go where it was sent."""

    def __init__(self, rid, xy, deck="top_deck"):
        self.rid, self.x, self.y, self.yaw, self.deck, self.path, self.goal = rid, *xy, 0.0, deck, [], None

    def go(self, key, pts):
        if self.goal != key:
            self.goal, self.path = key, list(pts)

    def step(self, dt):
        left = WALK_MS * max(0.0, dt)
        while self.path and left > 0:
            tx, ty = self.path[0]
            d = float(np.hypot(tx - self.x, ty - self.y))
            if d > 1e-6:
                self.yaw = float(np.arctan2(ty - self.y, tx - self.x))
            if d <= left:
                self.x, self.y, left = tx, ty, left - d
                self.path.pop(0)
            else:
                self.x, self.y, left = self.x + (tx - self.x) * left / d, self.y + (ty - self.y) * left / d, 0.0

    def pose(self):
        return {"x": round(self.x, 2), "y": round(self.y, 2), "z": 42.5, "yaw": round(self.yaw, 3)}


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
        self.clock = 0.0
        self.port = MemoryCoordinator([{"id": r, "zone": z} for r, z in (("go2-1", "A"), ("go2-2", "B"), ("go2-3", "dock"))],
                                      JOBS, clock=lambda: self.clock)
        self.since = 0.0

    def uplink_down(self, pt: float) -> bool:
        return UPLINK_DOWN[0] <= pt < UPLINK_DOWN[1]

    def jobs(self) -> list[dict]:
        return [{"id": j["id"], "kind": j["kind"], "zone": j.get("zone") or ZONE_OF.get(j["target"], ""),
                 "robot": j.get("robot"), "status": j["status"], "priority": j["priority"]}
                for j in self.port.jobs.values()]

    def at(self, pt: float) -> SensorSnapshot:
        self.clock = pt
        self.r2.battery = round(0.71 - pt * 0.0012, 3)
        self.r3.battery = round(min(1.0, 0.64 + pt * 0.003), 3)
        gas = [GasSensor("G-F3", zone="B", lel=round(ramp(pt, F3_LEL), 1), acoustic_db=round(ramp(pt, F3_DB), 1)),
               GasSensor("G-A1", zone="A", lel=round(ramp(pt, A1_LEL), 1), acoustic_db=39.0)]
        return SensorSnapshot(t=pt, robots=[Robot(**vars(self.r2)), Robot(**vars(self.r3))], gas=gas,
                              plume_kg_h=round(ramp(pt, PLUME), 2), plume_source="F3",
                              storm_level=round(ramp(pt, STORM), 2), wind_deg=250.0,
                              jobs=self.jobs(), sources=["fleet_mock"])

    def _job(self, jid, **kw):
        if jid in self.port.jobs:
            self.port.jobs[jid].update(kw)

    def apply(self, actions: list[dict], pt: float) -> None:
        r2, r3 = self.r2, self.r3
        for a in actions:
            if not a.get("auto"):
                continue
            k, rid, tgt = a["kind"], a["robot"], a["target"]
            if k == "retreat" and rid == "go2-2":
                r2.state, r2.caption, r2.zone, r2.task = "GAS_RETREAT", "retreated upwind to deck A", "A", None
                self._job("L-F3", robot=None, status="BLOCKED")
            elif k == "dispatch" and rid == "go2-2" and r2.state == "PATROL":
                r2.state, r2.caption, r2.task = "LEAK_APPROACH", "walking to flange F3 (job L-F3)", tgt
                self._job(tgt, robot="go2-2", status="RUNNING")
                self.since = pt
            elif k == "second_view" and rid == "go2-3" and r3.state == "DOCKED":
                r3.state, r3.caption, r3.zone = "SECOND_VIEW", "viewing F3 from deck A, upwind", "A"
            elif k == "hold" and rid == "go2-3" and r3.state == "SECOND_VIEW":
                r3.state, r3.caption, r3.zone = "DOCKED", "back on the dock, charging", "dock"
            elif k == "hold" and rid == "go2-2" and r2.state == "GAS_RETREAT":
                r2.state, r2.caption = "HOLD", "holding upwind until F3 < 5% LEL"
            elif k == "dispatch" and rid == "go2-3" and r3.state == "DOCKED":
                r3.state, r3.caption, r3.zone, r3.task = "LEAK_APPROACH", "heading to F3 to re-check bolts", "B", tgt
                self._job(tgt, robot="go2-3", status="RUNNING")
            elif k == "resume" and rid == "go2-2" and r2.state == "HOLD":
                r2.state, r2.caption, r2.zone = "PATROL", "patrolling the deck B pipe rack", "B"
        if r2.state == "LEAK_APPROACH" and pt - self.since > 3.0:
            r2.state, r2.caption = "LEAK_INSPECT", "inspecting F3 bolts for torque stripes"


def attach_control(snaps: list[dict], log_path=None, robot_id: str = "go2-1", tail_s: float = 5.0) -> list[dict]:
    if log_path is not None:
        Path(log_path).unlink(missing_ok=True)
    scenario = FleetScenario()
    flaky = FlakyBackend(ReplayBackend())
    agent = OrchestrationAgent(scenario.port, ReplayAgentBackend(), today=lambda: MOCK_DAY,
                               log_path=Path(log_path).with_name("agent_calls.jsonl") if log_path else None)
    if agent.log_path is not None:
        agent.log_path.unlink(missing_ok=True)
    centre = ControlCentre(flaky, log_path=log_path, interval_s=5.0, alarm_gap_s=1.5, per_minute=30,
                           sim_latency=True, today=lambda: MOCK_DAY, agent=agent)
    state = {"prev": None, "person": None, "done": set()}

    def once(key, fn):
        if key not in state["done"]:
            state["done"].add(key)
            fn()

    def story(pt, hud_state):
        prev = state["prev"] or {}
        once("intent", lambda: centre.intent(INTENT, now=pt))
        if any(a["kind"] == "probable_leak" for a in prev.get("anomalies", [])):
            once("leak", lambda: centre.finding({"kind": "gas_leak", "severity": "critical", "target": "F3", "zone": "B",
                                                 "summary": "Gas and ultrasonic rising together at F3"}, now=pt))
        if any(a["kind"] == "evacuate" and a["source"] == "edge" for a in prev.get("actions", [])):
            iso = next((j["id"] for j in scenario.port.jobs.values() if j["kind"] == "valve_isolate"), None)
            if iso:
                once("approve", lambda: scenario.port.approve_request(iso, "operator", "Crew isolating the F3 line"))
        if hud_state == "DONE":
            once("rust", lambda: centre.finding({"kind": "rust_found", "severity": "warning", "target": "P-118",
                                                 "zone": "A", "summary": "Deck 2 round found fresh rust on P-118"},
                                                now=pt))

    def tick(s, pt):
        flaky.down = scenario.uplink_down(pt)
        story(pt, s.get("state"))
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

    rig = MockRig(scenario)
    for s in snaps:
        ctl = tick(s, s["pt"])
        if ctl:
            s["control"] = ctl
        s.update(rig.sample(s.get("state"), s["pt"]))
    last = snaps[-1]
    end = last["pt"] + tail_s
    pt = last["pt"]
    while (centre.pending is not None or "rust" not in state["done"]) and pt < end:
        pt = round(pt + 0.5, 3)
        ctl = tick({**last, "events": []}, pt)
        keep = {k: last[k] for k in ("caption", "robots", "camera") if k in last}
        snaps.append({"t": last["t"], "pt": pt, "state": last["state"], "events": [], **keep,
                      "control": ctl or state["prev"], **rig.sample(last["state"], pt)})
    return snaps


class MockRig:
    """Fixed detectors and a top-deck fleet snapshot for the scripted F3 leak: GD-07 on F3 alarms as the
    plume grows, and the map shows the dog the control centre sent (job L-F3) walking there."""

    def __init__(self, scenario: FleetScenario):
        self.sc, self.pt, self.sent = scenario, None, None
        self.w = {"unit-1": Walker("unit-1", (-1.0, 5.0)), "go2-2": Walker("go2-2", (-19.0, 20.5)),
                  "go2-3": Walker("go2-3", DOCK)}
        self.w["unit-1"].yaw = 1.571

    @staticmethod
    def to_f3(w):
        return ROUTE_F3[2:] if abs(w.y - 20.5) < 0.6 else ROUTE_F3[1:] if abs(w.x + 10.0) < 0.6 else ROUTE_F3

    def steer(self):
        r2, r3, w2, w3 = self.sc.r2, self.sc.r3, self.w["go2-2"], self.w["go2-3"]
        if r2.state == "PATROL" and not w2.path:
            w2.go(("patrol", w2.x < -15), [(-12.0, 20.5)] if w2.x < -15 else [(-20.0, 20.5)])
        elif r2.state in ("LEAK_APPROACH", "LEAK_INSPECT"):
            w2.go("f3", self.to_f3(w2))
        elif r2.state == "GAS_RETREAT":
            w2.go("retreat", [(-15.94, 20.5), (-10.0, 20.5), (-10.0, 11.0), (-3.0, 11.0)])
        if r3.state == "SECOND_VIEW":
            w3.go("view", [(-12.0, 11.0), (-10.0, 11.0), (-10.0, 15.5)])
        elif r3.state == "LEAK_APPROACH":
            w3.go("f3", self.to_f3(w3))
        elif r3.state == "DOCKED":
            w3.go("dock", [(-10.0, 11.0), (-12.0, 11.0), DOCK] if abs(w3.x - DOCK[0]) > 0.5 or abs(w3.y - DOCK[1]) > 0.5 else [])

    def sample(self, hud_state, pt: float) -> dict:
        dt = 0.0 if self.pt is None else pt - self.pt
        self.pt = pt
        self.steer()
        for w in self.w.values():
            w.step(dt)
        job = self.sc.port.jobs.get("L-F3") or {}
        self.sent = job.get("robot") or self.sent
        f3 = ramp(pt, F3_LEL) / 100.0
        reads = {k: f3 * v for k, v in F3_SEEN.items()}
        reads["GD-06"] = ramp(pt, A1_LEL) / 100.0
        dets = detectors(reads, pt, {"GD-07": self.sent})
        r2, r3 = self.sc.r2, self.sc.r3
        info = {"unit-1": ("Unit 1 · rust", hud_state or "PATROL", 0.8, "rust:P-104"),
                "go2-2": ("Dog-2", r2.state, r2.battery, r2.task and "gas:" + r2.task),
                "go2-3": ("Dog-3", r3.state, r3.battery, r3.task and "gas:" + r3.task)}
        robots = [{"id": rid, "name": n, "state": st, "job_state": st, "pose": self.w[rid].pose(), "deck": "top_deck",
                   "battery": {"soc": soc}, "task": task} for rid, (n, st, soc, task) in info.items()]
        f3xyz = [-15.942, 23.167, 43.35]
        hot = next(d for d in dets if d["id"] == "GD-07")["state"] != "ok"
        fleet = {"schema": "fleet_state.v1", "t": round(pt, 2), "robots": robots,
                 "zones": [{"kind": "gas", "id": "gas:F3", "xyz": f3xyz, "radius": round(3 + 6 * min(1.0, f3 / 0.4), 2)}] if hot else [],
                 "tasks": [{"id": "rust:P-104", "kind": "rust", "xyz": [-1.0, 5.0, 43.6], "status": "active", "robot": "unit-1"},
                           {"id": "gas:L-F3", "kind": "gas", "xyz": f3xyz, "status": "active" if hot else "done",
                            "robot": self.sent}]}
        return {"detectors": dets, "fleet": fleet}
