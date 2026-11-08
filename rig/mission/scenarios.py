from __future__ import annotations

import math
import time

import numpy as np

from policy.fleet.fleet import walk_along
from policy.fleet.jobs import SETUP_S
from policy.fleet.reservations import Reservations
from policy.fleet.scenario import JUDGES_ROBOTS, JUDGES_RUST, JUDGES_WELD, CHECK_DEADLINE_S, nearest_node
from rig.mission.gas import ALARM_LEL, FLOW_GS, INFOTAXIS, METHODS, GasLeak

IDS = ("gas", "patrol", "repair", "flyby", "nav")
NAMES = {"gas": "Gas leak", "patrol": "Deck patrol", "repair": "Pipe repair", "flyby": "Fly-by"}
WEATHER = {"level": "rain", "time": "dusk", "rain": 0.8, "fog": 0.6, "wind_dir": 240.0}
DEMO_WEATHER = {"level": "rain", "time": "dusk", "rain": 0.35, "fog": 0.25, "wind_dir": 240.0}
ALERTS_KEEP = 10
OPEN = ("pending", "assigned", "active")
BEAT_S = 6.0
ARGON_WAIT_S = 4.0
GUST_PEAK_N = 14.0
GAS_FLOW_GS = 30.0
GUST_S = 6.0
TOAST = {"drop_obstacle": "{kind} on the walkway", "drop_can": "Can on the walkway", "close_walkway": "Walkway closed",
         "close_stairs": "Stairs closed", "clear": "All clear", "set_weather": "Weather · {level}"}


def unit(rid) -> str:
    d = "".join(c for c in str(rid) if c.isdigit())
    return f"Unit {d}" if d else str(rid)


CAM_DEFAULTS = {"deck": [18.75, -5.75, 42.5], "rust": [18.75, -5.75, 43.1], "flange": [17.0, -9.0, 42.5], "weld": [10.0, -8.0, 42.5]}


class Script:
    id = name = main = ""
    steps: tuple = ()
    disturbances: tuple = ()
    robots: tuple = ()
    specs: tuple = ()
    weather = WEATHER

    def __init__(self, run: "Scenarios"):
        self.run = run
        self.m = run.m
        self.fired: list[str] = []
        self.i = 0
        self.focus = self.main

    @property
    def f(self):
        return self.m.fleet.f

    def setup(self) -> None:
        pass

    def tick(self) -> None:
        pass

    def done(self) -> bool:
        return self.i == len(self.steps) - 1

    def key(self) -> dict | None:
        return None

    def fire(self, did: str) -> tuple[str, str]:
        return did, ""

    def task(self, kind, poi, xyz=None, only=None, **kw):
        with self.m.fleet.lock:
            t = self.f.add_task(kind, poi, xyz, only=only, **kw)
            self.m.fleet.snap = self.f.snapshot()
        return t

    def open_tasks(self, kind=None, robot=None):
        return [t for t in self.f.tasks.values() if t.status in OPEN and (kind is None or t.kind == kind)
                and (robot is None or t.only == robot or t.robot == robot)]

    def robot(self, rid):
        return self.m._robot(rid)

    def state_of(self, rid) -> str:
        r = self.robot(rid)
        if r is None:
            return ""
        st, _ = self.f.job_state(r)
        return st or r.state


class Gas(Script):
    id, name, main = "gas", "Gas leak", "R2"
    flange, valve_xyz = "F5", [17.0, -9.0, 42.5]
    seam = ("weld_seam_f5", [10.0, -8.0, 42.5])
    steps = ("Fixed gas sensor trips", "Argon reads the sensor data", "Gas robot dispatched",
             "Gas robot localising the leak", "Valve isolation waits for your approval", "Unit 1 closing the isolation valve",
             "Gas clearing — hot work still frozen", "Leak isolated, weld resumed")
    disturbances = (("leak_reopens", "Leak reopens"), ("second_leak", "Second flange leaks"),
                    ("cut_link", "Gemini link cut"))
    robots = (("R1", [10.0, -9.0, 42.5], 0.9), ("R2", [-1.0, -4.5, 42.5], 0.9), ("R3", [10.0, 12.0, 20.0], 0.8))
    second = "F6"

    def setup(self) -> None:
        m = self.m
        m.valve_by, m.wait_isolation, m.flow_gs = "R1", True, GAS_FLOW_GS
        m.leak.method = "infotaxis" if INFOTAXIS else "cast"
        m.valves = {self.flange: self.valve_xyz, self.second: [27.0, -9.0, 42.5]}
        tr = m.fleet.jobs.get("weld", "crack")
        start = next((s[0] for s in tr.segments if s[1] == "WELD"), 0.0)
        self.weld = self.task("weld", self.seam[0], self.seam[1], scenario="crack", only="R1")
        self.weld.progress = SETUP_S["weld"] + start
        self.task("rust", JUDGES_RUST[2][0], JUDGES_RUST[2][1], severity=0.5, exposure=0.9, only="R3")
        self.leak_at = self.trip_t = self.ctl_seen = self.ctl0 = None

    def gas_max(self) -> float:
        return max([d["lel"] for d in self.m.leak.detectors()] or [0.0])

    def tick(self) -> None:
        m, lk = self.m, self.m.leak
        if self.leak_at is None and self.state_of("R1") == "WELD":
            self.leak_at = m.t
            self.run.alert("weld", "Arc on — Unit 1 seam-tracking the riser weld")
        if self.leak_at is not None and lk.fid is None and m.t >= self.leak_at + 1.0:
            lk.start(self.flange, m.leak_flow())
            m.note("Sensor", "GAS_LEAK", f"flange {self.flange} leaking")
        if lk.on and self.trip_t is None and self.gas_max() >= ALARM_LEL:
            self.trip(lk.fid)
        if self.trip_t is not None and self.ctl_seen is None and (
                m.ctl is not None and m.ctl is not self.ctl0 or m.t >= self.trip_t + ARGON_WAIT_S):
            self.ctl_seen = m.t
        self.i = self.beat()

    def trip(self, fid: str) -> None:
        m = self.m
        self.trip_t, self.ctl_seen, self.ctl0 = m.t, None, m.ctl
        if not any(t.kind == "gas" and t.poi == fid for t in self.open_tasks()):
            with m.fleet.lock:
                self.f.schedule(self.f.t, "gas_alarm", flange=fid, scenario="loose", only="R2")
        g = self.gas_max()
        self.run.alert("gas_alarm", f"Gas alarm · {fid} at {g * 100:.0f} %LEL")
        m.note("Sensor", "GAS_ALARM", f"GD-{fid} {g * 100:.0f} %LEL -> alarm, hot work frozen within 15 m")
        m.flash(f"Gas alarm · {fid}", f" — fixed sensor at {g * 100:.0f} %LEL, hot work frozen", "stop", "GAS_CHECK", prio=3)
        if m.control is not None and hasattr(m.control, "nudge"):
            m.control.nudge(f"event:gas_alarm:{m.t:.0f}")

    def beat(self) -> int:
        m, lk = self.m, self.m.leak
        gas = [t for t in self.open_tasks("gas")]
        weld = self.weld
        valve = m.port.extra.get(m.valve_job or "")
        if self.trip_t is None:
            return 0
        if not lk.on and not lk.queued and not gas and self.gas_max() < ALARM_LEL and (
                weld.status == "done" or self.state_of("R1") in ("GRIND", "WELD", "INSPECT_WELD", "INSPECT")
                and weld.robot == "R1"):
            self.focus = "R1"
            return 7
        if lk.isolated and not lk.on:
            self.focus = "R1"
            return 6
        if m.valve_ft is not None:
            self.focus = "R1"
            return 5
        if valve is not None and valve["status"] == "WAITING_APPROVAL":
            self.focus = "R2"
            return 4
        if lk.seek is not None:
            self.focus = "R2"
            return 3
        if any(t.robot == "R2" for t in gas):
            self.focus = "R2"
            return 2
        if self.ctl_seen is not None:
            return 2 if gas else 1
        return 1

    def key(self) -> dict:
        return {"label": "gas", "value": round(self.gas_max() * 100, 1), "unit": "%LEL"}

    def fire(self, did: str) -> tuple[str, str]:
        m, lk = self.m, self.m.leak
        if did == "leak_reopens":
            fid = lk.fid or self.flange
            if lk.on:
                lk.plume.set_flow(lk.plume.q * 2.0)
                return "Leak worsens", f" — {fid} flow doubled, Unit 2 keeps localising"
            lk.start(fid, m.leak_flow())
            m.override.pop("R2", None)
            self.trip_t = None
            return "Leak reopens", f" at {fid} — the fixed sensor will trip again, Unit 2 re-dispatched"
        if did == "second_leak":
            fid = self.second
            if lk.add(fid, m.leak_flow()) == "now":
                self.trip_t = None
                return f"Second leak at {fid}", " — the fixed sensor will trip, Unit 2 re-dispatched"
            with m.fleet.lock:
                self.f.schedule(self.f.t, "gas_alarm", flange=fid, scenario="loose", only="R2")
            self.run.alert("gas_alarm", f"Gas alarm · {fid}")
            return f"Second leak at {fid}", " — queued for Unit 2 after the current flange"
        if did == "cut_link":
            if m.control is not None:
                m.control.cut_link(True)
            m.note("Operator", "LINK_CUT", "Gemini link cut -> edge rules only")
            return "Gemini link cut", " — the fleet carries on with its rule-based fallback"
        return did, ""


class Patrol(Script):
    id, name, main = "patrol", "Deck patrol", "R3"
    steps = ("Unit 3 walking the patrol route", "Re-routing around a blocked walkway", "Picking up a fallen can",
             "Can stowed in the back tote", "Patrol done — back to the dock", "Unit 3 back at the dock")
    disturbances = (("more_cans", "More small cans"), ("unfamiliar_object", "Unfamiliar object"),
                    ("drum", "205 L drum on the walkway"))
    robots = (("R1", [-12.0, 9.4, 42.5], 0.9), ("R2", [-21.0, 4.0, 31.25], 0.9), ("R3", [-1.0, -4.5, 42.5], 0.42))
    route = ("gauge_06", "gauge_05", "gauge_07", "gauge_05")
    crate_xyz, puddle_xyz, drum_xyz, drum_edge = [9.0, -6.8, 42.5], [13.0, -9.0, 42.5], [14.0, -4.5, 42.5], "w5"

    def setup(self) -> None:
        self.task("rust", JUDGES_RUST[1][0], JUDGES_RUST[1][1], severity=0.6, exposure=0.6, only="R1")
        self.task("leak_check", "F1", only="R2", deadline=CHECK_DEADLINE_S)
        self.k = 0
        self.leg = self.task("inspect", self.route[0], only="R3")
        self.laps = 0
        self.dropped = self.can_dropped = None
        self.cans = 0
        self.rr0 = None
        self.seen_cans: set = set()
        self.my_cans: set = set()
        self.drum_t = None

    def tick(self) -> None:
        m, f = self.m, self.f
        r = self.robot("R3")
        if self.leg is not None and self.leg.status == "done":
            self.k += 1
            more = self.k < len(self.route) or len(self.fired) < len(self.disturbances) and self.laps < 6
            if self.k >= len(self.route) and more:
                self.k, self.laps = 0, self.laps + 1
            self.leg = self.task("inspect", self.route[self.k], only="R3") if self.k < len(self.route) else None
        if self.dropped is None and r is not None and r.legs and r.state == "TRANSIT" and r.pos[0] > 10.0:
            self.rr0 = len(f.reroutes)
            try:
                self.run.hazard("drop_obstacle", kind="crate", at={"robot": "R3", "ahead_m": 5.0})
            except ValueError:
                self.run.hazard("drop_obstacle", kind="crate", at=self.crate_xyz)
            self.run.hazard("drop_obstacle", kind="puddle", at=self.puddle_xyz)
            self.dropped = m.t
        if self.dropped is not None and self.can_dropped is None and m.t > self.dropped + 8.0 and r.state == "TRANSIT":
            self.drop_cans(1)
            self.can_dropped = m.t
        for t in f.tasks.values():
            if t.kind == "pick_can" and t.status == "done" and t.id not in self.seen_cans and t.robot is None \
                    and t.id in self.my_cans:
                self.seen_cans.add(t.id)
                self.cans += 1
        self.i = self.beat()

    def drop_cans(self, n: int, **kw) -> list:
        out = []
        for k in range(n):
            ids = []
            for at in ({"robot": "R3", "ahead_m": 4.0 + 5.0 * k}, "R3"):
                try:
                    ids = self.run.hazard("drop_can", size="small", at=at, **kw)
                    break
                except ValueError:
                    continue
            for hid in ids:
                ob = self.f.hz.obstacles.get(hid) or {}
                t = self.f.tasks.get(ob.get("task") or "")
                if t is not None:
                    t.only = "R3"
                    self.my_cans.add(t.id)
                    out.append(t)
        return out

    def beat(self) -> int:
        r = self.robot("R3")
        cans = self.open_tasks("pick_can", "R3")
        if self.leg is None and not cans and r.state in ("CHARGING", "DOCKED", "DOCK_QUEUE"):
            return 5
        if self.leg is None and not cans:
            return 4
        if cans and any(t.robot == "R3" for t in cans):
            return 2
        if self.cans:
            return 3
        if self.rr0 is not None and any(x["robot"] == "R3" for x in self.f.reroutes[self.rr0:]):
            return max(self.i, 1)
        return max(self.i, 0)

    def key(self) -> dict:
        return {"label": "cans", "value": self.cans, "unit": "collected"}

    def fire(self, did: str) -> tuple[str, str]:
        m = self.m
        if did == "more_cans":
            ts = self.drop_cans(2)
            return "More small cans", f" — {len(ts)} on Unit 3's walkway, the pickup policy pre-empts the patrol"
        if did == "unfamiliar_object":
            ts = self.drop_cans(1, colour="yellow", unknown=True)
            m.note("Unit 3", "UNFAMILIAR", "object not in the can class -> grasp from the geometric depth prior")
            return "Unfamiliar object", " — Unit 3 grasps it from the geometric grasp prior" if ts else " — dropped"
        if did == "drum":
            ids = self.run.hazard("drop_obstacle", kind="barrel", at=self.drum_xyz, static=True, size="205L")
            self.drum_t = m.t
            m.note("Unit 3", "TOO_BIG", "205 L drum on w5: ~200 kg, beyond the gripper -> reported to the coordinator")
            closed = self.run.hazard("close_walkway", edge=self.drum_edge)
            who = "Argon notified; " if m._budget_ok() else ""
            m.note("Coordinator", "CLOSE_WALKWAY", f"{self.drum_edge} closed around the drum -> fleet re-routes")
            if m.control is not None and getattr(m.control, "agent", None) is not None and m._budget_ok():
                m._agent(lambda: m.control.finding({"kind": "drum_on_walkway", "severity": "warning",
                                                    "target": self.drum_edge, "detail": {"hazards": ids + closed}},
                                                   now=m.t))
            return "205 L drum — too big to lift", f" — Unit 3 reports it; {who}coordinator closes {self.drum_edge}, fleet re-routes"
        return did, ""


class Repair(Script):
    id, name, main = "repair", "Pipe repair", "R1"
    poi = "hero_bay"
    seam = ("weld_riser_east", [10.0, -8.0, 42.5])
    steps = ("Rust detector finds the patch", "Unit 1 walking to the pipe", "Spray, soak, scrub — force-limited",
             "Verified clean — paint waits for your approval", "Painting the patch", "Weld inspection finds a crack",
             "Crack repair — grind and seam-tracked weld", "Pipe repaired")
    disturbances = (("rust_returns", "Rust reappears"), ("gust", "Wind gust"))
    robots = (("R1", [9.0, -4.5, 42.5], 0.95), ("R2", [-21.0, 4.0, 31.25], 0.9), ("R3", [10.0, 12.0, 20.0], 0.8))

    def setup(self) -> None:
        self.rust = self.task("rust", self.poi, severity=0.9, exposure=1.0, only="R1", scenario="repair")
        self.task("leak_check", "F1", only="R2", deadline=CHECK_DEADLINE_S)
        self.task("rust", JUDGES_RUST[2][0], JUDGES_RUST[2][1], severity=0.5, exposure=0.9, only="R3")
        self.paint_job = None
        self.weld = None
        self.gust_wait = False
        tr = self.m.fleet.jobs.get("rust", "repair")
        p = next((s[0] for s in tr.segments if s[1] == "PAINT"), None)
        self.paint_at = None if p is None else SETUP_S["rust"] + p
        self.m.note("Detector", "RUST", f"rust patch confirmed at {self.poi} -> repair queued for Unit 1")

    def tick(self) -> None:
        m = self.m
        st = self.state_of("R1")
        rt = self.rust
        if self.paint_at is not None and rt.status in OPEN:
            j = m.port.extra.get(self.paint_job or "")
            if rt.progress >= self.paint_at - 1.0 and j is None:
                self.paint_job = m.port.queue("paint", self.poi, "SUPV", 2, params={"gate": "R1"}, by="Unit 1",
                                              why=f"Patch at {self.poi} verified clean; paint needs your approval.")
                j = m.port.extra[self.paint_job]
            if j is None or j["status"] == "WAITING_APPROVAL":
                with m.fleet.lock:
                    rt.progress = min(rt.progress, self.paint_at - 1.0)
            elif j["status"] == "QUEUED" and j.get("approved_by"):
                j["status"] = "RUNNING"
        if rt.status == "done":
            j = m.port.extra.get(self.paint_job or "")
            if j is not None and j["status"] == "RUNNING":
                j["status"] = "DONE"
            if self.weld is None:
                self.weld = self.task("weld", self.seam[0], self.seam[1], scenario="crack", only="R1")
        if self.gust_wait and st == "SCRUB":
            self.gust_wait = False
            self.gust()
        self.i = self.beat(st)

    def gust(self) -> None:
        m = self.m
        m.gusts.append((m.t, GUST_PEAK_N, GUST_S))
        m.note("Storm", "GUST", f"gust +{GUST_PEAK_N:.0f} N on the arm -> force shield holds the brush under 40 N")
        m.flash("Gust on the arm", f" — +{GUST_PEAK_N:.0f} N, the force shield holds the brush under 40 N", "warn", "SCRUB",
                prio=3)

    def beat(self, st: str) -> int:
        w = self.weld
        j = self.m.port.extra.get(self.paint_job or "")
        if w is not None and w.status == "done" and not self.open_tasks("rust", "R1"):
            return 7
        if w is not None and st in ("GRIND", "WELD", "INSPECT_WELD"):
            return 6
        if w is not None and st == "INSPECT":
            return 5
        if st == "PAINT":
            return 4
        if st == "VERIFY" or j is not None and j["status"] == "WAITING_APPROVAL":
            return 3
        if st in ("SPRAY", "SOAK", "SCRUB", "BACK_OFF"):
            return 2
        if st == "DETECT":
            return 0 if self.i <= 1 else self.i
        if self.rust.status == "assigned" or w is not None:
            return 5 if w is not None and self.rust.status == "done" else 1
        return 0

    def key(self) -> dict:
        f = self.m.brush_force("R1") if self.state_of("R1") == "SCRUB" else 0.0
        return {"label": "brush force", "value": round(float(f), 1), "unit": "N"}

    def fire(self, did: str) -> tuple[str, str]:
        m = self.m
        if did == "rust_returns":
            rt = self.rust
            j = m.port.extra.get(self.paint_job or "")
            if rt.status in OPEN:
                with m.fleet.lock:
                    rt.progress = min(rt.progress, SETUP_S["rust"])
                if j is not None and j["status"] in ("WAITING_APPROVAL", "QUEUED", "RUNNING"):
                    j["status"] = "CANCELLED"
                self.paint_job = None
            else:
                self.rust = self.task("rust", self.poi, severity=0.9, exposure=1.0, only="R1", scenario="repair")
                self.paint_job = None
            self.i = 0
            m.note("Detector", "RUST_RETURNS", f"rust back at {self.poi} -> re-detect and re-scrub")
            return "Rust reappears", f" at {self.poi} — Unit 1 re-detects and re-scrubs"
        if did == "gust":
            if self.state_of("R1") == "SCRUB":
                self.gust()
                return "Gust on the arm", " — the force shield holds the brush under 40 N"
            if self.open_tasks("rust", "R1"):
                self.gust_wait = True
                return "Gust forecast", " — it hits when the brush is back on the pipe"
            self.gust()
            return "Gust on the arm", " — no brush contact, Unit 1 holds the tool clear"
        return did, ""


class Nav(Script):
    id, name, main = "nav", "Navigation", "R3"
    steps = ("Unit 3 walking the deck loop", "Crate ahead — re-routing the other way")
    robots = (("R1", [-12.0, 9.4, 42.5], 0.9), ("R2", [-21.0, 4.0, 31.25], 0.9), ("R3", [9.0, -4.5, 42.5], 0.98))
    route = ("hero_bay_mouth", "gauge_06", "gauge_05", "nav_west")
    weather = DEMO_WEATHER
    west, center = [9.0, -6.75, 42.5], [16.5, -6.75, 42.5]

    def setup(self) -> None:
        self.run.hazard("clear", id="all")
        if "nav_west" not in self.f.rm.poi_node:
            self.f.rm.add_poi("nav_west", self.west, "top_deck", standoff=0.0)
        self.k = 0
        self.leg = self.go()
        self.run.cam_at = list(self.center)

    def sim_events(self) -> list:
        return [{"name": "camera", "args": {"mode": "deck", "at": list(self.center), "view": None}},
                {"name": "select", "args": {"robot": "R3"}}, {"name": "views", "args": {"robot": "R3", "on": True, "kind": "fpv"}}]

    def go(self):
        t = self.task("inspect", self.route[self.k % len(self.route)], only="R3")
        t.duration = 1.0
        return t

    def tick(self) -> None:
        if self.leg.status not in OPEN:
            self.k += 1
            self.leg = self.go()
        self.i = int(any(o.get("effect") == "block" for o in self.f.hz.obstacles.values()))

    def done(self) -> bool:
        return False

    def ahead(self, d: float) -> list:
        rm, r = self.f.rm, self.robot("R3")
        pts, node = [r.pos] + [rm.pos[lg.v] for lg in r.legs], r.legs[-1].v if r.legs else r.node
        for j in range(len(self.route) + 1):
            path = rm.route(node, rm.poi_node[self.route[(self.k + j) % len(self.route)]])[1]
            pts += [rm.pos[v] for v in path[1:]]
            node = path[-1] if path else node
        stops = [rm.pos[rm.poi_node[p]][:2] for p in self.route]
        for s in np.arange(d, d + 12.0, 0.5):
            q = walk_along(pts, float(s))
            if min(float(np.linalg.norm(q[:2] - w)) for w in stops) > 1.8:
                return [round(float(v), 3) for v in q[:3]]
        return [round(float(v), 3) for v in walk_along(pts, d)[:3]]


class Flyby(Script):
    id, name, main = "flyby", "Fly-by", ""
    robots = tuple(JUDGES_ROBOTS)

    def setup(self) -> None:
        self.task("leak_check", "F5", deadline=CHECK_DEADLINE_S)
        for poi, xyz, sev, exp in JUDGES_RUST:
            self.task("rust", poi, xyz, severity=sev, exposure=exp)
        self.task("weld", JUDGES_WELD[0], JUDGES_WELD[1], scenario=JUDGES_WELD[2])

    def done(self) -> bool:
        return False


class Demo(Script):
    deck = 42.5
    cam: dict = {}

    def setup(self) -> None:
        self.t0 = self.run.clock()
        with self.m.fleet.lock:
            self.f.hz.clear("all")
            self.f.tasks = {}
            self.f.hz.apply(self.f.rm)
            self.m.fleet.snap = self.f.snapshot()
        for r in self.f.robots:
            if r.id != self.main:
                self.pose(r.id, r.pos[:2], 0.0, float(r.pos[2]))

    def e(self) -> float:
        return self.run.clock() - self.t0

    def pose(self, rid, xy, yaw, z=None, state="IDLE", task=None, job_state=None, rollout=None, rust=None) -> None:
        self.m.override[rid] = {"xy": np.asarray(xy, float)[:2].copy(), "yaw": float(yaw), "z": self.deck if z is None else z,
                                "mode": "DEMO", "state": state, "task": task, "job_state": job_state, "rollout": rollout,
                                "rust": rust}

    def sim_events(self) -> list:
        return [{"name": "camera", "args": dict(self.cam)}, {"name": "select", "args": {"robot": self.main}},
                {"name": "views", "args": {"robot": self.main, "on": True, "kind": "fpv"}}]

    def done(self) -> bool:
        return self.i == len(self.steps) - 1


def walk(a, b, u: float) -> np.ndarray:
    u = min(max(u, 0.0), 1.0)
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * u


class DemoCan(Demo):
    id, name, main = "demo_can", "Demo: Can pick-up", "R3"
    steps = ("A can drops on the walkway", "Unit 3 walks to the can", "Suction cup seals on the can",
             "Learned pick-up carries it to the back tote", "Can stowed, Unit 3 moves on")
    robots = (("R1", [-21.0, 4.0, 31.25], 0.9), ("R2", [10.0, 12.0, 20.0], 0.9), ("R3", [1.4, -4.6, 42.5], 0.95))
    weather = DEMO_WEATHER
    start, stop, away, yaw = (1.4, -4.6), (3.5, -4.6), (4.1, -4.6), 0.0
    drop_s, walk_s, rep_s, away_s = 0.8, (1.5, 4.5), 5.0, 1.5
    lamp = [3.4, -4.6, 46.5, 25000.0]
    cam = {"mode": "zoom", "at": [3.0, -4.4, 42.5], "lens": {"dist": 4.0, "elev": 32.0, "focal": 24.0, "az": [70.0, 110.0, -65.0, -115.0]}}

    def setup(self) -> None:
        from rig.sim.taskmotion import Rollout

        super().setup()
        ro = self.ro = Rollout.load()
        self.f0, self.f1 = 40, len(ro.arm) - 25
        self.seal, self.tote = ro.first("LIFT"), ro.first("DONE")
        self.end_s = self.rep_s + (self.f1 - self.f0) / ro.hz
        self.can = None
        self.pose(self.main, self.start, self.yaw)

    def tick(self) -> None:
        e, ro = self.e(), self.ro
        k = self.f0 + (e - self.rep_s) * ro.hz
        if self.can is None and e >= self.drop_s:
            c, q = ro.can[self.f0], self.yaw
            xy = np.asarray(self.stop) + [c[0] * math.cos(q) - c[1] * math.sin(q), c[0] * math.sin(q) + c[1] * math.cos(q)]
            with self.m.fleet.lock:
                self.can = self.f.hz.drop("can", [*xy, self.deck], None, size="small", colour="red", radius=0.0,
                                          rollout=self.main, anchor=[*self.stop, self.deck, self.yaw],
                                          frames=[self.f0, self.f1], phase="ground", lamp=self.lamp)
                self.m.fleet.snap = self.f.snapshot()
            self.run.mine.add(self.can["id"])
        if e < self.walk_s[0]:
            self.pose(self.main, self.start, self.yaw)
        elif e < self.walk_s[1]:
            u = (e - self.walk_s[0]) / (self.walk_s[1] - self.walk_s[0])
            self.pose(self.main, walk(self.start, self.stop, u), self.yaw, state="TRANSIT", task="pick_can:demo")
        elif e < self.rep_s or k >= self.f1 and e < self.end_s + 0.5:
            self.pose(self.main, self.stop, self.yaw)
        elif k < self.f1:
            self.pose(self.main, self.stop, self.yaw, state="WORK", task="pick_can:demo", job_state="ROLLOUT",
                      rollout=[self.f0, self.f1])
        else:
            u = (e - self.end_s - 0.5) / self.away_s
            self.pose(self.main, walk(self.stop, self.away, u), self.yaw, state="TRANSIT" if u < 1 else "IDLE")
        ph = "ground" if e < self.rep_s else "held" if k < self.f1 else "tote"
        if self.can is not None and self.can["phase"] != ph:
            self.can["phase"] = ph
            with self.m.fleet.lock:
                self.m.fleet.snap = self.f.snapshot()
        self.i = 0 if e < self.walk_s[0] else 1 if e < self.rep_s else 2 if k < self.seal else 3 if k < self.tote else 4

    def key(self) -> dict:
        return {"label": "cans", "value": int(self.i >= 4), "unit": "in the tote"}


class DemoRust(Demo):
    id, name, main = "demo_rust", "Demo: Rust scrub", "R1"
    steps = ("Rust patch on the main line", "Unit 1 walks to the pipe", "Learned scrub policy clears the rust",
             "Patch clean, arm parked")
    robots = (("R1", [16.4, -5.25, 42.5], 0.95), ("R2", [-21.0, 4.0, 31.25], 0.9), ("R3", [10.0, 12.0, 20.0], 0.8))
    weather = DEMO_WEATHER
    patch, axis, base_z, back = (18.2567, -5.655, 43.7199), (-5.75, 43.6, 0.15), 0.32, 1.9
    walk_s, scrub_s, clear_s, park_s = (1.0, 5.0), 6.0, (6.5, 16.5), 17.5
    cam = {"mode": "zoom", "view": "rust", "at": [17.8, -5.4, 42.9], "lens": {"eye": [20.2, -2.6, 44.7], "focal": 27.0}}

    def setup(self) -> None:
        from rig.sim.gait import GaitLoop
        from rig.sim.taskmotion import scrub_base

        super().setup()
        g = GaitLoop.load()
        self.z = round(self.deck + self.base_z - (g.zmean if g else 0.36), 2)
        x, y, self.yaw = scrub_base(self.patch, *self.axis, trunk_z=self.z + (g.zmean if g else 0.36))
        self.stop = (x, y)
        self.start = (x - self.back * math.cos(self.yaw), y - self.back * math.sin(self.yaw))
        self.pose(self.main, self.start, self.yaw, self.z, rust=1.0)

    def left(self, e: float) -> float:
        a, b = self.clear_s
        u = min(max((e - a) / (b - a), 0.0), 1.0)
        return round(1.0 - u * u * (3 - 2 * u), 3)

    def tick(self) -> None:
        e = self.e()
        task, rust = "rust:demo", self.left(e)
        if e < self.walk_s[0]:
            self.pose(self.main, self.start, self.yaw, self.z, rust=rust)
        elif e < self.walk_s[1]:
            u = (e - self.walk_s[0]) / (self.walk_s[1] - self.walk_s[0])
            self.pose(self.main, walk(self.start, self.stop, u), self.yaw, self.z, state="TRANSIT", task=task, rust=rust)
        elif e < self.scrub_s:
            self.pose(self.main, self.stop, self.yaw, self.z, state="WORK", task=task, job_state="DEPLOY", rust=rust)
        elif e < self.park_s:
            self.pose(self.main, self.stop, self.yaw, self.z, state="WORK", task=task, job_state="SCRUB", rust=rust)
        else:
            self.pose(self.main, self.stop, self.yaw, self.z, task=task, rust=rust)
        self.i = 0 if e < self.walk_s[0] else 1 if e < self.scrub_s else 2 if e < self.park_s else 3

    def key(self) -> dict:
        return {"label": "rust left", "value": round(100 * self.left(self.e())), "unit": "%"}


SCRIPTS = {"gas": Gas, "patrol": Patrol, "repair": Repair, "flyby": Flyby, "demo_can": DemoCan, "demo_rust": DemoRust, "nav": Nav}
DEMOS = {"can": "demo_can", "rust": "demo_rust", "nav": "nav", "reset": "flyby"}


class Scenarios:
    def __init__(self, mission, clock=time.time):
        self.m, self.clock = mission, clock
        self.cur: Script | None = None
        self.alerts: list[dict] = []
        self.seq = 0
        self.selected: str | None = None
        self.camera = {"mode": "flyby", "robot": None, "view": None}
        self.mine: set = set()
        self.asked: set = set()
        self.last_i = None

    def alert(self, kind: str, text: str) -> dict:
        self.seq += 1
        a = {"seq": self.seq, "kind": kind, "text": text, "t": round(self.clock(), 2)}
        self.alerts = (self.alerts + [a])[-ALERTS_KEEP:]
        return a

    def hazard(self, name: str, **args) -> list[str]:
        f = self.m.fleet
        hz = f.f.hz
        before = set(hz.obstacles) | set(hz.closures)
        with f.lock:
            f.f.event(name, **args)
            f.snap = f.f.snapshot()
        new = [k for k in list(hz.obstacles) + list(hz.closures) if k not in before]
        self.mine |= set(new)
        return new

    def start(self, sid: str) -> dict:
        if sid not in SCRIPTS:
            raise ValueError(f"unknown scenario {sid!r}; expected one of {IDS}")
        m = self.m
        self.reset(SCRIPTS[sid].robots)
        s = SCRIPTS[sid](self)
        self.cur = s
        s.setup()
        if sid == "flyby":
            m.fleet.specs = [(t.kind, t.poi, t.scenario, t.severity, t.exposure) for t in m.fleet.f.tasks.values()
                             if t.kind != "pick_can"]
        self.hazard("set_weather", **s.weather)
        m.sim_out.append({"name": "set_weather", "args": dict(s.weather)})
        m.sim_out.extend(s.sim_events() if hasattr(s, "sim_events") else [])
        self.selected = s.main or None
        self.camera = {"mode": "flyby" if sid == "flyby" else (getattr(s, "cam", None) or {}).get("mode", "deck"), "robot": None, "view": None}
        self.last_i = None
        self.alerts = []
        m.note("Operator", "SCENARIO", f"{s.name} started")
        if sid != "flyby":
            m.flash(s.name, f" — {s.steps[0]}", "act", "SCENARIO", prio=3)
        with m.fleet.lock:
            m.fleet.snap = m.fleet.f.snapshot()
        return {"scenario": sid, "selected": self.selected}

    def reset(self, robots) -> None:
        m = self.m
        fl = m.fleet
        f = fl.f
        with fl.lock:
            hz = f.hz
            gone = [k for k in self.mine if k in hz.obstacles or k in hz.closures]
            for k in gone:
                hz.clear(k)
            self.mine = set()
            keep = {k for k, ob in hz.obstacles.items() if ob.get("task")}
            f.tasks = {tid: t for tid, t in f.tasks.items() if t.kind == "pick_can" and t.poi in keep
                       and t.status in OPEN}
            for t in f.tasks.values():
                t.status, t.robot, t.progress, t.arrived = "pending", None, 0.0, None
            f.pending_events.clear()
            f.alarms.clear()
            f.persons.clear()
            f.res = Reservations()
            f._marks = {}
            spec = {rid: (xyz, soc) for rid, xyz, soc in robots}
            for r in f.robots:
                xyz, soc = spec.get(r.id, (r.pos, r.battery.soc))
                r.node = nearest_node(f.rm, xyz)
                r.pos = f.rm.pos[r.node].copy()
                r.legs, r.trail, r.task, r.goal = [], [], None, None
                r.to_dock = r.low = r.paused = r.failed = r.stuck = False
                r.state, r.prev_path, r.reroute_t, r.planned_t = "IDLE", None, None, -1.0
                r.battery.wh = soc * r.battery.capacity_wh
            hz.apply(f.rm)
            f._event_tick = f._hz_tick = True
            fl.specs = []
            fl.snap = f.snapshot()
        old = m.leak
        m.leak = GasLeak(old.flanges, old.bounds, old.seed, old.field)
        m.leak.t = old.t
        m.override.clear()
        m.isolate_at = m.valve_job = m.valve_ft = None
        m.valve_by, m.wait_isolation, m.valves = None, False, {}
        m.gusts = []
        m.flow_gs = FLOW_GS
        with m.port.lock:
            m.port.extra.clear()
            m.port.prio.clear()
        if m.control is not None and getattr(m.control, "link_cut", False):
            m.control.cut_link(False)
        m._flash = None

    def demo(self, did: str) -> dict:
        if did not in DEMOS:
            raise ValueError(f"unknown demo {did!r}; expected one of {tuple(DEMOS)}")
        return self.start(DEMOS[did])

    def disturb(self) -> dict:
        s = self.cur
        if s is None or not s.disturbances:
            raise ValueError("no scenario with disturbances is running")
        left = [d for d, _ in s.disturbances if d not in s.fired]
        if not left:
            return {"disturb": None, "left": 0}
        did = left[0]
        s.fired.append(did)
        lead, rest = s.fire(did)
        self.alert("disturbance", lead + rest)
        self.m.note("Operator", "DISTURB", f"{did}: {lead}{rest}")
        self.m.flash(lead, rest, "warn", "DISTURB", prio=3)
        return {"disturb": did, "left": len(left) - 1}

    def select(self, rid) -> dict:
        if rid in (None, "", "null"):
            self.selected = None
            return {"selected": None}
        from rig.mission.live import robot_id
        out = robot_id(rid, self.m.ids())
        if out is None:
            raise ValueError(f"unknown robot {rid!r}; expected one of {self.m.ids()}")
        self.selected = out
        return {"selected": out}

    def pos(self, poi):
        f = self.m.fleet.f
        n = f.rm.poi_node.get(poi)
        return None if n is None else [float(v) for v in f.rm.pos[n][:3]]

    def targets(self) -> dict:
        s, f = self.cur, self.m.fleet.f
        out = {}
        obs = [o for o in f.hz.obstacles.values() if o.get("xyz")]
        if obs:
            out["obstacle"] = list(obs[-1]["xyz"])
        cans = [o for o in obs if o.get("kind") == "can" or o.get("task")]
        if cans:
            out["can"] = list(cans[-1]["xyz"])
        if s is None or s.id == "flyby":
            return {**CAM_DEFAULTS, **out}
        if s.id == "gas":
            fl = self.m.valves.get(s.flange, s.valve_xyz)
            out["flange"], out["weld"] = list(fl), list(s.seam[1])
            out["deck"] = [(a + b) / 2 for a, b in zip(fl, s.seam[1])]
        elif s.id == "patrol":
            r = s.robot("R3")
            out["deck"] = list(cans[-1]["xyz"]) if cans else [11.0, -6.0, 42.5]
            if cans:
                out["can"] = list(cans[-1]["xyz"])
            elif r is not None:
                out["can"] = [float(v) for v in r.pos[:3]]
        elif s.id == "nav":
            r = s.robot("R3")
            out["deck"] = list(s.center)
            if r is not None:
                out["can"] = [float(v) for v in r.pos[:3]]
        elif s.id == "repair":
            rust = (self.pos(s.poi) or [18.75, -5.75, 42.5])[:2] + [43.1]
            out["rust"], out["weld"], out["deck"] = rust, list(s.seam[1]), rust
        return {**CAM_DEFAULTS, **out}

    def view_at(self) -> list:
        return list(getattr(self, "cam_at", None) or self.targets()["deck"])

    def set_camera(self, args: dict) -> dict:
        mode, view = args.get("mode"), args.get("view")
        if mode in ("flyby", "rig"):
            self.camera, self.cam_at = {"mode": mode, "robot": None, "view": None}, None
            return {"camera": self.camera}
        tg = self.targets()
        key = "deck" if mode == "deck" else "obstacle" if mode == "obstacle" else view
        at = tg.get(key)
        if at is None:
            raise ValueError(f"no {key or mode} view here; have {sorted(tg)}")
        cam = "zoom" if mode == "zoom" else mode
        self.camera = {"mode": cam, "robot": None, "view": key if cam == "zoom" else None}
        self.cam_at = list(at)
        self.m.sim_out.append({"name": "camera", "args": {"mode": cam, "at": at, "view": self.camera["view"]}})
        return {"camera": self.camera, "views": [k for k in ("rust", "can", "flange", "weld") if k in tg]}

    def tick(self) -> None:
        s = self.cur
        if s is not None:
            s.tick()
        self._approvals()
        if s is None:
            return
        if s.id != "flyby" and s.i != self.last_i:
            if self.last_i is not None:
                self.m.flash(f"{s.name} · {s.steps[s.i]}", f" — {self.beat_caption(s)}" if self.beat_caption(s) else "",
                             "ok" if s.done() else "act", "STEP", BEAT_S, prio=2)
            self.last_i = s.i

    def beat_caption(self, s: Script) -> str:
        m = self.m
        if s.id == "gas":
            cloud = m._budget_ok() and getattr(m.control, "agent", None) is not None
            return {1: "Argon reads the fixed sensors" if cloud else "Argon offline — edge rules read the fixed sensors",
                    2: "fleet coordinator pre-empts Unit 2",
                    3: f"{METHODS[m.leak.method]} on the trunk and arm-tip sensors",
                    4: "approve the valve isolation in the inbox", 5: "Unit 1 walks to the valve upstream",
                    6: "weld frozen while gas is above 10 %LEL", 7: "hot-work gate released"}.get(s.i, "")
        if s.id == "patrol":
            return {1: "puddles are a soft cost", 2: "pickup policy on depth geometric priors",
                    4: f"battery {self.m._robot('R3').battery.soc:.0%}"}.get(s.i, "")
        if s.id == "repair":
            return {2: "brush force held under 40 N", 3: "approve paint in the inbox",
                    6: "seam tracking on the laser profile"}.get(s.i, "")
        return ""

    def _approvals(self) -> None:
        for j in self.m.port.extra.values():
            if j["status"] == "WAITING_APPROVAL" and j["id"] not in self.asked:
                self.asked.add(j["id"])
                self.alert("approval", f"Approval needed · {j['kind'].replace('_', ' ')} {j['target']}")

    def toast(self, name: str, args: dict) -> None:
        if name not in TOAST:
            return
        text = TOAST[name].format(kind=str(args.get("kind", "obstacle")).capitalize(), level=args.get("level", ""))
        self.alert("toast", text)

    def header(self) -> dict:
        s = self.cur
        out = {"selected": self.selected, "camera_mode": dict(self.camera), "cam_views": [k for k in ("rust", "can", "flange", "weld", "obstacle") if k in self.targets()], "alerts": list(self.alerts)}
        if s is None or s.id == "flyby":
            out["scenario"] = None
            return out
        nxt = next((k for k, (d, _) in enumerate(s.disturbances) if d not in s.fired), None)
        out["scenario"] = {"id": s.id, "name": s.name, "step": s.steps[s.i], "step_i": s.i, "steps": list(s.steps),
                           "key": s.key(), "done": s.done(),
                           "disturbances": [{"id": d, "name": n, "fired": d in s.fired} for d, n in s.disturbances],
                           "next": nxt, "main": s.main, "focus": s.focus}
        return out

    def ahead(self, d: float = 3.0):
        s = self.cur
        if isinstance(s, Nav):
            return s.ahead(d)
        return {"robot": self.focus() or self.selected or "R3", "ahead_m": d}

    def focus(self) -> str | None:
        s = self.cur
        if s is None or s.id == "flyby":
            return None
        return s.focus or s.main


def highlight(d: dict) -> list[str]:
    st = d.get("job_state") or d.get("state") or ""
    task = (d.get("task") or "").split(":")[0]
    keys = []
    if task in ("gas", "leak_check") or float(d.get("gas_lel") or 0) >= 0.05:
        keys.append("gas_lel")
    if st in ("SCRUB", "BACK_OFF", "RETORQUE", "VALVE_CLOSE"):
        keys.append("force_n")
    if task == "weld" or st in ("GRIND", "WELD"):
        keys.append("thermal_c")
    if task in ("pick_can", "rust", "valve") and st not in ("TRANSIT", "WAIT"):
        keys.append("reach")
    if st in ("TO_DOCK", "CHARGING", "DOCK_QUEUE") or float((d.get("battery") or {}).get("soc") or 1) < 0.3:
        keys.append("battery")
    return keys


def shielded(raw, limit: float = 38.0, give: float = 0.15):
    raw = np.asarray(raw, float)
    return np.where(raw > limit, limit + (raw - limit) * give, raw)


def gust_force(ts, gusts) -> np.ndarray:
    ts = np.asarray(ts, float)
    g = np.zeros_like(ts)
    for t0, peak, dur in gusts or ():
        u = (ts - t0) / dur
        g += np.where((u >= 0) & (u <= 1), peak * 0.5 * (1 - np.cos(2 * math.pi * u)), 0.0)
    return g
