from __future__ import annotations

import argparse
import importlib
import json
import math
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from integration.dual_argon import (FIX_LEAK, FIXER, GO_CHECK, HOLD, LEAK_UNSEEN_BY_FIXED, NEAR_M, OK,
                                    PIPE_SENSOR_FAULT, RESUME_RUST, SCRUBBER, SPARK_RADIUS_M, ArgonLink, Finding, LiveCaller,
                                    ReplayCaller, Timeline, fuse_fleet, ids_for, led)
from integration.fleet_beats import banner, cut_link, stuck_sensor
from integration.fleet_scenario import SNIFF_Z, Scenario
from policy.fleet_units.fixer import Fixer, FixerConfig
from policy.fleet_units.scrubber import build_scrubber
from policy.leak.world import LeakWorld
from rig.captions import caption, robot_entry

DT = 0.1
DECK_Z = 42.5
LIDAR_RANGE, BEAMS = 4.0, 72
V_MIN, W_MAX = -0.3, 1.5
U1_SPAWN = np.array([18.26, -5.25])
LEAK, STUCK = "F5", "G-F6"
NAMES = {SCRUBBER: "Unit 1 · Scrubber", FIXER: "Unit 2 · Fixer"}
TONE_STATE = {"SNIFF": "LEAK_APPROACH", "INSPECT": "LEAK_INSPECT", "VERIFY": "LEAK_VERIFY", "RETURN": "TO_DOCK",
              "RETREAT": "GAS_RETREAT", "ESCALATE": "ESCALATE_TO_HUMAN"}


def _mod(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def fixer_dock(scn: Scenario) -> np.ndarray:
    paths = _mod("integration.fleet_paths")
    dock = getattr(paths, "DOCK", None) if paths else None
    if isinstance(dock, dict) and FIXER in dock:
        return np.asarray(dock[FIXER], float)[:2]
    g = scn.flanges[LEAK]
    return g.center[:2] + 3.0 * unit(g.normal[:2])


class DeckMap:
    def __init__(self, posts=()):
        self.circles = [(np.asarray(c, float)[:2], float(r)) for c, r in posts]
        self.objects: dict[str, tuple] = {}

    def drop(self, oid: str, xy, r: float = 0.25) -> None:
        self.objects[oid] = (np.asarray(xy, float)[:2], float(r))

    def scan(self, pose, others=()):
        x, y, yaw = pose
        ang = np.linspace(-math.pi, math.pi, BEAMS, endpoint=False)
        d = np.stack([np.cos(ang + yaw), np.sin(ang + yaw)], 1)
        o = np.array([x, y])
        out = np.full(BEAMS, LIDAR_RANGE)
        for c, r in self.circles + list(self.objects.values()) + [(np.asarray(p, float)[:2], 0.3) for p in others]:
            oc = o - c
            b = d @ oc
            disc = b * b - (oc @ oc - r * r)
            t = -b - np.sqrt(np.where(disc >= 0, disc, 0.0))
            out = np.where((disc >= 0) & (t > 0), np.minimum(out, t), out)
        return out, ang


def _pursuit(path, pose, ranges, angles, vmax):
    x, y, yaw = pose
    i = int(np.linalg.norm(path - np.array([x, y]), axis=1).argmin())
    rel = path[min(i + 8, len(path) - 1)] - np.array([x, y])
    lx = math.cos(yaw) * rel[0] + math.sin(yaw) * rel[1]
    ly = -math.sin(yaw) * rel[0] + math.cos(yaw) * rel[1]
    curv = 2 * ly / max(lx * lx + ly * ly, 1e-3)
    cone = np.abs(angles) < 0.5
    ahead = float(ranges[cone].min()) if cone.any() else LIDAR_RANGE
    v = max(vmax * float(np.clip((ahead - 0.35) / 1.0, 0, 1)), 0.1)
    return v, float(np.clip(curv * v, -W_MAX, W_MAX))


class Nav:
    def __init__(self, unit_id: str, xy, yaw: float, deck: DeckMap, speed: float = 0.8, others=lambda: ()):
        self.unit, self.deck, self.speed, self.others = unit_id, deck, speed, others
        self.pose = np.array([*np.asarray(xy, float)[:2], yaw], float)
        self.v = self.w = 0.0
        self.path: Optional[np.ndarray] = None
        self.goal_yaw: Optional[float] = None
        self.state = "IDLE"
        self.lidar_min = LIDAR_RANGE
        self.drive = getattr(_mod("integration.fleet_nav"), "drive", None)
        self.Controller = getattr(_mod("patrol.controller"), "PatrolController", None)
        self.ctl = None
        self.reported: set = set()
        self.recover_until = -1.0
        self.reports: list[dict] = []
        self.t = 0.0

    @property
    def xy(self) -> np.ndarray:
        return self.pose[:2].copy()

    @property
    def yaw(self) -> float:
        return float(self.pose[2])

    @property
    def backend(self) -> str:
        return "fleet_nav.drive" if self.drive else ("patrol.PatrolController" if self.Controller else "pure_pursuit")

    def go(self, goal_xy, yaw: Optional[float] = None, via=()) -> None:
        pts = [self.xy] + [np.asarray(p, float)[:2] for p in via] + [np.asarray(goal_xy, float)[:2]]
        seg = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(int(np.linalg.norm(b - a) / 0.1), 1)
            seg += [a + (b - a) * k / n for k in range(n)]
        seg.append(pts[-1])
        self.path, self.goal_yaw = np.array(seg), yaw
        if self.state != "PUSH_RECOVER":
            self.state = "DRIVE"
        self.ctl = self.Controller("pure_pursuit", self.path) if self.Controller and len(self.path) > 2 else None

    def push(self, t: float, dxy=(0.0, 0.35), dyaw: float = 0.5) -> None:
        self.pose[:2] += np.asarray(dxy, float)
        self.pose[2] = wrap(self.pose[2] + dyaw)
        self.state, self.recover_until = "PUSH_RECOVER", t + 1.5

    def step(self, dt: float) -> bool:
        self.t += dt
        ranges, angles = self.deck.scan(self.pose, self.others())
        self.lidar_min = round(float(ranges.min()), 3)
        if self.path is None:
            self.v = self.w = 0.0
            if self.state not in ("STOP_OBJECT",):
                self.state = "IDLE"
            return True
        if self.state == "STOP_OBJECT":
            self.v = self.w = 0.0
            return False
        goal = self.path[-1]
        gap = float(np.linalg.norm(goal - self.xy))
        if gap < 0.12:
            self.pose[:2] = goal
            if self.goal_yaw is not None:
                self.pose[2] = self.goal_yaw
            self.path, self.v, self.w, self.state = None, 0.0, 0.0, "ARRIVED"
            return True
        for oid, (c, r) in self.deck.objects.items():
            rel = c - self.xy
            dist = float(np.linalg.norm(rel)) - r
            if oid not in self.reported and dist < 1.0 and abs(wrap(math.atan2(rel[1], rel[0]) - self.yaw)) < 0.6:
                self.reported.add(oid)
                self.state, self.v, self.w = "STOP_OBJECT", 0.0, 0.0
                self.reports.append({"object": oid, "xy": [round(float(v), 2) for v in c], "dist_m": round(dist, 2),
                                     "robot": self.unit})
                return False
        if gap < 1.5:
            self._hop(goal, dt)
        else:
            if self.drive is not None:
                vx, wz = self.drive(self.unit, (ranges, angles), tuple(self.pose), self.v, self.w, self.path)[:2]
            elif self.ctl is not None:
                vx, wz, _ = self.ctl.step(ranges, angles, tuple(self.pose), self.v, self.w)
            else:
                vx, wz = _pursuit(self.path, self.pose, ranges, angles, self.speed)
            i = int(np.linalg.norm(self.path - self.xy, axis=1).argmin())
            look = self.path[min(i + 5, len(self.path) - 1)] - self.xy
            err = wrap(math.atan2(look[1], look[0]) - self.yaw)
            if abs(err) > 1.0:
                vx, wz = 0.0, float(np.clip(2.5 * err, -W_MAX, W_MAX))
            self._apply(float(np.clip(vx, V_MIN, self.speed)), float(np.clip(wz, -W_MAX, W_MAX)), dt)
        if self.state == "PUSH_RECOVER" and self.t >= self.recover_until:
            self.state = "DRIVE"
        return False

    def _hop(self, goal, dt: float) -> None:
        d = goal - self.xy
        n = float(np.linalg.norm(d))
        step = min(n, self.speed * dt)
        yaw = math.atan2(d[1], d[0])
        self.w = float(np.clip(wrap(yaw - self.yaw) / dt, -W_MAX, W_MAX))
        self.pose[2] = wrap(self.yaw + self.w * dt)
        self.pose[:2] += d / max(n, 1e-9) * step
        self.v = step / dt

    def _apply(self, vx: float, wz: float, dt: float) -> None:
        self.v, self.w = vx, wz
        self.pose[2] = wrap(self.pose[2] + wz * dt)
        self.pose[:2] += vx * dt * np.array([math.cos(self.pose[2]), math.sin(self.pose[2])])

    def detour(self, xy, side: float = 0.9) -> None:
        if self.path is None:
            return
        goal, yaw = self.path[-1], self.goal_yaw
        c = np.asarray(xy, float)[:2]
        h = unit(goal - self.xy)
        n = np.array([-h[1], h[0]]) * side
        self.state = "DRIVE"
        self.go(goal, yaw, via=[c - 0.8 * h + n, c + 0.8 * h + n])

    def status(self) -> dict:
        return {"vx": round(self.v, 3), "wz": round(self.w, 3), "lidar_min": self.lidar_min, "state": self.state}


class Lockstep:
    def __init__(self, scrubber):
        self.s, self.o = scrubber, scrubber.orch
        self.inner = self.o._interrupt
        self.o._interrupt = self._hook
        self.cv = threading.Condition()
        self.until = 0.0
        self.parked = False
        self.th: Optional[threading.Thread] = None

    def _hook(self):
        with self.cv:
            while self.o.t >= self.until - 1e-9:
                self.parked = True
                self.cv.notify_all()
                self.cv.wait()
        return self.inner()

    def _run(self):
        try:
            self.s.step()
        finally:
            with self.cv:
                self.cv.notify_all()

    def advance(self, t: float) -> None:
        with self.cv:
            self.until, self.parked = t, False
            self.cv.notify_all()
            if self.th is None or not self.th.is_alive():
                if self.s.done or self.o.t > t - 1e-9:
                    return
                self.th = threading.Thread(target=self._run, daemon=True)
                self.th.start()
            while self.th.is_alive() and not self.parked:
                self.cv.wait(0.5)


class Unit1:
    def __init__(self, scn: Scenario, deck: DeckMap, soak_s: float = 3.0, others=lambda: ()):
        self.s, parts = build_scrubber()
        self.bridge = parts["bridge"]
        self.s.orch.cfg.soak_s = soak_s
        self.work = np.asarray(self.bridge.work_pos, float)[:2].copy()
        ey = unit(scn.flanges[LEAK].center[:2] - U1_SPAWN)
        self.M = np.stack([np.array([ey[1], -ey[0]]), ey], 1)
        self.face = math.atan2(ey[1], ey[0])
        deck.circles.append((U1_SPAWN + 0.75 * ey, 0.2))
        self.nav = Nav(SCRUBBER, U1_SPAWN, self.face, deck, speed=0.6, others=others)
        self.lock = Lockstep(self.s)
        self.target = U1_SPAWN.copy()

    @property
    def hot_xy(self) -> np.ndarray:
        b = self.s.work_base if self.s.held and self.s.work_base is not None else self.s.pos
        return U1_SPAWN if b is None else U1_SPAWN + self.M @ (np.asarray(b, float)[:2] - self.work)

    def world(self) -> np.ndarray:
        p = self.s.pos
        return U1_SPAWN if p is None else U1_SPAWN + self.M @ (np.asarray(p, float)[:2] - self.work)

    def step(self, t: float, gas_frac: float) -> None:
        self.bridge.gas_override = gas_frac
        self.lock.advance(t)
        tgt = self.world()
        if float(np.linalg.norm(tgt - self.target)) > 0.05:
            self.target = tgt
            self.nav.go(tgt, self.face)
        self.nav.step(DT)

    @property
    def state(self) -> str:
        return self.s.state

    def entry(self, lel: float) -> dict:
        h = self.s.hud()
        st = "HOLD" if self.s.held else self.state
        ev = {"force": h["force_n"], "force_limit": 40.0, "rust_fraction": h["rust_fraction"], "gas": lel / 100.0,
              "soak_left": h["soak_left_s"], "job": "rust"}
        return {"state": st, "caption": self.s.caption, "held": self.s.held, "paused_state": h["paused_state"],
                "force_n": h["force_n"], "rust": h["rust_fraction"], "ev": ev}


class Unit2:
    def __init__(self, scn: Scenario, deck: DeckMap, clock, others=lambda: (), sniff_s: float = 3.0):
        self.scn, self.clock = scn, clock
        self.lw = LeakWorld([scn.flanges[f] for f in scn.flanges])
        self.dock = fixer_dock(scn)
        g = scn.flanges[LEAK]
        self.nav = Nav(FIXER, self.dock, math.atan2(*(g.center[:2] - self.dock)[::-1]), deck, speed=1.0,
                       others=others)
        self.walker = NavWalker(self.nav)
        self.f = Fixer(self.lw, self.dock, scn.detectors, walker=self.walker, gas_at=self.gas, tighten=self.tighten,
                       config=FixerConfig(sniff_s=sniff_s))
        self.fixed_at: Optional[float] = None
        self.loose = False

    def gas(self, p) -> float:
        p = np.asarray(p, float)
        return self.scn.gas_frac((p[0], p[1], SNIFF_Z if len(p) < 3 else p[2]), self.clock())

    def tighten(self, fid: str, bolt: int):
        before = self.lw.tighten(fid, bolt)
        if fid == self.scn.leak_flange and self.fixed_at is None:
            self.fixed_at = self.clock()
            fix = getattr(self.scn, "fix", None)
            if fix is not None:
                fix(self.fixed_at)
            else:
                self.scn.t_fixed = self.fixed_at
        return before

    def step(self, t: float) -> None:
        if not self.loose and t >= self.scn.t_leak:
            self.loose = True
            self.lw.loosen(self.scn.leak_flange, self.scn.bolt, self.scn.loosen_rad)
            self.lw.settle()
        self.f.step(DT)

    @property
    def state(self) -> str:
        return self.f.state

    def entry(self, lel: float) -> dict:
        ev = self.f.evidence
        return {"state": self.f.state, "caption": self.f.caption, "job": self.f.job, "target": self.f.target,
                "bolt": ev.get("bolt"), "seek_mode": ev.get("seek_mode"), "diagnosis": ev.get("diagnosis"),
                "force_n": round(12.0 * float(getattr(self.f.arm, "progress", 0.0)), 2)
                if self.f.torquing else 0.0,
                "ev": {"gas": lel / 100.0, "gas_stop": 0.20, "flange": self.f.target, "bolt": ev.get("bolt"),
                       "job": "gas"}}


class NavWalker:
    def __init__(self, nav: Nav):
        self.nav = nav
        self.goal = nav.xy
        self.start = nav.xy
        self.arrived = True

    @property
    def xy(self) -> np.ndarray:
        return self.nav.xy

    @property
    def yaw(self) -> float:
        return self.nav.yaw

    @property
    def progress(self) -> float:
        total = float(np.linalg.norm(self.goal - self.start))
        return 1.0 if total < 1e-6 else float(np.clip(1 - np.linalg.norm(self.goal - self.xy) / total, 0, 1))

    def go(self, goal_xy, yaw: Optional[float] = None) -> None:
        self.goal, self.start = np.asarray(goal_xy, float)[:2].copy(), self.nav.xy
        self.arrived = False
        self.nav.go(self.goal, yaw)

    def step(self, dt: float) -> bool:
        if dt <= 0.0:
            return self.arrived
        if self.arrived:
            self.nav.step(dt)
            return True
        self.arrived = self.nav.step(dt)
        return self.arrived


class FleetDemo:
    def __init__(self, control: str = "edge", out: Optional[Path] = None, cut_link_at: Optional[float] = None,
                 replay_from: Optional[str] = None, stuck_at=None,
                 drop_at: Optional[float] = None, push_at: Optional[float] = None, log_hz: float = 5.0,
                 max_t: float = 150.0, t_leak: float = 6.0):
        self.out = Path(out) if out else None
        self.timeline = Timeline(self.out / "timeline.jsonl" if self.out else None)
        self.hud_path = self.out / "hud.jsonl" if self.out else None
        if self.hud_path:
            self.hud_path.write_text("")
        caller = None
        if control == "live":
            caller = LiveCaller()
            if not caller.available:
                raise SystemExit("no GEMINI_API_KEY for --control live")
        elif control == "replay":
            caller = ReplayCaller(replay_from or "runs/fleet_demo/live_timeline.jsonl")
        self.control = control
        self.link = ArgonLink(control, self.timeline, caller)
        self.wait = control in ("live", "replay")
        self.cut = cut_link(self.link, cut_link_at) if cut_link_at is not None else None
        self.every = max(1, round(1.0 / (log_hz * DT)))
        self.max_t, self.drop_at, self.push_at = max_t, drop_at, push_at
        self.t, self.k, self.seen = 0.0, 0, 0
        self.scn = Scenario(t_leak=t_leak, t_stuck=None, decay_tau_s=2.5)
        self.deck = DeckMap([(d.pos, 0.08) for d in self.scn.detectors])
        self.u1 = Unit1(self.scn, self.deck, others=lambda: [self.u2.nav.xy])
        self.u2 = Unit2(self.scn, self.deck, lambda: self.t, others=lambda: [self.u1.nav.xy])
        self.units = {SCRUBBER: self.u1, FIXER: self.u2}
        self.stuck_enabled = stuck_at is not False
        self.stuck_auto = stuck_at is None
        self.stuck = stuck_sensor(self.scn, math.inf if stuck_at in (None, False) else stuck_at, STUCK, 14.0)
        self.backlog: list = []
        self.battery = {SCRUBBER: 0.86, FIXER: 0.97}
        self.ids = ids_for(self.scn.detectors)
        self.events_seen: set = set()
        self.last_states: dict = {}
        self.dropped = self.pushed = False

    def lel(self, rid: str) -> float:
        return self.scn.sniff(self.units[rid].nav.xy, self.t)

    def sniffers(self) -> dict:
        return {rid: (self.lel(rid), u.nav.xy) for rid, u in self.units.items()}

    def far(self, target: str) -> float:
        d = next((d for d in self.scn.detectors if target in (d.id, d.flange)), None)
        r = 0.0 if d is None else d.dist(self.u1.hot_xy)
        return r if r > SPARK_RADIUS_M else 0.0

    def drain(self) -> None:
        for rid, u in self.units.items():
            moving = abs(u.nav.v) > 0.02 or abs(u.nav.w) > 0.02
            work = u.state in ("SCRUB", "SPRAY", "PAINT", "RETORQUE", "SNIFF", "INSPECT", "VERIFY")
            rate = 0.0012 if moving else 0.0009 if work else 0.0002
            if u.state == "DOCKED":
                rate = -0.0005
            self.battery[rid] = float(np.clip(self.battery[rid] - rate * DT, 0.0, 1.0))

    def findings(self, t: float) -> list:
        sn = self.sniffers()
        fds = self.stuck.findings(t, self.scn.detectors, sn) if self.stuck_enabled else \
            [fuse_fleet(d, sn) for d in self.scn.detectors]
        out = []
        for fd in fds:
            if fd.kind == LEAK_UNSEEN_BY_FIXED and (fd.robot_dist or 0.0) > NEAR_M:
                fd = Finding(OK, fd.sensor, fd.flange, fd.fixed_lel, fd.robot_lel, fd.robot_dist,
                             f"{fd.robot} {fd.robot_lel:.1f}%LEL at {fd.robot_dist:.1f} m is not at {fd.sensor}",
                             fd.robot, fd.hazard)
            out.append(fd)
        return out

    def robots(self) -> dict:
        out = {}
        for rid, u in self.units.items():
            x, y, yaw = u.nav.pose
            e = u.entry(self.lel(rid))
            e.pop("ev")
            out[rid] = {**e, "pose": [round(float(x), 3), round(float(y), 3), DECK_Z, round(float(yaw), 3)],
                        "lel": self.lel(rid), "nav": u.nav.status(), "battery": round(self.battery[rid], 4)}
        return out

    def _script_events(self) -> None:
        for e in self.scn.at(self.t):
            if e.name not in self.events_seen and e.t < 1e8:
                self.events_seen.add(e.name)
                self.timeline.log("scenario", self.t, **e.to_dict())

    def _disturb(self) -> None:
        nav = self.u2.nav
        if self.drop_at is not None and not self.dropped and self.t >= self.drop_at and nav.path is not None \
                and float(np.linalg.norm(nav.path[-1] - nav.xy)) > 3.0:
            self.dropped = True
            h = unit(nav.path[min(int(np.linalg.norm(nav.path - nav.xy, axis=1).argmin()) + 18,
                                  len(nav.path) - 1)] - nav.xy)
            self.deck.drop("box", nav.xy + 1.6 * h)
            self.timeline.log("scenario", self.t, name="object_dropped", object="box",
                              xy=[round(float(v), 2) for v in self.deck.objects["box"][0]], robot=FIXER)
        if self.push_at is not None and not self.pushed and self.t >= self.push_at and nav.path is not None:
            self.pushed = True
            nav.push(self.t)
            self.timeline.log("scenario", self.t, name="push", robot=FIXER)
        while nav.reports:
            rep = nav.reports.pop(0)
            self.timeline.log("object_report", self.t, **rep)
            self.timeline.log("command", self.t, kind="reroute", robot=FIXER, target=rep["object"], source="edge",
                              reason=f"{rep['object']} on the walkway {rep['dist_m']:.1f} m ahead: logged for the "
                                     f"crew, detour around it", confidence=1.0, finding="OBJECT_ON_WALKWAY")
            nav.detour(rep["xy"])

    def step(self) -> None:
        t = self.t
        if self.cut is not None:
            self.cut.tick(t)
        self._script_events()
        self.scn.detector_readings(t)
        self.u1.step(t + DT, self.scn.gas_frac((*self.u1.nav.xy, SNIFF_Z), t))
        self.u2.step(t)
        self.drain()
        self._disturb()
        fds = self.findings(t)
        robots = self.robots()
        brief = {rid: {k: r[k] for k in ("state", "lel", "pose")} | {"paused": r.get("held", False)}
                 for rid, r in robots.items()}
        cmds = self.link.tick(t, fds, brief, self.scn.detectors, self.ids, scrubber_pos=self.u1.hot_xy,
                              wait=self.wait)
        for c in cmds:
            if c.robot == SCRUBBER and c.kind == HOLD and self.far(c.target):
                self.timeline.log("hold_skipped", t, robot=SCRUBBER, target=c.target, source=c.source,
                                  why=f"{c.target} is {self.far(c.target):.1f} m from hot work, outside the "
                                      f"{SPARK_RADIUS_M:.0f} m spark radius")
            elif c.robot == SCRUBBER:
                self.u1.s.handle(c)
            elif c.robot == FIXER and not self.u2.f.command(c) and c.kind == GO_CHECK:
                self.backlog.append(c)
        if self.backlog and self.u2.state in ("DOCKED", "RETURN") and self.u2.f.command(self.backlog[0]):
            c = self.backlog.pop(0)
            self.timeline.log("dispatch", t, kind=c.kind, robot=c.robot, target=c.target, queued_at=c.t)
        if self.stuck_auto and math.isinf(self.stuck.at) and self.u2.fixed_at is not None \
                and self.u2.state in ("RETURN", "DOCKED") and not self.u1.s.held and self.link.verdict.get(f"G-{LEAK}") == OK:
            self.stuck.at = t + DT
        for rec in self.stuck.sync(self.link):
            self.timeline.log("inbox", t, **{k: v for k, v in rec.items() if k != "envelope"})
        states = {rid: (r["state"], r["nav"]["state"]) for rid, r in robots.items()}
        if self.k % self.every == 0 or cmds or states != self.last_states:
            self.log_tick(fds, robots)
        self.last_states = states
        self.k += 1
        self.t = round(t + DT, 3)

    def done(self) -> bool:
        if self.t >= self.max_t:
            return True
        quiet = self.u1.s.done and self.u2.state == "DOCKED" and self.u2.fixed_at is not None
        return quiet and (not self.stuck_enabled or bool(self.link.tickets)) and self.link.pending is None \
            and not self.link.queue

    def log_tick(self, fds, robots) -> None:
        dets = [{"id": d.id, "flange": d.flange, "lel": d.last, "led": led(d.last, d.id in self.link.faulted),
                 "pos": [round(float(v), 3) for v in d.pos]} for d in self.scn.detectors]
        row = self.timeline.log("tick", self.t, robots=robots, detectors=dets, fusion=[f.to_dict() for f in fds],
                                argon=self.link.header(), banner=banner(self.link), stuck=self.stuck.hud())
        self.write_header(row)

    def write_header(self, row: dict) -> None:
        if self.hud_path is None:
            return
        rows = self.timeline.rows[self.seen:]
        self.seen = len(self.timeline.rows)
        events = []
        for r in rows:
            if r["type"] in ("command", "ticket", "vetoed", "finding", "object_report", "scenario", "link_cut",
                             "link_restored"):
                src = r.get("source")
                events.append({"t": r["t"], "from": "Argon" if src == "argon" else "Edge" if src == "edge" else
                               r["type"].replace("_", " ").capitalize(),
                               "to": str(r.get("kind") or r.get("name") or r["type"]).upper(),
                               "reason": r.get("reason") or r.get("why") or r.get("banner") or r.get("sensor")
                               or r.get("object") or "", "robot": r.get("robot")})
        ents = []
        for rid, r in row["robots"].items():
            u = self.units[rid]
            ev = u.entry(r["lel"])["ev"]
            e = robot_entry(rid, NAMES[rid], r["state"], ev, r["pose"][:3])
            c = caption(TONE_STATE.get(r["state"], r["state"]), ev)
            e["caption"] = {"state": r["state"], "lead": r["caption"], "rest": "", "text": r["caption"],
                            "tone": c["tone"]}
            e.update(yaw=r["pose"][3], battery=r["battery"], lel=r["lel"], force_n=r.get("force_n"), rust=r.get("rust"), nav=r["nav"])
            ents.append(e)
        sensors = [{"id": d["id"], "at": f"flange {d['flange']}", "kind": "fixed", "lel": d["lel"] / 100.0,
                    "led": d["led"]} for d in row["detectors"] if d["flange"] in (LEAK, STUCK[2:])]
        sensors += [{"id": f"{rid}-sniffer", "at": NAMES[rid], "kind": "robot", "lel": r["lel"] / 100.0}
                    for rid, r in row["robots"].items()]
        mx = max(s["lel"] for s in sensors)
        u1 = row["robots"][SCRUBBER]
        hdr = {"t": row["t"], "state": u1["state"], "caption": ents[0]["caption"],
               "hud": [e["caption"]["text"] for e in ents], "events": events, "robots": ents,
               "force_n": u1.get("force_n"), "force_limit": 40.0, "rust_fraction": u1.get("rust"),
               "gas": {"sensors": sensors, "max": round(mx, 4), "alarm": any(d["led"] == "red" for d in row["detectors"])},
               "detectors": row["detectors"], "fusion": row["fusion"], "argon": row["argon"], "banner": row["banner"],
               "stuck": row["stuck"], "source": "EDGE" if self.link.link_cut or self.control == "edge" else "CLOUD",
               "fleet": {"schema": "fleet_state.v1", "t": row["t"],
                         "robots": [{"id": e["id"], "name": e["name"], "state": e["state"], "job_state": e["state"],
                                     "deck": "top_deck", "battery": {"soc": e["battery"]}, "pose": {"x": e["world_pos"][0], "y": e["world_pos"][1],
                                                                  "z": e["world_pos"][2], "yaw": e["yaw"]}}
                                    for e in ents],
                         "zones": [{"kind": "gas", "id": d["id"], "xyz": d["pos"], "radius": 6.0}
                                   for d in row["detectors"] if d["led"] in ("amber", "red")]}}
        with self.hud_path.open("a") as f:
            f.write(json.dumps(hdr, default=str) + "\n")

    def run(self) -> dict:
        while not self.done():
            self.step()
        self.link.wait_idle()
        self.log_tick(self.findings(self.t), self.robots())
        s = {"control": self.control, "t_end": self.t, "beats": beats(self.timeline.rows),
             "argon": self.link.summary(), "nav": self.u2.nav.backend,
             "inbox": list(self.stuck.inbox.filed.values()), "dock_unit2": [round(float(v), 3) for v in self.u2.dock]}
        if self.out:
            (self.out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
        return s


def _first(rows, pred, t0=None) -> Optional[float]:
    if t0 is False:
        return None
    return next((r["t"] for r in rows if (t0 is None or r["t"] >= t0) and pred(r)), None)


def _state(rid, *states):
    return lambda r: r["type"] == "tick" and r["robots"][rid]["state"] in states


def _det(sid, colour):
    return lambda r: r["type"] == "tick" and any(d["id"] == sid and d["led"] == colour for d in r["detectors"])


def _cmd(kind, robot=None, target=None):
    return lambda r: (r["type"] == "command" and r["kind"] == kind and (robot is None or r["robot"] == robot)
                      and (target is None or r["target"] == target))


def _finding(kind, sensor=None, robot=None):
    return lambda r: (r["type"] == "finding" and r["kind"] == kind and (sensor is None or r["sensor"] == sensor)
                      and (robot is None or r["robot"] == robot))


def beats(rows: list[dict]) -> dict:
    b = {}

    def at(name, pred, after=None):
        t0 = None if after is None else (b.get(after) if b.get(after) is not None else False)
        b[name] = _first(rows, pred, t0)

    g = f"G-{LEAK}"
    at("1_unit1_scrubbing", _state(SCRUBBER, "SCRUB"))
    at("1_unit2_docked", _state(FIXER, "DOCKED"))
    at("2_leak_start", lambda r: r["type"] == "scenario" and r.get("name") == "leak_start")
    at("2_gf5_amber", _det(g, "amber"), "2_leak_start")
    at("2_gf5_red", _det(g, "red"), "2_gf5_amber")
    at("3_spark_hold", _cmd(HOLD, SCRUBBER), "2_gf5_amber")
    at("3_unit1_held", _state(SCRUBBER, "HOLD"), "3_spark_hold")
    at("3_unverified", _finding("UNVERIFIED", g), "2_gf5_red")
    at("3_dispatch_unit2", _cmd(GO_CHECK, FIXER, LEAK), "3_unverified")
    at("4_unit2_walks", _state(FIXER, "TRANSIT"), "3_dispatch_unit2")
    at("4_unit2_sniffs", _state(FIXER, "SNIFF"), "4_unit2_walks")
    at("4_leak_confirmed", _finding("LEAK_CONFIRMED", g, FIXER), "4_unit2_walks")
    at("4_fix_leak", _cmd(FIX_LEAK, FIXER, LEAK), "4_leak_confirmed")
    at("4_retorque", _state(FIXER, "RETORQUE"), "4_fix_leak")
    at("4_gf5_green", _det(g, "green"), "4_retorque")
    at("5_resume_rust", _cmd(RESUME_RUST, SCRUBBER), "4_retorque")
    at("5_unit1_scrubs_again", _state(SCRUBBER, "SCRUB"), "5_resume_rust")
    at("5_unit1_done", _state(SCRUBBER, "DONE"), "5_resume_rust")
    at("5_unit2_returns", _state(FIXER, "RETURN"), "4_retorque")
    at("5_unit2_docked", _state(FIXER, "DOCKED"), "5_unit2_returns")
    at("6_gf6_stuck", lambda r: r["type"] == "tick" and r["stuck"]["led"] != "green" and
       any(d["id"] == STUCK and d["lel"] >= 10 for d in r["detectors"]))
    at("6_gf6_fault", _finding(PIPE_SENSOR_FAULT, STUCK), "6_gf6_stuck")
    at("6_ticket", lambda r: r["type"] == "ticket" and r.get("sensor") == STUCK, "6_gf6_stuck")
    at("6_inbox", lambda r: r["type"] == "inbox" and r.get("sensor") == STUCK, "6_ticket")
    at("6_no_repair_at_f6", lambda r: False)
    b["6_no_repair_at_f6"] = None if any(_cmd(FIX_LEAK, None, STUCK[2:])(r) for r in rows) else b["6_ticket"]
    at("6_object_report", lambda r: r["type"] == "object_report")
    at("6_link_cut", lambda r: r["type"] == "link_cut")
    return b


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="CPU scenario, no Isaac")
    ap.add_argument("--control", choices=("live", "edge", "replay"), default="edge")
    ap.add_argument("--cut-link-at", type=float, default=None)
    ap.add_argument("--replay-from", default="runs/fleet_demo/live_timeline.jsonl")
    ap.add_argument("--out", default="runs/fleet_demo")
    ap.add_argument("--stuck-at", type=float, default=None, help="default: when Unit 2 heads home after the fix")
    ap.add_argument("--no-stuck", action="store_true")
    ap.add_argument("--drop-object-at", type=float, default=None)
    ap.add_argument("--push-at", type=float, default=None)
    ap.add_argument("--log-hz", type=float, default=5.0)
    a = ap.parse_args(argv)
    out = Path(a.out) / a.control
    out.mkdir(parents=True, exist_ok=True)
    demo = FleetDemo(a.control, out, a.cut_link_at, a.replay_from, False if a.no_stuck else a.stuck_at,
                     a.drop_object_at, a.push_at, a.log_hz)
    s = demo.run()
    if a.control == "live":
        (Path(a.out) / "live_timeline.jsonl").write_text((out / "timeline.jsonl").read_text())
        (Path(a.out) / "live_hud.jsonl").write_text((out / "hud.jsonl").read_text())
    print(json.dumps({k: s[k] for k in ("control", "t_end", "beats", "nav", "dock_unit2")}, indent=1))
    ar = s["argon"]
    print(f"argon {ar['model']}: {ar['calls']} calls, latency {ar['latency_ms']}, cost ${ar['cost_usd']}")
    print("\n".join(ar["commands"]))


if __name__ == "__main__":
    main()
