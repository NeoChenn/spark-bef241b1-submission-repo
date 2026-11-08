from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from integration.dual_argon import (ALARM_LEL, CLEAR_LEL, FIX_LEAK, FIXER, GO_CHECK, HOLD, LOG_SENSOR_FAULT,
                                    PIPE_SENSOR_FAULT, RESUME_RUST, RETREAT, SCRUBBER, ArgonLink, FixedDetector,
                                    LiveCaller, ReplayCaller, Timeline, fuse_fleet, ids_for, led)
from rig.captions import caption, robot_entry

ZONE = np.array([18.75, -5.75, 42.5])
LIDAR_RANGE, BEAMS = 4.0, 72
V_MAX, V_MIN, W_MAX = 1.0, -0.3, 1.5
DT = 0.1
NAMES = {SCRUBBER: "Unit 1 · Scrubber", FIXER: "Unit 2 · Fixer"}


def W(x, y) -> np.ndarray:
    return ZONE[:2] + np.array([x, y], float)


LAYOUT = {
    "rust": W(0.0, -0.9), "backoff": W(-1.6, -1.4), "dock": W(12.0, -2.4), "walk_y": -1.3,
    "flanges": {"F1": W(8.0, 0.0), "F2": W(4.0, 0.0)},
    "detectors": {"G-F1": ("F1", W(8.3, 0.15)), "G-F2": ("F2", W(4.3, 0.15))},
    "boxes": [(W(-4.0, 0.25), W(16.0, 0.75))],
    "posts": [(W(2.0, -0.2), 0.12), (W(6.0, -0.2), 0.12), (W(10.0, -0.2), 0.12)],
}


def _mod(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class DeckMap:
    def __init__(self, boxes=(), posts=()):
        self.boxes = [(np.asarray(a, float), np.asarray(b, float)) for a, b in boxes]
        self.circles = [(np.asarray(c, float), float(r)) for c, r in posts]
        self.objects: dict[str, tuple] = {}

    def drop(self, oid: str, xy, r: float = 0.25) -> None:
        self.objects[oid] = (np.asarray(xy, float), float(r))

    def scan(self, pose, others=(), n: int = BEAMS, rng: float = LIDAR_RANGE):
        x, y, yaw = pose
        ang = np.linspace(-math.pi, math.pi, n, endpoint=False)
        d = np.stack([np.cos(ang + yaw), np.sin(ang + yaw)], 1)
        o = np.array([x, y])
        out = np.full(n, rng)
        circles = self.circles + list(self.objects.values()) + [(np.asarray(p, float)[:2], 0.3) for p in others]
        for c, r in circles:
            oc = o - c
            b = d @ oc
            disc = b * b - (oc @ oc - r * r)
            ok = disc >= 0
            t = -b - np.sqrt(np.where(ok, disc, 0.0))
            hit = ok & (t > 0)
            out = np.where(hit, np.minimum(out, t), out)
        for lo, hi in self.boxes:
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - o) / d, (hi - o) / d
            tmin = np.nanmax(np.minimum(t1, t2), 1)
            tmax = np.nanmin(np.maximum(t1, t2), 1)
            hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
            out = np.where(hit, np.minimum(out, tmin), out)
        return out, ang


def _pursuit(path, pose, ranges, angles):
    x, y, yaw = pose
    d = np.linalg.norm(path - np.array([x, y]), axis=1)
    i = int(d.argmin())
    j = min(i + 8, len(path) - 1)
    rel = path[j] - np.array([x, y])
    lx = math.cos(yaw) * rel[0] + math.sin(yaw) * rel[1]
    ly = -math.sin(yaw) * rel[0] + math.cos(yaw) * rel[1]
    curv = 2 * ly / max(lx * lx + ly * ly, 1e-3)
    cone = np.abs(angles) < 0.5
    ahead = float(ranges[cone].min()) if cone.any() else LIDAR_RANGE
    v = max(0.8 * float(np.clip((ahead - 0.35) / 1.0, 0, 1)), 0.1)
    if lx < 0:
        v = 0.1
    return v, float(np.clip(curv * v, -W_MAX, W_MAX))


class Nav:
    def __init__(self, unit: str, pose, deck: DeckMap, speed: float = 0.8):
        self.unit, self.deck, self.speed = unit, deck, speed
        self.pose = np.array([*pose], float)
        self.v = self.w = 0.0
        self.path: Optional[np.ndarray] = None
        self.state = "IDLE"
        self.lidar_min = LIDAR_RANGE
        self.ctl = None
        self.drive = getattr(_mod("integration.fleet_nav"), "drive", None)
        ctl = _mod("patrol.controller")
        self.Controller = getattr(ctl, "PatrolController", None)
        self.reported: set = set()
        self.recover_until = -1.0

    @property
    def xy(self) -> np.ndarray:
        return self.pose[:2]

    def go(self, waypoints) -> None:
        pts = [self.xy.copy()] + [np.asarray(p, float) for p in waypoints]
        seg = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(int(np.linalg.norm(b - a) / 0.1), 1)
            seg += [a + (b - a) * k / n for k in range(n)]
        seg.append(pts[-1])
        self.path = np.array(seg)
        self.state = "DRIVE"
        self.ctl = self.Controller("pure_pursuit", self.path) if self.Controller else None

    def push(self, t: float, dxy=(0.0, 0.35), dyaw: float = 0.5) -> None:
        self.pose[:2] += np.asarray(dxy, float)
        self.pose[2] = wrap(self.pose[2] + dyaw)
        self.state, self.recover_until = "PUSH_RECOVER", t + 1.5

    def step(self, t: float, dt: float, others=()) -> Optional[dict]:
        ranges, angles = self.deck.scan(self.pose, others)
        self.lidar_min = round(float(ranges.min()), 3)
        report = None
        if self.path is None or self.state in ("IDLE", "ARRIVED", "STOP_OBJECT"):
            self.v = self.w = 0.0
            return None
        goal = self.path[-1]
        if np.linalg.norm(goal - self.xy) < 0.15:
            self.path, self.state, self.v, self.w = None, "ARRIVED", 0.0, 0.0
            return None
        for oid, (c, r) in self.deck.objects.items():
            rel = c - self.xy
            dist = float(np.linalg.norm(rel)) - r
            bearing = wrap(math.atan2(rel[1], rel[0]) - self.pose[2])
            if dist < 0.9 and abs(bearing) < 0.6 and oid not in self.reported:
                self.reported.add(oid)
                self.state, self.v, self.w = "STOP_OBJECT", 0.0, 0.0
                return {"object": oid, "xy": [round(float(c[0]), 2), round(float(c[1]), 2)],
                        "dist_m": round(dist, 2)}
        if self.drive is not None:
            vx, wz = self.drive(self.unit, (ranges, angles), tuple(self.pose), self.v, self.w, self.path)[:2]
        elif self.ctl is not None:
            vx, wz, _ = self.ctl.step(ranges, angles, tuple(self.pose), self.v, self.w)
        else:
            vx, wz = _pursuit(self.path, self.pose, ranges, angles)
        err = wrap(math.atan2(*(self.path[min(int(np.linalg.norm(self.path - self.xy, axis=1).argmin()) + 5,
                                               len(self.path) - 1)] - self.xy)[::-1]) - self.pose[2])
        if abs(err) > 1.2:
            vx, wz = 0.0, float(np.clip(2.0 * err, -W_MAX, W_MAX))
        vx = float(np.clip(vx, V_MIN, self.speed))
        wz = float(np.clip(wz, -W_MAX, W_MAX))
        self.v, self.w = vx, wz
        self.pose[2] = wrap(self.pose[2] + wz * dt)
        self.pose[:2] += vx * dt * np.array([math.cos(self.pose[2]), math.sin(self.pose[2])])
        if self.state == "PUSH_RECOVER" and t >= self.recover_until:
            self.state = "DRIVE"
        return report

    def clear_stop(self) -> None:
        if self.state == "STOP_OBJECT":
            self.state = "DRIVE"

    def status(self) -> dict:
        return {"vx": round(self.v, 3), "wz": round(self.w, 3), "lidar_min": self.lidar_min, "state": self.state}


class Scenario:
    def __init__(self, loose_at: float = 8.0, fault: bool = True, drop_at: Optional[float] = None,
                 push_at: Optional[float] = None):
        self.loose_at, self.fault, self.drop_at, self.push_at = loose_at, fault, drop_at, push_at
        self.level = {"F1": 0.0, "F2": 0.0}
        self.target = {"F1": 0.0, "F2": 0.0}
        self.fixed: set = set()
        self.deck = DeckMap(LAYOUT["boxes"], LAYOUT["posts"])
        self.detectors = [FixedDetector(d, f, (float(p[0]), float(p[1]), float(ZONE[2]) + 1.2))
                          for d, (f, p) in LAYOUT["detectors"].items()]
        self.stuck_at: Optional[float] = None
        self.events: list[dict] = []

    def gas_at(self, p) -> float:
        p = np.asarray(p, float)[:2]
        g = 0.02
        for f, c in LAYOUT["flanges"].items():
            g += 0.4 * self.level[f] * math.exp(-float(np.linalg.norm(p - c)) / 1.5)
        return g

    def retorque(self, flange: str) -> None:
        self.fixed.add(flange)
        self.target[flange] = 0.0

    def step(self, t: float, dt: float, units: dict) -> list[dict]:
        ev = []
        if t >= self.loose_at and "F2" not in self.fixed:
            self.target["F2"] = min(0.3, 0.3 * (t - self.loose_at) / 5.0)
            if not any(e["name"] == "bolt_loose" for e in self.events):
                ev.append({"name": "bolt_loose", "flange": "F2", "bolt": 3})
        for f in self.level:
            tau = 2.0 if f in self.fixed else 0.6
            self.level[f] += (1 - math.exp(-dt / tau)) * (self.target[f] - self.level[f])
        fx = units[FIXER]
        g1 = next(d for d in self.detectors if d.id == "G-F1")
        if (self.fault and self.stuck_at is None and "F2" in self.fixed and fx.state == "TO_DOCK"
                and g1.dist(fx.nav.xy) < 1.8):
            self.stuck_at = g1.stuck_at = 14.0
            ev.append({"name": "sensor_stuck", "sensor": "G-F1", "lel": 14.0})
        if self.drop_at is not None and t >= self.drop_at and "box" not in self.deck.objects:
            ahead = fx.nav.xy + np.array([-1.4, 0.0])
            self.deck.drop("box", (ahead[0], LAYOUT["walk_y"] + ZONE[1]))
            ev.append({"name": "object_dropped", "object": "box",
                       "xy": [round(float(v), 2) for v in self.deck.objects["box"][0]]})
        if self.push_at is not None and t >= self.push_at and not any(e["name"] == "push" for e in self.events):
            fx.nav.push(t)
            ev.append({"name": "push", "robot": FIXER})
        for d in self.detectors:
            d.read(self.gas_at)
        self.events += ev
        return ev


class Scrubber:
    def __init__(self, deck: DeckMap):
        self.nav = Nav(SCRUBBER, (*LAYOUT["rust"], math.pi / 2), deck, speed=0.6)
        self.state, self.rust, self.force = "DETECT", 0.62, 0.0
        self.until = 1.0
        self.paused = False
        self.why = ""
        self.rng = np.random.default_rng(1)

    def hold(self, t: float, why: str) -> None:
        if self.state in ("DONE",) or self.paused:
            return
        self.paused, self.resume_state, self.why = True, self.state, why
        self.state = "HOLD"
        self.nav.go([LAYOUT["backoff"]])

    def resume(self, t: float) -> None:
        if not self.paused:
            return
        self.paused = False
        self.state = "RESUME"
        self.nav.go([LAYOUT["rust"]])

    def retreat(self, t: float) -> None:
        self.hold(t, "retreat")

    def step(self, t: float, dt: float, gas: float, others=()) -> None:
        self.nav.step(t, dt, others)
        s = self.state
        self.force = 0.0
        if s == "HOLD":
            return
        if s == "RESUME":
            if self.nav.state == "ARRIVED":
                self.nav.pose[2] = math.pi / 2
                self.state = self.resume_state if self.resume_state in ("SPRAY", "SOAK", "SCRUB") else "SCRUB"
            return
        if s == "SCRUB":
            self.force = round(24.0 + 3.0 * float(self.rng.standard_normal()), 2)
            self.rust = max(0.0, self.rust - 0.04 * dt)
            if self.rust <= 0.03:
                self.state, self.until = "VERIFY", t + 1.5
            return
        if t < self.until:
            return
        nxt = {"DETECT": ("SPRAY", 1.5), "SPRAY": ("SOAK", 2.5), "SOAK": ("SCRUB", 0.0), "VERIFY": ("PAINT", 4.0),
               "PAINT": ("VERIFY_PAINT", 1.0), "VERIFY_PAINT": ("DONE", 0.0)}.get(s)
        if nxt:
            self.state, self.until = nxt[0], t + nxt[1]

    def evidence(self, gas: float) -> dict:
        return {"force": self.force, "force_limit": 40.0, "rust_fraction": round(self.rust, 4), "gas": gas,
                "job": "rust", "soak_left": max(0.0, self.until - 0.0) if self.state == "SOAK" else None}


class Fixer:
    def __init__(self, deck: DeckMap, scenario: Scenario):
        self.nav = Nav(FIXER, (*LAYOUT["dock"], math.pi), deck, speed=0.9)
        self.scn = scenario
        self.state, self.flange, self.fix_ok = "DOCKED", None, False
        self.until = 0.0
        self.bolt = None

    def _standoff(self, flange: str) -> np.ndarray:
        return LAYOUT["flanges"][flange] + np.array([0.0, -0.85])

    def dispatch(self, t: float, flange: str) -> None:
        if self.state not in ("DOCKED", "TO_DOCK"):
            return
        self.flange, self.state = flange, "TRANSIT"
        y = LAYOUT["walk_y"] + ZONE[1]
        so = self._standoff(flange)
        self.nav.go([np.array([self.nav.xy[0] - 0.8, y]), np.array([so[0] + 3.0, y])])

    def fix(self, t: float, flange: str) -> None:
        if self.flange is None:
            self.dispatch(t, flange)
        self.fix_ok = True

    def retreat(self, t: float) -> None:
        self.state = "GAS_RETREAT"
        self.nav.go([LAYOUT["dock"]])

    def reroute(self, xy) -> None:
        if self.nav.path is None:
            return
        goal = self.nav.path[-1]
        c = np.asarray(xy, float)
        self.nav.go([np.array([c[0] + 0.7, c[1] - 0.8]), np.array([c[0] - 0.7, c[1] - 0.8]), goal])

    def step(self, t: float, dt: float, others=()) -> Optional[dict]:
        rep = self.nav.step(t, dt, others)
        s = self.state
        if s == "TRANSIT" and self.nav.state == "ARRIVED":
            so = self._standoff(self.flange)
            k = np.arange(1, 7)
            cast = [so + np.array([3.0 * (1 - i / 6), 0.35 * (-1) ** i]) for i in k]
            self.state = "LEAK_APPROACH"
            self.nav.go(cast + [so])
        elif s == "LEAK_APPROACH" and self.nav.state == "ARRIVED":
            self.nav.pose[2] = math.pi / 2
            self.state, self.bolt = "LEAK_INSPECT", 3
        elif s == "LEAK_INSPECT" and self.fix_ok:
            self.state, self.until = "RETORQUE", t + 3.0
        elif s == "RETORQUE" and t >= self.until:
            self.scn.retorque(self.flange)
            self.state, self.until = "LEAK_VERIFY", t + 2.0
        elif s == "LEAK_VERIFY" and t >= self.until:
            d = next(d for d in self.scn.detectors if d.flange == self.flange)
            if 100 * self.scn.gas_at(self.nav.xy) < CLEAR_LEL and led(d.last) == "green":
                self.state = "TO_DOCK"
                y = LAYOUT["walk_y"] + ZONE[1]
                self.nav.go([np.array([self.nav.xy[0] + 0.8, y]), np.array([LAYOUT["dock"][0] - 0.8, y]),
                             LAYOUT["dock"]])
        elif s in ("TO_DOCK", "GAS_RETREAT") and self.nav.state == "ARRIVED":
            self.state, self.flange, self.fix_ok, self.bolt = "DOCKED", None, False, None
        return rep

    def evidence(self, gas: float) -> dict:
        return {"gas": gas, "gas_stop": 0.20, "flange": self.flange, "bolt": self.bolt, "job": "gas"}


CAPTIONS = {
    "HOLD": ("Holding — hot work stopped", " spark rule, rust job paused"),
    "RESUME": ("Resuming", " the rust job — zone clear"),
    "STOP_OBJECT": ("Stopped", " — object on the walkway, reported to Argon"),
    "PUSH_RECOVER": ("Recovering", " from a push"),
    "DOCKED": ("Docked", " — waiting for a job"),
}


def _caption(state: str, nav_state: str, ev: dict) -> dict:
    key = nav_state if nav_state in ("STOP_OBJECT", "PUSH_RECOVER") else state
    if key in CAPTIONS:
        lead, rest = CAPTIONS[key]
        return {"state": state, "lead": lead, "rest": rest, "text": lead + rest,
                "tone": "warn" if key != "DOCKED" else "ok"}
    return caption(state, ev)


class FleetDemo:
    def __init__(self, control: str = "edge", out: Optional[Path] = None, cut_link_at: Optional[float] = None,
                 replay_from: Optional[str] = None, fault: bool = True, drop_at: Optional[float] = None,
                 push_at: Optional[float] = None, log_hz: float = 5.0, max_t: float = 120.0):
        out = Path(out) if out else None
        self.timeline = Timeline(out / "timeline.jsonl" if out else None)
        self.hud_path = out / "hud.jsonl" if out else None
        if self.hud_path:
            self.hud_path.write_text("")
        self.out = out
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
        self.cut_link_at, self.log_every, self.max_t = cut_link_at, max(1, round(1.0 / (log_hz * DT))), max_t
        self.scn = Scenario(fault=fault, drop_at=drop_at, push_at=push_at)
        self.units = {SCRUBBER: Scrubber(self.scn.deck), FIXER: Fixer(self.scn.deck, self.scn)}
        self.ids = ids_for(self.scn.detectors)
        self.fleet = _fleet_adapter()
        self.t = 0.0
        self.k = 0
        self.seen = 0
        self.findings: list[dict] = []

    def lel(self, rid: str) -> float:
        return round(100.0 * self.scn.gas_at(self.units[rid].nav.xy), 2)

    def robots(self) -> dict:
        out = {}
        for rid, u in self.units.items():
            x, y, yaw = u.nav.pose
            out[rid] = {"state": u.state, "pose": [round(float(x), 3), round(float(y), 3), float(ZONE[2]),
                                                   round(float(yaw), 3)],
                        "lel": self.lel(rid), "nav": u.nav.status()}
            if rid == SCRUBBER:
                out[rid].update(force_n=u.force, rust=round(u.rust, 4), paused=u.paused)
            else:
                out[rid].update(flange=u.flange, bolt=u.bolt, force_n=8.0 if u.state == "RETORQUE" else 0.0)
        return out

    def apply(self, cmds) -> None:
        u1, u2 = self.units[SCRUBBER], self.units[FIXER]
        for c in cmds:
            if c.kind == HOLD:
                u1.hold(self.t, c.reason)
                self._fleet("hold", SCRUBBER, why=c.reason)
            elif c.kind == RESUME_RUST:
                u1.resume(self.t)
                self._fleet("resume", SCRUBBER, why=c.reason)
            elif c.kind == GO_CHECK:
                u2.dispatch(self.t, c.target)
                self._fleet("assign", FIXER, kind="flange_leak_check", target=c.target, why=c.reason)
            elif c.kind == FIX_LEAK:
                u2.fix(self.t, c.target)
            elif c.kind == RETREAT:
                self.units.get(c.robot, u1).retreat(self.t)
            elif c.kind == LOG_SENSOR_FAULT:
                self._fleet("ticket", None, sensor=c.target, why=c.reason)

    def _fleet(self, op: str, rid, **kw) -> None:
        if self.fleet is None:
            return
        try:
            a, ids = self.fleet
            if op == "hold":
                r = a.hold(ids[rid], kw["why"])
            elif op == "resume":
                r = a.resume(ids[rid], kw["why"])
            elif op == "assign":
                f = LAYOUT["flanges"][kw["target"]]
                r = a.assign(kw["kind"], kw["target"], (float(f[0]), float(f[1]), float(ZONE[2])), why=kw["why"],
                             robot=ids[rid])
            else:
                f = next(d.pos for d in self.scn.detectors if d.id == kw["sensor"])
                r = a.ticket(kw["sensor"], kw["why"], f)
            a.c.tick(DT)
            self.timeline.log("fleet", self.t, op=op, robot=rid, ok=bool(r.get("ok")),
                              job=r.get("job_id"), refused=r.get("refused"))
        except Exception as e:
            self.timeline.log("fleet", self.t, op=op, robot=rid, ok=False, error=f"{type(e).__name__}: {e}")

    def step(self) -> None:
        t = self.t
        if self.cut_link_at is not None and t >= self.cut_link_at and not self.link.link_cut:
            self.link.cut(True)
            self.timeline.log("link", t, cut=True)
        for e in self.scn.step(t, DT, self.units):
            self.timeline.log("scenario", t, **e)
        u1, u2 = self.units[SCRUBBER], self.units[FIXER]
        u1.step(t, DT, self.lel(SCRUBBER) / 100.0, [u2.nav.xy])
        rep = u2.step(t, DT, [u1.nav.xy])
        if rep is not None:
            self.timeline.log("object_report", t, robot=FIXER, **rep)
            self.timeline.log("command", t, kind="reroute", robot=FIXER, target=rep["object"], source="edge",
                              reason=f"{rep['object']} on the walkway {rep['dist_m']:.1f} m ahead: logged, "
                                     f"detour around it", confidence=1.0, finding="OBJECT")
            u2.reroute(rep["xy"])
        sniff = {rid: (self.lel(rid), u.nav.xy) for rid, u in self.units.items()}
        fds = [fuse_fleet(d, sniff) for d in self.scn.detectors]
        robots = self.robots()
        cmds = self.link.tick(t, fds, robots, self.scn.detectors, self.ids, scrubber_pos=u1.nav.xy, wait=self.wait)
        self.apply(cmds)
        if self.k % self.log_every == 0 or cmds:
            self.log_tick(fds, robots)
        self.k += 1
        self.t = round(t + DT, 3)

    def done(self) -> bool:
        return (self.units[SCRUBBER].state == "DONE" and self.units[FIXER].state == "DOCKED"
                and "F2" in self.scn.fixed and (not self.scn.fault or bool(self.link.tickets))) or self.t >= self.max_t

    def log_tick(self, fds, robots) -> None:
        dets = [{"id": d.id, "flange": d.flange, "lel": d.last, "led": led(d.last, d.id in self.link.faulted),
                 "pos": [round(float(v), 3) for v in d.pos]} for d in self.scn.detectors]
        for rid, r in robots.items():
            ev = self.units[rid].evidence(r["lel"] / 100.0)
            r["caption"] = _caption(r["state"], r["nav"]["state"], ev)["text"]
        row = self.timeline.log("tick", self.t, robots=robots, detectors=dets, fusion=[f.to_dict() for f in fds],
                                argon=self.link.header())
        self.write_header(row)

    def write_header(self, row: dict) -> None:
        if self.hud_path is None:
            return
        rows = self.timeline.rows[self.seen:]
        self.seen = len(self.timeline.rows)
        events = []
        for r in rows:
            if r["type"] in ("command", "ticket", "vetoed", "finding", "object_report", "scenario", "link"):
                events.append({"t": r["t"], "from": "Argon" if r.get("source") == "argon" else
                               ("Edge" if r["type"] in ("command", "vetoed") else r["type"].capitalize()),
                               "to": (r.get("kind") or r.get("name") or r["type"]).upper(),
                               "reason": r.get("reason") or r.get("why") or r.get("sensor") or r.get("object") or "",
                               "robot": r.get("robot")})
        ents = []
        for rid, r in row["robots"].items():
            u = self.units[rid]
            ev = u.evidence(r["lel"] / 100.0)
            e = robot_entry(rid, NAMES[rid], r["state"], ev, r["pose"][:3])
            e["caption"] = _caption(r["state"], r["nav"]["state"], ev)
            e.update(yaw=r["pose"][3], lel=r["lel"], force_n=r.get("force_n"), rust=r.get("rust"), nav=r["nav"])
            ents.append(e)
        sensors = [{"id": d["id"], "at": f"flange {d['flange']}", "kind": "fixed", "lel": d["lel"] / 100.0,
                    "led": d["led"]} for d in row["detectors"]]
        sensors += [{"id": f"{rid}-sniffer", "at": NAMES[rid], "kind": "robot", "lel": r["lel"] / 100.0}
                    for rid, r in row["robots"].items()]
        lead = ents[0]
        mx = max(s["lel"] for s in sensors)
        hdr = {"t": row["t"], "state": lead["state"], "caption": lead["caption"],
               "hud": [e["caption"]["text"] for e in ents], "events": events, "robots": ents,
               "force_n": row["robots"][SCRUBBER].get("force_n"), "force_limit": 40.0,
               "rust_fraction": row["robots"][SCRUBBER].get("rust"),
               "gas": {"sensors": sensors, "max": round(mx, 4), "alarm": mx * 100 >= ALARM_LEL},
               "detectors": row["detectors"], "fusion": row["fusion"], "argon": row["argon"],
               "source": "EDGE" if self.link.link_cut or self.control == "edge" else "CLOUD",
               "fleet": {"schema": "fleet_state.v1", "t": row["t"],
                         "robots": [{"id": e["id"], "name": e["name"], "state": e["state"], "job_state": e["state"],
                                     "deck": "top_deck", "pose": {"x": e["world_pos"][0], "y": e["world_pos"][1],
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
        self.log_tick([fuse_fleet(d, {rid: (self.lel(rid), u.nav.xy) for rid, u in self.units.items()})
                       for d in self.scn.detectors], self.robots())
        s = {"control": self.control, "t_end": self.t, "beats": beats(self.timeline.rows),
             "argon": self.link.summary(), "nav": "fleet_nav" if self.units[FIXER].nav.drive else
             ("patrol.PatrolController" if self.units[FIXER].nav.Controller else "pure_pursuit")}
        if self.out:
            (self.out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
        return s


def _fleet_adapter():
    fa = _mod("policy.control.fleet_adapter")
    if fa is None:
        return None
    try:
        from fleet.contract import Robot, RobotKind, World
        from fleet.coordinator import Coordinator
        from fleet.rigplanner import RigPlanner

        world = World.from_rig_meta()
        world.state["flanges"] = {}
        z = float(ZONE[2])
        u1 = Robot("U1", RobotKind.QUADRUPED, {"arm", "spray", "brush", "camera", "gas_sensor"},
                   (*map(float, LAYOUT["rust"]), z), battery=0.9)
        u2 = Robot("U2", RobotKind.QUADRUPED, {"arm", "gripper", "gas_sensor", "camera"},
                   (*map(float, LAYOUT["dock"]), z), battery=0.95)
        dock = (*map(float, LAYOUT["dock"]), z)
        c = Coordinator([u1, u2], world, {}, RigPlanner(), docks={"U1": dock, "U2": dock})
        a = fa.FleetAdapter(c)
        a.assign("rust_repair", "main_pipe", (*map(float, LAYOUT["rust"]), z), why="rust on the main pipe")
        c.tick(1.0)
        return a, {SCRUBBER: "U1", FIXER: "U2"}
    except Exception:
        return None


def _first(rows, pred) -> Optional[float]:
    return next((r["t"] for r in rows if pred(r)), None)


def _after(rows, t0, pred) -> Optional[float]:
    if t0 is None:
        return None
    return next((r["t"] for r in rows if r["t"] >= t0 and pred(r)), None)


def _tick(rid, state):
    return lambda r: r["type"] == "tick" and r["robots"][rid]["state"] == state


def _det(sid, colour):
    return lambda r: r["type"] == "tick" and any(d["id"] == sid and d["led"] == colour for d in r["detectors"])


def _cmd(kind, robot=None):
    return lambda r: r["type"] == "command" and r["kind"] == kind and (robot is None or r["robot"] == robot)


def _finding(kind, robot=None):
    return lambda r: r["type"] == "finding" and r["kind"] == kind and (robot is None or r["robot"] == robot)


def beats(rows: list[dict]) -> dict:
    b = {}
    b["1_scrubbing"] = _first(rows, _tick(SCRUBBER, "SCRUB"))
    b["2_gf2_amber"] = _first(rows, _det("G-F2", "amber"))
    b["2_gf2_red"] = _after(rows, b["2_gf2_amber"], _det("G-F2", "red"))
    b["3_hold_unit1"] = _first(rows, _cmd(HOLD, SCRUBBER))
    b["3_unverified"] = _first(rows, _finding("UNVERIFIED", SCRUBBER))
    b["3_dispatch_unit2"] = _first(rows, _cmd(GO_CHECK, FIXER))
    b["4_unit2_walks"] = _after(rows, b["3_dispatch_unit2"], _tick(FIXER, "TRANSIT"))
    b["4_leak_confirmed"] = _first(rows, _finding("LEAK_CONFIRMED", FIXER))
    b["4_fix_leak"] = _first(rows, _cmd(FIX_LEAK, FIXER))
    b["4_retorque"] = _first(rows, _tick(FIXER, "RETORQUE"))
    b["4_gf2_green"] = _after(rows, b["4_retorque"], _det("G-F2", "green"))
    b["5_resume_rust"] = _first(rows, _cmd(RESUME_RUST, SCRUBBER))
    b["5_unit1_scrubs_again"] = _after(rows, b["5_resume_rust"], _tick(SCRUBBER, "SCRUB"))
    b["5_unit1_paints"] = _after(rows, b["5_resume_rust"], _tick(SCRUBBER, "PAINT"))
    b["5_unit1_done"] = _after(rows, b["5_resume_rust"], _tick(SCRUBBER, "DONE"))
    b["5_unit2_to_dock"] = _after(rows, b["4_gf2_green"], _tick(FIXER, "TO_DOCK"))
    b["5_unit2_docked"] = _after(rows, b["5_unit2_to_dock"], _tick(FIXER, "DOCKED"))
    b["6_gf1_fault"] = _first(rows, _finding(PIPE_SENSOR_FAULT))
    b["6_ticket"] = _first(rows, lambda r: r["type"] == "ticket")
    b["6_object_report"] = _first(rows, lambda r: r["type"] == "object_report")
    b["6_link_cut"] = _first(rows, lambda r: r["type"] == "link" and r.get("cut") and r["t"] >= 0)
    return b


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="CPU scenario, no Isaac")
    ap.add_argument("--control", choices=("live", "edge", "replay"), default="edge")
    ap.add_argument("--cut-link-at", type=float, default=None)
    ap.add_argument("--replay-from", default="runs/fleet_demo/live_timeline.jsonl")
    ap.add_argument("--out", default="runs/fleet_demo")
    ap.add_argument("--no-fault", action="store_true")
    ap.add_argument("--drop-object-at", type=float, default=None)
    ap.add_argument("--push-at", type=float, default=None)
    ap.add_argument("--log-hz", type=float, default=5.0)
    a = ap.parse_args(argv)
    out = Path(a.out) / a.control
    out.mkdir(parents=True, exist_ok=True)
    demo = FleetDemo(a.control, out, a.cut_link_at, a.replay_from, not a.no_fault, a.drop_object_at, a.push_at,
                     a.log_hz)
    s = demo.run()
    if a.control == "live":
        (Path(a.out) / "live_timeline.jsonl").write_text((out / "timeline.jsonl").read_text())
    print(json.dumps({k: s[k] for k in ("control", "t_end", "beats", "nav")}, indent=1))
    ar = s["argon"]
    print(f"argon {ar['model']}: {ar['calls']} calls, latency {ar['latency_ms']}, cost ${ar['cost_usd']}")
    print("\n".join(ar["commands"]))


if __name__ == "__main__":
    main()
