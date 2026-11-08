from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from integration.fleet_paths import DECK_Z, LANE_Y, DeckMap, Leg, leg as get_leg, polyline
from patrol.controller import PatrolController
from patrol.detect import DetectCfg, Tracker, detect_frame

REPO = Path(__file__).resolve().parent.parent
NAV_V3A = "patrol/models/nav_v3a/policy.onnx"
TRUNK_Z = 0.32

ALIGN, FOLLOW, REPORT, BLOCKED, FACE, ARRIVED = "ALIGN", "FOLLOW", "REPORT", "BLOCKED", "FACE", "ARRIVED"

STAND_Q = {"FL_hip_joint": 0.1, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
           "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
           "RL_hip_joint": 0.1, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
           "RR_hip_joint": -0.1, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5}
TROT_PAIRS = {"FL": 0.0, "RR": 0.0, "FR": math.pi, "RL": math.pi}


@dataclass(frozen=True)
class NavProfile:
    name: str
    policy: str = NAV_V3A
    v_max: float = 0.8
    v_min: float = -0.3
    w_max: float = 1.2
    lidar_z: float = 0.30
    lidar_beams: int = 360
    lidar_range: float = 4.0
    stop_m: float = 1.0
    lane_m: float = 0.25
    goal_tol: float = 0.25
    align_tol: float = 0.1
    turn_rate: float = 1.0
    blocked_s: float = 8.0
    push_m: float = 0.25
    map_heights: tuple = (0.7, 0.9)
    goal_xy_tol: float = 0.04
    turn_clear_m: float = 0.2
    map_tol_m: float = 0.2

    def resolved_policy(self) -> str:
        p = Path(self.policy)
        p = p if p.is_absolute() else REPO / p
        return str(p) if p.exists() else "pure_pursuit"

    def angles(self) -> np.ndarray:
        return np.deg2rad(np.arange(self.lidar_beams) * 360.0 / self.lidar_beams - 180.0 + 180.0 / self.lidar_beams)


PROFILES = {"unit1": NavProfile("scrubber"), "unit2": NavProfile("fixer")}


@dataclass
class NavCmd:
    vx: float
    wz: float
    state: str
    event: Optional[dict] = None


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class FleetNav:
    def __init__(self, unit: str, profile: Optional[NavProfile] = None, on_event: Optional[Callable] = None,
                 policy=None, deck: Optional[DeckMap] = None):
        self.unit = unit
        self.deck = deck if deck is not None else DeckMap.rig_merged()
        self.profile = profile or PROFILES.get(unit, NavProfile(unit))
        self.policy = policy if policy is not None else self.profile.resolved_policy()
        self.on_event = on_event
        self.events: list[dict] = []
        self.cfg = DetectCfg(tube=self.profile.lane_m, ahead_max=self.profile.stop_m + 2.0,
                             lidar_range=self.profile.lidar_range, pick_max_width=0.0, confirm=2)
        self.ctl: Optional[PatrolController] = None
        self.leg: Optional[Leg] = None
        self.state = ARRIVED
        self.t = 0.0

    @property
    def mode(self) -> str:
        return self.ctl.mode if self.ctl is not None else ""

    def start(self, leg_or_path, goal_yaw: Optional[float] = None, name: str = "") -> None:
        lg = get_leg(leg_or_path) if isinstance(leg_or_path, str) else leg_or_path
        if not isinstance(lg, Leg):
            p = np.asarray(lg, float)[:, :2]
            h = math.atan2(p[-1, 1] - p[-2, 1], p[-1, 0] - p[-2, 0])
            lg = Leg(self.unit, name or "path", p, h if goal_yaw is None else goal_yaw)
        self.leg = lg
        self.ctl = PatrolController(self.policy, lg.path)
        self.tracker = Tracker(self.cfg)
        self.blocker = None
        self.best, self.stall, self.timer, self.off = -1, 0.0, 0.0, False
        self.state = ALIGN
        self._emit("leg_start", leg=lg.name, length_m=round(lg.length, 2), policy=self.mode)

    def resume(self) -> None:
        if self.state in (REPORT, BLOCKED):
            self.blocker, self.stall, self.timer = None, 0.0, 0.0
            self.tracker.reset()
            self.state = FOLLOW
            self._emit("resumed")

    def with_map(self, ranges, angles, pose) -> np.ndarray:
        r = np.asarray(ranges, float)
        for h in self.profile.map_heights:
            r = np.fmin(r, self.deck.raycast(pose[0], pose[1], pose[2], angles, h, self.profile.lidar_range))
        return r

    def unmapped(self, ranges, angles, pose) -> np.ndarray:
        r = np.asarray(ranges, float)
        known = self.deck.raycast(pose[0], pose[1], pose[2], angles, None, self.profile.lidar_range + 1.0)
        return np.where(r < known - self.profile.map_tol_m, r, np.inf)

    def _emit(self, kind: str, **kw) -> dict:
        ev = {"type": kind, "unit": self.unit, "t": round(self.t, 2), "state": self.state, **kw}
        self.events.append(ev)
        if self.on_event is not None:
            self.on_event(ev)
        return ev

    def _turn(self, err: float) -> float:
        p = self.profile
        return float(np.clip(-1.5 * err, -p.turn_rate, p.turn_rate))

    def step(self, t: float, ranges, angles, pose, v: float, w: float) -> NavCmd:
        self.t = float(t)
        p = self.profile
        if self.ctl is None or self.state == ARRIVED:
            return NavCmd(0.0, 0.0, ARRIVED)
        pose = tuple(float(q) for q in pose)
        raw = np.asarray(ranges, float)
        ranges = self.with_map(raw, angles, pose)
        vx, wz, _ = self.ctl.step(ranges, angles, pose, v, w)
        L = (self.ctl.M - 1) * self.ctl.ds
        s0 = self.ctl.dist_from_start
        dets = self.tracker.update(detect_frame(self.unmapped(raw, angles, pose), angles, pose, self.ctl.ref,
                                                self.ctl.progress_idx, self.cfg))
        dets = [d for d in dets if s0 + d.ahead < L - 0.3]
        if self.state == ALIGN:
            err = self.ctl.heading_error(pose[2])
            if abs(err) > 0.35 and self.deck.clearance(*pose) < p.turn_clear_m and self.timer < 3.0:
                self.timer += 0.1
                return NavCmd(-0.2, 0.0, ALIGN)
            if abs(err) > 0.35:
                return NavCmd(0.0, self._turn(err), ALIGN)
            self.state = FOLLOW
            self.best = self.ctl.progress_idx
        if self.state == FOLLOW:
            near_goal = math.hypot(pose[0] - self.leg.goal[0], pose[1] - self.leg.goal[1]) < p.goal_tol
            if self.ctl.dist_to_goal < p.goal_tol or near_goal:
                self.state = FACE
            else:
                ahead = [d for d in dets if d.ahead < p.stop_m + 0.5]
                if ahead:
                    d = min(ahead, key=lambda q: q.ahead)
                    self.blocker = d.xy.copy()
                    self.state = REPORT
                    ev = self._emit("object_on_walkway", xy=[round(float(c), 3) for c in d.xy],
                                    width_m=round(float(d.width), 3), ahead_m=round(float(d.ahead), 2),
                                    s_m=round(float(s0 + d.ahead), 2), leg=self.leg.name)
                    return NavCmd(0.0, 0.0, REPORT, ev)
                lat = float(((np.array(pose[:2]) - self.ctl.P[self.ctl.progress_idx]) * self.ctl.nrm[self.ctl.progress_idx]).sum())
                ev = None
                if not self.off and abs(lat) > p.push_m:
                    self.off = True
                    ev = self._emit("off_path", lateral_m=round(lat, 3))
                elif self.off and abs(lat) < 0.1:
                    self.off = False
                    ev = self._emit("recovered", lateral_m=round(lat, 3))
                if self.ctl.progress_idx > self.best:
                    self.best, self.stall = self.ctl.progress_idx, 0.0
                else:
                    self.stall += 0.1
                if self.stall >= p.blocked_s:
                    self.state = BLOCKED
                    ev = self._emit("blocked", s_m=round(s0, 2), leg=self.leg.name)
                    return NavCmd(0.0, 0.0, BLOCKED, ev)
                return NavCmd(float(np.clip(vx, p.v_min, p.v_max)), float(np.clip(wz, -p.w_max, p.w_max)), FOLLOW, ev)
        if self.state == REPORT:
            here = any(np.linalg.norm(d.xy - self.blocker) < 0.3 for d in dets)
            if not here:
                self.state = FOLLOW
                self.blocker = None
                self.best, self.stall = self.ctl.progress_idx, 0.0
                return NavCmd(0.0, 0.0, FOLLOW, self._emit("cleared"))
            return NavCmd(0.0, 0.0, REPORT)
        if self.state == BLOCKED:
            return NavCmd(0.0, 0.0, BLOCKED)
        if self.state == FACE:
            gx, gy = self.leg.goal
            d = math.hypot(gx - pose[0], gy - pose[1])
            if d > p.goal_xy_tol:
                b = _wrap(math.atan2(gy - pose[1], gx - pose[0]) - pose[2])
                back = abs(b) > math.pi / 2
                eb = _wrap(b - math.pi) if back else b
                if abs(eb) > 0.3:
                    return NavCmd(0.0, self._turn(-eb), FACE)
                speed = min(0.25, 0.8 * d) * (-1.0 if back else 1.0)
                return NavCmd(max(speed, p.v_min), float(np.clip(1.5 * eb, -0.5, 0.5)), FACE)
            err = _wrap(pose[2] - self.leg.goal_yaw)
            if abs(err) > p.align_tol:
                return NavCmd(0.0, self._turn(err), FACE)
            self.state = ARRIVED
            return NavCmd(0.0, 0.0, ARRIVED, self._emit("arrived", leg=self.leg.name,
                                                         pose=[round(pose[0], 3), round(pose[1], 3), round(pose[2], 3)]))
        return NavCmd(0.0, 0.0, self.state)


_NAVS: dict[str, FleetNav] = {}


def nav(unit: str, **kw) -> FleetNav:
    if unit not in _NAVS or kw:
        _NAVS[unit] = FleetNav(unit, **kw)
    return _NAVS[unit]


def drive(unit: str, scan, pose, v: float, w: float, path, t: Optional[float] = None) -> tuple[float, float]:
    n = nav(unit)
    lg = get_leg(path) if isinstance(path, str) else path
    key = lg.name if isinstance(lg, Leg) else np.asarray(lg, float).tobytes()
    if getattr(n, "_key", None) != key:
        n._key = key
        if not isinstance(lg, Leg):
            p = np.asarray(lg, float)[:, :2]
            if not _path_clear(n.deck, p):
                p = plan(p[0], p[-1], n.deck)
            lg = p
        n.start(lg)
    ranges, angles = scan
    cmd = n.step(n.t + 0.1 if t is None else t, ranges, angles, pose, v, w)
    return cmd.vx, cmd.wz


@dataclass
class BaseSim:
    x: float
    y: float
    yaw: float
    lag: float = 0.5
    v: float = 0.0
    w: float = 0.0
    push_v: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def push(self, vx: float, vy: float) -> None:
        self.push_v = np.array([vx, vy], float)

    def step(self, vx: float, wz: float, dt: float = 0.1) -> tuple:
        self.v += self.lag * (vx - self.v)
        self.w += self.lag * (wz - self.w)
        self.yaw = _wrap(self.yaw + self.w * dt)
        self.x += math.cos(self.yaw) * self.v * dt + self.push_v[0] * dt
        self.y += math.sin(self.yaw) * self.v * dt + self.push_v[1] * dt
        self.push_v *= 0.8
        return self.pose

    @property
    def pose(self) -> tuple:
        return (self.x, self.y, self.yaw)


@dataclass
class Rollout:
    unit: str
    leg: str
    t: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    v: np.ndarray
    w: np.ndarray
    states: list
    events: list
    min_clearance: float
    collided: Optional[str]
    gait: np.ndarray = None

    def __post_init__(self):
        dg = (np.abs(self.v) + 0.2 * np.abs(self.w)) * np.gradient(self.t) if len(self.t) > 1 else np.zeros(1)
        self.gait = np.cumsum(dg)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def arrived(self) -> bool:
        return any(e["type"] == "arrived" for e in self.events)

    def pose_at(self, t: float) -> tuple:
        x = float(np.interp(t, self.t, self.xy[:, 0]))
        y = float(np.interp(t, self.t, self.xy[:, 1]))
        yaw = float(np.interp(t, self.t, np.unwrap(self.yaw)))
        return x, y, DECK_Z + TRUNK_Z, _wrap(yaw)

    def speed_at(self, t: float) -> tuple:
        return float(np.interp(t, self.t, self.v)), float(np.interp(t, self.t, self.w))

    def gait_q(self, t: float) -> dict:
        v, w = self.speed_at(t)
        return trot_q(float(np.interp(t, self.t, self.gait)), abs(v) + 0.25 * abs(w))

    def samples(self) -> list[dict]:
        return [{"t": round(float(a), 2), "x": round(float(p[0]), 3), "y": round(float(p[1]), 3),
                 "yaw": round(float(h), 4), "v": round(float(v), 3), "w": round(float(w), 3), "state": s}
                for a, p, h, v, w, s in zip(self.t, self.xy, self.yaw, self.v, self.w, self.states)]


def trot_q(gait_m: float, speed: float, stride_m: float = 0.36, stand=STAND_Q) -> dict:
    a = float(np.clip(speed / 0.6, 0.0, 1.0))
    phi = 2 * math.pi * gait_m / stride_m
    q = dict(stand)
    for legname, off in TROT_PAIRS.items():
        s = math.sin(phi + off)
        q[f"{legname}_thigh_joint"] += a * 0.25 * s
        q[f"{legname}_calf_joint"] -= a * 0.45 * max(0.0, math.cos(phi + off))
    return q


def run_leg(unit: str, leg, deck: Optional[DeckMap] = None, start=None, t0: float = 0.0, t_max: float = 90.0,
            dt: float = 0.1, hooks=(), profile: Optional[NavProfile] = None, policy=None, lag: float = 0.5,
            on_event: Optional[Callable] = None, resume_after: Optional[float] = None) -> Rollout:
    lg = get_leg(leg) if isinstance(leg, str) else leg
    deck = deck or DeckMap.rig_merged()
    n = FleetNav(unit, profile, on_event, policy)
    p = n.profile
    angles = p.angles()
    sx, sy, syaw = start if start is not None else (*lg.start, math.atan2(*(lg.path[1] - lg.path[0])[::-1]))
    base = BaseSim(sx, sy, syaw, lag)
    n.t = t0
    n.start(lg)
    hooks = sorted(hooks, key=lambda h: h[0])
    T, XY, YAW, V, W, S = [], [], [], [], [], []
    clear, hit, t, held = math.inf, None, t0, 0.0
    while t <= t0 + t_max:
        while hooks and hooks[0][0] <= t - t0:
            hooks.pop(0)[1](base, deck, n)
        ranges = deck.raycast(base.x, base.y, base.yaw, angles, p.lidar_z, p.lidar_range)
        cmd = n.step(t, ranges, angles, base.pose, base.v, base.w)
        T.append(t), XY.append((base.x, base.y)), YAW.append(base.yaw), V.append(base.v), W.append(base.w)
        S.append(cmd.state)
        if cmd.state == ARRIVED:
            break
        if cmd.state in (REPORT, BLOCKED) and resume_after is not None:
            held += dt
            if held >= resume_after:
                n.resume()
                held = 0.0
        base.step(cmd.vx, cmd.wz, dt)
        c = deck.clearance(base.x, base.y, base.yaw)
        clear = min(clear, c)
        if c < 0 and hit is None:
            hit = deck.name_at(base.x, base.y, base.yaw) or "obstacle"
            n._emit("collision", what=hit, xy=[round(base.x, 3), round(base.y, 3)])
        t = round(t + dt, 6)
    return Rollout(unit, lg.name, np.array(T), np.array(XY), np.array(YAW), np.array(V), np.array(W), S,
                   list(n.events), clear, hit)


def run_route(unit: str, legs, deck: Optional[DeckMap] = None, t0: float = 0.0, **kw) -> list[Rollout]:
    out, start, t = [], None, t0
    for name in legs:
        r = run_leg(unit, name, deck, start=start, t0=t, **kw)
        out.append(r)
        start, t = (float(r.xy[-1, 0]), float(r.xy[-1, 1]), float(r.yaw[-1])), float(r.t[-1]) + 0.1
    return out


def with_profile(unit: str, **changes) -> NavProfile:
    PROFILES[unit] = replace(PROFILES.get(unit, NavProfile(unit)), **changes)
    return PROFILES[unit]


def plan(start_xy, goal_xy, deck: Optional[DeckMap] = None, lane_y: float = LANE_Y) -> np.ndarray:
    deck = deck or DeckMap.rig_merged()
    s, g = np.asarray(start_xy, float)[:2], np.asarray(goal_xy, float)[:2]
    if _line_clear(deck, s, g):
        return polyline(s, g)
    sign = 1.0 if g[0] >= s[0] else -1.0
    k = min(1.4, abs(g[0] - s[0]) / 3)
    pts = [s]
    if abs(s[1] - lane_y) > 0.05:
        pts.append((s[0] + sign * k, lane_y))
    if abs(g[1] - lane_y) > 0.05:
        pts.append((g[0] - sign * k, lane_y))
    pts.append(g)
    return polyline(*pts)


def _path_clear(deck: DeckMap, p, margin: float = 0.05) -> bool:
    if len(p) < 2:
        return True
    h = np.arctan2(*np.diff(p, axis=0).T[::-1])
    return all(deck.clearance(x, y, a) > margin for (x, y), a in zip(p[:-1], h))


def _line_clear(deck: DeckMap, a, b, margin: float = 0.05) -> bool:
    h = math.atan2(b[1] - a[1], b[0] - a[0])
    return all(deck.clearance(x, y, h) > margin for x, y in polyline(a, b))


class NavWalker:
    def __init__(self, unit: str, xy, yaw: float = 0.0, deck: Optional[DeckMap] = None,
                 profile: Optional[NavProfile] = None, on_event: Optional[Callable] = None, policy=None,
                 lag: float = 0.5):
        self.deck = deck or DeckMap.rig_merged()
        self.nav = FleetNav(unit, profile, on_event, policy)
        self.base = BaseSim(float(xy[0]), float(xy[1]), float(yaw), lag)
        self.angles = self.nav.profile.angles()
        self.t, self.acc, self.goal = 0.0, 0.0, None

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.base.x, self.base.y])

    @property
    def yaw(self) -> float:
        return self.base.yaw

    @property
    def state(self) -> str:
        return self.nav.state

    @property
    def events(self) -> list:
        return self.nav.events

    def go(self, goal_xy, yaw: Optional[float] = None) -> None:
        g = np.asarray(goal_xy, float)[:2]
        key = (round(float(g[0]), 3), round(float(g[1]), 3), None if yaw is None else round(float(yaw), 3))
        if key == self.goal:
            return
        self.goal = key
        path = plan(self.xy, g, self.deck)
        if len(path) < 2 or np.linalg.norm(path[-1] - path[0]) < 0.05:
            path = np.array([self.xy, self.xy + 0.05 * np.array([math.cos(self.yaw), math.sin(self.yaw)])])
        h = math.atan2(path[-1, 1] - path[-2, 1], path[-1, 0] - path[-2, 0])
        self.nav.start(Leg(self.nav.unit, "go", path, h if yaw is None else float(yaw)))

    def resume(self) -> None:
        self.nav.resume()

    def tick(self) -> NavCmd:
        b, p = self.base, self.nav.profile
        ranges = self.deck.raycast(b.x, b.y, b.yaw, self.angles, p.lidar_z, p.lidar_range)
        cmd = self.nav.step(self.t, ranges, self.angles, b.pose, b.v, b.w)
        b.step(cmd.vx, cmd.wz, 0.1)
        self.t = round(self.t + 0.1, 6)
        return cmd

    def step(self, dt: float) -> bool:
        self.acc += dt
        while self.acc >= 0.1 - 1e-9 and self.nav.state != ARRIVED:
            self.acc -= 0.1
            self.tick()
        if self.nav.state == ARRIVED:
            self.acc = 0.0
        return self.nav.state == ARRIVED
