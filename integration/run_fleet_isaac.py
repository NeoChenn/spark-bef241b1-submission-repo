from __future__ import annotations

import argparse
import json
import math
import os
import sys
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

UNITS = ("unit1", "unit2")
TIMELINES = ("runs/fleet_demo/live_timeline.jsonl", "runs_sample/fleet_demo/live_timeline.jsonl")
LED_STATE = {"green": "ok", "amber": "warn", "red": "alarm", "grey": "fault", "gray": "fault",
             "ok": "ok", "warn": "warn", "alarm": "alarm", "fault": "fault"}
POS_KEYS = ("pos", "position", "xy", "pose")
ROBOT_KEYS = ("robot", "unit", "rid")
STATE_KEYS = ("state", "mode", "status", "activity", "phase")
NAV_DT = 0.1
MASK_KEYS = ("rust_grid", "sprayed", "painted")
LIDAR_RANGE = 4.0
STAND_HEIGHT = 0.32
UNIT1_USD = REPO / "sim/output/robot/go2_ur3_lift/go2_ur3_lift.usda"
UNIT1_URDF = REPO / "sim/assets/go2_ur3/go2_ur3_lift.urdf"
UNIT2_USD = REPO / "sim/output/robot/go2_ur3_fixer/go2_ur3_fixer.usda"
UNIT2_URDF = REPO / "sim/assets/go2_ur3/go2_ur3_fixer.urdf"
LIDAR_X = 0.345
NAV_POLICY = REPO / "patrol/models/nav_v3a/policy.onnx"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Render the fleet demo timeline in Isaac Sim (no overlays).")
    p.add_argument("--timeline", default="", help="JSONL timeline; 'demo' = built-in scenario")
    p.add_argument("--scene", default="sim/output/rig_merged.usda")
    p.add_argument("--meta", default="sim/output/rig_merged_meta.json")
    p.add_argument("--flanges", default="sim/output/leak_flanges_meta.json")
    p.add_argument("--out", default="runs/fleet_isaac")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = to the end of the timeline")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--every", type=int, default=1, help="render every n-th frame (quick tests)")
    p.add_argument("--at", default="", help="comma-separated times: render only these frames (checks)")
    p.add_argument("--renderer", default="RaytracedLighting",
                   choices=["RaytracedLighting", "PathTracing", "RealTimePathTracing"])
    p.add_argument("--spp", type=int, default=16)
    p.add_argument("--subframes", type=int, default=1)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--look", default="night", choices=["night", "none"])
    p.add_argument("--dome", type=float, default=120.0)
    p.add_argument("--moon", type=float, default=25.0)
    p.add_argument("--lamp-gain", type=float, default=1.0)
    p.add_argument("--rig-gain", type=float, default=0.1)
    p.add_argument("--nav", default="timeline", choices=["timeline", "lidar"])
    p.add_argument("--nav-policy", default=str(NAV_POLICY), help="ONNX policy, or 'pure_pursuit'")
    p.add_argument("--lidar-beams", type=int, default=360)
    p.add_argument("--lidar-height", type=float, default=0.28)
    p.add_argument("--scrub-log", default="", help="render_demo raw_*.json with recorded PPO brush positions")
    p.add_argument("--unit2-usd", default="")
    p.add_argument("--unit2-urdf", default="")
    p.add_argument("--camera", default="auto", choices=["auto", "wide", "unit1", "unit2"])
    p.add_argument("--focal", type=float, default=24.0)
    p.add_argument("--jpeg", type=int, default=95)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--write-demo", default="", help="write the built-in timeline to this path and exit")
    return p.parse_args(argv)


def resolve(path: str) -> Path:
    q = Path(path)
    return q if q.is_absolute() else REPO / q


def pick_timeline(arg: str) -> str:
    if arg:
        return arg
    for c in TIMELINES:
        if resolve(c).exists():
            return str(resolve(c))
    return "demo"


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _first(d: dict, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _xyyaw(d: dict):
    p = _first(d, POS_KEYS)
    if isinstance(p, dict):
        p = [p.get("x"), p.get("y"), p.get("yaw", d.get("yaw"))]
    if p is None or len(p) < 2:
        return None
    yaw = d.get("yaw", d.get("heading"))
    if yaw is None and "pose" in d and len(p) >= 3:
        yaw = p[3] if len(p) >= 4 else p[2]
    return float(p[0]), float(p[1]), None if yaw is None else float(yaw)


def _led(v):
    if isinstance(v, dict):
        v = _first(v, ("led", "state", "status"))
    return LED_STATE.get(str(v).lower()) if v is not None else None


def _det_id(raw) -> str:
    s = str(raw)
    flange = s.split("-")[-1].split("_")[-1]
    return f"G-{flange}"


@dataclass
class Scenario:
    poses: dict = field(default_factory=dict)
    states: dict = field(default_factory=dict)
    leds: list = field(default_factory=list)
    events: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    end: float = 0.0

    def add_pose(self, rid, t, x, y, yaw, state):
        self.poses.setdefault(rid, []).append([t, x, y, yaw])
        if state is not None:
            self.states.setdefault(rid, []).append((t, str(state)))
        self.end = max(self.end, t)

    def finish(self):
        for rid, rows in self.poses.items():
            rows.sort(key=lambda r: r[0])
            last = 0.0
            for i, r in enumerate(rows):
                if r[3] is None:
                    nxt = rows[i + 1] if i + 1 < len(rows) else None
                    prv = rows[i - 1] if i else None
                    a, b = (r, nxt) if nxt is not None else (prv, r)
                    d = None if a is None or b is None else (b[1] - a[1], b[2] - a[2])
                    r[3] = math.atan2(d[1], d[0]) if d and math.hypot(*d) > 0.05 else last
                last = r[3]
            self.poses[rid] = np.asarray(rows, float)
        for rid in self.states:
            self.states[rid].sort(key=lambda s: s[0])
        self.leds.sort(key=lambda e: e[0])
        self.masks.sort(key=lambda e: e[0])
        self.events.sort(key=lambda e: e["t"])
        return self

    def pose(self, rid, t):
        a = self.poses.get(rid)
        if a is None or not len(a):
            return None
        if t <= a[0, 0]:
            return tuple(a[0, 1:])
        if t >= a[-1, 0]:
            return tuple(a[-1, 1:])
        i = int(np.searchsorted(a[:, 0], t, side="right")) - 1
        t0, t1 = a[i, 0], a[i + 1, 0]
        s = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        x = a[i, 1] + s * (a[i + 1, 1] - a[i, 1])
        y = a[i, 2] + s * (a[i + 1, 2] - a[i, 2])
        yaw = wrap(a[i, 3] + s * wrap(a[i + 1, 3] - a[i, 3]))
        return float(x), float(y), float(yaw)

    def state(self, rid, t, default="IDLE"):
        seq = self.states.get(rid, [])
        i = bisect_right([s[0] for s in seq], t) - 1
        return seq[i][1] if i >= 0 else default

    def detectors(self, t) -> dict:
        out = {}
        for tt, did, st in self.leds:
            if tt > t:
                break
            out[did] = st
        return out

    def masks_at(self, t):
        i = bisect_right([m[0] for m in self.masks], t) - 1
        return self.masks[i][1] if i >= 0 else None

    def events_between(self, t0, t1) -> list:
        return [e for e in self.events if t0 < e["t"] <= t1]

    def segments(self, rid, min_step=0.05):
        a = self.poses.get(rid)
        if a is None or len(a) < 2:
            return []
        moving = np.hypot(np.diff(a[:, 1]), np.diff(a[:, 2])) > min_step
        out, i = [], 0
        while i < len(moving):
            if moving[i]:
                j = i
                while j + 1 < len(moving) and moving[j + 1]:
                    j += 1
                out.append((float(a[i, 0]), float(a[j + 1, 0]), a[i:j + 2, 1:3].copy()))
                i = j + 1
            else:
                i += 1
        return out


def grid(v):
    if isinstance(v, list) and v and isinstance(v[0], str):
        return np.array([[c == "1" for c in row] for row in v], float)
    if isinstance(v, list) and v and isinstance(v[0], list):
        return np.asarray(v, float)
    return None


def parse_rows(rows) -> Scenario:
    scn = Scenario(rows=list(rows))
    for r in rows:
        t = r.get("t", r.get("time"))
        if t is None:
            continue
        t = float(t)
        scn.end = max(scn.end, t)
        robots = r.get("robots")
        if isinstance(robots, dict):
            for rid, d in robots.items():
                p = _xyyaw(d) if isinstance(d, dict) else None
                if p:
                    scn.add_pose(rid, t, *p, _first(d, STATE_KEYS))
        rid = _first(r, ROBOT_KEYS)
        if isinstance(rid, str) and _first(r, POS_KEYS) is not None:
            p = _xyyaw(r)
            if p:
                scn.add_pose(rid, t, *p, _first(r, STATE_KEYS))
        u1 = robots.get("unit1", {}) if isinstance(robots, dict) else {}
        src = u1.get("surface") if isinstance(u1.get("surface"), dict) else u1 if isinstance(u1, dict) else {}
        m = {k: grid(src.get(k)) for k in MASK_KEYS}
        m = {k: v for k, v in m.items() if v is not None}
        if m:
            if "nozzle_on" in src:
                m["nozzle_on"] = bool(src["nozzle_on"])
            scn.masks.append((t, m))
        dets = r.get("detectors")
        if isinstance(dets, dict):
            for did, v in dets.items():
                st = _led(v)
                if st:
                    scn.leds.append((t, _det_id(did), st))
        elif isinstance(dets, list):
            for d in dets:
                if isinstance(d, dict) and _first(d, ("id", "sensor")) is not None and _led(d):
                    scn.leds.append((t, _det_id(_first(d, ("id", "sensor"))), _led(d)))
        typ = r.get("type", r.get("event"))
        if typ in ("detector", "led", "finding") and _first(r, ("id", "sensor", "detector")) is not None:
            st = _led(_first(r, ("led", "state")))
            if st:
                scn.leds.append((t, _det_id(_first(r, ("detector", "sensor", "id"))), st))
        if typ in ("command", "ticket", "finding", "vetoed", "argon_call", "event") or "event" in r:
            scn.events.append({"t": t, **{k: v for k, v in r.items() if k != "t" and k not in ("robots", "detectors")}})
    return scn.finish()


def load_scenario(path: str, meta: dict | None = None, flanges: dict | None = None) -> Scenario:
    if path == "demo":
        return parse_rows(demo_rows(meta, flanges))
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return parse_rows(rows)


def _flange(flanges: dict | None, fid: str) -> dict:
    for f in (flanges or {}).get("flanges", []):
        if f["id"] == fid:
            return f
    return {"id": fid, "center_m": [20.4602, -10.5044, 43.4], "normal": [0.9203, 0.3911, 0.0]}


def flange_stance(f: dict, robot_xy, standoff: float = 0.75):
    c, n = np.asarray(f["center_m"][:2], float), np.asarray(f["normal"][:2], float)
    side = np.array([-n[1], n[0]])
    if float((np.asarray(robot_xy, float) - c) @ side) < 0:
        side = -side
    p = c + side * standoff
    yaw = math.atan2(*(c - p)[::-1])
    return float(p[0]), float(p[1]), wrap(yaw + math.pi / 2)


def demo_rows(meta: dict | None = None, flanges: dict | None = None, fid: str = "F5") -> list[dict]:
    spawn = ((meta or {}).get("world") or {}).get("robot_spawn", {}).get("position_m", [18.2567, -5.25, 42.82])
    u1 = np.array(spawn[:2], float)
    dock = np.array([14.2, -3.6])
    f = _flange(flanges, fid)
    sx, sy, syaw = flange_stance(f, dock)
    stance = np.array([sx, sy])
    back = u1 + np.array([-1.8, 0.9])
    rows = []

    def keyframes(keys, t):
        for (t0, p0, s0), (t1, p1, _) in zip(keys, keys[1:]):
            if t <= t1:
                s = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                return np.asarray(p0) + s * (np.asarray(p1) - np.asarray(p0)), s0
        return np.asarray(keys[-1][1]), keys[-1][2]

    mid = np.array([stance[0] - 1.5, dock[1] - 2.2])
    k1 = [(0, u1, "SCRUB"), (11, u1, "HOLD"), (12, u1, "RETREAT"), (15, back, "HOLD"), (33, back, "RESUME"),
          (36, u1, "SCRUB"), (46, u1, "PAINT"), (52, u1, "DONE"), (55, u1, "DONE")]
    k2 = [(0, dock, "DOCKED"), (13, dock, "GO_CHECK"), (19, mid, "GO_CHECK"), (23, stance, "SNIFF"),
          (26, stance, "RETORQUE"), (31, stance, "VERIFY"), (34, stance, "RETURN"), (40, mid, "RETURN"),
          (45, dock, "DOCKED"), (55, dock, "DOCKED")]
    leds = [(0, "green"), (8, "amber"), (10, "red"), (31, "amber"), (33, "green")]
    for t in range(0, 56):
        for rid, keys, yaw0 in (("unit1", k1, 0.0), ("unit2", k2, None)):
            p, st = keyframes(keys, float(t))
            row = {"type": "robot", "t": float(t), "robot": rid, "pos": [round(float(p[0]), 3), round(float(p[1]), 3)],
                   "state": st}
            if rid == "unit1" and st in ("SCRUB", "PAINT", "DONE", "HOLD") and np.allclose(p, u1):
                row["yaw"] = yaw0
            if rid == "unit2" and st in ("SNIFF", "RETORQUE", "VERIFY"):
                row["yaw"] = round(syaw, 4)
            rows.append(row)
    for t, c in leds:
        rows.append({"type": "detector", "t": float(t), "id": f"G-{fid}", "led": c})
    for t, kind, robot in ((11, "hold", "unit1"), (12, "go_check", "unit2"), (23, "fix_leak", "unit2"),
                           (33, "resume_rust", "unit1")):
        rows.append({"type": "command", "t": float(t), "kind": kind, "robot": robot, "target": fid})
    return rows


def activity(rid: str, state: str) -> str:
    s = state.upper()
    if rid == "unit1":
        for key, act in (("VERIFY", "stow"), ("SOAK", "stow"), ("PAINT", "paint"), ("SPRAY", "spray"),
                         ("SCRUB", "scrub"), ("RUST", "scrub")):
            if key in s:
                return act
        return "stow"
    for key, act in (("DOCK", "stow"), ("TORQUE", "retorque"), ("FIX", "retorque"), ("REPAIR", "retorque"), ("CLOSE", "retorque"),
                     ("SNIFF", "inspect"), ("INSPECT", "inspect"), ("VERIFY", "inspect"), ("CONFIRM", "inspect")):
        if key in s:
            return act
    return "stow"


def activity_totals(scn: Scenario, rid: str = "unit1") -> dict:
    out = {}
    seq = scn.states.get(rid, [])
    for (t0, st), nxt in zip(seq, [*seq[1:], (scn.end, None)]):
        a = activity(rid, st)
        out[a] = out.get(a, 0.0) + max(0.0, nxt[0] - t0)
    return out


def rust_masks(shape, pipe: dict, side: float, z0: float, prog: dict, radius: float = 0.13, length: float = 0.8,
               full: dict | None = None):
    full = {"spray": 3.0, "scrub": 14.0, "paint": 4.0, **(full or {})}
    n_z, n_t = shape
    z = -length / 2 + (np.arange(n_z) + 0.5) * length / n_z
    th = (np.arange(n_t) + 0.5) * 2 * math.pi / n_t
    c = pipe["theta"] if side > 0 else math.pi - pipe["theta"]
    dth = np.abs((th - c + math.pi) % (2 * math.pi) - math.pi) * pipe["radius"]
    d = np.hypot((z - z0)[:, None], dth[None, :]) / radius
    sprayed = d < 1.15 * min(1.0, prog.get("spray", 0.0) / full["spray"])
    rust = np.where(d < 1.0, np.clip(1.0 - d, 0.2, 1.0), 0.0) * (d >= min(1.0, prog.get("scrub", 0.0) / full["scrub"]))
    painted = d < 1.15 * min(1.0, prog.get("paint", 0.0) / full["paint"])
    return {"rust_grid": rust, "sprayed": sprayed.astype(float), "painted": painted.astype(float)}


def resample(a: np.ndarray, shape) -> np.ndarray:
    a = np.asarray(a, float)
    if a.shape == tuple(shape):
        return a
    i = np.minimum((np.arange(shape[0]) + 0.5) * a.shape[0] / shape[0], a.shape[0] - 1).astype(int)
    j = np.minimum((np.arange(shape[1]) + 0.5) * a.shape[1] / shape[1], a.shape[1] - 1).astype(int)
    return a[i][:, j]


def gait(stand: dict, phase: float, amp: float) -> dict:
    q = {}
    for leg, sign in (("FL", 1), ("RR", 1), ("FR", -1), ("RL", -1)):
        lift = max(0.0, sign * math.sin(phase)) * amp
        q[f"{leg}_thigh_joint"] = stand[f"{leg}_thigh_joint"] - 0.25 * lift * 0.6
        q[f"{leg}_calf_joint"] = stand[f"{leg}_calf_joint"] - 0.25 * lift * 0.9
    return q


def scan_boxes(boxes: np.ndarray, x: float, y: float, yaw: float, n: int = 360, rmax: float = LIDAR_RANGE):
    angles = np.deg2rad(np.arange(n) * 360.0 / n - 180.0 + 180.0 / n)
    ranges = np.full(n, np.inf)
    if boxes is None or not len(boxes):
        return ranges, angles
    b = np.asarray(boxes, float)
    near = (b[:, 0] - rmax < x) & (b[:, 2] + rmax > x) & (b[:, 1] - rmax < y) & (b[:, 3] + rmax > y)
    inside = (b[:, 0] <= x) & (x <= b[:, 2]) & (b[:, 1] <= y) & (y <= b[:, 3])
    b = b[near & ~inside]
    if not len(b):
        return ranges, angles
    a = angles + yaw
    dx, dy = np.cos(a)[:, None], np.sin(a)[:, None]
    dx = np.where(np.abs(dx) < 1e-9, 1e-9, dx)
    dy = np.where(np.abs(dy) < 1e-9, 1e-9, dy)
    tx1, tx2 = (b[None, :, 0] - x) / dx, (b[None, :, 2] - x) / dx
    ty1, ty2 = (b[None, :, 1] - y) / dy, (b[None, :, 3] - y) / dy
    tn = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    tf = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    hit = (tn <= tf) & (tn > 0)
    d = np.where(hit, tn, np.inf).min(axis=1)
    ranges = np.where(d <= rmax, d, np.inf)
    return ranges, angles


def robot_box(x: float, y: float, r: float = 0.35) -> list:
    return [x - r, y - r, x + r, y + r]


@dataclass
class NavUnit:
    rid: str
    x: float
    y: float
    yaw: float
    v: float = 0.0
    w: float = 0.0
    dist: float = 0.0
    seg: int = -1
    ctrl: object = None
    vx: float = 0.0
    wz: float = 0.0
    mode: str = "hold"
    min_range: float = float("inf")


def make_controller(policy: str, path_xy):
    from patrol.controller import PatrolController

    if policy != "pure_pursuit" and not Path(policy).exists():
        policy = "pure_pursuit"
    return PatrolController(policy, path_xy)


def nav_step(u: NavUnit, scn: Scenario, segs: list, t: float, boxes, args, others=()) -> None:
    tgt = scn.pose(u.rid, t)
    active = next((k for k, (ta, tb, _) in enumerate(segs) if ta <= t <= tb + 1.5), -1)
    if args.nav == "lidar" and active >= 0:
        if active != u.seg:
            u.seg = active
            u.ctrl = make_controller(args.nav_policy, segs[active][2])
        extra = np.asarray([robot_box(ox, oy) for ox, oy in others], float).reshape(-1, 4)
        allb = extra if boxes is None or not len(boxes) else np.vstack([boxes, extra])
        lx, ly = u.x + LIDAR_X * math.cos(u.yaw), u.y + LIDAR_X * math.sin(u.yaw)
        ranges, angles = scan_boxes(allb, lx, ly, u.yaw, args.lidar_beams)
        u.min_range = float(np.min(ranges))
        vx, wz, _ = u.ctrl.step(ranges, angles, (u.x, u.y, u.yaw), u.v, u.w)
        if u.ctrl.dist_to_goal < 0.15:
            vx, wz = 0.0, 0.0
        u.vx, u.wz, u.mode = float(vx), float(wz), "lidar"
        u.v += 0.5 * (u.vx - u.v)
        u.w += 0.5 * (u.wz - u.w)
        u.yaw = wrap(u.yaw + u.w * NAV_DT)
        u.x += u.v * math.cos(u.yaw) * NAV_DT
        u.y += u.v * math.sin(u.yaw) * NAV_DT
        u.dist += abs(u.v) * NAV_DT + 0.15 * abs(u.w) * NAV_DT
        return
    u.seg, u.mode, u.vx, u.wz = -1, "timeline", 0.0, 0.0
    if tgt is None:
        return
    k = min(1.0, 3.0 * NAV_DT) if args.nav == "lidar" else 1.0
    nx, ny = u.x + k * (tgt[0] - u.x), u.y + k * (tgt[1] - u.y)
    nyaw = wrap(u.yaw + k * wrap(tgt[2] - u.yaw))
    step = math.hypot(nx - u.x, ny - u.y)
    u.v, u.w = step / NAV_DT, wrap(nyaw - u.yaw) / NAV_DT
    u.dist += step + 0.15 * abs(nyaw - u.yaw)
    u.x, u.y, u.yaw = nx, ny, nyaw


def load_brush_log(path: str, offset) -> np.ndarray | None:
    if not path or not Path(path).exists():
        return None
    log = json.loads(Path(path).read_text()).get("log", [])
    pts = [[e["t"], *e["brush"]] for e in log if "brush" in e and "SCRUB" in str(e.get("state", "")).upper()]
    if len(pts) < 4:
        return None
    a = np.asarray(pts, float)
    a[:, 0] -= a[0, 0]
    a[:, 1:] += np.asarray(offset, float)
    return a


def brush_target(t: float, pipe: dict, log: np.ndarray | None = None, robot_xy=None):
    o, ax = np.asarray(pipe["origin"], float), np.asarray(pipe["axis"], float)
    ref = np.asarray(pipe["ref"], float)
    bi = np.cross(ax, ref)
    r = float(pipe["radius"])
    side, z0 = 1.0, pipe["z"]
    if robot_xy is not None:
        rel = np.array([robot_xy[0], robot_xy[1], o[2]]) - o
        side = 1.0 if rel @ ref >= 0 else -1.0
        z0 = float(rel @ ax)
    if log is not None:
        tt = t % log[-1, 0]
        p = np.array([np.interp(tt, log[:, 0], log[:, k]) for k in (1, 2, 3)])
        rel = p - o
        along = rel @ ax
        radial = rel - along * ax
        radial = (radial @ ref) * side * ref + (radial @ bi) * bi
        p = o + (along - pipe["z"] + z0) * ax + radial
        n = radial / max(np.linalg.norm(radial), 1e-6)
        return p, -n
    th = pipe["theta"] + 0.28 * math.sin(1.7 * t)
    if side < 0:
        th = math.pi - th
    z = z0 + 0.14 * math.sin(0.55 * t)
    n = math.cos(th) * ref + math.sin(th) * bi
    return o + z * ax + (r - 0.004) * n, -n


def scrub_stance(pipe: dict, xy, offset: float = 0.5):
    o, ax, ref = (np.asarray(pipe[k], float) for k in ("origin", "axis", "ref"))
    rel = np.array([xy[0], xy[1], o[2]]) - o
    side = 1.0 if rel @ ref >= 0 else -1.0
    p = o + float(rel @ ax) * ax + side * offset * ref
    yaw = math.atan2(ax[1], ax[0]) + (0.0 if side > 0 else math.pi)
    return float(p[0]), float(p[1]), wrap(yaw)


def pipe_from_meta(meta: dict) -> dict:
    w = meta["world"]
    mp = w["main_pipe"]
    o = np.asarray(w.get("segment_origin_m", [18.2567, mp["y_m"], mp["z_m"]]), float)
    d = w.get("damage", {})
    n = np.asarray(d.get("surface_normal", [0.0, 0.6212, 0.7836]), float)
    ref = np.asarray(mp.get("ref", [0, 1, 0]), float)
    ax = np.asarray(mp.get("axis", [1, 0, 0]), float)
    theta = math.atan2(float(n @ np.cross(ax, ref)), float(n @ ref))
    dz = float((np.asarray(d.get("position_m", o), float) - o) @ ax)
    return {"origin": o.tolist(), "axis": ax.tolist(), "ref": ref.tolist(), "radius": float(mp["radius_m"]),
            "theta": theta, "z": dz}


def bolt_target(f: dict, robot_xy, k: int = 0):
    c = np.asarray(f["center_m"], float)
    n = np.asarray(f["normal"], float)
    ref = np.asarray(f.get("ref", [0, 0, 1]), float)
    a = 2 * math.pi * k / int(f.get("n_bolts", 8))
    p = c + float(f.get("bolt_circle_m", 0.13)) * (math.cos(a) * ref + math.sin(a) * np.cross(n, ref))
    side = 1.0 if float((np.asarray(robot_xy, float) - c[:2]) @ n[:2]) >= 0 else -1.0
    return p + side * 0.05 * n, -side * n


def nearest_flange(flanges: dict | None, xy, within: float = 3.5):
    best = None
    for f in (flanges or {}).get("flanges", []):
        d = math.dist(f["center_m"][:2], xy)
        if d <= within and (best is None or d < best[0]):
            best = (d, f)
    return None if best is None else best[1]


def wide_shot(scn: Scenario, meta: dict, focal: float = 24.0):
    pts = [a[:, 1:3] for a in scn.poses.values() if len(a)]
    deck = float(((meta or {}).get("world") or {}).get("deck_z_m", 42.5))
    c = np.concatenate(pts).mean(0) if pts else np.array([18.75, -5.75])
    span = float(np.ptp(np.concatenate(pts), axis=0).max()) if pts else 8.0
    dist = max(6.0, 0.6 * span + 3.0)
    eye = np.array([c[0] - 0.3 * dist, c[1] + 0.75 * dist, deck + 0.28 * dist + 1.0])
    target = np.array([c[0], c[1], deck + 0.6])
    return eye.tolist(), target.tolist(), focal


def beat_times(scn: Scenario, start: float = 0.0) -> dict:
    def first(seq, pred, after=None):
        return next((t for t, v in seq if pred(v) and (after is None or t >= after)), None)

    leds = [(t, st) for t, _, st in scn.leds if t >= start]
    s1 = [(t, st) for t, st in scn.states.get("unit1", []) if t >= start]
    s2 = [(t, st) for t, st in scn.states.get("unit2", []) if t >= start]
    b = {"start": start, "spray": first(s1, lambda v: activity("unit1", v) == "spray"),
         "scrub": first(s1, lambda v: activity("unit1", v) == "scrub"), "amber": first(leds, lambda v: v == "warn")}
    b["red"] = first(leds, lambda v: v == "alarm", b["amber"])
    b["step_back"] = first(s1, lambda v: activity("unit1", v) == "stow" and "DONE" not in v.upper(), b["amber"])
    b["arrive"] = first(s2, lambda v: activity("unit2", v) != "stow", b["red"])
    rt = first(s2, lambda v: activity("unit2", v) == "retorque", b["arrive"])
    b["fixed"] = first(s2, lambda v: activity("unit2", v) != "retorque", rt)
    red_id = next((d for t, d, st in scn.leds if st == "alarm" and t == b["red"]), None)
    b["green"] = first([(t, st) for t, d, st in scn.leds if d == red_id], lambda v: v == "ok", b["red"])
    b["resume"] = first(s1, lambda v: activity("unit1", v) != "stow", b["step_back"])
    b["paint"] = first(s1, lambda v: activity("unit1", v) == "paint", b["resume"])
    b["end"] = start + min(120.0, max(1.0, scn.end - start))
    out, prev = {}, None
    for k in ("start", "spray", "scrub", "amber", "red", "step_back", "arrive", "fixed", "green", "resume", "paint"):
        v = b.get(k)
        if v is None or (prev is not None and v <= prev + 0.2):
            continue
        out[k] = prev = v
    out["end"] = max(b["end"], (prev or start) + 1.0)
    return out


def summary_for(timeline: str) -> str:
    q = Path(timeline)
    for name in (q.name.replace("timeline.jsonl", "summary.json"), "live_summary.json", "summary.json"):
        if (q.parent / name).exists():
            return str(q.parent / name)
    return ""


def camera_source(args, scn: Scenario, meta: dict):
    wide = wide_shot(scn, meta, args.focal)
    if args.camera in UNITS:
        deck = float(meta["world"]["deck_z_m"])

        def follow(t):
            x, y, _ = scn.pose(args.camera, t)
            return [x - 1.6, y + 2.4, deck + 1.9], [x, y, deck + 0.8], 24.0

        return follow, f"follow {args.camera}"
    if args.camera == "auto":
        try:
            from integration import fleet_camera as fc
        except ImportError:
            fc = None
        fn = None
        if fc is not None:
            for name in ("camera_at", "shot_at", "at", "camera"):
                if callable(getattr(fc, name, None)):
                    fn = getattr(fc, name)
                    break
            if fn is not None and hasattr(fc, "Beats"):
                try:
                    try:
                        beats = fc.Beats.from_events(json.loads(Path(summary_for(args.timeline_path)).read_text()))
                    except Exception as exc:
                        print(f"[fleet] summary beats unusable ({exc}), using timeline beats", flush=True)
                        beats = fc.Beats.from_events(beat_times(scn, args.start))
                    base_fn = fn
                    fn = lambda t: base_fn(t, beats)
                    print(f"[fleet] camera beats {beats}", flush=True)
                except Exception as exc:
                    print(f"[fleet] camera beats default ({exc})", flush=True)
        if fn is not None:
            def cam(t):
                try:
                    c = fn(t)
                except Exception:
                    return wide
                if isinstance(c, dict):
                    return list(c["eye"]), list(c.get("target", c.get("look"))), float(c.get("focal", args.focal))
                return list(c[0]), list(c[1]), float(c[2]) if len(c) > 2 else args.focal

            return cam, "fleet_camera"
    return (lambda t: wide), "wide"


def look_rows(eye, target):
    e, tg = np.asarray(eye, float), np.asarray(target, float)
    f = tg - e
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= max(np.linalg.norm(r), 1e-9)
    u = np.cross(r, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = r, u, -f, e
    return m


def header_row(k: int, t: float, cam, args, units: dict, dets: dict, events: list, acts: dict) -> dict:
    from rig.camera import from_usd

    eye, target, focal = cam
    rows = look_rows(eye, target)
    camd = from_usd(rows.tolist(), focal, 20.955, args.width, args.height)
    robots = {rid: {"pos": [round(u.x, 4), round(u.y, 4)], "yaw": round(u.yaw, 4), "v": round(u.v, 3),
                    "w": round(u.w, 3), "nav": u.mode, "cmd": [round(u.vx, 3), round(u.wz, 3)],
                    "lidar_min": None if not math.isfinite(u.min_range) else round(u.min_range, 3),
                    "activity": acts.get(rid)} for rid, u in units.items()}
    return {"frame": k, "t": round(t, 4), "camera": {**camd, "eye": [round(v, 4) for v in eye],
                                                     "target": [round(v, 4) for v in target], "focal": focal},
            "robots": robots, "detectors": dets, "events": events}


def main(argv=None):
    args = parse_args(argv)
    meta = json.loads(resolve(args.meta).read_text())
    flanges = json.loads(resolve(args.flanges).read_text()) if resolve(args.flanges).exists() else None
    if args.write_demo:
        out = resolve(args.write_demo)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in demo_rows(meta, flanges)))
        print(f"[fleet] wrote {out}", flush=True)
        return
    tl = pick_timeline(args.timeline)
    args.timeline_path = tl
    scn = load_scenario(tl, meta, flanges)
    end = args.start + args.duration if args.duration > 0 else scn.end
    n_frames = int(round((end - args.start) * args.fps))
    at_frames = None
    if args.at:
        at_frames = {int(round((float(v) - args.start) * args.fps)) for v in args.at.split(",") if v.strip()}
        n_frames = min(n_frames, max(at_frames) + 1)
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames * max(1, args.every))
    print(f"[fleet] timeline {tl} end={scn.end:.1f}s robots={sorted(scn.poses)} leds={len(scn.leds)} "
          f"frames={n_frames}", flush=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": args.renderer, "width": args.width, "height": args.height})

    import carb
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom

    from integration.kinematics import ARM_STOWED, STANDING, RobotModel, pose
    from integration.repair_scene import RobotPoser

    settings = carb.settings.get_settings()
    for k, v in {"/rtx/rendermode": args.renderer, "/rtx/pathtracing/spp": args.spp,
                 "/rtx/pathtracing/totalSpp": args.spp, "/rtx/pathtracing/optixDenoiser/enabled": True,
                 "/rtx/post/motionblur/enabled": False, "/rtx/post/dof/enabled": False,
                 "/rtx/post/aa/op": 4, "/rtx/rtpt/maxBounces": 6}.items():
        settings.set(k, v)
    ctx = omni.usd.get_context()
    ctx.open_stage(str(resolve(args.scene)))
    for _ in range(240):
        app.update()
        _, loaded, total = ctx.get_stage_loading_status()
        if total == 0 or loaded >= total:
            break
    stage = ctx.get_stage()
    for path in ("/World/Pipes/damage_patch", "/World/Robots"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(False)

    boxes = None
    if args.nav == "lidar":
        boxes = scene_boxes(stage, meta, args.lidar_height)
        print(f"[fleet] lidar boxes {len(boxes)} at {args.lidar_height} m", flush=True)

    UsdGeom.Xform.Define(stage, "/World/Fleet")
    m1 = RobotModel.from_urdf(UNIT1_URDF)
    u2_usd = resolve(args.unit2_usd) if args.unit2_usd else (UNIT2_USD if UNIT2_USD.exists() else None)
    u2_urdf = resolve(args.unit2_urdf) if args.unit2_urdf else UNIT2_URDF
    import integration.repair_scene as rsmod

    posers, models, tools = {}, {}, {}
    posers["unit1"] = RobotPoser(stage, m1, "/World/Fleet/unit1")
    models["unit1"], tools["unit1"] = m1, "brush_tip"
    if u2_usd is not None and u2_urdf is not None and Path(u2_urdf).exists():
        m2 = RobotModel.from_urdf(u2_urdf)
        saved = rsmod.ROBOT_USD
        rsmod.ROBOT_USD = u2_usd
        try:
            posers["unit2"] = RobotPoser(stage, m2, "/World/Fleet/unit2")
        finally:
            rsmod.ROBOT_USD = saved
        links = {j.child for j in m2.joints.values()}
        tl2 = next((n for n in ("socket_tip", "tool_tip", "grasp_frame", "brush_tip") if n in links), "grasp_frame")
        models["unit2"], tools["unit2"] = m2, tl2
    else:
        posers["unit2"] = RobotPoser(stage, m1, "/World/Fleet/unit2")
        models["unit2"], tools["unit2"] = m1, "brush_tip"
    print(f"[fleet] unit2 usd={u2_usd} urdf={u2_urdf} tool={tools['unit2']}", flush=True)

    if args.look == "night":
        from policy import night_look

        zone = meta.get("prims", {}).get("zone", "/World/RepairZone")
        work = tuple(np.asarray(meta["damage"]["position_m"], float))
        print("[fleet] night look", night_look.apply(stage, zone, work, full=True, dome=args.dome, moon=args.moon,
                                                     lamp_gain=args.lamp_gain, rig_gain=args.rig_gain,
                                                     robot_root="/World/Fleet/unit1", robot_color="graphite",
                                                     rig_root="/World"), flush=True)

    detectors = None
    try:
        from integration.gas_detectors import GasDetectors

        detectors = GasDetectors(stage, meta_path=resolve(args.flanges))
    except Exception as exc:
        print(f"[fleet] gas detectors unavailable: {exc}", flush=True)

    cam_path = "/World/Fleet/camera"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 60000))
    cam.CreateFocalLengthAttr(args.focal)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * args.height / args.width)
    cam_op = UsdGeom.Xformable(cam).AddTransformOp()
    cam_fn, cam_kind = camera_source(args, scn, meta)
    print(f"[fleet] camera {cam_kind}", flush=True)

    pipe = pipe_from_meta(meta)
    spawn = meta["world"]["robot_spawn"]["position_m"]
    from integration.repair_scene import RustOverlay, add_spray_cone, set_spray
    from policy.interfaces import PipeGeometry

    seg_len = float(meta["world"].get("segment_length_m", 0.8))
    first = next((tt for tt, st in scn.states.get("unit1", []) if activity("unit1", st) != "stow"), args.start)
    sx0, sy0, _ = scrub_stance(pipe, scn.pose("unit1", first)[:2] if "unit1" in scn.poses else spawn[:2])
    rel0 = np.array([sx0, sy0, pipe["origin"][2]]) - np.asarray(pipe["origin"])
    side0 = 1.0 if rel0 @ np.asarray(pipe["ref"]) >= 0 else -1.0
    z_rust = float(rel0 @ np.asarray(pipe["axis"]))
    pipe_c = dict(pipe)
    pipe_c["origin"] = (np.asarray(pipe["origin"]) + z_rust * np.asarray(pipe["axis"])).tolist()
    pipe_c["z"] = 0.0
    shape0 = next((m["rust_grid"].shape for _, m in scn.masks if "rust_grid" in m), (40, 64))
    if scn.masks:
        pipe_c = dict(pipe)
    geom = PipeGeometry(pipe_c["origin"], pipe["axis"], pipe["ref"], pipe["radius"], seg_len, *shape0)
    overlay = RustOverlay(stage, geom)
    cone, cone_op = add_spray_cone(stage)
    brush = None
    try:
        from policy import night_look as nl
        from policy.brush_squish import BrushPad, UsdBrush

        root = stage.GetPrimAtPath("/World/Fleet/unit1")
        while True:
            inst = [p_ for p_ in Usd.PrimRange(root) if p_.IsInstanceable()]
            if not inst:
                break
            for p_ in inst:
                p_.SetInstanceable(False)
        for p_ in Usd.PrimRange(root):
            if "scrubber_link" in str(p_.GetPath()) and p_.IsA(UsdGeom.Gprim):
                UsdGeom.Imageable(p_).MakeInvisible()
        sponge = nl._material(stage, "/World/Looks/brush_sponge", nl.CORAL_LIN, 0.85, 0.0, 0.0,
                              emissive=tuple(c_ * 0.015 for c_ in nl.CORAL_LIN))
        hubm = nl._material(stage, "/World/Looks/brush_hub", (0.05, 0.05, 0.055), 0.4, 0.6, 0.3)
        UsdGeom.Scope.Define(stage, "/World/Fleet/brush")
        brush = UsdBrush(stage, "/World/Fleet/brush", BrushPad(4000.0), sponge, hubm)
    except Exception as exc:
        print(f"[fleet] brush squish unavailable: {exc}", flush=True)
    try:
        from integration.fleet_nav import trot_q
    except Exception:
        trot_q = None
    print(f"[fleet] rust at z={z_rust:.2f} side={side0:+.0f} squish={brush is not None} trot={trot_q is not None}",
          flush=True)
    prog = {"spray": 0.0, "scrub": 0.0, "paint": 0.0}
    tot = activity_totals(scn)
    full = {k: max(0.5, 0.8 * tot.get(k, 0.0)) for k in prog}
    print(f"[fleet] rust job seconds {tot} -> full {full}", flush=True)
    brush_log = load_brush_log(args.scrub_log, (pipe["origin"][0], spawn[1], meta["world"]["deck_z_m"]))
    print(f"[fleet] scrub {'recorded ' + args.scrub_log if brush_log is not None else 'scripted sweep'}", flush=True)
    deck = float(meta["world"]["deck_z_m"])
    units, segs, q = {}, {}, {}
    for rid in UNITS:
        p = scn.pose(rid, args.start) or (spawn[0], spawn[1] - (0 if rid == "unit1" else 2.0), 0.0)
        units[rid] = NavUnit(rid, *p)
        segs[rid] = scn.segments(rid)
        q[rid] = {**models[rid].default_q(), **ARM_STOWED, **STANDING}

    out = resolve(args.out)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    import omni.timeline

    omni.timeline.get_timeline_interface().play()
    rp = rep.create.render_product(cam_path, (args.width, args.height))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach(rp)
    writer = None
    if not args.no_video:
        import imageio

        writer = imageio.get_writer(str(out / "fleet_isaac.mp4"), fps=max(1, args.fps // max(1, args.every)),
                                    codec="libx264", quality=9, macro_block_size=8, pixelformat="yuv420p")
    head = (out / "fleet_isaac_header.jsonl").open("w")
    nav_t = args.start
    scrub_t, t_prev, rendered, work_w = 0.0, args.start, 0, 0.0
    seeds = {rid: dict(q[rid]) for rid in UNITS}

    def arm_target(rid, act, u, t):
        if rid == "unit1" and act in ("scrub", "spray", "paint"):
            p, appr = brush_target(scrub_t, pipe, brush_log if act == "scrub" else None, (u.x, u.y))
            if act != "scrub":
                p = p - appr * (0.12 if act == "paint" else 0.2)
            return p, appr, 0.0
        if rid == "unit2" and act in ("retorque", "inspect"):
            f = nearest_flange(flanges, (u.x, u.y))
            if f is None:
                fwd = np.array([math.cos(u.yaw), math.sin(u.yaw), 0.0])
                p, appr = np.array([u.x, u.y, deck + 1.1]) + 0.62 * fwd, fwd
            else:
                p, appr = bolt_target(f, (u.x, u.y))
            if act == "inspect":
                return p - appr * 0.25, appr, 0.0
            return p - appr * 0.005 * (1 + math.sin(2.0 * t)), appr, 2.5 * t
        return None

    for k in range(n_frames):
        t = args.start + k / args.fps
        while nav_t + NAV_DT <= t + 1e-9:
            nav_t += NAV_DT
            for rid, u in units.items():
                others = [(o.x, o.y) for r2, o in units.items() if r2 != rid]
                nav_step(u, scn, segs[rid], nav_t, boxes, args, others)
        acts = {rid: activity(rid, scn.state(rid, t)) for rid in UNITS}
        if acts["unit1"] in ("scrub", "spray", "paint"):
            scrub_t += t - t_prev
            prog[acts["unit1"]] += t - t_prev
        t_prev = t
        working = acts["unit1"] in ("scrub", "spray", "paint")
        work_w = min(1.0, work_w + 1.0 / args.fps) if working else max(0.0, work_w - 1.0 / args.fps)
        for rid, u in units.items():
            model = models[rid]
            moving = abs(u.v) > 0.05 or abs(u.w) > 0.15
            bx, by, byaw = u.x, u.y, u.yaw
            if rid == "unit1" and work_w > 0:
                sx, sy, syaw = scrub_stance(pipe, (u.x, u.y))
                s3 = work_w * work_w * (3 - 2 * work_w)
                bx, by, byaw = bx + s3 * (sx - bx), by + s3 * (sy - by), wrap(byaw + s3 * wrap(syaw - byaw))
                moving = moving or 0 < work_w < 1
            base = pose(bx, by, deck + STAND_HEIGHT + (0.008 * abs(math.sin(u.dist / 0.18 * math.pi)) if moving else 0.0),
                        byaw)
            target_q = dict(q[rid])
            if trot_q is not None:
                legs = trot_q(u.dist, abs(u.v) + 0.25 * abs(u.w) if moving else 0.0)
                target_q.update({j: v for j, v in legs.items() if "thigh" in j or "calf" in j})
            else:
                target_q.update(gait(STANDING, u.dist / 0.18 * math.pi, 1.0 if moving else 0.0))
            goal = arm_target(rid, acts[rid], u, t) if not moving else None
            if goal is None:
                target_q.update(ARM_STOWED)
            else:
                pt, appr, roll = goal
                seed = dict(seeds[rid])
                seed["lift_joint"] = 0.42 if rid == "unit1" else 0.40
                res = model.solve_ik(base, seed, np.asarray(pt, float), np.asarray(appr, float), iterations=60,
                                     tip=tools[rid])
                if res.position_error < 0.03:
                    seeds[rid] = dict(res.q)
                    target_q.update({j: res.q[j] for j in res.q if "ur3" in j or j == "lift_joint"})
                    if roll and "socket_drive_joint" in q[rid]:
                        target_q["socket_drive_joint"] = roll
                    elif roll:
                        target_q["ur3_wrist_3_joint"] = target_q.get("ur3_wrist_3_joint", 0.0) + 0.8 * math.sin(roll)
                else:
                    target_q.update(ARM_STOWED)
            blend = 1.0 if k == 0 else min(1.0, 4.0 / args.fps)
            for j, v in target_q.items():
                if "thigh" in j or "calf" in j:
                    q[rid][j] = v
                else:
                    q[rid][j] = q[rid].get(j, v) + blend * (v - q[rid].get(j, v))
            posers[rid].set(base, q[rid])
            if rid == "unit1":
                tip = model.tool_pose(base, q[rid], "brush_tip")
                act = acts["unit1"]
                live = scn.masks_at(t) or {}
                on = bool(live.get("nozzle_on", act in ("spray", "paint"))) and goal is not None and work_w >= 1.0
                set_spray(cone, cone_op, tip[:3, 3], tip[:3, 0], 0.22 if act == "spray" else 0.14, on)
                if brush is not None:
                    force = 25.0 if act == "scrub" and goal is not None and work_w >= 1.0 else 0.0
                    o_, a_ = np.asarray(pipe["origin"]), np.asarray(pipe["axis"])
                    brush._set("sponge", brush.pad.sponge(tip[:3, 3], tip[:3, :3], o_, a_, pipe["radius"], force))
                    brush._set("hub", brush.pad.hub(tip[:3, 3], tip[:3, :3], o_, a_, pipe["radius"]))
        masks = scn.masks_at(t) or rust_masks(geom.shape, pipe_c, side0, 0.0, prog, length=seg_len, full=full)
        painted = resample(masks.get("painted", np.zeros(geom.shape)), geom.shape) > 0.5
        overlay.wet.update((resample(masks.get("sprayed", np.zeros(geom.shape)), geom.shape) > 0.5) & ~painted)
        overlay.rust.update(np.clip(resample(masks.get("rust_grid", np.zeros(geom.shape)), geom.shape) / 0.1, 0, 1)
                            * ~painted)
        overlay.primer.update(painted.astype(np.float32))
        dets = {}
        for did, st in scn.detectors(t).items():
            dets[did] = st
            if detectors is not None:
                try:
                    if detectors.state(did) != st:
                        detectors.set_detector_state(did, st)
                except KeyError:
                    pass
        c = cam_fn(t)
        cam.GetFocalLengthAttr().Set(float(c[2]))
        cam_op.Set(Gf.Matrix4d(*look_rows(c[0], c[1]).reshape(-1)))
        if at_frames is not None and k not in at_frames:
            continue
        if at_frames is None and k % max(1, args.every):
            continue
        warm = args.warmup if rendered == 0 else 1
        rgb = None
        for i_ in range(warm + 20):
            rep.orchestrator.step(rt_subframes=args.subframes, pause_timeline=False, wait_for_render=True)
            data = np.asarray(ann.get_data())
            if i_ >= warm - 1 and data.size and data.ndim == 3:
                rgb = np.ascontiguousarray(data[..., :3].astype(np.uint8))
                break
        if rgb is None:
            raise RuntimeError(f"annotator stayed empty at t={t:.2f}")
        Image.fromarray(rgb).save(frames_dir / f"{rendered:05d}.jpg", quality=args.jpeg)
        if writer is not None:
            writer.append_data(rgb)
        evs = scn.events_between(t - max(1, args.every) / args.fps, t)
        row = header_row(rendered, t, c, args, units, dets, evs, acts)
        row["rust"] = {"progress": {k: round(v, 2) for k, v in prog.items()}, "source": "timeline" if
                       scn.masks_at(t) is not None else "synthetic"}
        head.write(json.dumps(row) + "\n")
        rendered += 1
        if rendered % 30 == 1:
            print(f"[fleet] frame {rendered}/{n_frames // max(1, args.every)} t={t:.2f} mean={rgb.mean():.1f} "
                  f"u1={units['unit1'].mode}:{acts['unit1']} u2={units['unit2'].mode}:{acts['unit2']}", flush=True)
    head.close()
    if writer is not None:
        writer.close()
    (out / "fleet_isaac.json").write_text(json.dumps({
        "timeline": tl, "frames": rendered, "fps": args.fps // max(1, args.every), "size": [args.width, args.height],
        "renderer": args.renderer, "spp": args.spp, "subframes": args.subframes, "look": args.look, "nav": args.nav,
        "nav_policy": args.nav_policy if Path(args.nav_policy).exists() else "pure_pursuit", "camera": cam_kind,
        "unit2_usd": str(u2_usd) if u2_usd else None, "scrub": "recorded" if brush_log is not None else "scripted",
        "detectors": detectors is not None}, indent=1))
    print(f"[fleet] DONE {rendered} frames -> {out}", flush=True)
    os._exit(0)


def scene_boxes(stage, meta: dict, height: float, radius: float = 30.0, max_side: float = 6.0) -> np.ndarray:
    from pxr import Usd, UsdGeom

    deck = float(meta["world"]["deck_z_m"])
    z = deck + height
    c = np.asarray(meta["zone_to_world"]["translation"][:2], float)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
    out = []
    it = iter(Usd.PrimRange(stage.GetPrimAtPath("/World")))
    for prim in it:
        path = str(prim.GetPath())
        if path.startswith(("/World/Fleet", "/World/GasDetectors", "/World/Looks")) or not prim.IsActive():
            it.PruneChildren()
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        lo, hi = r.GetMin(), r.GetMax()
        if not (lo[2] <= z <= hi[2]):
            continue
        if max(hi[0] - lo[0], hi[1] - lo[1]) > max_side:
            continue
        if math.hypot((lo[0] + hi[0]) / 2 - c[0], (lo[1] + hi[1]) / 2 - c[1]) > radius:
            continue
        out.append([lo[0], lo[1], hi[0], hi[1]])
    return np.asarray(out, float).reshape(-1, 4)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
