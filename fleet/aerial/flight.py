from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CINEMATIC = REPO / "sim/cinematic.py"
RIG_META = REPO / "sim/output/full_rig_meta.json"

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class FlareGeometry:
    base: Vec3
    tip: Vec3
    flame_top_z: float
    flame_reach_xy: float
    source: str


def _literal_tuple(node) -> Optional[tuple]:
    try:
        value = ast.literal_eval(node)
    except Exception:
        return None
    return tuple(value) if isinstance(value, (tuple, list)) else None


def flare_geometry(path: Path = CINEMATIC) -> FlareGeometry:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_flare"), None)
    if fn is None:
        raise ValueError(f"build_flare not found in {path}")
    names: dict[str, tuple] = {}
    blobs: list[tuple] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = _literal_tuple(node.value)
            if value is not None and len(value) == 3:
                names[node.targets[0].id] = tuple(float(v) for v in value)
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and node.iter.args:
            table = node.iter.args[0]
            if isinstance(table, ast.Tuple):
                for row in table.elts:
                    if isinstance(row, ast.Tuple) and len(row.elts) >= 4:
                        try:
                            nums = [float(ast.literal_eval(e)) for e in row.elts[:4]]
                        except Exception:
                            continue
                        blobs.append(tuple(nums))
    if "tip" not in names or "base" not in names:
        raise ValueError(f"build_flare in {path} no longer defines literal base/tip tuples")
    tip = names["tip"]
    if blobs:
        top = max(tip[2] + dz + r * s for dx, dz, r, s in blobs)
        reach = max(abs(dx) + r for dx, dz, r, s in blobs)
    else:
        top, reach = tip[2] + 12.0, 5.0
    return FlareGeometry(names["base"], tip, top, reach, f"{path.name}:build_flare")


def rig_bounds(path: Path = RIG_META) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    lo, hi = np.array(meta["bounds_m"]["min"], float), np.array(meta["bounds_m"]["max"], float)
    lo[2] = max(lo[2], 0.0)
    return lo, hi


def deck_heights(path: Path = RIG_META) -> list[float]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    return sorted(float(v) for v in meta.get("deck_heights_m", {}).values())


@dataclass
class Wind:
    speed_mps: float = 0.0
    toward_deg: float = 0.0
    gust_mps: float = 0.0

    @property
    def vec(self) -> np.ndarray:
        a = math.radians(self.toward_deg)
        return np.array([math.cos(a), math.sin(a)]) * self.speed_mps

    @classmethod
    def from_state(cls, state: dict) -> "Wind":
        w = state.get("wind") or {}
        return cls(float(w.get("speed_mps", 0.0)), float(w.get("toward_deg", 0.0)), float(w.get("gust_mps", 0.0)))


@dataclass
class DroneSpec:
    airspeed_mps: float = 6.0
    climb_mps: float = 3.0
    descend_mps: float = 2.0
    battery_wh: float = 262.0
    hover_w: float = 380.0
    reserve_frac: float = 0.25
    max_wind_mps: float = 12.0
    max_gust_mps: float = 15.0
    max_airspeed_mps: float = 17.0

    def power_w(self, airspeed_mps: float) -> float:
        return self.hover_w + 1.2 * airspeed_mps ** 2


@dataclass
class NoFlyCylinder:
    center_xy: np.ndarray
    radius: float
    z_min: float
    z_max: float = math.inf

    def contains(self, p: Sequence[float]) -> bool:
        return self.z_min <= p[2] <= self.z_max and float(np.hypot(*(np.asarray(p[:2]) - self.center_xy))) < self.radius

    def clearance(self, p: Sequence[float]) -> float:
        radial = float(np.hypot(*(np.asarray(p[:2]) - self.center_xy))) - self.radius
        below = self.z_min - p[2]
        above = p[2] - self.z_max
        vertical = max(below, above)
        if vertical <= 0:
            return radial
        if radial <= 0:
            return vertical
        return math.hypot(radial, vertical)


def flare_no_fly(wind: Wind, geom: Optional[FlareGeometry] = None, heat_margin_m: float = 12.0,
                 plume_height_m: float = 20.0, plume_rise_mps: float = 8.0, below_tip_m: float = 10.0) -> NoFlyCylinder:
    geom = geom or flare_geometry()
    tip = np.array(geom.tip, float)
    drift = wind.vec * (plume_height_m / plume_rise_mps)
    radius = geom.flame_reach_xy + heat_margin_m + 0.5 * float(np.linalg.norm(drift))
    return NoFlyCylinder(tip[:2] + 0.5 * drift, radius, tip[2] - below_tip_m)


@dataclass
class FlightPlan:
    waypoints: list[Vec3]
    length_m: float
    time_s: float
    energy_wh: float
    battery_needed: float
    feasible: bool
    reasons: list[str] = field(default_factory=list)
    min_nofly_clearance_m: float = math.inf
    min_box_clearance_m: float = math.inf
    detours: int = 0

    def summary(self) -> dict:
        return {"waypoints": len(self.waypoints), "length_m": round(self.length_m, 1), "time_s": round(self.time_s),
                "energy_wh": round(self.energy_wh, 1), "battery_needed": round(self.battery_needed, 3),
                "feasible": self.feasible, "reasons": self.reasons, "detours": self.detours,
                "min_nofly_clearance_m": round(float(self.min_nofly_clearance_m), 2),
                "min_box_clearance_m": round(float(self.min_box_clearance_m), 2)}


def box_clearance(p: Sequence[float], lo: np.ndarray, hi: np.ndarray) -> float:
    p = np.asarray(p, float)
    d = np.maximum(lo - p, p - hi)
    outside = np.linalg.norm(np.maximum(d, 0.0))
    return float(outside) if outside > 0 else float(d.max())


def sample_path(waypoints: Sequence[Vec3], step: float = 0.5) -> np.ndarray:
    pts = [np.asarray(waypoints[0], float)]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        for i in range(1, n + 1):
            pts.append(a + (b - a) * i / n)
    return np.array(pts)


def check_path(waypoints: Sequence[Vec3], nofly: NoFlyCylinder, lo: np.ndarray, hi: np.ndarray,
               skip_first: int = 0) -> tuple[float, float]:
    pts = sample_path(waypoints)
    nf = min(nofly.clearance(p) for p in pts)
    start = 0
    if skip_first:
        start = len(sample_path(waypoints[:skip_first + 1])) - 1
    bx = min((box_clearance(p, lo, hi) for p in pts[start:]), default=math.inf)
    return nf, bx


def _segment_hits_circle(a: np.ndarray, b: np.ndarray, c: np.ndarray, r: float) -> bool:
    d = b - a
    t = 0.0 if not d.any() else float(np.clip(np.dot(c - a, d) / np.dot(d, d), 0.0, 1.0))
    return float(np.linalg.norm(a + t * d - c)) < r


def _arc(c: np.ndarray, r: float, a0: float, a1: float, ccw: bool, z: float, step_deg: float = 10.0) -> list[Vec3]:
    span = (a1 - a0) % (2 * math.pi) if ccw else -((a0 - a1) % (2 * math.pi))
    n = max(2, int(abs(math.degrees(span)) / step_deg) + 1)
    return [(float(c[0] + r * math.cos(a0 + span * i / n)), float(c[1] + r * math.sin(a0 + span * i / n)), z)
            for i in range(n + 1)]


def avoid_no_fly(waypoints: list[Vec3], nofly: NoFlyCylinder, lo: np.ndarray, hi: np.ndarray,
                 margin_m: float = 3.0, standoff_m: float = 8.0) -> tuple[list[Vec3], int]:
    detours = 0
    r_safe = nofly.radius + margin_m
    z_under = nofly.z_min - margin_m
    lowered = []
    for p in waypoints:
        inside = float(np.hypot(p[0] - nofly.center_xy[0], p[1] - nofly.center_xy[1])) < r_safe
        if inside and p[2] > z_under:
            lowered.append((p[0], p[1], z_under))
            detours += 1
        else:
            lowered.append(tuple(p))
    waypoints = lowered
    out: list[Vec3] = [waypoints[0]]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        A, B = np.asarray(a, float), np.asarray(b, float)
        z_hi = max(A[2], B[2])
        if z_hi < nofly.z_min or not _segment_hits_circle(A[:2], B[:2], nofly.center_xy, nofly.radius):
            out.append(tuple(b))
            continue
        detours += 1
        c = nofly.center_xy
        a0 = math.atan2(A[1] - c[1], A[0] - c[0])
        a1 = math.atan2(B[1] - c[1], B[0] - c[0])
        best = None
        for ccw in (True, False):
            arc = _arc(c, r_safe, a0, a1, ccw, z_hi)
            if all(box_clearance(p, lo, hi) >= 0.5 * standoff_m for p in arc):
                length = sum(math.dist(p, q) for p, q in zip(arc[:-1], arc[1:]))
                if best is None or length < best[0]:
                    best = (length, arc)
        if best is not None and np.hypot(*(A[:2] - c)) >= r_safe - 1e-6 and np.hypot(*(B[:2] - c)) >= r_safe - 1e-6:
            out.append(best[1][0])
            out.extend(best[1][1:])
            out.append(tuple(b))
            continue
        out.extend([(A[0], A[1], z_under), (B[0], B[1], z_under), tuple(b)])
    return out, detours


def ring_waypoints(lo: np.ndarray, hi: np.ndarray, standoff: float, z: float) -> list[Vec3]:
    x0, y0, x1, y1 = lo[0] - standoff, lo[1] - standoff, hi[0] + standoff, hi[1] + standoff
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z), (x0, y0, z)]


def _perimeter_param(p: Sequence[float], x0, y0, x1, y1) -> tuple[float, tuple[float, float]]:
    w, h = x1 - x0, y1 - y0
    x, y = float(np.clip(p[0], x0, x1)), float(np.clip(p[1], y0, y1))
    dists = {"s": abs(y - y0), "e": abs(x1 - x), "n": abs(y1 - y), "w": abs(x - x0)}
    if not (x0 < p[0] < x1 and y0 < p[1] < y1):
        dists = {"s": abs(p[1] - y0) if p[1] <= y0 else 1e9, "e": abs(p[0] - x1) if p[0] >= x1 else 1e9,
                 "n": abs(p[1] - y1) if p[1] >= y1 else 1e9, "w": abs(p[0] - x0) if p[0] <= x0 else 1e9}
    side = min(dists, key=dists.get)
    if side == "s":
        return x - x0, (x, y0)
    if side == "e":
        return w + (y - y0), (x1, y)
    if side == "n":
        return w + h + (x1 - x), (x, y1)
    return 2 * w + h + (y1 - y), (x0, y)


def perimeter_route(a: Vec3, b: Vec3, lo: np.ndarray, hi: np.ndarray, standoff: float) -> list[Vec3]:
    x0, y0, x1, y1 = lo[0] - standoff, lo[1] - standoff, hi[0] + standoff, hi[1] + standoff
    w, h = x1 - x0, y1 - y0
    L = 2 * (w + h)
    corners = [(0.0, (x0, y0)), (w, (x1, y0)), (w + h, (x1, y1)), (2 * w + h, (x0, y1))]
    sa, pa = _perimeter_param(a, x0, y0, x1, y1)
    sb, pb = _perimeter_param(b, x0, y0, x1, y1)
    z = max(a[2], b[2])
    fwd = (sb - sa) % L
    out: list[Vec3] = [tuple(a)]
    if abs(a[2] - z) > 1e-6:
        out.append((a[0], a[1], z))
    out.append((pa[0], pa[1], z))
    if fwd <= L - fwd:
        cs = sorted(((s - sa) % L, xy) for s, xy in corners if 0 < (s - sa) % L < fwd)
    else:
        cs = sorted(((sa - s) % L, xy) for s, xy in corners if 0 < (sa - s) % L < L - fwd)
    out.extend((xy[0], xy[1], z) for _, xy in cs)
    out.append((pb[0], pb[1], z))
    out.append((b[0], b[1], z))
    if abs(b[2] - z) > 1e-6:
        out.append(tuple(b))
    dedup = [out[0]]
    for p in out[1:]:
        if math.dist(p, dedup[-1]) > 1e-6:
            dedup.append(p)
    return dedup


def estimate(waypoints: Sequence[Vec3], spec: DroneSpec, wind: Wind, dwell_s: float = 0.0) -> tuple[float, float, float]:
    w = wind.vec
    length = time_s = energy_j = 0.0
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        d = np.asarray(b, float) - np.asarray(a, float)
        horiz, vert = float(np.hypot(d[0], d[1])), float(d[2])
        length += float(np.linalg.norm(d))
        t_h, air = 0.0, wind.speed_mps
        if horiz > 1e-9:
            u = d[:2] / horiz
            ground = spec.airspeed_mps
            air = float(np.linalg.norm(u * ground - w))
            if air > spec.max_airspeed_mps:
                along = float(np.dot(w, u))
                cross2 = float(np.dot(w, w)) - along ** 2
                ground = max(along + math.sqrt(max(spec.max_airspeed_mps ** 2 - cross2, 0.0)), 0.5)
                air = spec.max_airspeed_mps
            t_h = horiz / ground
        t_v = vert / spec.climb_mps if vert > 0 else -vert / spec.descend_mps
        t = max(t_h, t_v)
        time_s += t
        energy_j += spec.power_w(air) * t
    time_s += dwell_s
    energy_j += spec.power_w(wind.speed_mps) * dwell_s
    return length, time_s, energy_j / 3600.0


def finalize(waypoints: list[Vec3], spec: DroneSpec, wind: Wind, nofly: NoFlyCylinder, lo, hi,
             battery: float = 1.0, dwell_s: float = 0.0, detours: int = 0, skip_first: int = 0) -> FlightPlan:
    length, time_s, energy = estimate(waypoints, spec, wind, dwell_s)
    need = energy / spec.battery_wh
    nf, bx = check_path(waypoints, nofly, lo, hi, skip_first)
    reasons = []
    if wind.speed_mps > spec.max_wind_mps:
        reasons.append(f"wind {wind.speed_mps:.1f} m/s above limit {spec.max_wind_mps:.1f}")
    if wind.gust_mps > spec.max_gust_mps:
        reasons.append(f"gust {wind.gust_mps:.1f} m/s above limit {spec.max_gust_mps:.1f}")
    if need > battery - spec.reserve_frac:
        reasons.append(f"needs {need:.0%} battery, has {battery:.0%} with {spec.reserve_frac:.0%} reserve")
    if nf <= 0:
        reasons.append("path enters the flare no-fly volume")
    if bx <= 0:
        reasons.append("path enters the rig volume")
    return FlightPlan(list(waypoints), length, time_s, energy, need, not reasons, reasons, nf, bx, detours)


def _takeoff(start: Vec3, lo, hi, standoff: float) -> list[Vec3]:
    if box_clearance(start, lo, hi) < standoff:
        return [tuple(start), (start[0], start[1], float(hi[2] + standoff))]
    return [tuple(start)]


def plan_survey(start: Vec3, wind: Wind, spec: DroneSpec = DroneSpec(), battery: float = 1.0,
                standoff_m: float = 8.0, ring_spacing_m: float = 12.0, z_low: Optional[float] = None,
                nofly: Optional[NoFlyCylinder] = None, bounds=None) -> FlightPlan:
    lo, hi = bounds or rig_bounds()
    nofly = nofly or flare_no_fly(wind)
    z_low = z_low if z_low is not None else min(deck_heights() or [20.0]) + 4.0
    zs = list(np.arange(z_low, hi[2] + 1e-6, ring_spacing_m))
    pts = _takeoff(start, lo, hi, standoff_m)
    skip = len(pts) - 1
    for z in zs:
        ring = ring_waypoints(lo, hi, standoff_m, float(z))
        pts.extend(perimeter_route(pts[-1], ring[0], lo, hi, standoff_m)[1:] if pts else ring[:1])
        pts.extend(ring[1:])
    pts.extend(perimeter_route(pts[-1], (pts[0][0], pts[0][1], pts[1][2] if skip else pts[0][2]), lo, hi, standoff_m)[1:])
    if skip:
        pts.append(tuple(start))
    pts, detours = avoid_no_fly(pts, nofly, lo, hi, standoff_m=standoff_m)
    plan = finalize(pts, spec, wind, nofly, lo, hi, battery, detours=detours, skip_first=skip)
    if skip:
        nf, bx = check_path(pts[:-1], nofly, lo, hi, skip)
        plan.min_box_clearance_m = bx
        plan.reasons = [r for r in plan.reasons if r != "path enters the rig volume"] + (
            ["path enters the rig volume"] if bx <= 0 else [])
        plan.feasible = not plan.reasons
    return plan


def plan_visit(start: Vec3, targets: Sequence[Vec3], wind: Wind, spec: DroneSpec = DroneSpec(),
               battery: float = 1.0, standoff_m: float = 8.0, dwell_s: float = 20.0,
               nofly: Optional[NoFlyCylinder] = None, bounds=None, return_home: bool = True) -> FlightPlan:
    lo, hi = bounds or rig_bounds()
    nofly = nofly or flare_no_fly(wind)
    pts = _takeoff(start, lo, hi, standoff_m)
    skip = len(pts) - 1
    todo = [tuple(map(float, t)) for t in targets]
    while todo:
        cur = pts[-1]
        nxt = min(todo, key=lambda t: math.dist(cur, t))
        todo.remove(nxt)
        pts.extend(perimeter_route(cur, nxt, lo, hi, standoff_m)[1:])
    if return_home:
        top = (pts[skip][0], pts[skip][1], pts[skip][2]) if skip else tuple(start)
        pts.extend(perimeter_route(pts[-1], top, lo, hi, standoff_m)[1:])
    pts, detours = avoid_no_fly(pts, nofly, lo, hi, standoff_m=standoff_m)
    box_pts = pts
    if return_home and skip:
        pts = pts + [tuple(start)]
    plan = finalize(pts, spec, wind, nofly, lo, hi, battery, dwell_s * len(targets), detours, skip)
    if return_home and skip:
        nf, bx = check_path(box_pts, nofly, lo, hi, skip)
        plan.min_box_clearance_m = bx
        plan.reasons = [r for r in plan.reasons if r != "path enters the rig volume"] + (
            ["path enters the rig volume"] if bx <= 0 else [])
        plan.feasible = not plan.reasons
    return plan


def viewpoint_for(target: Vec3, standoff_m: float = 8.0, bounds=None) -> Vec3:
    lo, hi = bounds or rig_bounds()
    p = np.asarray(target, float)
    faces = {(0, -1): p[0] - lo[0], (0, 1): hi[0] - p[0], (1, -1): p[1] - lo[1], (1, 1): hi[1] - p[1]}
    (axis, sign), _ = min(faces.items(), key=lambda kv: kv[1])
    q = p.copy()
    q[axis] = (lo[axis] - standoff_m) if sign < 0 else (hi[axis] + standoff_m)
    return tuple(float(v) for v in q)


def flare_viewpoint(nofly: NoFlyCylinder, geom: FlareGeometry, wind: Wind, margin_m: float = 6.0,
                    bounds=None, standoff_m: float = 8.0) -> Vec3:
    lo, hi = bounds or rig_bounds()
    r = nofly.radius + margin_m
    up = math.atan2(-wind.vec[1], -wind.vec[0]) if wind.speed_mps > 0.1 else math.pi / 2
    for k in range(0, 36):
        for sgn in (1, -1):
            a = up + sgn * math.radians(10 * k)
            p = (float(nofly.center_xy[0] + r * math.cos(a)), float(nofly.center_xy[1] + r * math.sin(a)),
                 float(geom.tip[2]))
            if box_clearance(p, lo, hi) >= standoff_m:
                return p
    raise ValueError("no viewpoint outside both the no-fly volume and the rig standoff")
