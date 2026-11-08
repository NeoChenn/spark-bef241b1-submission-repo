from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
META = REPO / "sim/output/rig_merged_meta.json"
DECK_Z = 42.5
ZONE_TO_WORLD = (18.75, -5.75, 42.5)
BODY_TOP_M = 0.9
TRUNK_TOP_M = 0.5
ARM_R_M = 0.15
LANE_Y = -9.68

F5 = (20.4602, -10.5044, 43.4)
F5_NORMAL = (0.9203, 0.3911, 0.0)
WORK_STANDOFF_M = 0.6
F6 = (31.0124, -10.5895, 43.4)
G_F5 = (20.153, -10.396, 43.46)
G_F6 = (31.198, -10.857, 43.46)
WIND = (0.0, -1.0, 0.0)

SCRUB = (18.2567, -5.25, 0.0)
F5_WORK = (F5[0] + WORK_STANDOFF_M * F5_NORMAL[0], F5[1] + WORK_STANDOFF_M * F5_NORMAL[1],
           math.atan2(-F5_NORMAL[1], -F5_NORMAL[0]))
DOCK = {"unit2": (29.6, LANE_Y, math.pi), "unit1": SCRUB}
G_F6_PASS = (31.6, LANE_Y)
HOLD = (17.4, -4.0, math.atan2(-5.655 - -4.0, 18.2567 - 17.4))
RETREAT = (16.0, -3.6, math.atan2(-5.655 - -3.6, 18.2567 - 16.0))
SAFE = {"unit1": {"hold": HOLD, "retreat": RETREAT},
        "unit2": {"hold": DOCK["unit2"], "retreat": DOCK["unit2"]}}

DECK_BOUNDS = ((10.9, 35.5), (-11.05, -0.75))

BOXES = (
    ("main_pipe", 11.15, 26.35, -5.9, -5.6, 0.95, 1.25),
    ("side_line", 11.0, 26.5, -8.9, -8.6, 0.6, 0.8),
    ("side_line", 11.0, 26.5, -9.32, -9.18, 0.63, 0.77),
    ("side_line", 11.0, 26.5, -2.67, -2.43, 0.68, 0.92),
    ("f5_f6_line", 15.5, 35.0, -10.82, -10.58, 0.78, 1.02),
    ("valve_f5", 19.52, 20.48, -11.0, -10.4, 0.72, 1.08),
    ("valve_f6", 31.02, 31.98, -11.0, -10.4, 0.72, 1.08),
    ("flange_f5", 20.36, 20.53, -10.68, -10.35, 0.73, 1.07),
    ("flange_f6", 30.97, 31.09, -10.76, -10.42, 0.73, 1.07),
    ("flange", 15.45, 15.55, -10.87, -10.53, 0.73, 1.07),
    ("flange", 34.95, 35.05, -10.87, -10.53, 0.73, 1.07),
    ("rail_south", 11.2, 36.2, -11.65, -11.05, 0.0, 1.1),
    ("rail_east", 35.55, 36.95, -12.65, 0.0, 0.0, 1.1),
    ("support", 16.93, 17.07, -10.95, -10.45, 0.0, 0.75),
    ("support", 22.93, 23.07, -10.95, -10.45, 0.0, 0.75),
    ("support", 28.93, 29.07, -10.95, -10.45, 0.0, 0.75),
    ("support", 33.93, 34.07, -10.95, -10.45, 0.0, 0.75),
    ("column", 11.62, 11.88, -10.28, -10.03, 0.0, 4.0),
    ("column", 11.62, 11.88, -1.47, -1.22, 0.0, 4.0),
    ("column", 18.62, 18.88, -1.47, -1.22, 0.0, 4.0),
    ("column", 25.62, 25.88, -10.28, -10.03, 0.0, 4.0),
    ("column", 25.62, 25.88, -1.47, -1.22, 0.0, 4.0),
    ("support", 12.68, 12.82, -6.0, -5.5, 0.0, 0.95),
    ("support", 15.28, 15.42, -6.0, -5.5, 0.0, 0.95),
    ("support", 22.08, 22.22, -6.0, -5.5, 0.0, 0.95),
    ("support", 24.68, 24.82, -6.0, -5.5, 0.0, 0.95),
    ("side_support", 12.2, 12.3, -8.88, -8.62, 0.0, 0.6),
    ("side_support", 16.2, 16.3, -8.88, -8.62, 0.0, 0.6),
    ("side_support", 21.2, 21.3, -8.88, -8.62, 0.0, 0.6),
    ("side_support", 25.2, 25.3, -8.88, -8.62, 0.0, 0.6),
    ("side_support", 12.2, 12.3, -9.38, -9.12, 0.0, 0.63),
    ("side_support", 16.2, 16.3, -9.38, -9.12, 0.0, 0.63),
    ("side_support", 21.2, 21.3, -9.38, -9.12, 0.0, 0.63),
    ("side_support", 25.2, 25.3, -9.38, -9.12, 0.0, 0.63),
    ("side_support", 12.2, 12.3, -2.67, -2.42, 0.0, 0.68),
    ("side_support", 16.2, 16.3, -2.67, -2.42, 0.0, 0.68),
    ("side_support", 21.2, 21.3, -2.67, -2.42, 0.0, 0.68),
    ("side_support", 25.2, 25.3, -2.67, -2.42, 0.0, 0.68),
    ("industrial_pipes", 11.25, 12.65, -2.9, -2.6, 0.0, 1.95),
    ("welding_cart", 21.54, 22.56, -8.2, -7.3, 0.0, 1.56),
    ("generator", 24.1, 24.8, -4.2, -3.3, 0.0, 0.58),
    ("utility_box", 14.49, 15.41, -1.47, -1.03, 0.0, 1.12),
    ("utility_box", 15.59, 16.11, -1.47, -1.03, 0.0, 1.12),
    ("power_box", 13.74, 14.36, -7.93, -7.37, 0.0, 0.51),
    ("propane_tank", 21.78, 22.12, -3.72, -3.38, 0.0, 0.55),
    ("propane_tank", 22.11, 22.59, -3.59, -3.11, 0.0, 0.55),
    ("extinguisher", 15.36, 15.74, -8.07, -7.63, 0.0, 0.66),
    ("toolbox", 21.42, 21.88, -8.54, -8.16, 0.0, 0.35),
    ("barrel", 25.33, 25.97, -1.77, -1.13, 0.0, 0.93),
    ("barrel", 24.54, 25.36, -1.76, -0.94, 0.0, 0.93),
    ("rack", 11.09, 12.01, -10.45, -9.85, 0.0, 1.9),
)
DISCS = (("crane_pedestal", 31.75, -4.65, 2.0, 0.0, 2.0),)


@dataclass
class DeckMap:
    boxes: np.ndarray
    names: list
    discs: np.ndarray
    bounds: tuple = DECK_BOUNDS

    @classmethod
    def rig_merged(cls, extra=()) -> "DeckMap":
        rows = list(BOXES) + list(extra)
        return cls(np.array([r[1:] for r in rows], float), [r[0] for r in rows],
                   np.array([d[1:] for d in DISCS], float))

    def add_box(self, name, x0, x1, y0, y1, z0=0.0, z1=0.5) -> int:
        self.boxes = np.vstack([self.boxes, [x0, x1, y0, y1, z0, z1]])
        self.names.append(name)
        return len(self.names) - 1

    def remove(self, name) -> None:
        keep = [i for i, n in enumerate(self.names) if n != name]
        self.boxes, self.names = self.boxes[keep], [self.names[i] for i in keep]

    def raycast(self, x, y, yaw, angles, h, max_range=4.0) -> np.ndarray:
        th = yaw + np.asarray(angles, float)
        dx, dy = np.cos(th)[:, None], np.sin(th)[:, None]
        b = self.boxes[(self.boxes[:, 4] <= h) & (self.boxes[:, 5] >= h)]
        r = np.full(len(th), np.inf)
        if len(b):
            with np.errstate(divide="ignore", invalid="ignore"):
                tx0, tx1 = (b[:, 0] - x) / dx, (b[:, 1] - x) / dx
                ty0, ty1 = (b[:, 2] - y) / dy, (b[:, 3] - y) / dy
            tn = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
            tf = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
            hit = (tf >= np.maximum(tn, 0)) & np.isfinite(tn)
            t = np.where(hit, np.maximum(tn, 0), np.inf)
            r = np.minimum(r, t.min(1))
        for cx, cy, rad, z0, z1 in self.discs:
            if not z0 <= h <= z1:
                continue
            ox, oy = x - cx, y - cy
            bb = ox * dx[:, 0] + oy * dy[:, 0]
            disc = bb * bb - (ox * ox + oy * oy - rad * rad)
            t = -bb - np.sqrt(np.maximum(disc, 0))
            r = np.minimum(r, np.where((disc >= 0) & (t >= 0), t, np.inf))
        return np.where(r <= max_range, r, np.inf)

    def _parts(self, x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        legs = [(x + k * 0.2 * c, y + k * 0.2 * s, 0.17, 0.0, TRUNK_TOP_M) for k in (-1, 0, 1)]
        return legs + [(x, y, ARM_R_M, TRUNK_TOP_M, BODY_TOP_M)]

    def clearance_parts(self, x, y, yaw=0.0):
        out = []
        b, n = self.boxes, np.array(self.names)
        for px, py, rad, lo, hi in self._parts(x, y, yaw):
            m = (b[:, 4] < hi) & (b[:, 5] > lo)
            d = np.hypot(np.maximum(np.maximum(b[m, 0] - px, px - b[m, 1]), 0),
                         np.maximum(np.maximum(b[m, 2] - py, py - b[m, 3]), 0)) - rad
            out += list(zip(d.tolist(), n[m].tolist()))
            out += [(math.hypot(px - cx, py - cy) - r - rad, "disc") for cx, cy, r, z0, z1 in self.discs
                    if z0 < hi and z1 > lo]
            (x0, x1), (y0, y1) = self.bounds
            out.append((min(px - x0, x1 - px, py - y0, y1 - py) - rad, "deck_edge"))
        return out

    def clearance(self, x, y, yaw=0.0) -> float:
        return min(d for d, _ in self.clearance_parts(x, y, yaw))

    def name_at(self, x, y, yaw=0.0) -> str:
        d, n = min(self.clearance_parts(x, y, yaw))
        return n if d < 0 else ""


def polyline(*pts, ds: float = 0.1) -> np.ndarray:
    p = np.asarray([q[:2] for q in pts], float)
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(int(math.ceil(np.linalg.norm(b - a) / ds)), 1)
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


@dataclass
class Leg:
    unit: str
    name: str
    path: np.ndarray
    goal_yaw: float

    @property
    def start(self) -> np.ndarray:
        return self.path[0]

    @property
    def goal(self) -> np.ndarray:
        return self.path[-1]

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.path, axis=0), axis=1).sum())

    def to_json(self) -> dict:
        return {"frame": "world", "unit": self.unit, "leg": self.name, "goal_yaw": round(self.goal_yaw, 4),
                "waypoints": [[round(float(x), 3), round(float(y), 3)] for x, y in self.path],
                "start_bin": [round(float(v), 3) for v in self.start], "end_bin": [round(float(v), 3) for v in self.goal]}


def legs() -> dict[str, Leg]:
    d2 = DOCK["unit2"]
    jog = (22.4, LANE_Y)
    return {
        "unit2.dispatch": Leg("unit2", "dispatch", polyline(d2, jog, F5_WORK), F5_WORK[2]),
        "unit2.sweep": Leg("unit2", "sweep", polyline(F5_WORK, jog, G_F6_PASS), 0.0),
        "unit2.home": Leg("unit2", "home", polyline(G_F6_PASS, d2), d2[2]),
        "unit1.hold": Leg("unit1", "hold", polyline(SCRUB, HOLD), HOLD[2]),
        "unit1.retreat": Leg("unit1", "retreat", polyline(HOLD, RETREAT), RETREAT[2]),
        "unit1.resume": Leg("unit1", "resume", polyline(HOLD, SCRUB), SCRUB[2]),
        "unit1.resume_from_retreat": Leg("unit1", "resume_from_retreat", polyline(RETREAT, SCRUB), SCRUB[2]),
    }


ROUTES = {"unit2": ("unit2.dispatch", "unit2.sweep", "unit2.home"), "unit1": ("unit1.hold", "unit1.resume")}


def leg(name: str) -> Leg:
    return legs()[name]


def safe_pose(unit: str, kind: str, base_xyz) -> np.ndarray:
    x, y, yaw = SAFE.get(unit, SAFE["unit1"])["retreat" if kind == "retreat" else "hold"]
    b = np.asarray(base_xyz, float).reshape(-1)
    if len(b) == 4:
        return np.array([x, y, b[2], yaw])
    if len(b) == 3:
        return np.array([x, y, b[2]])
    return np.array([x, y])


def safe_yaw(unit: str, kind: str) -> float:
    return SAFE.get(unit, SAFE["unit1"])["retreat" if kind == "retreat" else "hold"][2]


def to_world(local_xyz) -> np.ndarray:
    return np.asarray(local_xyz, float) + np.asarray(ZONE_TO_WORLD)


def meta() -> dict:
    return json.loads(META.read_text())


def export(out) -> dict:
    data = {k: v.to_json() for k, v in legs().items()}
    data["poses"] = {"dock_unit2": DOCK["unit2"], "f5_work": F5_WORK, "scrub": SCRUB, "hold": HOLD, "retreat": RETREAT}
    Path(out).write_text(json.dumps(data, indent=1))
    return data
