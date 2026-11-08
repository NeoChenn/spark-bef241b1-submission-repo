import json
import math

import numpy as np

from rig.sim.gait import ARM, LEGS, MOTION_DIR, PARK, smoothstep

BLEND_S = 0.8
SCRUB_HOLD = ("SETUP", "DETECT", "DEPLOY", "SOAK", "VERIFY", "VERIFY_PAINT")
PICK_PHASES = {"APPROACH": ("APPROACH",), "GRASP": ("DESCEND",), "LIFT": ("LIFT",),
               "STOW": ("CARRY", "LOWER", "RELEASE")}
WELD_WORK = {"lift_joint": 0.3, "ur3_shoulder_pan_joint": 0.0, "ur3_shoulder_lift_joint": -1.1,
             "ur3_elbow_joint": 1.5, "ur3_wrist_1_joint": -1.95, "ur3_wrist_2_joint": -1.57, "ur3_wrist_3_joint": 0.0}
WELD_SOURCE = "weld seam-tracking sweep (scripted)"
SCRUB_TIP = (-0.058, -0.404, 0.904)
BRUSH_R = 0.025


def trim_jump(rows: list, max_step: float = 1.0) -> list:
    for k in range(1, len(rows)):
        if max(abs(a - b) for a, b in zip(rows[k]["q"], rows[k - 1]["q"])) > max_step:
            return rows[:k]
    return rows


def export_clip(raw: dict, every: int = 1) -> dict:
    rows = trim_jump(raw["rows"])[::every]
    keep = ("t", "q", "phase")
    return {"source": raw.get("source", ""), "hz": raw["hz"] / every, "joints": raw["joints"],
            "rows": [{k: r[k] for k in keep if k in r} for r in rows]}


class Clip:
    def __init__(self, hz, joints, q, phases=None, source=""):
        self.hz, self.source = float(hz), source
        q = np.asarray(q, np.float64)
        self.q = q[:, [list(joints).index(j) for j in ARM]]
        self.phases = list(phases) if phases else []

    @classmethod
    def from_json(cls, d: dict) -> "Clip":
        rows = d["rows"]
        return cls(d["hz"], d["joints"], [r["q"] for r in rows], [r.get("phase") for r in rows] if rows and
                   "phase" in rows[0] else None, d.get("source", ""))

    @classmethod
    def load(cls, name: str):
        p = MOTION_DIR / f"{name}.json"
        return cls.from_json(json.loads(p.read_text())) if p.exists() else None

    @property
    def duration(self) -> float:
        return len(self.q) / self.hz

    def frame(self, x: float):
        x = min(max(x, 0.0), len(self.q) - 1.0)
        i = int(x)
        k = min(i + 1, len(self.q) - 1)
        u = x - i
        return self.q[i] * (1 - u) + self.q[k] * u

    def at(self, t: float):
        return self.frame(t * self.hz)

    def pingpong(self, t: float, start: float = 0.0):
        a = start * self.hz
        span = max(len(self.q) - 1 - a, 1.0)
        x = (t * self.hz) % (2 * span)
        return self.frame(a + (x if x <= span else 2 * span - x))

    def span(self, names) -> tuple[float, float]:
        idx = [i for i, p in enumerate(self.phases) if p in names]
        if not idx:
            return 0.0, 0.0
        return idx[0] / self.hz, idx[-1] / self.hz


class Rollout:
    def __init__(self, d: dict):
        self.hz, self.source, self.base_z = float(d["hz"]), d.get("source", ""), float(d.get("base_z", 0.32))
        rows = d["rows"]
        q = np.asarray([r["q"] for r in rows], np.float64)
        self.legs = q[:, [d["joints"].index(j) for j in LEGS]]
        self.arm = q[:, [d["joints"].index(j) for j in ARM]]
        self.can = np.asarray([r["can"] for r in rows], np.float64)
        self.quat = np.asarray([r["quat"] for r in rows], np.float64)
        self.phase = [r.get("phase") for r in rows]
        self.radius, self.length = float(d["can"]["radius"]), float(d["can"]["length"])

    @classmethod
    def load(cls, name: str = "pick_rollout"):
        p = MOTION_DIR / f"{name}.json"
        return cls(json.loads(p.read_text())) if p.exists() else None

    def first(self, phase: str) -> int:
        return self.phase.index(phase)

    def at(self, k: float):
        k = min(max(int(k), 0), len(self.arm) - 1)
        return self.legs[k], self.arm[k], self.can[k], self.quat[k]


def scrub_base(patch, axis_y: float, axis_z: float, pipe_r: float, trunk_z: float):
    tx, ty, tz = SCRUB_TIP
    side = 1.0 if patch[1] >= axis_y else -1.0
    dz = trunk_z + tz - axis_z
    dy = math.sqrt(max((pipe_r + BRUSH_R) ** 2 - dz * dz, 0.0))
    yaw = 0.0 if side > 0 else math.pi
    c, s = math.cos(yaw), math.sin(yaw)
    return patch[0] - (c * tx - s * ty), axis_y + side * dy - (s * tx + c * ty), yaw


def weld_sweep(t: float):
    q = dict(WELD_WORK)
    q["ur3_shoulder_pan_joint"] += 0.35 * math.sin(2 * math.pi * t / 9.0)
    q["ur3_wrist_3_joint"] += 0.18 * math.sin(2 * math.pi * 1.6 * t)
    q["ur3_wrist_1_joint"] += 0.06 * math.sin(2 * math.pi * 1.6 * t + 1.2)
    q["ur3_shoulder_lift_joint"] += 0.05 * math.sin(2 * math.pi * t / 4.5)
    return np.array([q[j] for j in ARM])


def task_kind(r: dict):
    if r.get("state") != "WORK":
        return None
    kind = (r.get("task") or "").split(":")[0]
    js = r.get("job_state") or ""
    if kind == "weld" or js.startswith("WELD"):
        return "weld"
    if kind == "pick_can":
        return "pick_can"
    if kind in ("rust", "rust_repair") or js in ("SPRAY", "SOAK", "SCRUB", "PAINT", "DEPLOY", "DETECT", "VERIFY"):
        return "scrub"
    return None


class TaskMotion:
    def __init__(self, clips: dict):
        self.clips = {k: v for k, v in clips.items() if v is not None}
        self.park = np.array([PARK[j] for j in ARM])
        self.out = self.park.copy()
        self.key = None
        self.blend_from, self.blend_t = self.park.copy(), -1e9
        self.t0 = 0.0
        self.source = None

    @classmethod
    def load(cls):
        return cls({"scrub": Clip.load("scrub"), "pick_can": Clip.load("pick_can")})

    def _pose(self, kind, js, t):
        e = t - self.t0
        if kind == "weld":
            return weld_sweep(e), WELD_SOURCE
        clip = self.clips.get(kind)
        if clip is None:
            return self.park, None
        if kind == "scrub":
            start = min(1.5, clip.duration / 3)
            return (clip.frame(start * clip.hz) if js in SCRUB_HOLD else clip.pingpong(e, start)), clip.source
        names = PICK_PHASES.get(js)
        if names is None:
            return (clip.frame(0) if js in ("DETECT", "SETUP") else clip.frame(len(clip.q) - 1)), clip.source
        a, b = clip.span(names)
        return clip.at(min(a + e, b)), clip.source

    def step(self, robot: dict | None, t: float):
        kind = task_kind(robot) if robot else None
        js = (robot or {}).get("job_state")
        key = (kind, ("hold" if js in SCRUB_HOLD else "run") if kind == "scrub" else js if kind == "pick_can" else None)
        if key != self.key:
            self.key, self.t0 = key, t
            self.blend_from, self.blend_t = self.out.copy(), t
        target, self.source = self._pose(kind, js, t) if kind else (self.park, None)
        u = smoothstep((t - self.blend_t) / BLEND_S)
        self.out = self.blend_from + (target - self.blend_from) * u
        return self.out
