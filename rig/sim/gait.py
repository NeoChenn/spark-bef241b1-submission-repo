import json
import math
from pathlib import Path

import numpy as np

MOTION_DIR = Path(__file__).resolve().parent / "motion"
LEGS = [f"{leg}_{j}_joint" for leg in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]
ARM = ["ur3_shoulder_pan_joint", "ur3_shoulder_lift_joint", "ur3_elbow_joint", "ur3_wrist_1_joint",
       "ur3_wrist_2_joint", "ur3_wrist_3_joint", "lift_joint"]
STAND = {"FL_hip_joint": 0.1, "RL_hip_joint": 0.1, "FR_hip_joint": -0.1, "RR_hip_joint": -0.1,
         "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
         "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5}
PARK = {"lift_joint": 0.0, "ur3_shoulder_pan_joint": 0.0, "ur3_shoulder_lift_joint": -1.57, "ur3_elbow_joint": 2.4,
        "ur3_wrist_1_joint": -2.4, "ur3_wrist_2_joint": -1.57, "ur3_wrist_3_joint": 0.0}
STAND_Z = 0.33
SPEED_TAU = 0.35
YAW_TAU = 0.2
TELEPORT_M = 3.0
RATE_RANGE = (0.6, 2.0)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def gait_weight(speed: float) -> float:
    return smoothstep((speed - 0.03) / 0.22)


def phase_rate(speed: float, v0: float, period: float) -> float:
    return min(max(speed / max(v0, 1e-6), RATE_RANGE[0]), RATE_RANGE[1]) / period


def _fold(q, hz, period, n):
    ph = (np.arange(len(q)) / hz / period) % 1.0
    b = np.minimum((ph * n).astype(int), n - 1)
    out = np.zeros((n, q.shape[1]))
    cnt = np.bincount(b, minlength=n)
    for j in range(q.shape[1]):
        out[:, j] = np.bincount(b, q[:, j], minlength=n) / np.maximum(cnt, 1)
    resid = q - out[b]
    return out, float((resid ** 2).mean()), cnt.min()


def fold_cycle(q, hz: float, pmin: float, pmax: float):
    q = np.asarray(q, np.float64)
    cands = []
    for p in np.arange(pmin, pmax, 0.25 / hz):
        n = max(int(round(p * hz)), 4)
        _, err, low = _fold(q, hz, p, n)
        if low:
            cands.append((p, n, err))
    emin = min(c[2] for c in cands)
    p, n, _ = next(c for c in cands if c[2] <= 1.5 * emin + 1e-9)
    near = [c for c in cands if abs(c[0] - p) < 0.1 * p]
    p, n, _ = min(near, key=lambda c: c[2])
    return float(p), _fold(q, hz, p, n)[0]


def gait_from_raw(raw: dict, skip_s: float = 2.0, pmin: float = 0.25, pmax: float = 1.0) -> dict:
    rows = [r for r in raw["rows"] if r["t"] >= skip_s]
    hz = float(raw["hz"])
    q = np.array([r["q"] for r in rows])
    p, loop = fold_cycle(q, hz, pmin, pmax)
    zl = _fold(np.array([[r["z"]] for r in rows]), hz, p, len(loop))[0]
    return {"source": raw.get("source", ""), "period": round(p, 4), "speed": round(float(np.mean([r["vx"] for r in rows])), 3),
            "joints": list(raw["joints"]), "hz": hz, "q": np.round(loop, 4).tolist(), "z": np.round(zl[:, 0], 4).tolist()}


class GaitLoop:
    def __init__(self, period, q, z, speed=0.8, source="", joints=None):
        self.period, self.speed, self.source = float(period), float(speed), source
        self.q = np.asarray(q, np.float64)
        self.z = np.asarray(z, np.float64)
        self.zmean = float(self.z.mean())
        joints = joints or LEGS
        self.q = self.q[:, [joints.index(j) for j in LEGS]]

    @classmethod
    def from_json(cls, d: dict) -> "GaitLoop":
        return cls(d["period"], d["q"], d["z"], d.get("speed", 0.8), d.get("source", ""), d.get("joints"))

    @classmethod
    def load(cls, path=None):
        p = Path(path) if path else MOTION_DIR / "gait_loop.json"
        return cls.from_json(json.loads(p.read_text())) if p.exists() else None

    def at(self, phase: float):
        n = len(self.q)
        x = (phase % 1.0) * n
        i = int(x) % n
        u = x - int(x)
        k = (i + 1) % n
        return self.q[i] * (1 - u) + self.q[k] * u, float(self.z[i] * (1 - u) + self.z[k] * u)


class Walker:
    def __init__(self, loop: GaitLoop, pose, t: float):
        self.loop = loop
        self.a = self.b = tuple(float(v) for v in pose)
        self.t0, self.dur, self.last = t, 1.0, t
        self.prev_t = t
        self.speed = 0.0
        self.phase = 0.0
        self.yaw = self.a[3]
        self.stand = np.array([STAND[j] for j in LEGS])

    def _pos(self, t):
        u = min(max((t - self.t0) / self.dur, 0.0), 1.0)
        return tuple(x + (y - x) * u for x, y in zip(self.a[:3], self.b[:3])), u

    def target(self, pose, t: float) -> None:
        pose = tuple(float(v) for v in pose)
        cur, _ = self._pos(t)
        if math.dist(cur, pose[:3]) > TELEPORT_M:
            self.a = self.b = pose
            self.yaw, self.speed = pose[3], 0.0
            self.t0, self.last = t, t
            return
        self.a = (*cur, self.yaw)
        self.b = pose
        self.dur = min(max(t - self.last, 0.2), 1.0)
        self.t0 = self.last = t

    def step(self, t: float):
        dt = max(t - self.prev_t, 0.0)
        self.prev_t = t
        pos, u = self._pos(t)
        raw = math.dist(self.a[:3], self.b[:3]) / self.dur if u < 1.0 else 0.0
        self.speed += (raw - self.speed) * (1 - math.exp(-dt / SPEED_TAU)) if dt else 0.0
        if self.speed < 1e-4:
            self.speed = 0.0
        self.yaw += wrap(self.b[3] - self.yaw) * (1 - math.exp(-dt / YAW_TAU)) if dt else 0.0
        self.yaw = wrap(self.yaw)
        if abs(wrap(self.b[3] - self.yaw)) < 1e-4:
            self.yaw = self.b[3]
        w = gait_weight(self.speed)
        if w == 0.0:
            return (*pos, self.yaw), self.stand.copy(), 0.0
        self.phase = (self.phase + phase_rate(self.speed, self.loop.speed, self.loop.period) * dt) % 1.0
        q, z = self.loop.at(self.phase)
        return (*pos, self.yaw), self.stand + (q - self.stand) * w, (z - self.loop.zmean) * w


class Quantiser:
    def __init__(self, step: float = 1e-3):
        self.step = step
        self.last: dict = {}
        self.writes = 0

    def changed(self, key, value) -> bool:
        v = tuple(round(float(x) / self.step) for x in np.atleast_1d(value))
        if self.last.get(key) == v:
            return False
        self.last[key] = v
        self.writes += 1
        return True
