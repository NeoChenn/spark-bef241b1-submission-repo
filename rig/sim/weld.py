import json
import math

import numpy as np

from rig.sim.gait import ARM, MOTION_DIR

CLIP = "weld_clip"
LIFT_Z = 0.37
PLAN = {"frames": 390, "fps": 30, "weld": (20, 330)}
AGE_EDGES = (0.3, 0.8, 1.6, 2.6, 3.6)
HOT = (((1.0, 0.88, 0.6), 1.0), ((1.0, 0.6, 0.2), 0.55), ((1.0, 0.34, 0.07), 0.28), ((0.9, 0.14, 0.02), 0.11),
       ((0.55, 0.05, 0.01), 0.035))
COOLED = (0.62, 0.58, 0.52)
ARC_RGB = (0.75, 0.85, 1.0)
POOL_RGB = (1.0, 0.45, 0.12)
OFF = (0.0, 0.0, -50.0)
SPARKS, TRAIL, TRAIL_S = 30, 3, 0.012
SPARK_LIFE, SPARK_V, SPARK_R = (0.2, 0.6), (1.2, 3.6), 0.0035
GRAVITY = np.array([0.0, 0.0, -9.81])
CRACK_R, CRACK_STEP = 0.0022, 0.0015
LEAD_S = 1.0


def _hash(i, seed):
    x = (np.asarray(i, np.int64) * 374761393 + int(seed) * 668265263) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((x ^ (x >> 16)) & 0xFFFFFF) / float(0xFFFFFF)


def _noise(t, rate, seed):
    x = t * rate
    i = math.floor(x)
    f = x - i
    f = f * f * (3 - 2 * f)
    return float(_hash(i, seed) * (1 - f) + _hash(i + 1, seed) * f)


def arc_flicker(t: float, seed: int = 0) -> float:
    v = 0.72 + _noise(t, 47.0, seed) * 0.45 + _noise(t, 113.0, seed + 1) * 0.25 + (_noise(t, 6.0, seed + 2) - 0.5) * 0.2
    gate = _noise(t, 9.0, seed + 3)
    if gate > 0.86:
        v *= 1.0 - min((gate - 0.86) / 0.14, 1.0) * 0.35
    return float(np.clip(v, 0.6, 1.3))


def densify(pts, step: float) -> np.ndarray:
    p = np.asarray(pts, float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    u = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(u, s, p[:, k]) for k in range(3)], 1)


def export(path=None) -> dict:
    from policy.weld.clip import plan

    d = plan(**PLAN)
    last = d["frames"][-1]
    fps, n = d["fps"], d["n_frames"]
    up = np.array([0.0, 0.0, LIFT_Z])
    bead = np.asarray(last["bead_world"]) + up
    bead_t = (n - 1) / fps - np.asarray(last["bead_age_s"])
    crack = densify(np.asarray(d["crack_world"]) + up, CRACK_STEP)
    crack = crack[(crack[:, 0] >= bead[:, 0].min()) & (crack[:, 0] <= bead[:, 0].max())]
    q = [[f["q"][j] + (LIFT_Z if j == "lift_joint" else 0.0) for j in ARM] for f in d["frames"]]
    r = lambda a: np.round(np.asarray(a, float), 5).tolist()
    out = {"source": "policy.weld.clip.plan: SeamTracker closed loop on the laser profile, UR3 IK per frame",
           "fps": fps, "n": n, "weld": d["weld"], "lift_z": LIFT_Z, "q": r(q),
           "tip": r(np.asarray([f["tip"] for f in d["frames"]]) + up),
           "pipe": {"origin": r(np.asarray(d["pipe"]["origin"]) + up), "axis": d["pipe"]["axis"], "radius": d["pipe"]["radius"]},
           "bead": r(bead), "bead_w": r(last["bead_width_m"]), "bead_t": r(bead_t), "crack": r(crack),
           "metrics": {k: round(d["metrics"][k], 4) for k in ("tracker_rmse_mm", "open_loop_rmse_mm", "ik_max_err_mm_weld",
                                                               "weld_length_mm")}}
    (path or MOTION_DIR / f"{CLIP}.json").write_text(json.dumps(out, separators=(",", ":")))
    return out


class WeldClip:
    def __init__(self, d: dict):
        self.fps, self.n, self.source = float(d["fps"]), int(d["n"]), d.get("source", "")
        self.on, self.off = (w / self.fps for w in d["weld"])
        self.q, self.tip = np.asarray(d["q"], float), np.asarray(d["tip"], float)
        self.origin, self.axis = np.asarray(d["pipe"]["origin"], float), np.asarray(d["pipe"]["axis"], float)
        self.radius = float(d["pipe"]["radius"])
        self.bead, self.bead_w, self.bead_t = (np.asarray(d[k], float) for k in ("bead", "bead_w", "bead_t"))
        self.crack = np.asarray(d["crack"], float)
        self.metrics = d.get("metrics", {})

    @classmethod
    def load(cls, name: str = CLIP):
        p = MOTION_DIR / f"{name}.json"
        return cls(json.loads(p.read_text())) if p.exists() else None

    @property
    def duration(self) -> float:
        return (self.n - 1) / self.fps

    def _lerp(self, a, c: float):
        x = min(max(c * self.fps, 0.0), self.n - 1.0)
        i = int(x)
        k = min(i + 1, self.n - 1)
        return a[i] + (a[k] - a[i]) * (x - i)

    def arm(self, c: float) -> np.ndarray:
        return self._lerp(self.q, c)

    def tip_at(self, c) -> np.ndarray:
        x = np.clip(np.asarray(c, float) * self.fps, 0.0, self.n - 1.0)
        return np.stack([np.interp(x, np.arange(self.n), self.tip[:, k]) for k in range(3)], -1)

    def arc(self, c) -> np.ndarray:
        c = np.asarray(c, float)
        return (c >= self.on) & (c <= self.off)

    def normals(self, p) -> np.ndarray:
        d = np.asarray(p, float) - self.origin
        r = d - np.outer(d @ self.axis, self.axis)
        return r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-9)


def to_world(p, at) -> np.ndarray:
    x, y, z, yaw = at
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(p, float)
    return np.stack([x + c * p[..., 0] - s * p[..., 1], y + s * p[..., 0] + c * p[..., 1], z + p[..., 2]], -1)


def rotate(v, yaw: float) -> np.ndarray:
    return to_world(v, (0.0, 0.0, 0.0, yaw))


def bead_state(clip: WeldClip, c: float):
    shown = clip.bead_t <= c
    return shown, np.searchsorted(AGE_EDGES, np.maximum(c - clip.bead_t, 0.0))


def sparks(clip: WeldClip, c: float, floor: float = -1e9):
    i = np.arange(SPARKS)
    life = SPARK_LIFE[0] + (SPARK_LIFE[1] - SPARK_LIFE[0]) * _hash(i, 3)
    age = (c + _hash(i, 5) * life) % life
    born = c - age
    alive = clip.arc(born) & (born >= 0)
    p0 = clip.tip_at(born)
    nrm = clip.normals(p0)
    side = np.cross(nrm, clip.axis)
    a = 2 * math.pi * _hash(i, 7)
    spread = 0.35 + 0.75 * _hash(i, 9)
    d = nrm + spread[:, None] * (np.cos(a)[:, None] * side + np.sin(a)[:, None] * clip.axis) - 0.35 * clip.axis
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    v = d * (SPARK_V[0] + (SPARK_V[1] - SPARK_V[0]) * _hash(i, 11))[:, None]
    pos = []
    for k in range(TRAIL):
        t = np.maximum(age - k * TRAIL_S, 0.0)[:, None]
        p = p0 + nrm * 0.004 + v * t + 0.5 * GRAVITY * t * t
        p[:, 2] = np.maximum(p[:, 2], floor)
        pos.append(p)
    return np.concatenate(pos), np.tile(age / life < 0.45, TRAIL), np.tile(alive, TRAIL)


def spark_scales() -> np.ndarray:
    return np.concatenate([np.full(SPARKS, SPARK_R * (1.0 - 0.3 * k)) for k in range(TRAIL)])
