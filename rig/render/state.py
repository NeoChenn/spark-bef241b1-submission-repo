import json
import math

import numpy as np

SCENE_KEYS = ("type", "seq", "t", "ts", "scene", "view", "specs", "attrs", "key")
HEAD_KEYS = ("seq", "t", "ts", "scene", "view")
LERP_TYPES = ("float", "double", "half", "float2", "float3", "float4", "double2", "double3", "double4", "half3",
              "color3f", "vector3f", "point3f", "normal3f", "vector3d", "point3d")
QUAT_TYPES = ("quatf", "quatd", "quath")


def under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


class StateLog:
    def __init__(self) -> None:
        self.head: dict = {}
        self.specs: dict[str, str | None] = {}
        self.attrs: dict[tuple[str, str], tuple[str, object]] = {}

    def merge(self, msg: dict) -> "StateLog":
        self.head.update({k: msg[k] for k in HEAD_KEYS if k in msg})
        for path, usda in msg.get("specs") or ():
            self.attrs = {k: v for k, v in self.attrs.items() if not under(k[0], path)}
            self.specs.pop(path, None)
            self.specs[path] = usda
        for path, name, kind, value in msg.get("attrs") or ():
            self.attrs.pop((path, name), None)
            self.attrs[(path, name)] = (kind, value)
        return self

    def message(self, key: bool = False) -> dict:
        return {"type": "state", **self.head, "key": key, "specs": [[p, u] for p, u in self.specs.items()],
                "attrs": [[p, n, k, v] for (p, n), (k, v) in self.attrs.items()]}

    def take(self) -> dict:
        msg = self.message()
        self.specs, self.attrs = {}, {}
        return msg

    def empty(self) -> bool:
        return not self.specs and not self.attrs


def merge_messages(old: dict, new: dict) -> dict:
    log = StateLog().merge(old).merge(new)
    out = {**old, **new, **log.message(bool(old.get("key") or new.get("key")))}
    events = list(old.get("events") or []) + list(new.get("events") or [])
    if events:
        out["events"] = events
    return out


def split(header: dict) -> tuple[dict, dict]:
    scene = {k: header[k] for k in SCENE_KEYS if k in header}
    meta = {k: v for k, v in header.items() if k not in SCENE_KEYS}
    if "t" in header:
        meta["t"] = header["t"]
    return scene, meta


def _slerp(a, b, u: float) -> list:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + u * (b - a)
    else:
        th = math.acos(min(d, 1.0))
        q = (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / math.sin(th)
    return (q / np.linalg.norm(q)).tolist()


def blend_value(kind: str, a, b, u: float):
    if a is None or type(a) is not type(b):
        return b
    if kind in QUAT_TYPES:
        return _slerp(a, b, u)
    if kind in LERP_TYPES or kind == "matrix4d":
        if isinstance(a, bool) or isinstance(b, bool):
            return b
        return (np.asarray(a, np.float64) * (1 - u) + np.asarray(b, np.float64) * u).tolist() if isinstance(a, list) \
            else a * (1 - u) + b * u
    return b


def blend_view(a: dict | None, b: dict, u: float) -> dict:
    if not a:
        return b
    ma, mb = np.asarray(a["m"], np.float64).reshape(4, 4), np.asarray(b["m"], np.float64).reshape(4, 4)
    out = mb.copy()
    out[3, :3] = ma[3, :3] * (1 - u) + mb[3, :3] * u
    qa, qb = _mat_quat(ma[:3, :3]), _mat_quat(mb[:3, :3])
    out[:3, :3] = _quat_mat(_slerp(qa, qb, u))
    return {**b, "m": out.reshape(-1).tolist(), "focal": a["focal"] * (1 - u) + b["focal"] * u}


def _mat_quat(m) -> list:
    m = np.asarray(m, np.float64).T
    m = m / np.linalg.norm(m, axis=0, keepdims=True)
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    return [w, x, y, z]


def _quat_mat(q) -> np.ndarray:
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return r.T


def interpolate(a: dict, b: dict, u: float) -> dict:
    prev = {(p, n): v for p, n, _, v in a.get("attrs") or ()}
    attrs = [[p, n, k, blend_value(k, prev.get((p, n)), v, u)] for p, n, k, v in b.get("attrs") or ()]
    out = {**b, "attrs": attrs, "t": a["t"] * (1 - u) + b["t"] * u}
    if b.get("view"):
        out["view"] = blend_view(a.get("view"), b["view"], u)
    return out


class Inbox:
    def __init__(self, record=None) -> None:
        import threading

        self._lock = threading.Lock()
        self.ready = threading.Event()
        self.log = StateLog()
        self.received = 0
        self._file = open(record, "a") if record else None

    def put(self, header: dict, payload: bytes = b"") -> None:
        if header.get("type") != "state":
            return
        with self._lock:
            self.log.merge(header)
            self.received += 1
            if self._file is not None:
                self._file.write(json.dumps(header, separators=(",", ":")) + "\n")
        self.ready.set()

    def take(self) -> dict | None:
        with self._lock:
            if not self.ready.is_set():
                return None
            self.ready.clear()
            return self.log.take()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def replay_plan(log: list[dict], fps: float = 0.0):
    full, cum = [], StateLog()
    for record in log:
        full.append({**cum.merge(record).message(True), "specs": []})
    if not fps:
        for i, record in enumerate(log):
            yield [record], full[i]
        return
    done = -1
    for b, u in resample([r["t"] for r in log], fps):
        upto = b if u > 0 or b == 0 else b - 1
        deltas = log[done + 1:upto + 1]
        done = max(done, upto)
        yield deltas, full[b] if u >= 1 or b == 0 else interpolate(full[b - 1], full[b], u)


def resample(times: list[float], fps: float) -> list[tuple[int, float]]:
    if not times:
        return []
    out, i, t = [], 0, times[0]
    while t <= times[-1] + 1e-9:
        while i + 1 < len(times) and times[i + 1] <= t:
            i += 1
        if i + 1 >= len(times):
            out.append((i, 1.0))
        else:
            span = times[i + 1] - times[i]
            out.append((i + 1, (t - times[i]) / span if span > 0 else 1.0))
        t += 1.0 / fps
    return out
