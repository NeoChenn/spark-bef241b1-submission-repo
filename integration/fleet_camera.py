from __future__ import annotations

import math
from dataclasses import astuple, dataclass, fields

import numpy as np

from rig.camera import anchor, look_at, project
from rig.sim.orbit import look_at_rows
from rig.sim.zoom import ease, spherical

Vec3 = tuple[float, float, float]

APERTURE = 20.955
CLIP = (0.05, 4000.0)
WIDE, MID, CLOSE = 24.0, 32.0, 40.0
W, H = 1280, 720
LABEL_LIFT = 0.85
TANGENT = 0.75
TRUNK_Z = 42.82
ALIASES = {"retorque_done": "fixed", "gas_clear": "green", "hold": "step_back", "retreat": "step_back",
           "resume_rust": "resume", "unit2_arrive": "arrive", "duration_s": "end", "SPRAY": "spray",
           "PAINT": "paint", "spray_start": "spray", "paint_start": "paint"}
RIG_BOXES = (
    ((11.15, -5.9, 43.45), (26.35, -5.6, 43.75)),
    ((15.5, -10.82, 43.28), (35.0, -10.58, 43.52)),
    ((11.15, -8.85, 42.5), (26.35, -8.65, 43.3)),
    ((11.15, -9.38, 42.5), (26.35, -9.12, 43.27)),
    ((11.15, -2.72, 42.5), (26.35, -2.38, 43.47)),
    ((11.1, -5.97, 42.5), (11.2, -5.53, 43.82)),
    ((15.28, -6.1, 42.5), (15.8, -5.4, 43.82)),
    ((21.7, -6.1, 42.5), (22.35, -5.4, 43.82)),
    ((24.55, -6.1, 42.5), (24.95, -5.4, 43.45)),
    ((26.3, -5.97, 42.5), (26.4, -5.53, 43.82)),
    ((13.0, -6.44, 43.24), (14.5, -5.06, 43.96)),
    ((19.52, -11.0, 43.22), (20.53, -10.35, 43.58)),
    ((11.62, -10.28, 46.45), (25.88, -10.03, 46.75)),
    ((11.62, -1.47, 46.45), (25.88, -1.22, 46.75)),
    ((11.25, -7.44, 46.01), (26.25, -6.89, 46.19)),
    ((11.25, -4.55, 46.05), (26.25, -3.89, 46.25)),
    ((11.62, -10.28, 46.45), (11.88, -1.22, 46.75)),
    ((11.62, -10.28, 42.5), (11.88, -10.03, 46.5)),
    ((11.62, -1.47, 42.5), (11.88, -1.22, 46.5)),
    ((18.62, -10.28, 45.96), (18.88, -1.22, 46.75)),
    ((18.62, -1.47, 42.5), (18.88, -1.22, 46.5)),
    ((25.62, -10.28, 45.96), (25.88, -1.22, 46.75)),
    ((25.62, -10.28, 42.5), (25.88, -10.03, 46.5)),
    ((25.62, -1.47, 42.5), (25.88, -1.22, 46.5)),
    ((18.66, -11.69, 42.5), (18.84, -11.51, 53.3)),
    ((27.9, -7.9, 42.5), (34.7, -1.2, 47.0)),
    ((19.5, -6.5, 47.0), (33.3, -2.9, 51.3)),
)


@dataclass(frozen=True)
class Beats:
    start: float = 0.0
    push: float = 0.5
    amber: float = 6.75
    red: float = 8.55
    step_back: float = 10.0
    arrive: float = 27.0
    fixed: float = 34.0
    green: float = 40.2
    resume: float = 42.7
    spray: float = 45.0
    paint: float = 50.0
    end: float = 60.0

    def __post_init__(self):
        ts = astuple(self)
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError(f"beat times must increase: {ts}")
        if self.end - self.start > 60.0 + 1e-9:
            raise ValueError("the take must be 60 s or shorter")

    @classmethod
    def from_events(cls, events) -> "Beats":
        if isinstance(events, dict) and "script" in events:
            events = [*events["script"], {"name": "end", "t": events.get("duration_s")}]
        if isinstance(events, dict):
            pairs = list(events.items())
        else:
            pairs = [(e.get("beat") or e.get("event") or e.get("name"), e.get("t", e.get("time"))) for e in events]
        names, got = {f.name for f in fields(cls)}, {}
        for name, t in pairs:
            key = ALIASES.get(str(name), str(name).rsplit(" ", 1)[-1])
            if key in names and t is not None and key not in got:
                got[key] = float(t)
        d = cls()
        g = lambda k, v: got.get(k, v)
        start = g("start", d.start)
        amber = g("amber", d.amber)
        red = g("red", amber + 1.8)
        fixed = g("fixed", d.fixed)
        green = g("green", fixed + 6.2)
        resume = g("resume", green + 2.5)
        end = g("end", start + 60.0)
        paint = g("paint", end - 10.0)
        return cls(start, g("push", max(start + 0.5, amber - 6.25)), amber, red, g("step_back", red + 1.5),
                   g("arrive", fixed - 7.0), fixed, green, resume, g("spray", (resume + paint) / 2), paint, end)


@dataclass(frozen=True)
class Scene:
    scrub: Vec3 = (18.2567, -5.655, 43.7199)
    normal: Vec3 = (0.0, 0.6212, 0.7836)
    unit1_work: Vec3 = (18.2567, -5.25, TRUNK_Z)
    unit1_back: Vec3 = (17.4, -4.0, TRUNK_Z)
    flange: Vec3 = (20.4602, -10.5044, 43.4)
    bolt: Vec3 = (20.5072, -10.5843, 43.3081)
    detector: Vec3 = (20.1533, -10.3958, 43.46)
    detector_id: str = "G-F5"
    rack: tuple[float, float, float, float] = (11.75, 25.75, -10.15, -1.35)
    gate: Vec3 = (8.0, -4.4, 45.5)
    unit2_work: Vec3 = (21.01, -10.27, TRUNK_Z)
    unit2_dock: Vec3 = (29.6, -9.68, TRUNK_Z)
    deck_z: float = 42.5

    def obstacles(self) -> list[tuple[np.ndarray, np.ndarray]]:
        d, f = np.asarray(self.detector), np.asarray(self.flange)
        return [(np.asarray(lo), np.asarray(hi)) for lo, hi in RIG_BOXES] + [(d - 0.1, d + 0.15), (f - 0.2, f + 0.2)]


DEFAULT_BEATS = Beats()
DEFAULT_SCENE = Scene()


def _walk(t: float, t0: float, t1: float, a: Vec3, b: Vec3) -> Vec3:
    s = ease((t - t0) / (t1 - t0))
    return tuple(x + (y - x) * s for x, y in zip(a, b))


def robots_at(t: float, beats: Beats = DEFAULT_BEATS, scene: Scene = DEFAULT_SCENE) -> dict[str, Vec3]:
    b, s = beats, scene
    if t < b.resume:
        u1 = _walk(t, b.step_back + 0.3, b.step_back + 3.3, s.unit1_work, s.unit1_back)
    else:
        u1 = _walk(t, b.resume, b.resume + 3.0, s.unit1_back, s.unit1_work)
    if t < b.resume:
        u2 = _walk(t, min(b.step_back + 5.0, b.arrive - 4.0), b.arrive, s.unit2_dock, s.unit2_work)
    else:
        u2 = _walk(t, b.resume, min(b.resume + 12.0, b.end), s.unit2_work, s.unit2_dock)
    return {"unit1": u1, "unit2": u2}


def _fit(points, az: float, el: float, focal: float, margin: float = 60.0):
    pts = np.asarray(points, float)
    target = tuple((pts.min(0) + pts.max(0)) / 2)
    lo, hi = 0.5, 80.0
    for _ in range(40):
        d = (lo + hi) / 2
        cam = look_at(spherical(target, d, az, el), target, focal, APERTURE, W, H)
        uv = [project(cam, p) for p in pts]
        ok = all(z > 0 and margin <= u <= W - margin and margin <= v <= H - margin for u, v, z in uv)
        lo, hi = (lo, d) if ok else (d, hi)
    return spherical(target, hi, az, el), target, focal


def _up(p: Vec3, dz: float) -> Vec3:
    return (p[0], p[1], p[2] + dz)


def _both(*pts: Vec3) -> list[Vec3]:
    return [q for p in pts for q in (p, _up(p, LABEL_LIFT))]


def _drift(shot, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
    eye, target, focal = shot
    return (eye[0] + dx, eye[1] + dy, eye[2] + dz), target, focal


def _close(target: Vec3, d: float, az: float, el: float, focal: float):
    return spherical(target, d, az, el), target, focal


def shots(scene: Scene = DEFAULT_SCENE) -> dict[str, tuple]:
    s = scene
    both = _both(s.unit1_work, s.unit1_back, s.unit2_dock, s.unit2_work)
    return {
        "establish": _fit(_both(s.unit1_work, s.unit2_dock, s.flange), 172.0, 28.0, WIDE),
        "scrub": _close(_up(s.scrub, 0.02), 1.75, 150.0, 20.0, CLOSE),
        "detector": _close(s.detector, 3.0, 64.0, 25.0, CLOSE),
        "pull_back": _fit(both, 180.0, 26.0, WIDE),
        "gripper": _close(s.bolt, 1.7, 60.0, 32.0, CLOSE),
        "spray": _close(tuple(np.asarray(s.scrub) + 0.18 * np.asarray(s.normal)), 2.0, 55.0, 24.0, 35.0),
        "paint": _close(_up(s.scrub, 0.02), 1.6, 100.0, 24.0, CLOSE),
        "green": _close(tuple((np.asarray(s.detector) + np.asarray(s.bolt)) / 2), 2.6, 70.0, 25.0, MID),
        "wide": _fit(both, 185.0, 26.0, WIDE),
    }


def _move(t0: float, t1: float, cap: float = 4.0, frac: float = 0.55) -> float:
    return t0 + min(cap, frac * (t1 - t0))


def beat_spans(beats: Beats = DEFAULT_BEATS) -> list[tuple[str, float, float, float]]:
    b = beats
    plan = [
        ("establish", b.start, b.push),
        ("scrub", b.amber - 1.4, b.amber - 0.6),
        ("detector", min(b.amber + 1.4, b.red - 0.3), b.step_back),
        ("pull_back", _move(b.step_back, b.arrive, cap=6.0), max(b.arrive - 4.0, b.step_back + 4.0)),
        ("gripper", _move(b.arrive - 4.0, b.fixed, cap=6.0), min(b.fixed + 1.5, b.green - 1.5)),
        ("green", b.green - 1.0, b.resume),
        ("spray", b.spray, b.paint - 2.0),
        ("paint", b.paint, min(b.paint + 3.0, b.end - 7.0)),
        ("wide", _move(min(b.paint + 3.0, b.end - 7.0), b.end, cap=6.0, frac=0.85), b.end),
    ]
    out, t0 = [], b.start
    for name, arrive, end in plan:
        arrive = b.start if name == "establish" else min(max(arrive, t0 + 0.5), end - 0.1)
        out.append((name, t0, end, arrive))
        t0 = end
    return out


DRIFT = {"establish": (0.6, -0.4, -0.2), "scrub": (0.05, 0.06, 0.0), "detector": (0.12, 0.08, 0.0),
         "pull_back": (-0.6, -0.3, -0.2), "gripper": (0.05, 0.05, 0.0), "green": (0.1, 0.1, 0.05),
         "spray": (0.08, 0.05, 0.0), "paint": (-0.06, 0.05, 0.0), "wide": (-0.8, 0.5, 0.3)}


def inside(eye, scene: Scene = DEFAULT_SCENE) -> bool:
    x0, x1, y0, y1 = scene.rack
    return x0 < eye[0] < x1 and y0 < eye[1] < y1 and eye[2] < 46.0


def _gate(a: Vec3, b: Vec3, scene: Scene):
    return None if inside(a, scene) == inside(b, scene) else scene.gate


def keyframes(beats: Beats = DEFAULT_BEATS, scene: Scene = DEFAULT_SCENE) -> list[tuple]:
    sh = shots(scene)
    keys = []
    for name, _, end, arrive in beat_spans(beats):
        keys += [(arrive, sh[name]), (end, _drift(sh[name], *DRIFT[name]))]
    out = []
    for t, (eye, target, focal) in keys:
        if out and t <= out[-1][0]:
            out.pop()
        eye, target = tuple(map(float, eye)), tuple(map(float, target))
        via = _gate(out[-1][1], eye, scene) if out else None
        out.append((float(t), eye, target, float(focal), via))
    return out


_CACHE: dict = {}


def _keys(beats: Beats, scene: Scene):
    k = (beats, scene)
    if k not in _CACHE:
        _CACHE.clear()
        _CACHE[k] = keyframes(beats, scene)
    return _CACHE[k]


def _hermite(p0, m0, p1, m1, u: float):
    u2, u3 = u * u, u * u * u
    return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1


def _path(a, b, via, s: float, scene: Scene):
    a, b = np.asarray(a), np.asarray(b)
    if via is None:
        return a + (b - a) * s
    v, z = np.asarray(via), np.zeros(3)
    d = (b - v) if inside(b, scene) else (v - a)
    d = d / np.linalg.norm(d)
    m = TANGENT * float((b - a) @ d) * d
    return _hermite(a, z, v, m, 2 * s) if s < 0.5 else _hermite(v, m, b, z, 2 * s - 1)


def camera_at(t: float, beats: Beats = DEFAULT_BEATS, scene: Scene = DEFAULT_SCENE) -> tuple[Vec3, Vec3, float]:
    keys = _keys(beats, scene)
    if t <= keys[0][0]:
        return keys[0][1], keys[0][2], keys[0][3]
    for (t0, e0, g0, f0, _), (t1, e1, g1, f1, via) in zip(keys, keys[1:]):
        if t <= t1:
            s = ease((t - t0) / (t1 - t0))
            eye = tuple(float(c) for c in _path(e0, e1, via, s, scene))
            return eye, tuple(x + (y - x) * s for x, y in zip(g0, g1)), f0 + (f1 - f0) * s
    return keys[-1][1], keys[-1][2], keys[-1][3]


def beat_at(t: float, beats: Beats = DEFAULT_BEATS) -> str:
    for name, _, end, _ in beat_spans(beats):
        if t < end:
            return name
    return "wide"


def view(t: float, beats: Beats = DEFAULT_BEATS, scene: Scene = DEFAULT_SCENE, w: int = W, h: int = H) -> dict:
    eye, target, focal = camera_at(t, beats, scene)
    return look_at(eye, target, focal, APERTURE, w, h)


def project_points(t: float, points: dict, beats: Beats = DEFAULT_BEATS, scene: Scene = DEFAULT_SCENE,
                   w: int = W, h: int = H, lift: float = LABEL_LIFT, margin: float = 24.0) -> dict[str, dict]:
    cam = view(t, beats, scene, w, h)
    return {k: anchor(cam, p, lift=lift, margin=margin) for k, p in points.items()}


def labels(t: float, beats: Beats = DEFAULT_BEATS, scene: Scene = DEFAULT_SCENE, positions: dict | None = None,
           w: int = W, h: int = H) -> dict[str, dict]:
    pts = dict(positions or robots_at(t, beats, scene))
    out = project_points(t, pts, beats, scene, w, h)
    out.update(project_points(t, {scene.detector_id: scene.detector}, beats, scene, w, h, lift=0.12))
    return out


def usd_matrix(eye: Vec3, target: Vec3) -> tuple:
    return look_at_rows(tuple(eye), tuple(target))


def apply_usd(stage, path: str, t: float, beats: Beats = DEFAULT_BEATS, scene: Scene = DEFAULT_SCENE, time=None):
    from pxr import Gf, Usd, UsdGeom

    cam = UsdGeom.Camera.Get(stage, path)
    if not cam:
        cam = UsdGeom.Camera.Define(stage, path)
        cam.CreateHorizontalApertureAttr(APERTURE)
        cam.CreateVerticalApertureAttr(APERTURE * H / W)
        cam.CreateClippingRangeAttr(Gf.Vec2f(*CLIP))
    xf = UsdGeom.Xformable(cam)
    ops = xf.GetOrderedXformOps()
    if len(ops) != 1 or ops[0].GetOpType() != UsdGeom.XformOp.TypeTransform:
        xf.ClearXformOpOrder()
        ops = [xf.AddTransformOp()]
    op = ops[0]
    code = Usd.TimeCode.Default() if time is None else Usd.TimeCode(time)
    eye, target, focal = camera_at(t, beats, scene)
    op.Set(Gf.Matrix4d(usd_matrix(eye, target)), code)
    cam.GetFocalLengthAttr().Set(float(focal), code)
    return cam


def author_usd(stage, path: str, beats: Beats = DEFAULT_BEATS, scene: Scene = DEFAULT_SCENE, fps: float = 30.0):
    n = int(math.floor((beats.end - beats.start) * fps)) + 1
    for i in range(n):
        apply_usd(stage, path, beats.start + i / fps, beats, scene, time=float(i))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(n - 1)
    stage.SetTimeCodesPerSecond(fps)
    return n
