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
TRUNK_Z = 42.82
ALIASES = {"retorque_done": "fixed", "gas_clear": "green", "hold": "step_back", "retreat": "step_back",
           "resume_rust": "resume", "unit2_arrive": "arrive", "duration_s": "end"}
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
    ((11.25, -10.28, 45.96), (26.25, -1.22, 46.75)),
    ((11.62, -10.28, 42.5), (11.88, -10.03, 46.5)),
    ((11.62, -1.47, 42.5), (11.88, -1.22, 46.5)),
    ((18.62, -1.47, 42.5), (18.88, -1.22, 46.5)),
    ((25.62, -10.28, 42.5), (25.88, -10.03, 46.5)),
    ((25.62, -1.47, 42.5), (25.88, -1.22, 46.5)),
    ((18.66, -11.69, 42.5), (18.84, -11.51, 53.3)),
)


@dataclass(frozen=True)
class Beats:
    start: float = 0.0
    push: float = 1.5
    amber: float = 6.75
    red: float = 8.55
    step_back: float = 10.0
    arrive: float = 27.0
    fixed: float = 34.0
    green: float = 40.2
    resume: float = 42.7
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
        return cls(start, g("push", max(start + 0.5, amber - 5.25)), amber, red, g("step_back", red + 1.5),
                   g("arrive", fixed - 7.0), fixed, green, resume, g("end", start + 60.0))


@dataclass(frozen=True)
class Scene:
    scrub: Vec3 = (18.2567, -5.655, 43.7199)
    unit1_work: Vec3 = (18.2567, -5.25, TRUNK_Z)
    unit1_back: Vec3 = (16.0, -4.5, TRUNK_Z)
    flange: Vec3 = (20.4602, -10.5044, 43.4)
    bolt: Vec3 = (20.5072, -10.5843, 43.3081)
    detector: Vec3 = (20.1533, -10.3958, 43.46)
    detector_id: str = "G-F5"
    unit2_work: Vec3 = (21.242, -10.172, TRUNK_Z)
    unit2_dock: Vec3 = (32.5, -9.6, TRUNK_Z)
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


def _fit(points, az: float, el: float, focal: float, margin: float = 90.0, lift: float = 0.0):
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
        "establish": _fit(_both(s.unit1_work, s.unit2_dock, s.flange), 75.0, 24.0, WIDE),
        "scrub": _close(_up(s.scrub, 0.02), 1.75, 28.0, 20.0, CLOSE),
        "detector": _close(s.detector, 3.0, 64.0, 25.0, CLOSE),
        "pull_back": _fit(both, 35.0, 20.0, WIDE),
        "gripper": _close(s.bolt, 1.7, 60.0, 32.0, CLOSE),
        "green": _close(tuple((np.asarray(s.detector) + np.asarray(s.bolt)) / 2), 2.6, 70.0, 25.0, MID),
        "wide": _fit(both, 120.0, 22.0, WIDE),
    }


def _move(t0: float, t1: float, cap: float = 4.0, frac: float = 0.55) -> float:
    return t0 + min(cap, frac * (t1 - t0))


def beat_spans(beats: Beats = DEFAULT_BEATS) -> list[tuple[str, float, float, float]]:
    b = beats
    scrub = _move(b.push, b.amber)
    plan = [
        ("establish", b.start, b.push),
        ("scrub", scrub, max(b.amber - 0.6, scrub + 0.8)),
        ("detector", min(b.amber + 1.4, b.red - 0.3), b.step_back),
        ("pull_back", _move(b.step_back, b.arrive), b.arrive),
        ("gripper", _move(b.arrive, b.fixed, cap=2.5), min(b.fixed + 1.5, b.green - 1.5)),
        ("green", b.green - 1.0, b.resume),
        ("wide", _move(b.resume, b.end, cap=5.0), b.end),
    ]
    out, t0 = [], b.start
    for name, arrive, end in plan:
        arrive = b.start if name == "establish" else min(max(arrive, t0 + 0.5), end - 0.1)
        out.append((name, t0, end, arrive))
        t0 = end
    return out


DRIFT = {"establish": (0.6, -0.4, -0.2), "scrub": (0.05, 0.06, 0.0), "detector": (0.12, 0.08, 0.0),
         "pull_back": (-0.4, -0.3, 0.0), "gripper": (0.05, 0.05, 0.0), "green": (0.1, 0.1, 0.05),
         "wide": (-0.8, 0.5, 0.3)}


def keyframes(beats: Beats = DEFAULT_BEATS, scene: Scene = DEFAULT_SCENE) -> list[tuple[float, Vec3, Vec3, float]]:
    sh = shots(scene)
    keys = []
    for name, _, end, arrive in beat_spans(beats):
        keys += [(arrive, sh[name]), (end, _drift(sh[name], *DRIFT[name]))]
    out = []
    for t, (eye, target, focal) in keys:
        if out and t <= out[-1][0]:
            out.pop()
        out.append((float(t), tuple(map(float, eye)), tuple(map(float, target)), float(focal)))
    return out


_CACHE: dict = {}


def _keys(beats: Beats, scene: Scene):
    k = (beats, scene)
    if k not in _CACHE:
        _CACHE.clear()
        _CACHE[k] = keyframes(beats, scene)
    return _CACHE[k]


def camera_at(t: float, beats: Beats = DEFAULT_BEATS, scene: Scene = DEFAULT_SCENE) -> tuple[Vec3, Vec3, float]:
    keys = _keys(beats, scene)
    if t <= keys[0][0]:
        return keys[0][1], keys[0][2], keys[0][3]
    for (t0, e0, g0, f0), (t1, e1, g1, f1) in zip(keys, keys[1:]):
        if t <= t1:
            s = ease((t - t0) / (t1 - t0))
            lerp = lambda a, b: tuple(x + (y - x) * s for x, y in zip(a, b))
            return lerp(e0, e1), lerp(g0, g1), f0 + (f1 - f0) * s
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
        xf.ClearXformOpOrder()
        xf.AddTransformOp()
    op = UsdGeom.Xformable(cam).GetOrderedXformOps()[0]
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
