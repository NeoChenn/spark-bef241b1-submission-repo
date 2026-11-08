from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from integration.fleet_camera import APERTURE, CLIP, H, W, _body, _fit, usd_matrix
from integration.fleet_paths import DOCK
from rig.camera import look_at, project
from rig.sim.zoom import ease, spherical

Vec3 = tuple[float, float, float]
TRUNK_Z = 42.82
RIG_CENTER = (5.0, 0.0, 44.0)
CONTROL_ROOM = {"prim": "/World/Rig/module_n83_40_module_white", "windows": "/World/Details/windows_module_n83_40",
                "box": ((-21.0, 10.1, 31.2), (-11.4, 21.1, 41.2)),
                "interior": ((-21.0, 10.1, 38.2), (-11.4, 21.1, 41.2)), "floor_z": 38.2, "ceiling_z": 41.2,
                "window_y": 21.1, "window_rows_z": ((32.75, 33.9), (35.95, 37.1), (39.15, 40.3)),
                "north_windows": (((-19.8, 21.08, 39.15), (-18.2, 21.08, 40.3)),
                                  ((-16.8, 21.08, 39.15), (-15.2, 21.08, 40.3))),
                "interior_eye": (-16.2, 12.6, 39.7), "window_target": (-16.2, 21.1, 39.5),
                "windows_center": (-17.5, 21.1, 39.7), "push_end": (-17.5, 30.0, 40.6)}


@dataclass(frozen=True)
class FilmScene:
    unit1_dock: Vec3 = (12.2, -7.5, TRUNK_Z)
    unit1_work: Vec3 = (18.225, -5.262, TRUNK_Z)
    patch: Vec3 = (18.2567, -5.655, 43.7199)
    patch_normal: Vec3 = (0.0, 0.6212, 0.7836)
    unit2_dock: Vec3 = (*DOCK["unit2"][:2], TRUNK_Z)
    unit2_work: Vec3 = (21.012, -10.27, TRUNK_Z)
    bolt: Vec3 = (20.5072, -10.5843, 43.3081)
    flange: Vec3 = (20.4602, -10.5044, 43.4)


DEFAULT_FILM_SCENE = FilmScene()


def _lerp(a, b, s: float) -> Vec3:
    return tuple(float(x + (y - x) * s) for x, y in zip(a, b))


def _up(p, dz: float) -> Vec3:
    return (float(p[0]), float(p[1]), float(p[2] + dz))


ORBIT = (-62.0, 18.0, 23.0, 26.0)


RIG_DECK = [(x, y, z) for x in (-37.0, 37.5) for y in (-39.0, 27.4) for z in (18.0, 42.5)]
RIG_TOPS = [(-62.6, 8.0, 82.5), (-25.6, 8.0, 41.6), (15.0, 15.0, 81.3), (25.0, 25.0, 81.3), (15.0, 25.0, 81.3),
            (25.0, 15.0, 81.3)]
RIG_KEEP = [p for p in RIG_DECK if p[2] > 40.0] + RIG_TOPS
ORBIT_TARGET = (1.0, 4.0, 52.0)
EDGE = 0.05


def width_fraction(eye, focal: float, pts=RIG_DECK, target=ORBIT_TARGET) -> float:
    cam = look_at(eye, target, focal, APERTURE, W, H)
    us = [project(cam, p)[0] for p in pts]
    return (max(us) - min(us)) / W


def inside_edges(eye, target, focal: float, pts, edge: float = EDGE) -> bool:
    cam = look_at(eye, target, focal, APERTURE, W, H)
    return all(z > 0 and edge * W <= u <= (1 - edge) * W and edge * H <= v <= (1 - edge) * H
               for u, v, z in (project(cam, p) for p in pts))


def rig_radius(az0: float, sweep: float, el: float, focal: float) -> float:
    lo, hi = 20.0, 900.0
    for _ in range(40):
        d = (lo + hi) / 2
        ok = all(inside_edges(spherical(ORBIT_TARGET, d, az0 + sweep * k / 16, el), ORBIT_TARGET, focal, RIG_KEEP)
                 for k in range(17))
        lo, hi = (lo, d) if ok else (d, hi)
    return hi


RIG_RADIUS = rig_radius(*ORBIT)


def rig_orbit(u: float) -> tuple[Vec3, Vec3, float]:
    az0, sweep, el, focal = ORBIT
    return spherical(ORBIT_TARGET, RIG_RADIUS, az0 + sweep * ease(u), el), ORBIT_TARGET, focal


def control_push(u: float) -> tuple[Vec3, Vec3, float]:
    s = ease(u)
    eye = _lerp((-13.5, 45.0, 43.5), CONTROL_ROOM["push_end"], s)
    return eye, _lerp((-15.0, 18.0, 38.0), CONTROL_ROOM["windows_center"], s), 26.0 + 2.0 * s


def control_room(u: float) -> tuple[Vec3, Vec3, float]:
    eye = CONTROL_ROOM["interior_eye"]
    return _lerp(eye, (eye[0] + 0.6, eye[1] + 0.3, eye[2]), u), CONTROL_ROOM["window_target"], 22.0


def follow(robot: Vec3, eye_offset: Vec3, focal: float) -> tuple[Vec3, Vec3, float]:
    target = _up(robot, 0.25)
    return tuple(float(t + o) for t, o in zip(target, eye_offset)), target, focal


def unit2_job(scene: FilmScene = DEFAULT_FILM_SCENE) -> tuple[Vec3, Vec3, float]:
    s = scene
    pts = [s.bolt, s.flange, _up(s.unit2_work, 0.3), _up(s.unit2_work, 1.1)]
    return _fit(pts, 40.0, 18.0, 40.0, margin=70.0, target=_lerp(s.bolt, _up(s.unit2_work, 0.6), 0.3))


def unit1_job(scene: FilmScene = DEFAULT_FILM_SCENE) -> tuple[Vec3, Vec3, float]:
    s = scene
    n = np.asarray(s.patch_normal)
    w = s.unit1_work
    pts = _body(w) + [(w[0] + 0.35, w[1], w[2] + 0.2), _up(w, 1.1), s.patch, tuple(np.asarray(s.patch) + 0.2 * n)]
    return _fit(pts, 45.0, 22.0, 21.0, margin=42.0, target=_lerp(s.patch, (w[0], w[1], 42.9), 0.5))


WALK_HOME = {"unit1": (-75.0, 55.0, 8.5, 24.0), "unit2": (-135.0, 55.0, 8.5, 24.0)}


def walk_home(robot: str, p: Vec3) -> tuple[Vec3, Vec3, float]:
    az, el, d, focal = WALK_HOME[robot]
    target = (p[0], p[1], 42.9)
    return spherical(target, d, az, el), target, focal


def conclusion(u: float) -> tuple[Vec3, Vec3, float]:
    return spherical(RIG_CENTER, RIG_RADIUS * 1.15, 120.0 + 6.0 * u, 14.0), RIG_CENTER, 24.0


def view(eye, target, focal, w: int = W, h: int = H) -> dict:
    return look_at(eye, target, focal, APERTURE, w, h)


def apply_usd_pose(stage, path: str, eye, target, focal: float, time=None):
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
    code = Usd.TimeCode.Default() if time is None else Usd.TimeCode(time)
    ops[0].Set(Gf.Matrix4d(usd_matrix(eye, target)), code)
    cam.GetFocalLengthAttr().Set(float(focal), code)
    return cam


def bodies(*robots: Vec3) -> list[Vec3]:
    return [p for r in robots for p in _body(r)]


def dist(a, b) -> float:
    return float(math.dist(a, b))
