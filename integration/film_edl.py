from __future__ import annotations

import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from integration.film_camera import (DEFAULT_FILM_SCENE, TRUNK_Z, FilmScene, birdseye, conclusion, control_push,
                                     control_room, follow, rig_orbit, unit1_job, unit2_job)
from rig.sim.zoom import ease

REPO = Path(__file__).resolve().parent.parent
FILM_RUN = REPO / "runs_sample/film"
FOLLOW_FOCAL = 33.0
MOVE_S = 1.2
LENGTH = 60.0
CUTS = {"short": (60.0, 3), "long": (66.0, 5)}
ALIASES = {"u1_walk": "dispatch_u1", "end": "t_end"}


@dataclass(frozen=True)
class FilmBeats:
    leak_start: float = 4.0
    alarm: float = 6.6
    dispatch_u2: float = 18.3
    u2_arrive: float = 40.5
    u2_engage: float = 46.8
    torque_done: float = 50.2
    gf5_green: float = 54.2
    u2_return: float = 54.7
    u1_walk: float = 63.2
    u1_arrive: float = 76.9
    spray: float = 79.2
    soak: float = 82.3
    scrub: float = 88.0
    scrub_done: float = 107.9
    paint: float = 109.5
    paint_done: float = 113.6
    u1_return: float = 114.8
    both_docked: float = 129.6
    end: float = 129.7

    @classmethod
    def from_summary(cls, summary: dict) -> "FilmBeats":
        beats, d = summary.get("beats", summary), cls()
        vals = {}
        for f in fields(cls):
            key = ALIASES.get(f.name, f.name)
            v = summary.get(key) if key == "t_end" else beats.get(key, beats.get(f.name))
            vals[f.name] = float(v) if v is not None else getattr(d, f.name)
        return cls(**vals)

    @classmethod
    def load(cls, path=FILM_RUN / "summary.json") -> "FilmBeats":
        return cls.from_summary(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class Shot:
    name: str
    film: tuple[float, float]
    sim: Optional[tuple[float, float]]
    rate: float
    camera: Callable
    t_in: str = "cut"
    t_out: str = "cut"
    two_d: bool = False
    badge: str = ""
    cards: int = 0

    def sim_at(self, film_t: float) -> Optional[float]:
        if self.sim is None:
            return None
        return min(self.sim[0] + (film_t - self.film[0]) * self.rate, self.sim[1])

    def to_dict(self) -> dict:
        return {"name": self.name, "film": [round(v, 3) for v in self.film],
                "sim": None if self.sim is None else [round(v, 3) for v in self.sim], "rate": round(self.rate, 3),
                "in": self.t_in, "out": self.t_out, "two_d": self.two_d, "badge": self.badge, "cards": self.cards}


def _walk(t: float, t0: float, t1: float, a, b):
    s = ease((t - t0) / max(t1 - t0, 1e-6))
    return tuple(float(x + (y - x) * s) for x, y in zip(a, b))


def default_track(b: FilmBeats, s: FilmScene) -> Callable:
    def track(robot: str, t: float):
        if robot == "unit2":
            if t < b.u2_return:
                return _walk(t, b.dispatch_u2, b.u2_arrive, s.unit2_dock, s.unit2_work)
            return _walk(t, b.u2_return, b.u1_arrive, s.unit2_work, s.unit2_dock)
        if t < b.u1_return:
            return _walk(t, b.u1_walk, b.u1_arrive, s.unit1_dock, s.unit1_work)
        return _walk(t, b.u1_return, b.both_docked, s.unit1_work, s.unit1_dock)
    return track


def timeline_track(rows) -> Callable:
    poses = {"unit1": [], "unit2": []}
    for r in rows:
        if r.get("type") == "tick":
            for k in poses:
                p = r["robots"][k]["pose"]
                poses[k].append((r["t"], p[0], p[1]))
    arr = {k: np.array(v, float) for k, v in poses.items()}

    def track(robot: str, t: float):
        a = arr[robot]
        return (float(np.interp(t, a[:, 0], a[:, 1])), float(np.interp(t, a[:, 0], a[:, 2])), TRUNK_Z)
    return track


def load_track(path=FILM_RUN / "timeline.jsonl") -> Optional[Callable]:
    p = Path(path)
    if not p.exists():
        return None
    return timeline_track(json.loads(line) for line in p.read_text().splitlines() if line.strip())


def _pan(track: Callable, robot: str, eye, t_hold: float) -> Callable:
    def cam(u, t):
        p = track(robot, t_hold if t is None else t)
        return follow(p, tuple(e - q - d for e, q, d in zip(eye, p, (0.0, 0.0, 0.25))), FOLLOW_FOCAL)
    return cam


def _blend(prev: Callable, cur: Callable, seconds: float, length: float) -> Callable:
    def cam(u, t):
        a, c = prev(1.0, None), cur(u, t)
        k = ease(u * length / seconds)
        mix = lambda p, q: tuple(x + (y - x) * k for x, y in zip(p, q))
        return mix(a[0], c[0]), mix(a[1], c[1]), a[2] + (c[2] - a[2]) * k
    return cam


def _still(pose) -> Callable:
    return lambda u, t: pose


def edl(beats: Optional[FilmBeats] = None, scene: FilmScene = DEFAULT_FILM_SCENE, track: Optional[Callable] = None,
        length: float | str | None = LENGTH) -> list[Shot]:
    b, s = beats or FilmBeats(), scene
    tr = track or default_track(b, s)
    j2, j1 = _still(unit2_job(s)), _still(unit1_job(s))
    home = min(b.u1_return, b.u2_return) if b.u2_return >= b.paint_done else b.u1_return
    p2, p1 = _pan(tr, "unit2", j2(0, 0)[0], b.u2_arrive), _pan(tr, "unit1", j1(0, 0)[0], b.u1_arrive)
    plan = [
        ("rig", 6.0, (0.0, 0.0), lambda u, t: rig_orbit(u), "fade", "xfade", False),
        ("control_push", 4.0, (0.0, 0.0), lambda u, t: control_push(u), "xfade", "xfade", False),
        ("board", 8.5, (0.0, b.dispatch_u2 + 0.3), lambda u, t: control_room(u * 0.85), "xfade", "cut", True),
        ("board_hold", 1.5, (b.dispatch_u2 + 0.3, b.dispatch_u2 + 0.3), lambda u, t: control_room(0.85 + 0.15 * u),
         "cut", "xfade", True),
        ("unit2_walk", 1.5, (b.u2_arrive - 1.5, b.u2_arrive), p2, "xfade", "cut", False),
        ("unit2_job", 10.5, (b.u2_engage - 2.5, b.gf5_green + 0.8), j2, "cut", "cut", False),
        ("unit1_walk", 1.5, (b.u1_arrive - 1.5, b.u1_arrive), p1, "cut", "cut", False),
        ("unit1_spray", 3.4, (b.spray - 0.3, b.soak), j1, "cut", "cut", False),
        ("unit1_soak", 1.0, (b.soak, b.scrub), j1, "cut", "cut", False),
        ("unit1_scrub", 8.0, (b.scrub, b.scrub_done), j1, "cut", "cut", False),
        ("unit1_paint", 4.6, (b.scrub_done, b.paint_done + 0.3), j1, "cut", "xfade", False),
        ("walk_back", 4.0, (home, home + 10.0), _still(birdseye(s)), "xfade", "xfade", False),
    ]
    shots, t, prev = [], 0.0, None
    for name, dur, sim, cam, t_in, t_out, two_d in plan:
        rate = 2.5 if name == "walk_back" else (sim[1] - sim[0]) / dur
        if name in ("unit2_job", "unit1_spray"):
            cam = _blend(prev, cam, MOVE_S, dur)
        badge = "" if rate <= 1.05 or two_d else f"{rate:.1f}x".replace(".0x", "x")
        shots.append(Shot(name, (t, t + dur), sim, rate, cam, t_in, t_out, two_d, badge))
        t, prev = t + dur, cam
    end, cards = CUTS["long" if length in (None, "long") else "short"]
    if isinstance(length, (int, float)):
        end = float(length)
    shots.append(Shot("conclusion", (t, max(end, t + 4.0)), None, 1.0, lambda u, tt: conclusion(u), "xfade",
                      "fade", True, cards=cards))
    return shots


def camera_at_film(film_t: float, shots: Optional[list[Shot]] = None):
    shots = shots or edl()
    shot = next((s for s in shots if s.film[0] <= film_t < s.film[1]), shots[-1])
    u = min(max((film_t - shot.film[0]) / (shot.film[1] - shot.film[0]), 0.0), 1.0)
    sim_t = shot.sim_at(film_t)
    eye, target, focal = shot.camera(u, sim_t)
    return tuple(map(float, eye)), tuple(map(float, target)), float(focal), sim_t


def to_json(shots: list[Shot]) -> str:
    return json.dumps([s.to_dict() for s in shots], indent=1)


if __name__ == "__main__":
    b = FilmBeats.load() if (FILM_RUN / "summary.json").exists() else FilmBeats()
    print(to_json(edl(b, track=load_track(), length=sys.argv[1] if len(sys.argv) > 1 else LENGTH)))
