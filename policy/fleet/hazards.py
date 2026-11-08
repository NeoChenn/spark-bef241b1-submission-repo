from __future__ import annotations

import math

import numpy as np

from policy.fleet.rigmap import DECKS

OBSTACLES = {"crate": ("block", 1.2), "debris": ("block", 1.5), "scaffold": ("block", 1.8),
             "hose": ("slow", 1.0), "spill": ("slip", 2.0)}
SLOW = {"slow": 0.45, "slip": 0.3}
WEATHER = {"calm": (0.0, 0.0, 0.0), "wind": (0.8, 1.0, 0.0), "rain": (0.7, 0.25, 1.0), "fog": (0.6, 0.0, 0.8),
           "storm": (1.0, 1.0, 1.0)}
FOG = {"fog": 1.0, "storm": 0.6}
GUST_REF_N = 18.0
WIND_CLOSE = 0.75
WIND_DIR = {"wind": 0.0, "storm": 90.0}
FLARE_XY = (-26.0, 8.0)
FLARE_R = 7.0
PERCEPTION_M = 12.0
OUTER_TOL = 0.5
CATWALK_DECKS = ("middle_deck", "lower_deck")
SIDES = {"S": (0.0, -1.0), "N": (0.0, 1.0), "W": (-1.0, 0.0), "E": (1.0, 0.0)}
PERCEIVE = {"inspect": 0.4, "leak_check": 0.4, "rust": 0.2, "crack": 0.2, "gas": 0.1}


def seg_dist(p, a, b) -> float:
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    d = b - a
    s = float(np.clip(np.dot(p - a, d) / max(float(np.dot(d, d)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + s * d)))


def weather_state(level="calm", wind_dir=None, gust=None) -> dict:
    if isinstance(level, (int, float)):
        x = float(np.clip(level, 0.0, 1.0))
        name, (lv, gs, rain), fog = f"{x:.2f}", (x, x, x), 0.6 * x
    else:
        if level not in WEATHER:
            raise ValueError(f"unknown weather {level!r}; expected one of {list(WEATHER)}")
        name, (lv, gs, rain), fog = level, WEATHER[level], FOG.get(level, 0.0)
    wind = gs * (float(gust) / GUST_REF_N if gust is not None else 1.0)
    wdir = float(wind_dir if wind_dir is not None else WIND_DIR.get(name, 90.0)) % 360.0
    base = (1 - 0.2 * fog) * (1 - 0.1 * min(wind, 1.5))
    return {"level": name, "intensity": round(lv, 3), "wind_dir": wdir, "gust": gust, "wind": round(wind, 3),
            "wet": round(rain, 3), "fog": round(fog, 3),
            "speed": {"walk": round(base * (1 - 0.2 * rain), 3), "catwalk": round(base * (1 - 0.2 * rain), 3),
                      "spur": round(base * (1 - 0.2 * rain), 3), "stairs": round(base * (1 - 0.5 * rain), 3)},
            "perception_range_m": round(PERCEPTION_M * (1 - 0.6 * fog), 1),
            "second_view_p": round(min(1.0, 0.15 + 0.7 * fog), 2),
            "closes_outer": wind >= WIND_CLOSE}


def outer_sides(rm) -> dict[int, str]:
    out = {}
    for deck in DECKS:
        idx = [i for i, c in enumerate(rm.coarse) if c["deck"] == deck and c["kind"] in ("walk", "catwalk")]
        if not idx:
            continue
        pts = np.array([rm.coarse[i][k][:2] for i in idx for k in "ab"])
        lo, hi = pts.min(0), pts.max(0)
        for i in idx:
            a, b = np.array(rm.coarse[i]["a"][:2]), np.array(rm.coarse[i]["b"][:2])
            for side, axis, v in (("S", 1, lo[1]), ("N", 1, hi[1]), ("W", 0, lo[0]), ("E", 0, hi[0])):
                if abs(a[axis] - v) < OUTER_TOL and abs(b[axis] - v) < OUTER_TOL:
                    out[i] = side
    return out


class Hazards:
    def __init__(self):
        self.obstacles: dict[str, dict] = {}
        self.closures: dict[str, dict] = {}
        self.weather = weather_state()
        self.n = 0
        self.sig = None
        self._wk = None
        self.why: dict[tuple[int, int], str] = {}

    def active(self) -> bool:
        return bool(self.obstacles or self.closures or self.weather["level"] != "calm")

    def stale(self, rm) -> bool:
        return self.sig != (len(rm.pos), len(rm.coarse))

    def _next(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}{self.n}"

    def walkway_edges(self, rm) -> dict[int, list[tuple[int, int]]]:
        if self._wk is not None and self._wk[0] == (len(rm.pos), len(rm.coarse)):
            return self._wk[1]
        out: dict[int, list] = {i: [] for i in range(len(rm.coarse))}
        for a, b, e in rm.edges():
            for i, c in enumerate(rm.coarse):
                if c["kind"] == e.kind and seg_dist(rm.pos[a], c["a"], c["b"]) < 1e-3 \
                        and seg_dist(rm.pos[b], c["a"], c["b"]) < 1e-3:
                    out[i].append((a, b))
                    break
        self._wk = ((len(rm.pos), len(rm.coarse)), out)
        return out

    def targets(self, rm, target: str, kind: str | None = None) -> list[int]:
        idx = [i for i, c in enumerate(rm.coarse) if c["id"] == target and (kind is None or c["kind"] == kind)]
        if not idx:
            ids = sorted({c["id"] for c in rm.coarse if kind is None or c["kind"] == kind})
            raise ValueError(f"unknown {kind or 'walkway'} {target!r}; expected one of {ids}")
        return idx

    def drop(self, kind: str, xyz, at=None) -> dict:
        if kind not in OBSTACLES:
            raise ValueError(f"unknown obstacle {kind!r}; expected one of {list(OBSTACLES)}")
        effect, radius = OBSTACLES[kind]
        ob = {"id": self._next(kind[:2]), "kind": kind, "xyz": [round(float(v), 3) for v in xyz],
              "radius": radius, "effect": effect, "at": at}
        self.obstacles[ob["id"]] = ob
        return ob

    def close(self, rm, target: str, kind: str | None = None) -> dict:
        self.targets(rm, target, kind)
        cl = {"id": self._next("x"), "kind": kind or "walkway", "target": target}
        self.closures[cl["id"]] = cl
        return cl

    def clear(self, hid: str) -> list[str]:
        if hid == "all":
            gone = list(self.obstacles) + list(self.closures)
            self.obstacles.clear()
            self.closures.clear()
            return gone
        if hid in self.obstacles:
            del self.obstacles[hid]
        elif hid in self.closures:
            del self.closures[hid]
        else:
            raise ValueError(f"no obstacle or closure {hid!r}; active: {list(self.obstacles) + list(self.closures)}")
        return [hid]

    def set_weather(self, level, wind_dir=None, gust=None) -> dict:
        self.weather = weather_state(level, wind_dir, gust)
        return self.weather

    def work_slow(self, kind: str) -> float:
        return 1.0 + PERCEIVE.get(kind, 0.0) * self.weather["fog"]

    def apply(self, rm) -> None:
        wk = self.walkway_edges(rm)
        closed: dict[tuple[int, int], str] = {}
        factor: dict[tuple[int, int], float] = {}
        for cl in self.closures.values():
            for i in self.targets(rm, cl["target"], None if cl["kind"] == "walkway" else cl["kind"]):
                for k in wk[i]:
                    closed.setdefault(k, cl["id"])
        for ob in self.obstacles.values():
            for a, b, e in rm.edges():
                if seg_dist(ob["xyz"], rm.pos[a], rm.pos[b]) >= ob["radius"]:
                    continue
                if ob["effect"] == "block" or (ob["effect"] == "slip" and e.kind == "stairs"):
                    closed.setdefault((a, b), ob["id"])
                else:
                    factor[(a, b)] = min(factor.get((a, b), 1.0), SLOW[ob["effect"]])
        w = self.weather
        wv = np.array([math.cos(math.radians(w["wind_dir"])), math.sin(math.radians(w["wind_dir"]))])
        for i, side in outer_sides(rm).items():
            exposed = float(np.dot(SIDES[side], -wv)) > 0.5 and rm.coarse[i]["deck"] in CATWALK_DECKS
            for k in wk[i]:
                if exposed and w["closes_outer"]:
                    closed.setdefault(k, f"wind {side}")
                elif w["wind"] > 0:
                    factor[k] = min(factor.get(k, 1.0), 1 - 0.25 * min(w["wind"], 1.5))
        top = [c for c in rm.coarse if c["deck"] == "top_deck" and c["kind"] == "walk"]
        if w["closes_outer"] and top:
            ctr = np.mean([c[k][:2] for c in top for k in "ab"], axis=0)
            u = ctr - np.array(FLARE_XY)
            if float(np.dot(wv, u / np.linalg.norm(u))) > 0.3:
                fz = (*FLARE_XY, DECKS["top_deck"])
                for a, b, e in rm.edges():
                    if e.kind != "stairs" and seg_dist(fz, rm.pos[a], rm.pos[b]) < FLARE_R:
                        closed.setdefault((a, b), "flare side")
        rm.closed = set(closed)
        rm.factor = factor
        rm.kind_factor = dict(w["speed"]) if w["level"] != "calm" else {}
        self.why = closed
        self.sig = (len(rm.pos), len(rm.coarse))

    def to_json(self, rm) -> dict:
        r3 = lambda p: [round(float(v), 2) for v in p]
        return {"obstacles": list(self.obstacles.values()), "closures": list(self.closures.values()),
                "weather": self.weather,
                "closed": [{"a": r3(rm.pos[a]), "b": r3(rm.pos[b]), "why": why} for (a, b), why in self.why.items()],
                "slow": [{"a": r3(rm.pos[a]), "b": r3(rm.pos[b]), "factor": round(f, 2)}
                         for (a, b), f in rm.factor.items()]}
