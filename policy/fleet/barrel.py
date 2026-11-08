from __future__ import annotations

import math

import numpy as np

R = 0.29
V_WIND = 2.0
K = 0.35
MU = 0.13
KICK = 2.0
V_STOP = 0.06
STRAIGHT = 0.7
TURN_KEEP = 0.4
RESTITUTION = 0.35
SUB_DT = 0.05
ROLL_MAX_S = 90.0
GUST = 0.25
AUTO_MAX = 2
AUTO_WIND = 0.8
MOVE_APPLY_M = 0.5
SETTLE_V = 0.4


def wind_xy(weather: dict) -> tuple[float, float]:
    w, a = float(weather.get("wind", 0.0)), math.radians(float(weather.get("wind_dir", 0.0)))
    return w * math.cos(a), w * math.sin(a)


def flat(rm, n: int, skip: int | None = None) -> list[int]:
    return [m for m, e in rm.adj[n].items() if e.kind != "stairs" and m != skip]


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)[:2]
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(2)


def nearest_edge(rm, xyz) -> tuple[int, int, float]:
    p = np.asarray(xyz, float)
    best = None
    for a, b, e in rm.edges():
        if e.kind == "stairs":
            continue
        pa, d = rm.pos[a], rm.pos[b] - rm.pos[a]
        L2 = max(float(np.dot(d, d)), 1e-12)
        s = float(np.clip(np.dot(p - pa, d) / L2, 0.0, 1.0))
        q = pa + s * d - p
        dist = float(np.hypot(np.linalg.norm(q[:2]), 3.0 * q[2]))
        if best is None or dist < best[0]:
            best = (dist, a, b, s * math.sqrt(L2))
    if best is None:
        raise ValueError(f"no walkway near {np.round(p, 1).tolist()} for a barrel")
    return best[1], best[2], best[3]


def _sync(rm, ob) -> None:
    a, b = ob["edge"]
    d = rm.pos[b] - rm.pos[a]
    L = float(np.linalg.norm(d))
    p = rm.pos[a] + d * (ob["s"] / max(L, 1e-9))
    u = unit(d)
    ob["xyz"] = [round(float(v), 4) for v in p]
    ob["vel"] = [round(float(ob["v"] * u[0]), 4), round(float(ob["v"] * u[1]), 4)]
    ob["yaw"] = round(math.degrees(math.atan2(u[1], u[0])), 2)


def start(rm, ob: dict, toward=None, wind=(0.0, 0.0)) -> dict:
    a, b, s = nearest_edge(rm, ob["xyz"])
    u = unit(rm.pos[b] - rm.pos[a])
    v = 0.0
    if toward is not None:
        h = unit(np.asarray(toward, float)[:2] - (rm.pos[a][:2] + u * s))
        v = KICK * (1.0 if float(np.dot(h, u)) >= 0 else -1.0) if abs(float(np.dot(h, u))) > 0.3 else 0.0
    ob.update(edge=[int(a), int(b)], s=float(s), v=v, spin=0.0, age=0.0, rolling=True)
    _sync(rm, ob)
    return ob


def _node(rm, ob, n: int, prev: int, speed: float, over: float, wind) -> bool:
    h = unit(rm.pos[n] - rm.pos[prev])
    w = np.asarray(wind, float)
    nxt = [(m, unit(rm.pos[m] - rm.pos[n])) for m in flat(rm, n, prev)]
    straight = max(nxt, key=lambda c: float(np.dot(c[1], h)), default=None)
    if straight is not None and float(np.dot(straight[1], h)) > STRAIGHT:
        ob.update(edge=[n, straight[0]], s=over, v=speed)
        return False
    push = max(nxt, key=lambda c: float(np.dot(c[1], w)), default=None)
    if push is not None and V_WIND * float(np.dot(push[1], w)) * K > MU * 1.4 and float(np.dot(push[1], h)) > -0.5:
        ob.update(edge=[n, push[0]], s=over * TURN_KEEP, v=speed * TURN_KEEP)
        return False
    L = float(np.linalg.norm(rm.pos[n] - rm.pos[prev]))
    settle = speed < SETTLE_V or not nxt
    ob.update(edge=[prev, n], s=max(L - (R if settle else over), 0.0), v=-speed * RESTITUTION)
    ob["bounces"] = ob.get("bounces", 0) + 1
    return settle


def step(rm, ob: dict, wind, dt: float) -> bool:
    if not ob.get("rolling"):
        return False
    n = max(1, int(math.ceil(dt / SUB_DT)))
    h = dt / n
    for _ in range(n):
        ob["age"] += h
        g = 1.0 + GUST * math.sin(1.9 * ob["age"]) * math.sin(0.7 * ob["age"] + 1.0)
        w = (wind[0] * g, wind[1] * g)
        a, b = ob["edge"]
        d = rm.pos[b] - rm.pos[a]
        L = float(np.linalg.norm(d))
        push = V_WIND * float(np.dot(unit(d), w))
        v = ob["v"] + K * (push - ob["v"]) * h
        if abs(v) > 1e-9:
            v -= math.copysign(min(abs(v), MU * h), v)
        ob["v"] = v
        if abs(v) < V_STOP and abs(push) * K <= MU * 1.4:
            return _stop(rm, ob)
        ob["s"] += v * h
        ob["spin"] = round(ob["spin"] + abs(v) * h / R, 4)
        if ob["s"] > L and _node(rm, ob, b, a, abs(v), ob["s"] - L, w):
            return _stop(rm, ob)
        if ob["s"] < 0 and _node(rm, ob, a, b, abs(v), -ob["s"], w):
            return _stop(rm, ob)
        if ob["age"] >= ROLL_MAX_S or ob.get("bounces", 0) > 6:
            return _stop(rm, ob)
    _sync(rm, ob)
    return True


def _stop(rm, ob) -> bool:
    ob["v"], ob["rolling"] = 0.0, False
    _sync(rm, ob)
    return False
