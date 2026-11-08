import bisect
import math

LEVELS = {"calm": (0.0, 0.0, 0.0), "wind": (0.8, 1.0, 0.0), "rain": (0.7, 0.25, 1.0), "fog": (0.6, 0.0, 0.8),
          "storm": (1.0, 1.0, 1.0)}
FOGGY = {"fog": 1.0, "storm": 0.6}
GUST_REF_N = 18.0
NIGHT = {"fog_color": (0.05, 0.16, 0.18), "fog_height": 0.002, "wave_height": 1.5, "dome": 400.0, "lights": 1.0, "sun": 0.0,
         "fog_dist": 0.0}
STORM = {"fog_color": (0.2, 0.24, 0.27), "fog_height": 0.03, "wave_height": 3.2, "dome": 900.0, "lights": 0.8, "sun": 90.0,
         "fog_dist": 0.004}
FOG_HEIGHT_MAX = 0.05
FOG_DIST_MAX = 0.012
LEAN_PER_M = 0.45
DROP_H = 1.5
DROP_S = 0.7
DRUM_R = 0.29
PREDICT_S = 0.8
BLEND_S = 0.35


def lerp(a, b, s):
    if isinstance(a, tuple):
        return tuple(x + (y - x) * s for x, y in zip(a, b))
    return a + (b - a) * s


def weather_params(level="calm", wind_dir=None, gust=None) -> dict:
    if isinstance(level, (int, float)):
        x = min(max(float(level), 0.0), 1.0)
        lv, gs, rain, fog = x, x, x, 0.6 * x
    else:
        if level not in LEVELS:
            raise ValueError(f"unknown weather {level!r}")
        (lv, gs, rain), fog = LEVELS[level], FOGGY.get(level, 0.0)
    wind = gs * (float(gust) / GUST_REF_N if gust is not None else 1.0)
    wdir = float(wind_dir if wind_dir is not None else 90.0)
    out = {k: lerp(NIGHT[k], STORM[k], lv) for k in NIGHT}
    out["wave_height"] = lerp(NIGHT["wave_height"], STORM["wave_height"], min(wind, 1.5))
    out["fog_height"] = max(out["fog_height"], lerp(NIGHT["fog_height"], FOG_HEIGHT_MAX, fog),
                            lerp(NIGHT["fog_height"], STORM["fog_height"], rain))
    out["fog_dist"] = max(out["fog_dist"], FOG_DIST_MAX * fog)
    out.update(blend=lv, wind=wind, wind_dir=wdir, rain=rain, fog=fog, storm_sky=lv >= 0.5)
    return out


def wind_vec(wind_dir: float, wind: float) -> tuple[float, float]:
    a = math.radians(wind_dir)
    return wind * math.cos(a), wind * math.sin(a)


def flame_offset(dz: float, wind_dir: float, wind: float) -> tuple[float, float, float]:
    wx, wy = wind_vec(wind_dir, wind)
    k = LEAN_PER_M * dz
    return wx * k, wy * k, -0.15 * k * min(wind, 1.0)


def drop_offset(age: float, h: float = DROP_H, dur: float = DROP_S) -> float:
    if age >= dur:
        return 0.0
    u = max(age, 0.0) / dur
    return h * (1.0 - u * u)


def segments(path, lift: float = 0.0, min_len: float = 1e-3) -> list[tuple[tuple, tuple]]:
    pts = [(float(p[0]), float(p[1]), float(p[2]) + lift) for p in path or []]
    return [(a, b) for a, b in zip(pts, pts[1:]) if math.dist(a, b) > min_len]


def dashes(a, b, dash: float = 0.6, gap: float = 0.4) -> list[tuple[tuple, tuple]]:
    L = math.dist(a, b)
    out, s = [], 0.0
    while s < L - 1e-9:
        e = min(s + dash, L)
        out.append((lerp(tuple(a), tuple(b), s / L), lerp(tuple(a), tuple(b), e / L)))
        s += dash + gap
    return out


def seg_frame(a, b) -> tuple[tuple, float, float, float]:
    dx, dy, dz = (b[i] - a[i] for i in range(3))
    L = math.sqrt(dx * dx + dy * dy + dz * dz)
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = -math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return lerp(tuple(a), tuple(b), 0.5), L, yaw, pitch


def tick_index(times: list[float], t: float) -> int:
    return max(0, bisect.bisect_right(times, t) - 1)


def ghost_alpha(age: float, hold: float = 6.0, fade: float = 6.0) -> float:
    if age <= hold:
        return 1.0
    return max(0.0, 1.0 - (age - hold) / fade)


def drum_pose(mv: dict, t: float, prev, yaw: float, spin: float) -> tuple[tuple, float, float]:
    dt = min(max(t - mv["t"], 0.0), PREDICT_S)
    p = [mv["p"][0] + mv["v"][0] * dt, mv["p"][1] + mv["v"][1] * dt, mv["p"][2]]
    w = max(0.0, 1.0 - (t - mv["t"]) / BLEND_S)
    p = tuple(p[i] + (mv["from"][i] - mv["p"][i]) * w for i in range(3))
    if prev is not None:
        dx, dy = p[0] - prev[0], p[1] - prev[1]
        step = math.hypot(dx, dy)
        if step > 1e-4:
            yaw = math.degrees(math.atan2(dy, dx))
            spin = (spin + math.degrees(step / DRUM_R)) % 360.0
    return p, yaw, spin


def puddle_edge(a: float, seed: int) -> float:
    k = (seed % 97) / 97.0 * 6.283
    return 0.78 + 0.14 * math.sin(2 * a + k) + 0.08 * math.sin(3 * a + 2.1 * k) + 0.05 * math.sin(5 * a + 0.7 * k)
