from __future__ import annotations

import math

import numpy as np

KINDS = {
    "calm": {"wind_speed": 3.0, "rain": 0.0, "fog": 0.35, "spray": 0.15},
    "wind": {"wind_speed": 12.0, "rain": 0.0, "fog": 0.25, "spray": 0.6},
    "rain": {"wind_speed": 8.0, "rain": 1.0, "fog": 0.55, "spray": 0.45},
    "fog": {"wind_speed": 2.5, "rain": 0.0, "fog": 1.0, "spray": 0.1},
    "storm": {"wind_speed": 18.0, "rain": 1.0, "fog": 0.7, "spray": 1.0},
}
DEFAULT_KIND = {"night": "calm", "golden": "calm", "storm": "storm"}
TIERS = {
    "live": {"width": 1280, "height": 720, "rain": 1500, "embers": 160, "spray": 300, "smoke": 48, "steam": 40, "flow": True,
             "flow_cell": 0.25, "slices": 96, "pixel_ratio": 8, "accum": 32, "renderer": "RealTimePathTracing",
             "subframes": 1, "spp": 1},
    "offline": {"width": 1920, "height": 1080, "rain": 6000, "embers": 600, "spray": 1200, "smoke": 140, "steam": 120, "flow": True,
                "flow_cell": 0.18, "slices": 256, "pixel_ratio": 4, "accum": 128, "renderer": "RealTimePathTracing",
                "subframes": 16, "spp": 2},
}
MEDIA = {
    "night": {"vis": 240.0, "albedo": (0.6, 0.57, 0.53), "transmit": (0.5, 0.58, 0.62), "height": 52.0, "g": 0.55},
    "storm": {"vis": 170.0, "albedo": (0.82, 0.86, 0.9), "transmit": (0.5, 0.55, 0.6), "height": 70.0, "g": 0.35},
    "golden": {"vis": 420.0, "albedo": (0.95, 0.82, 0.66), "transmit": (0.62, 0.55, 0.48), "height": 60.0, "g": 0.6},
}


def weather_state(level="calm", wind_dir: float | None = None, gust: float | None = None) -> dict:
    if isinstance(level, str):
        w = dict(KINDS[level])
    else:
        u = float(np.clip(level, 0.0, 1.0))
        w = {k: (1 - u) * KINDS["calm"][k] + u * KINDS["storm"][k] for k in KINDS["calm"]}
    if gust is not None:
        w["wind_speed"] *= float(gust) / 18.0
    a = math.radians(30.0 if wind_dir is None else float(wind_dir))
    w["wind_dir"] = math.degrees(a)
    w["wind"] = (w["wind_speed"] * math.cos(a), w["wind_speed"] * math.sin(a))
    return w


def parse_schedule(text: str) -> list[tuple[float, str]]:
    out = []
    for item in filter(None, (s.strip() for s in text.split(","))):
        kind, _, at = item.partition("@")
        out.append((float(at or 0.0), kind if kind in KINDS else float(kind)))
    return sorted(out)


def schedule_at(schedule, t: float):
    current = None
    for at, kind in schedule:
        if t >= at:
            current = kind
    return current if current is not None else (schedule[0][1] if schedule else None)


def volumetric_settings(preset: str, tier: str, weather: dict) -> dict:
    m, q = MEDIA[preset], TIERS[tier]
    return {
        "/rtx/raytracing/globalVolumetricEffects/enabled": True,
        "/rtx/raytracing/fractionalCutoutOpacity": True,
        "/rtx/pathtracing/fractionalCutoutOpacity": True,
        "/rtx/shadows/fractionalCutoutOpacity": True,
        "/rtx/raytracing/inscattering/detailNoiseScale": 0.012,
        "/rtx/raytracing/inscattering/noiseScaleRangeMin": 0.15,
        "/rtx/raytracing/inscattering/noiseScaleRangeMax": 1.0,
        "/rtx/raytracing/inscattering/densityMult": 1.0,
        "/rtx/raytracing/inscattering/singleScatteringAlbedo": list(m["albedo"]),
        "/rtx/raytracing/inscattering/transmittanceColor": list(m["transmit"]),
        "/rtx/raytracing/inscattering/transmittanceMeasurementDistance": m["vis"] * 0.35 / max(weather["fog"], 0.1),
        "/rtx/raytracing/inscattering/atmosphereHeight": m["height"],
        "/rtx/raytracing/inscattering/anisotropyFactor": m["g"],
        "/rtx/raytracing/inscattering/depthSlices": q["slices"],
        "/rtx/raytracing/inscattering/pixelRatio": q["pixel_ratio"],
        "/rtx/raytracing/inscattering/maxAccumulationFrames": q["accum"],
        "/rtx/raytracing/inscattering/useDetailNoise": 1,
        "/rtx/raytracing/inscattering/noiseAnimationSpeedX": weather["wind"][0] * 0.05,
        "/rtx/raytracing/inscattering/noiseAnimationSpeedY": weather["wind"][1] * 0.05,
        "/rtx/raytracing/inscattering/maxDistance": 3000.0,
    }


def wind_lean(wind, height: float, stiffness: float = 25.0) -> tuple[float, float]:
    speed = math.hypot(*wind)
    if speed == 0:
        return (0.0, 0.0)
    bend = height * min(1.0, speed / stiffness) * (height / (height + 2.0))
    return (wind[0] / speed * bend, wind[1] / speed * bend)


class Particles:
    def __init__(self, n: int, box, velocity=(0.0, 0.0, 0.0), spread: float = 0.0, gravity: float = 0.0,
                 drag: float = 0.0, life=(math.inf, math.inf), seed: int = 0, source=None):
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = np.asarray(box[0], float), np.asarray(box[1], float)
        self.v0, self.spread, self.gravity, self.drag, self.life = np.asarray(velocity, float), spread, gravity, drag, life
        self.source = source
        self.pos, self.vel = np.zeros((n, 3)), np.zeros((n, 3))
        self.age, self.ttl = np.zeros(n), np.zeros(n)
        self._spawn(np.ones(n, bool), anywhere=True)

    def _spawn(self, mask, anywhere: bool = False) -> None:
        k = int(mask.sum())
        if not k:
            return
        extra = 0.0
        if self.source is not None:
            got = self.source(self.rng, k)
            self.pos[mask], extra = got if isinstance(got, tuple) else (got, 0.0)
        else:
            self.pos[mask] = self.rng.uniform(self.lo, self.hi, (k, 3))
            if not anywhere and self.v0[2] < 0:
                self.pos[mask, 2] = self.hi[2]
        self.vel[mask] = self.v0 + extra + self.rng.normal(0.0, self.spread, (k, 3))
        lo, hi = self.life
        self.ttl[mask] = self.rng.uniform(lo, hi, k) if math.isfinite(hi) else math.inf
        self.age[mask] = self.rng.uniform(0.0, np.minimum(self.ttl[mask], 1.0)) if anywhere else 0.0

    def recentre(self, centre) -> None:
        half = (self.hi - self.lo) / 2
        shift = np.asarray(centre, float) - (self.lo + half)
        self.lo, self.hi = self.lo + shift, self.hi + shift
        self.pos += shift

    def step(self, dt: float, wind=(0.0, 0.0)) -> None:
        air = np.array([wind[0], wind[1], 0.0])
        self.vel[:, 2] += self.gravity * dt
        if self.drag:
            self.vel += (air - self.vel) * min(1.0, self.drag * dt)
            self.pos += self.vel * dt
        else:
            self.pos += (self.vel + air) * dt
        self.age += dt
        size = self.hi - self.lo
        if self.source is None:
            wrap = (self.pos[:, :2] - self.lo[:2]) % size[:2] + self.lo[:2]
            self.pos[:, :2] = wrap
        dead = (self.age > self.ttl) | (self.pos[:, 2] < self.lo[2]) | (self.pos[:, 2] > self.hi[2])
        dead |= ((self.pos < self.lo) | (self.pos > self.hi)).any(1)
        self._spawn(dead)


def streak_points(pos, vel, shutter: float) -> np.ndarray:
    out = np.empty((2 * len(pos), 3))
    out[0::2] = pos - vel * shutter
    out[1::2] = pos
    return out


def soft_texture(size: int, seed: int = 0, octaves: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size))
    for o in range(octaves):
        n = 2 ** (o + 2)
        g = rng.uniform(0, 1, (n + 1, n + 1))
        x = np.linspace(0, n, size)
        i = np.minimum(x.astype(int), n - 1)
        f = x - i
        f = f * f * (3 - 2 * f)
        rows = g[i] * (1 - f)[:, None] + g[i + 1] * f[:, None]
        acc += (rows[:, i] * (1 - f) + rows[:, i + 1] * f) / 2 ** o
    acc = (acc - acc.min()) / (np.ptp(acc) + 1e-9)
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1) * 2 - 1
    edge = np.clip(1 - np.hypot(xx, yy), 0, 1) ** 1.5
    return (np.clip(acc * 1.3 - 0.05, 0, 1) * edge * 255).astype(np.uint8)


FIRE_MAP = ([(0.07, 0.07, 0.075, 0.28), (0.04, 0.038, 0.035, 0.45), (0.035, 0.033, 0.03, 0.5), (1.0, 0.16, 0.013, 0.8),
             (13.5, 3.0, 0.126, 0.8), (78.0, 39.0, 6.1, 0.7)], [0, 0.05, 0.15, 0.6, 0.85, 1])


def billboard_quats(pos, eye) -> np.ndarray:
    d = np.asarray(eye, float) - np.asarray(pos, float)
    yaw = np.arctan2(d[:, 1], d[:, 0]) + np.pi / 2
    pitch = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
    cy, sy, cp, sp = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(-pitch / 2), np.sin(-pitch / 2)
    return np.column_stack([cy * cp, cy * sp, sy * sp, sy * cp])


def fade_bucket(age, ttl, n: int) -> np.ndarray:
    u = np.clip(np.asarray(age) / np.maximum(np.asarray(ttl), 1e-6), 0, 1)
    level = np.minimum(np.minimum(u / 0.15, 1.0), (1 - u) / 0.5)
    return np.clip((level * n).astype(int), 0, n - 1)


def flow_overrides(flare, vents, wind, cell: float, flame_radius: float = 2.6, fuel: float = 0.8, cooling: float = 2.2,
                   steam: float = 0.9, smoke_fade: float = 0.25, jet: float = 3.0) -> dict:
    s = flame_radius / 10.0
    lean = (float(wind[0]) * 0.3, float(wind[1]) * 0.3)
    rgba, xs = FIRE_MAP

    def emitter(pos, radius, up, **emit):
        return {"layer": 0, "position": tuple(map(float, pos)), "radius": float(radius),
                "velocity": (lean[0], lean[1], up), "velocityIsWorldSpace": True, **emit}

    return {
        "flowEmitterSphere": emitter(flare, flame_radius, jet, fuel=fuel, temperature=2.0),
        "flowSimulate": {"layer": 0, "densityCellSize": float(cell)},
        "flowSimulate/advection": {"gravity": (0.0, 0.0, -100.0 * s), "coolingRate": cooling, "combustionEnabled": True},
        "flowSimulate/advection/smoke": {"fade": smoke_fade},
        "flowOffscreen": {"layer": 0},
        "flowOffscreen/colormap": {"rgbaPoints": list(rgba), "xPoints": [float(x) for x in xs],
                                   "colorScalePoints": [1.0] * len(xs)},
        "flowRender": {"layer": 0},
        "vents": [emitter(p, 1.0, 3.0, fuel=0.0, temperature=0.08, smoke=steam, coupleRateSmoke=2.0) for p in vents],
    }
