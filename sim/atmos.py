from __future__ import annotations

import math
import re

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
    "live": {"width": 1280, "height": 720, "rain": 1500, "embers": 160, "spray": 300, "fog_cards": 10, "flow": True,
             "flow_cell": 0.35, "slices": 96, "pixel_ratio": 8, "accum": 32, "renderer": "RealTimePathTracing",
             "subframes": 1, "spp": 1},
    "offline": {"width": 1920, "height": 1080, "rain": 6000, "embers": 600, "spray": 1200, "fog_cards": 18, "flow": True,
                "flow_cell": 0.18, "slices": 256, "pixel_ratio": 4, "accum": 128, "renderer": "RealTimePathTracing",
                "subframes": 16, "spp": 2},
}
MEDIA = {
    "night": {"density": 0.4, "albedo": (0.9, 0.86, 0.8), "transmit": (0.55, 0.62, 0.66), "height": 120.0, "g": 0.55},
    "storm": {"density": 0.7, "albedo": (0.82, 0.86, 0.9), "transmit": (0.5, 0.55, 0.6), "height": 160.0, "g": 0.35},
    "golden": {"density": 0.3, "albedo": (0.95, 0.82, 0.66), "transmit": (0.7, 0.6, 0.5), "height": 140.0, "g": 0.6},
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
        "/rtx/raytracing/inscattering/densityMult": m["density"] * (0.5 + 1.5 * weather["fog"]),
        "/rtx/raytracing/inscattering/singleScatteringAlbedo": list(m["albedo"]),
        "/rtx/raytracing/inscattering/transmittanceColor": list(m["transmit"]),
        "/rtx/raytracing/inscattering/transmittanceMeasurementDistance": 900.0 / (0.4 + weather["fog"]),
        "/rtx/raytracing/inscattering/atmosphereHeight": m["height"],
        "/rtx/raytracing/inscattering/anisotropyFactor": m["g"],
        "/rtx/raytracing/inscattering/depthSlices": q["slices"],
        "/rtx/raytracing/inscattering/pixelRatio": q["pixel_ratio"],
        "/rtx/raytracing/inscattering/maxAccumulationFrames": q["accum"],
        "/rtx/raytracing/inscattering/useDetailNoise": 1,
        "/rtx/raytracing/inscattering/noiseAnimationSpeedX": weather["wind"][0] * 0.02,
        "/rtx/raytracing/inscattering/noiseAnimationSpeedY": weather["wind"][1] * 0.02,
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


def fog_cards(n: int, radius: float, z=(2.0, 14.0), seed: int = 0, t: float = 0.0, wind=(0.0, 0.0)) -> dict:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-radius, radius, (n, 2)) + np.asarray(wind) * 0.35 * t
    xy = (xy + radius) % (2 * radius) - radius
    return {"centre": np.column_stack([xy, rng.uniform(*z, n)]), "size": rng.uniform(0.35, 0.7, n) * radius,
            "yaw": rng.uniform(0.0, 360.0, n)}


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


FIRE_MAP = ("[(0.012, 0.012, 0.012, 0.0), (0.03, 0.028, 0.026, 0.55), (0.05, 0.045, 0.04, 0.6), (1.0, 0.16, 0.013, 0.8), "
            "(13.5, 3.0, 0.13, 0.8), (78.0, 39.0, 6.1, 0.7)]", "[0, 0.05, 0.15, 0.6, 0.85, 1]")
STEAM_MAP = ("[(0.5, 0.52, 0.55, 0.0), (0.7, 0.72, 0.75, 0.25), (0.8, 0.82, 0.85, 0.45), (0.85, 0.86, 0.88, 0.5)]",
             "[0, 0.2, 0.6, 1]")


def _vec(v) -> str:
    return ", ".join(f"{float(c):g}" for c in v)


def _emitter(name, pos, radius, layer, velocity, fuel=0.0, temp=0.0, smoke=0.0) -> str:
    return f"""    def FlowEmitterSphere "{name}"
    {{
        float coupleRateFuel = 2
        float coupleRateSmoke = 2
        float coupleRateTemperature = 10
        float coupleRateVelocity = 2
        float fuel = {fuel:g}
        int layer = {layer}
        float3 position = ({_vec(pos)})
        float radius = {radius:g}
        bool radiusIsWorldSpace = 1
        float smoke = {smoke:g}
        float temperature = {temp:g}
        float3 velocity = ({_vec(velocity)})
        bool velocityIsWorldSpace = 1
    }}
"""


def _wind(layer, wind, centre) -> str:
    return f"""    def FlowEmitterBox "wind_{layer}"
    {{
        float allocationScale = 0
        float coupleRateVelocity = 0.4
        float coupleRateTemperature = 0
        float coupleRateFuel = 0
        float coupleRateSmoke = 0
        float3 halfSize = (400, 400, 200)
        int layer = {layer}
        float3 position = ({_vec(centre)})
        float3 velocity = ({_vec((wind[0], wind[1], 0.0))})
    }}
"""


def _layer(layer, cell, rgba, xs, combustion, fade, buoyancy) -> str:
    return f"""    def FlowSimulate "sim_{layer}"
    {{
        float densityCellSize = {cell:g}
        int layer = {layer}
        float stepsPerSecond = 30
        uint maxStepsPerSimulate = 2
        def FlowAdvectionCombustionParams "advection"
        {{
            float buoyancyPerTemp = {buoyancy:g}
            float buoyancyPerSmoke = {0.0 if combustion else 0.6:g}
            float burnPerTemp = 4
            bool combustionEnabled = {int(combustion)}
            float coolingRate = 1.0
            float fuelPerBurn = 0.25
            float3 gravity = (0, 0, -9.8)
            float ignitionTemp = 0.05
            float smokePerBurn = 4
            float tempPerBurn = 5
            def FlowAdvectionChannelParams "smoke" {{ float damping = 0.05
                float fade = {fade:g} }}
            def FlowAdvectionChannelParams "velocity" {{ float damping = 0.01
                float fade = 1 }}
            def FlowAdvectionChannelParams "divergence" {{}}
            def FlowAdvectionChannelParams "temperature" {{ float damping = 0
                float fade = 0 }}
            def FlowAdvectionChannelParams "fuel" {{ float damping = 0
                float fade = 0 }}
            def FlowAdvectionChannelParams "burn" {{ float damping = 0
                float fade = 0 }}
        }}
        def FlowVorticityParams "vorticity" {{ bool enabled = 1
            float forceScale = 2.5
            float velocityMask = 1 }}
        def FlowPressureParams "pressure" {{ bool enabled = 1 }}
        def FlowSummaryAllocateParams "summaryAllocate" {{ bool enableNeighborAllocation = 1
            float smokeThreshold = 0.02
            float speedThreshold = 1 }}
    }}
    def FlowOffscreen "offscreen_{layer}"
    {{
        int layer = {layer}
        def FlowRayMarchColormapParams "colormap"
        {{
            float colorScale = 2.5
            uint resolution = 32
            float4[] rgbaPoints = {rgba}
            float[] xPoints = {xs}
        }}
        def FlowShadowParams "shadow" {{ bool enabled = 1
            float attenuation = 0.045
            float minIntensity = 0.125
            uint numSteps = 16 }}
        def FlowDebugVolumeParams "debugVolume" {{}}
    }}
    def FlowRender "render_{layer}"
    {{
        int layer = {layer}
        def FlowRayMarchParams "rayMarch"
        {{
            float attenuation = 0.05
            float colorScale = 1
            float stepSizeScale = 0.75
            def FlowRayMarchCloudParams "cloud" {{}}
        }}
    }}
"""


def flow_usda(flare, vents, wind, cell: float) -> str:
    lean = (wind[0] * 0.4, wind[1] * 0.4)
    body = _emitter("flare", flare, 1.3, 0, (lean[0] - 2.0, lean[1], 14.0), fuel=1.0, temp=2.0)
    body += "".join(_emitter(f"vent_{i}", p, 0.6, 1, (lean[0], lean[1], 5.0), smoke=0.8, temp=0.4)
                    for i, p in enumerate(vents))
    body += _wind(0, wind, flare) + _wind(1, wind, flare)
    body += _layer(0, cell, *FIRE_MAP, True, 0.25, 4.0) + _layer(1, cell, *STEAM_MAP, False, 1.2, 1.5)
    body = re.sub(r"(\S) \}$", "\\1\n}", body, flags=re.M)
    return f'#usda 1.0\n(\n    defaultPrim = "Flow"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\ndef Xform "Flow"\n{{\n{body}}}\n'
