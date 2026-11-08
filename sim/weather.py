from __future__ import annotations

import math

LEVELS = {"calm": (0.0, 0.0, 0.0), "fog": (0.15, 0.0, 1.0), "wind": (0.6, 0.0, 0.0), "rain": (0.6, 1.0, 0.3),
          "storm": (1.0, 1.0, 0.6)}
FOG = {"height": (0.002, 0.06), "dist": (0.0, 0.014), "color": ((0.05, 0.16, 0.18), (0.17, 0.21, 0.23))}
PRESET_LEVEL = {"storm": 1.0}
RAMP_S = 4.0
STORM_HEIGHT = 2.5
CALM_WIND, STORM_WIND, GUST_REF = 3.0, 18.0, 18.0
DEFAULT_DIR = 30.0
OCEAN = {"chop": (1.0, 1.45), "slope_gain": (2.5, 3.0), "coverage": (0.035, 0.09), "foam_amount": (1.0, 1.25),
         "leg_foam": (1.0, 1.35), "roughness": (0.015, 0.03)}
SKIP = ("/World/Looks/HeroRustOverlay", "/World/RepairZone/Looks/rust_coarse_01")
ATM_EVERY_S = 0.25

_flare = {"wind": None}


def components(level) -> tuple[float, float, float]:
    if isinstance(level, str):
        if level not in LEVELS:
            raise ValueError(f"unknown weather {level!r}")
        return LEVELS[level]
    u = min(max(float(level), 0.0), 1.0)
    return u, u, 0.6 * u


def level_of(level) -> float:
    return components(level)[0]


def lerp(a, b, s):
    if isinstance(a, tuple):
        return tuple(x + (y - x) * s for x, y in zip(a, b))
    return a + (b - a) * s


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def target(level, wind_dir=None, gust=None, base_height: float = 1.5, fog_base=None) -> dict:
    u, rain, fog = components(level)
    speed = lerp(CALM_WIND, STORM_WIND, u) * (float(gust) / GUST_REF if gust is not None else 1.0)
    a = math.radians(DEFAULT_DIR if wind_dir is None else float(wind_dir))
    out = {k: lerp(lo, hi, u) for k, (lo, hi) in OCEAN.items()}
    out.update(level=u, wave_height=lerp(base_height, max(base_height, STORM_HEIGHT), u), rain=rain, fog=fog,
               wind=(speed * math.cos(a), speed * math.sin(a)))
    base = fog_base or {k: v[0] for k, v in FOG.items()}
    out.update(fog_height=lerp(base["height"], max(base["height"], FOG["height"][1]), fog),
               fog_dist=lerp(base["dist"], max(base["dist"], FOG["dist"][1]), fog),
               fog_color=lerp(tuple(base["color"]), FOG["color"][1], fog))
    return out


def wet(base_wetness: float, base_puddles: float, rain: float) -> tuple[float, float]:
    return lerp(base_wetness, 1.0, 0.85 * rain), lerp(base_puddles, 1.0, rain)


class Ramp:
    def __init__(self, state: dict, now: float = 0.0, duration: float = RAMP_S):
        self.src, self.dst, self.t0, self.duration = dict(state), dict(state), now, duration

    def at(self, now: float) -> dict:
        s = ease((now - self.t0) / self.duration)
        return {k: lerp(self.src[k], v, s) for k, v in self.dst.items()}

    def set(self, state: dict, now: float) -> None:
        self.src, self.dst, self.t0 = self.at(now), dict(state), now

    def done(self, now: float) -> bool:
        return now - self.t0 >= self.duration


def flare_wind(default):
    return _flare["wind"] or default


class Controller:
    def __init__(self, stage, wave: dict | None, preset: str, atm=None):
        from pxr import UsdShade

        try:
            from sim import ocean_fx
        except ImportError:
            import ocean_fx
        self.stage, self.wave, self.atm, self.fx = stage, wave, atm, ocean_fx
        self.base_height = wave["height"] if wave and preset != "storm" else 1.5
        self.ocean = UsdShade.Shader.Get(stage, ocean_fx.SHADER)
        self.wet = []
        for prim in stage.Traverse():
            attr = prim.GetAttribute("inputs:wetness")
            if prim.GetTypeName() != "Shader" or not attr or attr.Get() is None:
                continue
            if any(str(prim.GetPath()).startswith(p) for p in SKIP):
                continue
            puddles = prim.GetAttribute("inputs:puddles")
            self.wet.append((attr, float(attr.Get()), puddles if puddles and puddles.Get() is not None else None,
                             float(puddles.Get()) if puddles and puddles.Get() is not None else 0.0))
        self.args, self.now, self.atm_at, self.live = {}, 0.0, -1.0, False
        self.fog_base = self._fog_base()
        start = self._target(PRESET_LEVEL.get(preset, 0.0), None, None)
        self.ramp = Ramp(start, 0.0)

    def _fog_base(self) -> dict:
        import carb

        st = carb.settings.get_settings()
        color = st.get("/rtx/fog/fogColor")
        return {"height": float(st.get("/rtx/fog/fogHeightDensity") or FOG["height"][0]),
                "dist": float(st.get("/rtx/fog/fogDistanceDensity") or 0.0),
                "color": tuple(float(c) for c in color) if color and len(color) == 3 else FOG["color"][0]}

    def _target(self, level, wind_dir, gust) -> dict:
        w = target(level, wind_dir, gust, self.base_height, self.fog_base)
        w["foam_threshold"] = self.fx.foam_threshold(self.fx.big_waves(), self.fx.detail_waves(), w["chop"], w["coverage"])
        return w

    def set(self, level=None, wind_dir=None, gust=None, time=None) -> None:
        if time:
            import cinematic

            if hasattr(cinematic, "set_time"):
                cinematic.set_time(self.stage, time)
                self.fog_base = self._fog_base()
                print(f"rig sim: time of day {time}", flush=True)
        if level is None:
            level = self.args.get("level", "calm")
        self.args = {"level": level, "wind_dir": wind_dir, "gust": gust}
        self.ramp.set(self._target(level, wind_dir, gust), self.now)
        self.live = True
        import photoreal

        if hasattr(photoreal, "set_rain"):
            photoreal.set_rain(self.stage, components(level)[1])
        print(f"rig sim: weather ramp to {level} ({level_of(level):.2f})", flush=True)

    def tick(self, t: float) -> None:
        from pxr import Sdf

        self.now = t
        if not self.live:
            return
        w = self.ramp.at(t)
        if self.wave is not None:
            self.wave["height"] = w["wave_height"]
        if self.ocean:
            for key in ("chop", "slope_gain", "foam_amount", "leg_foam", "roughness", "foam_threshold"):
                self.ocean.CreateInput(key, Sdf.ValueTypeNames.Float).Set(float(w[key]))
            self.ocean.CreateInput("height", Sdf.ValueTypeNames.Float).Set(float(w["wave_height"]))
        for attr, base, puddles, base_p in self.wet:
            wv, pv = wet(base, base_p, w["rain"])
            attr.Set(wv)
            if puddles is not None:
                puddles.Set(pv)
        _flare["wind"] = tuple(w["wind"])
        import carb

        st = carb.settings.get_settings()
        st.set_bool("/rtx/fog/enabled", True)
        st.set_float("/rtx/fog/fogHeightDensity", float(w["fog_height"]))
        st.set_float("/rtx/fog/fogDistanceDensity", float(w["fog_dist"]))
        st.set_float_array("/rtx/fog/fogColor", [float(c) for c in w["fog_color"]])
        if self.atm is not None and not isinstance(self.args.get("level"), str) and t - self.atm_at >= ATM_EVERY_S:
            self.atm_at = t
            self.atm.set_weather(level=w["level"], wind_dir=self.args.get("wind_dir"), gust=self.args.get("gust"))
        if self.ramp.done(t):
            self.live = False
