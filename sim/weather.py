from __future__ import annotations

import math

LEVELS = {"calm": (0.0, 0.0, 0.0), "fog": (0.15, 0.0, 1.0), "wind": (0.6, 0.0, 0.0), "rain": (0.6, 1.0, 0.3),
          "storm": (1.0, 1.0, 0.6)}
FOG = {"height": (0.002, 0.28), "dist": (0.0, 0.064), "color": ((0.05, 0.16, 0.18), (0.2, 0.25, 0.27))}
ROLL = 0.25
PRESET_LEVEL = {"storm": 1.0}
RAMP_S = 4.0
STORM_HEIGHT = 2.5
CALM_WIND, STORM_WIND, GUST_REF = 3.0, 18.0, 18.0
DEFAULT_DIR = 30.0
OCEAN = {"chop": (1.0, 1.45), "slope_gain": (2.5, 3.0), "coverage": (0.035, 0.09), "foam_amount": (1.0, 1.25),
         "leg_foam": (1.0, 1.35), "roughness": (0.015, 0.03)}
SKIP = ("/World/Looks/HeroRustOverlay", "/World/RepairZone/Looks/rust_coarse_01")
ATM_EVERY_S = 0.25
ROLL_EVERY_S = 0.25
RAIN_ROOT = "/World/Weather/rain"
RAIN_COUNT, RAIN_BOX, RAIN_STREAK, RAIN_WIDTH, RAIN_SPEED, RAIN_BELOW = 4200, (14.0, 24.0), (0.25, 0.8), 0.006, 9.0, 14.0
RAIN_CLEAR = 3.5

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


def mix(current=(0.0, 0.0, 0.0), level=None, rain=None, fog=None) -> tuple[float, float, float]:
    sea, r, f = 0.0, current[1], current[2]
    if level is not None:
        if level == "rain":
            r = 1.0
        elif level == "fog":
            f = 1.0
        else:
            sea, r, f = components(level)
    r = r if rain is None else min(max(float(rain), 0.0), 1.0)
    f = f if fog is None else min(max(float(fog), 0.0), 1.0)
    return max(sea, 0.6 * r, 0.15 * f), r, f


def parse_default(text: str) -> dict:
    text = (text or "").strip()
    if not text or "@" in text:
        return {}
    if "=" in text:
        return {k.strip(): float(v) for k, v in (p.split("=") for p in text.split(",") if "=" in p) if k.strip() in ("rain", "fog")}
    if "+" in text:
        return {part: 1.0 for part in text.split("+") if part in ("rain", "fog")}
    return {"level": float(text) if text.replace(".", "", 1).isdigit() else text}


def atmos_kind(text: str) -> str:
    d = parse_default(text)
    if "level" in d or not d:
        return text
    return "rain" if d.get("rain", 0.0) > 0 else ("fog" if d.get("fog", 0.0) > 0 else "calm")


def lerp(a, b, s):
    if isinstance(a, tuple):
        return tuple(x + (y - x) * s for x, y in zip(a, b))
    return a + (b - a) * s


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def target(level, wind_dir=None, gust=None, base_height: float = 1.5, fog_base=None) -> dict:
    u, rain, fog = level if isinstance(level, tuple) else components(level)
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


def roll(t: float, fog: float) -> float:
    return 1.0 + ROLL * fog * math.sin(t * 0.35) * math.sin(t * 0.13 + 1.0)


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
        self.mixed = mix(level=PRESET_LEVEL.get(preset, 0.0))
        self.curtain, self.roll_at = None, -1.0
        start = self._target(self.mixed, None, None)
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

    def set(self, level=None, wind_dir=None, gust=None, time=None, rain=None, fog=None) -> None:
        if time:
            import cinematic

            if hasattr(cinematic, "set_time"):
                cinematic.set_time(self.stage, time)
                self.fog_base = self._fog_base()
                print(f"rig sim: time of day {time}", flush=True)
        self.mixed = mix(self.mixed, level, rain, fog)
        self.args = {"level": level if level is not None else self.args.get("level"), "wind_dir": wind_dir, "gust": gust}
        self.ramp.set(self._target(self.mixed, wind_dir, gust), self.now)
        self.live = True
        import photoreal

        if hasattr(photoreal, "set_rain"):
            photoreal.set_rain(self.stage, self.mixed[1])
        print(f"rig sim: weather sea={self.mixed[0]:.2f} rain={self.mixed[1]:.2f} fog={self.mixed[2]:.2f}", flush=True)

    def _rain(self, t: float, amount: float) -> None:
        from pxr import Gf, Usd, UsdGeom

        if self.curtain is None:
            if amount < 0.1:
                return
            self.curtain = rain_curtain(self.stage)
        vis = "inherited" if amount >= 0.1 else "invisible"
        if self.curtain["vis"] != vis:
            self.curtain["vis"] = vis
            UsdGeom.Imageable(self.curtain["prim"]).GetVisibilityAttr().Set(vis)
        if vis == "invisible":
            return
        for path in self.fx.CAMERAS:
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid():
                eye = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
                x, y, z = curtain_offset(tuple(eye), t)
                self.curtain["op"].Set(Gf.Vec3d(x, y, z))
                return

    def tick(self, t: float) -> None:
        from pxr import Sdf

        self.now = t
        w = self.ramp.at(t) if (self.live or self.mixed[2] > 0.05 or self.mixed[1] > 0.05) else None
        if w is None:
            return
        self._rain(t, w["rain"])
        import carb

        st = carb.settings.get_settings()
        if w["fog"] > 0.05 and t - self.roll_at >= ROLL_EVERY_S:
            self.roll_at = t
            st.set_bool("/rtx/fog/enabled", True)
            st.set_float("/rtx/fog/fogHeightDensity", float(w["fog_height"]) * roll(t, w["fog"]))
            st.set_float("/rtx/fog/fogDistanceDensity", float(w["fog_dist"]))
            st.set_float_array("/rtx/fog/fogColor", [float(c) for c in w["fog_color"]])
        if not self.live:
            return
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
        st.set_bool("/rtx/fog/enabled", True)
        st.set_float("/rtx/fog/fogHeightDensity", float(w["fog_height"]) * roll(t, w["fog"]))
        st.set_float("/rtx/fog/fogDistanceDensity", float(w["fog_dist"]))
        st.set_float_array("/rtx/fog/fogColor", [float(c) for c in w["fog_color"]])
        if self.atm is not None and not isinstance(self.args.get("level"), str) and t - self.atm_at >= ATM_EVERY_S:
            self.atm_at = t
            self.atm.set_weather(level=w["level"], wind_dir=self.args.get("wind_dir"), gust=self.args.get("gust"))
        if self.ramp.done(t):
            self.live = False


def streaks(rng, count: int):
    import numpy as np

    top = shell(rng, count)
    length = rng.uniform(*RAIN_STREAK, count)
    drift = np.array([0.12, 0.07, -1.0])
    pts = np.stack([top, top + 0.5 * length[:, None] * drift, top + length[:, None] * drift], 1).reshape(-1, 3)
    widths = np.tile(np.array([0.15, 1.0, 0.35]) * RAIN_WIDTH, count) * np.repeat(rng.uniform(0.6, 1.3, count), 3)
    shade = rng.uniform(0.3, 0.75, count)
    return pts, widths, shade


def rain_curtain(stage, count: int = RAIN_COUNT, seed: int = 5) -> dict:
    import numpy as np
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    rng = np.random.default_rng(seed)
    root = UsdGeom.Xform.Define(stage, RAIN_ROOT)
    op = root.AddTranslateOp()
    curves = UsdGeom.BasisCurves.Define(stage, f"{RAIN_ROOT}/streaks")
    curves.CreateTypeAttr("linear")
    pts, widths, shade = streaks(rng, count)
    pts = np.concatenate([pts, pts + (0.0, 0.0, RAIN_BOX[1])])
    curves.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    curves.CreateCurveVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(2 * count, 3, np.int32)))
    curves.CreateWidthsAttr(Vt.FloatArray.FromNumpy(np.tile(widths, 2).astype(np.float32)))
    curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    tint = np.repeat(np.tile(shade, 2)[:, None], 3, 1) * np.array([0.95, 0.98, 1.0])
    UsdGeom.PrimvarsAPI(curves).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.uniform).Set(
        Vt.Vec3fArray.FromNumpy(tint.astype(np.float32)))
    mat = UsdShade.Material.Define(stage, f"{RAIN_ROOT}/look")
    sh = UsdShade.Shader.Define(stage, f"{RAIN_ROOT}/look/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    rd = UsdShade.Shader.Define(stage, f"{RAIN_ROOT}/look/tint")
    rd.CreateIdAttr("UsdPrimvarReader_float3")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.CreateOutput("result", Sdf.ValueTypeNames.Float3))
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.035, 0.037, 0.04))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    UsdShade.MaterialBindingAPI.Apply(curves.GetPrim()).Bind(mat)
    return {"prim": root.GetPrim(), "op": op, "vis": "inherited"}


def shell(rng, count: int, half: float = RAIN_BOX[0], height: float = RAIN_BOX[1], clear: float = RAIN_CLEAR):
    import numpy as np

    r = clear + (half - clear) * rng.uniform(0.0, 1.0, count) ** 1.5
    a = rng.uniform(0.0, 2 * np.pi, count)
    return np.column_stack([r * np.cos(a), r * np.sin(a), rng.uniform(0.0, height, count)])


def curtain_offset(eye, t: float) -> tuple[float, float, float]:
    fall = (t * RAIN_SPEED) % RAIN_BOX[1]
    return eye[0], eye[1], eye[2] - RAIN_BELOW - fall
