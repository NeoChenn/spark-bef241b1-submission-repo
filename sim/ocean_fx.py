from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

try:
    from sim.ocean import FADE_DISTANCE_M, GRAVITY, WAVES
    from sim.ocean_waves import Waves, make_waves
except ImportError:
    from ocean import FADE_DISTANCE_M, GRAVITY, WAVES
    from ocean_waves import Waves, make_waves

OUTPUT = Path(__file__).resolve().parent / "output"
LAYER = OUTPUT / "ocean_fx.usda"
MDL_FILE = OUTPUT / "ocean_fx.mdl"
MODULE = "OceanFx"
MATERIAL = "/World/OceanFx/Looks/OceanFx"
SHADER = f"{MATERIAL}/Shader"
OCEAN = "/World/Cinematic/ocean"
UV_TILE_M = 26.0
WIND_DIR_DEG = 18.0
CAMERAS = ("/World/Judge/Cam", "/World/Cinematic/Cameras/stream")
REFERENCE_WIDTH_PX = 1280.0

FX = {
    "night": {"chop": 1.0, "slope_gain": 2.5, "coverage": 0.035, "foam_amount": 1.0, "leg_foam": 1.0, "roughness": 0.015,
              "deep_color": (0.0015, 0.007, 0.010), "crest_color": (0.006, 0.03, 0.032),
              "aerated_color": (0.03, 0.09, 0.085), "foam_color": (0.9, 0.93, 0.93)},
    "storm": {"chop": 1.45, "slope_gain": 3.0, "coverage": 0.09, "foam_amount": 1.25, "leg_foam": 1.35, "roughness": 0.03,
              "deep_color": (0.004, 0.012, 0.014), "crest_color": (0.012, 0.045, 0.045),
              "aerated_color": (0.05, 0.12, 0.11), "foam_color": (0.7, 0.74, 0.75)},
    "golden": {"chop": 0.85, "slope_gain": 2.0, "coverage": 0.015, "foam_amount": 0.9, "leg_foam": 0.9, "roughness": 0.015,
               "deep_color": (0.005, 0.02, 0.03), "crest_color": (0.02, 0.07, 0.07),
               "aerated_color": (0.06, 0.16, 0.15), "foam_color": (0.8, 0.8, 0.78)},
}


def big_waves() -> tuple[np.ndarray, np.ndarray]:
    n = len(WAVES)
    a, b = [], []
    for wavelength, relative, heading, steepness, phase in WAVES:
        k = math.tau / wavelength
        dx, dy = math.cos(math.radians(heading)), math.sin(math.radians(heading))
        a.append((dx, dy, k, relative))
        b.append((math.sqrt(GRAVITY * k), phase, 0.8 * min(steepness, 1.0) / n, 0.0))
    return np.array(a), np.array(b)


def detail_waves(wind_speed: float = 9.0, max_wavelength: float = 10.0, min_wavelength: float = 0.05,
                 choppiness: float = 0.7, seed: int = 7) -> Waves:
    w = make_waves(wind_speed, WIND_DIR_DEG, seed=seed, bands=24, per_band=4, min_wavelength=min_wavelength,
                   spread=0.6, choppiness=choppiness)
    keep = w.wavelength <= max_wavelength
    k, amp = w.k[keep], w.amp[keep]
    q = choppiness / (k * amp * keep.sum())
    return Waves(dir=w.dir[keep], k=k, omega=w.omega[keep], amp=amp, phase=w.phase[keep], q=q.astype(np.float32))


def jacobian(big: tuple[np.ndarray, np.ndarray], detail: Waves, chop: float, xy: np.ndarray, t: float) -> np.ndarray:
    a, b = big
    fade = np.exp(-np.linalg.norm(xy, axis=1) / FADE_DISTANCE_M)
    th = xy @ (a[:, :2] * a[:, 2:3]).T - (b[:, 0] * t - b[:, 1])
    qs = np.sin(th) * b[:, 2] * fade[:, None]
    jxx, jyy, jxy = qs @ a[:, 0] ** 2, qs @ a[:, 1] ** 2, qs @ (a[:, 0] * a[:, 1])
    th = xy @ (detail.dir * detail.k[:, None]).T - (detail.omega * t - detail.phase)
    qs = np.sin(th) * (detail.q * detail.k * detail.amp * chop)
    jxx += qs @ detail.dir[:, 0] ** 2
    jyy += qs @ detail.dir[:, 1] ** 2
    jxy += qs @ (detail.dir[:, 0] * detail.dir[:, 1])
    return (1 - jxx) * (1 - jyy) - jxy**2


def foam_threshold(big, detail: Waves, chop: float, coverage: float, samples: int = 12_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-150, 150, (samples, 2))
    return float(np.quantile(jacobian(big, detail, chop, xy, float(rng.uniform(0, 60))), coverage))


def _f(v: float) -> str:
    return f"{float(v):.7g}"


def _vecs(rows, kind: str = "float4") -> str:
    return ",\n    ".join(f"{kind}({', '.join(_f(x) for x in row)})" for row in rows)


def build_mdl(big, detail: Waves, legs) -> str:
    a, b = big
    da = np.column_stack([detail.dir, detail.k, detail.amp])
    db = np.column_stack([detail.omega, detail.phase, detail.q, np.zeros_like(detail.q)])
    slots = list(legs) or [(0.0, 0.0, 0.0)]
    drift = (math.cos(math.radians(WIND_DIR_DEG)), math.sin(math.radians(WIND_DIR_DEG)))
    template = (Path(__file__).resolve().parent / "ocean_fx.mdl.in").read_text()
    return (
        template.replace("@BIG_COUNT@", str(len(a)))
        .replace("@BIG_A@", _vecs(a))
        .replace("@BIG_B@", _vecs(b))
        .replace("@WAVE_COUNT@", str(len(da)))
        .replace("@WAVE_A@", _vecs(da))
        .replace("@WAVE_B@", _vecs(db))
        .replace("@LEG_COUNT@", str(len(legs)))
        .replace("@LEG_SLOTS@", str(len(slots)))
        .replace("@LEGS@", _vecs(slots, "float3"))
        .replace("@DRIFT@", f"float2({_f(drift[0])}, {_f(drift[1])})")
        .replace("@FADE@", _f(FADE_DISTANCE_M))
    )


def preset_inputs(name: str, height: float, threshold: float) -> dict:
    fx = FX.get(name, FX["night"])
    return {"height": float(height), "chop": fx["chop"], "slope_gain": fx["slope_gain"], "foam_threshold": threshold,
            "foam_amount": fx["foam_amount"], "leg_foam": fx["leg_foam"], "roughness": fx["roughness"],
            "deep_color": fx["deep_color"], "crest_color": fx["crest_color"],
            "aerated_color": fx["aerated_color"], "foam_color": fx["foam_color"]}


def pixel_angle(aperture_mm: float, focal_mm: float, width_px: float = REFERENCE_WIDTH_PX) -> float:
    return aperture_mm / max(focal_mm, 1e-3) / width_px


def wave_stride(quality: str) -> int:
    return 1 if quality == "offline" else 3


def moved(last, new, tol: float) -> bool:
    return last is None or max(abs(a - b) for a, b in zip(last, new)) > tol


def parse_overrides(text: str) -> dict:
    return {k.strip(): float(v) for k, v in (item.split("=") for item in text.split(",") if "=" in item)}


_state: dict = {}


def attach(stage) -> bool:
    if not LAYER.exists() or os.environ.get("RIG_OCEAN_FX", "1") == "0":
        return False
    root = stage.GetRootLayer()
    target = os.path.relpath(LAYER, Path(root.realPath).parent) if root.realPath else str(LAYER)
    if not any(Path(p).name == LAYER.name for p in root.subLayerPaths):
        root.subLayerPaths.insert(0, target)
    return True


def apply(stage, name: str, height: float) -> bool:
    from pxr import Gf, Sdf, UsdShade

    _state.clear()
    if not attach(stage):
        return False
    shader = UsdShade.Shader.Get(stage, SHADER)
    if not shader:
        return False
    fx = FX.get(name, FX["night"])
    threshold = foam_threshold(big_waves(), detail_waves(), fx["chop"], fx["coverage"])
    for key, value in preset_inputs(name, height, threshold).items():
        if isinstance(value, tuple):
            shader.CreateInput(key, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*value))
        else:
            shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(float(value))
    quality = os.environ.get("RIG_QUALITY", "live")
    shader.CreateInput("wave_stride", Sdf.ValueTypeNames.Int).Set(wave_stride(quality))
    shader.CreateInput("live", Sdf.ValueTypeNames.Bool).Set(quality != "offline")
    shader.CreateInput("debug", Sdf.ValueTypeNames.Float).Set(float(os.environ.get("RIG_OCEAN_FX_DEBUG", "0")))
    for key, value in parse_overrides(os.environ.get("RIG_OCEAN_FX_INPUTS", "")).items():
        shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(value)
    print(f"ocean_fx: preset={name} height={height} threshold={threshold:.3f}", flush=True)
    _state.update(stage=stage, time=shader.CreateInput("time", Sdf.ValueTypeNames.Float),
                  cam=shader.CreateInput("camera_position", Sdf.ValueTypeNames.Float3),
                  angle=shader.CreateInput("pixel_angle", Sdf.ValueTypeNames.Float))
    return True


def animate(stage, time: float) -> None:
    from pxr import Gf, Usd, UsdGeom

    if "stage" not in _state or _state["stage"] != stage:
        return
    _state["time"].Set(float(time))
    for path in CAMERAS:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            cam = UsdGeom.Camera(prim)
            eye = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            eye = tuple(float(v) for v in eye)
            if moved(_state.get("last_eye"), eye, 1.0):
                _state["cam"].Set(Gf.Vec3f(*eye))
                _state["last_eye"] = eye
            angle = (pixel_angle(cam.GetHorizontalApertureAttr().Get(), cam.GetFocalLengthAttr().Get()),)
            if moved(_state.get("last_angle"), angle, angle[0] * 0.02):
                _state["angle"].Set(angle[0])
                _state["last_angle"] = angle
            if os.environ.get("RIG_OCEAN_FX_DEBUG"):
                print(f"ocean_fx: t={time:.2f} cam={path} eye={tuple(round(v, 1) for v in eye)}", flush=True)
            return
