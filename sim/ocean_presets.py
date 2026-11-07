"""Look presets for sim/environment.py: sky, sun, sea state, water colour and renderer settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    hdri: str
    dome_intensity: float
    dome_rotate_z: float = 0.0
    wind_speed: float = 8.0
    wind_dir_deg: float = 30.0
    swell: tuple[float, float, float] | None = None
    choppiness: float = 0.8
    deep_color: tuple[float, float, float] = (0.004, 0.016, 0.03)
    crest_color: tuple[float, float, float] = (0.02, 0.09, 0.08)
    foam_threshold: float = 0.35
    foam_amount: float = 1.0
    roughness: float = 0.03
    haze_distance: float = 0.0
    haze_radiance: tuple[float, float, float] = (0.0, 0.0, 0.0)
    render: dict = field(default_factory=dict)


BASE_RENDER = {
    "/rtx/post/motionblur/enabled": False,
    "/rtx/post/dof/enabled": False,
    "/rtx/post/lensFlares/enabled": False,
}

PRESETS: dict[str, Preset] = {
    "clear": Preset(
        hdri="kloofendal_48d_partly_cloudy_puresky",
        dome_intensity=1000.0,
        wind_speed=8.0,
        swell=(1.2, 120.0, 10.0),
    ),
    "golden": Preset(
        hdri="industrial_sunset_puresky",
        dome_intensity=1000.0,
        wind_speed=6.0,
        swell=(1.0, 140.0, 200.0),
        deep_color=(0.003, 0.012, 0.022),
    ),
    "overcast": Preset(
        hdri="overcast_soil_puresky",
        dome_intensity=1000.0,
        wind_speed=14.0,
        swell=(2.0, 160.0, 300.0),
        choppiness=0.95,
        deep_color=(0.006, 0.016, 0.017),
        crest_color=(0.02, 0.06, 0.05),
        foam_threshold=0.45,
        roughness=0.06,
    ),
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]


def render_settings(preset: Preset) -> dict:
    return {**BASE_RENDER, **preset.render}
