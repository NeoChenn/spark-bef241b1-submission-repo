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
    crest_color: tuple[float, float, float] = (0.012, 0.055, 0.05)
    foam_coverage: float = 3.0
    foam_amount: float = 1.0
    roughness: float = 0.03
    haze_distance: float = 20000.0
    haze_gain: float = 1.0
    sun: tuple[float, float, float, tuple[float, float, float]] | None = None
    render: dict = field(default_factory=dict)


BASE_RENDER = {
    "/rtx/post/motionblur/enabled": False,
    "/rtx/post/dof/enabled": False,
    "/rtx/post/lensFlares/enabled": False,
}

def hdri_azimuth(u: float) -> float:
    """World azimuth (deg, from +X towards +Y) of lat-long column u on an unrotated RTX dome in a Z-up stage."""
    return (360.0 * (1.0 - u)) % 360.0


PRESETS: dict[str, Preset] = {
    "clear": Preset(
        hdri="kloofendal_48d_partly_cloudy_puresky",
        dome_intensity=900.0,
        wind_speed=8.0,
        swell=(1.2, 120.0, 10.0),
    ),
    "dusk": Preset(
        hdri="industrial_sunset_puresky",
        dome_intensity=650.0,
        wind_speed=6.0,
        swell=(1.0, 140.0, 200.0),
        deep_color=(0.003, 0.012, 0.022),
        haze_distance=12000.0,
    ),
    "golden": Preset(
        hdri="belfast_sunset_puresky",
        dome_intensity=500.0,
        wind_speed=6.0,
        swell=(1.0, 140.0, 200.0),
        deep_color=(0.003, 0.012, 0.022),
        haze_distance=12000.0,
        sun=(hdri_azimuth(0.605), 2.1, 4000.0, (1.0, 0.55, 0.25)),
    ),
    "overcast": Preset(
        hdri="overcast_soil_puresky",
        dome_intensity=750.0,
        wind_speed=14.0,
        swell=(2.0, 160.0, 300.0),
        choppiness=0.95,
        foam_coverage=2.0,
        deep_color=(0.006, 0.016, 0.017),
        crest_color=(0.02, 0.06, 0.05),
        roughness=0.06,
        haze_distance=5000.0,
    ),
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]


def render_settings(preset: Preset) -> dict:
    return {**BASE_RENDER, **preset.render}
