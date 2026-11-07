import json
from pathlib import Path

import pytest

from sim.ocean_presets import PRESETS, get_preset, hdri_azimuth, render_settings

ASSETS = Path(__file__).resolve().parents[1] / "sim" / "assets"


def test_required_presets_exist_and_use_fetched_skies():
    manifest = json.loads((ASSETS / "ocean_manifest.json").read_text())
    assert {"clear", "golden", "overcast", "mood"} <= set(PRESETS)
    for p in PRESETS.values():
        assert p.sky is not None or p.hdri in manifest["hdris"]
        assert p.dome_intensity > 0 and p.wind_speed > 0
        assert "/rtx/post/motionblur/enabled" in render_settings(p)


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError):
        get_preset("stormy_mars")


def test_hdri_azimuth_wraps():
    assert hdri_azimuth(0.5) == pytest.approx(180.0)
    assert 0.0 <= hdri_azimuth(1.0) < 360.0


def test_fog_settings_and_sky_opacity():
    from sim.ocean_presets import Fog

    fog = Fog(color=(0.1, 0.1, 0.2), distance_density=0.5)
    s = fog.settings()
    assert s["/rtx/fog/enabled"] and s["/rtx/fog/fogColor"] == [0.1, 0.1, 0.2]
    assert fog.sky_opacity == pytest.approx(0.3935, abs=1e-3)
    assert render_settings(get_preset("clear"))["/rtx/fog/enabled"] is False
    assert render_settings(get_preset("mood"))["/rtx/fog/enabled"] is True
