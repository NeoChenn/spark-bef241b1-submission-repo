import math

import pytest

from sim import weather as wx


def test_levels_and_clamp():
    assert wx.level_of("calm") == 0.0 and wx.level_of("storm") == 1.0
    assert wx.level_of(1.7) == 1.0 and wx.level_of(-1) == 0.0
    with pytest.raises(ValueError):
        wx.level_of("hail")


def test_target_scales_sea_and_wind():
    calm, storm = wx.target(0.0), wx.target(1.0)
    assert calm["wave_height"] == 1.5 and storm["wave_height"] == wx.STORM_HEIGHT
    assert storm["chop"] > calm["chop"] and storm["coverage"] > calm["coverage"]
    assert math.isclose(math.hypot(*calm["wind"]), wx.CALM_WIND)
    assert math.isclose(math.hypot(*storm["wind"]), wx.STORM_WIND)
    gusty = wx.target(1.0, wind_dir=90.0, gust=36.0)
    assert math.isclose(gusty["wind"][1], 2 * wx.STORM_WIND) and abs(gusty["wind"][0]) < 1e-9
    assert wx.target(1.0, base_height=3.0)["wave_height"] == 3.0


def test_wetness_rises_with_rain():
    assert wx.wet(0.4, 0.2, 0.0) == (0.4, 0.2)
    w, p = wx.wet(0.4, 0.2, 1.0)
    assert math.isclose(w, 0.4 + 0.6 * 0.85) and p == 1.0


def test_ramp_eases_from_current_state():
    r = wx.Ramp({"a": 0.0, "v": (0.0, 0.0)}, 0.0, duration=4.0)
    r.set({"a": 1.0, "v": (2.0, 4.0)}, 10.0)
    assert r.at(10.0)["a"] == 0.0 and r.at(14.0)["a"] == 1.0
    mid = r.at(12.0)
    assert math.isclose(mid["a"], 0.5) and mid["v"] == (1.0, 2.0)
    r.set({"a": 0.0, "v": (0.0, 0.0)}, 12.0)
    assert math.isclose(r.at(12.0)["a"], 0.5) and not r.done(15.9) and r.done(16.0)


def test_flare_wind_default():
    assert wx.flare_wind((2.6, 1.5)) == (2.6, 1.5)


def test_named_conditions_split_rain_and_fog():
    fog, rain = wx.target("fog"), wx.target("rain")
    assert fog["fog"] == 1.0 and fog["rain"] == 0.0 and fog["fog_height"] > rain["fog_height"]
    assert rain["rain"] == 1.0 and rain["wave_height"] > wx.target("calm")["wave_height"]
    base = {"height": 0.004, "dist": 0.001, "color": (0.3, 0.2, 0.1)}
    calm = wx.target("calm", fog_base=base)
    assert calm["fog_height"] == 0.004 and calm["fog_color"] == (0.3, 0.2, 0.1)


def test_fog_rolls_only_when_foggy():
    assert wx.roll(3.0, 0.0) == 1.0
    vals = [wx.roll(t, 1.0) for t in range(0, 60, 3)]
    assert max(vals) > 1.0 > min(vals) and all(0.7 < v < 1.3 for v in vals)


def test_rain_and_fog_combine_and_level_toggles():
    assert wx.mix(level="calm") == (0.0, 0.0, 0.0)
    both = wx.mix(rain=1.0, fog=1.0)
    assert both == (0.6, 1.0, 1.0)
    assert wx.mix(wx.mix(level="rain"), level="fog") == (0.6, 1.0, 1.0)
    assert wx.mix(both, level="calm") == (0.0, 0.0, 0.0)
    assert wx.mix(level="storm") == (1.0, 1.0, 0.6)
    assert wx.mix(level=0.5) == (0.5, 0.5, 0.3)
    t = wx.target(both)
    assert t["rain"] == 1.0 and t["fog"] == 1.0 and math.isclose(t["fog_dist"], wx.FOG["dist"][1])


def test_parse_default_weather_env():
    assert wx.parse_default("") == {} and wx.parse_default("calm@0,storm@20") == {}
    assert wx.parse_default("rain+fog") == {"rain": 1.0, "fog": 1.0}
    assert wx.parse_default("rain=1,fog=0.5") == {"rain": 1.0, "fog": 0.5}
    assert wx.parse_default("storm") == {"level": "storm"} and wx.parse_default("0.7") == {"level": 0.7}
    assert wx.atmos_kind("rain+fog") == "rain" and wx.atmos_kind("fog=1") == "fog" and wx.atmos_kind("storm") == "storm"


def test_rain_curtain_wraps_around_camera():
    for t in (0.0, 1.3, 3.33, 10.0):
        x, y, z = wx.curtain_offset((5.0, -2.0, 40.0), t)
        assert (x, y) == (5.0, -2.0)
        lo, hi = z, z + 2 * wx.RAIN_BOX[1]
        assert lo <= 40.0 - wx.RAIN_BELOW and hi >= 40.0 + 9.0
