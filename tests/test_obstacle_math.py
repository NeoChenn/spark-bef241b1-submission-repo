import math

import pytest

from rig.sim.obstacle_math import (NIGHT, STORM, dashes, drop_offset, flame_offset, ghost_alpha, seg_frame,
                                   segments, tick_index, weather_params)


def test_calm_is_night_and_storm_is_storm():
    calm, storm = weather_params("calm"), weather_params("storm")
    assert calm["fog_height"] == pytest.approx(NIGHT["fog_height"]) and calm["wave_height"] == NIGHT["wave_height"]
    assert storm["wave_height"] == pytest.approx(STORM["wave_height"]) and storm["fog_color"] == STORM["fog_color"]
    assert storm["storm_sky"] and not calm["storm_sky"] and calm["wind"] == 0.0
    assert calm["fog_dist"] == 0.0 < storm["fog_dist"] < weather_params("fog")["fog_dist"] and storm["sun"] > calm["sun"]


def test_fog_is_thick_but_calm_sea_and_wind_raises_waves():
    fog, wind = weather_params("fog"), weather_params("wind")
    assert fog["fog_height"] > weather_params("storm")["fog_height"] and fog["wave_height"] == NIGHT["wave_height"]
    assert wind["wave_height"] > NIGHT["wave_height"] and wind["rain"] == 0.0
    assert weather_params("wind", gust=36.0)["wave_height"] > wind["wave_height"]
    assert weather_params(0.5)["blend"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        weather_params("hail")


def test_flame_leans_downwind_more_at_the_tip():
    x0, y0, _ = flame_offset(2.0, 90.0, 1.0)
    x1, y1, _ = flame_offset(11.0, 90.0, 1.0)
    assert abs(x0) < 1e-9 and y1 > y0 > 0
    assert flame_offset(5.0, 0.0, 0.0) == (0.0, 0.0, -0.0)


def test_drop_falls_then_rests():
    assert drop_offset(0.0) == pytest.approx(1.5)
    assert 0 < drop_offset(0.35) < 1.5
    assert drop_offset(5.0) == 0.0


def test_segments_lift_and_skip_repeats():
    s = segments([[0, 0, 42.5], [0, 0, 42.5], [3, 0, 42.5], [3, 4, 42.5]], lift=0.1)
    assert len(s) == 2 and s[0][0][2] == pytest.approx(42.6)


def test_dashes_cover_segment_with_gaps():
    d = dashes((0, 0, 0), (10, 0, 0), 0.6, 0.4)
    assert len(d) == 10 and d[0] == ((0, 0, 0), (0.6, 0.0, 0.0))
    assert d[-1][1][0] == pytest.approx(9.6)


def test_seg_frame_yaw_pitch_length():
    mid, L, yaw, pitch = seg_frame((0, 0, 0), (0, 3, 4))
    assert L == pytest.approx(5.0) and yaw == pytest.approx(90.0) and pitch == pytest.approx(-math.degrees(math.atan2(4, 3)))
    assert mid == (0.0, 1.5, 2.0)


def test_tick_index_and_ghost_fade():
    ts = [1.0, 2.0, 3.0]
    assert tick_index(ts, 0.5) == 0 and tick_index(ts, 2.5) == 1 and tick_index(ts, 9) == 2
    assert ghost_alpha(1.0) == 1.0 and 0 < ghost_alpha(9.0) < 1 and ghost_alpha(20.0) == 0.0
