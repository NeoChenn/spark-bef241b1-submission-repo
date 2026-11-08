import math

import numpy as np
import pytest

from sim.atmos import (
    KINDS,
    TIERS,
    Particles,
    fog_cards,
    parse_schedule,
    schedule_at,
    streak_points,
    volumetric_settings,
    weather_state,
    wind_lean,
)


def test_kinds_match_storm_weather_names():
    assert set(KINDS) == {"calm", "wind", "rain", "fog", "storm"}


def test_weather_state_from_kind():
    calm, storm = weather_state("calm"), weather_state("storm")
    assert calm["rain"] == 0 and storm["rain"] == 1
    assert storm["wind_speed"] > calm["wind_speed"]
    assert storm["spray"] > calm["spray"]
    assert weather_state("fog")["fog"] > weather_state("wind")["fog"]


def test_weather_state_float_level_interpolates_calm_to_storm():
    mid = weather_state(0.5)
    assert weather_state("calm")["wind_speed"] < mid["wind_speed"] < weather_state("storm")["wind_speed"]
    assert mid["rain"] == pytest.approx(0.5)
    assert weather_state(3.0) == weather_state(1.0)


def test_wind_dir_and_gust_override():
    w = weather_state("wind", wind_dir=90.0)
    assert w["wind"][0] == pytest.approx(0.0, abs=1e-9)
    assert w["wind"][1] == pytest.approx(w["wind_speed"])
    assert weather_state("wind", gust=36.0)["wind_speed"] == pytest.approx(2 * weather_state("wind", gust=18.0)["wind_speed"])


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        weather_state("hail")


def test_schedule_parse_and_lookup():
    sched = parse_schedule("calm@0,storm@20,fog@35")
    assert sched == [(0.0, "calm"), (20.0, "storm"), (35.0, "fog")]
    assert schedule_at(sched, 5.0) == "calm"
    assert schedule_at(sched, 20.0) == "storm"
    assert schedule_at(sched, 99.0) == "fog"
    assert parse_schedule("rain") == [(0.0, "rain")]
    assert parse_schedule("0.4@0,storm@9") == [(0.0, 0.4), (9.0, "storm")]
    assert parse_schedule("") == []
    assert schedule_at([], 3.0) is None


def test_tiers_live_is_cheaper_than_offline():
    live, off = TIERS["live"], TIERS["offline"]
    assert live["rain"] < off["rain"] and live["embers"] < off["embers"]
    assert live["fog_cards"] <= off["fog_cards"]
    assert off["width"] == 1920 and live["width"] == 1280


def test_volumetric_settings_enable_media_and_scale_with_fog():
    calm = volumetric_settings("night", "live", weather_state("calm"))
    fog = volumetric_settings("night", "live", weather_state("fog"))
    key = "/rtx/raytracing/inscattering/transmittanceMeasurementDistance"
    assert calm["/rtx/raytracing/globalVolumetricEffects/enabled"] is True
    assert 0 < fog[key] < calm[key]
    assert volumetric_settings("night", "offline", weather_state("calm"))["/rtx/raytracing/inscattering/depthSlices"] > calm[
        "/rtx/raytracing/inscattering/depthSlices"]
    assert len(calm["/rtx/raytracing/inscattering/transmittanceColor"]) == 3


def test_wind_lean_bends_with_height_and_speed():
    still = wind_lean((0.0, 0.0), 10.0)
    assert still == (0.0, 0.0)
    lo, hi = wind_lean((5.0, 0.0), 2.0), wind_lean((5.0, 0.0), 10.0)
    assert hi[0] > lo[0] > 0 and hi[1] == 0
    assert math.hypot(*wind_lean((80.0, 0.0), 10.0)) <= 10.0


def test_particles_fall_and_respawn_inside_box():
    box = (np.array([-5.0, -5.0, 0.0]), np.array([5.0, 5.0, 10.0]))
    p = Particles(200, box, velocity=(0.0, 0.0, -9.0), seed=1)
    assert p.pos.shape == (200, 3)
    for _ in range(100):
        p.step(0.05)
    assert (p.pos >= box[0] - 1e-6).all() and (p.pos <= box[1] + 1e-6).all()
    assert p.vel[:, 2].mean() < -5


def test_particles_wind_drifts_and_gravity_pulls():
    box = (np.array([-50.0, -50.0, 0.0]), np.array([50.0, 50.0, 100.0]))
    p = Particles(500, box, velocity=(0.0, 0.0, 3.0), gravity=-1.0, life=(50.0, 60.0), seed=2)
    x0 = p.pos[:, 0].copy()
    p.step(0.5, wind=(4.0, 0.0))
    assert np.median(p.pos[:, 0] - x0) == pytest.approx(2.0)
    assert np.median(p.vel[:, 2]) < 3.0


def test_particles_life_expires_and_resets_age():
    box = (np.zeros(3), np.ones(3))
    p = Particles(50, box, life=(0.1, 0.2), seed=3)
    p.step(0.5)
    assert (p.age < 0.2).all()


def test_particles_move_with_box():
    p = Particles(30, (np.zeros(3), np.ones(3)), seed=4)
    p.recentre(np.array([100.0, 0.0, 0.0]))
    p.step(0.01)
    assert p.pos[:, 0].min() > 99


def test_particles_source_sets_position_and_velocity():
    box = (np.array([-10.0, -10.0, 0.0]), np.array([10.0, 10.0, 10.0]))
    src = lambda rng, k: (np.tile([1.0, 2.0, 0.5], (k, 1)), np.tile([3.0, 0.0, 6.0], (k, 1)))
    p = Particles(20, box, gravity=-9.8, life=(5.0, 5.0), source=src, seed=6)
    np.testing.assert_allclose(p.vel, np.tile([3.0, 0.0, 6.0], (20, 1)))
    p.step(0.1)
    assert (p.pos[:, 0] > 1.0).all()


def test_streak_points_two_per_particle_along_velocity():
    pos = np.array([[0.0, 0.0, 10.0], [1.0, 1.0, 5.0]])
    vel = np.array([[0.0, 0.0, -10.0], [2.0, 0.0, 0.0]])
    pts = streak_points(pos, vel, 0.05)
    assert pts.shape == (4, 3)
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 10.5])
    np.testing.assert_allclose(pts[1], pos[0])
    np.testing.assert_allclose(pts[2], [0.9, 1.0, 5.0])


def test_fog_cards_low_and_wrap_while_drifting():
    cards = fog_cards(12, radius=400.0, z=(2.0, 14.0), seed=5)
    assert cards["centre"].shape == (12, 3)
    assert (cards["centre"][:, 2] >= 2.0).all() and (cards["centre"][:, 2] <= 14.0).all()
    moved = fog_cards(12, radius=400.0, z=(2.0, 14.0), seed=5, t=1e4, wind=(6.0, 0.0))
    assert (np.hypot(moved["centre"][:, 0], moved["centre"][:, 1]) <= 400.0 * math.sqrt(2) + 1e-6).all()
    assert not np.allclose(moved["centre"], cards["centre"])


def test_soft_texture_fades_to_clear_edges():
    from sim.atmos import soft_texture

    a = soft_texture(64, seed=1)
    assert a.shape == (64, 64) and a.dtype == np.uint8
    assert a[24:40, 24:40].mean() > 40
    assert a[0].max() == 0 and a[:, -1].max() == 0
    assert not np.array_equal(a, soft_texture(64, seed=2))


def test_flow_usda_has_fire_steam_layers_and_wind():
    from pxr import Sdf

    from sim.atmos import flow_usda

    text = flow_usda(flare=(-58.0, 8.0, 70.0), vents=[(0.0, 0.0, 40.0), (5.0, 5.0, 41.0)], wind=(6.0, -2.0), cell=0.3)
    layer = Sdf.Layer.CreateAnonymous("f.usda")
    assert layer.ImportFromString(text)
    flame = layer.GetPrimAtPath("/Flow/flare")
    assert flame.typeName == "FlowEmitterSphere"
    assert flame.attributes["layer"].default == 0
    assert [layer.GetPrimAtPath(f"/Flow/vent_{i}").attributes["layer"].default for i in range(2)] == [1, 1]
    assert layer.GetPrimAtPath("/Flow/wind_0").attributes["velocity"].default == (6.0, -2.0, 0.0)
    assert layer.GetPrimAtPath("/Flow/wind_0").attributes["allocationScale"].default == 0.0
    assert layer.GetPrimAtPath("/Flow/sim_1").attributes["densityCellSize"].default == pytest.approx(0.3)
    assert {p.typeName for p in layer.GetPrimAtPath("/Flow").nameChildren} >= {"FlowSimulate", "FlowOffscreen", "FlowRender"}


def test_billboard_quats_turn_minus_y_face_towards_eye():
    from sim.atmos import billboard_quats

    pos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    eyes = np.array([0.0, -50.0, 0.0])
    q = billboard_quats(pos, eyes)
    assert q.shape == (3, 4)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0)

    def rotate(qq, v):
        w, x, y, z = qq
        u = np.array([x, y, z])
        return v + 2 * np.cross(u, np.cross(u, v) + w * v)

    for p, qq in zip(pos, q):
        to_eye = (eyes - p) / np.linalg.norm(eyes - p)
        np.testing.assert_allclose(rotate(qq, np.array([0.0, -1.0, 0.0])), to_eye, atol=1e-6)
    above = billboard_quats(np.zeros((1, 3)), np.array([0.0, -10.0, 10.0]))[0]
    np.testing.assert_allclose(rotate(above, np.array([0.0, -1.0, 0.0])), np.array([0, -1, 1]) / np.sqrt(2), atol=1e-6)


def test_fade_bucket_ramps_in_and_out():
    from sim.atmos import fade_bucket

    age = np.array([0.0, 0.5, 5.0, 9.9])
    ttl = np.full(4, 10.0)
    b = fade_bucket(age, ttl, 4)
    assert b.dtype.kind == "i" and b.min() >= 0 and b.max() <= 3
    assert b[2] == 3 and b[0] < b[2] and b[3] < b[2]
