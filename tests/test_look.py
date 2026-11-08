import numpy as np
import pytest

from sim import look


def test_ev_clamps_and_hits_keys():
    keys = look.EV_KEYS
    assert look.ev_at(0.1) == keys[0][1]
    assert look.ev_at(1e5) == keys[-1][1]
    for d, ev in keys:
        assert look.ev_at(d) == pytest.approx(ev)


def test_ev_brighter_when_farther():
    evs = [look.ev_at(d) for d in (1, 3, 10, 40, 150, 400)]
    assert evs == sorted(evs)


def test_exposure_doubles_iso_per_stop():
    assert look.exposure(0.0)["/rtx/post/tonemap/filmIso"] == pytest.approx(look.BASE_ISO)
    assert look.exposure(1.0)["/rtx/post/tonemap/filmIso"] == pytest.approx(2 * look.BASE_ISO)
    assert look.exposure(-2.0)["/rtx/post/tonemap/filmIso"] == pytest.approx(look.BASE_ISO / 4)


def test_rtx_settings_per_tier():
    assert look.rtx_settings("none", 1.0) == {}
    s = look.rtx_settings("offline", -1.0)
    assert s["/rtx/post/tonemap/filmIso"] == pytest.approx(look.BASE_ISO / 2)
    assert "/rtx/post/tonemap/op" in s
    live = look.rtx_settings("live")
    assert not any("motionblur/enabled" in k and v for k, v in live.items())


def test_view_look_table_and_fallback():
    for name in ("wide", "deck_mid", "repair", "waterline", "flare"):
        assert "ev" in look.view_look(name, 10.0)
    assert look.view_look(None, 330.0)["ev"] == pytest.approx(look.ev_at(330.0))
    assert look.view_look("repair", 2.0)["ev"] < look.view_look("wide", 2.0)["ev"]


def test_lens():
    assert look.lens(2.5, 2.8) == {"fStop": pytest.approx(2.8 * look.LENS_SCALE), "focusDistance": 2.5}
    assert look.lens(9.0, 0.0)["fStop"] == 0.0


def test_curves_monotone_split_tone():
    c = look.curves(look.TIERS["offline"]["grade"]).astype(int)
    assert c.shape == (3, 256)
    assert (np.diff(c, axis=1) >= 0).all()
    assert c[2, 70] > c[0, 70]
    assert c[0, 200] > c[2, 200]
    assert c[:, 4].max() <= 5


def test_grade_none_is_identity():
    img = np.full((4, 6, 3), 77, np.uint8)
    assert look.grade(img, "none") is img


@pytest.mark.parametrize("tier", ["live", "offline"])
def test_grade_vignette_and_dtype(tier):
    img = np.full((90, 160, 3), 128, np.uint8)
    out = look.grade(img, tier, 0.5)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out[0, 0].mean() < out[45, 80].mean()


def test_grade_offline_deterministic_and_live_drops_alpha():
    img = (np.random.default_rng(0).random((32, 48, 4)) * 255).astype(np.uint8)
    assert np.array_equal(look.grade(img, "offline", 1.25), look.grade(img, "offline", 1.25))
    assert look.grade(img, "live").shape == (32, 48, 3)


def test_fixture_retunes_colour_and_shapes_rects():
    r = look.fixture("rect", 4000.0, 100.0)
    assert r["kelvin"] == look.KELVIN[4000.0]
    assert r["intensity"] == pytest.approx(100.0 * look.FIXTURES["rect"]["gain"])
    assert r["cone"] > 0 and 0 < r["softness"] <= 1
    s = look.fixture("spot", 2000.0, 10.0, 30.0)
    assert s["cone"] == 30.0 and s["kelvin"] < 2500
    assert look.fixture("spot", 2000.0, 10.0, 60.0)["cone"] == look.FIXTURES["spot"]["cone_max"]
    b = look.fixture("bulb", 6000.0, 5.0)
    assert "cone" not in b and b["kelvin"] > 6000


def test_work_light_aims_at_target_from_above():
    pos, aim = look.work_light((1.0, 2.0, 3.0))
    assert pos[2] > 3.0
    assert np.allclose(np.add(pos, np.multiply(aim, np.linalg.norm(look.WORK_LIGHT["offset"]))), (1.0, 2.0, 3.0))
    assert np.linalg.norm(aim) == pytest.approx(1.0)


def test_atmos_darkens_far_views_only():
    assert look.view_look("wide", 250.0, True)["ev"] < look.view_look("wide", 250.0)["ev"]
    assert look.view_look("repair", 1.9, True)["ev"] == look.view_look("repair", 1.9)["ev"]
    assert look.view_look(None, 330.0, True)["ev"] == pytest.approx(look.ev_at(330.0) + look.ATMOS_EV_KEYS[-1][1])
    assert look.view_look(None, 3.0, True)["ev"] == pytest.approx(look.ev_at(3.0))


def test_preset_exposure_offsets():
    night = look.view_look("wide", 250.0, True)["ev"]
    assert look.view_look("wide", 250.0, True, "storm")["ev"] == pytest.approx(night + look.PRESET_EV["storm"])
    assert look.view_look("wide", 250.0, True, "golden")["ev"] > night


def test_dusk_grade_warmer_highlights():
    img = np.full((8, 8, 3), 200, np.uint8)
    night, dusk = look.grade(img, "live", 0.0), look.grade(img, "live", 0.0, "dusk")
    assert int(dusk[4, 4, 0]) - int(dusk[4, 4, 2]) > int(night[4, 4, 0]) - int(night[4, 4, 2])


def test_dusk_exposure_darkens_far_hazy_views_and_lifts_close_ones():
    assert look.view_look("wide", 250.0, True, "dusk")["ev"] < look.view_look("wide", 250.0, True)["ev"]
    assert look.view_look("repair", 1.9, True, "dusk")["ev"] > look.view_look("repair", 1.9, True)["ev"]


def test_sun_direction_points_at_sky_source():
    d = look.sun_direction((85.0, 0.0, 205.0))
    assert np.linalg.norm(d) == pytest.approx(1.0)
    assert 0.05 < d[2] < 0.12
    az = np.degrees(np.arctan2(d[1], d[0])) % 360
    assert az == pytest.approx(115.0, abs=1.0)
    assert look.sun_direction((0.0, 0.0, 0.0))[2] == pytest.approx(1.0)


def test_bloom_threshold_raised():
    b = look.bloom(0.1)
    assert b["/rtx/post/lensFlares/cutoffPoint"] == [look.BLOOM_CUTOFF] * 3


def _cam(eye, target, w=320, h=180):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rig.camera import look_at

    return look_at(eye, target, 24.0, 20.955, w, h)


def test_inscatter_glows_toward_sun_and_with_distance():
    img = np.full((180, 320, 3), 20, np.uint8)
    sun = look.sun_direction((85.0, 0.0, 205.0))
    eye = (0.0, 0.0, 10.0)
    toward = _cam(eye, np.add(eye, np.multiply(sun, 100.0)))
    away = _cam(eye, np.subtract(eye, np.multiply(sun, 100.0)))
    far = np.full((180, 320), 800.0, np.float32)
    near = np.full((180, 320), 3.0, np.float32)
    a = look.inscatter(img, far, toward, "dusk", sun)
    b = look.inscatter(img, far, away, "dusk", sun)
    c = look.inscatter(img, near, toward, "dusk", sun)
    assert a.shape == img.shape and a.dtype == np.uint8
    assert a.mean() > b.mean() > img.mean()
    assert c.mean() < a.mean()
    assert a[..., 0].mean() > a[..., 2].mean()
    assert look.inscatter(img, far, toward, "night", sun) is img
    assert look.inscatter(img, None, toward, "dusk", sun) is img


def test_live_tier_skips_haze_without_reading_depth():
    img = np.full((180, 320, 3), 20, np.uint8)
    sun = look.sun_direction((85.0, 0.0, 205.0))
    cam = _cam((0.0, 0.0, 10.0), np.add((0.0, 0.0, 10.0), np.multiply(sun, 100.0)))

    def boom():
        raise AssertionError("depth read")

    assert look.inscatter(img, boom, cam, "dusk", sun, tier="live") is img
    out = look.inscatter(img, lambda: np.full((180, 320), 800.0, np.float32), cam, "dusk", sun)
    assert out.mean() > img.mean()
