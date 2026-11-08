import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import hero_mat as hm

META = json.loads((Path(__file__).resolve().parents[1] / "sim/output/rig_merged_meta.json").read_text())


def test_wrap_scale_tiles_the_circumference_exactly():
    s = hm.wrap_scale(0.15, 0.32)
    n = 2 * math.pi * 0.15 / s
    assert abs(n - round(n)) < 1e-9 and abs(s - 0.32) < 0.1


def test_wrap_scale_never_below_one_tile():
    assert hm.wrap_scale(0.01, 5.0) == pytest.approx(2 * math.pi * 0.01)


def test_patch_st_encodes_ring_fraction():
    st = hm.patch_st(1 + 8 * 48)
    assert st[0] == (0.0, 0.0) and st[1] == (0.125, 0.0) and st[-1] == (1.0, 47 / 48)
    assert len(st) == 385


def test_patch_st_rejects_unknown_topology():
    with pytest.raises(ValueError):
        hm.patch_st(100)


def test_night_is_wet_with_puddles_and_golden_dry():
    night = hm.weather("night")
    assert 0.5 < night["wetness"] <= 1.0 and 0.5 < night["puddles"] <= 1.0
    assert hm.weather("golden") == {"wetness": 0.0, "puddles": 0.0}


def test_pipe_stays_olive_for_the_colour_classifier():
    assert not hm.warm(hm.PIPE_TINT)
    assert not hm.warm((0.05, 0.074, 0.034))


def test_recipes_cover_every_target_and_textures_exist():
    r = hm.recipes(META)
    assert set(hm.TARGETS.values()) <= set(r)
    for sub, values in r.values():
        assert sub in hm.SUBS
        for v in values.values():
            if isinstance(v, Path) and v != hm.NOISE:
                assert v.exists(), v


def test_pipe_recipe_tracks_the_damage_spot():
    _, v = hm.recipes(META)["pipe"]
    assert v["damage"] == tuple(META["world"]["damage"]["position_m"])
    assert v["pipe_z"] == META["world"]["main_pipe"]["z_m"]


def test_noise_is_tileable_and_normalised():
    n = hm.noise_rgb(64, seed=3)
    assert n.shape == (64, 64, 3) and n.min() >= 0.0 and n.max() <= 1.0
    edge = abs(n[0] - n[-1]).mean()
    inner = abs(n[0] - n[32]).mean()
    assert edge < inner


def test_mdl_exports_every_sub_identifier():
    text = hm.MDL.read_text()
    for sub in hm.SUBS:
        assert f"export material {sub}(" in text


def test_structure_paints_are_muted_and_wet():
    r = hm.recipes(META)
    for kind in ("frame", "zone_steel"):
        sub, v = r[kind]
        assert sub == "rig_hero_paint" and max(v["paint_color"]) < 0.3 and kind in hm.WET_PUDDLES
