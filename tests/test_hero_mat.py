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


def test_stage_inputs_follow_the_repair():
    assert hm.stage_inputs("PATROL") == {"amount": 1.0, "wetness": 0.0}
    assert hm.stage_inputs("SPRAY", 0.0)["wetness"] == 0.4 and hm.stage_inputs("SPRAY", 1.0)["wetness"] == 1.0
    assert hm.stage_inputs("SOAK", 0.5) == {"amount": 1.0, "wetness": 1.0}
    scrub = [hm.stage_inputs("SCRUB", p)["amount"] for p in (0.0, 0.5, 1.0)]
    assert scrub[0] == 1.0 and scrub[0] > scrub[1] > scrub[2] == hm.PATCH_LEFT
    assert hm.stage_inputs("PAINT", 1.0)["amount"] == 0.0 and hm.stage_inputs("done")["amount"] == 0.0
    assert hm.stage_inputs("PATROL", base_wetness=0.64)["wetness"] == 0.64
    assert hm.stage_inputs("SCRUB", 7.0)["amount"] == hm.PATCH_LEFT


def test_overlay_rgba_packs_levels_and_primer_hides_rust():
    import numpy as np

    rust = np.array([[0.0, 0.5, 1.0, 1.0]])
    wet = np.array([[1.0, 0.0, 0.0, 0.0]])
    primer = np.array([[0.0, 0.0, 0.0, 1.0]])
    px = hm.overlay_rgba(rust, wet, primer)[0]
    assert px.dtype == np.uint8 and px.shape == (4, 4)
    assert tuple(px[0]) == (0, 255, 0, 255)
    assert tuple(px[1]) == (128, 0, 0, 128)
    assert tuple(px[3]) == (0, 0, 255, 255)
    assert tuple(hm.overlay_rgba(np.zeros((1, 1)))[0, 0]) == (0, 0, 0, 0)


def test_bind_overlay_and_set_state_on_a_stage():
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(stage, "/World/RustOverlay")
    hm.bind_overlay(stage, "/World/RustOverlay", "dynamic://rust_overlay", 0.154, 1.2)
    mat, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/World/RustOverlay")).ComputeBoundMaterial()
    assert mat.GetPath() == hm.OVERLAY_MATERIAL
    shader = UsdShade.Shader.Get(stage, f"{hm.OVERLAY_MATERIAL}/{hm.SHADER}")
    assert shader.GetSourceAssetSubIdentifier("mdl") == hm.OVERLAY_SUB
    assert shader.GetInput("mask").Get().path == "dynamic://rust_overlay"
    assert abs(shader.GetInput("uv_scale").Get()[0] - 2 * math.pi * 0.154) < 1e-5
    hm.set_state(stage, "SCRUB", 0.5)
    assert shader.GetInput("amount").Get() == pytest.approx(hm.stage_inputs("SCRUB", 0.5)["amount"])
    assert shader.GetInput("wetness").Get() == pytest.approx(hm.stage_inputs("SCRUB", 0.5)["wetness"])


def test_set_state_can_drive_wetness_only():
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(stage, "/World/RustOverlay")
    hm.bind_overlay(stage, "/World/RustOverlay", "dynamic://rust_overlay", 0.154, 1.2)
    assert hm.set_state(stage, "SCRUB", 0.9, keys=("wetness",)) == {"wetness": 0.75}
    shader = UsdShade.Shader.Get(stage, f"{hm.OVERLAY_MATERIAL}/{hm.SHADER}")
    assert shader.GetInput("amount").Get() == 1.0


def test_bind_overlay_accepts_a_file_mask():
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(stage, "/World/RustOverlay")
    hm.bind_overlay(stage, "/World/RustOverlay", "/tmp/mask.png", 0.154, 1.2)
    shader = UsdShade.Shader.Get(stage, f"{hm.OVERLAY_MATERIAL}/{hm.SHADER}")
    assert shader.GetInput("mask").Get().path == "/tmp/mask.png"


def test_tube_recipes_use_seamless_tube_mapping():
    r = hm.recipes(META)
    for kind in ("rail", "zone_pipe_grey", "zone_pipe_orange"):
        sub, v = r[kind]
        assert sub == "rig_hero_paint" and v["tube"] is True and v["tube_r"] > 0
    assert "/World/Looks/rail" in hm.TARGETS


def test_apply_turns_detail_off_for_the_live_tier(monkeypatch):
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(hm.LAYER))
    monkeypatch.setattr(hm, "attach", lambda s: True)
    live = hm.apply(stage, "night", "live")
    deck = UsdShade.Shader.Get(stage, f"/World/Looks/deck/{hm.SHADER}")
    assert live["detail"] is False and deck.GetInput("detail").Get() is False
    assert hm.apply(stage, "night", "offline")["detail"] is True and deck.GetInput("detail").Get() is True
