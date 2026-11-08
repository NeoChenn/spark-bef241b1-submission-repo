import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import photoreal as pr

SCENE_MATERIALS = (
    "deck", "concrete", "frame", "rail", "crane", "derrick", "helideck", "pipe", "tank", "vessel",
    "module_white", "module_blue", "module_grey", "support_steel", "line_water", "line_gas", "line_process",
    "splash_zone", "walkway_line", "topside_steel", "topside_steel_white", "topside_steel_rust", "topside_yellow",
    "topside_galv", "topside_insulated", "topside_pipe_grey", "topside_pipe_red", "topside_pipe_yellow",
    "topside_pipe_green", "topside_vessel_white", "topside_vessel_grey", "topside_machine_blue",
    "topside_machine_green", "topside_valve_red", "topside_black", "topside_lifeboat", "topside_grating",
    "worn_corrugated_iron", "metal_plate", "rusty_metal_02", "rusty_painted_metal", "blue_metal_plate",
    "metal_grate_rusty", "pipe_grey", "pipe_orange", "safety_yellow", "flare_steel",
)


def test_every_scene_material_has_a_recipe():
    missing = [name for name in SCENE_MATERIALS if name not in pr.RECIPES]
    assert missing == []


def test_perception_targets_are_never_overridden():
    for name in ("green_metal_rust", "rust_coarse_01", "window_glass", "cinematic_sea", "LeakCrack", "TorqueStripe"):
        assert name not in pr.RECIPES


def test_recipe_kinds_are_known():
    assert {r["kind"] for r in pr.RECIPES.values()} <= set(pr.KINDS)


@pytest.mark.parametrize("rain", [-1.0, 0.0, 0.3, 0.55, 1.0, 2.0])
def test_weather_is_bounded(rain):
    w = pr.weather(rain)
    assert 0.0 <= w["wetness"] <= 1.0 and 0.0 <= w["puddles"] <= 1.0


def test_dry_weather_has_no_water():
    assert pr.weather(0.0) == {"wetness": 0.0, "puddles": 0.0}


def test_weather_increases_with_rain():
    levels = [pr.weather(r) for r in (0.1, 0.4, 0.7, 1.0)]
    for a, b in zip(levels, levels[1:]):
        assert a["wetness"] < b["wetness"] and a["puddles"] <= b["puddles"]
    assert levels[-1] == {"wetness": 1.0, "puddles": 1.0}


def test_presets_map_to_rain():
    assert pr.rain_for("storm") == 1.0
    assert pr.rain_for("golden") == 0.0
    assert 0.0 < pr.rain_for("night") < 1.0
    assert pr.rain_for("unknown") == 0.0


def test_quality_switch():
    assert pr.quality_settings("live")["triplanar"] is False and pr.quality_settings("live")["lite"] is True
    assert pr.quality_settings("offline")["lite"] is False
    assert pr.quality_settings("live")["wet_coat"] is False and pr.quality_settings("offline")["wet_coat"] is True
    assert pr.quality_settings("offline")["triplanar"] is True
    assert pr.quality_settings("offline")["offline_props"] is True
    assert pr.quality_settings("anything")["offline_props"] is False


def test_material_roots_cover_zone():
    paths = pr.material_paths("deck")
    assert "/World/Looks/deck" in paths and "/World/RepairZone/Looks/deck" in paths


def test_paint_recipe_tints_shared_paint_and_rust_sets():
    sub, values = pr.shader_inputs(pr.RECIPES["rail"])
    assert sub == "rig_weathered"
    assert values["tint_paint"] is True and values["paint_color"] == pr.YELLOW
    assert values["paint_diff"].parent.name == pr.PAINT and values["rust_diff"].parent.name == pr.RUST
    assert values["masks"].name == "rig_masks.png" and values["streaks"].name == "rig_streaks.png"
    assert values["use_opacity"] is False


def test_textured_recipes_use_their_own_set_untinted():
    sub, values = pr.shader_inputs(pr.RECIPES["deck"])
    assert values["tint_paint"] is False and values["paint_diff"].parent.name == "DiamondPlate008C_4k"
    assert values["paint_scale"] == 1.6 and values["metallic"] == 0.4


def test_grating_cuts_out():
    _, values = pr.shader_inputs(pr.RECIPES["topside_grating"])
    assert values["use_opacity"] is True and values["opacity_tex"].name.startswith("opacity")


def test_brushed_uses_anisotropic_shader():
    sub, values = pr.shader_inputs(pr.RECIPES["topside_insulated"])
    assert sub == "rig_brushed" and values["brush_axis"] == (0.0, 0.0, 1.0)


def test_all_recipe_textures_exist():
    for recipe in pr.RECIPES.values():
        _, values = pr.shader_inputs(recipe)
        for value in values.values():
            if isinstance(value, Path):
                assert value.exists(), value


def test_floor_faces_pick_up_facing_triangles_at_deck_height():
    import numpy as np

    points = np.array([(0, 0, 42.5), (1, 0, 42.5), (0, 1, 42.5), (0, 0, 40.0), (1, 0, 41.0), (0, 1, 40.0),
                       (0, 0, 42.5), (0, 0, 43.5), (1, 0, 42.5)], float)
    tris = np.array([(0, 1, 2), (2, 1, 0), (3, 4, 5), (6, 8, 7)])
    assert pr.floor_faces(points, tris, (20.0, 42.5)).tolist() == [0]
