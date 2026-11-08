import json
from pathlib import Path

import numpy as np

from policy.envs.rust_state import rig_meta_scene
from policy.rust_overlay import rig_translation
from rig.sim.workzone import DECK_Z, ZONE_TRANSLATION, merged_meta, to_world, zone_footprint

DECK_META_PATH = Path(__file__).resolve().parent.parent / "sim/output/rig_deck_meta.json"
DECK_META = json.loads(DECK_META_PATH.read_text())
T = (10.0, -5.0, 42.5)


def test_zone_sits_on_top_deck():
    assert ZONE_TRANSLATION[2] == DECK_Z == 42.5


def test_to_world_is_pure_translation():
    assert np.allclose(to_world((1.0, -2.0, 1.1), T), (11.0, -7.0, 43.6))


def test_footprint_is_deck_rectangle_in_world():
    assert zone_footprint(T, half=(8.0, 5.0)) == (2.0, 18.0, -10.0, 0.0)


def test_merged_meta_world_pipe_and_damage():
    w = merged_meta(DECK_META, T)["world"]
    assert w["deck_z_m"] == 42.5
    assert np.isclose(w["main_pipe"]["z_m"], 43.6)
    assert w["main_pipe"]["y_m"] == -5.0
    assert w["main_pipe"]["radius_m"] == 0.15
    assert w["main_pipe"]["axis"] == [1.0, 0.0, 0.0]
    assert w["main_pipe"]["ref"] == [0.0, 1.0, 0.0]
    dx, dy, dz = DECK_META["damage"]["position_m"]
    assert np.allclose(w["damage"]["position_m"], [dx + 10.0, dy - 5.0, dz + 42.5])
    assert w["damage"]["surface_normal"] == DECK_META["damage"]["surface_normal"]
    assert np.allclose(w["segment_origin_m"], [dx + 10.0, -5.0, 43.6])
    assert np.allclose(w["main_pipe"]["x_range_m"], [2.4, 17.6])


def test_merged_meta_keeps_local_deck_schema():
    m = merged_meta(DECK_META, T)
    assert m["main_pipe"] == DECK_META["main_pipe"]
    assert m["damage"] == DECK_META["damage"]
    assert m["zone_to_world"] == {"translation": [10.0, -5.0, 42.5], "yaw": 0.0}


def test_robot_spawn_beside_pipe_on_deck():
    r = merged_meta(DECK_META, T)["world"]["robot_spawn"]
    assert np.isclose(r["position_m"][0], DECK_META["damage"]["position_m"][0] + 10.0)
    assert np.isclose(r["position_m"][1], -4.5)
    assert np.isclose(r["position_m"][2], 42.5 + 0.32)
    assert r["yaw_rad"] == 0.0


def test_backdrop_translation_puts_zone_pipe_on_env_pipe(tmp_path):
    m = merged_meta(DECK_META, T)
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(m))
    scene = rig_meta_scene(str(path), standoff=0.35, base_height=0.32)
    t = rig_translation(scene)
    env_pipe = np.array(scene["pipe_origin"]) - np.array(scene["env_to_world"]["translation"])
    assert np.allclose(np.array(m["world"]["segment_origin_m"]) + t, env_pipe)


def test_deck_meta_backdrop_translation_unchanged():
    scene = rig_meta_scene(str(DECK_META_PATH), standoff=0.35)
    assert np.allclose(rig_translation(scene), -np.array(scene["env_to_world"]["translation"]))
