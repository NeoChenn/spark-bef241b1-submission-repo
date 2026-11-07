import numpy as np
import pytest

from sim import dress_plan as dp


def quad(x0, y0, x1, y1, z):
    a, b, c, d = (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)
    return [[a, b, c], [a, c, d]]


def wall(x0, y0, x1, y1, z0, z1):
    a, b, c, d = (x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1)
    return [[a, b, c], [a, c, d]]


def test_mesh_tile_follows_build_full_rig():
    assert dp.mesh_tile("legs_and_lower_deck_deck") == 2.5
    assert dp.mesh_tile("legs_and_lower_deck_concrete") == 6.0
    assert dp.mesh_tile("crane_crane") == 3.0


def test_uv_scale_turns_mesh_tiles_into_texture_metres():
    assert dp.uv_scale(3.0, 1.5) == pytest.approx(2.0)
    assert dp.uv_scale(2.5, 2.5) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "name, material",
    [
        ("top_deck_rail", "rail_yellow"),
        ("crane_crane", "crane_yellow"),
        ("derrick_derrick", "derrick_white"),
        ("pipework_22_pipe", "pipe_orange"),
        ("tank_n73_32_tank", "tank_white"),
        ("vessel_n3_52_vessel", "vessel_blue"),
        ("module_n76_0_module_white", "module_white"),
        ("module_n75_n20_module_blue", "module_blue"),
        ("module_n75_n40_module_grey", "module_grey"),
        ("legs_and_lower_deck_concrete", "concrete_leg"),
        ("legs_and_lower_deck_deck", "deck_plate"),
        ("upper_structure_frame", "steel_grey"),
    ],
)
def test_mesh_material(name, material):
    assert dp.mesh_material(name) == material
    assert material in dp.MATERIALS


def test_lower_deck_top_is_grating_and_undersides_are_weathered():
    tris = np.array(quad(-5, -5, 5, 5, 20.0) + quad(-5, -5, 5, 5, 18.75) + quad(-5, -5, 5, 5, 31.25))
    labels = dp.classify_faces("legs_and_lower_deck_deck", tris)
    assert list(labels) == ["grating"] * 2 + ["underside"] * 2 + ["deck_plate"] * 2


def test_top_deck_plate_edges_and_equipment_in_upper_structure():
    tris = np.array(quad(0, 0, 10, 10, 42.5) + wall(0, 0, 10, 0, 41.25, 42.5) + wall(2, 2, 4, 2, 42.5, 46.0))
    labels = dp.classify_faces("upper_structure_frame", tris)
    assert list(labels) == ["deck_plate"] * 2 + ["deck_edge"] * 2 + ["steel_grey"] * 2


def test_slab_edges_of_lower_frame_are_deck_edges():
    tris = np.array(wall(0, 0, 10, 0, 30.0, 31.25) + wall(0, 0, 0, 1, 20.0, 30.0))
    labels = dp.classify_faces("legs_and_lower_deck_frame", tris)
    assert list(labels) == ["deck_edge"] * 2 + ["steel_weathered"] * 2


def test_big_walls_of_lower_frame_are_cladding():
    tris = np.array(wall(0, 0, 10, 0, 20.0, 30.0))
    assert list(dp.classify_faces("legs_and_lower_deck_frame", tris)) == ["cladding"] * 2


def test_helideck_top_gets_markings():
    tris = np.array(quad(-30, -30, -15, -15, 51.25) + quad(-30, -30, -15, -15, 50.0) + wall(-30, -30, -15, -30, 50.0, 51.25))
    labels = dp.classify_faces("helideck_helideck", tris)
    assert list(labels) == ["helideck"] * 2 + ["underside"] * 2 + ["deck_edge"] * 2


def test_helideck_markings_put_h_white_ring_yellow_and_field_green():
    img = np.asarray(dp.helideck_markings(512)).astype(int)
    px = lambda x, y: img[dp.helideck_pixel(x, y, 512)]
    hx, hy = dp.HELIDECK_CENTRE
    assert px(hx - 2.5, hy).min() > 200
    assert px(hx + dp.TDPM_RADIUS - 0.5, hy)[2] < 80 and px(hx + dp.TDPM_RADIUS - 0.5, hy)[0] > 180
    field = px(hx + 5.5, hy + 5.5)
    assert field[1] > field[0] and field[1] > field[2]


def test_helideck_uv_transform_maps_world_to_image():
    scale, offset = dp.helideck_uv_transform(3.0)
    x0, y0 = dp.HELIDECK_CENTRE[0] - dp.HELIDECK_HALF, dp.HELIDECK_CENTRE[1] - dp.HELIDECK_HALF
    st = np.array([x0, y0]) / 3.0
    assert st * scale + offset == pytest.approx([0.0, 0.0], abs=1e-6)
    st = (np.array([x0, y0]) + 2 * dp.HELIDECK_HALF) / 3.0
    assert st * scale + offset == pytest.approx([1.0, 1.0], abs=1e-6)


def test_streak_mask_is_dense_at_top_and_fades_down():
    mask = np.asarray(dp.streak_mask(128, 256, seed=1), dtype=float)
    top, bottom = mask[:32].mean(), mask[-32:].mean()
    assert top > 2 * bottom
    assert 0 < mask.max() <= 255


def test_aim_matrix_points_local_minus_z_at_target():
    m = np.array(dp.aim_matrix((0, 0, 10), (10, 0, 0)))
    assert m[3, :3] == pytest.approx([0, 0, 10])
    assert -m[2, :3] == pytest.approx(np.array([10, 0, -10]) / np.sqrt(200))
    assert np.allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-9)


def test_aim_matrix_handles_straight_down():
    m = np.array(dp.aim_matrix((1, 2, 30), (1, 2, 20)))
    assert -m[2, :3] == pytest.approx([0, 0, -1])


def test_point_cloud_blocks_boxes_that_touch_geometry():
    tris = np.array(wall(0, 0, 4, 0, 42.5, 45.0))
    cloud = dp.surface_points(tris, spacing=0.1)
    assert dp.box_blocked(cloud, (1, -0.5, 43.0), (2, 0.5, 44.0))
    assert not dp.box_blocked(cloud, (1, 0.5, 43.0), (2, 1.5, 44.0))


def test_lights_have_sane_values_and_unique_names():
    lights = dp.lights()
    names = [l.name for l in lights]
    assert len(names) == len(set(names))
    assert any(l.color[0] > 0.9 and l.color[1] < 0.2 for l in lights)
    for light in lights:
        assert light.kind in ("rect", "sphere", "disk")
        assert light.intensity > 0
        assert all(0 <= c <= 1 for c in light.color)


def test_props_avoid_walkways_and_each_other():
    props = dp.props()
    assert len(props) > 20
    boxes = [p.box() for p in props if p.on_floor]
    for i, a in enumerate(boxes):
        for walk in dp.WALKWAYS:
            assert not dp.boxes_overlap(a, walk), props[i].name
        for b in boxes[i + 1:]:
            assert not dp.boxes_overlap(a, b)


def test_boxes_overlap():
    a = ((0, 0, 0), (1, 1, 1))
    assert dp.boxes_overlap(a, ((0.5, 0.5, 0.5), (2, 2, 2)))
    assert not dp.boxes_overlap(a, ((1.1, 0, 0), (2, 1, 1)))


def test_windows_cover_the_living_quarters_faces():
    windows = dp.windows()
    assert len(windows) > 20
    assert any(w.lit for w in windows) and any(not w.lit for w in windows)


def test_flare_boom_leaves_the_deck_and_puts_flame_past_the_tip():
    members, flame = dp.flare_boom()
    starts = np.array([m[0] for m in members])
    assert starts[:, 2].min() >= dp.DECK_TOPS[2]
    tip = np.array([m[1] for m in members[:3]]).mean(0)
    assert np.linalg.norm(np.array(flame) - tip) < 4.0
    assert flame[2] > dp.DECK_TOPS[2] + 15
    assert all(r > 0 for _, _, r in members)
