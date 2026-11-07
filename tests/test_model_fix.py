import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim" / "tools"))

from model_fix import (
    converter_for,
    deck_level,
    guess_scale,
    guess_up_axis,
    is_walkable,
    preview_cameras,
    unitless,
    TextureMap,
)


@pytest.mark.parametrize(
    "extent, mpu, expected",
    [
        (60.0, 1.0, 1.0),
        (6000.0, 0.01, 0.01),
        (6000.0, 1.0, 0.01),
        (60000.0, 1.0, 0.001),
        (0.6, 1.0, 100.0),
    ],
)
def test_guess_scale_brings_platform_into_plausible_size(extent, mpu, expected):
    assert guess_scale(extent, mpu) == pytest.approx(expected)


def test_guess_scale_trusts_metadata_when_plausible():
    assert guess_scale(4000.0, 0.01) == pytest.approx(0.01)
    assert guess_scale(150.0, 1.0) == pytest.approx(1.0)


def test_guess_scale_ignores_bogus_cm_metadata_on_metre_model():
    assert guess_scale(80.0, 0.01) == pytest.approx(1.0)


def test_guess_scale_trusts_small_but_sane_metadata():
    assert guess_scale(2.0, 1.0) == pytest.approx(1.0)
    assert guess_scale(200.0, 0.01) == pytest.approx(0.01)


def test_guess_scale_override():
    assert guess_scale(80.0, 1.0, units="cm") == pytest.approx(0.01)
    assert guess_scale(80.0, 1.0, units="in") == pytest.approx(0.0254)


def test_guess_up_axis_follows_metadata_when_geometry_agrees():
    assert guess_up_axis("Y", {"X": 10.0, "Y": 100.0, "Z": 12.0}) == "Y"
    assert guess_up_axis("Z", {"X": 10.0, "Y": 12.0, "Z": 100.0}) == "Z"


def test_guess_up_axis_overrides_wrong_metadata():
    assert guess_up_axis("Y", {"X": 10.0, "Y": 12.0, "Z": 100.0}) == "Z"


def test_guess_up_axis_keeps_metadata_when_ambiguous():
    assert guess_up_axis("Y", {"X": 50.0, "Y": 60.0, "Z": 70.0}) == "Y"


def test_guess_up_axis_override():
    assert guess_up_axis("Y", {"X": 0.0, "Y": 100.0, "Z": 0.0}, forced="Z") == "Z"


def test_deck_level_picks_largest_upward_area_band():
    samples = [(0.0, 5.0), (12.0, 400.0), (12.1, 300.0), (20.0, 300.0), (35.0, 50.0)]
    assert deck_level(samples) == pytest.approx(12.0, abs=0.3)


def test_deck_level_ignores_sea_plane_at_bottom():
    samples = [(-25.0, 100000.0), (0.0, 50.0), (10.0, 900.0), (18.0, 600.0), (40.0, 80.0)]
    assert deck_level(samples) == pytest.approx(10.0, abs=0.3)


def test_deck_level_empty():
    assert deck_level([]) is None


@pytest.mark.parametrize(
    "name", ["Main_Deck", "helideck_01", "Stairs.003", "grating_plate", "catwalk", "Floor", "walkway"]
)
def test_walkable_names(name):
    assert is_walkable(name, up_fraction=0.0)


@pytest.mark.parametrize("name", ["Derrick", "pipe_12", "Crane_boom", "railing"])
def test_non_walkable_names(name):
    assert not is_walkable(name, up_fraction=0.1)


def test_walkable_by_geometry():
    assert is_walkable("Mesh_123", up_fraction=0.6, min_up_area=20.0, up_area=50.0)
    assert not is_walkable("Mesh_123", up_fraction=0.6, min_up_area=20.0, up_area=1.0)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("rig.glb", "assimp"),
        ("rig.GLTF", "assimp"),
        ("rig.fbx", "assimp"),
        ("rig.obj", "assimp"),
        ("rig.stl", "assimp"),
        ("rig.usdz", "usdz"),
        ("rig.usd", "usd"),
        ("rig.usda", "usd"),
        ("rig.step", "hoops"),
        ("rig.STP", "hoops"),
        ("rig.iges", "hoops"),
        ("rig.blend", "blender"),
        ("rig.dae", "blender"),
        ("rig.3ds", "blender"),
        ("rig.x3d", "blender"),
    ],
)
def test_converter_for(path, expected):
    assert converter_for(path) == expected


def test_converter_for_unknown():
    with pytest.raises(ValueError):
        converter_for("rig.xyz")


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(path.name.encode())
    return path


def test_texture_map_resolves_relative_to_layer(tmp_path):
    touch(tmp_path / "raw" / "textures" / "steel.png")
    tex = TextureMap(out_dir=tmp_path / "out", search_dirs=[tmp_path / "src"])
    assert tex.map("./textures/steel.png", tmp_path / "raw") == "./textures/steel.png"
    assert (tmp_path / "out" / "textures" / "steel.png").read_bytes() == b"steel.png"
    assert tex.missing == []


def test_texture_map_finds_windows_absolute_path_by_basename(tmp_path):
    touch(tmp_path / "src" / "maps" / "Rust_D.jpg")
    tex = TextureMap(out_dir=tmp_path / "out", search_dirs=[tmp_path / "src"])
    assert tex.map("C:\\Users\\bob\\maps\\Rust_D.jpg", tmp_path / "raw") == "./textures/Rust_D.jpg"


def test_texture_map_basename_lookup_is_case_insensitive(tmp_path):
    touch(tmp_path / "src" / "rust_d.JPG")
    tex = TextureMap(out_dir=tmp_path / "out", search_dirs=[tmp_path / "src"])
    assert tex.map("textures/Rust_D.jpg", tmp_path / "raw") == "./textures/rust_d.JPG"


def test_texture_map_dedupes_same_source(tmp_path):
    touch(tmp_path / "raw" / "a.png")
    tex = TextureMap(out_dir=tmp_path / "out", search_dirs=[])
    assert tex.map("a.png", tmp_path / "raw") == tex.map("./a.png", tmp_path / "raw")
    assert len(list((tmp_path / "out" / "textures").iterdir())) == 1


def test_texture_map_renames_on_basename_clash(tmp_path):
    touch(tmp_path / "raw" / "one" / "base.png")
    (tmp_path / "raw" / "two").mkdir(parents=True)
    (tmp_path / "raw" / "two" / "base.png").write_bytes(b"other")
    tex = TextureMap(out_dir=tmp_path / "out", search_dirs=[])
    first = tex.map("one/base.png", tmp_path / "raw")
    second = tex.map("two/base.png", tmp_path / "raw")
    assert first == "./textures/base.png"
    assert second == "./textures/base_1.png"
    assert (tmp_path / "out" / "textures" / "base_1.png").read_bytes() == b"other"


def test_texture_map_records_missing(tmp_path):
    tex = TextureMap(out_dir=tmp_path / "out", search_dirs=[tmp_path])
    assert tex.map("gone.png", tmp_path / "raw") is None
    assert tex.missing == ["gone.png"]


def test_texture_map_leaves_builtin_mdl_alone(tmp_path):
    tex = TextureMap(out_dir=tmp_path / "out", search_dirs=[])
    assert tex.map("OmniPBR.mdl", tmp_path / "raw") == "OmniPBR.mdl"
    assert tex.missing == []


def test_preview_cameras_wide_shot_sees_whole_model():
    cams = preview_cameras([-30.0, -20.0, -40.0], [30.0, 20.0, 50.0])
    eye, target = cams["wide"]["eye"], cams["wide"]["target"]
    assert target == pytest.approx([0.0, 0.0, 5.0])
    distance = sum((e - t) ** 2 for e, t in zip(eye, target)) ** 0.5
    assert distance > 100.0
    assert eye[2] > target[2]


def test_preview_cameras_deck_shot_stands_on_deck():
    cams = preview_cameras([-30.0, -20.0, -40.0], [30.0, 20.0, 50.0])
    eye, target = cams["deck"]["eye"], cams["deck"]["target"]
    assert eye[2] == pytest.approx(1.7)
    assert abs(eye[0]) < 60.0 and abs(eye[1]) < 40.0
    assert target[2] > eye[2]


def test_guess_scale_ignores_metadata_for_unitless_formats():
    assert guess_scale(550.0, 0.01, trust_metadata=False) == pytest.approx(0.1)
    assert guess_scale(60.0, 0.01, trust_metadata=False) == pytest.approx(1.0)


def test_unitless_formats():
    assert unitless("rig.stl") and unitless("rig.OBJ") and unitless("rig.ply")
    assert not unitless("rig.glb") and not unitless("rig.fbx")
