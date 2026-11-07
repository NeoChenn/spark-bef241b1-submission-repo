import pytest

from asset_manifest import (
    Asset,
    ambientcg_map_name,
    ambientcg_zip_url,
    licences_markdown,
    load_assets,
    polyhaven_hdri_file,
    polyhaven_model_files,
    polyhaven_texture_files,
    safe_target,
    select,
)

LEGACY = {
    "resolution": "1k",
    "hdri_resolution": "2k",
    "models": ["barrel_03"],
    "textures": ["metal_plate"],
    "hdris": ["gray_pier"],
}

V2 = {
    "version": 2,
    "assets": [
        {
            "id": "MetalWalkway004",
            "kind": "texture",
            "source": "ambientcg",
            "url": "https://ambientcg.com/a/MetalWalkway004",
            "licence": "CC0-1.0",
            "resolution": "2k",
            "path": "textures/MetalWalkway004",
            "category": "grating",
        },
        {
            "id": "iso7010_e007",
            "kind": "decal",
            "source": "wikimedia",
            "url": "https://commons.wikimedia.org/wiki/File:ISO_7010_E007.svg",
            "download": "https://upload.wikimedia.org/x/1024px-ISO_7010_E007.svg.png",
            "licence": "Public domain",
            "resolution": "1024px",
            "path": "decals/iso7010_e007/iso7010_e007.png",
        },
        {
            "id": "nv_pallet",
            "kind": "model",
            "source": "nvidia",
            "url": "https://example.com/pallet.usd",
            "licence": "NVIDIA Omniverse",
            "resolution": "n/a",
            "path": "https://example.com/pallet.usd",
            "remote": True,
        },
    ],
}


def test_legacy_manifest_maps_to_polyhaven_assets_at_old_paths():
    assets = {a.id: a for a in load_assets(LEGACY)}
    assert assets["barrel_03"] == Asset(
        id="barrel_03",
        kind="model",
        source="polyhaven",
        url="https://polyhaven.com/a/barrel_03",
        licence="CC0-1.0",
        resolution="1k",
        path="models/barrel_03",
    )
    assert assets["metal_plate"].path == "textures/metal_plate"
    assert assets["gray_pier"].kind == "hdri"
    assert assets["gray_pier"].resolution == "2k"
    assert assets["gray_pier"].path == "hdris/gray_pier.hdr"


def test_v2_manifest_keeps_extra_fields():
    assets = {a.id: a for a in load_assets(V2)}
    assert assets["MetalWalkway004"].category == "grating"
    assert assets["iso7010_e007"].download.endswith(".png")
    assert assets["nv_pallet"].remote is True
    assert assets["MetalWalkway004"].remote is False


def test_source_id_defaults_to_id():
    item = dict(V2["assets"][0], id="sky_8k", source_id="sky")
    assets = load_assets({"version": 2, "assets": [item, V2["assets"][1]]})
    assert assets[0].upstream_id == "sky"
    assert assets[1].upstream_id == "iso7010_e007"


def test_v2_manifest_rejects_duplicate_ids():
    data = {"version": 2, "assets": V2["assets"] + V2["assets"][:1]}
    with pytest.raises(ValueError, match="duplicate"):
        load_assets(data)


def test_v2_manifest_rejects_unknown_kind():
    bad = dict(V2["assets"][0], kind="sound")
    with pytest.raises(ValueError, match="kind"):
        load_assets({"version": 2, "assets": [bad]})


def test_v2_manifest_requires_fields():
    bad = {k: v for k, v in V2["assets"][0].items() if k != "licence"}
    with pytest.raises(ValueError, match="licence"):
        load_assets({"version": 2, "assets": [bad]})


def test_select_filters_by_kind_and_id_and_skips_remote():
    assets = load_assets(V2)
    assert [a.id for a in select(assets)] == ["MetalWalkway004", "iso7010_e007"]
    assert [a.id for a in select(assets, kinds={"decal"})] == ["iso7010_e007"]
    assert [a.id for a in select(assets, ids={"MetalWalkway004"})] == ["MetalWalkway004"]


def test_ambientcg_zip_url():
    assert ambientcg_zip_url("Rust007", "2k") == "https://ambientcg.com/get?file=Rust007_2K-JPG.zip"


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        ("Rust007_2K-JPG_Color.jpg", "diff.jpg"),
        ("Rust007_2K-JPG_NormalGL.jpg", "nor_gl.jpg"),
        ("Rust007_2K-JPG_NormalDX.jpg", None),
        ("Rust007_2K-JPG_Roughness.jpg", "rough.jpg"),
        ("Rust007_2K-JPG_Metalness.jpg", "metal.jpg"),
        ("Rust007_2K-JPG_AmbientOcclusion.jpg", "ao.jpg"),
        ("Rust007_2K-JPG_Displacement.jpg", "disp.jpg"),
        ("Rust007_2K-JPG_Opacity.jpg", "opacity.jpg"),
        ("Rust007_2K-JPG.usdc", None),
        ("Rust007.png", None),
        ("../evil_Color.jpg", None),
    ],
)
def test_ambientcg_map_name(member, expected):
    assert ambientcg_map_name(member) == expected


PH_TEXTURE = {
    "Diffuse": {"2k": {"jpg": {"url": "https://dl/x_diff_2k.jpg", "md5": "a"}}},
    "nor_gl": {"2k": {"jpg": {"url": "https://dl/x_nor_gl_2k.jpg", "md5": "b"}}},
    "Rough": {"2k": {"jpg": {"url": "https://dl/x_rough_2k.jpg", "md5": "c"}}},
    "Displacement": {"2k": {"jpg": {"url": "https://dl/x_disp_2k.jpg", "md5": "d"}}},
    "nor_dx": {"2k": {"jpg": {"url": "https://dl/x_nor_dx_2k.jpg", "md5": "e"}}},
}


def test_polyhaven_texture_files_picks_gl_normals_and_standard_names():
    files = polyhaven_texture_files(PH_TEXTURE, "2k")
    assert files == [
        ("https://dl/x_diff_2k.jpg", "a", "diff.jpg"),
        ("https://dl/x_nor_gl_2k.jpg", "b", "nor_gl.jpg"),
        ("https://dl/x_rough_2k.jpg", "c", "rough.jpg"),
        ("https://dl/x_disp_2k.jpg", "d", "disp.jpg"),
    ]


def test_polyhaven_hdri_file_flat_and_folder_paths():
    files = {"hdri": {"4k": {"hdr": {"url": "https://dl/sky_4k.hdr", "md5": "m"}}}}
    assert polyhaven_hdri_file(files, "4k", "hdris/sky.hdr") == ("https://dl/sky_4k.hdr", "m", "sky.hdr")
    assert polyhaven_hdri_file(files, "4k", "hdris/sky") == ("https://dl/sky_4k.hdr", "m", "sky_4k.hdr")


def test_polyhaven_model_files_includes_textures_and_missing_rough():
    files = {
        "usd": {
            "1k": {
                "usd": {
                    "url": "https://dl/b_1k.usdc",
                    "md5": "u",
                    "include": {"textures/b_diff_1k.jpg": {"url": "https://dl/b_diff_1k.jpg", "md5": "t"}},
                }
            }
        },
        "Rough": {"1k": {"exr": {"url": "https://dl/b_rough_1k.exr", "md5": "r"}}},
    }
    assert polyhaven_model_files(files, "1k") == [
        ("https://dl/b_1k.usdc", "u", "b_1k.usdc"),
        ("https://dl/b_diff_1k.jpg", "t", "textures/b_diff_1k.jpg"),
        ("https://dl/b_rough_1k.exr", "r", "textures/b_rough_1k.exr"),
    ]


def test_safe_target_refuses_escape(tmp_path):
    assert safe_target(tmp_path, "textures/a.jpg") == tmp_path / "textures" / "a.jpg"
    for bad in ("../a", "/etc/passwd", "", "a/../../b"):
        with pytest.raises(ValueError):
            safe_target(tmp_path, bad)


def test_licences_markdown_lists_one_line_per_asset_sorted_by_kind():
    text = licences_markdown(load_assets(V2))
    lines = [line for line in text.splitlines() if line.startswith("| ") and "---" not in line]
    assert lines[0].startswith("| id ")
    assert "| iso7010_e007 | decal | wikimedia | Public domain | https://commons.wikimedia.org/wiki/File:ISO_7010_E007.svg |" in lines
    assert len(lines) == 4
