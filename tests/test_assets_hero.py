import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import assets_hero as ah

ASSETS = Path(__file__).resolve().parents[1] / "sim" / "assets"


def test_layout_keeps_work_zone_route_and_eyes_clear():
    assert ah.layout_issues() == []


def test_every_asset_resolves_locally_or_has_remote_bounds():
    bounds = {**json.loads((ASSETS / "nv_bounds.json").read_text()), **json.loads((ASSETS / "hero_bounds.json").read_text())}
    manifest = {a["id"] for a in json.loads((ASSETS / "manifest.json").read_text())["assets"]}
    for asset, *_ in ah.PLACE + ah.WALL:
        assert asset in manifest
        if asset.startswith("nv_"):
            assert asset in bounds
        else:
            assert list((ASSETS / "third_party" / "models" / asset).glob("*.usdc"))


def test_hero_props_stay_on_the_top_deck():
    for x, y in ah.footprints():
        assert -27.0 < x < 37.0 and -27.0 < y < 27.0


def test_wall_pose_sits_on_the_column_face():
    assert ah.wall_pose(11.75, -10.15, "+y") == (11.75, -10.025, 90.0)
    assert ah.wall_pose(18.75, -1.35, "-x") == (18.625, -1.35, 180.0)


def test_inside_respects_margin():
    assert ah.inside((16.0, -5.0), ah.WORK, margin=0.5)
    assert not ah.inside((15.0, -5.0), ah.WORK, margin=0.5)
