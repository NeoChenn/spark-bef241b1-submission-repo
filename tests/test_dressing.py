import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import dressing

ROOT = Path(__file__).resolve().parents[1] / "sim" / "assets"


def test_layout_keeps_walkways_zone_and_camera_paths_clear():
    assert dressing.layout_issues() == []


def test_props_reference_known_assets():
    manifest = {a["id"] for a in json.loads((ROOT / "manifest.json").read_text())["assets"]}
    bounds = json.loads((ROOT / "nv_bounds.json").read_text())
    for asset, *_ in dressing.PROPS + dressing.OFFLINE_PROPS:
        assert asset in manifest
        if asset.startswith("nv_"):
            assert asset in bounds


def test_dressing_stays_on_the_top_deck():
    for x, y in dressing.footprints():
        assert -27.0 < x < 37.0 and -27.0 < y < 27.0
