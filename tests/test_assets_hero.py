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
    for asset, *_ in ah.PLACE + ah.WALL + ah.HUNG + ah.REPAIR_BG + ah.DENSE + ah.DENSE_OFFLINE:
        assert asset in manifest
        if asset.startswith("nv_"):
            assert asset in bounds
        else:
            assert list((ASSETS / "third_party" / "models" / asset).glob("*.usdc"))


def test_hero_props_stay_on_the_top_deck():
    for x, y in ah.footprints():
        assert -27.6 < x < 37.5 and -27.6 < y < 27.4


def test_wall_pose_sits_on_the_column_face():
    assert ah.wall_pose(11.75, -10.15, "+y") == (11.75, -10.025, 90.0)
    assert ah.wall_pose(18.75, -1.35, "-x") == (18.625, -1.35, 180.0)


def test_inside_respects_margin():
    assert ah.inside((16.0, -5.0), ah.WORK, margin=0.5)
    assert not ah.inside((15.0, -5.0), ah.WORK, margin=0.5)


def test_repair_background_stays_behind_the_pipe_away_from_the_robot():
    for _, x, y, *_ in ah.REPAIR_BG:
        assert ah.inside((x, y), ah.BEHIND_PIPE)
        assert y < -6.5


def test_unreachable_remote_props_are_skipped_fast(monkeypatch):
    import time

    monkeypatch.setattr(ah, "PROBE", "http://127.0.0.1:9/missing.usd")
    monkeypatch.setenv("RIG_QUALITY", "offline")
    start = time.monotonic()
    assert ah.remote_ok(timeout=1.0) is False
    assert time.monotonic() - start < 2.0
    monkeypatch.setenv("RIG_ASSETS_REMOTE", "0")
    assert ah.remote_ok() is False


def test_live_quality_skips_remote_props_by_default(monkeypatch):
    monkeypatch.setenv("RIG_QUALITY", "live")
    monkeypatch.delenv("RIG_ASSETS_REMOTE", raising=False)
    assert ah.remote_ok() is False


def test_repair_background_clears_arm_reach_and_zoom_camera():
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rig.sim.zoom import ZoomPath

    meta = json.loads((ASSETS.parent / "output" / "rig_merged_meta.json").read_text())
    zoom = ZoomPath(wide_target=tuple(meta["zoom"]["wide_target"]), close_target=tuple(meta["zoom"]["close_target"]))
    damage = np.array(meta["world"]["damage"]["position_m"])
    boxes = [(np.array([x - 0.5, y - 0.5, ah.DECK]), np.array([x + 0.5, y + 0.5, ah.DECK + 3.2]))
             for _, x, y, *_ in ah.REPAIR_BG]

    def dist(p, box):
        return float(np.linalg.norm(np.maximum(0, np.maximum(box[0] - p, p - box[1]))))

    for box in boxes:
        assert dist(damage, box) > 0.6 + 0.5
        for t in np.arange(0, zoom.period, 0.25):
            eye, target, _ = zoom.at(t)
            eye, target = np.array(eye), np.array(target)
            assert min(dist(eye + (target - eye) * s, box) for s in np.linspace(0, 1, 60)) > 0.5


def test_dense_props_keep_the_zoom_line_of_sight_clear():
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import build_assets_hero as b
    from rig.sim.zoom import ZoomPath

    meta = json.loads((ASSETS.parent / "output" / "rig_merged_meta.json").read_text())
    close = np.array(meta["zoom"]["close_target"])
    zoom = ZoomPath(wide_target=tuple(meta["zoom"]["wide_target"]), close_target=tuple(close))
    boxes = []
    for asset, x, y, _, lift in ah.DENSE + ah.DENSE_OFFLINE:
        _, lo, hi = b.source(asset)
        top = ah.DECK + lift + (hi[2] - lo[2]) * ah.DENSE_SCALE.get(asset, 1.0)
        boxes.append((np.array([x - 0.6, y - 0.6, ah.DECK]), np.array([x + 0.6, y + 0.6, top + 0.3])))
    for t in np.arange(0, zoom.period, 0.25):
        eye, target, _ = zoom.at(t)
        eye, target = np.array(eye), np.array(target)
        if np.linalg.norm(target - close) > 3.0:
            continue
        line = eye + (target - eye) * np.linspace(0, 1, 400)[:, None]
        for lo, hi in boxes:
            assert np.linalg.norm(np.maximum(0, np.maximum(lo - line, line - hi)), axis=1).min() > 0.0
