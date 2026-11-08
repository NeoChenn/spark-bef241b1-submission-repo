import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import flare_fx as fx


def test_flicker_stays_in_band():
    vals = [fx.flicker(t * 0.01, 1.0) for t in range(3000)]
    assert 0.8 < min(vals) < max(vals) < 1.05
    assert np.std(vals) > 0.02


def test_flame_frames_are_masked_at_tile_borders_and_hot_at_base():
    fr = fx.flame_frames(n=4, w=64, h=128)
    assert fr.shape == (4, 128, 64, 4)
    a = fr[..., 3]
    assert a[:, :, 0].max() == 0 and a[:, :, -1].max() == 0 and a[:, 0, :].max() == 0
    lum = fr[..., :3].sum(-1)
    assert lum[:, 100:, 24:40].mean() > 3 * lum[:, 10:40].mean()
    assert not np.allclose(fr[0], fr[1])


def test_atlas_packs_frames_in_rows():
    fr = np.zeros((6, 4, 2, 4))
    fr[5, ..., 3] = 1.0
    img = fx.atlas(fr, cols=4)
    assert img.shape == (8, 8, 4) and img.dtype == np.uint8
    assert img[4:8, 2:4, 3].min() == 255 and img[:4, :, 3].max() == 0


def test_centerline_starts_at_tip_and_bends_downwind():
    line = fx.centerline(np.array([0.0, 1.0]), 0.0, wind=(5.0, 0.0))
    assert line[0] == pytest.approx(fx.TIP)
    assert line[1][2] - fx.TIP[2] > 0.6 * fx.HEIGHT
    calm = fx.centerline(np.array([1.0]), 0.0, wind=(0.0, 0.0))[0]
    assert line[1][0] > calm[0] + 3.0


def test_facing_right_is_horizontal_and_perpendicular_to_view():
    r = fx.facing_right((0, 0, 0), (0, -10, 5))
    assert r[2] == 0 and np.linalg.norm(r) == pytest.approx(1.0)
    assert abs(r @ np.array([0, -1.0, 0])) < 1e-9


def test_smoke_is_deterministic_and_drifts_downwind():
    a = fx.smoke_state(3.0, wind=(4.0, 0.0))
    b = fx.smoke_state(3.0, wind=(4.0, 0.0))
    for x, y in zip(a, b):
        assert np.allclose(x, y)
    pos, scale, bucket, roll = a
    assert pos.shape == (fx.SMOKE_N, 3) and bucket.min() >= 0 and bucket.max() < fx.SMOKE_LEVELS
    old = bucket >= fx.SMOKE_LEVELS // 2
    assert pos[old, 0].mean() > pos[~old, 0].mean() + 5.0
    assert scale[old].mean() > scale[~old].mean()


def test_smoke_starts_inside_flame_top():
    pos, _, bucket, _ = fx.smoke_state(3.0)
    top = fx.centerline(np.array([1.0]), 3.0)[0]
    assert np.linalg.norm(pos[bucket == 0] - top, axis=1).min() < 6.0


def test_smoke_levels_thin_out_and_glow_young():
    levels = [fx.smoke_level(k) for k in range(fx.SMOKE_LEVELS)]
    alphas, glows = zip(*levels)
    assert alphas[1] > alphas[-1] and glows[0] > 5 * glows[-1]
    assert max(alphas) <= fx.SMOKE_ALPHA


def test_billboard_quats_face_eye():
    q = fx.billboard_quats(np.zeros((1, 3)), (0, -10, 0))
    assert np.linalg.norm(q[0]) == pytest.approx(1.0)
    assert q[0] == pytest.approx([1, 0, 0, 0])


def test_billboard_roll_keeps_unit_quats():
    q = fx.billboard_quats(np.zeros((3, 3)), (0, -10, 0), np.array([0.0, 1.0, 2.5]))
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
    assert q[0] == pytest.approx([1, 0, 0, 0])


def test_parts_by_quality(monkeypatch):
    monkeypatch.delenv("RIG_FLARE_PARTS", raising=False)
    monkeypatch.setenv("RIG_QUALITY", "offline")
    assert {"side", "halo", "smoke"} <= fx.parts()
    monkeypatch.setenv("RIG_QUALITY", "live")
    assert fx.parts() == {"flame", "anim"}
    monkeypatch.setenv("RIG_FLARE_PARTS", "flame")
    assert fx.parts() == {"flame"}
