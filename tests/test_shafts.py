import math

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

from sim import shafts

AREA = (
    ("rect", (18.75, -11.0, 53.0), 4000.0, 1.0, (0.0, 0.7, -1.0), (3.0, 1.2)),
    ("rect", (0.0, 0.0, 50.0), 4000.0, 1.0, (0.0, 0.0, -1.0), (1.0, 1.0)),
    ("rect", (0.0, 0.0, 50.0), 4000.0, 1.0, (0.0, 0.0, -1.0), (1.0, 1.0)),
    ("spot", (16.0, 16.0, 60.0), 6000.0, 1.0, (0.25, 0.25, -1.0), 38.0),
    ("spot", (24.0, 24.0, 60.0), 6000.0, 1.0, (-0.25, -0.25, -1.0), 38.0),
)
SEA = (((2.0, -31.0, 17.0), 2000.0, 1.0), ((14.0, 8.0, 17.0), 6000.0, 1.0))
DECK = tuple(((12.0 + k, -2.0, 46.8), (16.0, -4.0, 42.5), 2000.0, 1.0, 40.0) for k in range(4))


def test_specs_per_tier_reach_floor():
    assert shafts.specs(AREA, SEA, DECK, "none") == []
    assert shafts.specs(AREA, SEA, DECK, "live") == []
    off = shafts.specs(AREA, SEA, DECK, "offline")
    assert len(off) == len(shafts.PICKS)
    for sp in off:
        end = np.add(sp["apex"], np.multiply(sp["aim"], sp["length"]))
        assert end[2] == pytest.approx(shafts.DECK_Z if sp["apex"][2] > 30 else shafts.SEA_Z)
        assert np.linalg.norm(sp["aim"]) == pytest.approx(1.0)


def test_inside_cone():
    sp = shafts.specs(AREA, SEA, DECK, "offline")[6]
    assert shafts.inside(np.add(sp["apex"], np.multiply(sp["aim"], sp["length"] * 0.5)), sp)
    assert not shafts.inside((200.0, -200.0, 10.0), sp)
    assert not shafts.inside(np.add(sp["apex"], (0.0, 0.0, 3.0)), sp)


def _cam(eye, target):
    from rig.camera import look_at

    return look_at(eye, target, 24.0, 20.955, 320, 180)


def test_layer_draws_beam_where_it_projects():
    sp = shafts.specs(AREA, SEA, DECK, "offline")[6]
    mid = np.add(sp["apex"], np.multiply(sp["aim"], sp["length"] * 0.4))
    cam = _cam((120.0, -31.0, 8.0), mid)
    beam = shafts.layer(cam, [sp], 0.0, scale=2)
    assert beam.shape == (90, 160, 3)
    assert beam[40:50, 75:85].sum() > 10 * beam[:, :20].sum() + 1
    assert beam[..., 0].sum() > beam[..., 2].sum()
    away = _cam((120.0, -31.0, 8.0), (300.0, -31.0, 8.0))
    assert not shafts.layer(away, [sp]).any()
    inner = _cam(np.add(sp["apex"], np.multiply(sp["aim"], sp["length"] * 0.5)), (0.0, 0.0, 0.0))
    assert not shafts.layer(inner, [sp]).any()


def test_add_upsamples_and_brightens():
    img = np.full((180, 320, 3), 20, np.uint8)
    beam = np.zeros((45, 80, 3), np.float32)
    beam[20:25, 38:42] = 100.0
    out = shafts.add(img, beam)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out[90, 160].mean() > 60 and out[0, 0].mean() == 20
    assert shafts.add(img, np.zeros_like(beam)) is img


def test_depth_occludes_beam_behind_geometry():
    sp = shafts.specs(AREA, SEA, DECK, "offline")[6]
    mid = np.add(sp["apex"], np.multiply(sp["aim"], sp["length"] * 0.4))
    cam = _cam((120.0, -31.0, 8.0), mid)
    free = shafts.layer(cam, [sp], 0.0, scale=2, depth=np.full((180, 320), np.inf, np.float32))
    assert np.allclose(free, shafts.layer(cam, [sp], 0.0, scale=2))
    wall = np.full((180, 320, 1), 50.0, np.float32)
    assert not shafts.layer(cam, [sp], 0.0, scale=2, depth=wall).any()
    assert shafts.layer(cam, [sp], 0.0, scale=2, depth=wall * 10).sum() == pytest.approx(free.sum())
