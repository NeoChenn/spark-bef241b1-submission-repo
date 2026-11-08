import math

import numpy as np
import pytest

from sim import shafts

AREA = (
    ("rect", (18.75, -11.0, 53.0), 4000.0, 1.0, (0.0, 0.7, -1.0), (3.0, 1.2)),
    ("rect", (0.0, 0.0, 50.0), 4000.0, 1.0, (0.0, 0.0, -1.0), (1.0, 1.0)),
    ("rect", (0.0, 0.0, 50.0), 4000.0, 1.0, (0.0, 0.0, -1.0), (1.0, 1.0)),
    ("spot", (16.0, 16.0, 60.0), 6000.0, 1.0, (0.25, 0.25, -1.0), 38.0),
    ("spot", (24.0, 24.0, 60.0), 6000.0, 1.0, (-0.25, -0.25, -1.0), 38.0),
)
SEA = (((2.0, -31.0, 17.0), 2000.0, 1.0), ((14.0, 8.0, 17.0), 6000.0, 1.0))


def test_specs_per_tier_reach_floor():
    assert shafts.specs(AREA, SEA, "none") == []
    assert len(shafts.specs(AREA, SEA, "live")) == 2
    off = shafts.specs(AREA, SEA, "offline")
    assert len(off) == len(shafts.PICKS)
    for sp in off:
        end = np.add(sp["apex"], np.multiply(sp["aim"], sp["length"]))
        assert end[2] == pytest.approx(shafts.DECK_Z if sp["apex"][2] > 30 else shafts.SEA_Z)
        assert np.linalg.norm(sp["aim"]) == pytest.approx(1.0)


def test_cone_geometry():
    sp = shafts.specs(AREA, SEA, "offline")[3]
    pts, nrm, s, counts, idx = shafts.cone(sp, segs=16, rings=5)
    assert pts.shape == (80, 3) and nrm.shape == (80, 3) and s.shape == (80,)
    assert counts.sum() == len(idx) and idx.max() < len(pts)
    assert np.allclose(pts[:16], sp["apex"])
    rim = pts[-16:]
    r = np.linalg.norm(rim[:, :2] - np.asarray(sp["apex"][:2]), axis=-1)
    assert np.allclose(r, sp["length"] * math.tan(math.radians(sp["angle"])))
    assert np.allclose(np.linalg.norm(nrm, axis=-1), 1.0)


def test_inside_hides_cone():
    sp = shafts.specs(AREA, SEA, "offline")[3]
    pts, nrm, s, _, _ = shafts.cone(sp)
    inner = np.add(sp["apex"], np.multiply(sp["aim"], sp["length"] * 0.5))
    assert shafts.inside(inner, sp)
    assert not shafts.alpha(pts, nrm, s, inner, sp).any()
    far = (200.0, -200.0, 10.0)
    assert not shafts.inside(far, sp)
    a = shafts.alpha(pts, nrm, s, far, sp, 1.0)
    assert a.max() > 0.2 and a.min() >= 0.0 and a.max() <= 1.0


def test_alpha_fades_at_apex_floor_and_silhouette():
    sp = shafts.specs(AREA, SEA, "offline")[3]
    pts, nrm, s, _, _ = shafts.cone(sp, segs=64, rings=12)
    a = shafts.alpha(pts, nrm, s, (200.0, -31.0, 8.0), sp)
    assert a[s == 0].max() == 0.0 and a[s == 1].max() == 0.0
    mid = (s > 0.2) & (s < 0.5)
    facing = np.abs(nrm[mid] @ np.array([1.0, 0.0, 0.0]))
    assert a[mid][facing > 0.8].mean() > 3 * a[mid][facing < 0.1].mean()


def test_usd_cones_update_and_hide():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
    from pxr import Usd, UsdGeom

    import shafts_fx

    stage = Usd.Stage.CreateInMemory()
    fx = shafts_fx.Shafts(stage, "offline", AREA, SEA)
    assert len(fx.items) == len(shafts.PICKS)
    fx.update((200.0, -200.0, 10.0), 0.5)
    st = np.array(fx.items[3][5].Get())
    assert 0 < st[:, 0].max() <= 1 and st[:, 0].min() >= 0
    fx.update(np.add(fx.items[3][1]["apex"], np.multiply(fx.items[3][1]["aim"], 5.0)), 0.5)
    assert UsdGeom.Imageable(fx.items[3][0]).ComputeVisibility() == "invisible"
    assert not np.array(fx.items[3][5].Get())[:, 0].any()
    fx.show(False)
    assert UsdGeom.Imageable(fx.items[0][0]).ComputeVisibility() == "invisible"
    assert shafts_fx.Shafts(Usd.Stage.CreateInMemory(), "none", AREA, SEA).items == []
