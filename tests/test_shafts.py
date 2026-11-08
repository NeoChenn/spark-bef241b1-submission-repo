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


def test_card_faces_camera_and_spans_cone():
    sp = shafts.specs(AREA, SEA, "offline")[3]
    eye = (200.0, -31.0, 8.0)
    pts = shafts.card(sp, eye, rings=5, cols=3).reshape(5, 3, 3)
    assert np.allclose(pts[0], sp["apex"])
    half = sp["length"] * math.tan(math.radians(sp["angle"]))
    assert np.linalg.norm(pts[-1, 0] - pts[-1, 1]) == pytest.approx(half)
    across = pts[-1, 2] - pts[-1, 0]
    view = np.subtract(eye, pts[-1, 1])
    assert abs(across @ view) < 1e-6 * np.linalg.norm(view) * np.linalg.norm(across) + 1e-6
    counts, idx, st = shafts.topology(5, 3)
    assert counts.sum() == len(idx) and idx.max() == 14 and st.shape == (15, 2)
    assert st.min() == 0.0 and st.max() == 1.0


def test_inside_cone():
    sp = shafts.specs(AREA, SEA, "offline")[3]
    assert shafts.inside(np.add(sp["apex"], np.multiply(sp["aim"], sp["length"] * 0.5)), sp)
    assert not shafts.inside((200.0, -200.0, 10.0), sp)
    assert not shafts.inside(np.add(sp["apex"], (0.0, 0.0, 3.0)), sp)


def test_usd_beams_follow_camera_and_hide():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
    from pxr import Usd, UsdGeom

    import shafts_fx

    stage = Usd.Stage.CreateInMemory()
    fx = shafts_fx.Shafts(stage, "offline", AREA, SEA)
    assert len(fx.items) == len(shafts.PICKS) and shafts_fx.MDL.exists()
    m, sp, sh = fx.items[3]
    fx.update((200.0, -200.0, 10.0), 0.5)
    assert np.allclose(np.array(m.GetPointsAttr().Get()), shafts.card(sp, (200.0, -200.0, 10.0)), atol=1e-4)
    assert sh.GetInput("time").Get() == 0.5
    assert UsdGeom.Imageable(m).ComputeVisibility() != "invisible"
    fx.update(np.add(sp["apex"], np.multiply(sp["aim"], 5.0)), 0.6)
    assert UsdGeom.Imageable(m).ComputeVisibility() == "invisible"
    fx.show(False)
    fx.update((200.0, -200.0, 10.0), 0.7)
    assert UsdGeom.Imageable(m).ComputeVisibility() == "invisible"
    assert shafts_fx.Shafts(Usd.Stage.CreateInMemory(), "none", AREA, SEA).items == []
