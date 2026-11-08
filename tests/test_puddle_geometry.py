import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import puddle_geometry as pg


def test_outline_is_irregular_but_bounded():
    o = pg.outline(1.0, seed=3)
    r = np.hypot(o[:, 0], o[:, 1])
    assert len(o) == 48 and r.min() >= 0.55 - 1e-9 and r.max() <= 1.4 + 1e-9
    assert r.std() > 0.03
    assert not np.allclose(pg.outline(1.0, seed=3), pg.outline(1.0, seed=4))


def test_outline_is_star_shaped_and_ordered():
    o = pg.outline(0.8, seed=1)
    ang = np.unwrap(np.arctan2(o[:, 1], o[:, 0]))
    assert np.all(np.diff(ang) > 0)


def test_triangulation_covers_every_vertex_with_valid_indices():
    m = pg.mesh((1.0, 2.0, 3.0), 0.6, seed=2, n=24)
    n_pts = len(m["points"])
    assert n_pts == 1 + 24 * (len(pg.RINGS) - 1)
    assert len(m["indices"]) == 3 * len(m["counts"]) == 3 * (24 + 2 * 24 * (len(pg.RINGS) - 2))
    assert m["indices"].min() == 0 and m["indices"].max() == n_pts - 1
    assert set(range(n_pts)) == set(m["indices"].tolist())


def test_triangles_face_up():
    m = pg.mesh((0.0, 0.0, 0.0), 1.0, seed=5)
    p = m["points"][m["indices"].reshape(-1, 3)]
    normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    assert np.all(normals[:, 2] > 0)


def test_mesh_sits_just_above_the_deck():
    m = pg.mesh((0.0, 0.0, 42.5), 0.5, lift=0.004)
    assert np.allclose(m["points"][:, 2], 42.504)


def test_rim_is_zero_inside_and_one_at_the_edge():
    assert pg.rim_weight(0.0) == 0.0 and pg.rim_weight(0.5) == 0.0 and pg.rim_weight(1.0) == 1.0
    f = np.linspace(0, 1, 50)
    assert np.all(np.diff(pg.rim_weight(f)) >= 0)


def test_uv_radius_encodes_ring_fraction():
    m = pg.mesh((0.0, 0.0, 0.0), 2.0, seed=1)
    uv = m["uvs"] * 2 - 1
    assert np.allclose(np.hypot(uv[:, 0], uv[:, 1]), m["f"])
    assert math.isclose(m["rim"][-1], 1.0)


def test_make_puddle_and_set_rain_author_usd():
    from pxr import Usd, UsdShade

    import photoreal

    stage = Usd.Stage.CreateInMemory()
    prim = photoreal.make_puddle(stage, "/World/P0", (1.0, 2.0, 42.5), 0.6, seed=4)
    photoreal.make_puddle(stage, "/World/P1", (3.0, 2.0, 42.5), 0.4)
    assert prim.GetAttribute(photoreal.PUDDLE_ATTR).Get()
    assert photoreal.set_rain(stage, 1.7) == 2
    shader = UsdShade.Shader.Get(stage, "/World/P0/look/shader")
    assert shader.GetInput("rain").Get() == 1.0
    assert shader.GetSourceAssetSubIdentifier("mdl") == "rig_puddle"


def test_puddles_can_share_one_material():
    from pxr import Usd, UsdShade

    import photoreal

    stage = Usd.Stage.CreateInMemory()
    look = photoreal.puddle_material(stage, "/World/Looks/puddle")
    for k in range(3):
        photoreal.make_puddle(stage, f"/World/P{k}", (k, 0.0, 0.0), 0.5, seed=k, material=look)
    assert not stage.GetPrimAtPath("/World/P0/look")
    assert photoreal.set_rain(stage, 0.4) == 3
    assert UsdShade.Shader.Get(stage, "/World/Looks/puddle/shader").GetInput("rain").Get() == pytest.approx(0.4)
