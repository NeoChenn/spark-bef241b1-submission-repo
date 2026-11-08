import numpy as np
import pytest

pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom


def test_deck_flanges_match_the_geometry_the_bot_uses():
    from integration.gas_bot_bridge import add_deck_flanges
    from policy.gasbot import deck
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    prims = add_deck_flanges(stage)
    cache = UsdGeom.XformCache()
    for g in deck.flanges():
        for k, path in enumerate(prims[g.id]["bolt_prims"]):
            M = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            top = np.array(M.Transform(Gf.Vec3d(g.head_height, 0, 0)))
            assert np.linalg.norm(top - g.bolt_center(k)) < 1e-3
