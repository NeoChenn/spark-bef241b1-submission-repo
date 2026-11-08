import numpy as np

import sim.look as look


def test_mist_level_from_fog_density():
    assert look.mist_level(None) == 0.0 and look.mist_level(0.0) == 0.0
    assert look.mist_level(look.MIST["fog_ref"]) == 1.0 and look.mist_level(1.0) == 1.0


def test_mist_thickens_with_depth_and_glows_near_lights():
    small = np.zeros((20, 20, 3), np.uint8)
    small[10, 10] = 255
    d = np.full((20, 20), 5.0, np.float32)
    d[:, 10:] = 200.0
    f, col = look.mist_layer(small, d, 1.0)
    assert f[0, 15] > f[0, 5] and f.max() <= look.MIST["max"]
    assert col[10, 11].mean() > col[0, 0].mean()


def test_inscatter_mist_only_for_presets_without_haze():
    rgb = np.full((40, 64, 3), 20, np.uint8)
    depth = np.full((40, 64), 100.0, np.float32)
    cam = {"K": [[50, 0, 32], [0, 50, 20], [0, 0, 1]], "T": np.eye(4).tolist()}
    assert np.array_equal(look.inscatter(rgb, depth, cam, "night", (0, 0, 1)), rgb)
    out = look.inscatter(rgb, depth, cam, "night", (0, 0, 1), mist=1.0)
    assert out.mean() > rgb.mean()
