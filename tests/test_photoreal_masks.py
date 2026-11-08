import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import photoreal_masks as pm


def test_noise_is_tileable_and_normalised():
    n = pm.tileable_noise(128, 6.0, seed=3)
    assert n.shape == (128, 128)
    assert abs(n.min()) < 1e-6 and abs(n.max() - 1.0) < 1e-6
    inner = np.abs(np.diff(n, axis=0)).mean()
    wrap = np.abs(n[0] - n[-1]).mean()
    assert wrap < 3 * inner


def test_noise_is_deterministic_and_seeded():
    assert np.array_equal(pm.tileable_noise(64, 4.0, seed=1), pm.tileable_noise(64, 4.0, seed=1))
    assert not np.array_equal(pm.tileable_noise(64, 4.0, seed=1), pm.tileable_noise(64, 4.0, seed=2))


def test_lower_frequency_is_smoother():
    rough = np.abs(np.diff(pm.tileable_noise(128, 32.0, seed=0), axis=1)).mean()
    smooth = np.abs(np.diff(pm.tileable_noise(128, 4.0, seed=0), axis=1)).mean()
    assert smooth < rough


def test_puddle_coverage_grows_with_rain():
    field = pm.tileable_noise(128, 3.0, seed=5)
    covers = [pm.puddle_coverage(field, level) for level in (0.0, 0.3, 0.6, 1.0)]
    assert covers[0] == 0.0
    assert all(a < b for a, b in zip(covers, covers[1:]))
    assert covers[-1] < 0.6


def test_masks_pack_four_channels():
    rgba = pm.masks(64, seed=2)
    assert rgba.shape == (64, 64, 4) and rgba.dtype == np.uint8
    assert all(rgba[..., c].std() > 5 for c in range(4))


def test_streaks_run_vertically_and_tile():
    s = pm.streaks(128, seed=4)
    assert s.shape == (128, 128) and s.dtype == np.uint8
    vertical = np.abs(np.diff(s.astype(float), axis=0)).mean()
    horizontal = np.abs(np.diff(s.astype(float), axis=1)).mean()
    assert vertical < horizontal
    assert np.abs(s[0].astype(float) - s[-1]).mean() < 3 * vertical + 1
    assert 0.08 < (s > 64).mean() < 0.6
