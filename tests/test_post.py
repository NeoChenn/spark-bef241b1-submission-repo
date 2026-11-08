import numpy as np

from rig.post import LensDrops, grade, heat_shimmer


def frame(h=120, w=160):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([(xx * 7) % 255, (yy * 5) % 255, ((xx + yy) * 3) % 255], -1).astype(np.uint8)


def test_shimmer_zero_strength_is_identity():
    img = frame()
    assert np.array_equal(heat_shimmer(img, (80, 60), 30, 1.0, 0.0), img)


def test_shimmer_only_touches_column_above_source():
    img = frame()
    out = heat_shimmer(img, (80, 100), 20, 0.7, 3.0)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out[40:100, 70:90], img[40:100, 70:90])
    assert np.array_equal(out[:, :40], img[:, :40])
    assert np.array_equal(out[110:], img[110:])


def test_shimmer_offscreen_source_is_identity():
    img = frame()
    assert np.array_equal(heat_shimmer(img, (-500, 60), 20, 0.3, 3.0), img)


def test_lens_starts_clean_and_collects_rain():
    img = frame()
    lens = LensDrops(seed=0)
    assert np.array_equal(lens.apply(img), img)
    for _ in range(40):
        lens.step(0.1, rain=1.0)
    assert len(lens.drops) > 0
    out = lens.apply(img)
    assert out.shape == img.shape and not np.array_equal(out, img)


def test_lens_drops_dry_off_without_rain():
    lens = LensDrops(seed=1)
    for _ in range(40):
        lens.step(0.1, rain=1.0)
    for _ in range(400):
        lens.step(0.1, rain=0.0)
    assert len(lens.drops) == 0


def test_grade_neutral_is_identity_and_lift_brightens_shadows():
    img = frame()
    assert np.array_equal(grade(img), img)
    dark = np.full((4, 4, 3), 10, np.uint8)
    assert grade(dark, lift=(0.05, 0.04, 0.03)).mean() > dark.mean()
