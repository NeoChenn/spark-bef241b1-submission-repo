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


def test_height_fog_thickens_with_distance_and_settles_low():
    from rig.camera import look_at
    from rig.post import height_fog

    cam = look_at((0.0, -100.0, 5.0), (0.0, 0.0, 5.0), 24.0, w=64, h=36)
    img = np.zeros((36, 64, 3), np.uint8)
    near = np.full((36, 64), 20.0, np.float32)
    far = np.full((36, 64), 800.0, np.float32)
    a = height_fog(img, near, cam, density=0.01, falloff=15.0, colour=(200, 200, 200))
    b = height_fog(img, far, cam, density=0.01, falloff=15.0, colour=(200, 200, 200))
    assert b.mean() > a.mean() > 0
    assert b[-1].mean() > b[0].mean()
    assert np.array_equal(height_fog(img, far, cam, density=0.0, falloff=15.0, colour=(200, 200, 200)), img)


def test_god_rays_spread_light_from_source():
    from rig.post import god_rays

    img = np.zeros((90, 160, 3), np.uint8)
    img[40:50, 75:85] = 255
    out = god_rays(img, (80, 45), strength=0.8, samples=24)
    assert out[45, 88].sum() > 30 and out[10, 10].sum() == 0
    assert np.array_equal(god_rays(img, (80, 45), strength=0.0), img)
