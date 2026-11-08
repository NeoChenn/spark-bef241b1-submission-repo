import math

import numpy as np
import torch

from policy.storm import (Lens, Weather, bump, deck_offset, gust_force, lens_quality, sample_deck, sample_gusts,
                          wet_factor)


def scene(h=120, w=160, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 60, np.uint8)
    for _ in range(40):
        y, x = rng.integers(0, h - 12), rng.integers(0, w - 12)
        img[y:y + 10, x:x + 10] = rng.integers(0, 255, 3)
    return img


def test_bump_is_a_single_raised_cosine():
    u = torch.tensor([-0.1, 0.0, 0.25, 0.5, 1.0, 1.2])
    b = bump(u)
    assert b[0] == 0 and b[-1] == 0 and abs(b[3] - 1) < 1e-6 and abs(b[2] - 0.5) < 1e-6 and b[1] == 0


def test_calm_has_no_gust_and_no_deck_motion():
    g = sample_gusts(8, 0.0, 30.0, generator=torch.Generator().manual_seed(0))
    d = sample_deck(8, 0.0, generator=torch.Generator().manual_seed(0))
    for t in np.linspace(0, 30, 61):
        tt = torch.full((8,), float(t))
        assert torch.all(gust_force(g, tt) == 0)
        assert torch.all(deck_offset(d, tt) == 0)


def test_storm_gusts_rise_and_fall_within_bounds():
    gen = torch.Generator().manual_seed(1)
    g = sample_gusts(64, 1.0, 15.0, generator=gen, peak=(10.0, 30.0), mean=(0.0, 4.0))
    ts = torch.linspace(0, 15, 301)
    f = torch.stack([gust_force(g, torch.full((64,), float(t))) for t in ts])
    assert f.min() >= 0
    assert f.max() <= 4.0 + 30.0 * g["peak"].shape[1] + 1e-4
    assert (f.max(0).values > 10.0).float().mean() > 0.9
    assert (f.max(0).values - f.min(0).values > 5.0).all()
    assert torch.allclose(g["dir"].norm(dim=-1), torch.ones(64), atol=1e-5)
    assert torch.equal(sample_gusts(4, 1.0, 15.0, generator=torch.Generator().manual_seed(3))["t0"],
                       sample_gusts(4, 1.0, 15.0, generator=torch.Generator().manual_seed(3))["t0"])


def test_gust_force_scales_with_level():
    gen = lambda: torch.Generator().manual_seed(2)
    lo, hi = sample_gusts(32, 0.3, 15.0, generator=gen()), sample_gusts(32, 1.0, 15.0, generator=gen())
    ts = torch.linspace(0, 15, 151)
    flo = max(float(gust_force(lo, torch.full((32,), float(t))).max()) for t in ts)
    fhi = max(float(gust_force(hi, torch.full((32,), float(t))).max()) for t in ts)
    assert flo < fhi


def test_deck_motion_is_slow_and_a_few_cm():
    d = sample_deck(16, 1.0, generator=torch.Generator().manual_seed(4), amp=(0.02, 0.05), freq=(0.1, 0.3))
    ts = np.linspace(0, 60, 1201)
    off = np.stack([deck_offset(d, torch.full((16,), float(t))).numpy() for t in ts])
    assert off.shape == (1201, 16, 2)
    assert np.abs(off).max() <= 0.05 + 1e-6 and np.abs(off).max(0).min() > 0.015
    crossings = (np.diff(np.sign(off[:, :, 1]), axis=0) != 0).sum(0)
    assert crossings.min() >= 2 * 0.1 * 60 - 2 and crossings.max() <= 2 * 0.3 * 60 + 2


def test_wet_factor():
    assert wet_factor(0.0) == 1.0 and 0.5 <= wet_factor(1.0) < 0.8 and wet_factor(0.5) > wet_factor(1.0)


def test_clean_lens_is_identity():
    img = scene()
    lens = Lens(seed=0)
    assert np.array_equal(lens.apply(img, visibility_loss=0.0), img)


def test_rain_accumulates_and_wipe_clears():
    lens = Lens(seed=0)
    for _ in range(20):
        lens.rain(0.5, 1.0)
    n = len(lens.drops)
    assert n > 5
    for _ in range(20):
        lens.rain(0.5, 0.0)
    assert len(lens.drops) == n
    lens.wipe()
    assert len(lens.drops) <= max(1, n // 8)
    assert lens.dirt() < 0.05


def test_degradation_lowers_quality_monotonically():
    img = scene()
    qs = []
    for level in (0.0, 0.3, 0.6, 1.0):
        lens = Lens(seed=1)
        for _ in range(10):
            lens.rain(1.0, level)
        out = lens.apply(img, visibility_loss=level)
        assert out.shape == img.shape and out.dtype == np.uint8
        qs.append(lens_quality(out))
    assert all(a > b for a, b in zip(qs, qs[1:])), qs
    assert qs[0] > 0.6 and qs[-1] < 0.35
    assert float(np.std(Lens(seed=1).apply(img, visibility_loss=1.0))) < float(np.std(img))


def test_wipe_restores_quality_but_not_haze():
    img = scene()
    lens = Lens(seed=2)
    for _ in range(12):
        lens.rain(1.0, 1.0)
    dirty = lens_quality(lens.apply(img, visibility_loss=0.2))
    lens.wipe()
    wiped = lens_quality(lens.apply(img, visibility_loss=0.2))
    hazy = lens_quality(lens.apply(img, visibility_loss=1.0))
    assert wiped > dirty + 0.15 and hazy < wiped


def test_lens_is_deterministic():
    img = scene()
    a, b = Lens(seed=5), Lens(seed=5)
    for _ in range(5):
        a.rain(1.0, 0.8)
        b.rain(1.0, 0.8)
    assert np.array_equal(a.apply(img, 0.5), b.apply(img, 0.5))


def test_weather_calm_and_storm_with_lulls():
    w = Weather(seed=0)
    s = w.at(10.0)
    assert s["level"] == 0 and s["gust_n"] == 0 and s["rain"] == 0 and s["deck_m"] == (0.0, 0.0)
    w.set_level(1.0, t=5.0)
    xs = [w.at(t) for t in np.arange(5.0, 65.0, 0.25)]
    rain = np.array([x["rain"] for x in xs])
    gust = np.array([x["gust_n"] for x in xs])
    assert rain.max() > 0.8 and rain.min() < 0.35
    assert gust.max() > 10.0 and (gust < 3.0).mean() > 0.2
    assert max(abs(x["deck_m"][1]) for x in xs) > 0.01
    lull = w.next_lull(5.0)
    assert lull is not None and lull >= 5.0 and w.at(lull)["rain"] < 0.35
    w.set_level(0.0, t=70.0)
    assert w.at(71.0)["gust_n"] == 0 and w.at(71.0)["rain"] == 0
    assert math.isfinite(w.at(1e4)["rain"])


def test_scripted_gusts_peak_at_given_times():
    from policy.storm import scripted_gusts
    g = scripted_gusts([4.0, 9.0], peak=24.0, dur=2.0, mean=2.0, direction=(0.0, -1.0, 0.0))
    f = lambda t: float(gust_force(g, torch.tensor([t]))[0])
    assert abs(f(4.0) - 26.0) < 1e-4 and abs(f(9.0) - 26.0) < 1e-4 and abs(f(6.5) - 2.0) < 1e-4
    assert torch.allclose(g["dir"][0], torch.tensor([0.0, -1.0, 0.0]))


def test_gust_gate_hysteresis():
    from policy.storm import GustGate
    gate = GustGate(limit=12.0, resume=6.0)
    seq = [gate.update(x) for x in (3, 11, 13, 10, 7, 5.9, 8, 12.5)]
    assert seq == [False, False, True, True, True, False, False, True]


def test_storm_grade_rain_overlay():
    from policy.storm import storm_grade
    img = scene()
    assert np.array_equal(storm_grade(img, 0.0, t=1.0), img)
    a, b = storm_grade(img, 1.0, t=1.0), storm_grade(img, 1.0, t=1.0)
    assert np.array_equal(a, b) and a.shape == img.shape and a.dtype == np.uint8
    assert not np.array_equal(storm_grade(img, 1.0, t=1.1), a)
    assert a.astype(float)[..., 0].mean() < img.astype(float)[..., 0].mean()


def test_set_weather_kinds_map_onto_storm():
    from policy.storm import WEATHER
    ts = np.arange(0.0, 60.0, 0.25)
    peak = {}
    for kind in WEATHER:
        w = Weather(seed=0)
        w.set_weather(kind, t=0.0)
        xs = [w.at(t) for t in ts]
        peak[kind] = (max(x["gust_n"] for x in xs), max(x["rain"] for x in xs))
    assert peak["calm"] == (0.0, 0.0)
    assert peak["wind"][0] > 10 and peak["wind"][1] == 0
    assert peak["rain"][1] > 0.6 and peak["rain"][0] < peak["wind"][0]
    assert peak["fog"][0] == 0 and peak["fog"][1] > 0.4
    assert peak["storm"][0] >= peak["wind"][0] and peak["storm"][1] >= peak["rain"][1]
    w = Weather(seed=0)
    w.set_weather("wind", t=0.0, wind_dir=90.0, gust=40.0)
    assert abs(w.at(1.0)["gust_dir"][1] - 1.0) < 1e-6
    assert max(w.at(t)["gust_n"] for t in ts) > peak["wind"][0]
    w.set_weather(0.4, t=0.0)
    assert w.level == 0.4
