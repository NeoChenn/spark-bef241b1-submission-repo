import math

import numpy as np
import pytest

from bench.stats import bootstrap_ci, episode_seeds, mean_ci, rate_ci, summarize


def test_mean_ci_known_values():
    m, lo, hi = mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m == pytest.approx(3.0)
    half = 2.7764451051977987 * math.sqrt(2.5) / math.sqrt(5)
    assert lo == pytest.approx(3.0 - half)
    assert hi == pytest.approx(3.0 + half)


def test_mean_ci_single_and_empty():
    assert mean_ci([7.0]) == (7.0, 7.0, 7.0)
    m, lo, hi = mean_ci([])
    assert math.isnan(m) and math.isnan(lo) and math.isnan(hi)


def test_mean_ci_ignores_nan():
    assert mean_ci([1.0, float("nan"), 3.0])[0] == pytest.approx(2.0)


def test_rate_ci_wilson():
    p, lo, hi = rate_ci(5, 10)
    assert p == 0.5
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)
    p, lo, hi = rate_ci(0, 100)
    assert p == 0.0 and lo == 0.0 and 0.03 < hi < 0.04
    p, lo, hi = rate_ci(100, 100)
    assert p == 1.0 and hi == pytest.approx(1.0) and 0.96 < lo < 0.97


def test_rate_ci_from_bools():
    assert rate_ci([True, False, True, True])[0] == 0.75


def test_bootstrap_ci_deterministic_and_brackets():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 1.0, 200)
    a = bootstrap_ci(np.mean, x, seed=1)
    b = bootstrap_ci(np.mean, x, seed=1)
    assert a == b
    est, lo, hi = a
    assert lo < est < hi
    assert lo < 2.0 < hi


def test_bootstrap_ci_paired_arrays():
    y = np.array([0, 0, 1, 1] * 25)
    s = np.array([0.1, 0.2, 0.8, 0.9] * 25)
    est, lo, hi = bootstrap_ci(lambda a, b: float(np.mean((b > 0.5) == a)), y, s, seed=0)
    assert est == lo == hi == 1.0


def test_episode_seeds_stable_distinct():
    s = episode_seeds(100)
    assert s == episode_seeds(100)
    assert len(set(s)) == 100
    assert episode_seeds(10) == s[:10]
    assert episode_seeds(5, base=1) != episode_seeds(5)


def test_summarize():
    d = summarize([1.0, 2.0, 3.0])
    assert d["n"] == 3 and d["mean"] == pytest.approx(2.0) and d["lo"] < 2.0 < d["hi"]
    assert set(d) == {"mean", "lo", "hi", "n"}
    p = summarize([True, False], rate=True)
    assert p["mean"] == 0.5 and p["n"] == 2
