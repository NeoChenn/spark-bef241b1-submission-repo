"""Condition-monitoring trends over inspection rounds: drift and step changes against a baseline.

A one-off reading above a threshold is not an alarm here; a sustained change is. Each series is median-filtered
(kills single spikes), turned into z-scores against the first ``baseline_n`` rounds, and run through a two-sided
CUSUM. A CUSUM alarm is then classified as a ``step`` (flat after the change point) or a ``drift`` (a straight
line fits the post-change segment and it keeps climbing).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class Trend:
    kind: str                    # "step" | "drift"
    direction: str               # "up" | "down"
    baseline_mean: float
    baseline_sd: float
    current_mean: float          # mean of the last 3 rounds
    delta: float                 # current_mean - baseline_mean
    delta_sigma: float
    slope_per_round: float       # least squares over the rounds since the change point
    change_round: int            # first round of the new behaviour (0-based)
    alarm_round: int             # round at which the CUSUM crossed its limit
    rounds: int

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _medfilt3(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return x.copy()
    y = x.copy()
    y[1:-1] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
    return y


def detect_change(series: Sequence[float], baseline_n: int = 10, min_sd: float = 0.0, k: float = 0.5,
                  h: float = 6.0, min_effect: float = 0.0, min_sigma: float = 3.0) -> Optional[Trend]:
    """Trend if the CUSUM alarms AND the last 3 rounds sit at least max(min_sigma * sd, min_effect) off baseline
    (the median filter makes neighbouring rounds correlated, so CUSUM alone over-alarms on small wobbles)."""
    tr = _cusum(series, baseline_n, min_sd, k, h)
    if tr is None or abs(tr.delta) < max(min_sigma * tr.baseline_sd, min_effect):
        return None
    return tr


def _cusum(series, baseline_n, min_sd, k, h) -> Optional[Trend]:
    x = np.asarray(series, float)
    if len(x) < baseline_n + 4:
        return None
    base = x[:baseline_n]
    mu = float(base.mean())
    sd = max(float(base.std(ddof=1)), min_sd, 1e-9)
    z = (_medfilt3(x) - mu) / sd
    hi = lo = 0.0
    hi_start = lo_start = baseline_n
    alarm = None
    for i in range(baseline_n, len(z)):
        hi_new, lo_new = max(0.0, hi + z[i] - k), max(0.0, lo - z[i] - k)
        if hi == 0.0 and hi_new > 0.0:
            hi_start = i
        if lo == 0.0 and lo_new > 0.0:
            lo_start = i
        hi, lo = hi_new, lo_new
        if hi > h:
            alarm = (i, hi_start, "up")
            break
        if lo > h:
            alarm = (i, lo_start, "down")
            break
    if alarm is None:
        return None
    alarm_i, _, direction = alarm
    # Locate the change and its shape: best step (baseline, then a new level) vs best hinge (baseline, then a
    # ramp) over every candidate start between the end of the baseline and the alarm. Uses the raw series.
    n = len(x)
    best_step = (math.inf, baseline_n)
    best_hinge = (math.inf, baseline_n, 0.0)
    for c in range(baseline_n, alarm_i + 1):
        pre = float(((x[baseline_n:c] - mu) ** 2).sum())
        post = x[c:]
        sse_s = pre + float(((post - post.mean()) ** 2).sum())
        d = np.arange(1, n - c + 1, dtype=float)
        s = float((d * (post - mu)).sum() / (d * d).sum())
        sse_h = pre + float(((post - mu - s * d) ** 2).sum())
        if sse_s < best_step[0]:
            best_step = (sse_s, c)
        if sse_h < best_hinge[0]:
            best_hinge = (sse_h, c, s)
    is_drift = best_hinge[0] < 0.8 * best_step[0] and abs(best_hinge[2]) * (n - best_hinge[1]) > 3.0 * sd
    if is_drift:
        kind, change_i, slope = "drift", best_hinge[1], best_hinge[2]
    else:
        kind, change_i = "step", best_step[1]
        post = x[change_i:]
        slope = float(np.polyfit(np.arange(len(post), dtype=float), post, 1)[0]) if len(post) >= 3 else 0.0
    current = float(x[-3:].mean())
    return Trend(kind, direction, mu, sd, current, current - mu, (current - mu) / sd, float(slope),
                 int(change_i), int(alarm_i), int(n))


def synth_condition_history(asset_ids: Sequence[str], rounds: int = 30, seed: int = 7,
                            faults: Optional[dict] = None) -> dict:
    """Per-asset temperature (deg C) and vibration velocity (mm/s RMS) over ``rounds`` inspection rounds.

    Default faults (on the first assets available, so the demo always has something to find):
      asset[1] temperature drift +0.6 C/round from round 18 (failing valve packing / bearing)
      asset[4] vibration step 2.0 -> 4.4 mm/s at round 22 (loosened support, cavitation)
      asset[2] one-off 12 C spike at round 25 (sun on the sensor; must NOT raise an alarm)
    """
    rng = np.random.default_rng(seed)
    ids = list(asset_ids)
    if faults is None:
        faults = {}
        if len(ids) > 1:
            faults[ids[1]] = {"temp_drift": (18, 0.6)}
        if len(ids) > 4:
            faults[ids[4]] = {"vib_step": (22, 2.4)}
        if len(ids) > 2:
            faults[ids[2]] = {"temp_spike": (25, 12.0)}
    out = {}
    for i, aid in enumerate(ids):
        t0 = 38.0 + 6.0 * rng.random()
        v0 = 1.6 + 0.8 * rng.random()
        temp = t0 + rng.normal(0, 0.5, rounds)
        vib = v0 + rng.normal(0, 0.12, rounds)
        f = faults.get(aid, {})
        if "temp_drift" in f:
            start, rate = f["temp_drift"]
            temp[start:] += rate * np.arange(1, rounds - start + 1)
        if "temp_step" in f:
            start, size = f["temp_step"]
            temp[start:] += size
        if "temp_spike" in f:
            at, size = f["temp_spike"]
            temp[at] += size
        if "vib_step" in f:
            start, size = f["vib_step"]
            vib[start:] += size
        if "vib_drift" in f:
            start, rate = f["vib_drift"]
            vib[start:] += rate * np.arange(1, rounds - start + 1)
        out[aid] = {"temp_c": [round(float(v), 2) for v in temp], "vib_mm_s": [round(float(v), 3) for v in vib]}
    return out


def rounds_to_limit(trend: Trend, limit: float) -> Optional[float]:
    """Rounds until a drifting series reaches ``limit`` at the current slope (None if not heading there)."""
    if trend.kind != "drift" or trend.slope_per_round == 0:
        return None
    n = (limit - trend.current_mean) / trend.slope_per_round
    return round(n, 1) if n > 0 and math.isfinite(n) else None
