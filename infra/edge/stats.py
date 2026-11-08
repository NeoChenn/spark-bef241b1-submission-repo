from __future__ import annotations

import numpy as np


def latency_stats(ms, batch: int = 1) -> dict:
    a = np.asarray(list(ms), np.float64)
    if a.size == 0:
        raise ValueError("no samples")
    p50 = float(np.percentile(a, 50))
    mean = float(a.mean())
    return {"n": int(a.size), "mean_ms": mean, "p50_ms": p50, "p90_ms": float(np.percentile(a, 90)),
            "p99_ms": float(np.percentile(a, 99)), "min_ms": float(a.min()), "max_ms": float(a.max()),
            "hz_p50": 1000.0 / p50, "throughput_per_s": batch * 1000.0 / mean, "batch": int(batch)}


def parse_smi_power(lines) -> list[float]:
    out = []
    for ln in lines:
        s = ln.strip().split(",")[0].replace("W", "").strip()
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


def power_stats(w) -> dict:
    a = np.asarray(list(w), np.float64)
    if a.size == 0:
        return {"n": 0, "mean_w": None, "p50_w": None, "max_w": None}
    return {"n": int(a.size), "mean_w": float(a.mean()), "p50_w": float(np.median(a)), "max_w": float(a.max())}


def summarize_power_window(load, base, mean_ms: float, batch: int = 1) -> dict:
    lp, bp = power_stats(load), power_stats(base)
    out = {"load_mean_w": lp["mean_w"], "load_max_w": lp["max_w"], "base_mean_w": bp["mean_w"],
           "n_load": lp["n"], "n_base": bp["n"], "delta_w": None, "mj_per_inference_total": None,
           "mj_per_inference_delta": None}
    if lp["n"] == 0:
        return out
    out["mj_per_inference_total"] = lp["mean_w"] * mean_ms / batch
    if bp["n"]:
        out["delta_w"] = lp["mean_w"] - bp["mean_w"]
        out["mj_per_inference_delta"] = out["delta_w"] * mean_ms / batch
    return out


def agreement(ref, got, thr_logit: float = 0.0) -> dict:
    r, g = np.asarray(ref, np.float64).reshape(-1), np.asarray(got, np.float64).reshape(-1)
    d = np.abs(r - g)
    return {"n": int(r.size), "decision_agreement": float(((r >= thr_logit) == (g >= thr_logit)).mean()),
            "max_abs_logit_diff": float(d.max()), "mean_abs_logit_diff": float(d.mean())}


def binary_metrics(z, y, thr_logit: float = 0.0) -> dict:
    z, y = np.asarray(z, np.float64).reshape(-1), np.asarray(y).astype(int).reshape(-1)
    p = (z >= thr_logit).astype(int)
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    tn, fn = int(((p == 0) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": int(y.size), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": float((p == y).mean()),
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def window_samples(samples, windows, settle: float = 0.15) -> list[float]:
    return [v for t, v in samples if any(t0 + settle <= t <= t1 for t0, t1 in windows)]


def loop_rate(stage_ms) -> float:
    t = float(sum(stage_ms))
    return float("inf") if t <= 0 else 1000.0 / t
