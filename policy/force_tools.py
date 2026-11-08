from __future__ import annotations

import numpy as np

F_MID, F_HALF = 25.0, 20.0


def clamp_force_action(a3, cap: float, mid: float = F_MID, half: float = F_HALF):
    hi = (cap - mid) / half
    return np.minimum(a3, hi) if isinstance(a3, np.ndarray) else min(float(a3), hi)


def setpoint_action(force: float, mid: float = F_MID, half: float = F_HALF) -> float:
    return (float(force) - mid) / half


def intervals(t, flags) -> list[tuple[float, float]]:
    out, start = [], None
    for ti, f in zip(t, flags):
        if f and start is None:
            start = ti
        elif not f and start is not None:
            out.append((start, ti))
            start = None
    if start is not None:
        out.append((start, t[-1]))
    return out


def crossing(x, y, level: float):
    x, y = np.asarray(x, float), np.asarray(y, float)
    for k in range(1, len(x)):
        if y[k - 1] < level <= y[k]:
            return float(x[k - 1] + (x[k] - x[k - 1]) * (level - y[k - 1]) / (y[k] - y[k - 1]))
    return None


def saturation_point(x, y, frac: float = 0.95):
    y = np.asarray(y, float)
    return crossing(x, y, frac * float(y.max()))


def resample_rows(grid, rows: int) -> np.ndarray:
    g = np.asarray(grid, np.float32)
    src = np.linspace(0.0, 1.0, g.shape[0])
    dst = np.linspace(0.0, 1.0, rows)
    return np.stack([np.interp(dst, src, g[:, j]) for j in range(g.shape[1])], 1)


def force_stats(force, dt: float, limit: float = 40.0, contact: float = 1.0) -> dict:
    f = np.asarray(force, float)
    on = f > contact
    over = f > limit
    return {
        "peak_n": float(f.max()) if f.size else 0.0,
        "mean_contact_n": float(f[on].mean()) if on.any() else 0.0,
        "p95_contact_n": float(np.percentile(f[on], 95)) if on.any() else 0.0,
        "frac_over_limit": float(over.mean()) if f.size else 0.0,
        "frac_contact_over_limit": float(over[on].mean()) if on.any() else 0.0,
        "time_over_limit_s": float(over.sum() * dt),
        "contact_s": float(on.sum() * dt),
    }


def telemetry_frame(t, state, force_n, rust_fraction, detector_conf, events, grid, rows: int = 40, **extra) -> dict:
    g = np.clip(resample_rows(grid, rows), 0.0, 1.0)
    fr = {"t": round(float(t), 3), "state": str(state), "force_n": round(float(force_n), 2),
          "rust_fraction": None if rust_fraction is None else round(float(rust_fraction), 4),
          "detector_conf": None if detector_conf is None else round(float(detector_conf), 3),
          "events": list(events), "rust_grid": np.round(g, 3).tolist()}
    fr.update(extra)
    return fr
