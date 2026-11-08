from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy.weld.seam import Seam


@dataclass
class SensorConfig:
    laser_n: int = 64
    laser_span_mm: float = 16.0
    laser_sigma_mm: float = 0.08
    dropout: float = 0.04
    lookahead_mm: float = 12.0
    eddy_r_mm: float = 2.0
    eddy_k: float = 3.0
    eddy_sigma: float = 0.05
    current_sigma: float = 1.5
    voltage_sigma: float = 0.2
    speed_sigma: float = 0.05
    thermal_sigma: float = 3.0


DEFAULT = SensorConfig()


def laser_profile(seam: Seam, x: float, y_ref: float, rng, cfg: SensorConfig = DEFAULT):
    ys = y_ref + np.linspace(-cfg.laser_span_mm, cfg.laser_span_mm, cfg.laser_n)
    z = seam.height(x, ys) + rng.normal(0, cfg.laser_sigma_mm, cfg.laser_n)
    z[rng.random(cfg.laser_n) < cfg.dropout] = np.nan
    return ys, z


def groove_center(ys, z, min_depth: float = 0.8) -> float | None:
    ok = np.isfinite(z)
    if ok.sum() < 8:
        return None
    zf = np.interp(ys, ys[ok], z[ok])
    zf = np.convolve(np.pad(zf, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    ref = np.median(zf[np.r_[:6, -6:0]])
    d = ref - zf
    if d.max() < min_depth:
        return None
    w = np.clip(d - 0.5 * d.max(), 0, None)
    return float((w * ys).sum() / w.sum())


def eddy(seam: Seam, x: float, y: float, rng, cfg: SensorConfig = DEFAULT) -> float:
    c = seam.cfg
    i0, i1 = seam.col(x - cfg.eddy_r_mm), seam.col(x + cfg.eddy_r_mm) + 1
    gx = seam.xs[i0:i1, None]
    near = np.hypot(gx - x, seam.ys[None, :] - y) < cfg.eddy_r_mm
    n = float((seam.open_crack()[i0:i1] & near).sum()) * c.dx_mm * c.dy_mm
    return float(np.clip(1 - np.exp(-n / cfg.eddy_k) + rng.normal(0, cfg.eddy_sigma), 0, 1))


def telemetry(current_a: float, voltage_v: float, speed_mm_s: float, rng, cfg: SensorConfig = DEFAULT) -> dict:
    return {"current_a": float(current_a + rng.normal(0, cfg.current_sigma)),
            "voltage_v": float(voltage_v + rng.normal(0, cfg.voltage_sigma)),
            "speed_mm_s": float(max(speed_mm_s + rng.normal(0, cfg.speed_sigma), 0.0))}


def thermal(seam: Seam, rng, cfg: SensorConfig = DEFAULT) -> float:
    return float(seam.temp_c + rng.normal(0, cfg.thermal_sigma))


def eddy_array(seam: Seam, x: float, y: float, rng, offsets=(-4.0, -2.0, 0.0, 2.0, 4.0),
               cfg: SensorConfig = DEFAULT) -> float:
    return max(eddy(seam, x, y + o, rng, cfg) for o in offsets)
