from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .plume import GAS_BASE, GasSensor, shape
from .safety import SafetyLimits

CHI2_95_2DOF = 5.99
A_MAX = 1.0


@dataclass
class Sample:
    xy: tuple[float, float]
    lel: float
    phase: str


@dataclass
class Survey:
    samples: list[Sample] = field(default_factory=list)
    path_m: float = 0.0
    time_s: float = 0.0
    events: list[str] = field(default_factory=list)
    detected: bool = False
    estimate_xy: tuple[float, float] | None = None
    radius_m: float | None = None
    amplitude: float | None = None
    peak_lel: float = 0.0
    fit_chi2_red: float | None = None
    radius_history: list[float] = field(default_factory=list)

    def arrays(self):
        P = np.array([s.xy for s in self.samples], float)
        c = np.array([s.lel for s in self.samples], float)
        return P, c


class Seeker:
    def __init__(self, state: dict, sensor: GasSensor, start_xy, speed_mps: float = 1.0,
                 limits: SafetyLimits = SafetyLimits(), decay_m: float = 1.5):
        self.state, self.sensor, self.limits = state, sensor, limits
        self.pos = np.asarray(start_xy, float)[:2].copy()
        self.speed = max(speed_mps, 0.1)
        self.decay_m = decay_m
        self.s = Survey()

    def goto(self, xy):
        xy = np.asarray(xy, float)
        d = float(np.linalg.norm(xy - self.pos))
        self.s.path_m += d
        self.s.time_s += d / self.speed
        self.pos = xy.copy()

    def read(self, phase: str) -> float:
        v = self.sensor.sample(self.state, self.pos)
        self.s.time_s += self.sensor.dwell_s
        self.s.samples.append(Sample((float(self.pos[0]), float(self.pos[1])), v, phase))
        self.s.peak_lel = max(self.s.peak_lel, v)
        return v

    def round(self, route, spacing_m: float = 2.0, detect_excess: float = 0.015) -> bool:
        pts = [np.asarray(p, float)[:2] for p in route]
        prev = self.pos.copy()
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(1, int(math.ceil(np.linalg.norm(b - a) / spacing_m)))
            for k in range(n):
                self.goto(a + (b - a) * k / n)
                v = self.read("round")
                if v - GAS_BASE > detect_excess:
                    self.s.detected = True
                if v >= self.limits.gas_stop_fraction_lel:
                    self.s.events.append(f"STOP limit on the round: {v:.3f} LEL; backed off and left the route")
                    self.goto(prev)
                    return True
                if v >= self.limits.gas_approach_fraction_lel:
                    self.s.events.append(f"approach limit on the round ({v:.3f} LEL); left the route")
                    return True
                prev = self.pos.copy()
        return self.s.detected

    def climb(self, steps: int = 12, step_m: float = 1.0, probe_m: float = 0.5):
        safe = [q for q in self.s.samples if q.lel < self.limits.gas_approach_fraction_lel] or self.s.samples
        best = max(safe, key=lambda q: q.lel)
        self.goto(best.xy)
        here = self.read("climb")
        w = np.asarray(self.state.get("wind") or (0.0, 0.0), float)[:2]
        upwind = -w / np.linalg.norm(w) if np.linalg.norm(w) > 1e-6 else np.zeros(2)
        prev = self.pos.copy()
        for _ in range(steps):
            if here >= self.limits.gas_approach_fraction_lel:
                self.s.events.append(f"approach limit {self.limits.gas_approach_fraction_lel:.2f} LEL reached "
                                     f"({here:.3f}); stop climbing")
                break
            centre = self.pos.copy()
            g = np.zeros(2)
            for k, axis in enumerate((np.array([1.0, 0.0]), np.array([0.0, 1.0]))):
                vals = []
                for sgn in (1, -1):
                    self.goto(centre + sgn * probe_m * axis)
                    vals.append(self.read("probe"))
                g[k] = (vals[0] - vals[1]) / (2 * probe_m)
            self.goto(centre)
            if np.linalg.norm(g) < 1e-4:
                self.s.events.append("gradient lost; cast upwind")
                d = upwind if np.linalg.norm(upwind) else np.array([1.0, 0.0])
            else:
                d = g / np.linalg.norm(g) + 0.3 * upwind
                d /= np.linalg.norm(d)
            e = max(here - GAS_BASE, 1e-4)
            k = float(np.linalg.norm(g)) / e
            room = math.log(max((self.limits.gas_approach_fraction_lel - GAS_BASE) / e, 1.0))
            s = min(step_m, room / k) if k > 1e-6 else step_m
            if s < 0.15:
                self.s.events.append(f"next step would pass the approach limit "
                                     f"{self.limits.gas_approach_fraction_lel:.2f} LEL; stop climbing")
                break
            prev = self.pos.copy()
            self.goto(self.pos + s * d)
            here = self.read("climb")
            if here >= self.limits.gas_stop_fraction_lel:
                self.s.events.append(f"STOP limit: {here:.3f} LEL >= {self.limits.gas_stop_fraction_lel:.2f}; "
                                     "backed off")
                self.goto(prev)
                break

    def fit(self, half_m: float = 8.0, res_m: float = 0.1):
        P, c = self.s.arrays()
        centre = P[np.argmax(c)]
        ax = np.arange(-half_m, half_m + 1e-9, res_m)
        gx, gy = np.meshgrid(centre[0] + ax, centre[1] + ax, indexing="ij")
        cand = np.stack([gx.ravel(), gy.ravel()], axis=1)
        y = c - GAS_BASE
        sig = np.array([self.sensor.sigma(v) for v in y])
        wts = 1.0 / sig ** 2
        chi = np.empty(len(cand))
        amp = np.empty(len(cand))
        wind = self.state.get("wind")
        for i0 in range(0, len(cand), 4096):
            S = shape(P[None, :, :], cand[i0:i0 + 4096, None, :], wind, self.decay_m)
            A = np.clip((S * y * wts).sum(1) / np.maximum((S * S * wts).sum(1), 1e-12), 0.0, A_MAX)
            chi[i0:i0 + 4096] = (((y - A[:, None] * S) ** 2) * wts).sum(1)
            amp[i0:i0 + 4096] = A
        k = int(np.argmin(chi))
        dof = max(len(y) - 3, 1)
        red = float(chi[k] / dof)
        scale = max(1.0, red)
        region = cand[(chi - chi[k]) / scale <= CHI2_95_2DOF]
        est = cand[k]
        radius = float(np.max(np.linalg.norm(region - est, axis=1))) + res_m
        self.s.estimate_xy = (float(est[0]), float(est[1]))
        self.s.amplitude = float(amp[k])
        self.s.radius_m = radius
        self.s.fit_chi2_red = red
        self.s.radius_history.append(radius)
        self._region = region
        return est, radius

    def refine(self, target_radius_m: float = 0.5, budget: int = 12, rng=None):
        rng = rng or np.random.default_rng(0)
        for _ in range(budget):
            if self.s.radius_m is not None and self.s.radius_m <= target_radius_m:
                break
            est = np.array(self.s.estimate_xy)
            region = self._region
            if len(region) > 200:
                region = region[rng.choice(len(region), 200, replace=False)]
            ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
            ring = np.stack([np.cos(ang), np.sin(ang)], 1)
            centres = [est, self.pos, np.array(max(self.s.samples, key=lambda q: q.lel).xy)]
            cands = np.concatenate([c + r * ring for c in centres for r in (1.0, 1.75, 2.5, 3.5)])
            pred_est = GAS_BASE + self.s.amplitude * shape(cands, est, self.state.get("wind"), self.decay_m)
            S = shape(cands[None, :, :], region[:, None, :], self.state.get("wind"), self.decay_m)
            s_est = (pred_est - GAS_BASE) / max(self.s.amplitude, 1e-9)
            worst = np.maximum(s_est, np.median(S, axis=0))
            safe = GAS_BASE + 1.5 * self.s.amplitude * worst < self.limits.gas_approach_fraction_lel
            if not safe.any():
                self.s.events.append("no safe information sample left")
                break
            cands, pred_est, S = cands[safe], pred_est[safe], S[:, safe]
            pred = self.s.amplitude * S
            noise = np.array([self.sensor.sigma(v - GAS_BASE) for v in pred_est]) ** 2
            gain = pred.var(0) / noise
            travel = np.linalg.norm(cands - self.pos, axis=1)
            nxt = cands[int(np.argmax(gain - 0.05 * travel))]
            self.goto(nxt)
            self.read("info")
            self.fit()
