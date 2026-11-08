from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from policy.gasbot import deck
from policy.gasbot.locate import CastSurge, SeekConfig, SourceEstimator, acoustic_flow, flow_estimate, unit
from policy.gasbot.plume import PlumeField
from policy.gasbot.sensors import SensorSuite
from policy.leak.job import LEAK_STATES, RigOrchestrator
from policy.orchestrator import S

GAS_DT = 0.1
UNIT = "unit 2 - gas"


@dataclass
class GasBotConfig:
    alarm_lel: float = 0.10
    confirm: int = 2
    patrol_dt: float = 1.0
    patrol_speed: float = 0.6
    patrol_s: float = 300.0
    acoustic_snr: float = 6.0
    acoustic_confirm: int = 3
    seek_dt: float = 0.5
    seek_max_s: float = 240.0
    found_sigma: float = 0.75
    found_dist: float = 1.8
    found_hits: int = 6
    snap_m: float = 1.5
    h2s_stop: float = 20.0
    deconv: float = 0.7


class FastReading:
    def __init__(self, t90: float, n: int = 11):
        self.tau = t90 / math.log(10.0)
        self.buf: deque = deque(maxlen=n)

    def update(self, t: float, r: float) -> None:
        self.buf.append((t, r))

    def value(self, k: float) -> float:
        if len(self.buf) < 3:
            return self.buf[-1][1] if self.buf else 0.0
        t, r = np.array(self.buf).T
        slope = np.polyfit(t - t.mean(), r, 1)[0]
        return float(r.mean() + slope * (t[-1] - t.mean()) + k * self.tau * slope)


class GasBotMixin:
    def gas_init(self, leak_world, wind, seed: int = 0, flow_full: Optional[dict] = None, h2s_ratio: float = 0.0,
                 t0: float = 0.0, machinery_err_db: float = 0.0, field=None) -> None:
        self.leak_world = leak_world
        self.wind = wind
        self.field = field or PlumeField(wind, {f: deck.leak_point(s.geom) for f, s in leak_world.flanges.items()},
                                         seed)
        self.flow_full = flow_full or {f: 2.0 for f in leak_world.flanges}
        self.suite = SensorSuite(seed=seed, h2s_ratio=h2s_ratio, machinery=deck.MACHINERY,
                                 machinery_err_db=machinery_err_db)
        self.fast = {"trunk": FastReading(self.suite.trunk.t90), "tip": FastReading(self.suite.tip.t90)}
        self.gas_t = t0
        self.at_flange: Optional[str] = None
        self.wind_est = wind.unit(mean=True)
        self.anemo_rng = np.random.default_rng(seed + 11)
        self.sensor_ok = True
        self.ogi_stats = None

    def base_xy(self) -> np.ndarray:
        raise NotImplementedError

    def base_yaw(self) -> float:
        raise NotImplementedError

    def heading(self) -> np.ndarray:
        a = self.base_yaw()
        return np.array([math.cos(a), math.sin(a)])

    def trunk_pos(self) -> np.ndarray:
        return np.r_[self.base_xy() + 0.15 * self.heading(), deck.TRUNK_Z]

    def tip_pos(self) -> np.ndarray:
        if self.at_flange is not None:
            g = self.leak_world.flanges[self.at_flange].geom
            return g.center + 0.22 * g.ref + 0.05 * g.normal
        h = self.heading()
        side = 0.3 * math.sin(2 * math.pi * self.gas_t / 6.0)
        return np.r_[self.base_xy() + 0.5 * h + side * np.array([-h[1], h[0]]), deck.TIP_Z]

    def gas_sync(self, t: float) -> None:
        while self.gas_t + GAS_DT <= t + 1e-9:
            self.gas_t += GAS_DT
            self.leak_world.step(GAS_DT)
            for f in self.leak_world.flanges:
                self.field.set_flow(f, self.leak_world.flanges[f].level * self.flow_full[f])
            self.field.step(GAS_DT)
            self.suite.update(self.field, self.trunk_pos(), self.tip_pos(), GAS_DT)
            self.fast["trunk"].update(self.gas_t, self.suite.trunk.lel)
            self.fast["tip"].update(self.gas_t, self.suite.tip.lel)
            w = self.wind.unit(mean=True)
            a = math.atan2(w[1], w[0]) + math.radians(5.0) * self.anemo_rng.normal()
            self.wind_est = unit(0.98 * self.wind_est + 0.02 * np.array([math.cos(a), math.sin(a)]))

    def gas_reading(self) -> Optional[float]:
        if not self.sensor_ok:
            return None
        return self.suite.tip.lel if self.at_flange is not None else self.suite.lel_max()

    def safety_reading(self) -> Optional[float]:
        return self.suite.trunk.lel if self.sensor_ok else None

    def h2s_ppm(self) -> float:
        return self.suite.h2s.reading

    def sense(self, k: float = 0.7) -> dict:
        lt, lp = self.suite.trunk.lel, self.suite.tip.lel
        ft = max(lt, self.fast["trunk"].value(k)) if lt > 0.004 else 0.0
        fp = max(lp, self.fast["tip"].value(k)) if lp > 0.004 else 0.0
        ac = self.suite.listen(self.base_xy(), self.field.leaks())
        return {"pos": self.base_xy(), "lel_trunk": ft, "lel_tip": fp, "raw": (lt, lp),
                "tip_rel": self.tip_pos()[:2] - self.base_xy(), "wind": self.wind_est.copy(), "acoustic": ac,
                "samples": [(self.trunk_pos(), self.suite.trunk.reading), (self.tip_pos(), self.suite.tip.reading)]}

    def nearest_flange_to(self, xy, exclude=()) -> Optional[tuple[str, float]]:
        best = None
        for f, s in self.leak_world.flanges.items():
            if f in exclude:
                continue
            d = float(np.linalg.norm(s.geom.center[:2] - np.asarray(xy, float)[:2]))
            if best is None or d < best[1]:
                best = (f, d)
        return best

    def upwind_target(self, distance: float) -> np.ndarray:
        up = -self.wind_est
        side = np.array([-up[1], up[0]])
        p = self.base_xy()
        cands = [p + distance * unit(up + s * side) for s in (0.0, 0.7, -0.7, 1.5, -1.5)]
        return max(cands, key=lambda c: (c - p) @ up - 3.0 * np.linalg.norm(c - deck.clamp(c)[0]))


class GasBotOrchestrator(RigOrchestrator):
    def __init__(self, *a, gas_cfg: Optional[GasBotConfig] = None, seek_cfg: Optional[SeekConfig] = None, **kw):
        super().__init__(*a, **kw)
        self.gcfg = gas_cfg or GasBotConfig()
        self.seek_cfg = seek_cfg or SeekConfig()
        self.confirm = 0
        self.ac_confirm = 0
        self.alarmed = False
        self.seek_t0 = None
        self.detect_t = None
        self.found_t = None
        self.fixed_t = None
        self.found_err = None
        self.est_track: list = []
        self.handoffs: list = []
        self.seeker: Optional[CastSurge] = None
        self.est: Optional[SourceEstimator] = None
        self.estimate = None
        self.flow = None
        self.samples: deque = deque(maxlen=240)
        self.ac_flows: deque = deque(maxlen=20)
        self.flow_conc = None
        self.sensed = None
        self.next_bearing = 0.0

    def _gas_preempt(self):
        g = self.bridge.safety_reading()
        h = self.bridge.h2s_ppm()
        if g is None:
            return S.GAS_RETREAT, "gas sensor reading invalid -> fail safe: stop and retreat upwind", self._evidence()
        if g >= self.lcfg.gas_stop:
            return (S.GAS_RETREAT, f"trunk gas {g:.0%} LEL >= stop limit {self.lcfg.gas_stop:.0%} "
                    "-> stop work, retreat upwind", self._evidence())
        if h >= self.gcfg.h2s_stop:
            return (S.GAS_RETREAT, f"H2S {h:.0f} ppm >= {self.gcfg.h2s_stop:.0f} ppm -> retreat upwind",
                    self._evidence())
        return None

    def _evidence(self, **extra) -> dict:
        ev = super()._evidence(**extra)
        s = self.bridge.suite
        ev["lel_trunk"], ev["lel_tip"] = round(s.trunk.lel, 4), round(s.tip.lel, 4)
        ev["h2s_ppm"] = round(s.h2s.reading, 2)
        if self.estimate is not None:
            ev["source_est"] = [round(float(v), 2) for v in self.estimate[0]] + [round(self.estimate[1], 2)]
        return ev

    def _check_alarm(self, lel: float) -> None:
        if not self.alarmed and lel >= self.gcfg.alarm_lel:
            self.alarmed = True
            x, y = self.bridge.base_xy()
            self._alert("gas_alarm", gas=f"{lel:.0%} LEL", where=f"x={x:.1f} y={y:.1f}")

    def _s_patrol(self):
        c = self.gcfg
        self.bridge.patrol_step(c.patrol_speed * c.patrol_dt, c.patrol_dt)
        s = self.bridge.sense()
        self.sensed = s
        lel = max(s["lel_trunk"], s["lel_tip"])
        self._check_alarm(max(s["raw"]))
        self.confirm = self.confirm + 1 if lel >= self.seek_cfg.hit_lel else 0
        ac = s["acoustic"]
        self.ac_confirm = self.ac_confirm + 1 if ac.snr_db >= c.acoustic_snr else 0
        if self.confirm >= c.confirm:
            self._start_seek()
            return (S.GAS_SEEK, f"gas {lel:.1%} LEL on {c.confirm} reads (>= {self.seek_cfg.hit_lel:.1%}) "
                    "-> localise source (cast and surge)", self._evidence())
        if self.ac_confirm >= c.acoustic_confirm:
            self._start_seek()
            return (S.GAS_SEEK, f"ultrasound {ac.db:.0f} dB, {ac.snr_db:.0f} dB over background on "
                    f"{c.acoustic_confirm} reads -> localise source", self._evidence())
        if self.t >= c.patrol_s:
            return S.DONE, f"patrol complete after {self.t:.0f} s, gas normal", self._evidence()
        return None, "", {}

    def _start_seek(self) -> None:
        self.detect_t = self.detect_t if self.detect_t is not None else self.t
        self.seek_t0 = self.t
        self.seeker = CastSurge(self.seek_cfg)
        self.est = SourceEstimator(deck.BOUNDS[0], (-0.4, 3.4))
        self.est.prior([s.geom.center for f, s in self.bridge.leak_world.flanges.items() if f not in self.resolved])
        self.leak_resume = S.PATROL

    def _s_gas_seek(self):
        c = self.gcfg
        s = self.bridge.sense()
        self.sensed = s
        hit = self.seeker.hit(s["lel_trunk"], s["lel_tip"])
        self.est.observe(s["pos"], s["wind"], hit)
        ac = s["acoustic"]
        if ac.bearing is not None and self.t >= self.next_bearing and not self._machinery_bearing(s["pos"], ac):
            self.est.bearing(s["pos"], ac.bearing, ac.bearing_sigma)
            self.next_bearing = self.t + 1.0
        self.estimate = self.est.estimate()
        mean, sigma, _ = self.estimate
        self.est_track.append((self.t, float(mean[0]), float(mean[1]), sigma))
        self.samples.extend(s["samples"])
        self.flow_conc = flow_estimate(list(self.samples), mean, s["wind"], self.bridge.wind.u, 1.0)
        a = self.bridge.suite.acoustic
        if ac.snr_db >= 3.0:
            self.ac_flows.append((s["pos"].copy(), ac.db, ac.background_db))
        qs = [acoustic_flow(db, bg, float(np.linalg.norm(mean - p)), a.l1m_db, a.absorb_db_m)
              for p, db, bg in self.ac_flows]
        qs = [math.log(q) for q in qs if q is not None]
        self.flow = (math.exp(float(np.median(qs))) if qs else self.flow_conc) if sigma < self.seek_cfg.trust_sigma else None
        self._check_alarm(max(s["raw"]))
        prev = self.seeker.mode
        v = self.seeker.command(self.t, s["pos"], s["lel_trunk"], s["lel_tip"], s["tip_rel"], s["wind"],
                                ac.bearing, ac.snr_db, self.estimate, getattr(self, "_blocked", False))
        dist = float(np.linalg.norm(mean - s["pos"]))
        evidence = self.est.n_hits >= c.found_hits or self.est.n_bear >= c.found_hits
        if sigma < c.found_sigma and dist < c.found_dist and evidence:
            near = self.bridge.nearest_flange_to(mean, exclude=self.resolved)
            if near is not None and near[1] <= c.snap_m:
                self.found_t = self.found_t if self.found_t is not None else self.t
                self.handoffs.append(near[0])
                self._start_flange(near[0])
                return (S.LEAK_APPROACH, f"source at ({mean[0]:.1f}, {mean[1]:.1f}) +/- {sigma:.1f} m, "
                        f"{near[1]:.1f} m from flange {near[0]} -> approach and inspect", self._evidence())
        if self.t - self.seek_t0 > c.seek_max_s:
            return self._escalate(f"no source found in {c.seek_max_s:.0f} s of seeking -> ask human",
                                  kind="source_lost", why=f"{c.seek_max_s:.0f} s")
        self._blocked = self.bridge.seek_move(v, c.seek_dt)
        if self.seeker.mode != prev:
            return (S.GAS_SEEK, f"{prev} -> {self.seeker.mode}: gas {max(s['raw']):.1%} LEL, est "
                    f"({mean[0]:.1f}, {mean[1]:.1f}) +/- {sigma:.1f} m", self._evidence())
        return None, "", {}

    def _machinery_bearing(self, pos, ac) -> bool:
        for c, l1 in deck.MACHINERY:
            d = np.asarray(c) - np.asarray(pos)[:2]
            off = abs((math.atan2(d[1], d[0]) - ac.bearing + math.pi) % (2 * math.pi) - math.pi)
            near = self.bridge.suite.acoustic.expected_db(pos) - self.bridge.suite.acoustic.background_db > 3.0
            if near and off < math.radians(25.0):
                return True
        return False

    def _resume(self, why: str, **extra):
        if self.fixed_t is None and "fixed" in why:
            self.fixed_t = self.t
        self.bridge.at_flange = None
        self.confirm = self.ac_confirm = 0
        self.seeker = None
        return super()._resume(why, **extra)

    def _escalate(self, reason: str, kind: str = "needs_human", retreat: bool = True, **fmt):
        self.bridge.at_flange = None
        return super()._escalate(reason, kind, retreat, **fmt)

    def _s_gas_retreat(self):
        self.bridge.at_flange = None
        return super()._s_gas_retreat()

    def alarm_level(self) -> str:
        s = self.bridge.suite
        if self.state == S.GAS_RETREAT or s.trunk.lel >= self.lcfg.gas_stop or s.h2s.reading >= self.gcfg.h2s_stop:
            return "RETREAT"
        if s.lel_max() >= self.gcfg.alarm_lel:
            return "ALARM"
        if s.lel_max() >= self.seek_cfg.hit_lel:
            return "DETECT"
        return "OK"

    def gas_telemetry(self) -> dict:
        b, s = self.bridge, self.bridge.suite
        ac = s.acoustic.last
        w = b.wind_est
        src = None
        if self.estimate is not None:
            m, sig, _ = self.estimate
            src = {"x": round(float(m[0]), 2), "y": round(float(m[1]), 2), "sigma_m": round(sig, 2)}
        mode = self.seeker.mode if self.seeker is not None and self.state == S.GAS_SEEK else self.state.value
        x, y = b.base_xy()
        level = self.alarm_level()
        sensors = [{"id": sid, "at": at, "kind": r.kind, "lel": round(r.lel, 4), "ppm": round(r.reading),
                    "sat": r.saturated} for r, sid, at in ((s.trunk, "IR-T", "Unit 2 trunk IR"), (s.tip, "CAT-A", "Unit 2 arm tip"))]
        return {"unit": UNIT, "mode": mode, "sensors": sensors, "max": round(s.lel_max(), 4),
                "alarm": level in ("ALARM", "RETREAT"), "level": level, "h2s_ppm": round(s.h2s.reading, 1),
                "acoustic": None if ac is None else {"db": ac.db, "bg_db": ac.background_db, "snr_db": ac.snr_db,
                                                     "bearing_deg": None if ac.bearing is None else
                                                     round(math.degrees(ac.bearing) % 360, 1)},
                "source": src, "flow_gs": None if self.flow is None else round(self.flow, 2),
                "flow_kg_h": None if self.flow is None else round(self.flow * 3.6, 2),
                "wind": {"dir_deg": round((math.degrees(math.atan2(w[1], w[0])) + 180) % 360, 1),
                         "speed_ms": round(b.wind.u, 1)},
                "robot": {"x": round(float(x), 2), "y": round(float(y), 2),
                          "yaw_deg": round(math.degrees(b.base_yaw()), 1)},
                "flange": self.flange_id, "ogi": b.ogi_stats,
                "limits": {"seek": self.seek_cfg.hit_lel, "alarm": self.gcfg.alarm_lel, "stop": self.lcfg.gas_stop,
                           "h2s_stop_ppm": self.gcfg.h2s_stop}}

    def overlay_text(self) -> list[str]:
        s = self.bridge.suite
        st = self.state.value
        if self.state == S.GAS_SEEK and self.seeker is not None:
            st += f" ({self.seeker.mode})"
        lines = [f"t={self.t:6.1f}s  UNIT 2 GAS  STATE: {st}",
                 f"CH4 trunk {s.trunk.lel:5.1%}  tip {s.tip.lel:5.1%} LEL  H2S {s.h2s.reading:4.1f} ppm  "
                 f"[{self.alarm_level()}]", "", f"why: {self.last_reason[:70]}"]
        ac = s.acoustic.last
        us = f"US {ac.db:.0f} dB" if ac else "US --"
        if self.estimate is not None:
            m, sig, _ = self.estimate
            q = "--" if self.flow is None else f"{self.flow:.1f} g/s"
            lines[2] = f"source ({m[0]:.1f}, {m[1]:.1f}) +/- {sig:.1f} m  flow {q}  {us}"
        else:
            lines[2] = f"source --  {us}"
        if self.state in LEAK_STATES and self.state != S.GAS_RETREAT:
            d = self.leak_diag.kind.value if self.leak_diag else "--"
            r = self.last_leak_reading.summary() if self.last_leak_reading else None
            lines[2] = f"flange {self.flange_id}  leak: {d}  " + (
                f"bolts {r['bolts_read']}  worst {r['max_abs_angle']:.2f} rad  crack p={r['crack_prob']:.2f}"
                if r else "")
        return lines
