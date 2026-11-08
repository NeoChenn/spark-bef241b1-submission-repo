from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np

from integration.dual_argon import CLEAR_LEL, FIX_LEAK, FIXER, GO_CHECK, LOG_SENSOR_FAULT, RETREAT, STOP_LEL, Command
from policy.gasbot.locate import CastSurge, SeekConfig, SourceEstimator, unit
from policy.gasbot.plume import Wind
from policy.leak.job import ALERTS, LeakConfig, LeakJob, RigOrchestrator
from policy.orchestrator import OrchestratorConfig, S

DOCKED, TRANSIT, SNIFF, INSPECT, RETORQUE, VERIFY, RETURN, RETREATING, ESCALATE = (
    "DOCKED", "TRANSIT", "SNIFF", "INSPECT", "RETORQUE", "VERIFY", "RETURN", "RETREAT", "ESCALATE")
STATES = (DOCKED, TRANSIT, SNIFF, INSPECT, RETORQUE, VERIFY, RETURN, RETREATING, ESCALATE)
INNER = {S.LEAK_APPROACH: INSPECT, S.LEAK_INSPECT: INSPECT, S.LEAK_SECOND_VIEW: INSPECT, S.RETORQUE: RETORQUE,
         S.LEAK_VERIFY: VERIFY, S.HOLD: INSPECT}


class Walker(Protocol):
    xy: np.ndarray
    yaw: float

    def go(self, goal_xy, yaw: Optional[float] = None) -> None: ...

    def step(self, dt: float) -> bool: ...


class ArmSkill(Protocol):
    progress: float

    def start(self, flange, bolt: int) -> None: ...

    def step(self, dt: float) -> str: ...


class StraightWalker:
    def __init__(self, xy, yaw: float = 0.0, speed: float = 0.8):
        self.xy = np.asarray(xy, float)[:2].copy()
        self.yaw, self.speed = float(yaw), float(speed)
        self.goal, self.goal_yaw = self.xy.copy(), None
        self.start = self.xy.copy()

    def go(self, goal_xy, yaw: Optional[float] = None) -> None:
        self.goal, self.goal_yaw = np.asarray(goal_xy, float)[:2].copy(), yaw
        self.start = self.xy.copy()

    @property
    def progress(self) -> float:
        total = float(np.linalg.norm(self.goal - self.start))
        return 1.0 if total < 1e-6 else float(np.clip(1.0 - np.linalg.norm(self.goal - self.xy) / total, 0.0, 1.0))

    def step(self, dt: float) -> bool:
        d = self.goal - self.xy
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            self.yaw = math.atan2(d[1], d[0])
            self.xy = self.xy + d * min(1.0, self.speed * dt / n)
        if float(np.linalg.norm(self.goal - self.xy)) < 1e-3:
            if self.goal_yaw is not None:
                self.yaw = self.goal_yaw
            return True
        return False


class TimedTorque:
    def __init__(self, duration_s: float = 4.0, fail: Callable[[str, int], bool] = lambda f, k: False):
        self.duration, self.fail = duration_s, fail
        self.t = 0.0
        self.job = None
        self.started: list[tuple[str, int]] = []
        self.progress = 0.0

    def start(self, flange, bolt: int) -> None:
        self.job, self.t, self.progress = (flange.id, bolt), 0.0, 0.0
        self.started.append(self.job)

    def step(self, dt: float) -> str:
        self.t += dt
        self.progress = min(1.0, self.t / self.duration)
        if self.progress < 1.0:
            return "running"
        return "failed" if self.fail(*self.job) else "done"


@dataclass
class FixerConfig:
    sniff_s: float = 8.0
    seek_dt: float = 0.5
    sniff_radius_m: float = 1.0
    stand_m: float = 0.55
    trunk_z: float = 0.45
    await_s: float = 20.0
    report_s: float = 3.0
    retreat_m: float = 3.0
    clear_lel: float = CLEAR_LEL
    stop_lel: float = STOP_LEL


class _Pipe:
    shape = (1, 1)


class _Bridge:
    def __init__(self, fx: "Fixer"):
        self.fx = fx

    def pipe(self):
        return _Pipe()

    def sim_time(self) -> float:
        return self.fx.inner_t

    def advance(self, dt: float) -> None:
        self.fx.inner_t += dt
        self.fx.debt += dt

    def gas_reading(self) -> float:
        return self.fx.onboard_lel / 100.0

    def flange(self, fid):
        return self.fx.flanges[fid]

    def approach_flange(self, fid) -> bool:
        self.fx.perception.focus(fid)
        return True

    def retreat(self, distance: float = 3.0) -> None:
        self.fx.backoff = True

    def return_to_work(self) -> None:
        pass


class _World:
    def step_soak(self, dt: float) -> None:
        pass


class _Arm:
    def __init__(self, fx: "Fixer"):
        self.fx = fx

    def current_force(self) -> float:
        return 0.0

    def torque(self, fid: str, bolt: int) -> bool:
        self.fx.arm.start(self.fx.flanges[fid], bolt)
        self.fx.torquing = (fid, bolt)
        return True


def _default_perception(world):
    from policy.leak.mock import LeakPerception
    return LeakPerception(world)


def _default_localizer():
    from detection.leak import LeakLocalizer
    return LeakLocalizer()


class Fixer:
    def __init__(self, world, dock_xy, detectors=(), walker: Optional[Walker] = None, arm: Optional[ArmSkill] = None,
                 gas_at: Optional[Callable] = None, tighten: Optional[Callable[[str, int], object]] = None,
                 wind: Optional[Wind] = None, stands: Optional[dict] = None, perception=None, localizer=None,
                 config: Optional[FixerConfig] = None, leak_config: Optional[LeakConfig] = None, robot: str = FIXER):
        self.id = robot
        self.cfg = config or FixerConfig()
        self.lcfg = leak_config or LeakConfig()
        self.flanges = {fid: s.geom for fid, s in world.flanges.items()}
        self.gas_at = gas_at or world.gas_at
        self.tighten = tighten or getattr(world, "tighten", None)
        self.detectors = {d.id: d for d in detectors}
        self.dock = np.asarray(dock_xy, float)[:2].copy()
        self.walker = walker or StraightWalker(self.dock)
        self.arm = arm or TimedTorque()
        self.wind = wind or Wind(3.0, 0.0)
        self.stands = {k: np.asarray(v, float)[:2] for k, v in (stands or {}).items()}
        self.perception = perception or _default_perception(world)
        self.inner = RigOrchestrator(_Bridge(self), self.perception, None, _World(), _Arm(self), None,
                                     OrchestratorConfig(log_path=None),
                                     leak=LeakJob(localizer or _default_localizer(), self.lcfg))
        self.t = self.inner_t = self.debt = 0.0
        self.state = DOCKED
        self.caption = "docked, waiting for Argon"
        self.job: Optional[str] = None
        self.target: Optional[str] = None
        self.sensor: Optional[str] = None
        self.torquing: Optional[tuple[str, int]] = None
        self.backoff = False
        self.alert: Optional[dict] = None
        self.alerts: list[dict] = []
        self.reports: list[dict] = []
        self.log: list[dict] = []
        self.onboard_lel = self._read()
        self._reset_sniff()

    def _reset_sniff(self) -> None:
        self.seeker: Optional[CastSurge] = None
        self.est: Optional[SourceEstimator] = None
        self.estimate = None
        self.sniff_t = 0.0
        self.sniff_next = 0.0
        self.sniff_done = False
        self.peak_lel = 0.0
        self.wait_t = 0.0
        self.reported = False
        self.diagnosis: Optional[str] = None

    @property
    def xy(self) -> np.ndarray:
        return np.asarray(self.walker.xy, float)[:2]

    def _z(self) -> float:
        return float(self.flanges[self.target].center[2]) if self.target else 1.0

    def _read(self, xy=None, z=None) -> float:
        p = np.r_[self.xy if xy is None else xy, self._z() if z is None else z]
        return round(100.0 * float(self.gas_at(p)), 2)

    def stand(self, fid: str) -> np.ndarray:
        if fid in self.stands:
            return self.stands[fid]
        g = self.flanges[fid]
        n = g.normal[:2]
        d = unit(n) if float(np.linalg.norm(n)) > 0.5 else unit(self.dock - g.center[:2])
        return g.center[:2] + self.cfg.stand_m * d

    def _face(self, fid: str) -> float:
        d = self.flanges[fid].center[:2] - self.stand(fid)
        return math.atan2(d[1], d[0])

    def _go(self, state: str, caption: str, **kw) -> None:
        self.log.append({"t": round(self.t, 2), "from": self.state, "to": state, "why": caption, **kw})
        self.state, self.caption = state, caption

    def command(self, cmd: Command) -> bool:
        if cmd.robot != self.id:
            return False
        if cmd.kind == RETREAT:
            if self.state in (DOCKED, RETREATING):
                return False
            self._retreat(f"Argon: retreat from {cmd.target or self.target}")
            return True
        if cmd.kind == LOG_SENSOR_FAULT:
            fid = self.detectors[cmd.target].flange if cmd.target in self.detectors else None
            if fid is None:
                return False
            if self.state in (DOCKED, RETURN):
                self._dispatch("fault", fid, cmd.target)
            elif self.target == fid:
                self.job, self.sensor = "fault", cmd.target
            return True
        if cmd.kind not in (GO_CHECK, FIX_LEAK) or cmd.target not in self.flanges:
            return False
        job = "fix" if cmd.kind == FIX_LEAK else "check"
        if self.state in (DOCKED, RETURN):
            self._dispatch(job, cmd.target, self._sensor_for(cmd.target))
            return True
        if self.target == cmd.target and self.state in (TRANSIT, SNIFF) and job == "fix":
            self.job = "fix"
            return True
        return False

    def handle(self, cmds) -> list[Command]:
        return [c for c in cmds if self.command(c)]

    def _sensor_for(self, fid: str) -> Optional[str]:
        return next((d.id for d in self.detectors.values() if d.flange == fid), None)

    def _dispatch(self, job: str, fid: str, sensor: Optional[str]) -> None:
        self.job, self.target, self.sensor = job, fid, sensor
        self.alert = None
        self._reset_sniff()
        self.walker.go(self.stand(fid), self._face(fid))
        verb = {"fix": "fix leak", "check": "check gas", "fault": "check head"}[job]
        self._go(TRANSIT, f"{verb} at {fid}: walking in")

    def _retreat(self, why: str) -> None:
        self.inner.state = S.DONE
        self.torquing = None
        self.walker.go(self.dock)
        self._alert("gas_limit", gas=f"{self.onboard_lel:.0f} %LEL")
        self._go(RETREATING, f"{why}: backing off to dock")

    def _alert(self, kind: str, inner: Optional[dict] = None, **fmt) -> dict:
        if inner is not None:
            a = dict(inner)
        else:
            sev, ack, msg, rec = ALERTS[kind]
            fmt.setdefault("fid", self.target or "-")
            fmt.setdefault("bolts", self.inner.repaired)
            fmt.setdefault("why", "")
            fmt.setdefault("gas", "--")
            a = {"severity": sev, "kind": kind, "message": msg.format(**fmt), "recommended_action": rec,
                 "requires_ack": ack}
        a.update(t=round(self.t, 2), unit=self.id)
        self.alert = a
        self.alerts.append(a)
        return a

    def step(self, dt: float) -> None:
        self.t += dt
        self.onboard_lel = self._read()
        if self.state not in (DOCKED, RETREATING) and self.onboard_lel >= self.cfg.stop_lel:
            self._retreat(f"sniffer {self.onboard_lel:.0f} %LEL >= stop {self.cfg.stop_lel:.0f}")
        getattr(self, "_s_" + self.state.lower())(dt)

    def _s_docked(self, dt: float) -> None:
        pass

    def _s_transit(self, dt: float) -> None:
        if self.walker.step(dt):
            self._start_sniff()
            return
        self.caption = f"walking to {self.target}: sniffer {self.onboard_lel:.1f} %LEL"

    def _start_sniff(self) -> None:
        self.seeker = CastSurge(SeekConfig(hit_lel=self.cfg.clear_lel / 100.0, surge_speed=0.4, cast_speed=0.35,
                                           cast_width=0.6, cast_max=self.cfg.sniff_radius_m))
        c = self.flanges[self.target].center
        self.est = SourceEstimator((c[0] - 4.0, c[0] + 4.0), (c[1] - 4.0, c[1] + 4.0))
        self.est.prior([c])
        self.sniff_t = self.sniff_next = 0.0
        self.peak_lel = self.onboard_lel
        self._go(SNIFF, f"at {self.target}: cast-and-surge sniff")

    def _s_sniff(self, dt: float) -> None:
        if self.sniff_done:
            self._after_sniff(dt)
            return
        self.sniff_t += dt
        self.peak_lel = max(self.peak_lel, self.onboard_lel)
        if self.sniff_t >= self.sniff_next:
            self.sniff_next += self.cfg.seek_dt
            self._seek()
        self.walker.step(dt)
        if self.sniff_t >= self.cfg.sniff_s:
            self.sniff_done = True
            self.walker.go(self.stand(self.target), self._face(self.target))

    def _seek(self) -> None:
        w = self.wind.unit()
        tip = self.onboard_lel / 100.0
        trunk = self._read(z=self.cfg.trunk_z) / 100.0
        hit = self.seeker.hit(trunk, tip)
        self.est.observe(self.xy, w, hit)
        self.estimate = self.est.estimate()
        h = np.array([math.cos(self.walker.yaw), math.sin(self.walker.yaw)])
        v = self.seeker.command(self.sniff_t, self.xy, trunk, tip, 0.3 * h, w, estimate=self.estimate)
        home = self.stand(self.target)
        nxt = self.xy + v * self.cfg.seek_dt
        off = nxt - home
        r = float(np.linalg.norm(off))
        if r > self.cfg.sniff_radius_m:
            nxt = home + off / r * self.cfg.sniff_radius_m
        self.walker.go(nxt)
        self.caption = (f"sniff {self.seeker.mode.lower()} at {self.target}: {self.onboard_lel:.1f} %LEL "
                        f"(peak {self.peak_lel:.1f})")

    def _after_sniff(self, dt: float) -> None:
        if not self.walker.step(dt):
            return
        self.wait_t += dt
        gas = self.peak_lel >= self.cfg.clear_lel
        if self.job == "fix":
            self._start_inspect(f"sniff peak {self.peak_lel:.1f} %LEL at {self.target}: inspect bolts and face")
            return
        if gas:
            if self.job == "fault":
                self._report("not_a_fault")
                self.job = "fix"
                self._start_inspect(f"{self.sensor} is right: {self.peak_lel:.1f} %LEL here -> inspect")
            elif self.wait_t >= self.cfg.await_s:
                self._start_inspect(f"no reply in {self.cfg.await_s:.0f} s, gas {self.peak_lel:.1f} %LEL -> inspect")
            else:
                self.caption = f"gas confirmed at {self.target}: {self.peak_lel:.1f} %LEL, awaiting Argon"
            return
        if not self.reported:
            self._report("head_suspect")
        self.caption = (f"{self.target} reads clean ({self.peak_lel:.1f} %LEL)"
                        + (f": {self.sensor} suspect, reported" if self.sensor else ""))
        if self.wait_t >= self.cfg.report_s:
            self._return(self.caption)

    def _report(self, verdict: str) -> dict:
        d = self.detectors.get(self.sensor)
        r = {"unit": self.id, "verdict": verdict, "sensor": self.sensor, "flange": self.target,
             "fixed_lel": None if d is None else d.last, "robot_lel": self.peak_lel,
             "robot_dist": None if d is None else round(d.dist(self.xy), 2), "t": round(self.t, 2),
             "message": (f"{self.sensor} reads {d.last:.1f} %LEL but unit2 sniffer reads {self.peak_lel:.1f} %LEL "
                         f"beside it: detector fault suspected, maintenance ticket" if verdict == "head_suspect" and d
                         else f"{self.target}: sniffer {self.peak_lel:.1f} %LEL")}
        self.reports.append(r)
        self.reported = True
        return r

    def _start_inspect(self, why: str) -> None:
        self.inner._start_flange(self.target)
        self.inner.state = S.LEAK_APPROACH
        self.debt = 0.0
        self._go(INSPECT, why)

    def _s_inspect(self, dt: float) -> None:
        self._run_inner(dt)

    _s_retorque = _s_verify = _s_inspect

    def _run_inner(self, dt: float) -> None:
        if self.torquing is not None:
            r = self.arm.step(dt)
            if r == "running":
                return
            fid, k = self.torquing
            self.torquing = None
            if r == "done" and self.tighten is not None:
                self.tighten(fid, k)
            self.caption = f"bolt #{k} {'re-torqued' if r == 'done' else 'tool slipped'}: verify"
        self.debt -= dt
        for _ in range(16):
            if self.debt > 1e-9 or self.torquing is not None or self.inner.state not in INNER:
                break
            self.inner.step()
        st = self.inner.state
        if st in INNER:
            nxt = RETORQUE if self.torquing is not None else INNER[st]
            if nxt != self.state:
                self._go(nxt, self.inner.last_reason)
            else:
                self.caption = self.inner.last_reason
            self.diagnosis = self.inner.leak_diag.kind.value if self.inner.leak_diag else None
            return
        a = self.inner.alerts[-1] if self.inner.alerts else None
        self.diagnosis = self.inner.leak_diag.kind.value if self.inner.leak_diag else self.diagnosis
        if st == S.ESCALATE_TO_HUMAN:
            self._alert("needs_human", a)
            g = self.flanges[self.target].center[:2]
            self.walker.go(g + self.cfg.retreat_m * unit(self.dock - g))
            self._go(ESCALATE, f"{self.inner.last_reason[:90]}")
            return
        if a is not None:
            self._alert(a["kind"], a)
        self._return(self.inner.last_reason.split(" -> ")[0])

    def _return(self, why: str) -> None:
        self.walker.go(self.dock)
        self._go(RETURN, f"{why[:80]}; returning to dock")

    def _s_return(self, dt: float) -> None:
        if self.walker.step(dt):
            self._go(DOCKED, "docked, waiting for Argon")
            self.job = None

    def _s_retreat(self, dt: float) -> None:
        if self.walker.step(dt):
            self._go(DOCKED, f"retreated to dock: {self.alert['message'] if self.alert else ''}"[:120])
            self.job = None

    def _s_escalate(self, dt: float) -> None:
        if not self.walker.step(dt):
            self.caption = f"backing off from {self.target}: humans needed"

    @property
    def progress(self) -> float:
        s = self.state
        if s in (TRANSIT, RETURN, RETREATING, ESCALATE):
            return float(getattr(self.walker, "progress", 0.0))
        if s == SNIFF:
            return 1.0 if self.sniff_done else min(1.0, self.sniff_t / self.cfg.sniff_s)
        if s == RETORQUE:
            return float(getattr(self.arm, "progress", 0.0)) if self.torquing else 0.0
        if s == INSPECT:
            return min(1.0, self.inner.leak_reads / self.lcfg.reads_per_view)
        if s == VERIFY:
            return min(1.0, self.inner.verify_reads / self.lcfg.verify_reads)
        return 1.0

    def sniffer(self) -> tuple[float, np.ndarray]:
        return self.onboard_lel, self.xy.copy()

    @property
    def evidence(self) -> dict:
        ev = {"unit": self.id, "state": self.state, "job": self.job, "flange": self.target, "sensor": self.sensor,
              "xy": [round(float(v), 2) for v in self.xy], "onboard_lel": self.onboard_lel,
              "peak_lel": round(self.peak_lel, 2), "progress": round(self.progress, 3), "diagnosis": self.diagnosis,
              "alert": self.alert, "report": self.reports[-1] if self.reports else None}
        if self.seeker is not None:
            ev["seek_mode"] = self.seeker.mode
        if self.estimate is not None:
            m, sig, _ = self.estimate
            ev["source_est"] = [round(float(m[0]), 2), round(float(m[1]), 2), round(float(sig), 2)]
        if self.target is not None and self.state in (INSPECT, RETORQUE, VERIFY, ESCALATE, RETURN):
            ev["bolts"] = self.inner.leak_ev.summary()["bolts"]
            ev["crack_prob"] = round(self.inner.leak_ev.crack_prob, 3)
            ev["repaired"] = list(self.inner.repaired)
            if self.torquing is not None:
                ev["bolt"] = self.torquing[1]
            elif self.inner.target_bolt is not None:
                ev["bolt"] = self.inner.target_bolt
        return ev

    def status(self) -> dict:
        return {"state": self.state, "job": self.job, "target": self.target, "pos": self.evidence["xy"],
                "sniffer_lel": self.onboard_lel, "caption": self.caption}
