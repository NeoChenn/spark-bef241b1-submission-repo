from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from policy.leak.diagnose import LeakEvidence, LeakKind, diagnose
from policy.orchestrator import TERMINAL, Orchestrator, S
from policy.orchestrator import overlay_text as rust_overlay

RUST_WORK = {S.APPROACH, S.SPRAY, S.SOAK, S.SCRUB, S.VERIFY, S.PAINT, S.VERIFY_PAINT}
LEAK_STATES = {S.GAS_CHECK, S.LEAK_APPROACH, S.LEAK_INSPECT, S.LEAK_SECOND_VIEW, S.RETORQUE, S.LEAK_VERIFY,
               S.GAS_RETREAT}


@dataclass
class LeakConfig:
    gas_clear: float = 0.05
    gas_stop: float = 0.20
    tolerance_rad: float = 0.06
    suspect_rad: float = 0.03
    max_suspects: int = 2
    min_conf: float = 0.7
    crack_threshold: float = 0.5
    reads_per_view: int = 3
    max_leak_views: int = 2
    read_dt: float = 1.0
    verify_reads: int = 4
    max_settle_reads: int = 8
    retry_limit: int = 1
    rise_margin: float = 0.03
    retreat_m: float = 3.0


@dataclass
class LeakJob:
    localizer: object
    config: LeakConfig = field(default_factory=LeakConfig)


ALERTS = {
    "repaired": ("info", False, "Leak at {fid} repaired automatically: re-torqued bolt(s) {bolts}; gas back to normal.",
                 "No action needed. Re-check this flange on the next round."),
    "clean": ("info", False, "Flange {fid} checked: bolts within tolerance, no crack.", "No action needed."),
    "crack_or_burst": ("critical", True, "Crack or burst suspected at {fid}; robot backed off and did not attempt repair.",
                       "Treat as loss of containment: keep people clear and follow site procedure before any repair."),
    "gas_limit": ("critical", True, "Gas {gas} at the robot; it stopped work and retreated.",
                  "Keep clear and send a gas-testing crew."),
    "needs_human": ("warning", True, "Robot stopped at {fid} and needs a human: {why}.",
                    "Send a person, or re-inspect when conditions allow."),
}


class RigOrchestrator(Orchestrator):
    def __init__(self, bridge, perception, localizer, world, arm, scrub_policy, config=None, spray=None, paint=None,
                 leak: Optional[LeakJob] = None):
        super().__init__(bridge, perception, localizer, world, arm, scrub_policy, config, spray, paint)
        self.leak = leak
        self.lcfg = leak.config if leak else LeakConfig()
        self.resolved: set[str] = set()
        self.alerts: list[dict] = []
        self.leak_resume: Optional[S] = None
        self._gas_paused = False
        self.flange_id: Optional[str] = None
        self._start_flange(None)

    def _start_flange(self, fid: Optional[str]) -> None:
        self.flange_id = fid
        n = self.bridge.flange(fid).n_bolts if fid is not None else 8
        self.leak_ev = LeakEvidence(n)
        self.leak_diag = None
        self.last_leak_reading = None
        self.leak_views = self.leak_reads = 0
        self.queue: list[int] = []
        self.target_bolt: Optional[int] = None
        self.verify_reads = self.settle_reads = 0
        self.retries: dict[int, int] = {}
        self.suspect_done = False
        self.gas_at_repair = 0.0
        self.repaired: list[int] = []

    def _gas(self) -> Optional[float]:
        if self.leak is None or not hasattr(self.bridge, "gas_reading"):
            return None
        g = self.bridge.gas_reading()
        return None if g is None or not math.isfinite(float(g)) else float(g)

    def _flange(self):
        return self.bridge.flange(self.flange_id)

    def _retreat(self) -> None:
        if hasattr(self.bridge, "retreat"):
            self.bridge.retreat(self.lcfg.retreat_m)

    def _alert(self, kind: str, **fmt) -> dict:
        sev, ack, msg, rec = ALERTS[kind]
        fmt.setdefault("fid", self.flange_id or "-")
        fmt.setdefault("bolts", self.repaired)
        fmt.setdefault("why", "")
        fmt.setdefault("gas", "--")
        a = {"severity": sev, "kind": kind, "message": msg.format(**fmt), "recommended_action": rec,
             "requires_ack": ack, "t": round(self.t, 2)}
        self.alerts.append(a)
        return a

    def _evidence(self, **extra) -> dict:
        ev = super()._evidence()
        if self.leak is not None:
            g = self._gas()
            ev["gas"] = None if g is None else round(g, 4)
            if self.flange_id is not None:
                ev["flange"] = self.flange_id
                ev["leak_diagnosis"] = self.leak_diag.kind.value if self.leak_diag else None
                if self.last_leak_reading is not None:
                    ev["leak_reading"] = self.last_leak_reading.summary()
        ev.update(extra)
        return ev

    def _occluded(self, img) -> bool:
        return img is None or float(np.asarray(img, np.float32).std()) < self.cfg.occluded_std

    def step(self):
        if self.state in TERMINAL:
            return
        if self.leak is not None and self.state != S.GAS_RETREAT:
            pre = self._gas_preempt()
            if pre is not None:
                self._record(*pre)
                return
        super().step()

    def _interrupt(self):
        if self.leak is None:
            return None
        return self._gas_preempt()

    def _gas_preempt(self):
        cfg = self.lcfg
        g = self._gas()
        if g is None:
            return S.GAS_RETREAT, "gas sensor reading invalid -> fail safe: stop and retreat", self._evidence()
        if g >= cfg.gas_stop:
            return (S.GAS_RETREAT, f"gas {g:.2f} LEL >= stop limit {cfg.gas_stop:.2f} -> stop work, retreat",
                    self._evidence())
        if g > cfg.gas_clear and self.state in RUST_WORK and not self._gas_paused:
            self._gas_paused = True
            self.leak_resume = self.state
            return (S.GAS_CHECK, f"gas {g:.2f} LEL above clear {cfg.gas_clear:.2f} during {self.state.value} "
                    "-> pause rust work, find the leak first", self._evidence())
        return None

    def _resume(self, why: str, **extra):
        nxt = self.leak_resume or S.DETECT
        self.leak_resume = None
        self._gas_paused = False
        if hasattr(self.bridge, "return_to_work"):
            self.bridge.return_to_work()
        verb = "resume " if nxt != S.DETECT else ""
        return nxt, f"{why} -> {verb}{nxt.value}", self._evidence(**extra)

    def _escalate(self, reason: str, kind: str = "needs_human", retreat: bool = True, **fmt):
        if retreat:
            self._retreat()
        return S.ESCALATE_TO_HUMAN, reason, self._evidence(alert=self._alert(kind, **fmt))

    def _s_patrol(self):
        nxt, reason, ev = super()._s_patrol()
        if self.leak is not None and nxt == S.DETECT:
            return S.GAS_CHECK, reason.replace("-> inspect", "-> gas check first"), ev
        return nxt, reason, ev

    def _s_gas_check(self):
        cfg = self.lcfg
        g = self._gas()
        if g is None:
            return S.GAS_RETREAT, "gas sensor reading invalid -> fail safe", self._evidence()
        if g <= cfg.gas_clear:
            return self._resume(f"gas {g:.3f} LEL normal")
        fid = self.bridge.nearest_flange() if hasattr(self.bridge, "nearest_flange") else None
        if fid is None or fid in self.resolved:
            return self._escalate(f"gas {g:.2f} LEL elevated but no unchecked flange nearby -> ask human",
                                  why="gas source not found")
        self._start_flange(fid)
        return (S.LEAK_APPROACH, f"gas {g:.2f} LEL above clear {cfg.gas_clear:.2f}, nearest flange {fid} "
                "-> go and look", self._evidence())

    def _s_leak_approach(self):
        if not self.bridge.approach_flange(self.flange_id):
            return self._escalate(f"flange {self.flange_id} unreachable -> ask human", why="flange unreachable")
        self.perception.request_view(0)
        return (S.LEAK_INSPECT, f"at flange {self.flange_id}, camera on the flange face "
                "-> read torque stripes, check for cracks", self._evidence())

    def _s_leak_inspect(self):
        cfg = self.lcfg
        self._advance(cfg.read_dt)
        img = self.perception.observe()
        if self._occluded(img):
            return self._hold(S.LEAK_INSPECT)
        r = self.leak.localizer.read(img, self.perception.camera(), self._flange())
        self.last_leak_reading = r
        self.leak_ev.add(r)
        self.leak_reads += 1
        if self.leak_reads < cfg.reads_per_view:
            s = r.summary()
            return (S.LEAK_INSPECT, f"view {self.leak_views} read {self.leak_reads}/{cfg.reads_per_view}: "
                    f"{s['bolts_read']} bolts read, crack p={s['crack_prob']:.2f}", self._evidence())
        g = self._gas()
        d = diagnose(self.leak_ev, g if g is not None else 0.0, cfg)
        self.leak_diag = d
        if d.kind == LeakKind.CRACK:
            return self._escalate(f"diagnosis: {d.reason} -> loss of containment, not a robot repair; backed off",
                                  kind="crack_or_burst")
        if d.kind == LeakKind.LOOSE_BOLT:
            self.queue = [k for k, _ in d.loose]
            return (S.RETORQUE, f"diagnosis: {d.reason} -> re-torque #{self.queue[0]} (no approval step)",
                    self._evidence(bolt_estimates=self.leak_ev.summary()["bolts"]))
        if d.kind == LeakKind.UNSURE:
            if self.leak_views < cfg.max_leak_views:
                return (S.LEAK_SECOND_VIEW, f"unsure: {d.reason} -> take view {self.leak_views + 1}",
                        self._evidence())
            return self._escalate(f"still unsure after {self.leak_views} extra views: {d.reason} -> ask human",
                                  retreat=g is not None and g > cfg.gas_clear, why=d.reason)
        if d.kind == LeakKind.UNEXPLAINED:
            return self._escalate(f"{d.reason} -> ask human", why=d.reason)
        self.resolved.add(self.flange_id)
        return self._resume(f"flange {self.flange_id}: {d.reason}", alert=self._alert("clean"))

    def _s_leak_second_view(self):
        self.leak_views += 1
        self.leak_reads = 0
        if not self.perception.request_view(self.leak_views):
            g = self._gas()
            return self._escalate(f"flange view {self.leak_views} unreachable while unsure -> ask human",
                                  retreat=g is not None and g > self.lcfg.gas_clear, why="no clear view")
        return S.LEAK_INSPECT, f"camera moved to flange view {self.leak_views} -> read again", self._evidence()

    def _s_retorque(self):
        cfg = self.lcfg
        k = self.queue[0]
        e = self.leak_ev.bolt(k)
        if e is None or e.conf < cfg.min_conf:
            return self._escalate(f"bolt #{k} not readable confidently before tightening -> ask human",
                                  why=f"bolt #{k} unreadable")
        if not (hasattr(self.arm, "torque") and self.arm.torque(self.flange_id, k)):
            self.retries[k] = self.retries.get(k, 0) + 1
            if self.retries[k] > cfg.retry_limit:
                return self._escalate(f"tool could not reach bolt #{k} after {self.retries[k]} tries -> ask human",
                                      why=f"bolt #{k} unreachable")
            return S.RETORQUE, f"tool missed bolt #{k} -> retry", self._evidence(bolt=k)
        g = self._gas()
        self.leak_ev.reset_bolt(k)
        self.target_bolt = k
        self.verify_reads = self.settle_reads = 0
        self.gas_at_repair = g if g is not None else 0.0
        if k not in self.repaired:
            self.repaired.append(k)
        return (S.LEAK_VERIFY, f"re-torqued bolt #{k} (read {abs(e.angle):.2f} rad off, conf {e.conf:.2f}) "
                "-> verify stripe and gas", self._evidence(bolt=k))

    def _s_leak_verify(self):
        cfg = self.lcfg
        k = self.target_bolt
        self._advance(cfg.read_dt)
        img = self.perception.observe()
        if self._occluded(img):
            return self._hold(S.LEAK_VERIFY)
        r = self.leak.localizer.read(img, self.perception.camera(), self._flange())
        self.last_leak_reading = r
        self.leak_ev.add(r)
        self.verify_reads += 1
        g = self._gas()
        g = g if g is not None else 0.0
        if self.verify_reads < cfg.verify_reads:
            return (S.LEAK_VERIFY, f"verify read {self.verify_reads}/{cfg.verify_reads}: gas {g:.3f} LEL",
                    self._evidence(bolt=k))
        e = self.leak_ev.bolt(k)
        if e is None or e.conf < cfg.min_conf:
            return self._escalate(f"cannot verify bolt #{k}: no confident reading -> ask human",
                                  retreat=g > cfg.gas_clear, why=f"bolt #{k} not verifiable")
        if abs(e.angle) > cfg.tolerance_rad:
            self.retries[k] = self.retries.get(k, 0) + 1
            if self.retries[k] > cfg.retry_limit:
                return self._escalate(f"bolt #{k} still {abs(e.angle):.2f} rad off after retry -> ask human",
                                      why=f"bolt #{k} will not align")
            return S.RETORQUE, f"bolt #{k} still {abs(e.angle):.2f} rad off -> tighten again", self._evidence(bolt=k)
        if self.queue and self.queue[0] == k:
            self.queue.pop(0)
        if g > self.gas_at_repair + cfg.rise_margin:
            return self._escalate(f"gas rising after repair ({self.gas_at_repair:.2f} -> {g:.2f} LEL) -> ask human",
                                  why="gas rising after repair")
        if self.queue:
            return S.RETORQUE, f"bolt #{k} aligned -> next bolt #{self.queue[0]}", self._evidence(bolt=k)
        if g <= cfg.gas_clear:
            self.resolved.add(self.flange_id)
            return self._resume(f"leak at {self.flange_id} fixed: bolt(s) {self.repaired} re-torqued, "
                                f"gas {g:.3f} LEL back to normal", alert=self._alert("repaired"))
        if not self.suspect_done:
            sus = sorted(((abs(est.angle), j) for j in range(self.leak_ev.n_bolts)
                          if (est := self.leak_ev.bolt(j)) is not None and est.conf >= cfg.min_conf
                          and abs(est.angle) > cfg.suspect_rad), reverse=True)[: cfg.max_suspects]
            if sus:
                self.suspect_done = True
                self.queue = [j for _, j in sus]
                return (S.RETORQUE, f"gas still {g:.3f} LEL: next most suspicious bolt(s) {self.queue} "
                        "-> tighten", self._evidence())
        if self.settle_reads < cfg.max_settle_reads:
            self.settle_reads += 1
            return S.LEAK_VERIFY, f"gas {g:.3f} LEL still settling", self._evidence()
        return self._escalate(f"gas still {g:.2f} LEL after repairs -> ask human", why="gas still elevated")

    def _s_gas_retreat(self):
        g = self._gas()
        self._retreat()
        self._advance(2.0)
        gas = "sensor failed" if g is None else f"{g:.2f} LEL"
        return (S.ESCALATE_TO_HUMAN, f"retreated {self.lcfg.retreat_m:.0f} m; gas: {gas} -> critical alert",
                self._evidence(alert=self._alert("gas_limit", gas=gas)))

    def overlay_text(self) -> list[str]:
        lines = rust_overlay(self)
        if self.leak is None:
            return lines
        g = self._gas()
        gs = "--" if g is None else f"{g:.2f}"
        flag = ""
        if g is None or g >= self.lcfg.gas_stop:
            flag = " STOP"
        elif g > self.lcfg.gas_clear:
            flag = " HIGH"
        leak_view = self.state in LEAK_STATES or (self.state == S.ESCALATE_TO_HUMAN and self.flange_id is not None)
        if leak_view:
            d = self.leak_diag.kind.value if self.leak_diag else "--"
            lines[1] = f"gas: {gs} LEL{flag}   leak: {d}   flange: {self.flange_id or '-'}"
            r = self.last_leak_reading.summary() if self.last_leak_reading else None
            lines[2] = (f"bolts read {r['bolts_read']}  worst {r['max_abs_angle']:.2f} rad  "
                        f"crack p={r['crack_prob']:.2f}") if r else "bolts: --"
        else:
            lines[1] = f"{lines[1]}   gas: {gs}{flag}"
        return lines
