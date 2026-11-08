from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from integration.dual_argon import (PIPE_SENSOR_FAULT, SCRUBBER, FIXER, EdgeOnly, FixedDetector, Finding, fuse_fleet,
                                    led)
from policy.control.envelope import Envelope
from policy.control.port import MemoryCoordinator

STUCK_LEL = 15.0
STUCK = "STUCK"
TICKET_KIND = "sensor_maintenance"
BANNER_OFFLINE = "CLOUD OFFLINE — edge rules only"
ROBOTS = ({"id": SCRUBBER, "kind": "scrubber", "state": "SCRUB"}, {"id": FIXER, "kind": "fixer", "state": "DOCKED"})


@dataclass
class Health:
    status: str
    evidence: dict


class FlatHealth:
    def __init__(self, min_span_s: float = 6.0, min_samples: int = 8, eps: float = 0.01, window_s: float = 30.0):
        self.min_span_s, self.min_samples, self.eps, self.window_s = min_span_s, min_samples, eps, window_s
        self.hist: dict[str, deque] = {}

    def observe(self, t: float, values: dict, kind: str = "gas", group: str = "") -> dict:
        out = {}
        for sid, v in values.items():
            h = self.hist.setdefault(sid, deque())
            if v is not None:
                h.append((float(t), float(v)))
            while h and t - h[0][0] > self.window_s:
                h.popleft()
            recent = np.array([x for x in h if x[0] >= t - self.min_span_s] or [(t, np.nan)])
            span = float(recent[-1, 0] - recent[0, 0])
            std = float(np.std(recent[:, 1]))
            flat = len(recent) >= self.min_samples and span >= self.min_span_s - 1e-9 and std <= self.eps
            out[sid] = Health(STUCK if flat else "OK",
                              {"std": round(std, 4), "span_s": round(span, 1), "value": h[-1][1] if h else None})
        return out


def health_monitor(min_span_s: float = 6.0, min_samples: int = 8):
    try:
        from policy.control.health import HealthMonitor
        return HealthMonitor(min_span_s=min_span_s, min_samples=min_samples)
    except ImportError:
        return FlatHealth(min_span_s=min_span_s, min_samples=min_samples)


def detector(scenario, sid: str) -> FixedDetector:
    if isinstance(scenario, FixedDetector):
        return scenario
    ds = getattr(scenario, "detectors", scenario)
    if isinstance(ds, dict):
        return ds[sid]
    return next(d for d in ds if d.id == sid)


class Inbox:
    def __init__(self, port=None, envelope: Optional[Envelope] = None):
        self.port = port if port is not None else MemoryCoordinator(list(ROBOTS))
        self.envelope = envelope or Envelope()
        self.filed: dict[str, dict] = {}

    def file(self, tk: dict, fd: Optional[Finding] = None, health: Optional[str] = None) -> dict:
        why = tk.get("reason") or f"maintenance ticket for {tk['sensor']}"
        params = {"ticket": tk["id"], "health": health, "source": tk.get("source")}
        if fd is not None:
            params.update(finding=fd.kind, flange=fd.flange, fixed_lel=fd.fixed_lel, robot=fd.robot,
                          robot_lel=fd.robot_lel, robot_dist=fd.robot_dist)
        jid = self.port.queue(TICKET_KIND, tk["sensor"], level="SUPV", priority=3, params=params, by="fusion", why=why)
        sent = None
        if tk.get("source") == "argon":
            sent = self.envelope.call(self.port, "approve_request",
                                      {"job_id": jid, "why": f"{tk['id']} {tk['sensor']}: {why}"[:200]})
        rec = {"ticket": tk["id"], "job_id": jid, "sensor": tk["sensor"], "source": tk.get("source"),
               "status": self.port.job(jid)["status"], "health": health, "envelope": sent}
        self.filed[tk["id"]] = rec
        return rec

    def sync(self, link, health: Optional[dict] = None) -> list[dict]:
        out = []
        for tk in link.tickets:
            if tk["id"] not in self.filed:
                out.append(self.file(tk, link.findings.get(tk["sensor"]), (health or {}).get(tk["sensor"])))
        return out

    def waiting(self) -> list[str]:
        return list(self.port.fleet_state()["approvals"])


class StuckSensor:
    def __init__(self, scenario, at: float, sensor: str = "G-F6", value: float = STUCK_LEL,
                 inbox: Optional[Inbox] = None, health=None):
        self.det = detector(scenario, sensor)
        self.at, self.sensor, self.value = float(at), sensor, float(value)
        self.inbox = inbox or Inbox()
        self.health = health if health is not None else health_monitor()
        self.active = False
        self.fault: Optional[Finding] = None
        self.health_status = "OK"
        self.last: Optional[Finding] = None

    def tick(self, t: float, sniffers: Optional[dict] = None, finding: Optional[Finding] = None) -> Finding:
        if not self.active and t >= self.at:
            self.active = True
            self.det.stuck_at = self.value
        if self.active:
            self.det.last = round(self.value, 2)
        h = self.health.observe(t, {self.sensor: self.det.last}, group=self.det.flange).get(self.sensor)
        self.health_status = getattr(h, "status", "OK")
        fd = self.fault or finding or fuse_fleet(self.det, sniffers or {})
        if self.fault is None and fd.kind == PIPE_SENSOR_FAULT and not fd.hazard:
            if self.health_status == STUCK:
                fd.reason += f"; health {STUCK}"
            self.fault = fd
        self.last = fd
        return fd

    def findings(self, t: float, detectors: list, sniffers: dict) -> list[Finding]:
        return [self.tick(t, sniffers) if d is self.det else fuse_fleet(d, sniffers) for d in detectors]

    def sync(self, link) -> list[dict]:
        return self.inbox.sync(link, {self.sensor: self.health_status})

    @property
    def led(self) -> str:
        return led(self.det.last, self.fault is not None)

    @property
    def verdict(self) -> str:
        if self.fault is None:
            return self.last.kind if self.last else "OK"
        return f"{PIPE_SENSOR_FAULT}+{STUCK}" if self.health_status == STUCK else PIPE_SENSOR_FAULT

    def ticket(self) -> Optional[dict]:
        return next((r for r in self.inbox.filed.values() if r["sensor"] == self.sensor), None)

    def hud(self) -> dict:
        tk = self.ticket()
        return {"sensor": self.sensor, "led": self.led, "verdict": self.verdict, "health": self.health_status,
                "ticket": tk and tk["ticket"], "job": tk and tk["job_id"],
                "status": tk and self.inbox.port.job(tk["job_id"])["status"]}

    def clear(self) -> None:
        self.det.stuck_at, self.active, self.fault = None, False, None


class CutLink:
    def __init__(self, link, at: float, until: Optional[float] = None, agent=None):
        self.link, self.at, self.until, self.agent = link, float(at), until, agent
        self.events: list[dict] = []

    def down(self, t: float) -> bool:
        return t >= self.at and (self.until is None or t < self.until)

    def tick(self, t: float) -> Optional[dict]:
        want = self.down(t)
        if want == self.link.link_cut:
            return None
        self.link.cut(want)
        if self.agent is not None:
            self.agent.cut_link(want)
        dropped = []
        if want:
            for r in ([self.link.pending] if self.link.pending else []) + list(self.link.queue):
                if not isinstance(r.result, EdgeOnly):
                    r.result, r.due = EdgeOnly(), t
                    for c in r.fallback:
                        c.reason = f"link cut -> {c.reason}"
                    dropped.append(r.key)
        ev = {"t": round(float(t), 2), "event": "link_cut" if want else "link_restored", "banner": self.banner(),
              "dropped": dropped}
        self.events.append(ev)
        self.link.timeline.log(ev["event"], t, banner=ev["banner"], dropped=dropped)
        return ev

    def banner(self) -> str:
        return banner(self.link)


def banner(link) -> str:
    return BANNER_OFFLINE if link.link_cut else ""


def stuck_sensor(scenario, at: float, sensor: str = "G-F6", value: float = STUCK_LEL, **kw) -> StuckSensor:
    return StuckSensor(scenario, at, sensor, value, **kw)


def cut_link(link, at: float, until: Optional[float] = None, agent=None) -> CutLink:
    return CutLink(link, at, until, agent)
