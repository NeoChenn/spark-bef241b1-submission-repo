from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from policy.control.snapshot import SensorHealth, SensorSnapshot

OK, DEAD, STUCK, DRIFT, OUT_OF_RANGE, NOISY = "OK", "DEAD", "STUCK", "DRIFT", "OUT_OF_RANGE", "NOISY"
STATUSES = (OK, DEAD, STUCK, DRIFT, OUT_OF_RANGE, NOISY)
RANGE = {"gas": (-0.5, 100.0), "thermal": (-40.0, 300.0), "acoustic": (20.0, 140.0), "pressure": (0.0, 60.0)}
STUCK_EPS = {"gas": 0.01, "thermal": 0.02, "acoustic": 0.05, "pressure": 0.002}
DRIFT_TOL = {"gas": 1.5, "thermal": 6.0, "acoustic": 8.0, "pressure": 0.6}
NOISE_FLOOR = {"gas": 0.15, "thermal": 0.4, "acoustic": 1.0, "pressure": 0.04}
UNITS = {"gas": "%LEL", "thermal": "C", "acoustic": "dB", "pressure": "bar"}
HAZARD_LEL = 5.0
GAS_ALARM_LEL = 10.0
STUCK_RATIO = 5.0
NOISY_RATIO = 4.0


@dataclass
class Reading:
    id: str
    kind: str
    value: Optional[float]
    zone: str = ""
    group: str = ""
    pos: Optional[tuple] = None


@dataclass
class Track:
    id: str
    kind: str
    zone: str = ""
    group: str = ""
    pos: Optional[tuple] = None
    hist: deque = field(default_factory=deque)
    last_t: Optional[float] = None
    last_v: Optional[float] = None
    status: str = OK
    since: float = 0.0
    evidence: dict = field(default_factory=dict)
    ok_since: Optional[float] = None
    job: Optional[str] = None
    confirmed: bool = False
    trusted_until: float = -math.inf


def readings(snap: SensorSnapshot) -> list[Reading]:
    out = [Reading(g.id, "gas", g.lel, g.zone, g.group, g.pos) for g in snap.gas]
    out += [Reading(f.id, f.kind, f.value, f.zone, f.group, f.pos) for f in snap.fixed]
    return out


def noise(v: np.ndarray) -> float:
    if len(v) < 4:
        return 0.0
    d = np.diff(v, 2)
    return float(np.median(np.abs(d - np.median(d))) / 0.6745 / math.sqrt(6.0))


class HealthMonitor:
    def __init__(self, window_s: float = 30.0, dead_s: float = 8.0, min_span_s: float = 15.0, min_samples: int = 8,
                 recover_s: float = 20.0):
        self.window_s, self.dead_s, self.min_span_s = window_s, dead_s, min_span_s
        self.min_samples, self.recover_s = min_samples, recover_s
        self.tracks: dict[str, Track] = {}
        self.t = self.t0 = 0.0

    def register(self, sid: str, kind: str, zone: str = "", group: str = "", pos=None) -> None:
        self.tracks.setdefault(sid, Track(sid, kind, zone, group, pos))

    def _ingest(self, t: float, rs: list[Reading]) -> None:
        for r in rs:
            tr = self.tracks.get(r.id)
            if tr is None:
                tr = self.tracks[r.id] = Track(r.id, r.kind, r.zone, r.group, r.pos)
            tr.zone, tr.group, tr.pos = r.zone or tr.zone, r.group or tr.group, r.pos or tr.pos
            if r.value is None:
                continue
            v = float(r.value)
            tr.hist.append((t, v))
            tr.last_t, tr.last_v = t, v
        for tr in self.tracks.values():
            while tr.hist and t - tr.hist[0][0] > self.window_s:
                tr.hist.popleft()

    def _peers(self, tr: Track) -> list[Track]:
        return [p for p in self.tracks.values() if p is not tr and p.group and p.group == tr.group
                and p.kind == tr.kind and p.hist]

    def _window(self, tr: Track) -> Optional[np.ndarray]:
        if len(tr.hist) < self.min_samples or tr.hist[-1][0] - tr.hist[0][0] < self.min_span_s:
            return None
        return np.array(tr.hist)

    def classify(self, tr: Track, t: float) -> tuple[str, dict]:
        k = tr.kind
        if tr.last_t is None or t - tr.last_t > self.dead_s:
            return DEAD, {"silent_s": round(t - (tr.last_t if tr.last_t is not None else self.t0), 1),
                          "last_value": tr.last_v}
        lo, hi = RANGE.get(k, (-math.inf, math.inf))
        if not math.isfinite(tr.last_v) or not lo <= tr.last_v <= hi:
            return OUT_OF_RANGE, {"value": tr.last_v, "lo": lo, "hi": hi}
        w = self._window(tr)
        if w is None:
            return OK, {}
        v = w[:, 1]
        peers = [(p, self._window(p)) for p in self._peers(tr)]
        peers = [(p, pw) for p, pw in peers if pw is not None and t - p.last_t <= self.dead_s]
        recent = w[w[:, 0] >= t - self.min_span_s]
        std = float(np.std(recent[:, 1]))
        eps = STUCK_EPS.get(k, 1e-3)
        nbr_std = float(np.median([np.std(pw[pw[:, 0] >= t - self.min_span_s, 1]) for _, pw in peers])) \
            if peers else None
        if std <= eps and ((nbr_std is not None and nbr_std >= STUCK_RATIO * max(std, eps))
                           or (nbr_std is None and std == 0.0)):
            return STUCK, {"std": round(std, 4), "nbr_std": None if nbr_std is None else round(nbr_std, 4),
                           "value": round(float(v[-1]), 3), "span_s": round(float(recent[-1, 0] - recent[0, 0]), 1)}
        if t < tr.trusted_until:
            return OK, {}
        if k == "gas" and max([float(v.max())] + [float(pw[:, 1].max()) for _, pw in peers]) >= GAS_ALARM_LEL:
            return OK, {}
        nz = noise(v)
        nbr_nz = float(np.median([noise(pw[:, 1]) for _, pw in peers])) if peers else 0.0
        ref = max(nbr_nz, NOISE_FLOOR.get(k, 0.0))
        if nz > NOISY_RATIO * ref:
            return NOISY, {"noise": round(nz, 3), "nbr_noise": round(nbr_nz, 3), "ratio": round(nz / ref, 1)}
        if len(peers) >= 2:
            at = {}
            for _, pw in peers:
                for pt, pv in pw:
                    at.setdefault(round(float(pt), 3), []).append(pv)
            res = [x - float(np.median(at[round(float(ti), 3)])) for ti, x in w if len(at.get(round(float(ti), 3), [])) >= 2]
            if len(res) >= self.min_samples:
                bias = float(np.median(res))
                tol = DRIFT_TOL.get(k, math.inf)
                if abs(bias) > tol:
                    nbr = float(np.median([pw[-1, 1] for _, pw in peers]))
                    return DRIFT, {"bias": round(bias, 2), "tol": tol, "nbr_median": round(nbr, 2),
                                   "value": round(float(v[-1]), 2)}
        return OK, {}

    def peer_max(self, sid: str) -> Optional[float]:
        tr = self.tracks[sid]
        vals = [p.last_v for p in self._peers(tr) if p.last_v is not None and p.status == OK]
        return max(vals) if vals else None

    def _hazard(self, tr: Track) -> bool:
        if tr.kind != "gas" or tr.status == OK:
            return False
        own = tr.last_v if tr.last_v is not None and math.isfinite(tr.last_v) else 0.0
        nbr = self.peer_max(tr.id) or 0.0
        return own >= HAZARD_LEL or nbr >= HAZARD_LEL or (tr.status == DRIFT and tr.evidence.get("bias", 0) > 0)

    def update(self, snap: SensorSnapshot) -> list[SensorHealth]:
        t = float(snap.t)
        if self.t == 0.0 and self.t0 == 0.0:
            self.t0 = t
        self.t = t
        self._ingest(t, readings(snap))
        out = []
        for tr in self.tracks.values():
            raw, ev = self.classify(tr, t)
            if raw != OK:
                if tr.status != raw and not tr.confirmed:
                    if tr.status == OK:
                        tr.since = t
                    tr.status = raw
                if tr.status == raw:
                    tr.evidence = ev
                tr.ok_since = None
            elif tr.status != OK and not tr.confirmed:
                tr.ok_since = t if tr.ok_since is None else tr.ok_since
                if t - tr.ok_since >= self.recover_s:
                    tr.status, tr.evidence, tr.ok_since, tr.job = OK, {}, None, None
            ev = dict(tr.evidence)
            if tr.status != OK:
                ev["unit"] = UNITS.get(tr.kind, "")
                if tr.kind == "gas":
                    ev["peer_max"] = self.peer_max(tr.id)
            out.append(SensorHealth(tr.id, tr.kind, tr.status, tr.zone, tr.since if tr.status != OK else t, ev,
                                    self._hazard(tr), job=tr.job))
        snap.health = out
        return out

    def attach(self, sid: str, job: Optional[str]) -> None:
        if sid in self.tracks:
            self.tracks[sid].job = job

    def clear(self, sid: str, trust_s: float = 0.0) -> None:
        tr = self.tracks.get(sid)
        if tr is None:
            return
        tr.status, tr.evidence, tr.job, tr.confirmed, tr.ok_since = OK, {}, None, False, None
        tr.trusted_until = self.t + trust_s
        tr.hist.clear()

    def confirm(self, sid: str, job: Optional[str]) -> None:
        tr = self.tracks.get(sid)
        if tr is not None:
            tr.confirmed, tr.job = True, job

    def faults(self) -> dict[str, Track]:
        return {k: v for k, v in self.tracks.items() if v.status != OK}
