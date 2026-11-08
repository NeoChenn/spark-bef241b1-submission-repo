from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LeakKind(str, Enum):
    LOOSE_BOLT = "loose_bolt"
    CRACK = "crack_or_burst"
    UNSURE = "unsure"
    UNEXPLAINED = "gas_unexplained"
    CLEAN = "clean"


@dataclass
class BoltEstimate:
    angle: float
    conf: float
    reads: int


class LeakEvidence:

    def __init__(self, n_bolts: int):
        self.n_bolts = n_bolts
        self._w = [0.0] * n_bolts
        self._wa = [0.0] * n_bolts
        self._conf = [0.0] * n_bolts
        self._n = [0] * n_bolts
        self.crack_prob = 0.0
        self.crack_unsure_reads = 0
        self.reads = 0

    def add(self, reading) -> None:
        self.reads += 1
        for b in reading.bolts:
            if b.angle is None:
                continue
            w = b.conf ** 2
            self._w[b.bolt] += w
            self._wa[b.bolt] += w * b.angle
            self._conf[b.bolt] = max(self._conf[b.bolt], b.conf)
            self._n[b.bolt] += 1
        self.crack_prob = max(self.crack_prob, reading.crack_prob)
        self.crack_unsure_reads += int(reading.crack_unsure)

    def reset_bolt(self, k: int) -> None:
        self._w[k] = self._wa[k] = self._conf[k] = 0.0
        self._n[k] = 0

    def bolt(self, k: int) -> Optional[BoltEstimate]:
        if self._n[k] == 0:
            return None
        a = self._wa[k] / self._w[k] if self._w[k] > 0 else 0.0
        return BoltEstimate(a, self._conf[k], self._n[k])

    def summary(self) -> dict:
        out = {}
        for k in range(self.n_bolts):
            e = self.bolt(k)
            out[k] = None if e is None else [round(e.angle, 3), round(e.conf, 2)]
        return {"bolts": out, "crack_prob": round(self.crack_prob, 3), "reads": self.reads}


@dataclass
class LeakDiagnosis:
    kind: LeakKind
    reason: str
    loose: list[tuple[int, float]] = field(default_factory=list)


def diagnose(ev: LeakEvidence, gas: float, cfg) -> LeakDiagnosis:
    if ev.crack_prob >= cfg.crack_threshold:
        return LeakDiagnosis(LeakKind.CRACK, f"crack-like feature on the flange face (p={ev.crack_prob:.2f})")
    if ev.crack_unsure_reads and ev.crack_prob < cfg.crack_threshold:
        return LeakDiagnosis(LeakKind.UNSURE, f"crack check unsure (p={ev.crack_prob:.2f})")
    loose, unreliable = [], []
    for k in range(ev.n_bolts):
        e = ev.bolt(k)
        if e is None or e.conf < cfg.min_conf:
            unreliable.append(k)
        elif abs(e.angle) > cfg.tolerance_rad:
            loose.append((k, abs(e.angle)))
    if loose:
        loose.sort(key=lambda x: x[1], reverse=True)
        desc = ", ".join(f"#{k} {a:.2f} rad" for k, a in loose)
        return LeakDiagnosis(LeakKind.LOOSE_BOLT, f"{len(loose)} loose bolt(s) by torque stripe: {desc}", loose)
    if unreliable:
        return LeakDiagnosis(LeakKind.UNSURE, f"{len(unreliable)} bolt(s) not readable confidently: {unreliable}")
    if gas > cfg.gas_clear:
        return LeakDiagnosis(LeakKind.UNEXPLAINED, f"gas {gas:.2f} LEL elevated but no loose bolt or crack seen")
    return LeakDiagnosis(LeakKind.CLEAN, "all bolts within tolerance, no crack, gas normal")
