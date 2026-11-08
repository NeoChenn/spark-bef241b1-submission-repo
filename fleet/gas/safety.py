from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class SafetyLimits:
    gas_stop_fraction_lel: float = 0.20
    person_min_distance_m: float = 3.0
    min_detection_confidence: float = 0.70
    gas_approach_fraction_lel: float = 0.10


class Verdict(str, Enum):
    ALLOW = "allow"
    HOLD_FOR_APPROVAL = "hold_for_approval"
    PAUSE = "pause"
    STOP_AND_RETREAT = "stop_and_retreat"
    FLAG_HUMAN = "flag_human"


@dataclass(frozen=True)
class SafetyInputs:
    gas_fraction_lel: Optional[float]
    nearest_person_m: Optional[float]
    detection_confidence: float
    action_is_actuation: bool
    permit_approved: bool


@dataclass(frozen=True)
class SafetyDecision:
    verdict: Verdict
    reason: str


def check(inputs: SafetyInputs, limits: SafetyLimits = SafetyLimits()) -> SafetyDecision:
    gas = inputs.gas_fraction_lel
    if gas is None or (isinstance(gas, float) and math.isnan(gas)):
        return SafetyDecision(Verdict.STOP_AND_RETREAT, "gas sensor reading invalid (fail safe)")
    if gas >= limits.gas_stop_fraction_lel:
        return SafetyDecision(Verdict.STOP_AND_RETREAT,
                              f"gas {gas:.2f} of LEL at or above limit {limits.gas_stop_fraction_lel:.2f}")
    person = inputs.nearest_person_m
    if person is not None and person < limits.person_min_distance_m:
        return SafetyDecision(Verdict.PAUSE,
                              f"person at {person:.1f} m, closer than {limits.person_min_distance_m:.1f} m")
    if inputs.detection_confidence < limits.min_detection_confidence:
        return SafetyDecision(Verdict.FLAG_HUMAN, f"confidence {inputs.detection_confidence:.2f} below "
                                                  f"{limits.min_detection_confidence:.2f}")
    if inputs.action_is_actuation and not inputs.permit_approved:
        return SafetyDecision(Verdict.HOLD_FOR_APPROVAL, "actuation needs an approved permit")
    return SafetyDecision(Verdict.ALLOW, "all checks passed")


def nearest_person_m(people, xy) -> Optional[float]:
    if not people:
        return None
    return min(math.hypot(p[0] - xy[0], p[1] - xy[1]) for p in people)
