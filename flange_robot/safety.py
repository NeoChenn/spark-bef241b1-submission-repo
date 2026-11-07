"""Deterministic safety supervisor.

This layer is rules, not learning. It sits between the policy and the actuators and
overrides the policy. Precedence, highest first:

1. Gas reading at or above the limit, or an invalid reading -> STOP_AND_RETREAT
2. A person closer than the minimum distance                -> PAUSE
3. Detection confidence below the minimum                    -> FLAG_HUMAN
4. Tightening without an approved permit                     -> HOLD_FOR_APPROVAL
5. Otherwise                                                 -> ALLOW

An invalid or missing gas reading is treated as unsafe (fail safe).
"""

import math
from dataclasses import dataclass
from enum import Enum

from .config import SafetyLimits


class Verdict(str, Enum):
    ALLOW = "allow"
    HOLD_FOR_APPROVAL = "hold_for_approval"
    PAUSE = "pause"
    STOP_AND_RETREAT = "stop_and_retreat"
    FLAG_HUMAN = "flag_human"


@dataclass(frozen=True)
class SafetyInputs:
    gas_fraction_lel: float | None  # None or NaN means the sensor failed
    nearest_person_m: float | None  # None means no person detected
    detection_confidence: float
    action_is_tightening: bool
    permit_approved: bool


@dataclass(frozen=True)
class SafetyDecision:
    verdict: Verdict
    reason: str


def check(inputs: SafetyInputs, limits: SafetyLimits = SafetyLimits()) -> SafetyDecision:
    """Return the safety verdict for a proposed action. Pure function, no side effects."""
    gas = inputs.gas_fraction_lel
    if gas is None or (isinstance(gas, float) and math.isnan(gas)):
        return SafetyDecision(Verdict.STOP_AND_RETREAT, "gas sensor reading invalid (fail safe)")
    if gas >= limits.gas_stop_fraction_lel:
        return SafetyDecision(
            Verdict.STOP_AND_RETREAT,
            f"gas {gas:.2f} of LEL at or above limit {limits.gas_stop_fraction_lel:.2f}",
        )

    person = inputs.nearest_person_m
    if person is not None and person < limits.person_min_distance_m:
        return SafetyDecision(
            Verdict.PAUSE,
            f"person at {person:.1f} m, closer than {limits.person_min_distance_m:.1f} m",
        )

    if inputs.detection_confidence < limits.min_detection_confidence:
        return SafetyDecision(
            Verdict.FLAG_HUMAN,
            f"detection confidence {inputs.detection_confidence:.2f} "
            f"below {limits.min_detection_confidence:.2f}",
        )

    if inputs.action_is_tightening and not inputs.permit_approved:
        return SafetyDecision(Verdict.HOLD_FOR_APPROVAL, "tightening needs an approved permit to work")

    return SafetyDecision(Verdict.ALLOW, "all checks passed")
