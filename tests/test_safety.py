import math

from flange_robot.config import SafetyLimits
from flange_robot.safety import SafetyInputs, Verdict, check

LIMITS = SafetyLimits(gas_stop_fraction_lel=0.2, person_min_distance_m=3.0, min_detection_confidence=0.7)


def inputs(**overrides):
    base = dict(gas_fraction_lel=0.02, nearest_person_m=None, detection_confidence=0.9,
                action_is_tightening=False, permit_approved=False)
    base.update(overrides)
    return SafetyInputs(**base)


def test_allows_when_everything_is_fine():
    assert check(inputs(), LIMITS).verdict == Verdict.ALLOW


def test_gas_at_limit_stops_and_retreats():
    assert check(inputs(gas_fraction_lel=0.2), LIMITS).verdict == Verdict.STOP_AND_RETREAT


def test_gas_just_below_limit_is_allowed():
    assert check(inputs(gas_fraction_lel=0.199), LIMITS).verdict == Verdict.ALLOW


def test_invalid_gas_reading_fails_safe():
    assert check(inputs(gas_fraction_lel=None), LIMITS).verdict == Verdict.STOP_AND_RETREAT
    assert check(inputs(gas_fraction_lel=math.nan), LIMITS).verdict == Verdict.STOP_AND_RETREAT


def test_person_too_close_pauses():
    assert check(inputs(nearest_person_m=2.0), LIMITS).verdict == Verdict.PAUSE


def test_person_far_away_is_ignored():
    assert check(inputs(nearest_person_m=10.0), LIMITS).verdict == Verdict.ALLOW


def test_low_confidence_flags_a_human():
    assert check(inputs(detection_confidence=0.5), LIMITS).verdict == Verdict.FLAG_HUMAN


def test_tightening_without_permit_is_held():
    d = check(inputs(action_is_tightening=True, permit_approved=False), LIMITS)
    assert d.verdict == Verdict.HOLD_FOR_APPROVAL


def test_tightening_with_permit_is_allowed():
    assert check(inputs(action_is_tightening=True, permit_approved=True), LIMITS).verdict == Verdict.ALLOW


def test_gas_overrides_everything_else():
    d = check(inputs(gas_fraction_lel=0.5, nearest_person_m=1.0, detection_confidence=0.1,
                     action_is_tightening=True, permit_approved=True), LIMITS)
    assert d.verdict == Verdict.STOP_AND_RETREAT


def test_person_overrides_permit_and_confidence():
    d = check(inputs(nearest_person_m=1.0, detection_confidence=0.1,
                     action_is_tightening=True, permit_approved=False), LIMITS)
    assert d.verdict == Verdict.PAUSE
