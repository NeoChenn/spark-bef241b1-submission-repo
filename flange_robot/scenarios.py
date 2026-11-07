"""Randomised leak scenarios and the scoring rule, shared by training and evaluation.

Training and the baseline comparison must see the same scenarios and be scored by the same
rule, otherwise the comparison means nothing. Both import from here.

Scenario kinds (the ground truth the robot is never told):
  none   no fault
  loose  one to three loose bolts, a small leak the robot can safely approach
  crack  a crack at the flange (a quarter of these also have a loose bolt)
  big    a leak strong enough that the gas limit stops the robot before it arrives

Scoring is by ground truth at the end of the episode. A false all-clear (saying "done" while
something is still wrong) is the worst outcome. An unnecessary escalation costs a little. Time
costs a little per simulation step, so the policy is pushed to act without waiting longer than
the evidence needs.

All numbers here are ILLUSTRATIVE design choices, not measurements.
"""

from dataclasses import dataclass

import numpy as np

from .env import N_BOLTS, FlangeEnv

TOLERANCE_RAD = 0.06   # matches PolicyConfig.tolerance_rad
GAS_CLEARED = 0.07     # true gas at or below this counts as fixed
STEP_COST = 0.05       # reward lost per simulation step before the final decision

KIND_WEIGHTS = {"none": 0.15, "loose": 0.45, "crack": 0.20, "big": 0.20}
MAX_REPAIRABLE_TOTAL_RAD = 0.26  # above this much total looseness, gas at the flange reaches the stop limit


@dataclass(frozen=True)
class ScenarioSpec:
    kind: str
    bolts: tuple[tuple[int, float], ...] = ()  # (bolt, radians)
    crack: float = 0.0
    dust: float = 0.0
    env_seed: int = 0


def sample_scenario(rng: np.random.Generator) -> ScenarioSpec:
    kinds = list(KIND_WEIGHTS)
    kind = str(rng.choice(kinds, p=[KIND_WEIGHTS[k] for k in kinds]))
    dust = 0.0 if rng.random() < 0.55 else float(rng.uniform(0.05, 0.6))
    env_seed = int(rng.integers(0, 2**31 - 1))

    def pick(n: int, lo: float, hi: float, cap: float | None = None) -> tuple[tuple[int, float], ...]:
        bolts = rng.choice(N_BOLTS, size=n, replace=False)
        angles = rng.uniform(lo, hi, size=n)
        if cap is not None and angles.sum() > cap:
            angles = np.maximum(angles * cap * rng.uniform(0.85, 1.0) / angles.sum(), 0.065)
        return tuple((int(b), float(a)) for b, a in zip(bolts, angles))

    if kind == "none":
        return ScenarioSpec("none", dust=dust, env_seed=env_seed)
    if kind == "loose":
        return ScenarioSpec("loose", bolts=pick(int(rng.integers(1, 4)), 0.07, 0.14, MAX_REPAIRABLE_TOTAL_RAD),
                            dust=dust, env_seed=env_seed)
    if kind == "crack":
        extra = pick(1, 0.07, 0.12) if rng.random() < 0.25 else ()
        return ScenarioSpec("crack", bolts=extra, crack=float(rng.uniform(0.1, 0.35)), dust=dust, env_seed=env_seed)
    if rng.random() < 0.5:  # big, from bolts
        return ScenarioSpec("big", bolts=pick(3, 0.15, 0.25), dust=dust, env_seed=env_seed)
    return ScenarioSpec("big", crack=float(rng.uniform(0.55, 0.9)), dust=dust, env_seed=env_seed)


def apply_scenario(env: FlangeEnv, spec: ScenarioSpec) -> None:
    """Set up the fault in a freshly built environment."""
    if spec.crack > 0:
        env.trigger_crack(spec.crack)
    for k, (bolt, angle) in enumerate(spec.bolts):
        if k == 0 and spec.crack == 0:
            env.trigger_gas_leak(bolt, angle)
        else:
            env.loosen_bolt(bolt, angle)
    env.set_dust(spec.dust)


def truth_of(env: FlangeEnv, tight_unneeded: int, tight_total: int) -> dict:
    return {
        "max_angle": max(abs(env.true_bolt_angle(i)) for i in range(N_BOLTS)),
        "gas_true": env.gas_fraction_lel(),
        "tight_in_crack": env.crack_severity > 0 and tight_total > 0,
        "tight_unneeded": tight_unneeded,
    }


def score_outcome(kind: str, outcome: str, truth: dict) -> float:
    """Reward for how the episode ended, judged against ground truth. Excludes the time cost."""
    if outcome == "timeout":
        return -10.0
    done = outcome in ("done_quiet", "done_repaired")
    if kind == "none":
        r = 10.0 if outcome == "done_quiet" else -3.0
    elif kind == "loose":
        if outcome == "done_repaired":
            r = 10.0 if truth["max_angle"] <= TOLERANCE_RAD and truth["gas_true"] <= GAS_CLEARED else -15.0
        elif outcome == "done_quiet":
            r = -15.0  # false all-clear: something was loose
        else:
            r = -3.0   # unnecessary escalation
    elif kind == "crack":
        r = 10.0 if outcome == "escalated_crack" else (-20.0 if done else 4.0)
        if truth["tight_in_crack"]:
            r -= 10.0
    else:  # big
        if outcome == "escalated_gas":
            r = 10.0
        elif outcome == "escalated_crack":
            r = 8.0
        else:
            r = -20.0 if done else 4.0
    return r - 0.5 * truth.get("tight_unneeded", 0)


def outcome_of_alert(alert_kind: str | None, finished: bool) -> str:
    """Map a RepairPolicy alert to a scenario outcome label."""
    if not finished or alert_kind is None:
        return "timeout"
    return {"repaired": "done_repaired", "no_fault": "done_quiet", "crack_or_burst": "escalated_crack",
            "gas_limit": "escalated_gas"}.get(alert_kind, "escalated_other")
