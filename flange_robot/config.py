"""Central configuration.

Everything configurable lives here: model names, safety limits, usage guards.
Do not scatter these values through the code.

All numeric limits below are ILLUSTRATIVE placeholders. Replace them with values
from a cited source before they appear in the technical summary.
"""

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# Model IDs come from the environment so switching after a 503 or a new release
# is one change. Left blank by default: look up the current IDs in AI Studio.
REASONING_MODEL = os.getenv("GEMINI_REASONING_MODEL", "")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "")

# Usage guard: the organisers disqualify abuse, so cap calls per minute.
GEMINI_MAX_CALLS_PER_MINUTE = _int_env("GEMINI_MAX_CALLS_PER_MINUTE", 20)

# Where run logs go (gitignored).
LOG_DIR = os.getenv("FLANGE_LOG_DIR", "logs")


@dataclass(frozen=True)
class SafetyLimits:
    """Deterministic limits for the safety supervisor. ILLUSTRATIVE, to be sourced."""

    gas_stop_fraction_lel: float = 0.20  # stop and retreat at or above this fraction of LEL
    person_min_distance_m: float = 3.0  # pause if a person is closer than this
    min_detection_confidence: float = 0.70  # below this, flag a human instead of acting


@dataclass(frozen=True)
class EdgeBudget:
    """Where each model runs. Fill in MEASURED values for the technical summary."""

    detector_hardware: str = "TODO"
    detector_latency_ms: float | None = None
    policy_hardware: str = "TODO"
    policy_latency_ms: float | None = None
    reasoning_runs: str = "remote (Gemini), off the control loop; local fallback = flag a human"
