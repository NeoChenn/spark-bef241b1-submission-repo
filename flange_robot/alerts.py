"""Supervisor alerts.

Every episode ends with an alert to the supervisor: a quiet "all clear, repaired" note, or a
request for a human when the robot has stopped. The alert is structured data first (so the
console can render it) and text second.

The text comes from a deterministic template. Gemini may rewrite it for readability on the
slow path, but if Gemini is unavailable the template is used. An alert must never depend on
a remote service.
"""

from dataclasses import asdict, dataclass, field
from typing import Callable

from .gemini_client import GeminiReasoner, GeminiUnavailable


@dataclass(frozen=True)
class Alert:
    severity: str          # "info", "warning" or "critical"
    kind: str              # short code: "repaired", "no_fault", "crack_or_burst", "gas_limit", ...
    message: str           # one-line headline
    recommended_action: str
    requires_ack: bool     # a human must acknowledge
    location_xy: tuple[float, float]
    gas_fraction_lel: float
    step: int
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def format_alert(a: Alert) -> str:
    """Deterministic plain-text alert. Always available."""
    ack = " ACKNOWLEDGEMENT REQUIRED." if a.requires_ack else ""
    return (
        f"[{a.severity.upper()}] {a.message}{ack}\n"
        f"Location: ({a.location_xy[0]:.1f}, {a.location_xy[1]:.1f}). "
        f"Gas: {a.gas_fraction_lel:.2f} of LEL. Step {a.step}.\n"
        f"Recommended: {a.recommended_action}\n"
        f"Evidence: {a.evidence}"
    )


def alert_text(a: Alert, reasoner: GeminiReasoner | None = None) -> str:
    """Readable alert. Uses Gemini if available, otherwise the template."""
    base = format_alert(a)
    if reasoner is None:
        return base
    prompt = (
        "Rewrite this robot alert for a busy offshore supervisor in 3 short lines. "
        "Keep every number and the recommended action exactly as given. Do not add advice.\n\n" + base
    )
    try:
        return reasoner.ask(prompt)
    except GeminiUnavailable:
        return base


Notifier = Callable[[Alert], None]
