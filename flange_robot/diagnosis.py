"""Leak diagnosis: is this a loose bolt, a cracked or burst pipe, or unclear?

This decides WHETHER the robot may attempt a repair at all. Only a loose bolt is a repair the
robot is allowed to attempt (it is reversible and low risk). A crack or burst means loss of
containment, so the robot backs off and alerts a supervisor.

STATUS: a rule-based baseline over SIMULATED camera features (a crack-likeness score with a
confidence) and the torque-stripe readings. It is a stand-in for a real image model. Replace
`Diagnoser.diagnose` with a trained classifier once there are rendered images to train on.

Precedence: a visible crack beats a loose bolt, because the more severe fault wins. A diagnosis
is only made from confident readings; otherwise the answer is UNCERTAIN, never a guess.
"""

from dataclasses import dataclass, field
from enum import Enum

from .config import SafetyLimits


class FaultKind(str, Enum):
    NO_FAULT = "no_fault"
    LOOSE_BOLT = "loose_bolt"
    CRACK_OR_BURST = "crack_or_burst"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class Diagnosis:
    kind: FaultKind
    reason: str
    evidence: dict = field(default_factory=dict)
    loose_bolts: tuple[tuple[int, float], ...] = ()  # (bolt, estimated misalignment), worst first


class Diagnoser:
    def __init__(
        self,
        limits: SafetyLimits = SafetyLimits(),
        tolerance_rad: float = 0.06,
        crack_threshold: float = 0.5,
        gas_clear: float = 0.05,
    ):
        self.limits = limits
        self.tolerance_rad = tolerance_rad
        self.crack_threshold = crack_threshold
        self.gas_clear = gas_clear
        self._w = 0.0
        self._ws = 0.0
        self._conf = 0.0

    def update(self, obs: dict) -> None:
        """Accumulate camera evidence, weighted by confidence squared."""
        cam = obs.get("camera") or {}
        conf = cam.get("crack_confidence", 0.0)
        if conf > 0:
            self._w += conf**2
            self._ws += conf**2 * cam["crack_score"]
            self._conf = conf

    @property
    def crack_score(self) -> float:
        return self._ws / self._w if self._w > 0 else 0.0

    def diagnose(self, gas: float, bolt_estimates: dict[int, dict]) -> Diagnosis:
        """Diagnose from accumulated evidence. `bolt_estimates` maps bolt -> {"w","wa","conf"}."""
        min_conf = self.limits.min_detection_confidence
        evidence = {"crack_score": round(self.crack_score, 3), "camera_confidence": round(self._conf, 3),
                    "gas": round(gas, 3)}

        if self._conf >= min_conf and self.crack_score >= self.crack_threshold:
            return Diagnosis(
                FaultKind.CRACK_OR_BURST,
                f"crack-like feature at the flange (score {self.crack_score:.2f})",
                evidence,
            )

        loose, unreliable = [], []
        for i, e in bolt_estimates.items():
            angle = e["wa"] / e["w"] if e["w"] > 0 else 0.0
            if e["conf"] < min_conf:
                unreliable.append(i)
            elif abs(angle) > self.tolerance_rad:
                loose.append((i, angle))
        if loose:
            loose.sort(key=lambda x: abs(x[1]), reverse=True)
            evidence["loose_bolts"] = {i: round(a, 3) for i, a in loose}
            return Diagnosis(FaultKind.LOOSE_BOLT, f"{len(loose)} loose bolt(s) by torque stripe", evidence, tuple(loose))

        if unreliable:
            return Diagnosis(FaultKind.UNCERTAIN, "perception too degraded to judge the bolts", evidence)
        if gas > self.gas_clear:
            return Diagnosis(FaultKind.UNCERTAIN, "gas elevated but no loose bolt found", evidence)
        return Diagnosis(FaultKind.NO_FAULT, "no leak source found; all bolts within tolerance", evidence)
