"""Gas-leak repair policy: the rule-based BASELINE.

This is a deterministic state machine, not a learned policy. It is the reference behaviour
and the baseline that a learned (PPO) policy has to beat. It is also what the demo can run
while training is still in progress.

Scope, deliberately narrow: the robot repairs EARLY-STAGE leaks, a weeping flange caught
before it becomes a release. Anything larger is never approached: the safety layer stops
the robot at the gas limit and the policy escalates to a human. It never welds, and it
never tightens without an approved permit.

Phases:

    APPROACH -> INSPECT -> REQUEST_PERMIT -> REPAIR -> VERIFY -> DONE
                   \\____________\\__________________\\______-> ESCALATED

- APPROACH        walk to the flange (position comes from the asset register, as on a real site)
- INSPECT         hold still and average noisy torque-stripe readings, weighted by confidence
- REQUEST_PERMIT  ask the human permit-to-work gate to approve, with the evidence
- REPAIR          re-torque the worst confident bolt in reach
- VERIFY          wait for fresh readings: stripe aligned? gas falling? retry once, else escalate.
                  If gas stays elevated after the confident repairs, ask the permit gate once more
                  for the next most suspicious bolts (fusing the gas and stripe evidence).
- DONE            all bolts within tolerance and gas back to background
- ESCALATED       hand over to a human with a reason; retreat if the cause was gas

The policy only uses what a sensor would report (the observation). It never reads ground truth.
The safety supervisor (`safety.py`) is checked on every step and overrides the policy.

All thresholds are ILLUSTRATIVE.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from .config import SafetyLimits
from .decision_log import DecisionLog
from .env import FLANGE_XY, REACH_M, Action
from .safety import SafetyInputs, Verdict, check


class Phase(str, Enum):
    APPROACH = "approach"
    INSPECT = "inspect"
    REQUEST_PERMIT = "request_permit"
    REPAIR = "repair"
    VERIFY = "verify"
    DONE = "done"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class PolicyConfig:
    flange_xy: tuple[float, float] = FLANGE_XY
    arrive_tol_m: float = 0.15
    tolerance_rad: float = 0.06   # a bolt is "loose" if its stripe is off by more than this
    inspect_steps: int = 10       # readings averaged before judging
    verify_steps: int = 15        # readings averaged (and gas settling time) after a repair
    gas_clear: float = 0.05       # gas must be at or below this to call the leak fixed
    gas_rise_margin: float = 0.03 # gas rising by more than this after a repair is a red flag
    retry_limit: int = 1
    reach_margin_m: float = 0.05
    # If gas is still elevated after the confident repairs, the next most suspicious bolts
    # (below `tolerance_rad` but above this) are queued once, behind a fresh permit request.
    suspect_tolerance_rad: float = 0.03
    max_suspects: int = 2


class RepairPolicy:
    def __init__(
        self,
        permit: Callable[[str], bool],
        limits: SafetyLimits = SafetyLimits(),
        config: PolicyConfig = PolicyConfig(),
        log: DecisionLog | None = None,
    ):
        self.permit = permit
        self.limits = limits
        self.config = config
        self.log = log
        self.phase = Phase.APPROACH
        self.reason = ""
        self.retreating = False
        self._est: dict[int, dict] = {}  # bolt -> weighted stripe estimate and latest confidence
        self._counter = 0
        self._queue: list[int] = []
        self._target: int | None = None
        self._retries: dict[int, int] = {}
        self._permit_ok = False
        self._gas_at_repair = 0.0
        self._suspect_pass_done = False

    # -- perception bookkeeping ---------------------------------------------
    def _update(self, obs: dict) -> None:
        for b in obs["bolts"]:
            w = b["confidence"] ** 2
            if w <= 0:
                continue
            e = self._est.setdefault(b["bolt"], {"w": 0.0, "wa": 0.0, "conf": 0.0, "dist": float("inf")})
            e["w"] += w
            e["wa"] += w * b["stripe_misalignment_rad"]
            e["conf"] = b["confidence"]
            e["dist"] = b["distance_m"]

    @staticmethod
    def _angle(e: dict) -> float:
        return e["wa"] / e["w"] if e["w"] > 0 else 0.0

    def _escalate(self, reason: str) -> None:
        self.phase = Phase.ESCALATED
        if not self.reason:
            self.reason = reason

    # -- the decision --------------------------------------------------------
    def act(self, obs: dict) -> tuple[Action, dict]:
        """Return the action to take and an info dict (phase, verdict, reason)."""
        self._update(obs)
        gas = obs["gas_fraction_lel"]
        proposed = self._propose(obs, gas)

        confidence = 1.0
        if proposed.kind == "retorque" and proposed.bolt in self._est:
            confidence = self._est[proposed.bolt]["conf"]
        decision = check(
            SafetyInputs(
                gas_fraction_lel=gas,
                nearest_person_m=None,
                detection_confidence=confidence,
                action_is_tightening=proposed.kind == "retorque",
                permit_approved=self._permit_ok,
            ),
            self.limits,
        )

        final = proposed
        verdict = decision.verdict
        if verdict == Verdict.STOP_AND_RETREAT:
            self._escalate(f"safety: {decision.reason}")
            self.retreating = True
            final = Action("retreat")
        elif verdict == Verdict.PAUSE:
            final = Action("wait")
        elif verdict == Verdict.FLAG_HUMAN:
            self._escalate(f"safety: {decision.reason}")
            final = Action("wait")
        elif verdict == Verdict.HOLD_FOR_APPROVAL:
            self.phase = Phase.REQUEST_PERMIT
            final = Action("wait")
        elif final.kind == "retorque":
            self._commit_retorque(final.bolt, gas)  # only now: the safety layer allowed it

        info = {"phase": self.phase.value, "verdict": verdict.value, "reason": decision.reason}
        if self.log:
            self.log.decision(
                evidence={"phase": self.phase.value, "gas": round(gas, 4),
                          "bolts_seen": len(obs["bolts"]), "queue": list(self._queue)},
                action=final.kind + (f":{final.bolt}" if final.bolt is not None else ""),
                verdict=verdict.value,
                reason=decision.reason,
            )
        return final, info

    def _propose(self, obs: dict, gas: float) -> Action:
        wait = Action("wait")
        cfg = self.config
        pos = np.array(obs["robot_xy"])

        if self.phase == Phase.APPROACH:
            if float(np.linalg.norm(pos - np.array(cfg.flange_xy))) <= cfg.arrive_tol_m:
                self.phase, self._counter = Phase.INSPECT, 0
                return wait
            return Action("move_xy", xy=cfg.flange_xy)

        if self.phase == Phase.INSPECT:
            self._counter += 1
            if self._counter < cfg.inspect_steps:
                return wait
            self._finish_inspection(gas)
            return wait

        if self.phase == Phase.REQUEST_PERMIT:
            if self.permit(self._permit_request(gas)):
                self._permit_ok = True
                self.phase = Phase.REPAIR
            else:
                self._escalate("permit to work denied")
            return wait

        if self.phase == Phase.REPAIR:
            bolt = self._queue[0]
            e = self._est.get(bolt)
            if e is None or e["dist"] > REACH_M - cfg.reach_margin_m:
                self._escalate(f"bolt {bolt} out of reach")
                return wait
            return Action("retorque", bolt=bolt)

        if self.phase == Phase.VERIFY:
            self._counter += 1
            if self._counter >= cfg.verify_steps:
                self._finish_verification(gas)
            return wait

        if self.phase == Phase.ESCALATED and self.retreating:
            return Action("retreat")
        return wait  # DONE, or ESCALATED and holding position

    def _commit_retorque(self, bolt: int, gas: float) -> None:
        self._target = bolt
        self._gas_at_repair = gas
        self._est.pop(bolt, None)  # discard old readings so verification uses fresh ones
        self.phase, self._counter = Phase.VERIFY, 0

    def _finish_inspection(self, gas: float) -> None:
        cfg = self.config
        loose, unreliable = [], []
        for i, e in self._est.items():
            if e["conf"] < self.limits.min_detection_confidence:
                unreliable.append(i)
            elif abs(self._angle(e)) > cfg.tolerance_rad:
                loose.append((abs(self._angle(e)), i))
        if loose:
            self._queue = [i for _, i in sorted(loose, reverse=True)]  # worst first
            self.phase = Phase.REQUEST_PERMIT
        elif unreliable:
            self._escalate("perception too degraded to judge the bolts")
        elif gas > cfg.gas_clear:
            self._escalate("gas elevated but no loose bolt found")
        else:
            self.phase, self.reason = Phase.DONE, "no leak source found; all bolts within tolerance"

    def _finish_verification(self, gas: float) -> None:
        cfg = self.config
        bolt = self._target
        e = self._est.get(bolt)
        if e is None or e["conf"] < self.limits.min_detection_confidence:
            self._escalate(f"cannot verify bolt {bolt}: no confident reading")
        elif abs(self._angle(e)) > cfg.tolerance_rad:
            self._retries[bolt] = self._retries.get(bolt, 0) + 1
            if self._retries[bolt] > cfg.retry_limit:
                self._escalate(f"bolt {bolt} still misaligned after retry")
            else:
                self.phase = Phase.REPAIR  # same bolt is still at the front of the queue
        else:
            self._queue.pop(0)
            if gas > self._gas_at_repair + cfg.gas_rise_margin:
                self._escalate("gas rising after repair")
            elif self._queue:
                self.phase = Phase.REPAIR
            elif gas <= cfg.gas_clear:
                self.phase, self.reason = Phase.DONE, "leak repaired; gas back to background"
            elif not self._suspect_pass_done and self._queue_suspects():
                # Gas is evidence that something is still loose: ask the human again for the next suspects.
                self._suspect_pass_done = True
                self._permit_ok = False
                self.phase = Phase.REQUEST_PERMIT
            else:
                self._escalate("gas still elevated after all bolts repaired")

    def _queue_suspects(self) -> bool:
        """Queue the most suspicious confident bolts still below tolerance. True if any were queued."""
        cfg = self.config
        suspects = [
            (abs(self._angle(e)), i)
            for i, e in self._est.items()
            if e["conf"] >= self.limits.min_detection_confidence and abs(self._angle(e)) > cfg.suspect_tolerance_rad
        ]
        self._queue = [i for _, i in sorted(suspects, reverse=True)[: cfg.max_suspects]]
        return bool(self._queue)

    def _permit_request(self, gas: float) -> str:
        parts = ", ".join(f"bolt {i}: {self._angle(self._est[i]):+.2f} rad" for i in self._queue)
        return f"Re-torque requested. Loose bolts: {parts}. Gas {gas:.2f} of LEL (limit {self.limits.gas_stop_fraction_lel:.2f})."
