"""Skill-level Gymnasium environment for training the repair DECISION policy with PPO.

What is learned here, and what is not:
- LEARNED: the timing of decisions. When to keep observing, when to commit to a diagnosis, when
  to judge a repair, when to escalate. These trade accuracy against time and are where fixed
  thresholds are brittle (for example, a dusty camera needs more readings).
- NOT LEARNED: safety, diagnosis, motion. The gas interlock stops the episode on its own, the
  diagnoser is the rule-based `Diagnoser`, and actions that are not allowed are masked out. A
  policy can therefore never tighten in gas at the limit, never tighten before diagnosing a loose
  bolt, and never tighten a bolt it cannot see confidently.

Actions (discrete, with a validity mask; use MaskablePPO):
  0 OBSERVE   hold still for one step and take another reading
  1 APPROACH  walk to the flange (a macro action: many simulation steps)
  2 CONCLUDE  commit to a diagnosis from the evidence so far (once per episode)
  3 RETORQUE  re-tighten the worst queued bolt
  4 VERIFY    judge the last repair from the readings since (aligned? gas cleared?)
  5 ESCALATE  hand over to a human now

Time costs `STEP_COST` per simulation step, so waiting longer than the evidence needs is penalised.
The terminal reward comes from `scenarios.score_outcome`, judged against ground truth.
There is no approval step: the robot re-tightens on its own (see docs/gas-leak-detection.md).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import SafetyLimits
from .diagnosis import Diagnoser, FaultKind
from .env import FLANGE_XY, REACH_M, Action, FlangeEnv
from .policy import PolicyConfig
from .safety import SafetyInputs, Verdict, check
from .scenarios import STEP_COST, TOLERANCE_RAD, apply_scenario, sample_scenario, score_outcome, truth_of

OBSERVE, APPROACH, CONCLUDE, RETORQUE, VERIFY, ESCALATE = range(6)
ACTION_NAMES = ["observe", "approach", "conclude", "retorque", "verify", "escalate"]
OBS_DIM = 22
_DIAG_ORDER = [FaultKind.LOOSE_BOLT, FaultKind.CRACK_OR_BURST, FaultKind.UNCERTAIN, FaultKind.NO_FAULT]


class RepairSkillEnv(gym.Env):
    metadata: dict = {}

    def __init__(self, seed: int = 0, max_sim_steps: int = 300):
        super().__init__()
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(-10.0, 10.0, (OBS_DIM,), np.float32)
        self.limits = SafetyLimits()
        self.cfg = PolicyConfig()
        self.max_sim_steps = max_sim_steps
        self._rng = np.random.default_rng(seed)
        self.sim: FlangeEnv | None = None
        self.outcome: str | None = None

    # -- episode setup -------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        spec = (options or {}).get("spec") or sample_scenario(self._rng)
        self.spec = spec
        self.sim = FlangeEnv(seed=spec.env_seed)
        apply_scenario(self.sim, spec)
        self.obs = self.sim.observe()
        self.est: dict[int, dict] = {}
        self.diagnoser = Diagnoser(self.limits, self.cfg.tolerance_rad, gas_clear=self.cfg.gas_clear)
        self.arrived = False
        self.n_read = 0
        self.diag = None
        self.queue: list[int] = []
        self.awaiting = False
        self.verify_reads = 0
        self.target: int | None = None
        self.retries: dict[int, int] = {}
        self.suspect_done = False
        self.gas_at_repair = 0.0
        self.sim_steps = 0
        self.prev_gas = self.obs["gas_fraction_lel"]
        self.repaired = 0
        self.tight_total = 0
        self.tight_unneeded = 0
        self.unsafe_tighten = False
        self.outcome = None
        self.terminal_score = 0.0
        self._terminated = False
        self._ingest(self.obs)
        return self._vec(), {"spec": spec}

    # -- bookkeeping ---------------------------------------------------------
    @staticmethod
    def _angle(e: dict) -> float:
        return e["wa"] / e["w"] if e["w"] > 0 else 0.0

    def _dist(self) -> float:
        return float(np.linalg.norm(np.array(self.obs["robot_xy"]) - np.array(FLANGE_XY)))

    def _ingest(self, obs: dict) -> None:
        for b in obs["bolts"]:
            w = b["confidence"] ** 2
            if w <= 0:
                continue
            e = self.est.setdefault(b["bolt"], {"w": 0.0, "wa": 0.0, "conf": 0.0, "dist": float("inf")})
            e["w"] += w
            e["wa"] += w * b["stripe_misalignment_rad"]
            e["conf"] = b["confidence"]
            e["dist"] = b["distance_m"]
        self.diagnoser.update(obs)
        if self.arrived:
            self.n_read += 1
        if self.awaiting:
            self.verify_reads += 1

    def _sim(self, action: Action, tightening: bool = False, conf: float = 1.0) -> bool:
        """One simulation step, behind the safety layer. False means the episode ended (gas stop)."""
        gas = self.obs["gas_fraction_lel"]
        verdict = check(SafetyInputs(gas, None, conf, tightening, True), self.limits).verdict
        if verdict == Verdict.STOP_AND_RETREAT:
            self._terminal("escalated_gas")
            return False
        self.prev_gas = gas
        self.obs = self.sim.step(action)
        self.sim_steps += 1
        self._ingest(self.obs)
        if self.obs["gas_fraction_lel"] >= self.limits.gas_stop_fraction_lel:
            self._terminal("escalated_gas")  # the supervisor stops and retreats the robot
            return False
        return True

    def _terminal(self, outcome: str) -> None:
        if self._terminated:
            return
        truth = truth_of(self.sim, self.tight_unneeded, self.tight_total)
        self.terminal_score = score_outcome(self.spec.kind, outcome, truth)
        self.outcome = outcome
        self._terminated = True

    # -- masks and observation ----------------------------------------------
    def action_masks(self) -> np.ndarray:
        m = np.zeros(len(ACTION_NAMES), dtype=bool)
        m[OBSERVE] = True
        m[ESCALATE] = True
        m[APPROACH] = not self.arrived
        m[CONCLUDE] = self.arrived and self.diag is None and self.n_read >= 1
        loose = self.diag is not None and self.diag.kind == FaultKind.LOOSE_BOLT
        if loose and self.queue and not self.awaiting:
            e = self.est.get(self.queue[0])
            m[RETORQUE] = (
                e is not None
                and e["conf"] >= self.limits.min_detection_confidence
                and e["dist"] <= REACH_M - self.cfg.reach_margin_m
                and self.obs["gas_fraction_lel"] < self.limits.gas_stop_fraction_lel
            )
        m[VERIFY] = self.awaiting and self.verify_reads >= 1
        return m

    def _vec(self) -> np.ndarray:
        o = self.obs
        gas = o["gas_fraction_lel"]
        cam = o["camera"]
        confident = [abs(self._angle(e)) for e in self.est.values() if e["conf"] >= self.limits.min_detection_confidence]
        seen = [b["confidence"] for b in o["bolts"]]
        diag_hot = [0.0] * 4
        if self.diag is not None:
            diag_hot[_DIAG_ORDER.index(self.diag.kind)] = 1.0
        target_angle = 0.0
        if self.target is not None and self.target in self.est:
            target_angle = abs(self._angle(self.est[self.target]))
        v = [
            gas / 0.2,
            (gas - self.prev_gas) / 0.05,
            float(self.arrived),
            self._dist() / 5.0,
            self.n_read / 20.0,
            self.diagnoser.crack_score,
            cam["crack_confidence"],
            sum(a > self.cfg.tolerance_rad for a in confident) / 3.0,
            max(confident, default=0.0) / 0.3,
            float(np.mean(seen)) if seen else 0.0,
            float(self.diag is not None),
            *diag_hot,
            len(self.queue) / 3.0,
            self.repaired / 3.0,
            float(self.awaiting),
            self.verify_reads / 20.0,
            target_angle / 0.2,
            (gas - self.cfg.gas_clear) / 0.1,
            self.sim_steps / self.max_sim_steps,
        ]
        return np.asarray(v, dtype=np.float32)

    # -- dynamics ------------------------------------------------------------
    def step(self, action: int):
        action = int(action)
        self._reward = 0.0
        steps_before = self.sim_steps
        if not self.action_masks()[action]:
            self._reward -= 1.0  # an invalid choice: charge it and treat it as "observe"
            action = OBSERVE

        if action == OBSERVE:
            self._sim(Action("wait"))
        elif action == APPROACH:
            for _ in range(150):
                if self._dist() <= self.cfg.arrive_tol_m:
                    break
                if not self._sim(Action("move_xy", xy=FLANGE_XY)):
                    break
            if not self._terminated:
                self.arrived = self._dist() <= self.cfg.arrive_tol_m
        elif action == CONCLUDE:
            self._conclude()
        elif action == RETORQUE:
            self._retorque()
        elif action == VERIFY:
            self._verify()
        else:
            self._terminal("escalated_other")

        if not self._terminated and self.sim_steps >= self.max_sim_steps:
            self._terminal("timeout")
        if self._terminated:
            self._reward += self.terminal_score
        reward = self._reward - STEP_COST * (self.sim_steps - steps_before)
        timed_out = self.outcome == "timeout"
        info = {"outcome": self.outcome, "kind": self.spec.kind, "sim_steps": self.sim_steps,
                "terminal_score": self.terminal_score if self._terminated else None,
                "unsafe_tighten": self.unsafe_tighten}
        return self._vec(), float(reward), self._terminated and not timed_out, timed_out, info

    def _conclude(self) -> None:
        d = self.diagnoser.diagnose(self.obs["gas_fraction_lel"], self.est)
        self.diag = d
        if d.kind == FaultKind.CRACK_OR_BURST:
            self._terminal("escalated_crack")
        elif d.kind == FaultKind.LOOSE_BOLT:
            self.queue = [i for i, _ in d.loose_bolts]
        elif d.kind == FaultKind.UNCERTAIN:
            self._terminal("escalated_other")
        else:
            self._terminal("done_quiet")

    def _retorque(self) -> None:
        bolt = self.queue[0]
        e = self.est[bolt]
        gas = self.obs["gas_fraction_lel"]
        if check(SafetyInputs(gas, None, e["conf"], True, True), self.limits).verdict == Verdict.STOP_AND_RETREAT:
            self._terminal("escalated_gas")
            return
        if self.sim.gas_fraction_lel() >= self.limits.gas_stop_fraction_lel:
            self.unsafe_tighten = True
        self.tight_total += 1
        self.repaired += 1
        if abs(self.sim.true_bolt_angle(bolt)) <= TOLERANCE_RAD:
            self.tight_unneeded += 1
        self.est.pop(bolt)  # discard old readings so verification uses fresh ones
        self.target, self.awaiting, self.verify_reads, self.gas_at_repair = bolt, True, 0, gas
        self._sim(Action("retorque", bolt=bolt), tightening=True, conf=e["conf"])

    def _verify(self) -> None:
        cfg = self.cfg
        gas = self.obs["gas_fraction_lel"]
        bolt = self.target
        e = self.est.get(bolt)
        self.awaiting = False
        if e is None or e["conf"] < self.limits.min_detection_confidence:
            self._terminal("escalated_other")
        elif abs(self._angle(e)) > cfg.tolerance_rad:
            self.retries[bolt] = self.retries.get(bolt, 0) + 1
            if self.retries[bolt] > cfg.retry_limit:
                self._terminal("escalated_other")
            # otherwise the same bolt stays at the front of the queue and can be tightened again
        else:
            self.queue.pop(0)
            if gas > self.gas_at_repair + cfg.gas_rise_margin:
                self._terminal("escalated_other")
            elif self.queue:
                pass
            elif gas <= cfg.gas_clear:
                self._terminal("done_repaired")
            elif not self.suspect_done and self._queue_suspects():
                self.suspect_done = True
            else:
                self._terminal("escalated_other")

    def _queue_suspects(self) -> bool:
        cfg = self.cfg
        suspects = [
            (abs(self._angle(e)), i) for i, e in self.est.items()
            if e["conf"] >= self.limits.min_detection_confidence and abs(self._angle(e)) > cfg.suspect_tolerance_rad
        ]
        self.queue = [i for _, i in sorted(suspects, reverse=True)[: cfg.max_suspects]]
        return bool(self.queue)
