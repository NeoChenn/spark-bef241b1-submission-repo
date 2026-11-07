"""Run one episode of a policy in the environment and report what happened.

The harness is the only place that may look at ground truth (true bolt angles, true gas).
It uses it to EVALUATE the policy, never to drive it. `hook(t, env)` is called before each
step and is where a live scene change is injected, such as a leak starting mid-run.
"""

from dataclasses import dataclass, field
from typing import Callable

from .env import N_BOLTS, FlangeEnv
from .policy import Phase, RepairPolicy


@dataclass
class EpisodeResult:
    outcome: str
    reason: str
    steps: int
    retorques: list[dict] = field(default_factory=list)  # {"t", "bolt", "gas_true"} at the moment of action
    safety_stops: int = 0
    final_angles: list[float] = field(default_factory=list)  # ground truth, evaluation only
    final_xy: list[float] = field(default_factory=list)
    gas_trace: list[float] = field(default_factory=list)  # true gas at the robot, evaluation only


def run_episode(
    env: FlangeEnv,
    policy: RepairPolicy,
    max_steps: int = 400,
    hook: Callable[[int, FlangeEnv], None] | None = None,
    settle_steps: int = 40,
) -> EpisodeResult:
    obs = env.observe()
    result = EpisodeResult(outcome="timeout", reason="", steps=0)
    settled = 0
    for t in range(max_steps):
        if hook:
            hook(t, env)
        action, info = policy.act(obs)
        gas_true = env.gas_fraction_lel()
        result.gas_trace.append(gas_true)
        if info["verdict"] == "stop_and_retreat":
            result.safety_stops += 1
        if action.kind == "retorque":
            result.retorques.append({"t": t, "bolt": action.bolt, "gas_true": gas_true})
        obs = env.step(action)
        result.steps = t + 1
        if policy.phase == Phase.DONE:
            break
        if policy.phase == Phase.ESCALATED:
            settled += 1
            if settled >= settle_steps:
                break

    result.outcome = policy.phase.value if policy.phase in (Phase.DONE, Phase.ESCALATED) else "timeout"
    result.reason = policy.reason
    result.final_angles = [env.true_bolt_angle(i) for i in range(N_BOLTS)]
    result.final_xy = env.robot_xy.tolist()
    return result
