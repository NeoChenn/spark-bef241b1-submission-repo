"""Compare the rule-based baseline with a trained PPO policy on the SAME held-out scenarios.

Both are scored by `scenarios.score_outcome` (against ground truth) minus the same time cost.
Run:

    python -m flange_robot.evaluate                                   # baseline only
    python -m flange_robot.evaluate --model checkpoints/ppo_repair/best.zip --n 300

The scenario set is generated from a fixed seed that training never uses, so the result is a
held-out comparison. It is still SIMULATED: the diagnosis input is a simulated feature and the
scenarios are our own design. A good number here does not show real-world performance.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from .env import FlangeEnv
from .episode import run_episode
from .policy import Phase, RepairPolicy
from .scenarios import STEP_COST, TOLERANCE_RAD, ScenarioSpec, apply_scenario, outcome_of_alert, sample_scenario, score_outcome, truth_of

EVAL_SEED = 99991  # training never uses this seed
SUCCESS_SCORE = 8.0  # a terminal score at or above this counts as handled correctly
LIMIT = 0.2


def heldout_specs(n: int, seed: int = EVAL_SEED) -> list[ScenarioSpec]:
    rng = np.random.default_rng(seed)
    return [sample_scenario(rng) for _ in range(n)]


def run_baseline(spec: ScenarioSpec) -> dict:
    env = FlangeEnv(seed=spec.env_seed)
    apply_scenario(env, spec)
    policy = RepairPolicy()
    r = run_episode(env, policy, max_steps=400)
    finished = policy.phase in (Phase.DONE, Phase.ESCALATED)
    outcome = outcome_of_alert(policy.alerts[0].kind if policy.alerts else None, finished)
    unneeded = sum(1 for x in r.retorques if x["angle_true"] <= TOLERANCE_RAD)
    score = score_outcome(spec.kind, outcome, truth_of(env, unneeded, len(r.retorques)))
    return {"kind": spec.kind, "outcome": outcome, "score": score, "steps": r.decision_step,
            "ret": score - STEP_COST * r.decision_step,
            "unsafe": any(x["gas_true"] >= LIMIT for x in r.retorques)}


def run_ppo(model, spec: ScenarioSpec) -> dict:
    from .rl_env import RepairSkillEnv

    env = RepairSkillEnv()
    obs, _ = env.reset(options={"spec": spec})
    total = 0.0
    for _ in range(500):
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, r, term, trunc, info = env.step(int(a))
        total += r
        if term or trunc:
            break
    return {"kind": spec.kind, "outcome": env.outcome or "timeout", "score": env.terminal_score,
            "steps": env.sim_steps, "ret": total, "unsafe": env.unsafe_tighten}


def summarise(rows: list[dict]) -> dict:
    out = {"n": len(rows), "mean_return": float(np.mean([r["ret"] for r in rows])),
           "success_rate": float(np.mean([r["score"] >= SUCCESS_SCORE for r in rows])),
           "mean_steps": float(np.mean([r["steps"] for r in rows])),
           "false_all_clears": int(sum(r["score"] <= -15 for r in rows)),
           "unsafe_tightenings": int(sum(r["unsafe"] for r in rows)),
           "timeouts": int(sum(r["outcome"] == "timeout" for r in rows)), "by_kind": {}}
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    for k, rs in sorted(by.items()):
        out["by_kind"][k] = {"n": len(rs), "success_rate": float(np.mean([r["score"] >= SUCCESS_SCORE for r in rs])),
                             "mean_steps": float(np.mean([r["steps"] for r in rs])),
                             "mean_return": float(np.mean([r["ret"] for r in rs]))}
    return out


def print_table(name: str, s: dict) -> None:
    print(f"\n{name}: n={s['n']}  mean return {s['mean_return']:+.2f}  success {s['success_rate']:.1%}  "
          f"mean steps {s['mean_steps']:.1f}  false all-clears {s['false_all_clears']}  "
          f"unsafe tightenings {s['unsafe_tightenings']}  timeouts {s['timeouts']}")
    for k, v in s["by_kind"].items():
        print(f"    {k:6s} n={v['n']:3d}  success {v['success_rate']:6.1%}  steps {v['mean_steps']:6.1f}  return {v['mean_return']:+6.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="path to a MaskablePPO .zip")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    ap.add_argument("--tag", default="eval")
    a = ap.parse_args()

    specs = heldout_specs(a.n, a.seed)
    results = {"baseline": summarise([run_baseline(s) for s in specs])}
    print_table("rule-based baseline", results["baseline"])
    if a.model:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(a.model, device="cpu")
        results["ppo"] = summarise([run_ppo(model, s) for s in specs])
        print_table("PPO", results["ppo"])
    os.makedirs("runs", exist_ok=True)
    path = f"runs/{a.tag}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
