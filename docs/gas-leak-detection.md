# Gas leak detection and response (branch: `gas-leak-detection`)

## The flow

```
camera + stripe readings + gas  ->  DIAGNOSE  ->  loose bolt?     -> re-torque (no approval) -> verify -> report
                                                  crack or burst?  -> back off, no repair      -> ALERT supervisor
                                                  unsure?          -> back off if gas is up    -> ALERT supervisor
```

| Step | Code | State |
|---|---|---|
| Diagnose | `flange_robot/diagnosis.py` | Rule-based baseline over **simulated** camera features. Not an image model. |
| Decide and act | `flange_robot/policy.py` | Rule-based baseline. PPO is **not trained yet**. |
| Alert the supervisor | `flange_robot/alerts.py` | Structured alert plus a template text. Gemini may rewrite it, with a template fallback. |
| Safety | `flange_robot/safety.py` | Deterministic. Overrides everything. |

## Decision: no duct tape on cracks

We considered training the policy to apply duct tape to a crack as the "fix". **Recommendation: don't.**

1. **It isn't a credible repair.** A crack or burst in a line carrying gas is a loss of containment. Taping it is not a recognised fix, and an informed judge will say so. I am not a site-safety authority: confirm with an engineer or a source before making any claim either way.
2. **Wrong place for a robot to improvise.** It would mean working next to a live leak with a fix that nobody has validated.
3. **PPO would learn our simulator's assumption, not reality.** If the sim marks a crack as "fixed" once tape is applied, the policy learns to apply tape because we defined the reward that way. That is reward hacking on our own model.
4. **It breaks the honest scope.** The problem statement says the robot does not handle loss of containment. A taping robot contradicts that.

**What the robot does instead for a crack or burst:** back off, keep a record, and alert the supervisor with evidence (location, gas reading, crack-likeness score, camera confidence), requiring acknowledgement.

**If the team wants a visible manipulation in the crack branch,** use a low-risk, reversible, genuinely useful action such as placing a warning tag or a gas-sensor beacon at the flange. Do not present it as a repair. This is not built.

## Decision: no human approval before re-tightening

The team decided the robot re-tightens loose bolts **on its own**. The robot re-tightens on its own, with no human approval step, so it is only allowed to act on a diagnosed loose bolt, never in gas at the limit, and it checks its own work afterwards. The supervisor is told about every repair.

What this changes and what it does not:
- The approval step is optional and **off by default** (`PolicyConfig.require_permit=False`). Turning it on restores the old behaviour.
- The gas interlock, the diagnosis check and the verification step still apply. Autonomy raises the stakes on diagnosis quality: a wrong "loose bolt" call now leads straight to action.
- The diagnosis is simulated and easy by construction, so do not claim real-world reliability for it. This is the main open risk of the decision.
- The DeepMind talk advised keeping humans in control for hardware actions, with an approval gate. We are choosing otherwise, so the pitch should explain why (reversible, low-risk action; strict preconditions; supervisor alerted) and be honest about the trade-off.

## What PPO should (and should not) learn

- **Not the diagnosis.** Telling a crack from a loose bolt is a classifier trained on labelled images (supervised learning).
- **Not safety.** Gas limits and "never repair a crack" are rules and action masks, not learned.
- **The decision layer.** Given the diagnosis and its confidence, the gas level and trend, and distance to the flange, choose among: approach, gather more evidence, tighten bolt *k*, back off, or alert. The value of learning is deciding *when to gather more evidence versus act versus escalate* when perception is noisy or degraded, where fixed thresholds are brittle.
- **Proposed reward (to tune):** a positive for a verified repair (stripe aligned and gas fell); a penalty for escalating when a repair was feasible; a penalty for a late or missing alert; a large penalty for any unsafe attempt (also blocked by masking, so it should never fire).
- **Baseline to beat:** the current rule-based policy. Compare repair success within the repairable range, steps to resolution, and unnecessary escalations. If PPO does not beat the baseline, say so in the submission.

## Training results: PPO v1

200k steps of MaskablePPO on CPU (4 parallel environments, about 23 minutes). The best checkpoint by
validation return is compared with the rule-based baseline on the **same 300 held-out scenarios**, scored by
the same rule (`python -m flange_robot.evaluate --model checkpoints/ppo_repair/best.zip`).

| | Rule-based baseline | PPO v1 |
|---|---|---|
| Mean return | +2.94 | **+3.50** |
| Handled correctly | 80.3% | 78.7% |
| Mean steps to a decision | 92.3 | **77.5** (16% fewer) |
| False all-clears | 0 | 0 |
| Unsafe tightenings | 0 | 0 |
| Loose bolt: correct / steps | 70.0% / 107 | 70.7% / 84 |
| Crack: correct | 91.5% | 88.1% |
| No fault: correct | 76.9% | 69.2% |
| Big leak: correct | 100% | 100% |

**What this shows:** PPO learned to decide about 16% faster at roughly the same accuracy. The difference in
"handled correctly" (80.3% vs 78.7%) is within noise for 300 scenarios (about ±2.3 points). It is slightly worse
on no-fault and crack cases and much quicker on loose bolts. The higher return comes from saved time.

**What it does not show:** zero false all-clears and zero unsafe tightenings are guaranteed by the action
masks and the safety layer for both policies; PPO did not learn them. All of this is in our own simulation
with simulated perception.

**Training was not smooth:** at 61k steps the policy collapsed (every validation episode timed out), then
recovered by 82k. The best checkpoint is kept, so the result is unaffected, but a second seed should be run
before quoting these numbers as stable.

**Why success stays near 80%:** almost all failures happen when dust is above about 0.23, where stripe
confidence falls under the 0.7 gate however long the robot waits. More training cannot fix that. The next
step is a perception change: let the robot move closer to a bolt, or clean the lens, when the camera is dusty.

## What the tests show (and don't)

- Crack, loose bolt, no fault, degraded camera, burst (stops on gas) and a crack plus a loose bolt are all handled as designed.
- In 20 random trials each, every crack was diagnosed as a crack, every loose bolt as a loose bolt, and there were no false crack diagnoses.
- **That result is easy by construction.** The simulated crack score is well separated from noise. It says the logic works, not that real vision would.
- **Not modelled:** look-alikes such as stains, pipe markings and shadows; partial occlusion; borderline confidence; a crack and a loose bolt close together.

## Known limits

- Camera input is simulated features. MuJoCo cannot render offscreen in the build container (no EGL), so real images need Isaac Sim on Modal.
- The safety layer checks the instantaneous gas reading only. It has no rate-of-rise rule, so a quickly growing leak could be approached.
- Gas, crack and noise models are illustrative.

## Next steps

1. Decide whether to add a warning-tag or beacon action for the crack branch.
2. Build the PPO environment (Gymnasium wrapper, action masking, reward above) and train against the rule-based baseline.
3. Render labelled images (loose bolt, crack, burst, look-alikes) in Isaac Sim and train a real classifier to replace the rule-based diagnoser.
4. Add a gas rate-of-rise rule to the safety layer.
