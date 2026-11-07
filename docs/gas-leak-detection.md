# Gas leak detection and response (branch: `gas-leak-detection`)

## The flow

```
camera + stripe readings + gas  ->  DIAGNOSE  ->  loose bolt?     -> permit request -> re-torque -> verify -> report
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
2. **Wrong place for a robot to improvise.** It would mean working next to a live leak, outside what the permit covers.
3. **PPO would learn our simulator's assumption, not reality.** If the sim marks a crack as "fixed" once tape is applied, the policy learns to apply tape because we defined the reward that way. That is reward hacking on our own model.
4. **It breaks the honest scope.** The problem statement says the robot does not handle loss of containment. A taping robot contradicts that.

**What the robot does instead for a crack or burst:** back off, keep a record, and alert the supervisor with evidence (location, gas reading, crack-likeness score, camera confidence), requiring acknowledgement.

**If the team wants a visible manipulation in the crack branch,** use a low-risk, reversible, genuinely useful action such as placing a warning tag or a gas-sensor beacon at the flange. Do not present it as a repair. This is not built.

## What PPO should (and should not) learn

- **Not the diagnosis.** Telling a crack from a loose bolt is a classifier trained on labelled images (supervised learning).
- **Not safety.** Gas limits, the permit gate and "never repair a crack" are rules and action masks, not learned.
- **The decision layer.** Given the diagnosis and its confidence, the gas level and trend, distance to the flange, and permit state, choose among: approach, gather more evidence, request a permit, tighten bolt *k*, back off, or alert. The value of learning is deciding *when to gather more evidence versus act versus escalate* when perception is noisy or degraded, where fixed thresholds are brittle.
- **Proposed reward (to tune):** a positive for a verified repair (stripe aligned and gas fell); a penalty for escalating when a repair was feasible; a penalty for a late or missing alert; a large penalty for any unsafe attempt (also blocked by masking, so it should never fire).
- **Baseline to beat:** the current rule-based policy. Compare repair success within the repairable range, steps to resolution, and unnecessary escalations. If PPO does not beat the baseline, say so in the submission.

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
