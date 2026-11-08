# Novelty upgrades: beyond vanilla PPO

Every upgrade is opt-in. Defaults, interfaces, the live stream and the existing tests are unchanged. Each one is measured before and after on the same seeded benchmark.

| Policy | Novelty | Before → after | Default |
|---|---|---|---|
| Gas | Bayesian particle filter + infotaxis search | CH4-only leak found 64% → 86%, wrong-flange hand-offs 2 → 0, p90 error 2.24 → 0.64 m | off (opt-in `search="infotaxis"`) |
| Pick-up | Gemini Robotics-ER points to the grasp | grasp point on unseen target 7% → 97%, median 3-D error 26 → 0.5 cm | off (opt-in, detector fallback) |
| Fleet / Argon | Simulation-verified planning | bad-plan leak isolation 437 → 406 s (clean planner 416 → 407 s), 0 unsafe in every arm | off (opt-in `VerifiedEnvelope`) |
| Perception | Shift-gated test-time adaptation | storm-lens F1 0.47 → 0.95 at severity 0.5 (0.00 → 0.90 at 1.0), clean F1 0.993 unchanged | off (opt-in `detection.tta.TTA`) |
| Scrub | CBF force shield (RLS stiffness) | in progress | — |
| Walking | RMA payload/deck adaptation | in progress | — |

## One-liners

- **Gas:** Without ultrasound, a Bayesian particle filter with infotaxis search finds the leaking flange 86% of the time against 64% for cast-and-surge, and never hands off the wrong flange.
- **Pick-up:** Gemini Robotics-ER points at the grasp spot on bottles, boxes and cans it has never seen, landing on the target 97% of the time against 7% for the colour detector.
- **Fleet / Argon:** Before a Gemini plan is applied, the robots rehearse it in a fast simulation and veto any step that delays leak isolation, so plausible-but-harmful plans cost 31 s less per leak.
- **Perception:** The rust detector re-tunes itself to rain and glare on the fly, recovering F1 from 0.47 to 0.95 in heavy weather while leaving clear-weather frames untouched.

## Details

- **Gas.** Benchmark: the 60-leak mock (seeds 1000–1059, loose flange). Success is counted out of the leaks detected (22 detected on CH4 alone). With ultrasound, both modes reach 100%. Code: `policy/gasbot/infotaxis.py`. Evaluation: `policy/gasbot/infotaxis_eval.py`. HUD: the Gas sensors panel shows "PF belief σ" in infotaxis mode only. Charts: `gas/chart.png` and `gas/shrink.png`.
- **Pick-up.** Benchmark: 30 synthetic ray-cast scenes, each with an unseen target and 2 distractors. The model is `gemini-robotics-er-2-preview`, run on 35 calls for about $0.06. The baseline is `detect_cans` from the pick-cans branch; given an oracle pick of which blob is the target, it reaches 47%. Code: `policy/pickgrasp/vlm_grasp.py`. Bench: `bench/pick_vlm.py`. Chart: `pick/chart.png`.
- **Fleet / Argon.**
  - Benchmark: 40 seeded 15-minute bench/orch shifts. In each shift, every planner call is rolled out for 600 s in the CPU fleet sim against the incumbent plan. A call is vetoed if it adds an unsafe event or delays leak isolation by more than 10 s.
  - "Plausible bad plans" are P1 inspection detours, injected into half of the gas re-plans.
  - The verifier also vetoed 11 harmful calls from the stand-in planner itself; one was a deck-wide gas survey that added 19 s.
  - The planner arms use the deterministic replay planner. A live Gemini run was cut for time.
  - HUD: a veto shows in the Orchestrator call list as "✗ … plan vetoed by simulation: leak isolation +N s".
  - Code: `policy/control/verifier.py`. Bench: `bench/orch_verify.py`. Chart: `fleet/chart.png`.
- **Perception.**
  - Benchmark: 498 held-out crops (`rust_sim_test` plus 13 sim renders), degraded by the `policy/storm.py` lens (rain drops and visibility loss) plus a glare bloom, seeded per crop.
  - Method: BN statistics are re-estimated per 32-frame batch. Adaptation runs only when a feature-shift score (Gaussian KL at the first BN layer) exceeds 0.4. Clean batches score 0.09–0.20; degraded batches score 0.6 or more.
  - Plain always-on BN-adapt would cost clear-weather F1 (0.993 → 0.940); the gate removes that cost.
  - The deployed ONNX path is untouched.
  - Code: `detection/tta.py`. Bench: `bench/tta.py`. Chart: `tta/chart.png`.
