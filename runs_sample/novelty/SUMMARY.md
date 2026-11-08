# Novelty upgrades: beyond vanilla PPO

Every upgrade is opt-in. Defaults, interfaces, the live stream and the existing tests are unchanged. Each one is measured before and after on the same seeded benchmark.

| Policy | Novelty | Before → after | Default |
|---|---|---|---|
| Gas | Bayesian particle filter + infotaxis search | CH4-only leak found 64% → 86%, wrong-flange hand-offs 2 → 0, p90 error 2.24 → 0.64 m | off (opt-in `search="infotaxis"`) |
| Pick-up | Gemini Robotics-ER points to the grasp | grasp point on unseen target 7% → 97%, median 3-D error 26 → 0.5 cm | off (opt-in, detector fallback) |
| Fleet / Argon | Simulation-verified planning | bad-plan leak isolation 437 → 406 s (clean planner 416 → 407 s), 0 unsafe in every arm | off (opt-in `VerifiedEnvelope`) |
| Perception | Shift-gated test-time adaptation | storm-lens F1 0.47 → 0.95 at severity 0.5 (0.00 → 0.90 at 1.0), clean F1 0.993 unchanged | off (opt-in `detection.tta.TTA`) |
| Scrub | Control-barrier-function force shield with online RLS stiffness estimate | episodes over 40 N: calm 0.8% → 0.0%, storm 11.5% → 6.6%; storm success 90.6% → 92.0% | off (opt-in `play_scrub --cbf`) |
| Walking | RMA payload/deck adaptation: privileged teacher + 1 s proprioceptive history estimator | velocity tracking error 0.077 → 0.063 m/s (−18%), falls 6.84 → 6.77 /km, lateral payload COM R² 0.84 | off (opt-in `policy.walk.rma.RMARuntime` + `novelty/walk/policy_rma.onnx`) |

## One-liners

- **Scrub:** A safety shield checks every brush command against a live estimate of the pipe's stiffness, eliminating over-40 N episodes in calm weather and cutting them by 42% in a full storm with no loss of success.
- **Gas:** Without ultrasound, a Bayesian particle filter with infotaxis search finds the leaking flange 86% of the time against 64% for cast-and-surge, and never hands off the wrong flange.
- **Pick-up:** Gemini Robotics-ER points at the grasp spot on bottles, boxes and cans it has never seen, landing on the target 97% of the time against 7% for the colour detector.
- **Fleet / Argon:** Before a Gemini plan is applied, the robots rehearse it in a fast simulation and veto any step that delays leak isolation, so plausible-but-harmful plans cost 31 s less per leak.
- **Perception:** The rust detector re-tunes itself to rain and glare on the fly, recovering F1 from 0.47 to 0.95 in heavy weather while leaving clear-weather frames untouched.
- **Walking:** The Go2 works out from one second of its own joint history where its payload sits and how the deck is moving, cutting velocity-tracking error by 18% on a rolling, heaving deck with up to 5 kg of payload and no rise in falls.

## Details

- **Scrub.** Benchmark: scrub v2 best, 512 episodes, held-out seed 11 (tuned on seed 7). The shield is a discrete-time CBF on the force setpoint; recursive least squares estimates contact stiffness from the force sensor. It is active on about 2% of steps and leaves clean time unchanged. It does not reach 0% in a storm: the arm closes only about 6% of the commanded normal gap per step, too slow for gust and deck impacts. The storm worst single-step peak rose (210 → 252 N), so it stays opt-in. Storm v3 was not on watson, so v2 was used. Code: `policy/cbf_shield.py`. Bench: `bench/cbf_scrub.py`. Chart: `scrub/chart.png`.
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
- **Walking.**
  - Benchmark: 1024 robots × 30 s, seed 42. Payload 0–5 kg with COM ±5 cm; arm deployed with p = 0.5; deck roll/pitch ±0.1 rad plus heave ±1.5 m/s².
  - Method: a privileged teacher encodes payload and deck state into a latent; a 1-D-conv student regresses that latent from 1 s of proprioceptive history; the fine-tuned policy consumes it.
  - Teacher upper bound: 0.062 m/s, 5.71 falls/km. Zeroing the latent gives 0.073 m/s and 7.90 falls/km, so the policy uses it.
  - Estimator R²: COM y 0.84, COM x 0.78, heave 0.47, mass 0.24.
  - The fall rate is unchanged, and no fall reduction is claimed.
  - Code: `policy/walk/rma.py`, `policy/walk/train_rma.py`. ONNX: `walk/policy_rma.onnx`. Chart: `walk/chart.png`.
