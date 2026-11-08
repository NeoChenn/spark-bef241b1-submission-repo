# PPO vs greedy scrub policy on the Gemini test scenarios

**Status: the PPO column is pending.** The trained scrub policy (`runs/scrub_v2/policy.onnx`) is not on the CPU
machine that ran this and is not in git, so only the greedy rerun has been done. The wiring
(`policy/skills/onnx_scrub.py`, `scrub_policy="ppo"` in `policy/scenario_runner.py`) is tested on CPU with a
random-weight ONNX of the same 327 -> 4 signature. These numbers are not PPO results.

## Method

`python -m policy.scenario_runner` re-runs the same 120 scenarios from `results.jsonl` (60 per Gemini round), with
the same parameters and seeds, once for each scrub policy. Nothing new is designed by Gemini, so the comparison is
paired. Everything except the scrub policy is the same: the CPU mock world, the colour stand-in detector, the
orchestrator and its force limit. Raw rows are in `ppo_vs_greedy/rerun_<policy>.jsonl`, and the totals are in
`ppo_vs_greedy/summary.json`.

```bash
# fetch the v2 export first (see "Missing file" below), then:
python -m policy.scenario_runner --policies greedy,ppo --onnx runs/scrub_v2/policy.onnx
```

## Results

| | greedy (baseline, `report.md`) | greedy (rerun) | PPO v2 (ONNX) |
|---|---|---|---|
| unsafe | 25 / 120 | 25 / 120 | pending |
| - missed_rust | 19 | 19 | pending (19 expected, see below) |
| - painted_over_rust | 6 | 6 | pending |
| - force_not_contained / timeout | 0 / 0 | 0 / 0 | pending |
| success | 95 | 95 | pending |
| runs with peak > 40 N | - | 101 | pending |
| share of scrub steps > 40 N | - | 0.43 % | pending |
| mean sim time | - | 61.6 s | pending |

The greedy rerun reproduces the baseline: every one of the 120 labels matches `results.jsonl`.

## What a scrub policy can and cannot change

- **The 19 `missed_rust` runs are a detector problem.** In all 19, DETECT saw no rust (faint `patch_level` < 0.65
  under the colour classifier), so SCRUB never ran: there were 0 scrub steps and 100 % of the rust was left. Any scrub
  policy, PPO included, gets the same 19. Fixing these needs the detector or the DETECT logic, for example a forced
  SECOND_VIEW or the trained MNv4 classifier.
- **Peaks above 40 N come from injected spikes.** All 120 scenarios inject a force spike, and none of the 101 runs
  with a peak above 40 N went over without one. The orchestrator's BACK_OFF handles these spikes. The PPO policy's
  own setpoint is capped at 38 N, and `MockArm` turns that into at most 25 N (`press` <= 1). So the > 40 N numbers
  measure the orchestrator, not the scrub policy.
- **The 6 `painted_over_rust` runs are the only label the scrub policy can move.** These are stubborn rust
  (`abrasion_coef` < 0.3) that left 5.7-10.9 % true residue after 1-3 cycles. The residue was invisible to the
  colour detector, so VERIFY passed it. A policy that presses harder could remove more rust in the same 200-step
  cycle, and so could one that covers the patch better. The PPO maps F* >= 25 N to `press` 1 (25 N), while greedy
  presses at 0.7 (17.5 N) when it is on a cell. Either effect could push the residue below 5 %. This is not measured
  yet. Even then, the underlying fault, a detector that cannot see faint residue, remains.

## Missing file

`runs/scrub_v2/policy.onnx` (relative to the repo root; `$SCRUB_ONNX` or `--onnx` override it). The v2 policy was
trained on the GB10 box (watson), so the export should be at
`~/Documents/spark-bef241b1-submission-repo/runs/scrub_v2/policy.onnx` there. If it only exists as a checkpoint,
re-export it with `policy/play_scrub.py`, which writes `<out_dir>/exported/policy.onnx` with the obs normaliser baked
in. On Modal:

```bash
MODAL_PROFILE=jacobyuan99 .venv/bin/modal run policy/play_scrub.py --run <v2 run> --robot lift_v2 --meta --force-patch
MODAL_PROFILE=jacobyuan99 .venv/bin/modal volume get rust-ckpt scrub/<v2 run>/exported/policy.onnx runs/scrub_v2/policy.onnx
```

## How the PPO runs inside the CPU orchestrator

The policy needs its 327-D Isaac Lab observation: UR3 joints, brush pose, contact force and rust-grid crops. The
mock orchestrator only knows the brush (theta, z), the perceived rust grid and the force. So `OnnxScrubPolicy` keeps a
shadow `bench.scrub_sim.KinematicScrub` (the same CPU kinematics as the scrub benchmark) and builds the observation
from that:

- The perceived rust grid (40 x 64 over 0.8 m) is resampled onto the policy grid (32 x 64 over 0.6 m). The target
  mask becomes the sprayed and soaked mask.
- The policy runs at 30 Hz, 3 steps per 0.1 s orchestrator step.
- Each step returns `[wrap(theta_brush - theta_arm), z_brush - z_arm, press]`, where `press = clip(F*/25 N, 0, 1)`
  while the shadow brush is within 1 cm of the surface, and 0 otherwise.
- The orchestrator still owns the force limit, BACK_OFF, the press scaling and the storm gating.
