# Handoff: rust detection + repair (Agentverse 2026, Robotics track)

Status as of **2026-11-07 ~19:15 UTC**. Hackathon ends roughly **2026-11-08 10:00 UTC** (17 h window started ~17:30 UTC).
Branch: **`rust-detection`** (pushed to `origin`). Never push to `main`; the sim team owns `main` and `sim/*` branches.

## 1. What we're building

A simulated **Unitree Go2 quadruped + UR3 arm** on an oil-rig deck (Isaac Sim) finds rust on a pipe and repairs it:
`detect → spray → wait 30 s → scrub (PPO policy) → re-detect → paint → verify`.

Judging (what to optimise): perception-driven action 25%, adaptation to mid-run change 25%, edge autonomy 15%,
relevance + critical failure mode on screen 20%, ambition/quality 15%.
Deliverables: 1-page PDF technical summary, ≤1 min simulation video (perception on screen + one unscripted change,
single take), 5-min live pitch.

## 2. State of each piece

| Piece | State | Where |
|---|---|---|
| Dataset | ✅ 5,894 imgs (2,203 rust / 3,691 clean), phash-deduped, split by source group | `detection/prepare_data.py`, `detection/DATA.md` |
| Detector | ✅ on real photos; ❌ misses rust in sim renders; sim-domain retrain **in progress, uncommitted** | `detection/` |
| Localizer (patch heatmap → mask → pipe grid) | ✅ | `detection/localize.py` |
| Orchestrator state machine + spray/paint skills + mock demo | ✅ 22 tests | `policy/orchestrator.py`, `policy/skills/`, `policy/demo_mock.py` |
| Robot with lift mast | ✅ | `sim/assets/go2_ur3/go2_ur3_lift.urdf`, generator `integration/make_go2_ur3_lift.py` |
| Isaac Sim 5.1 + Isaac Lab 2.3.2 on Modal | ✅ physics + PPO; ❌ rendering | `infra/` |
| Scrub PPO policy | ✅ 99.2% success; ⚠ presses too hard (mean max ~46–52 N vs 40 N limit) | `policy/envs/`, `policy/train_scrub.py` |
| Blender render on Modal (workaround for no RTX) | 🟡 first test launched, result unknown | `infra/render_blender.py` (uncommitted) |
| Sim integration contract | ✅ written, sim team hasn't implemented yet | `integration/INTERFACE.md` |
| Demo video, PDF summary, pitch | ❌ not started | — |

### Detector numbers (MobileNetV4-Conv-Small + LoRA r=16, 0.26M of 2.73M params trainable, local CPU run)
Test at val-picked threshold (recall ≥ 0.95): acc 0.977, P 0.970, R 0.975, **F1 0.973**, AUROC 0.998, PR-AUC 0.998
(tn 508, fp 11, fn 9, tp 353). Val F1 0.957. Temperature 2.25, ECE 0.017 after calibration, 1.4% of test in the
"unsure" band [0.35, 0.65]. 0.19 GMACs, ONNX CPU batch-1 1.1 ms (4 threads). Loss: BCEWithLogits + pos_weight.
LoRA beat the F1 ≥ 0.90 bar, so full fine-tune isn't needed.
**Sim gap:** on `sim/output/preview_*.png` painted pipes are correctly clean (p 0.01–0.08) but the rendered rust
damage patch scores 0.06 and a rusty pipe 0.00 → must fix before the demo.
Fix in progress (uncommitted `detection/synth_sim.py`): synthetic sim-style data made from the scene's own
Poly Haven textures (`sim/assets/third_party/textures/rust_coarse_01` is the damage-patch texture; also
`rusty_metal_02`, `metal_grate_rusty` as rust; `metal_plate`, `blue_metal_plate`, `box_profile_metal_sheet` and plain
paint colours as clean), with pipe-like shading/noise, → `rust_v2`, retrain, test on held-out preview renders.
Local checkpoints: `runs/detector/{best,mobilenetv4_conv_small_lora,...}` (gitignored). Best model for consumers:
`/ckpt/detector/best/{model.onnx, meta.json}` on Modal volume `rust-ckpt` (meta has input size, mean/std,
threshold, temperature, unsure band).

### Scrub policy numbers (`scrub_lift_v1`, rsl_rl PPO, 2048 envs, L40S, ~45k steps/s)
Solved by iter ~20, plateaued by ~100. `model_100` eval with a new rust patch injected mid-episode every time:
exact rig geometry (`--meta`) 99.2% success, 5.9 s mean clean time, mean episode-max force 52 N, peak 86 N.
Checkpoints every 50 iters at `rust-ckpt:/scrub/scrub_lift_v1/model_<it>.pt`; curve at `.../progress.jsonl`.
`play_scrub.py` exports `exported/policy.pt` + `policy.onnx` (obs normaliser baked in).
Env design: fixed-base Go2 at standing pose (trunk 0.32 m), `lift_joint` set to 0.38–0.45 m at reset (not an
action), 6 arm joints via diff-IK on `brush_tip`; pipe along x at z 1.1 ± 0.05, r 0.15 ± 0.03; rust grid (rows z,
cols theta) in `policy/envs/rust_state.py` (pure torch); 4-D action = brush moves in pipe frame + 5–45 N force
setpoint; 327-D obs. Reward: rust removed + completion bonus + in-band contact − over-force (w_over_force=0.1, too
weak) − unsprayed scrubbing − action rate − distance shaping.

## 3. Immediate next steps (priority order)

1. **Finish the sim-domain detector.** Make sure the rendered damage patch in `sim/output/preview_repair_closeup.png`
   scores > 0.5. Promote to `/ckpt/detector/best` + `runs/detector/best`, write `detection/RESULTS.md`, commit.
2. **Scrub v2 (force fix)** — user had not yet approved; ask first. Resume from latest `scrub_lift_v1` checkpoint
   with `w_over_force` ×5, clamp the force setpoint ≤ 38 N, harder randomisation (friction ±50%, patch count 1–4,
   pipe yaw ±10°). ~15 min on one L40S. Target: episode-max force < 40 N on ≥ 95% of episodes, success ≥ 95%.
3. **Ask the user to stop idle Modal apps** (agent permission to run `modal app stop` was denied — do not run it
   yourself; give the user the command): `ap-Zj4w6Vj35ANGLCNbKkPvOD` (scrub v1, plateaued),
   `ap-eM8tC9c15W155tg1T43vRU` (hung smoke test holding an L40S).
4. **Render path for the video.** Either (a) the sim team plays `policy.onnx` + orchestrator in their Isaac 6.2
   scene on the GB10 (preferred, real look), following `integration/INTERFACE.md`; or (b) finish
   `infra/render_blender.py` (Blender Cycles on CUDA reads the scene's UsdPreviewSurface materials) and replay
   logged rollouts (per-frame link poses + rust grid as npz) there.
5. **Wire it together in the sim:** orchestrator ← localizer on `repair_closeup`/wrist camera frames; ScrubPolicy
   wraps the ONNX policy; `RustState.to_texture()` drives a rust overlay on the pipe so scrubbing is visible; HUD
   lines from `orchestrator.overlay_text()` burned onto the streamed frames. Demo the unscripted change by spawning
   a new rust patch mid-soak/mid-scrub (orchestrator already re-plans; see `policy/demo_mock.py` trace).
6. **Deliverables:** 1-page PDF (user/problem, features, architecture, where each model runs + hardware + latency,
   limits, critical failure mode = over-force/missed rust → back-off/escalate, safety), video, pitch.

## 4. Gotchas (learned the hard way)

- **Modal profiles:** use `MODAL_PROFILE=jacobyuan99` (user's personal workspace, ~$77 credit left at 19:00).
  The `team` profile (workspace `agentverse`) is over its spend limit and refuses L40S. Volumes are per-workspace;
  both have `rust-data`, `rust-ckpt`, `rig-assets`, but the current data/checkpoints live in **jacobyuan99**.
  Check cost: `MODAL_PROFILE=jacobyuan99 .venv/bin/modal billing summary`.
- **No rendering on Modal.** Isaac RTX needs Vulkan; Modal's gVisor sandbox only forwards CUDA compute → any camera
  segfaults in `librtx.scenedb.plugin.so`. Train on privileged state; render elsewhere (or Blender Cycles/CUDA).
- **Long Modal runs:** `modal run --detach` runs got cancelled when the local client was killed by a `timeout`
  wrapper. Reliable pattern: `modal deploy` + `Function.spawn` (see `policy/SCRUB.md`).
- **Isaac on Modal:** run scripts with `/usr/local/bin/python` (`ISAACLAB_PY`), `cpu=8, memory=32768`;
  `simulation_app.close()` hangs → use `infra.isaac_image.run_isaac(...)` or end scripts with `os._exit(0)`;
  init joint pos must respect calf limits (`.*_thigh_joint: 0.8, .*_calf_joint: -1.5`).
- **Image pins:** `setuptools<80`, `flatdict==4.0.1 --no-build-isolation`, `tensordict==0.9.1` (see `infra/isaac_image.py`).
- **Reach:** UR3 on bare Go2 tops out at ~1.1 m; rust is at 1.22 m. That's why the 0–0.45 m `lift_joint` exists.
  With full lift, 6/9 points over the patch are reachable; the top/far edge isn't (training counts reachable cells only).
- **Sim team runs Isaac Sim 6.2** on an aarch64 GB10; their streaming is camera → JPEG → FastAPI web page (WebRTC
  doesn't work on aarch64). Their `/World/Cameras/repair_closeup` (seed 7) sits on the far side of the pipe from
  the damage — `INTERFACE.md` gives a fix.
- **Tests:** run `.venv/bin/python -m pytest -q tests` (bare `pytest` can't import the packages). 61 tests.
- Rusty decorative textures elsewhere in the scene will also be flagged as rust by a working detector.

## 5. Quick commands

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv modal torch numpy pytest   # setup
.venv/bin/python -m policy.demo_mock                      # full pipeline mock, CPU, ~1 s
export MODAL_PROFILE=jacobyuan99
.venv/bin/modal run detection/prepare_data.py             # rebuild dataset (wipes /data/rust_v1)
.venv/bin/modal run detection/modal_app.py --help         # detector training on GPU
.venv/bin/modal app logs rig-scrub-train                  # scrub training logs
.venv/bin/modal volume get rust-ckpt scrub/scrub_lift_v1/progress.jsonl .
.venv/bin/modal run infra/render_blender.py --camera repair_closeup,hero   # Blender test render
```

Read next: `policy/SCRUB.md`, `infra/README.md`, `integration/INTERFACE.md`, `detection/DATA.md`.
