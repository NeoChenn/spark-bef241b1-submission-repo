# SCRUB skill (PPO, Isaac Lab DirectRLEnv)

| File | What |
|---|---|
| `envs/rust_state.py` | Pure-torch, batched rust grid (H=32 z-rows x W=64 theta-cols per env): patches, spray/soak, Archard abrasion, crops, RGBA texture, `rig_meta_scene()` loader. CPU-tested (`pytest tests/test_rust_state.py`). |
| `envs/scrub_env_cfg.py` | `ScrubEnvCfg` (geometry, DR, rewards, robot asset) + `ScrubPPORunnerCfg` (rsl_rl) + `urdf_robot_cfg()`. |
| `envs/scrub_env.py` | `ScrubEnv` (registers `Rig-Scrub-Franka-v0`). |
| `train_scrub.py` | Modal L40S trainer; checkpoints at `rust-ckpt:/scrub/<run>/model_<it>.pt`, `progress.jsonl`. |
| `play_scrub.py` | Eval + export (`exported/policy.pt`, `policy.onnx`) + optional video. |

## Run

```bash
.venv/bin/modal run policy/train_scrub.py --smoke                     # 64 envs x 20 iters
.venv/bin/modal run --detach policy/train_scrub.py --run scrub_v1     # 2048 envs, 3000 iters, 6 h cap
.venv/bin/modal run --detach policy/train_scrub.py --run scrub_v1 --resume --iters 2000
.venv/bin/modal run policy/play_scrub.py --run scrub_v1               # stats -> /ckpt/scrub/scrub_v1/eval.json
.venv/bin/modal run policy/play_scrub.py --run scrub_v1 --meta --video --force-patch   # rig-scene demo video
.venv/bin/modal volume get rust-ckpt scrub/scrub_v1/progress.jsonl .  # reward / fraction_removed curve
```

## Model

* **Scene** (matches `sim/output/rig_deck_meta.json`, Z up, deck z=0): pipe horizontal along x, axis at z=1.1 m
  (DR 1.0-1.2), radius 0.15 m (DR 0.10-0.20), small yaw (+-0.1 rad). The quadruped stands during scrubbing, so the
  arm is a fixed-base articulation at dog-back height 0.55 m beside the pipe (base->surface 0.22-0.38 m), facing it.
  Rust patches are spawned on the reachable upper/side face: theta window centred at 0.90 rad (the rig damage normal
  `[0, .62, .78]`) +- 0.75 rad.
* **Pipe frame** (same as `policy/interfaces.py`): `axis` along the pipe, `ref` = horizontal normal pointing at the
  robot (theta=0), `binormal = axis x ref` = up (theta=pi/2). Grid rows = z in [-L/2, L/2] (L=0.6 m), cols = theta.
* **Contact** is analytic: sphere brush (r=2.5 cm, 13 cm in front of the EE link) vs cylinder; penetration x spring k
  (DR 1500-4000 N/m) = normal force; slide speed = brush velocity tangent to the surface. No contact sensor needed.
* **Abrasion**: removal = k * coef * F * v * mult * parabolic footprint; mult 4 (sprayed and soaked >= 30 s),
  1.5 (sprayed, still soaking), 1 (dry). Zero below 5 N; >40 N flagged and penalised. coef DR +-30 %.
* **Action (4)**: brush-target deltas (axial, ref, binormal) in the pipe frame, 1 cm/step max at 30 Hz, and a
  normal-force setpoint F* in [5, 45] N. Within 1 cm of the surface the target's radial coordinate becomes
  `r + brush_r - F*/k_nominal` (hybrid force/position); the true k is randomised so the policy closes the loop on the
  force observation. Joint targets via damped-least-squares IK on the brush-point Jacobian.
* **Obs (329 for 7 arm joints)**: arm q, qd*0.1, brush pos/vel (pipe frame), gap, sin/cos theta, axial, target-brush
  offset, force/40, F*/40, radius, 9x9 local rust crop, 9x9 local spray/soak crop (0 dry, .5 wet, 1 soaked), 8x16
  coarse rust map, offset+distance to nearest rust, fraction remaining, unsprayed-contact fraction, last action.
  Everything is grid-based: in deployment, fill `RustState.rust` from the classifier heatmap.
* **Reward**: 50 x fraction of rust removed this step, +20 on success (< 5 % left), +0.02 force in 10-40 N,
  -0.1 per 10 N over 40, -0.05 x unsprayed contact, -0.01 action rate, -0.2 x distance to nearest rust (m).
  Episode 15 s; terminates on success.
* **Adaptation**: with p=0.3 a new (sprayed, soaked) patch is injected at 3-10 s, or as soon as the first patches are
  clean, so the policy learns to re-target. Episode stats `Episode/new_patch[_success]` are logged.

## Switching to the teammate URDF

```python
from policy.envs.scrub_env_cfg import ScrubEnvCfg, urdf_robot_cfg
cfg = ScrubEnvCfg()
cfg.robot_cfg = urdf_robot_cfg("/assets/robot_dog.urdf")   # fix_base=True, all joints PD-held
cfg.arm_joint_names = ["arm_joint_.*"]     # regex of the arm joints (IK only drives these)
cfg.num_arm_joints = 6
cfg.ee_body_name = "arm_ee_link"           # REQUIRED: link the brush is rigidly mounted on
cfg.brush_offset = (0.0, 0.0, 0.05)        # brush-sphere centre in that link frame
cfg.base_height = 0.0                      # URDF root height (the dog body is in the URDF)
```
(`train_scrub.py --urdf /assets/robot_dog.urdf` sets only the asset; edit the link/joint names above in
`ScrubEnvCfg` to match.) Obs size changes with the joint count, so a Franka checkpoint does not load into it.

## Known limitations

* Arm links do not collide with the pipe (contact is analytic for the brush only).
* Position-only IK: EE orientation is free (fine for a sphere brush).
* Training visuals are off; the pipe/dog visuals in play mode sit at the nominal pose (play pins DR to it).
