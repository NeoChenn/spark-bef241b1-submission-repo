# SCRUB skill (PPO, Isaac Lab DirectRLEnv)

| File | What |
|---|---|
| `envs/rust_state.py` | Pure-torch, batched rust grid (H=32 z-rows x W=64 theta-cols per env): patches, spray/soak, Archard abrasion, crops, RGBA texture, `rig_meta_scene()` loader. CPU-tested (`pytest tests/test_rust_state.py`). |
| `envs/scrub_env_cfg.py` | `ScrubLiftEnvCfg` (main), `ScrubEnvCfg` (base / no-lift low pipe), `ScrubFrankaEnvCfg`, `ScrubPPORunnerCfg` (rsl_rl), `urdf_robot_cfg()`. |
| `envs/scrub_env.py` | `ScrubEnv`; registers `Rig-Scrub-Go2UR3Lift-v0` (`--robot lift`, default), `Rig-Scrub-Go2UR3-v0` (`--robot go2ur3`), `Rig-Scrub-Franka-v0`. |
| `train_scrub.py` | Modal L40S trainer; checkpoints at `rust-ckpt:/scrub/<run>/model_<it>.pt`, `progress.jsonl`. |
| `play_scrub.py` | Eval + export (`exported/policy.pt`, `policy.onnx`) + optional video. |

## Run (GPU jobs use the personal workspace, see infra/README.md)

```bash
MODAL_PROFILE=jacobyuan99 .venv/bin/modal run policy/train_scrub.py --smoke                     # 64 envs x 20 iters
# long runs: deploy + spawn (a killed local `modal run --detach` client cancelled two runs after ~4 min)
MODAL_PROFILE=jacobyuan99 .venv/bin/modal deploy policy/train_scrub.py
MODAL_PROFILE=jacobyuan99 .venv/bin/python -c "import modal; print(modal.Function.from_name('rig-scrub-train','train').spawn('scrub_lift_v1', 2048, 3000, False, '', 'lift', '').object_id)"
#   args: run, num_envs, iters, resume, urdf, robot, init_from
MODAL_PROFILE=jacobyuan99 .venv/bin/modal run --detach policy/train_scrub.py --run scrub_lift_v1 --resume --iters 2000
MODAL_PROFILE=jacobyuan99 .venv/bin/modal run policy/play_scrub.py --run scrub_lift_v1               # stats -> /ckpt/scrub/scrub_lift_v1/eval.json
MODAL_PROFILE=jacobyuan99 .venv/bin/modal run policy/play_scrub.py --run scrub_lift_v1 --meta --force-patch   # exact rig geometry
# --video needs RTX cameras, which crash on Modal (privileged-state only); render on the sim box with exported/policy.pt
MODAL_PROFILE=jacobyuan99 .venv/bin/modal volume get rust-ckpt scrub/scrub_lift_v1/progress.jsonl .  # reward / fraction_removed curve
```

## Run on the GB10 box (aarch64, Isaac Sim 5.1 container, RTX video works)

```bash
docker build -t rig-scrub-isaaclab:5.1 -f infra/watson/Dockerfile.isaaclab infra/watson
infra/watson/run.sh policy.train_scrub --worker --num_envs 2048 --iters 150 --run_dir /repo/runs/scrub_lift_watson   # ~9 min
infra/watson/run.sh policy.play_scrub --worker --ckpt /repo/runs/scrub_lift_watson/model_149.pt --out_dir /repo/runs/scrub_lift_watson \
  --num_envs 256 --episodes 512 --meta /repo/sim/output/rig_deck_meta.json --force_patch
infra/watson/run.sh policy.play_scrub --worker --ckpt /repo/runs/scrub_lift_watson/model_149.pt --out_dir /repo/runs/scrub_video_close \
  --num_envs 4 --episodes 100000 --video --video_steps 450 --cam close --meta /repo/sim/output/rig_deck_meta.json --force_patch --enable_cameras
```
torch 2.7 cu128 has no sm_121 NVRTC target, so `run.sh` sets `PYTORCH_JIT=0` (TorchScript export is skipped; ONNX still exported).

## Model

* **Robot (main, `--robot lift`)**: `sim/assets/go2_ur3/go2_ur3_lift.urdf` (Modal `rig-assets:/go2_ur3/`): Go2 trunk
  fixed at z=0.32 m, legs PD-held at the standing pose (hips +-0.1, thighs 0.8/1.0, calves -1.5), gravity off;
  prismatic `lift_joint` set to U(0.38, 0.45) m at reset and held (NOT a policy action: in deployment the
  orchestrator sets it from the detected rust height); only the 6 `ur3_*` joints move. IK/contact frame = URDF link
  `brush_tip` (0.12 m along `ur3_ee_link` +x); the contact sphere (r 2.5 cm) is centred 2.5 cm behind the tip.
* **Scene (main)**: the real rig pipe (`rig_deck_meta.json`): axis along x, z = 1.1 +- 0.05 m, r = 0.15 +- 0.03 m,
  yaw +-0.1 rad. Dog faces +x parallel to the pipe, trunk 0.45-0.55 m from the axis on the damage side (+y).
  Rust spawns around the damage normal theta = atan2(0.784, 0.621) = 0.90 rad +- 0.35 rad, and only on cells within
  0.12-0.62 m of the UR3 shoulder (per-env reachability mask; fraction_remaining / success are over those cells).
* `--robot go2ur3`: same dog without the lift, pipe lowered to z 0.5-0.75 (UR3 alone tops out at ~1.1 m EE height).
  `--robot franka`: Franka at 0.55 m beside a z~1.1 pipe. All three share the same obs/action layout (327 obs for
  6-joint arms) and pipe-frame semantics.
* `play_scrub.py --meta` pins the geometry to `sim/output/rig_deck_meta.json` (radius, damage theta); it lowers the
  pipe with a warning only if the rig z exceeds the variant's `max_reach_z`.
* **Pipe frame** (same as `policy/interfaces.py`): `axis` along the pipe, `ref` = horizontal normal pointing at the
  robot (theta=0), `binormal = axis x ref` = up (theta=pi/2). Grid rows = z in [-L/2, L/2] (L=0.6 m), cols = theta.
* **Contact** is analytic: sphere brush (r=2.5 cm) vs cylinder; penetration x spring k
  (DR 1500-4000 N/m) = normal force; slide speed = brush velocity tangent to the surface. No contact sensor needed.
* **Abrasion**: removal = k * coef * F * v * mult * parabolic footprint; mult 4 (sprayed and soaked >= 30 s),
  1.5 (sprayed, still soaking), 1 (dry). Zero below 5 N; >40 N flagged and penalised. coef DR +-30 %.
* **Action (4)**: brush-target deltas (axial, ref, binormal) in the pipe frame, 1 cm/step max at 30 Hz, and a
  normal-force setpoint F* in [5, 45] N. Within 1 cm of the surface the target's radial coordinate becomes
  `r + brush_r - F*/k_nominal` (hybrid force/position); the true k is randomised so the policy closes the loop on the
  force observation. Joint targets via damped-least-squares IK on the brush-point Jacobian.
* **Obs (327 for the 6 UR3 joints)**: arm q, qd*0.1, brush pos/vel (pipe frame), gap, sin/cos theta, axial, target-brush
  offset, force/40, F*/40, radius, 9x9 local rust crop, 9x9 local spray/soak crop (0 dry, .5 wet, 1 soaked), 8x16
  coarse rust map, offset+distance to nearest rust, fraction remaining, unsprayed-contact fraction, last action.
  Everything is grid-based: in deployment, fill `RustState.rust` from the classifier heatmap.
* **Reward**: 50 x fraction of rust removed this step, +20 on success (< 5 % left), +0.02 force in 10-40 N,
  -0.1 per 10 N over 40, -0.05 x unsprayed contact, -0.01 action rate, -0.2 x distance to nearest rust (m).
  Episode 15 s; terminates on success.
* **Adaptation**: with p=0.3 a new (sprayed, soaked) patch is injected at 3-10 s, or as soon as the first patches are
  clean, so the policy learns to re-target. Episode stats `Episode/new_patch[_success]` are logged.

## Another robot / URDF

```python
from policy.envs.scrub_env_cfg import ScrubEnvCfg, urdf_robot_cfg
cfg = ScrubEnvCfg()
cfg.robot_cfg = urdf_robot_cfg("/assets/other.urdf", init_joint_pos={...}, arm_joint_expr="arm_.*")
cfg.arm_joint_names = [...]; cfg.num_arm_joints = 6
cfg.ee_body_name = "tool_link"             # REQUIRED: link the brush is rigidly mounted on
cfg.brush_offset = (0.11, 0.0, 0.0)        # brush-sphere centre in that link frame
cfg.base_height = 0.32                     # root z
```
`SCRUB_URDF=/path/go2_ur3.urdf` overrides the default URDF location. Obs size = 2*J + 315.

## Known limitations

* Arm links do not collide with the pipe (contact is analytic for the brush only).
* Position-only IK: EE orientation is free (sphere brush), so the brush is not forced normal to the pipe; the
  reachability mask is a shoulder-distance proxy, not full IK with orientation.
* Training visuals are off; the pipe visual in play mode sits at the nominal pose (play pins DR to it).
* Legs are fixed (fixed base, gravity off): the trunk does not sway under arm reaction forces.
