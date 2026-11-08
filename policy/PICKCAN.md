# PICK-CAN skill (Isaac Lab DirectRLEnv, suction gripper)

Small cans (soda / paint tins) that fall on a walkway are a trip hazard and block the walking robots.
A `drop_can` judge event puts one on a walkway; the fleet dispatches the nearest robot with a gripper,
which detects the can on its wrist camera, picks it with a suction cup, lifts it and drops it in the
tote on its back.

| File | What |
|---|---|
| `integration/make_go2_ur3_lift.py --grip` | Writes `sim/assets/go2_ur3/go2_ur3_lift_grip.urdf`: the lift robot with a 0.20 m suction wand (`gripper_link`, tip `grip_tip`) instead of the scrub brush, and a yellow tote (`tote_link`, 0.26 x 0.22 x 0.09 m) on the rear of the trunk. The scrub URDF is unchanged. |
| `pickcan/core.py` | Pure torch: can pose helpers, cylinder surface distance, suction seal test, suction hold wrench (PD on the can pose relative to the cup, mass + full inertia tensor, gravity feed-forward), tote test. CPU-tested. |
| `envs/pickcan_env_cfg.py`, `envs/pickcan_env.py` | `Rig-PickCan-Go2UR3Lift-v0`. |
| `pickcan/script.py` | Scripted picker (batched torch state machine) on the same action space: APPROACH, DESCEND, LIFT, CARRY (arc around the mast), LOWER, RELEASE. |
| `pickcan/run.py` | Eval runner (scripted or rsl_rl checkpoint), first episode of every env, JSON summary. |
| `train_pickcan.py` | rsl_rl PPO trainer, time-boxed (`--minutes`), checkpoint every 50 iterations. |
| `pickcan/detect.py` | Can detector stand-in on an RGB-D camera: HSV saturation + colour classes, connected components, PCA, depth back-projection. |
| `pickcan/video.py` | One-take clip on the rig deck (night look), wrist-camera inset with detections. |
| `fleet/jobs/pick_can.py` | Fleet job (`kind="pick_can"`, needs arm + gripper). |

## Env

* Robot: fixed base standing at 0.32 m, legs held, UR3 on the lift mast; arm driven by 6-DoF damped
  least-squares IK on the tip (position + tool axis straight down, yaw free) at 120 Hz.
* Action (5, 30 Hz): tip target delta x/y/z (1.2 cm per step), suction on/off, lift-mast delta.
* Can: 8 sizes (diameter 0.053-0.10 m, length 0.10-0.20 m), mass 0.1-0.5 kg, can-deck friction
  0.35-1.0, lying (65 %) or upright, random yaw, 0.355-0.445 m from the shoulder axis all round the
  robot outside its footprint. The arm starts turned toward the can (the robot has parked next to it).
* Suction: seals when the cup is within 12 mm of the can surface, at least 10 mm inside its edge and
  within 25 degrees of the surface normal. While sealed, a PD wrench holds the can at the seal pose.
  The seal breaks (drop) when the hold needs more than 25 N or 0.8 N m about the cup, or the can is
  torn 3 cm away, e.g. by a collision.
* Observations (57): arm joints and velocities, lift, tip pose/velocity, detector can pose (bias
  sigma 12 mm xy / 6 mm z / 0.1 rad yaw, shrinking to a quarter as the wrist camera closes in, plus
  3 mm jitter), grasp point and tote relative to the tip, can size, can velocity and height, seal and
  suction flags, tool contact force, previous action.
* Success: can at rest inside the tote, seal released, lifted at least 0.15 m. Episode 12 s.
* Rewards: reach, seal bonus, hold, lift, carry to tote, in-tote, success; drop, action rate, tool
  contact force over 20 N, suction on while far from the can.

## Geometric perception (no pixels into the policy)

`pickcan/geom.py`: a depth sensor on the top of the lift mast (32 x 32 rays, 0.4 rad field of view, aimed at the coarse
detection) is ray-cast against the can in torch for every env each step, with 0.4 % range noise and 20 % ray dropout.
Points above the deck are the can segment. A cylinder is fitted with PCA plus extents: centre, axis, radius, length,
lying/upright, residual and a confidence, and the grasp point (top line or lid centre). `geom_obs()` is the compact
14-d observation (relative to the tip, yaw folded to 2θ). `PickCanEnv.geo_obs()` adds the tip state, IK target,
lift, arm joints, seal/suction flags, tote offset, tool force and the last action (41-d). The scripted picker grasps
at the fitted grasp point.

## Learning (`pickcan/learn.py`, `pickcan/diffusion.py`)

One container run: scripted demos on many envs (successful episodes only), then behaviour cloning on the 42-d
geometric observation: a small DDPM diffusion policy (MLP-residual denoiser, about 2 M parameters, chunks of 8
actions, cosine schedule, 8 DDIM steps, replan every 4 steps) and an MLP BC baseline with the same chunks. All
policies are then evaluated on the same seeded scenarios (first episode of every env).

## Results (same 768 seeded scenarios, first episode of each env, `runs_sample/pickcan/eval_learned_n768.json`)

Demos: 768 envs x 1000 steps of the scripted picker (half the envs with 0.3 action noise, DART style), 3.9 min of
collection, 1880 episodes, 1698 successful, 556 k (observation, 8-step chunk) samples, on a shared GB10.

| Policy (input: 42-d depth geometry, no pixels) | Train | Params | Success | Lying / upright | Time to stow | Episodes with a drop | Max tool force p95 / max | Inference, batch 1 |
|---|---|---|---|---|---|---|---|---|
| Scripted picker (fitted grasp + IK state machine) | - | - | 90.0 % | 88.1 / 93.2 % | 11.1 s | 10.8 % | 2.5 / 47.7 N | - |
| Diffusion policy (DDPM, 8 DDIM steps, chunk 8) | 15 min | 2.19 M | 89.7 % | 87.9 / 92.9 % | 11.2 s | 11.3 % | 2.8 / 226 N | 12.2 ms |
| MLP behaviour cloning (chunk 8) | 5 min | 1.62 M | **91.0 %** | 87.7 / 96.8 % | 11.1 s | 9.9 % | 2.6 / 8.7 N | 0.16 ms |

A diffusion policy on depth-derived cylinder geometry learns can pick-up from 1.7 k scripted demos in 15 minutes and
matches the scripted controller (89.7 % vs 90.0 %); the MLP clone of the same data takes 5 minutes and edges it
(91.0 %). The first round (512 envs, no noise injection, 10 / 3 min) reached 61 % (diffusion) and 77 % (MLP); the
noise-injected demos and the can-top height in the observation closed the gap. PPO from scratch was set up
(potential-based staged reward, `train_pickcan.py`) but not trained, because the BC route got there first.

Scripted iterations (n = 256): 49 % (stall at the tote, jerky IK drops), 44.5 % (pan limit on the carry), 87.5 %
(counter-clockwise carry, faster actions), 88.3 % (release low inside the tote).

Re-evaluated from the shipped fp16 weights (`runs_sample/pickcan/eval_shipped_weights_n768.json`, same 768
scenarios): scripted 90.2 %, diffusion 89.1 %, MLP-BC 91.1 %.

Weights: `pickcan/weights/{mlp,diffusion}.npz` (fp16, `diffusion.load_weights`). The fleet `pick_can` job uses the
MLP-BC numbers (`fleet/jobs/pick_can.py`: `POLICY`, `SUCCESS_P = 0.91`, `TIME_S = 11.1`); the scripted picker is
the fallback.

## Run (GB10, docker only: `rig-scrub-isaaclab:5.1`, `--init --rm --memory 16g`, repo at `/repo`)

```bash
/isaac-sim/python.sh -m policy.pickcan.run --mode script --num_envs 256 --out runs/pickup/eval.json --headless
/isaac-sim/python.sh -m policy.pickcan.learn --out runs/pickup/learn --headless
/isaac-sim/python.sh -m policy.pickcan.learn --load policy/pickcan/weights --out runs/pickup/eval_w --headless
/isaac-sim/python.sh -m policy.pickcan.video --policy mlp --ckpt policy/pickcan/weights/mlp.npz --out_dir runs/pickup/clip --headless
```

## For Mission Control (Deck patrol)

* Live judge event `drop_can {size?: "small"|"large", at?: robot id | walkway | xy}` (whitelisted in `rig/hud.py`):
  the nearest robot with a gripper (R1, R3) gets a `pick_can` task; the can blocks the walkway (0.6 m) until the
  job is done, then routes reopen.
* Hero motion: recorded rollouts `runs_sample/pickcan/rollout_*.json` at 30 fps. Each frame has the robot joint
  positions (`meta.joint_names`), can position and quaternion, tool tip, phase, seal flag, contact force, and the
  depth sensor eye plus up to 160 hit points on the can: the "D" view, i.e. the policy's real input before the
  cylinder fit. Frame: robot base at the origin, deck at z = 0 (the clip put the robot at deck (1.5, -1.9)).
* Running live needs the Isaac Lab env for the observation (depth ray cast, IK, suction); the MLP itself is 0.16 ms.
