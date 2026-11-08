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
