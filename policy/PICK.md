# PICK skill (litter pickup): computer vision finds it, RL picks it up

Same pattern as the SCRUB skill: a camera model decides *what and where*, a PPO policy does the arm motion. Here the
debris detector (`detection/rig_vision.py`) finds litter on the deck and gives a pick point; the PICK policy
(`policy/skills/pick_policy/policy.onnx`) grasps it and lifts it, closed loop at 30 Hz.

| File | What |
|---|---|
| `envs/pick_state.py` | Pure-torch, batched litter-on-deck model with an analytic parallel gripper (like `rust_state.py`). CPU-tested (`tests/test_pick_state.py`). |
| `envs/pick_scripted.py` | A hand-written pick (hover, line up, open, lower, close, lift): the baseline, and the teacher for the warm start. |
| `envs/pick_cpu_env.py` | `PickState` as an rsl_rl `VecEnv`: observations, actions, rewards; the team's scrub PPO settings. |
| `train_pick_cpu.py` | CPU training: behaviour cloning + DAgger from the script, critic warm-up, PPO fine-tune, best-checkpoint selection, ONNX export. |
| `envs/pick_env.py`, `envs/pick_env_cfg.py` | The same task in Isaac Lab (`Rig-Pick-Go2UR3Gripper-v0`): real UR3 + gripper via 6D IK on `grasp_frame`, the object and grasp events from `PickState`. **Not yet run in Isaac.** |
| `train_pick.py` | Isaac Lab trainer (Modal / GB10), a copy of `train_scrub.py`; `--init_from` takes a CPU checkpoint. |
| `deck/pick_skill.py` | `RLPickSkill`: the ONNX policy as the robot's pick action (mock and Isaac bridge). |

## Robot and task

* **Robot:** the patrol team's `go2_ur3_gripper.urdf` (branch `patrol-nav`): Go2 + lift + UR3 + parallel-jaw gripper
  (0.12 m max opening), TCP `grasp_frame`. This skill can replace the scripted grasp in their PICK state.
* **Reach (important for the patrol team):** standing, the UR3 shoulder is 0.64 m above the deck and the UR3 reaches
  0.5 m, so the gripper cannot reach the deck. The skill assumes the Go2 **crouches to a trunk height of 0.17-0.21 m**
  and stops with the object **0.20-0.32 m ahead** (|lateral| <= 0.10 m). The patrol mission's 0.45 m standoff is out of
  reach for a deck pick (wrist 0.65 m from the shoulder).
* **Objects:** lying drink cans (diameter 5.2-8 cm, length 9-20 cm) and flat packaging, any orientation.
* **Camera error:** the policy sees the object only through the detector's estimate: a per-episode bias up to 1.5 cm
  and 9 degrees (domain randomisation), plus 2 mm noise per step.
* **Action (5):** TCP target dx, dy, dz (1 cm per step), yaw (0.08 rad per step), gripper open / close.
  **Observation (23):** TCP, yaw, opening, holding, estimated grasp point and angle relative to the gripper, object
  size, crouch height, last action, time.
* **Events (analytic, like the scrub brush):** grasp when the jaws close centred (< 1.2 cm), lined up (< 20 deg) and at
  15-85 % of the object's height; knock (object pushed 1.5 cm) when the fingers come down on it; drop when opening
  while holding; success when held 15 cm up.

## Training (CPU, `python -m policy.train_pick_cpu`, about 15 min on 4 cores)

PPO from scratch never found a grasp (the jaws must close for several steps in a row while centred within 1 cm;
random actions did that about once in 10,000 steps). What worked:

1. **Behaviour cloning + DAgger** from the scripted pick (4 rounds, 1M samples): 80-83 % success.
2. **Critic warm-up:** 20 PPO iterations with the actor frozen.
3. **PPO fine-tune** (the team's network and PPO settings; fixed learning rate 1e-4), 300 iterations, 1024 envs,
   with 30 % of episodes starting hovered over the object (curriculum).
4. **Best checkpoint** on a fair test (normal starts only): iteration 40.

Bugs found on the way, each fixed and covered by a test or a comment:
* the success bonus did not cover the shaping reward still to come, so hovering beat finishing (reward hacking);
* the observation normaliser stayed at identity during cloning and then shifted under PPO;
* the adaptive learning rate climbed while the actor was frozen and wrecked the first real update.

## Results (1024 held-out episodes, same camera error for both)

| | picks | knocks per pick | median time |
|---|---|---|---|
| **RL policy (shipped)** | **88.4 %** | **1.9** | 1.8 s |
| scripted pick | 85.5 % | 3.3 | 1.6 s |
| behaviour cloning only | 80-83 % | 3.9 | 1.6 s |
| RL policy, perfect camera | **100 %** | **0** | 1.8 s |
| scripted pick, perfect camera | 96.9 % | 0.9 | 1.6 s |

In the robot mock (`python -m policy.leak.mock --leak none --debris litter`) the debris model finds a bottle and two
cans and the policy picks each in under 2 s with no knocks.

## Limits (say these in the summary)

* The grasp is analytic, like the scrub brush's contact: no finger-object physics, no slipping, no rolling cans.
* Trained on the CPU model of the arm (a first-order tracking lag), not the real UR3 dynamics. `pick_env.py` puts the
  same task on the real arm in Isaac Lab; run `train_pick.py --init_from` a CPU checkpoint there before trusting it.
* The gain over the scripted pick is modest (3 points, half the knocks). The policy learned to cope better with
  camera error; with a perfect camera both are near perfect.
* The object's angle has to come from the detector (the long axis of the changed pixels); `deck_bridge.pick_yaw` is
  still 0 by default.
