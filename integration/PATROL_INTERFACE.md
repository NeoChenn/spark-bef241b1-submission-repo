# Patrol ⇄ sim interface contract

Owner: the patrol side (`patrol/`). We train offline in a 2D sim (`patrol/sim2d.py`, no Isaac); **you
simulate, we test** means the sim team runs the trained policy and the mission state machine inside the
Isaac Sim 6.2 process and reports back. Scope of this milestone: follow a predefined path between two
bins, handle disturbances (objects lying in the path, pushes), pick up what fits, dump it at the bins.
Rust repair joins later as one more disturbance type (section 8).

```
Isaac Sim (rig/sim/main.py loop, 10 Hz decisions)            patrol (same process, numpy + onnxruntime)
  planar lidar on the trunk ── ranges + beam angles ──────►  PatrolMission.step(t, ranges, angles, pose, v, w)
  base_link pose (x, y, yaw) + measured vx, wz ───────────►     ├─ Tracker(detect_frame(...))  lidar detector
                                                               ├─ PatrolController  ── policy.onnx (PPO)
  Go2 locomotion policy (velocity tracking) ◄── (vx, wz) ◄────┤      FOLLOW: path following + obstacle avoidance
  UR3 diff-IK on grasp_frame ◄── arm_target_xy ◄──────────────┤      APPROACH / PICK: scripted grasp
  gripper_finger_left_joint ◄── gripper "open"/"close" ◄──────┤      DUMP at start_bin / end_bin, then reverse
  HUD ◄── Command.state / Command.event ◄─────────────────────┘      BLOCKED: no progress -> turn back
```

## 1. Frames and units

* Metres, seconds (sim time), radians. World Z up; the patrol only uses world **x, y** (z ignored).
* Robot frame: x forward, y left, yaw CCW from world +x. `pose = (x, y, yaw)` of `base_link` in world.
* `v` = measured forward speed (m/s, body x), `w` = measured yaw rate (rad/s, CCW +).
* Lidar beam angles are in the **robot frame**, 0 = straight ahead, + = to the left (CCW), any range of
  values (wrapped internally).

## 2. Robot additions

| Prim / link | What | Status |
|---|---|---|
| `sim/assets/go2_ur3/go2_ur3_gripper.urdf` | Go2 + lift + UR3 with a **parallel-jaw gripper** (0.12 m max opening) replacing the brush, plus an open-top **container** bin on the trunk. Generator: `integration/make_go2_ur3_gripper.py` | new |
| `grasp_frame` | IK target between the finger pads (fixed to `gripper_base` via `gripper_grasp_joint`) | URDF |
| `gripper_finger_left_joint` / `gripper_finger_right_joint` | prismatic jaws, right mimics left; open = upper limit (0.06 each side), close = drive to 0 with a force limit | URDF |
| `container` / `container_drop` | on-robot bin (inner 0.26 × 0.20 × 0.12 m) and the release point above it | URDF |
| `…/base/patrol_lidar` | **new** planar lidar, see below | new |

**Lidar to add:** a 2D, 360°, single-plane scan mounted on the trunk, scan plane **0.25–0.35 m above the
deck** so it sees small objects lying on the deck. Anything entirely below the scan plane is invisible to
it, so keep the plane as low as the legs allow. Range 4 m (`features.LIDAR_RANGE`; longer returns are
clipped). Beam count: default **360 beams (1°)**; the recommended minimum for small objects is in
`patrol/DETECT_RESULTS.md`. Any count and FOV work for the policy because the scan is min-pooled into 36
sectors of 10° (`features.pool_sectors_np`). Either an Isaac RTX lidar (a 2D rotary profile) or a PhysX
raycast lidar (`isaacsim.sensors.physx` `RotatingLidarPhysX`, `horizontal_resolution=1.0`, `fov=(360, 0)`)
is fine. Pass no-return beams as `inf`, `nan` or `>= 4.0`; they read as "free".

## 3. Path file

```json
{"frame": "world",
 "waypoints": [[x0, y0], [x1, y1], ...],
 "start_bin": [x, y],
 "end_bin":   [x, y]}
```

* Waypoints are an ordered polyline from the start bin to the end bin; extra z values are ignored.
* Any spacing: we resample at 0.1 m (`patrol.controller.resample_path`). Keep corners gentle (turn
  radius ≥ ~1.2 m, the training range) and the walkway ≥ 0.85 m wide (wall to wall).
* The robot shuttles start → end → start …; at each end it dumps its container into that bin.

## 4. Policy I/O

Files: `runs/patrol/<run>/policy.onnx`, `policy_meta.json`, `deploy.pt`, `eval.json`, `progress.jsonl`;
the same files are on Modal volume `rust-ckpt:/patrol/<run>/` (workspace `neochenn`):
`modal volume get rust-ckpt patrol/<run> runs/patrol/`.

ONNX: input `"obs"` float32 `[B, 53]`, output `"action"` float32 `[B, 2]` in [-1, 1]. The observation
normaliser is baked in, so feed raw features. You normally don't build `obs` yourself:
`PatrolController` does it from the scan, pose and path.

**obs (53 floats, in this order)** (copied from `patrol/features.py`):

```
[0:36]   lidar, min range per 10 deg sector, sector j covers [-pi + 10j deg, -pi + 10(j+1) deg) in the
         robot frame (x forward, y left, CCW), divided by LIDAR_RANGE, 1.0 = nothing within range
[36]     lateral offset from the path (m, + = robot is left of the path), / 0.5
[37]     heading error yaw - path_heading (rad, wrapped), / pi
[38:48]  path points 0.5, 1.0, 1.5, 2.0, 2.5 m ahead along the path, robot frame (x, y) pairs, / 2.5
[48:50]  measured forward speed (m/s) / V_MAX, yaw rate (rad/s) / W_MAX
[50:52]  previous action (policy output, in [-1, 1])
[52]     distance left to the end of the path (m), / 5, clipped to 1
```

**action → velocity command** for the Go2 locomotion policy (body frame):

```
vx = a0 * V_MAX if a0 >= 0 else a0 * -V_MIN      V_MAX = 1.0 m/s, V_MIN = -0.3 m/s
wz = a1 * W_MAX                                   W_MAX = 1.5 rad/s
vy = 0
```

**Control rate: 10 Hz** (one decision per 0.1 s sim time; hold the command in between). Training assumed
the base reaches the commanded velocity with a first-order lag of 0.3–0.8 per step, i.e. a few hundred ms.

## 5. Glue loop (pseudo-code inside `rig/sim/main.py`)

```python
import json, numpy as np
from patrol.controller import PatrolController
from patrol.mission import PatrolMission, MissionCfg

spec = json.load(open("path.json"))
ctrl = PatrolController("patrol/models/nav_v3a/policy.onnx", spec["waypoints"])   # or "pure_pursuit" baseline
mission = PatrolMission(ctrl, MissionCfg())
angles = np.deg2rad(np.arange(360) - 180 + 0.5)            # whatever your lidar reports, robot frame

while running:                                             # every 0.1 s of sim time
    ranges = lidar_annotator.get_data()["distance"]        # (B,) metres, inf for no return
    x, y, yaw = base_pose_xy_yaw()                         # base_link in world
    v, w = base_body_vel()                                 # measured forward speed, yaw rate
    cmd = mission.step(sim_time, ranges, angles, (x, y, yaw), v, w)

    loco.set_command(vx=cmd.vx, vy=0.0, wz=cmd.wz)         # Go2 velocity-tracking policy
    if cmd.state == "PICK" and cmd.arm_target_xy is not None:
        # robot is stopped ~0.45 m from the object, facing it
        ik.move("grasp_frame", xyz=(*cmd.arm_target_xy, deck_z + 0.03), approach=(0, 0, -1))
    if cmd.gripper == "open":  gripper.open()              # finger joints to upper limit
    if cmd.gripper == "close": gripper.close(force=20.0)   # then lift and move grasp_frame to container_drop, open
    if cmd.state == "DUMP":
        # at start_bin / end_bin: tip the on-robot container (or pick items out of it) into the station bin
        dump_container_into_station_bin()
    hud(mission.overlay_text())                             # 4 lines: state, event, items/legs, target
    # states: FOLLOW, APPROACH, PICK, DUMP, BLOCKED, TURN (turn in place after every reversal)
    advance(0.1)
```

Picked items go into the on-robot `container` first (released at `container_drop`); the container is
emptied into the station bin during `DUMP`. `MissionCfg` holds capacity, standoff, timings and the
detector thresholds; the mission reverses the path itself after a dump or a block.

## 6. Disturbance hooks for testing (not announced to the mission)

| Hook | Implementation | Expected reaction |
|---|---|---|
| `drop_object(s, size ≤ 0.11 m)` | spawn a small rigid body (bolt, can, rag-sized box) on the walkway ahead of the robot | `FOLLOW → APPROACH → PICK → FOLLOW`, item counter +1 |
| `drop_big(s, size 0.15–0.5 m)` that leaves a gap | spawn a crate beside the path centre | not pickable: policy steers around it and rejoins the path |
| `block_corridor(s)` | a crate or pipe spool spanning the walkway | no progress → `BLOCKED` → reverses and heads back to the previous bin |
| `push(dir, 0.3–0.8 m/s)` | apply a base velocity impulse | recovers to the path (trained with random pushes) |
| `fill_container()` | set the item counter to capacity | heads to the nearer bin, `DUMP`, resumes |
| `move_bin` / new path | load another path JSON (build a new `PatrolController` + `PatrolMission`) | follows the new path; no retraining |

## 7. Sim-to-real gaps: check these first

1. **Lidar angle convention.** Put a box 1 m straight ahead, then 1 m to the left: the minimum must land in
   sector 18 (ahead), then in sector 27 (left, +90°). If it's mirrored, negate your angles; if it's off by 90° or
   180°, fix the lidar's mount orientation or add an offset.
2. **Sign of lateral error.** Stand the robot 0.2 m left of the path: `obs[36]` must be positive (+0.4).
3. **Yaw-rate sign.** A positive `wz` must turn the robot left (CCW seen from above), and the measured `w`
   must agree with it.
4. **Velocity limits of your locomotion policy.** If it can't track 1.0 m/s or 1.5 rad/s, scale the command
   down; don't change the obs. Tell us the real limits and we'll retrain with them.
5. **Scan height.** Objects lower than the scan plane are missed; check the lidar sees your test objects.
6. **Pose source.** Use the true base pose for now. Odometry drift appears as lateral error.

## 8. Merging with the rust pipeline later

Rust becomes one more disturbance type. While patrolling, the wrist or front camera runs
`detection.export.OnnxRustClassifier` on pipe crops. A confident rust hit makes the mission stop and hand
control to `policy/orchestrator.py` (detect → spray → soak → scrub → paint → verify, per
`integration/INTERFACE.md`). When the orchestrator reports DONE or ESCALATE, the patrol resumes `FOLLOW`
from the same path index. The tool swap (gripper vs brush) is the open question: either a second arm
tool or a quick-change mount.
