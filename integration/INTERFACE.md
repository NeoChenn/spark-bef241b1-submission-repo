# Sim ⇄ policy interface contract

Owner: the orchestrator side (`policy/orchestrator.py`, `detection/localize.py`). The sim team implements
**one class satisfying `policy.interfaces.SimBridge` + `Perception` + `ArmController`** inside the Isaac Sim 6.2
process (`rig/sim/main.py` loop). Everything else is already written and tested against a CPU mock
(`.venv/bin/python -m policy.demo_mock`, which uses the same geometry as `sim/output/rig_deck_meta.json`).

```
Isaac Sim (rig/sim/main.py)                         policy (same process, pure python/numpy/torch)
  wrist / repair_closeup camera ─ rgb annotator ──► Perception.observe() ─► RustLocalizer (ONNX, CPU)
  camera prim xform + focal ───────────────────────► Perception.camera()  ─► project_mask_to_grid
  contact sensor on brush_link ────────────────────► ArmController.current_force()
  diff-IK on brush_link ◄── apply_ee_delta / move_to ◄─────────────── Orchestrator state machine
  rust overlay texture  ◄── RustState.to_texture(overlay=True) ◄──── RustState (spray / soak / abrade)
  JPEG frame header "hud" ◄── bridge.publish_overlay(lines) ◄─────── every DecisionRecord
```

## 1. Frames and units

* Metres, seconds (sim time), Z up, deck at z = 0 (same as `rig_deck_meta.json`).
* **Main pipe**: axis along world +X, centre-line at y = 0, z = 1.1, radius 0.15.
* **Pipe frame / angle**: `theta` is measured from **+Y toward +Z** (exactly the `center_angle` in
  `sim/build_rig_deck.py::damage_patch`), so `PipeGeometry(axis=+X, ref=+Y)`. Seed 7 damage: `theta = atan2(0.7836, 0.6212) = 0.90 rad`.
* **Work segment**: the rust grid covers a 0.8 m segment of the main pipe **centred on the damage x**
  (`origin = (damage.x, 0, 1.1)`, `z ∈ [-0.4, 0.4]` along +X). Grid = `RustState(num_envs=1, H=40, W=64, length=0.8)`,
  `radius[:] = 0.15`. Rows = z (axial), cols = theta (periodic). `policy.demo_mock.scene_geometry()` builds it from the meta JSON.
* **Cameras**: our `CameraModel` is OpenCV (x right, y down, +z forward). Convert a USD camera with
  `policy.interfaces.camera_from_usd(np.array(xform.ComputeLocalToWorldTransform(t)), focal, h_aperture, W, H)`
  (fx = focal / h_aperture · W, square pixels, principal point at the centre).

## 2. Prims and attributes we need

| Prim path | What | Status |
|---|---|---|
| `/World/Pipes/main_pipe` | existing tube. **Add** custom attrs `rig:axis = (1,0,0)`, `rig:radius = 0.15`, `rig:segmentOrigin = (damage.x, 0, 1.1)`, `rig:segmentLength = 0.8` so `bridge.pipe()` reads geometry instead of hard-coding it | add attrs |
| `/World/Pipes/rust_overlay` | **new** cylindrical sleeve mesh, radius 0.153 (just outside the pipe, like `damage_patch`), length 0.8, centred on the segment origin. UVs: `u = theta / 2π`, `v = (z + 0.4) / 0.8` (matches `RustState.surface_to_uv`). Material `/World/Looks/RustOverlay`: diffuse + opacity from a **dynamic texture** fed with `RustState.to_texture(overlay=True)[0]` (RGBA uint8, 40×64; rows = v, cols = u). Update ≥ 2 Hz. Use `omni.ui.DynamicTextureProvider("rust_state").set_bytes_data(...)` and texture path `dynamic://rust_state`. | new |
| `/World/Pipes/damage_patch` | the static seed patch. Either hide it once `rust_overlay` exists, or initialise `RustState.rust` from its footprint (centre `theta`, approx radius in metres) so they agree. | hide or keep in sync |
| `/World/Robots/scrubber_dog` | URDF import of `/assets/robot_dog.urdf` (fixed-base fallback: the existing Ridgeback Franka `/World/Robots/fitter`) | new |
| `…/scrubber_dog/base_link` | quadruped trunk; `approach()` places it (teleport is fine for the demo) next to the pipe, within reach of the segment | URDF |
| `…/arm_joint_1 … arm_joint_6` | arm revolute joints (names flexible; list them in `bridge.ARM_JOINTS`) | URDF |
| `…/brush_link` | **tool frame**: origin at bristle tip, +Z along the brush axis (pointing into the surface). Rigid body with a **contact sensor** (`isaacsim.sensors.physics` ContactSensor or Isaac Lab `ContactSensor`) → `current_force()` = normal-force magnitude (N) | URDF + sensor |
| `…/nozzle_link` | spray nozzle frame (+Z = spray direction). If absent we use `brush_link` with 0.15 m standoff | optional |
| `…/wrist_cam_link/wrist_camera` | USD camera on the arm, 35 mm focal, 20.955 mm aperture, 640×360 render product + `rgb` annotator — **this is the localizer input** | new |
| `/World/Cameras/repair_closeup` | fallback localizer input. **Problem (seed 7):** its eye `(dx+0.7, dy-1.2, dz+0.7)` is on the −Y side but the damage faces +Y/+Z: the ray to the damage centre meets the surface at cos = −0.13, so the damage is on the far side of the pipe. Put the eye on the damage-normal side instead, e.g. `eye = damage + 0.85·normal + (0.15, 0, 0)`, `target = damage` (what the mock uses). | fix eye |
| `/World/Cameras/hero` | video camera for the demo; burn the HUD into this stream | existing |

URDF on Modal: the `rig-assets` volume is mounted at `/assets`, so upload with
`modal volume put rig-assets robot_dog.urdf /robot_dog.urdf` (meshes under `/meshes/...`, with URDF paths
relative to the URDF file). The Isaac side reads `/assets/robot_dog.urdf` (`infra/common.py: ASSET_VOL, ASSET_DIR`).

## 3. Methods to implement (see `policy/interfaces.py` for signatures)

**Perception**
* `observe() -> np.ndarray | None`: `np.asarray(ann.get_data())[..., :3]` as uint8 H×W×3 from the wrist camera
  annotator (same as `rig/sim/main.py`). Return `None` if there is no frame yet (`data.size == 0`) or the mean is ≈ 0
  (shader warm-up). A black or flat frame (std < 6) counts as occluded → orchestrator `HOLD`.
* `camera() -> CameraModel`: `camera_from_usd(...)` of the camera that produced the last frame.
* `request_view(i) -> bool`: move the camera to alternative viewpoint i ∈ {1, 2} (re-pose the arm wrist, or swap
  to another camera prim). Suggested: rotate ±0.35 rad about the pipe axis and ±0.12 m along it, still looking at the
  damage. Return False if it can't be reached.

**ArmController** (all targets are world-frame, built from `PipeGeometry.surface_point`)
* `move_to(waypoints) -> list[bool]`: for each `Waypoint(position, approach, standoff, flow_rate)` drive
  `brush_link` (or `nozzle_link`) to `position` with tool +Z = `approach`, via Isaac Lab `DifferentialIKController`
  (`command_type="pose"`, `ik_method="dls"`). Return reached flags (pos error < 1 cm). Spray: 0.15 m standoff; paint:
  0.05 m (brush in contact).
* `apply_ee_delta(a)`: **scrub action** `a = [d_theta (rad), d_z (m), press ∈ 0..1]` in pipe frame. Implementation:
  `theta, z += clip(d_theta, ±0.15), clip(d_z, ±0.03)` → target `p, n = pipe.surface_point(theta, z)`, tool tip at
  `p − 0.005·press·n` (pushes into the surface; admittance gives ≈ 25·press N), orientation `approach = −n`, one diff-IK
  step. Then apply abrasion: `rust.abrade(rust.surface_to_uv(θ, z), 0.03, F_contact, slide_speed=0.3, dt)` where
  `F_contact` is the contact sensor reading (rotary brush → constant 0.3 m/s slide). The orchestrator multiplies `press`
  by a safety scale after every force back-off.
* `ee_theta_z()`, `current_force()` (contact sensor, N), `back_off(0.05)` (retract along +n).

**SimBridge**
* `sim_time()`, `advance(dt)`: step physics/render for `dt` of sim time with the arm holding pose (the orchestrator
  calls `RustState.step_soak(dt)` itself — **don't** step soak in the sim too).
* `pipe() -> PipeGeometry` (read the `rig:*` attrs; re-read each frame so "move pipe" works),
  `patrol_poses()`, `go_to_patrol_pose(p)`, `approach(theta, z, standoff)`.
* `person_distance() -> float | None` (hook: distance from base_link to the nearest `/World/People/*` prim).
* `scrub_obs() -> dict`: extra obs for the PPO policy, e.g. `{"policy": obs_tensor}` from `policy/envs/scrub_env.py`.
* `publish_overlay(lines)`: called on every decision with 4 short strings. Easiest: keep the latest `lines` and
  add them to the frame header — `sender.send({..., "hud": lines}, jpeg)` — and draw them in `rig/web/static`; or burn
  them in before JPEG encode with `policy.orchestrator.draw_hud(rgb, lines)`.

Control rates: physics 60–120 Hz; the orchestrator makes a scrub decision every 0.1 s sim time (10 Hz,
`OrchestratorConfig.scrub_dt`), re-detects every 5 s during the 30 s soak and every 40 scrub steps.

## 4. Mid-run change hooks for the video (not announced to the orchestrator)

| Hook | Implementation | Expected on-screen reaction |
|---|---|---|
| `spawn_patch(theta, z, size)` | `RustState.add_patch(..., sprayed=False)` (or `MockRustWorld.inject_patch`) during SOAK or SCRUB | `SOAK→SPRAY "RE-PLAN: new rust #1 …"`, soak extended 30 s; mid-scrub: `SCRUB→SPRAY` |
| `glare(on)` | raise the sun/dome intensity or add a spot light on the damage for view 0 | `DETECT→SECOND_VIEW` (unsure), then confident |
| `occlude_camera(s)` | black frames / spray mist on lens for s seconds | `→HOLD`, then resume the same state |
| `force_spike()` | a weld bead / collider on the scrub path | `SCRUB→BACK_OFF→SCRUB`, press scaled 0.7 |
| `person_at(2.4 m)` | move a `/World/People/worker` prim near the robot | `→PAUSE_PERSON`, resume when > 3 m |
| `move_pipe(dx)` | translate the segment origin attr + overlay | grid re-projected from the new `pipe()` |
| stubborn rust | `RustState.abrasion_coef[:] = 0.1` | 3 failed VERIFYs → `ESCALATE_TO_HUMAN` with residual cells |

## 5. Outputs

* `runs/decisions.jsonl`: one line per transition:
  `{"t", "from", "to", "reason", "evidence": {"detector_conf", "rust_fraction", "patch_ids", "force", …}}`.
* HUD (`Orchestrator.overlay_text()`): `STATE`, rust-left %, detector confidence, patch ids, force, last reason.
* Detector checkpoint: `/ckpt/detector/best/{model.onnx, meta.json}` on volume `rust-ckpt`. Load with
  `RustLocalizer.from_checkpoint("/ckpt/detector/best", patch=96, stride=48)` (patch is resized to `meta.input_size`).
  meta keys read: `input_size` (int or `[3,H,W]`), `mean`, `std`, `threshold`, `temperature`, `unsure_band`. ONNX input
  NCHW float32 normalised, output `(N, 1)` rust logit.

## 6. Gas-leak job (same robot, same plan engine)

Built on CPU and tested against the mock; **nothing here has run inside Isaac Sim yet**.

* **Plan engine:** `policy/leak/job.py` `RigOrchestrator` subclasses `Orchestrator`. With `leak=LeakJob(LeakLocalizer())`
  it adds `GAS_CHECK, LEAK_APPROACH, LEAK_INSPECT, LEAK_SECOND_VIEW, RETORQUE, LEAK_VERIFY, GAS_RETREAT`. Gas is checked
  before every step and every scrub step: at or above 0.20 LEL (or a failed sensor) means `GAS_RETREAT` from any state;
  above 0.05 during rust work pauses it, fixes or escalates the leak, then resumes. With `leak=None` it is the rust robot.
* **Perception (trained crack model, like rust):** `detection/leak.py` `LeakLocalizer.read(rgb, camera, flange)`, same flow as rust
  (evidence -> calibrated probability -> threshold -> unsure band -> another view). Stripe angles come from pixels (red
  stripe on each bolt head against its expected direction). Cracks: a MobileNetV4 + LoRA window classifier trained
  with the team's `detection/train.py` (`detection/train_leak.py`), shipped at `detection/checkpoints/crack_v1` and
  loaded by `LeakLocalizer.default()`; the classical line check is the fallback. If the view is too poor to read the
  stripes cleanly (`view_quality` < 0.85), "no crack" becomes "unsure", so a dirty lens never yields a silent
  all-clear. Results: `detection/LEAK_RESULTS.md`; re-check with `python -m detection.eval_leak_sim`.
* **Scene:** `python sim/build_leak_flanges.py` writes `sim/output/leak_flanges.usda` (a bolted flange with torque
  stripes beside each of the 6 valves), `full_rig_leak.usda` (that layer over `full_rig.usda`: **open this one**) and
  `leak_flanges_meta.json`. Placement is a guess from the valve positions: check it visually.
  The cinematic scene (`sim/output/cinematic.usda`, what `run_demo.py`, `render_sequence.py` and the stream open)
  also loads `leak_flanges.usda`, so the flanges appear in the team video and stream.

| Prim | What |
|---|---|
| `/World/Leak/F<n>` | flange Xform beside valve n; local +X = face normal, +Z = bolt 0; attrs `rig:flangeId`, `rig:crackSeverity` |
| `…/bolt_<k>` | bolt Xform; `xformOp:rotateX:loosen` (degrees) turns the head and its stripe |
| `…/mark_<k>` | the fixed stripe on the face |
| `…/crack` | hidden until a crack exists |

**Bridge methods** (`integration/leak_bridge.py` `LeakBridgeMixin`, mix into the Isaac bridge class): `gas_reading()`,
`flanges()`, `flange(id)`, `nearest_flange()`, `approach_flange(id)` (walks there, **checking gas every step**, and
stops early at the limit), `request_flange_view(i)`, `retreat(m)`, `return_to_work()`, and `torque_bolt(id, k)` for
`ArmController.torque(id, k)` (socket onto the head via `move_to`, then tighten in `LeakWorld`, then
`apply_leak_state(stage, world, prims)`). The host provides `robot_position()`, `move_base_to(xyz, facing)`,
`set_camera_pose(eye, target)`, `advance(dt)`, `work_position()`, `stage`, `leak_world`, `leak_prims`.

Isaac implementation notes (CPU-tested with usd-core in `tests/test_leak_isaac.py` and `tests/test_deck_bridge.py`;
not yet run in Isaac Sim): `load_leak_world(meta)` gives the `LeakWorld` and prim index. `gas_reading()` samples the
`LeakWorld` gas field at `gas_sensor_position()` (robot position + `gas_sensor_offset`); set `gas_override` to a
number for a fixed reading or to `float("nan")` for a failed sensor (the job then retreats). The host's `advance(dt)`
should call `leak_advance(dt)`, which steps the gas model and pushes bolt angles and cracks to the stage.
`retreat(m)` and `return_to_work()` walk with `move_base_to` (the mock jumps). `integration/rig_bridge_example.py`
shows the whole composition.

**Wrist camera:** 640x360 works from **1.1 m** (the whole flange fits; every bolt read; angle error at most 0.019 rad).
From 1.0 m the flange clips the frame.

**Hooks for the video** (call on `LeakWorld`, then `apply_leak_state`): `loosen(flange, bolt, rad)`,
`crack(flange, severity)`, `gas_spike(flange)`. The existing `occlude_camera` and `person_at` hooks work unchanged.

| Hook | Expected on-screen reaction |
|---|---|
| `loosen` while scrubbing | `SCRUB -> GAS_CHECK`, fix the bolt, `LEAK_VERIFY -> SCRUB` (resumes) |
| `crack` | `LEAK_INSPECT -> ESCALATE_TO_HUMAN`, critical alert, robot backs off, no tightening |
| `gas_spike` | `GAS_RETREAT -> ESCALATE_TO_HUMAN`, critical alert |
| dirty lens (`dust`) | two extra views, then `ESCALATE_TO_HUMAN` (unsure, does not guess) |

**Outputs:** the same `runs/decisions.jsonl`. Leak evidence adds `gas`, `flange`, `leak_diagnosis`, `leak_reading`, and
`alert` (`severity`, `kind`, `message`, `recommended_action`, `requires_ack`). The HUD keeps its 4 lines; leak states
show gas, diagnosis and bolt readings, and `GAS_RETREAT` shows in the alert colour.

Try it on CPU: `python -m policy.leak.mock --leak loose` (also `crack`, `big`, `none`, `dust`, `--gas-during-scrub`).

## 7. Deck sweep: debris pickup and a camera person check (same robot, same plan engine)

Two more trained detectors (`detection/rig_vision.py`), made the same way as the rust and crack models: real photos
plus sim-domain crops, MobileNetV4 + LoRA through the team's `detection/train.py`, calibrated, ONNX in
`detection/checkpoints/{debris_v1,person_v1}`. Results and limits: `detection/RIG_VISION_RESULTS.md`.

* **Debris** (`DEBRIS_CHECK -> DEBRIS_PICKUP -> DEBRIS_VERIFY`, `DEBRIS_SECOND_VIEW` when unsure): after the gas check
  says normal at a patrol stop, the robot looks down the deck, picks up loose objects one at a time and checks they
  are gone, then goes on to the rust check. If it cannot pick something up or cannot tell, it reports it (`warning`)
  and carries on: debris is not an emergency.
* **People:** the team's 3 m pause rule still uses `person_distance()`. The body camera now also counts: a person the
  model sees, at a distance under 3 m from the depth camera, pauses the robot, and it resumes only when both say
  clear. The camera can add pauses; it never removes one.

| Bridge method | Returns | Isaac notes |
|---|---|---|
| `deck_frame()` | `(rgb HxWx3 uint8, CameraModel)` | body or wrist camera looking down the deck ahead; posed `T_world_cam` (world z up) |
| `deck_reference()` | `rgb` | a clean picture of the same stop and view, stored when the deck was known clean (first patrol or commissioning). Only changes against it are classified, so fixed equipment is never picked up |
| `request_deck_view(i)` | `bool` | move sideways / closer for another look |
| `pick_debris(u, v)` | `bool` | project pixel (u, v) onto the deck with the camera model, grasp there, drop in a bin |
| `body_frame()` | `(rgb, CameraModel)` | the forward body camera |
| `body_range()` (optional) | `(H, W) float` metres | depth image as ground distance per pixel; without it the distance comes from the box and is cruder |

Try it on CPU: `python -m policy.leak.mock --leak none --debris litter --person-at 2` (also `--debris can`, `--no-deck`).

**Isaac implementation** (`integration/deck_bridge.py` `DeckBridgeMixin`; CPU-tested in `tests/test_deck_bridge.py`;
not yet run in Isaac Sim). Composition example with the leak half: `integration/rig_bridge_example.py`.

* Cameras: `deck_camera_path` and `body_camera_path` (default `/World/Robots/scrubber_dog/base_link/body_camera`, a
  guess: set them to the real prims). `CameraModel` comes from the prim's world transform, focal length and
  horizontal aperture (`usd_camera_model`, via `camera_from_usd`). Frames come from a replicator render product per
  camera with the `rgb` and `distance_to_image_plane` annotators (`grab_camera`, the only Isaac call), at
  `deck_resolution` (640x360).
* `body_range()`: `depth_to_ground_range(depth, camera)`, the horizontal distance to each pixel's 3D point
  (inf where there is no hit). It uses the measured depth, so it works at any deck height (the full rig's top deck
  is at z = 42.5, not 0).
* `deck_reference()`: references are kept per (patrol stop, view). The host sets `patrol_stop` in
  `go_to_patrol_pose`. If a stop has no reference yet, the current frame is stored and used: the first visit is
  taken as clean. For the demo, do a clean patrol first (or `capture_deck_reference()` at each stop) before
  spawning debris, or `load_references(dir)` from a run saved with `save_references(dir)` (PNG + `index.json`).
* `request_deck_view(i)`: walks to the view-0 pose plus `deck_view_offsets[i]` (forward, left) in metres, same heading.
* `pick_debris(u, v)`: pixel -> deck point (ray meets z = `deck_z`; use `deck_z_from_meta`) -> walks so the point is
  `pick_standoff` (0.26 m, with the Go2 crouched) ahead if it is out of reach -> robot base frame -> `pick_skill(target_xy_base, yaw)`.
  **`pick_skill` must be plugged in** (the RL pick policy); without it the call raises `NotImplementedError`. Yaw
  is 0 unless `pick_yaw()` is overridden. On success, the nearest registered debris prim (within 0.15 m) is hidden.
* Demo debris: `spawn_debris(name, (x, y))` / `spawn_debris_can(stage, path, (x, y, z))` adds a red can lying on the
  deck under `/World/Debris/` (`physics=True` adds a rigid body and collider).

Still needed from the team's bridge: `stage`, the real camera prim paths (and the wrist camera for flanges),
`robot_position()` / `robot_yaw()` from the articulation, `move_base_to(xyz, facing)` (locomotion step),
`set_camera_pose(eye, target)` (arm IK), `advance(dt)` (physics + render; call `leak_advance`), `work_position()`,
`patrol_stop` set in `go_to_patrol_pose`, `arm.move_to`, and the pick skill.
