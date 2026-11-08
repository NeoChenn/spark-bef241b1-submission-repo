# Sim bridge (Isaac Sim 6.2)

## What it is

`integration/isaac_bridge.py::IsaacRigBridge` is one object that satisfies `policy.interfaces.SimBridge`,
`Perception` and `ArmController` (the contract in `integration/INTERFACE.md`) on the rig deck scene
`sim/output/rig_deck.usda`. `integration/run_repair.py` opens the deck, adds the Go2 + lift + UR3 robot, the rust
overlay and the demo props (`integration/repair_scene.py`), and runs the real `policy.orchestrator.Orchestrator`
against the bridge. Arm maths (FK, IK, contact spring) is plain numpy in `integration/kinematics.py`.

## How to run (Windows)

One-time: install torch into a separate folder that `run_repair.py` adds to `sys.path`
(override the folder with the `ISAAC_DEPS` environment variable):

```bat
C:\isaacsim\python.bat -m pip install --target %USERPROFILE%\.isaac-deps --index-url https://download.pytorch.org/whl/cpu torch
```

Only if `sim/output/robot/` is missing, convert the URDF (`sim/assets/go2_ur3/go2_ur3_lift.urdf`) to USD:

```bat
C:\isaacsim\python.bat integration\import_robot.py
```

Run the repair loop (headless):

```bat
C:\isaacsim\python.bat integration\run_repair.py [--detector DIR] [--glare] [--no-events]
```

* `--detector DIR`: folder with `model.onnx` + `meta.json` (see below). Default: colour stand-in.
* `--glare`: turns on a glare light for inspection view 0 only, so the orchestrator has a reason to take a second view.
* `--no-events`: skip the scheduled mid-run scene changes.
* Other flags: `--out` (default `runs/repair`), `--max-steps 400`, `--subframes 4`, `--fps 12`, `--hero-size 1280x720`.

Outputs in `runs/repair/`: `decisions.jsonl` (one line per state transition), `repair.mp4` (hero camera with the
HUD and a detector inset: inspection frame with the rust mask in red), `frames/*.jpg` (the video frames) and
`inspection_last.png`.

Tests (CPU only, no Isaac; needs numpy + pytest), from the repo root:

```bat
python -m pytest tests/test_kinematics.py
```

## What is real and what is approximated

* **Perception is real.** `observe()` returns the RTX render of the inspection camera (rgb annotator, 640x360).
  Pixels whose ray misses the pipe segment are set to neutral grey (118), following the HANDOFF advice to classify
  the pipe surface and not wide views (the rusty deck grate reads as rust). If the pipe pixels themselves are flat or
  dark (lens misted or blocked), the raw frame is returned instead, so the orchestrator's occlusion check fires HOLD.
* **The arm clears the view before every look**: it swings to the stowed pose (lift unchanged) for the inspection
  frame, then swings back before the next tool command, so it never hides the rust it is checking.
* **The robot is posed kinematically, not simulated by PhysX.** The imported USD is referenced with its Physics
  variant set to `none`; each link transform is written from FK of the URDF. The brush is moved with damped
  least-squares IK on `brush_tip` (position + brush axis; roll about the axis is free).
* **Legs hold the standing pose** from `ScrubLiftEnvCfg`. Between stations the base glides along a smooth path
  while the legs play a trot animation. The lift goes to 0.42 m and the arm unfolds from a stowed pose at the pipe.
* **Contact force is analytic**: the same spring the scrub policy was trained with, `F = 2500 N/m x penetration`
  of `brush_tip` into the pipe surface. A weld bead raises the local surface, so the force rises over it.
  There is no PhysX ContactSensor.
* **Inspection camera** is the fixed `/World/Cameras/repair_closeup` fallback, re-posed to the damage-normal side
  (`eye = damage + 0.85 n + 0.15 axis`) as INTERFACE.md suggests. `request_view(1|2)` moves it to +-0.35 rad and
  +-0.12 m alternatives. There is no wrist camera.
* **Rust state** is a `policy.demo_mock.MockRustWorld` grid (40 x 64). It is shown as three thin sleeves on the
  pipe (wet remover, rust texture, primer) whose quads switch on per cell, so the render always matches the grid.
  The static `damage_patch` is hidden.

## Stand-ins until the trained models are shared

* **Detector**: without `--detector`, the localizer uses `detection.localize.fake_colour_classifier`
  (printed at start-up as `DETECTOR colour stand-in`).
* **Scrub policy**: `policy.demo_mock.GreedyScrub`. The trained `policy.onnx` expects the 327-D Isaac Lab observation;
  `scrub_obs()` returns `{}`, so it is not wired.
* Both the detector checkpoint and the scrub policy are on the Modal volume `rust-ckpt` (profile `jacobyuan99`),
  not in git.

To use the real detector, download `model.onnx` and `meta.json` from `rust-ckpt:/detector/best` into a folder
(for example `runs\detector\best`) and pass it:

```bat
C:\isaacsim\python.bat integration\run_repair.py --detector runs\detector\best
```

It is loaded with `RustLocalizer.from_checkpoint(DIR, patch=96, stride=48)`.

## Gemini second opinion

`policy/gemini_verifier.py` asks Gemini (default model `gemini-robotics-er-2-preview`, override with
`GEMINI_MODEL`) about the inspection frame at the two moments a human supervisor would otherwise decide:

* **DETECT still ambiguous after the extra views**: instead of escalating straight away, Gemini says rust / clean /
  cannot tell, with a confidence, a reason and up to 5 rust points. Points are projected onto the pipe grid to pick
  which candidate patches to repair.
* **VERIFY about to paint**: Gemini must not see rust left. If it does (confidence >= 0.6), the robot scrubs again
  (at most 2 vetoes per repair; after the last scrub cycle it escalates with Gemini's evidence).

The verdict, confidence and latency go into `decisions.jsonl` (`evidence.gemini`), onto a fifth HUD line, and into
`runs/repair/gemini_calls.json`. The fast loop (detector, localizer, scrub) never waits on Gemini; with no key, no
network or a bad reply, the orchestrator does exactly what it did without Gemini.

Setup: put `GEMINI_API_KEY=...` in a `.env` file at the repo root (gitignored) or the environment, then check one
frame before a full run:

```bat
C:\isaacsim\python.bat -m policy.gemini_verifier runs\repair\inspection_last.png --question clean
```

`run_repair.py` turns Gemini on automatically when a key is found (`--no-gemini` to skip). The REST call uses
`models/{model}:generateContent`; set `GEMINI_ENDPOINT=interactions` to use the newer `/v1beta/interactions` API.
Tests (no network): `tests/test_gemini_verifier.py`.

## Mid-run hooks (bridge methods)

| Method | What it does in the scene |
|---|---|
| `spawn_patch(theta, z, size, level)` | adds rust to the grid (`world.inject_patch`); the overlay shows it on the next frame |
| `set_glare(on)` | sphere light near view 0 at 4e5 intensity (0 when off) |
| `occlude_camera(on)` | shows / hides a dark card 3 cm in front of the inspection lens (lens mist) |
| `force_spike(height, radius)` | adds a weld bead prim under the brush and a bump in the spring surface (default 10 mm: +25 N, trips the 40 N limit once) |
| `person_at(distance)` | moves the high-vis worker prim `/World/People/worker` near the base; `None` removes it |
| `storm_level(level)` | 0-1 storm (`policy/storm.py` `Weather`): gusts add `0.9 x gust_n` to the brush force, rain drops collect on the inspection lens and haze the frame (`Lens`), wet pipe scales abrasion by `1 - 0.35 x level`; `weather()` reports gust/rain/deck, `wipe_lens()` clears the drops; 0 ends the storm |
| `set_weather(level, wind_dir, gust)` | same control as `storm_level`, shared with the fleet/navigation weather event: `level` is `calm`, `wind` (gusts, no rain), `rain`, `fog` (haze, no gusts), `storm`, or a 0-1 number; `wind_dir` in degrees sets the gust direction, `gust` (N) scales the gust peaks |

With events on, `run_repair.py` schedules: a new patch 12 s after SPRAY, a weld bead 4 s into SCRUB, and on
SCRUB -> VERIFY lens mist for 1.5 s plus a worker at 2.4 m from 0.5 s to 2.5 s.

Not implemented as bridge methods: `move_pipe` (the `rig:*` attrs are re-read in `pipe()`, but nothing moves the
overlay), and stubborn rust (set `world.abrasion_coef` directly).

## Deviations from INTERFACE.md

* The main pipe prim is `/World/Pipes/main_pipe_073`, found by the `main_pipe` prefix, not `/World/Pipes/main_pipe`.
* A `rig:ref = (0, 1, 0)` attribute is added next to `rig:axis`, `rig:radius`, `rig:segmentOrigin`, `rig:segmentLength`.
* No Isaac Lab ContactSensor or DifferentialIKController: force and IK come from `kinematics.py` (see above).
* The rust overlay is face-switched mesh sleeves, not a `dynamic://rust_state` texture.
* The hero camera is `/World/Cameras/repair_hero`; the HUD is burned into its frames with `draw_hud`.
* The URDF's orange brush head is recoloured to steel in this scene, because a colour-based detector reads it as rust.
* The robot is `/World/Robots/scrubber_dog`; the existing `fitter` and `welder` robots are deactivated.
