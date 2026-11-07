# prompt_daphe_1: game-style HUD for the rust-repair demo

Hand this file to a Claude session working in this repo. Everything below the line is the prompt.

Note: the sim bridge, Gemini verifier and repair runner it mentions (`integration/isaac_bridge.py`,
`integration/run_repair.py`, `policy/gemini_verifier.py`) are on branch `sim/bridge` (PR #6) until that is merged.
Check out or merge that branch first.

---

Build a game-style control and stats overlay for our hackathon robot demo. It's a web page that sits on top of a live Isaac Sim video stream.

## Project context

- Repo: NeoChenn/spark-bef241b1-submission-repo. Extend the existing viewer in `rig/web/`:
  - `server.py` is FastAPI. It has `GET /`, `GET /api/health`, `GET /api/frame.jpg` and a WebSocket at `/ws/video` that sends JPEG frames, each with a JSON header.
  - `static/index.html` and `static/viewer.js` draw the stream.
  - Keep the stream path as it is. Add the UI around it.
- Scenario: a Unitree Go2 quadruped with a lift mast and a UR3 arm patrols an offshore oil rig. It finds rust on a pipe, then sprays remover, waits 30 s for it to soak, scrubs, checks the result and paints.
- A state machine (`policy/orchestrator.py`) drives it. Its states are PATROL, DETECT, SECOND_VIEW, APPROACH, SPRAY, SOAK, SCRUB, VERIFY, PAINT, VERIFY_PAINT, DONE, ESCALATE_TO_HUMAN, BACK_OFF, HOLD and PAUSE_PERSON.
- Every transition writes a record shaped like this:
  `{"t", "from", "to", "reason", "evidence": {"detector_conf", "rust_fraction", "patch_ids", "force", "gemini": {"verdict", "confidence", "reason", "latency_ms", "points"}}}`
- `Orchestrator.overlay_text()` returns 4–5 HUD lines.
- Gemini (`gemini-robotics-er-2-preview`) gives a second opinion when the on-board detector is unsure, and it can veto painting if it still sees rust.
- Mid-run "unscripted change" hooks already exist on the sim bridge (`integration/isaac_bridge.py`):
  - `spawn_patch(theta, z)`
  - `set_glare(on)`
  - `occlude_camera(on)`
  - `force_spike()`
  - `person_at(distance)`

## Data to add

- In the frame header the sim sends, add:
  - `"hud"`: the lines from `overlay_text()`
  - `"state"`
  - `"events"`: decision records since the last frame
  - `"rust_grid"`: a 40x64 array of floats in 0–1, rows = position along the pipe, columns = angle around it
  - `"force_n"`
  - `"detector_conf"`
  - `"gemini"`: the last verdict
- Add `POST /api/event {"name": "...", "args": {...}}`. It queues one of the hooks above, and the sim loop runs it on its next step.

## Look

- Aim for a premium game-client HUD: minimal, technical, clean and lean. Not a sci-fi cliché, and not a copy of any existing game or client's UI or branding.
- Dark translucent panels (`rgba(10, 12, 16, 0.55)` with a 12 px backdrop blur), 10 px corner radius and a 1 px hairline border at `rgba(255, 255, 255, 0.08)`.
- One accent colour, a clean cyan-teal. Amber for warnings, red only for safety stops.
- Type: Inter for labels in small caps at 11–12 px with wide letter-spacing, and JetBrains Mono for numbers. Use tabular figures so numbers don't jitter.
- Everything sits at the edges of the screen. The middle stays clear so the video is the hero. No drop shadows on text, no gradients on panels, and no icon clutter.
- Panels can be toggled like game "mods": a small module list in one corner turns each panel on or off and remembers the choice. Default hotkeys: H HUD, M map, G Gemini, L log, C cinematic mode, which hides everything.
- Motion is subtle: 150 ms fades, numbers that count up and down, and a thin progress ring for the soak timer. Respect `prefers-reduced-motion`.

## Panels

1. **Top-left status pill:** the current STATE as a coloured dot plus its name, the sim time, the FPS, and an EDGE / CLOUD badge showing whether the last decision used Gemini.
2. **Top-right mission stepper:** PATROL → DETECT → SPRAY → SOAK → SCRUB → VERIFY → PAINT → DONE. Highlight the current step. Safety states (BACK_OFF, HOLD, PAUSE_PERSON, ESCALATE) flash a small amber or red tag beside it.
3. **Bottom-left vitals,** as small sparklines over the last 30 s:
   - brush force in newtons, with a dashed 40 N limit line;
   - rust remaining in %;
   - detector confidence.
4. **Bottom-right "pipe map":** the 40x64 rust grid unrolled as a small heatmap. Show sprayed cells tinted, painted cells grey-green, and the brush position as a crosshair.
5. **Gemini card,** only shown when Gemini was asked: verdict chip (RUST / CLEAN / CAN'T TELL), confidence bar, latency in ms, a one-line reason, and its points drawn as small rings on a thumbnail of the inspection frame.
6. **Decision log drawer:** the last 6 transitions, newest first, as "t  FROM → TO  reason", in mono font.
7. **"Unscripted change" panel,** collapsed by default: five buttons that call `POST /api/event` (new rust, glare, lens mist, weld bead, worker walks up). This is how a judge triggers the live change during the demo.

## Camera and demo flow

- Start with a cinematic wide shot of the whole rig (`sim/cinematic.py` has night, golden and storm presets and a camera sequence). Show a minimal title card: project name, one-line problem statement, and "Press Space to begin".
- On start, the camera eases from the wide shot down to the deck and the pipe work zone over about 4 s. Then the HUD panels fade in one by one.
- Hotkeys 1, 2 and 3 switch between the wide rig, hero deck and inspection close-up cameras.

The demo video must be one take of under a minute, with perception on screen and one unscripted change. Target this flow:

| Time | Shot | What judges see |
|---|---|---|
| 0–6 s | The night storm preset, slowly orbiting the whole rig, then title card | Scale and atmosphere |
| 6–12 s | Camera dives down to the deck as the HUD fades in. The dog trots to the pipe. | Mission stepper and the PATROL state |
| 12–20 s | The inspection close-up appears picture-in-picture, with the rust outlined in red and the pipe map filling in | Perception on screen |
| 20–28 s | The robot sprays, and the soak ring starts. A judge clicks "new rust", and the stepper jumps back to SPRAY with a RE-PLAN tag. | The unscripted change |
| 28–40 s | The robot scrubs. The force sparkline climbs, the weld bead pushes it over 40 N, it shows BACK_OFF and resumes. | The critical failure mode handled on screen |
| 40–50 s | VERIFY. The Gemini card pops up with a verdict, the robot scrubs again if needed, and the EDGE/CLOUD badge flips. | Use of Gemini |
| 50–60 s | The robot paints. The pipe map turns grey-green and shows DONE, then the camera pulls back to the rig. | A clean ending |

Known gap: the cinematic rig (`sim/output/cinematic.usda`) and the repair deck (`sim/output/rig_deck.usda`) are separate scenes today, so the wide-to-pipe zoom is a cut unless the deck is placed onto the full rig.

## Deliver

- Plain HTML, CSS and JS in `rig/web/static` (no build step).
- A mock mode (`?mock=1`) that replays `runs/repair/decisions.jsonl` and a recorded rust grid, so the UI can be designed without Isaac Sim running.
- Test at 1920x1080 and 1280x720.
- Keep it under about 600 lines of JS.
