# Spotlight variant

Open `http://localhost:8101/static/ui-lab/spotlight/index.html?mock=1&autostart=1`. Add `&t=20.6` to start at BACK OFF. Without `mock`, the page uses `/ws/video` (through `/static/viewer.js`) and `/ws/state`.

## Design decisions

- **Default screen.** Only the video shows, plus two glass objects:
  - A status capsule at the top left: state dot, state, time and one number. The number changes with context: soak seconds during SOAK, brush force during SCRUB, LEL % during a gas alarm, otherwise rust %.
  - The phase subtitle at the top centre: a plain-English phrase, with the planner's last reason underneath.
- **Everything else goes through ⌘K, Ctrl+K or `/`.** Clicking the capsule also opens the palette.
  - The left column of the palette holds fuzzy-matched results. They are grouped as Top hit, Panels, Robots, Cameras, Scenario events, Demo and Help. Matched letters are shown in bold.
  - The right column is a live preview pane in the style of Spotlight. Each panel previews as a real, live instance of itself, such as the force chart or the gas bars. Cameras preview as a still of the shot, and events preview with what will happen.
  - Each row has a live accessory value (for example 12.3 N, 40.7 % LEL or "New plan"), an on-dot and its shortcut.
- **Empty query.** Shows Suggested and Recent. Suggested depends on context: during a gas alarm, Gas sensors, Gas plume and Control centre come first; during SCRUB, Force and Pipe map come first.
- **Opening panels.** ↵ runs the selected command and closes the palette. ⌘↵ runs it and keeps the palette open.
- **Docking.** The opened card flies out of the preview pane's exact rect and docks to its home edge. Instruments (vitals, force, gas, fleet) dock left. Context panels (mission, map, PiP, Gemini, plume, control centre) dock right. The decision log docks at the bottom.
  - When an edge overflows the viewport, its oldest card is dropped.
  - Closing a card collapses its height, and the glass follows.
- **One blur per edge.** Each dock has a single `backdrop-filter` layer. Its `clip-path: path()` is rebuilt from its cards' rects, so the cards still look like separate rounded cards.
  - During flight, a card uses a plain tinted fill with no blur, then fades into the shared glass.
  - The path is rebuilt only when the string changes. The reads happen at the start of rAF, so layout is still clean.
- **Glass recipe.** Each surface is a tint of `rgba(20,20,24,.6)` over a 40 px tiled noise PNG (2 KB data URI) with `blur(26px) saturate(160%)`.
  - A 1 px top-weighted gradient border is drawn with a masked `::before`.
  - The shadow uses negative spread, so it never darkens the card above.
  - The palette is slightly denser: tint .68, `blur(30px) saturate(170%)`, 22 px radius.
- **Type and colour.** The SF system stack, falling back to Inter, with tabular numbers. Text uses three label tiers: .88, .6 and .32. Coral is used only for the selected row's icon tile, on-dots, the mission track and the search caret. Status colours appear only on values that are actually in alarm.
- **Alarm.**
  - The capsule tints red or amber and swaps its text for the alarm (for example "Gas at Flange B2 · 40 % LEL" or "Back off · 65 N").
  - The subtitle explains the alarm.
  - A soft inset vignette breathes at the screen edge. It animates opacity only, which is composited.
  - The palette's suggestions reorder themselves.
  - Panels never auto-open; the user stays in control.
- **Motion.** The palette fades and scales in 120 ms. Cards fly for 380 ms, and numbers count with a 0.2 lerp. `prefers-reduced-motion` turns off all of it.
- **Mock data for coming features.**
  - Six gas detectors (GD‑1 to GD‑6) at fixed plan positions. Shift+6 starts a leak at Flange B2 that spreads downwind.
  - A top-down plume canvas.
  - Control centre ("Gemini 4 Argon"): a situation report with ranked, clickable actions. It is rebuilt per situation (gas, back off, person, hold, nominal).
  - Fleet: Quad‑1, Quad‑2 and Drone‑1, with floating per-robot labels (R).
- **Dropped from the live HUD:** the module list, the unscripted-change card, the controls card, the title card, the always-on stepper and the always-on vitals. Each one is a palette command instead.

## Cost

Measured in Playwright Chromium at 1920×1080, from a CDP trace over 4 s windows.

- **Stream stand-in.** A 1280×720 canvas was redrawn at 30 fps behind everything, together with the orbiting backdrop. This means every blur re-samples every frame.

| State | fps | p95 frame | main thread per frame (rAF JS + style + layout + paint) | GPU task per frame |
|---|---|---|---|---|
| Empty (capsule and subtitle) | 60 | 16.7 ms | ≈0.20 ms | 0.20 ms |
| 6 panels (3 left, 2 right, log) | 60 | 16.7 ms | ≈0.87 ms | 0.64 ms |
| 6 panels and palette (live preview) | 60 | 16.8 ms | ≈0.92 ms | 0.67 ms |

- **Blur layers.** At most six blur layers are visible at once: capsule, subtitle, three dock layers and the palette. Adding cards to an edge does not add blur layers.
- **Screenshot runs.** Every screenshot run also held 60 fps (p95 16.8 ms) at 1280×720.

## Palette commands

| Group | Command | Key |
|---|---|---|
| Panels | Mission (track and soak ring) | H |
| | Vitals (force, rust, detector sparklines) | V |
| | Force sensor | F |
| | Pipe map | M |
| | Inspection camera | I |
| | Second opinion (Gemini Robotics‑ER) | O |
| | Decisions (log) | L |
| | Gas sensors | G |
| | Gas plume | P |
| | Control centre (Gemini 4 Argon) | A |
| Robots | Fleet | B |
| | Robot labels | R |
| | Quad‑1, Quad‑2, Drone‑1 (show the label and ping it) | |
| Cameras | Wide rig, Hero deck, Close‑up | 1 2 3 |
| Scenario events | New rust patch, Sun glare, Lens mist, Weld bead, Worker walks up, Gas leak | ⇧1 to ⇧6 |
| Demo | Start demo | Space |
| | Cinematic mode | C |
| | Close all panels | X |
| | Restart replay (mock only) | |
| Help | Keyboard shortcuts (opens the palette on it) | ? |
| | About this demo | |

Palette keys:

- ⌘K, Ctrl+K or `/` opens the palette.
- ↑ and ↓ move the selection.
- ↵ runs the command; ⌘↵ runs it and keeps the palette open.
- Esc or a click outside closes the palette.

In live mode, `gas_leak` is simulated only in the browser, because the server does not accept it as a hook yet.
