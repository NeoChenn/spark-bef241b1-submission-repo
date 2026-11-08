# Variant B: dock

Run `uv run uvicorn rig.web.server:app --port 8102` and open
`http://localhost:8102/static/ui-lab/dock/index.html?mock=1&autostart=1`.

Query flags:

- `t=<s>` seeks the mock replay.
- `panels=gas,plume,control` opens panels.
- `labels` shows the robot labels.

## Design decisions

- **Default screen.** It shows only the status capsule and the phase subtitle, both at top-centre. The robot labels are off by default and toggle with `R`. A 36 px handle at the bottom edge is the only sign that the dock exists.
- **Spotlight.** It is the main way in to everything:
  - `⌘K`, `Ctrl+K` or `/` opens it, and `Esc` closes it.
  - It is a centred 660 px field in 22 px type.
  - With an empty query it shows Recent and Suggested; the suggestions change once the demo has started.
  - Typing gives a Top hit, then grouped results: Panels, Cameras, Scenario events, Demo, Robots and Help.
  - Matched characters are bolded. A green dot marks a command that is on.
  - Every row shows its shortcut.
  - `⇧↵`, or the pin glyph on a hovered row, pins or unpins a panel to the dock.
  - It scales and fades in over 120 ms.
- **Dock.** It is a slim glass strip at bottom-centre that holds only the pinned panels, plus Search.
  - It stays hidden and rises when the pointer comes within 72 px of the bottom edge, when the palette is open, or for about 2 s after a command runs. That keeps it quiet during recording while judges can still see at a glance what is available.
  - A macOS-style dot under an icon means its panel is open.
  - A coral badge marks new content in a closed panel, such as a new Gemini verdict or a new decision. The badge turns red for a gas alarm. This replaces the old auto-popping Gemini card.
- **One sidebar.** Every opened panel is a section of a single glass column on the right: one blur layer, hairline separators, and a close button that appears on hover. Sections keep a fixed order, so the layout never reshuffles. The column scrolls when it overflows at 720p. The left side and the centre stay clear for the video.
- **Dynamic Island alarms.** On BACK OFF, PERSON, HOLD, ESCALATE, a gas level above 10 % LEL, or DONE, the capsule springs into a banner with an icon, a title and one line of detail. It collapses after 3.6 to 4.2 s. The capsule then keeps the status tint while the condition lasts.
  - While the banner is up, the subtitle hides.
  - Alarms are skipped while seeking, so a replay that starts mid-run does not flash old alarms.
- **The single number.** The capsule shows state, time and one contextual figure:
  - % LEL while gas is alarming
  - the soak countdown while soaking
  - force while the arm is in contact
  - rust remaining otherwise
- **Glass.**
  - Fill and blur: `rgba(22,22,24,.52)` with `blur(20px) saturate(160%)`.
  - Noise: a 64×64 static noise PNG (4.8 kB, data-URI).
  - Edge and depth: a 1 px masked top-edge gradient for the inner highlight, and one soft 60 px shadow.
  - There are no borders on the panels themselves.
  - While Spotlight is open, the island, sidebar and dock drop their blur and sit under a flat scrim. The palette is then the only blurred surface.
- **Type and colour.**
  - Type is system-ui (SF Pro on macOS), falling back to Inter, with tabular numbers.
  - Text uses Apple-style label, secondary and tertiary alphas.
  - Coral is the only accent. Green, amber and red appear only for status.
  - There is no all-caps text.

## Panels covered

| Panel | Contents |
|---|---|
| Mission | 8-segment stepper with an inline soak ring |
| Control centre | Gemini 4 Argon situation report, plus three ranked actions with confidence and Run buttons. Nominal and gas variants. |
| Gas sensors | 4 detectors, % LEL, bars tinted at 10 and 20 % |
| Gas plume | Deck schematic with a drifting plume and robot dots |
| Vitals | Force, rust and confidence sparklines |
| Force sensor | Large readout, gauge with the 40 N mark, 20 s chart |
| Pipe map | Heatmap |
| Inspection camera | Inspection image |
| Second opinion | Gemini verdict |
| Fleet | 3 robots with role, task and battery |
| Decisions | Last 5 decisions |
| Keyboard shortcuts | Shortcut list |
| Overlays | Robot labels with a stem line and a tinted status dot, and the phase subtitle |

The panels are unchanged from today's HUD unless the table says otherwise.

The gas data is a local stub. In live mode, `gas_leak` would need a server hook; today it only drives the mock.

## Cost

Method:

- `/ws/video` was simulated by drawing a 1280×720 frame into `#view` at 30 fps, with the mock replay playing.
- Frame times come from 5 s of `requestAnimationFrame` deltas.
- Paint, layout and raster times come from the CDP `devtools.timeline` trace and are averaged per frame.

Headless Chromium with a GPU (ANGLE Metal), on an M-series Mac:

| State | 1920×1080 fps | paint ms/frame | 1280×720 fps |
|---|---|---|---|
| Default | 60 | 0.05 | 60 |
| 4 panels | 60 | 0.11 | 60 |
| 6 panels incl. plume and map | 60 | 0.11 | 60 |
| Spotlight open | 60 | 0.11 | 60 |
| Gas alarm (banner, 3 panels) | 60 | 0.05 | 60 |

Software compositing (headless without a GPU, the worst case):

- 1080p: 60 fps with nothing open and 42 to 52 fps with 4 to 6 panels open. Blur over a moving video is the whole cost; disabling it gives 58 to 60.
- 720p: a steady 60 everywhere.

Canvases draw only while their section is open, and the gas model ticks at 2 Hz.

## Palette commands

| Group | Commands |
|---|---|
| Panels | Mission `I`, Control centre `A`, Gas sensors `S`, Gas plume `O`, Vitals `V`, Force sensor `F`, Pipe map `M`, Inspection camera `P`, Second opinion `G`, Fleet `T`, Decisions `L`, Robot labels `R`, Phase subtitle |
| Cameras | Wide rig `1`, Hero deck `2`, Close-up `3` |
| Scenario events | New rust patch `⇧1`, Glare `⇧2`, Lens mist `⇧3`, Weld bead `⇧4`, Worker walks up `⇧5`, Gas leak at separator `⇧6` |
| Demo | Start demo `Space`, Cinematic mode `C`, Close all panels, Reset dock |
| Robots | Locate Spot-1, Spot-2, Drone-1 (turns labels on and rings the robot) |
| Help | Keyboard shortcuts `?` |
