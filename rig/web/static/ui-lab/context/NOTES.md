# Variant C: `context`

Open `http://localhost:8103/static/ui-lab/context/index.html?mock=1&autostart=1`. It also takes `&t=<s>` to seek, `&pin=map,vitals` to pin panels and `&presenter`.

## Idea

The screen shows the video, a status capsule and the phase subtitle, plus at most one auto-surfaced panel. That panel is picked by the current phase, and the other panels are pinned through Spotlight.

| Situation | Panel that surfaces |
|---|---|
| Gas alarm, before a decision | Gas sensors |
| Gas alarm, after the control centre decides (or escalation) | Control centre |
| Gemini verdict, for 9 s | Gemini second opinion |
| SCRUB / BACK_OFF | Brush force |
| SOAK | Soak timer |
| DETECT / VERIFY / HOLD | Inspection camera |
| PAUSE_PERSON | Decision log |
| DONE | Pipe map |

- **One slot, with a minimum dwell.** The slot holds its panel for at least 1.6 s before switching, so short phases like HOLD → PAUSE_PERSON → HOLD do not flicker.
- **The gas PiP never surfaces by itself.** That respects the one-panel limit. During an alarm, the Gas plume PiP is the second "Suggested now" row, and one Enter pins it.
- **Pinned panels stay.** They show a coral pin, and the pin also appears on hover over any card. The "Context panels" switch turns auto-surfacing off.
- **Presenter mode** (`P` or the palette) hides everything except the subtitle, for clean capture. Spotlight still opens over it, and "Presenter mode" becomes the top suggestion so it is easy to exit.

## Layout and glass

- **Capsule (top left):** state dot, state name, time, and the one number that matters now: force in SCRUB, seconds left in SOAK, ppm during an alarm, otherwise rust %. Clicking it opens Spotlight.
- **Subtitle (top centre):** a kicker line ("Rust repair · step 5 of 7" or "Gas alarm · Manifold B") and one sentence. It sits on a radial scrim with no blur.
- **Docks:** the right dock holds the data cards and the bottom-left dock holds the media PiPs (inspection, plume).
  - Each dock is one glass surface, so there is one `backdrop-filter` per dock and none per card. Cards are hairline-separated sections inside it.
  - Cards expand and collapse with a `grid-template-rows` 0fr→1fr transition, so a dock grows and shrinks smoothly.
  - At most 4 blurred surfaces are visible: the capsule, two docks and the palette.
- **Glass recipe:**
  - `rgba(20,20,24,.64)` over `blur(24px) saturate(160%)`
  - a tiled 48 px LA-PNG noise data URI (3 KB)
  - a 1 px top-edge gradient highlight (`::before`) and a 0.5 px inset hairline
  - a soft 60 px shadow and a 16 px radius
  - The palette is denser: `.74`, `blur(30px)`, 22 px radius.
- **Robot labels:** a name only, as a small dark pill with a leader line and a dot. They have no blur, so they cost nothing. Atlas, the lead robot, has a coral dot.
- **Colour:**
  - coral is the single accent: the selected palette row, the force line and the chosen control-centre action
  - ok, warn and stop appear only on the capsule dot, the gas rows, and a soft red (alarm) or amber (caution) edge vignette
- **Type:** system-ui (SF Pro on macOS) falling back to Inter, with tabular numbers and label / secondary / tertiary alphas. Sentence case everywhere, with no all-caps.

## Spotlight

⌘K / Ctrl+K and `/` open it, and Esc or a click outside closes it.

- **Keys:** ↑↓ moves, ↵ runs and closes, and ⌘↵ runs without closing.
- **Switches:** clicking a row's switch toggles it in place.
- **Empty query:** "Suggested now" shows up to 4 rows ranked by a per-command context score, then "Recent" (up to 3), then every group.
- **With a query:**
  - Matching rows that have a context score form "Suggested now"; otherwise the best match is the "Top hit".
  - The rest are grouped, in order of each group's best score.
  - Matching is fuzzy. A label match is weighted ×1.4, keywords ×0.7 and the group name ×0.5, plus 0.6 × the context score.
  - The matched part of the label is shown bold.
- **Row hints:** panel rows show "Showing now" in coral when the context slot holds them, and their switch shows the pin state.

### Commands

| Group | Commands |
|---|---|
| Panels | Brush force, Soak timer, Inspection camera, Gemini second opinion, Gas sensors, Gas plume, Control centre, Vitals, Pipe map, Decision log, Fleet, Modules (each one pins or unpins); Phase subtitle, Robot labels, Context panels (switches); Unpin all panels |
| Cameras | Wide rig `1`, Hero deck `2`, Repair close-up `3` |
| Scenario events | Inject new rust `⇧1`, Sun glare `⇧2`, Lens mist `⇧3`, Weld bead force spike `⇧4`, Worker walks up `⇧5`, Gas leak at manifold B `⇧6` |
| Demo | Start demo `Space`, Presenter mode `P` |
| Robots | Atlas, Kestrel, Tern (highlights the label; the hint shows the robot's current task) |
| Help | Keyboard shortcuts `?` |

Examples of the context scores:
- "Start demo" scores 1 before the demo starts.
- During SCRUB: "Weld bead force spike" scores .8, "Repair close-up" .55 and "Worker walks up" .4.
- During VERIFY, "Lens mist" scores .6.
- During a gas alarm, "Gas plume" scores .8 and the scrub suggestions are suppressed.
- At DONE, "Wide rig" scores .6.

## Data

- **Mock replay:** reuses `/api/mock` and `/api/mock/shot/{n}`, using the same playback as `hud.js`.
- **Live:** `/static/viewer.js` is included unchanged for `/ws/video`. The page reads `/ws/state` and posts scenario events to `/api/event`.
- **Optional live fields:** the page also reads these header fields if they are present:
  - `gas: {sensors:[{id,at,ppm}], alarm}`
  - `control: {summary, actions:[[text,score]], chosen}`
  - `robots: [{id,name,x,y}]` (normalised screen positions)
  - `subtitle`
- **Mocked locally:** the gas leak, the control-centre report (Gemini 4 Argon) and the robot positions per camera shot. In live mode, the gas leak posts `gas_leak`, which the server does not accept yet.

## Cost

Measured with Playwright Chromium (ANGLE Metal, Apple M5 Max) at 1920×1080.

- **Test conditions:** a fake 30 fps "stream" is drawn into `#view`, so the blur has to be recomputed every frame. Each run is 4 s of `requestAnimationFrame` timing plus CDP `Performance.getMetrics` deltas, in main-thread ms per second.
- **Result:** every scenario holds 60 fps with p95 = 16.7 ms. Removing every `backdrop-filter` changes neither fps nor main-thread time, because the blur cost is compositor/GPU-side and stays inside the frame budget.
- **Repeat runs:** across three runs, one 4 s window had a single 100–150 ms hitch (57–58 fps average), and this happened with and without blur. A first run measured 66 ms/s, which was warm-up.
- **Main-thread cost:** it comes mostly from text and number updates at 4 Hz and the sparklines, which are drawn only while their card is visible.

| Scenario | fps | p95 ms | Task ms/s (blur) | Task ms/s (no blur) | ×4 CPU throttle: fps / Task ms/s |
|---|---|---|---|---|---|
| Default (capsule + subtitle) | 60 | 16.7 | 14.6 | 14.1 | 60 / 41 |
| Scrub, force auto + 3 pinned | 60 | 16.8 | 26.8 | 28.8 | 60 / 100 |
| Palette open over 2 pinned | 60 | 16.7 | 30.0 | 31.8 | 60 / 124 |
| Gas alarm + plume PiP | 60 | 16.7 | 25.8 | 29.3 | 60 / 123 |

## Files

- `index.html`: the markup for every card
- `ui.css`: tokens, glass and layout
- `ui.js`: about 520 lines; context engine, palette, mock replay, gas mock, plume canvas and sparklines
