# Mission Control: scenarios and Spotlight UI

## Intent

A live demo of the rig for hackathon judges. We (the team) drive the main screen from the keyboard. The judges only use the phone page at `/interactive` to drop obstacles and watch navigation re-route. The demo has to show every policy we built, look photoreal, and feel like an Apple Spotlight product: one hero image, almost nothing on screen, everything else one keystroke away.

## Goals

- Three scripted scenarios that, together, exercise every policy we have.
- Disturbances we can throw live to show that the robots and Gemini 4 Argon react to them, rather than following a script.
- Robot-perspective views (camera, detection boxes, lidar, depth) that show what each policy is conditioned on.
- Keep the hero render fully photoreal while all this is going on.

## Non-goals

- Slides, title cards or a presentation mode inside the app.
- Workers or people on the deck.
- Judges using the keyboard UI.

## Anti-goals (would be a failure even if it works)

- Clutter: more than one side card, permanent panels, all-caps labels, borders inside borders.
- Anything that reduces the hero render's quality or frame rate. The live stream must stay at 10 fps or better on jarvis.
- Rendering views nobody is looking at.
- Gemini approving its own requests. Only a human approves.

## Scenarios

Each scenario is started from Spotlight. Starting one resets the deck to a known state, including dusk + rain + fog, and pre-selects the robot doing its main task.

### 1. Gas leak, with a hot-work stop
1. A fixed gas sensor on a flange trips.
   - The screen edges pulse red and an alarm sounds, both for about 3 s.
   - The capsule shows the gas level in %LEL.
2. Gemini 4 Argon reads the sensor data.
   - It checks its plan in simulation before filing it.
   - Its requests go to the operator inbox.
3. The fleet coordinator dispatches the gas robot.
   - The robot localises the leak using a particle filter and infotaxis.
   - A plume simulation view is available.
4. On operator approval, a robot closes the isolation valve.
5. A welding robot on a nearby seam is frozen by the hot-work gate while gas is above 10% LEL. It resumes seam tracking once the gas clears.

Disturbances (I), in order:
- the leak reopens;
- a second flange leaks;
- the link to Gemini is cut, and the fleet carries on with its rule-based fallback.

### 2. Deck patrol and housekeeping
1. A robot walks a set patrol route using the walking policy, with load and sway adaptation.
2. Navigation re-routes around blocked walkways. Puddles are a soft cost: walk around if the detour is cheap, otherwise walk through slowly.
3. Fallen cans are collected by the can pickup policy and dropped into the robot's back tote.
   - The policy runs on depth-derived geometric priors.
   - Gemini Robotics-ER chooses the grasp on objects the robot hasn't seen before.
4. Battery management sends robots back to dock.

Disturbances (I), in order:
- more small cans;
- an unfamiliar object to pick up;
- the 205 L drum, which is too big to pick up. The robot reports it, Argon closes the walkway, and the fleet re-routes.

The judges' `/interactive` presses land in this scenario: puddle, small can, large can, crate, spill, debris, scaffold, hose and closures. Each shows up as a toast.

### 3. Pipe repair
1. The rust detector finds the patch. In rain it adapts at test time.
2. The robot walks to the pipe and scrubs it.
   - The scrub policy uses the force safety shield, and the storm policy when there are gusts.
   - A live force graph is shown.
3. The robot verifies the patch is clean. Painting waits for operator approval.
4. Weld inspection finds a crack and the robot repairs it.

Disturbances (I), in order:
- rust reappears in the same spot, so it is re-detected and re-scrubbed;
- a wind gust hits, and the force must stay under 40 N.

## Screen

### Launch
- The ambient fly-by plays full screen with only a faint "⌘K" hint.
- Spotlight opens automatically on first load.

### Always on during a scenario
- **Capsule, top:** scenario name, current step, and one key number (%LEL, brush force, cans collected).
- **Caption, bottom:** one line on what is happening, for example "Gas robot closing on F3".

### Hero views
- **0: Mission Control.** A wide view of the deck showing all robots, each labelled with its task ("R2 · gas survey"), plus route lines, blocked walkways and targets. A slim fleet list on the left shows robot, task and battery, alongside the operator inbox. Approving requests happens here. Scenarios start in this view.
- **1, 2, 3: robot.** The hero camera cuts to a chase shot of that robot, with a small coral label on it. The cut is instant, not a camera flight.

### Overlays for the selected robot
Only one side card is shown at a time, on the right. Opening another overlay replaces it, and pressing the same key again closes it.

| Key | Card |
|---|---|
| F | robot camera |
| B | robot camera with detection boxes (boxes appear only in this first-person card) |
| L | lidar |
| D | depth |

### Other panels
- **S:** sensor strip along the bottom. The readings that matter for the current task are highlighted in coral.
- **G:** Gemini panel on the left. It shows Argon's thought summaries as they stream (raw chain of thought is not exposed), each call with its one-line "why", and anything the safety envelope refused.

### Sound
Simple and sparse, in the browser (no sim audio):
- Alerts: gas alarm, an approval-needed chime, a short ping when a disturbance fires, and a soft click on Spotlight actions.
- If there is time: a welding crackle while the weld robot is working.
- **M** mutes all sound.

### Other keys
- **I:** throw the scenario's next disturbance and announce it in the caption.
- **Esc:** back to the fly-by.
- **?:** help.
- **⌘K or /:** Spotlight.

## Spotlight

Groups, fuzzy-searchable:
- **Scenarios:** Gas leak, Deck patrol, Pipe repair, Fly-by.
- **Views:** Mission Control (0), R1–R3, F, B, L, D, S, G.
- **Disturbances:** the current scenario's disturbances, listed by name.
- **Help.**

Every row shows its shortcut on the right.

## Robot motion in the hero (user decision)

- **Every visible robot is animated** as it walks through the hero view: legs cycle at a speed matched to its route speed (e.g. a recorded gait loop from the walking policy). For looks only; it does not need to be the live policy.
- **Task motions** (scrub, pick-can, weld) on the selected robot use the real policy where it is ready to run live; otherwise a recorded rollout of that policy, captioned honestly.
- Joint states stream to the hero within the 24–30 Hz state budget.
- L and D for the selected robot show that policy's actual input (e.g. walking height scan, pick-can depth); for others they show a generic ray-cast view.
- Disturbances in a robot's beat (rust reappears, gust, more cans) are handled live by its policy, not pre-recorded.

## Render budget

- **Jarvis:** renders only the hero view, at 1920×1080 with real-time path tracing, dusk + rain + fog. Nothing else runs on jarvis's GPU while the demo is live.
- **Robot camera card (F/B):** rendered on watson only while the card is open, and only for the selected robot. Low resolution (about 256×192 to 512×288), at 5 Hz or less, decoupled from the physics loop. The state stream to jarvis must stay at 24–30 Hz.
- **Lidar, depth and boxes:** come from simulation ray casts and detector outputs, not from photoreal renders.
- **During the live demo,** no other agents' GPU jobs run on watson.

## Visual language

Keep the existing HUD language:
- dark glass, with one blur per surface;
- system type and tabular numbers;
- one accent colour, Claude coral `#D97757`, plus ok/warn/stop status colours used sparingly;
- quick, motion-respecting transitions.

## Left to the builder

- Exact card sizes and the chase-camera framing.
- The alarm sound asset.
- How the disturbances are implemented in the mission brain.
- Which robot (R1, R2 or R3) plays which role.
- How the sensor strip picks which readings to show, using the "relevant to the task" rule.

## Builder's calls (not discussed with the user; flagged for review)

- Spotlight opens automatically on first load.
- The alarm stops after about 3 s.
- Scenarios start in Mission Control (0).
- Cuts are instant rather than camera flights.
- Disturbances run in a fixed order rather than at random.
- Welding sits in scenario 1 (through the hot-work gate) and also in scenario 3 (the crack repair).
- The robot camera card is rendered on watson at lower quality.
- Novelty modules already merged as opt-in (infotaxis gas search, Gemini-ER grasp, sim-verified plans, rain TTA) get wired into the live mission brain; anything not live in a run is captioned honestly (e.g. "cast-and-surge" not "infotaxis") — no claim on screen that isn't running.
- Approval: only today's supervised job kinds (valve isolate, paint, retorque, spray, …) wait for an operator click; AUTO kinds (inspection, gas survey, walkway closure/re-route) run straight away. Gemini can never approve.
- Judges' `/interactive` presses apply to whatever is running (every scenario has walking robots that re-route); starting a scenario does not clear their items — only "Clear all" does.
- Gemini panel: thought summaries are requested from the API and streamed; if unavailable, the panel shows each decision with its why.
- Render: jarvis is already the 1080p real-time path-traced hero (measured 13.5 fps, ~0.1 s latency with dusk + rain + fog on 2026-11-08). If it drops below 10 fps, resolution steps down first (DLSS upscale from a lower internal resolution) before any look feature is cut.
