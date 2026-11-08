# prompt_daphne_2: Mission Control, orchestration agent, benchmarks and the pitch

Hand this file to a Claude session in this repo. Everything below the line is the prompt. It continues
`prompt_daphe_1.md` (the game-style HUD).

Check first what exists. These pieces may be on branches or open PRs, not `main`; check `gh pr list` and merge or
check out what you need:
- `sim/bridge` (PR #6): Isaac Sim bridge, rust repair loop, Gemini verifier and test analyst.
- `fleet-coordinator` (PR #7): fleet coordinator, CPU multi-robot sim, `runs/fleet/timeline.jsonl`.
- Further PRs for the job modules: operator rounds, gas and valves, drone and crawler, and the Isaac fleet scene.

---

You are finishing a hackathon project (Agentverse 2026, Robotics track). The deadline is about 2026-11-08 10:00 UTC; check the
time first and plan to it. Work in this order, and stop adding features once the time-boxed checkpoints say so.

## 0. Ground rules

- Read `HANDOFF.md`, `integration/INTERFACE.md`, `integration/SIM_BRIDGE.md`, `fleet/contract.py`, `fleet/README.md` and
  `docs/gemini_test_report/report.md` first.
- Find the official hackathon rules and rubric (Discord or the event page; ask the user if you can't reach them) and
  re-check everything below against them. HANDOFF.md records:
  - **Judging:**
    - perception-driven action 25%;
    - adaptation to a mid-run change 25%;
    - edge autonomy 15%;
    - relevance plus the critical failure mode shown on screen 20%;
    - ambition and quality 15%.
  - **Deliverables:**
    - a 1-page PDF technical summary;
    - a simulation video of at most 1 minute, single take, with perception on screen and one unscripted change;
    - a 5-minute live pitch.
  - **Separate prize:** best use of Gemini.
- Every claim in the pitch, PDF or UI must be backed by a number we measured, with the file that produced it. Label
  stand-ins as stand-ins (the colour classifier, the greedy scrub policy, the CPU mock, kinematic robots). Never claim
  hardware results we don't have.
- Never print, log or commit API keys. The Gemini key lives in `.env` (gitignored).

## 1. Orchestration agent (on top of the existing coordinator)

The fleet coordinator (`fleet/coordinator.py`) already does assignment, deconfliction, approvals and follow-up jobs
with fixed rules. Add an orchestration agent above it:

- **Inputs:** fleet state, the job queue, findings, operator messages and the timeline.
- **Gemini's role:** Gemini acts as the mission planner. It turns operator intent ("inspect deck 2, isolate any leak,
  don't paint without me") into jobs. It re-plans when findings arrive and explains every priority change in one
  line. It never commands motors directly, and it can only call the coordinator's API (queue, cancel, approve-request,
  reprioritise) through function calling.
- **Edge autonomy:** the coordinator and robots keep working on their last plan if Gemini is slow or offline. Show this
  live with a "cut the link" toggle.
- **Safety envelope:** Gemini can't approve its own SUPV jobs, can't reset an ESD, and can't override the 40 N force
  limit or the 3 m person pause. Write tests for each refusal.
- **Logging:** log every Gemini call (latency, tokens if available, decision) to the timeline.

## 2. Mission Control web app

Extend `rig/web/` (FastAPI + static JS, no build step). Reuse the HUD style from `prompt_daphe_1.md`: minimal, technical,
clean and lean, like a premium game client. Dark translucent panels, one cyan-teal accent, amber for warnings, red only
for safety stops, Inter plus JetBrains Mono with tabular numbers, no clutter. Don't copy any existing product's branding.

Views, switchable by tabs and hotkeys:

1. **Mission map:** top-down rig map with decks as layers. It shows robots with heading and status colour, walkway
   segments, job markers, the gas plume, people, docks and drone no-fly zones. Click a robot to focus it.
2. **First person:** each robot's camera feed (Isaac Sim render, or recorded frames in mock mode). Overlay what that
   robot perceives: rust mask, gauge reading, gas reading, Gemini points.
3. **Sensors:** per robot, gas ppm, thermal, contact force against the 40 N line, battery and IK reach, as sparklines.
4. **Orchestrator:** the job queue by status, the approval inbox (approve and reject buttons, with Gemini's approval
   text), and the decision log with the "why" lines.
5. **Benchmarks:** the results from section 3, as small clean charts.

Also add:
- **Unscripted change panel:** buttons the judges can press (new rust, gas leak, worker walks up, lens mist, weld bead,
  robot battery fault, cut the Gemini link).
- **Replay mode:** `?replay=runs/fleet/timeline.jsonl`, so the UI works without Isaac Sim running.

## 3. Benchmarks: prove our policies with evidence

Build `bench/` with one command per benchmark that writes JSON and a chart. Use the same seeded scenarios for every
method, and report means with 95% confidence intervals over at least 100 episodes where the cost allows.

1. **Scrub control** (Isaac Lab env or the CPU `RustState` mock):
   - **Methods:**
     - our PPO policy (`policy.onnx`, from Modal `rust-ckpt`);
     - Gemini as the controller, choosing `[d_theta, d_z, press]` from the same observation at the rate it can
       manage. Use the model the team calls "Gemini Argon" if that's the one we have access to; confirm the exact
       model ID with the API's model list and record it;
     - the greedy scripted baseline (`policy.demo_mock.GreedyScrub`);
     - a random policy.
   - **Metrics:**
     - success (at most 5% rust left);
     - time to clean;
     - peak force and the share of episodes over 40 N;
     - rust removed per second;
     - decision latency and compute (CPU ms vs API seconds);
     - API cost per episode.
2. **Detection:**
   - **Methods:** our LoRA MobileNetV4 detector (`detection/`, `rust-ckpt:/detector/best`) vs Gemini vision vs the
     colour stand-in.
   - **Data:** held-out real test images and sim renders.
   - **Metrics:** F1, AUROC, latency, and whether it runs on the robot.
3. **Orchestration:**
   - **Methods:** the Gemini planner vs the rule-based coordinator.
   - **Data:** the same shifts with injected events.
   - **Metrics:** time to isolate a leak, jobs done per hour, unsafe actions (must be 0), approvals requested.
4. **Safety testing:** the Gemini test analyst report (`docs/gemini_test_report/`). Rerun it with the real detector
   once available, plus a glare-off control, to show the failure rate going down.

**Report honestly.** If Gemini beats our RL on some metric, say so and explain the trade-off. The expected and
defensible story is: a learned policy for the 10 Hz contact loop on the robot (fast, cheap, works offline, respects
force limits), and Gemini for perception second opinions, planning and explanation (slow but general). Don't fake
numbers or cherry-pick seeds.

## 4. Narrative and pitch (start by early morning at the latest)

- **The story in one line:** the rig's dangerous, repetitive work done by a coordinated robot fleet that knows when to
  ask a human. Tie every feature to a rig worker's job from the team's job table (AUTO / SUPV / ASSIST / HUMAN) and to a
  rubric line.
- **1-minute video:** single take, recorded live in Mission Control.
  - **Opening:** start wide on the rig, then the fleet working.
  - **Perception and decision:** one robot's first-person perception with its decision.
  - **Unscripted change:** a judge-style button press (gas leak), with the orchestrator re-planning on screen.
  - **Safety and edge:** a SUPV approval, a force back-off, the "cut the Gemini link" moment showing edge autonomy.
  - **Ending:** a verified repair.
  - Plan it shot by shot with timings, and rehearse it as a script.
- **5-minute pitch:**
  - the problem with real numbers (cite sources);
  - a live Mission Control demo;
  - the architecture (where each model runs, hardware, latency);
  - the benchmark charts;
  - the critical failure mode and how we handle it;
  - what's real vs simulated;
  - the ask.
  - Write speaker notes, and keep a fallback recording in case the live demo fails.
- **1-page PDF:** user and problem, features, architecture diagram, where each model runs with hardware and latency,
  benchmark table, limits, safety.
- **Rubric check:** a table mapping each judging criterion to the exact moment in the video, the slide and the metric that
  proves it, plus a "best use of Gemini" section showing Gemini's roles: verifier, test analyst, orchestrator and
  explainer.

## Deliver

- Code with tests, `bench/` results, the Mission Control views working in replay mode, `docs/pitch/` (script,
  storyboard, speaker notes, rubric map, PDF draft).
- One PR per area.
- Don't merge without the team's OK.
