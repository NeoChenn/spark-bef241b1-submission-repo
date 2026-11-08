# Pitch: Rig Robot, a two-robot fleet coordinated by Argon (5 minutes)

Agentverse 2026, Robotics track. There are 11 slides; the deck is `docs/pitch.html` (← → or space to move, F for fullscreen; `docs/pitch.pdf` is the same deck). The one-page summary is `docs/technical_summary.pdf`. The fleet demo itself is described in `docs/fleet_demo.md`.

The judging weights tell us where to spend the time. Perception-driven action (25 %) is covered on slides 4, 6 and 8, adaptation (25 %) on slides 4, 5, 8 and 9, edge autonomy (15 %) on slides 5 and 10, relevance and the critical failure mode (20 %) on slides 2 and 9, and ambition and quality (15 %) on slides 3, 5, 7 and throughout.

Items marked **TBD** are still being produced:
- the final one-take fleet video file (`runs/fleet/fleet_video.mp4`)
- the GB10 TensorRT numbers (`edge-bench`)

Fill them in before the talk, or drop the line.

| # | Slide | Time | Cumulative |
|---|---|---|---|
| 1 | Hook | 0:15 | 0:15 |
| 2 | The problem: two chronic jobs | 0:20 | 0:35 |
| 3 | Two robots, one control centre | 0:25 | 1:00 |
| 4 | **Demo video** (≤ 60 s) | 1:05 | 2:05 |
| 5 | Argon: two sensors, one verdict | 0:35 | 2:40 |
| 6 | Perception drives every action | 0:25 | 3:05 |
| 7 | The scrub policy and the force problem | 0:25 | 3:30 |
| 8 | Unit 2's gas-leak job: loose bolt or crack? | 0:20 | 3:50 |
| 9 | When things go wrong | 0:25 | 4:15 |
| 10 | Edge autonomy, and what's next | 0:20 | 4:35 |
| 11 | Limits and close | 0:25 | 5:00 |

---

## 1. Hook (0:00–0:15)

**On screen:** the night rig hero render. Title: *Rig Robot: two quadrupeds and Argon, the rig's control centre.*

**Say:**
> "Two jobs keep sending people into the most dangerous place on an offshore platform: fixing rust, and chasing small gas leaks. We built a two-robot fleet that does both, coordinated by a control centre we call Argon, and it knows when to call a human."

## 2. The problem: two chronic jobs (0:15–0:35)

**On screen:** two columns.
- **Corrosion**: pipework rusts constantly in salt spray, and people fix it by hand from rope access.
- **Small gas leaks**: bolted flanges work loose, and a person with a gas detector walks up to find them.

Underneath: **dangerous · costly · never finished**. The user is the asset-integrity and maintenance crews.

**Say:**
> "Neither job is hard. Both are dangerous, slow and endless. The robots have to be trusted, and there are two things they must never do: paint over rust, and put a wrench on a cracked flange."

*(This sets up the critical failure modes. Come back to them on slide 9.)*

## 3. Two robots, one control centre (0:35–1:00)

**On screen:** three cards, with Isaac frames of both robots.
- **Unit 1, Scrubber**: Go2 + 0–0.45 m lift mast + UR3, brush pad, spray and paint nozzle, its own gas sniffer. Spray → soak → scrub (PPO) → verify → paint.
- **Unit 2, Fixer**: Go2 + lift mast + UR3 with a socket tool, gas sensors. Lidar walk-in → sniff → inspect the bolts → re-torque → verify → back to the dock.
- **Argon** (gemini-4-argon): fuses each fixed gas detector with the nearest robot's sniffer, sends the robot whose tools fit the job, holds hot work near gas, and files tickets for a human.

Underneath: *fixed detectors on every flange (green, amber, red, grey for a faulty head); safety rules act locally and at once; Gemini's replies are checked before anything moves.*

**Say:**
> "Two Unitree Go2s with lift masts and UR3 arms, on a full rig in Isaac Sim. Unit 1 carries a brush and a spray nozzle; it fixes rust. Unit 2 carries a socket tool and gas sensors; it fixes leaks. Both robots, and the fixed gas detector on every flange, report to Argon. Argon decides who goes where. The safety rules don't wait for it."

## 4. Demo video (1:00–2:05)

**DEMO CUE:** say the intro line, then **play the video** (≤ 60 s, one take). The final file is **TBD** (`runs/fleet/fleet_video.mp4`). Don't talk over it except for the call-outs. Read them off the HUD as they happen.

**Intro (5 s):**
> "This is one take. A bolt works loose in the middle of the job, and nobody tells the robots. Watch the HUD."

**Call-outs while it plays:**
1. On the spray and the scrub: *"Unit 1 sprays, lets it soak, then scrubs. The brush squishes onto the pipe and stays under 40 newtons."*
2. When G-F5 goes amber (the loosening is scripted in the sim at a fixed time, not announced): *"Gas. A wire brush is a spark source, so Unit 1 stops and steps back at once. Its job is paused, not cancelled."*
3. When Unit 2 sets off: *"Unit 1's sniffer is too far away to confirm it, so Argon sends Unit 2 to check."*
4. At the flange: *"Confirmed. It finds the loose bolt and re-torques it. The detector goes green."*
5. When Unit 1 resumes: *"Argon clears the zone. Unit 1 picks up exactly where it stopped, then paints."*
6. If the stuck detector or the refused resume shows: *"That detector is stuck high, and the robot next to it reads clean, so it's a ticket for a human, not a repair."* / *"Gemini asked to resume while the leak was still open. The safety layer refused it."*

**Fallback if the video fails:** open the fleet HUD replay of the live Argon run (`?replay=` the run folder, then Space to start, H for the controls). If nothing loads, use https://rig.zeecloud.org or the mock replay at `/?mock=1`.

## 5. Argon: two sensors, one verdict (2:05–2:40)

**On screen:** the verdict table and Argon's numbers.

| Fixed detector | Robot sniffer | Verdict | Argon |
|---|---|---|---|
| High | High | **LEAK_CONFIRMED** | Unit 2 fixes the flange |
| High | Clean, beside it for 3 s | **PIPE_SENSOR_FAULT** | Ticket to a human, no repair |
| Clean | High | **LEAK_UNSEEN_BY_FIXED** | Inspect, flag the detector |
| High | Too far away to tell | **UNVERIFIED** | Send Unit 2 to check |
| Clean | Clean | **OK** | Carry on |

Numbers: **18/18** scenario evals correct; **4/4** valid live calls in the demo run; **4–9 s** per call, non-blocking; about **$0.003** per call.

Underneath: *hazard first (≥ 20 %LEL is always real gas); spark rule (hot work stops within 6 m of an amber or red detector); link cut, timeout or a bad reply → the same rules decide locally.*

**Say:**
> "A fixed detector on its own can't tell a leak from a broken sensor. A robot's sniffer on its own only knows where it's standing. Argon puts the two together. Both high: fix it. The detector high but the robot standing next to it reads clean: the detector is broken, so a human gets a ticket and nobody touches the pipe. And Argon is advice with guard rails: anything at the stop limit is treated as real gas, hot work stops near gas without waiting for Gemini, and if the link drops the robots make the same safe call on their own."

## 6. Perception drives every action (2:40–3:05)

**On screen:** two pipelines side by side.
- **Rust**: frame → 96 px patches → MobileNetV4 + LoRA heatmap → (θ, z) grid on the pipe → spray path, scrub targets, done check. Numbers: **F1 0.977** on 881 real photos, **13/13** held-out Isaac renders.
- **Leak**: frame → project the known bolt positions → read each red torque stripe (how far each bolt turned) → classical crack check → diagnosis. Numbers: **40/40 cracks found, 0/40 false alarms**; stripe angle error ≤ 0.019 rad (about 1°).

**Say:**
> "Nothing is a fixed script. For rust, a 1-millisecond MobileNet heatmap is projected onto the pipe and becomes the work plan. For leaks we didn't need ML. Every bolt has a torque stripe. When the bolt turns, the stripe breaks, and we measure that to about a degree. A classical crack check found 40 of 40 cracks with no false alarms. Both checks have an 'unsure' answer, and unsure means take another look."

## 7. The scrub policy and the force problem (3:05–3:30)

**On screen:** a v1 vs v2 table (rig geometry, a new rust patch every episode):

| | v1 | v2 |
|---|---|---|
| Success | 100 % | 97.6 % |
| Episodes over 40 N | **76.8 %** | **1.0 %** |
| Mean / p95 max force | 49.9 / 70.1 N | 29.9 / 35.7 N |

**Say:**
> "Scrubbing is PPO in Isaac Lab across 2,048 parallel environments, with a wear model where rust comes off four times faster once it has soaked. Version 1 cleaned everything by pressing too hard: three episodes in four broke our 40-newton limit. We retrained it with a quadratic over-force penalty and a 38-newton cap. Episodes over the limit fell from 77 % to 1 %, and success only dropped to 97.6 %."

## 8. Unit 2's gas-leak job: loose bolt or crack? (3:30–3:50)

**On screen:** the crack-check figure (`docs/leak/crack_check.png`: clean / cracked / cracked through a dirty lens) and three outcomes from the mock:
- loose bolts 1 and 5 → re-torque both → gas clears → **DONE**
- cracked flange → back off 3 m → **ESCALATE** (critical)
- dirty lens → 2 extra views → **ESCALATE** (unsure)
- leak near hot work → Unit 1 holds, Unit 2 re-torques, Argon resumes Unit 1 → **DONE**

**Say:**
> "The gas sensors tell Unit 2 that something is leaking. The camera tells it what. A loose bolt is something it may fix: it tightens the worst bolt, then checks that the stripe lines up and the gas falls. A crack it must never touch: it backs off and raises a critical alert. A dirty lens hides cracks before it hides the stripes, so a blurry 'no crack' isn't trusted. It becomes 'unsure', and a human gets the call."

## 9. When things go wrong (3:50–4:15)

**On screen:** the failure-mode table (the critical rows are highlighted) next to a HUD screenshot (`docs/img/hud_mock_backoff.png`, the mock replay at BACK_OFF):

| Failure | Response |
|---|---|
| **Paints over rust** | VERIFY: < 5 % rust, ≥ 80 % visible; Gemini veto; 3 failed cycles → ESCALATE |
| **Tightens a cracked flange** | crack possible → never touch bolts; back off 3 m, critical alert |
| Gas ≥ 20 %LEL or a robot sensor fault | GAS_RETREAT (fail-safe) |
| Detector amber within 6 m of hot work | Spark rule: HOLD Unit 1 at once; Unit 2 fixes; resume |
| Gemini says "resume" while the leak is open | Refused and logged (it happened in the live run) |
| Link to Argon cut | The same rules decide locally |
| Force > 40 N | BACK_OFF: retract 5 cm, press × 0.7 |
| Person < 3 m · camera blocked | PAUSE_PERSON · HOLD |

**Say:**
> "Before every step, the checks run in a fixed order: is the gas sensor working, is gas below the limit, is anyone near. The most severe hazard wins. In our live run Gemini did ask to resume the scrub while the leak was still open. The safety layer dropped that command, kept the rest of the reply, and logged why. Every response is on the HUD and in a decision log, with the evidence behind it."

## 10. Edge autonomy, and what's next (4:15–4:35)

**On screen:**
- **On the robot (CPU):** rust detector 1.1 ms; scrub policy 0.012 ms; leak check about 90 ms per frame (measured on a laptop CPU); lidar navigation and the safety rules. TensorRT on the GB10: **TBD**.
- **Cloud, Argon:** gemini-4-argon fleet decisions, non-blocking, 4–9 s per call; the Gemini rust second opinion. Offline, the edge rules make the safe call.
- **Training only:** Modal A10G / L40S and our GB10.
- **Next:** trained PPO lidar navigation on both units (the demo uses the pure-pursuit fallback), a 3D gas plume with orifice leak rate, storm adaptation, and more units and jobs under Argon.

**Say:**
> "The whole fast loop runs on the robots. Argon takes seconds to answer, so nothing safety-critical waits for it: the robots hold, retreat and back off on their own, and Argon's answer lands when it lands. If the satellite link drops, the edge rules make the same calls."

*(Cut the "next" line if you're over time.)*

## 11. Limits and close (4:35–5:00)

**On screen:** limits on the left, the tagline on the right.
- Simulation only. The robots are replayed kinematically from the logged run, the brush force is a spring model, and bolt torque isn't simulated.
- The leak is scripted at a fixed time, and not announced to the robots or Argon. Gas thresholds are illustrative. Navigation uses the pure-pursuit fallback. The leak checks still need re-testing on Isaac frames.

Tagline: *Find it. Fix it. Prove it. Ask when unsure.*

**Say:**
> "To be clear: this is simulation, and our gas numbers are illustrative. But the logic is the point. Two robots find rust and leaks, a control centre sends the right one, they fix the routine problems, prove they're fixed, and ask a human about everything else. Thank you."

---

## Q&A cheat sheet (keep each answer to about 30 s)

**Is this just a scripted animation?**
No. Every transition comes from perception. The rust mask becomes the target grid, the stripe reading picks which bolt to tighten, and the state machine moves on only when the camera or the gas sensor confirms the result. The mid-run changes, new rust or a bolt working loose, are scripted in the sim but aren't announced to the robot. Each decision is logged with its evidence in `decisions.jsonl`, and every Argon request, reply, command and refusal is in the fleet timeline.

**Why no ML for the leak checks?**
We didn't need it, and a classical check is easier to trust. A painted torque stripe is an industry convention: if it's broken, the bolt has turned. The crack check looks for long, thin, dark shapes on the flange face and measures their length in millimetres. It found 40/40 cracks with 0/40 false alarms on held-out renders; 6/40 clean flanges came back "unsure", which costs a second look, not a miss. An ONNX crack-model path exists (validation F1 0.98 after one epoch), but it is off by default.

**What if it tightens a bolt on a cracked flange?**
That's the failure we designed against. A crack probability ≥ 0.5 is never a robot repair. While a crack is merely possible (unsure), no bolt is touched. And a dirty lens can't produce a "no crack", because view quality below 0.85 turns it into "unsure". The safety sweep in the tests checks that it never tightens a cracked flange, never tightens in gas at the limit, and never finishes "done" with a loose bolt left.

**Is Argon really making the decisions in the video?**
Yes, with one exception we show on purpose. The video replays a recorded run in which four live gemini-4-argon calls made the dispatch to F5, the fix order, the sensor-fault ticket and the final resume (4.2–8.6 s each, about 1¢ for the whole run). The spark-rule hold came from the edge rule in the same tick as the amber alarm, because hot work near gas can't wait seconds for a model. Replaying the recording reproduces the same commands at the same times.

**What do the gas numbers mean?**
They are percentages of the lower explosive limit (%LEL): under 5 is clear, 10 is an alarm (red), and 20 or more means stop and retreat, and is always treated as real gas. They are illustrative, not from a site standard; the gas model is a simple distance-decay field. Unit 2 is the dedicated gas robot; a 3D plume model is next.

**How does it handle a leak in the middle of a scrub?**
In the demo run the bolt starts leaking at 18.0 s. The F5 detector goes amber 0.8 s later, and because it is within 6 m of Unit 1's wire brush, the edge spark rule holds Unit 1 in the same tick: it lifts the brush and steps back about 1.5 m, with its rust progress kept. Unit 1's sniffer is too far away to confirm, so Argon sends Unit 2. It sniffs, confirms, re-torques bolt 3, and the detector goes green. Argon then clears the zone and Unit 1 resumes exactly where it stopped, finishes the scrub and paints.

**How good is the rust detector, really?**
On 881 real photos it scores F1 0.977, recall 0.983 and AUROC 0.998. On held-out Isaac renders it gets 13/13, against 10/13 before we added sim-style data. The caveats: the render set is small, and one real source of weathered surfaces gives up to 13 % false positives. We set the threshold for recall, because a miss is the costly error.

**Why did scrub success drop from 100 % to 97.6 %?**
That was deliberate. v1 got 100 % by pressing too hard: 76.8 % of its episodes went over 40 N, with a peak of 87.5 N. v2 keeps 99 % of episodes under the limit. The episodes it doesn't finish go back to VERIFY, which scrubs again or escalates. It never paints over them.

**What stops it damaging the pipe?**
There are three layers: the policy is trained against over-force; the setpoint is clamped to 38 N outside the network; and the orchestrator's hard 40 N gate backs off (retract 5 cm, press × 0.7) and escalates after 3 back-offs.

**What if Gemini is wrong or offline?**
Argon's replies are parsed against a fixed schema: a command must name a known robot and flange, and some commands are only allowed for some robots (only Unit 1 can be held for the spark rule). Anything at the stop limit, and the spark rule, act on the edge first; Gemini can only add to that, never delay it. An unsafe command is dropped and logged: in the live run Gemini asked to resume Unit 1 while the leak was open, and that one command was refused. If the link is cut, a call times out or a reply is invalid, the edge rules make the same call; the cut-link replay shows it at 22 s. For rust, Gemini can veto a paint at most twice but can't make the robot skip a check.

**What runs on the edge?**
Everything in the fast loop: the rust detector (ONNX on CPU, 1.1 ms), the leak check (numpy, about 90 ms per frame on a laptop CPU, 3 frames per view), the scrub policy (0.012 ms) and the state machine. GB10 TensorRT numbers are TBD. Training used Modal GPUs and the GB10.

**Why two robots instead of one?**
The tools and the hazards differ. A wire brush is a spark source, so the scrubber has to stop near gas, which is exactly when the leak robot has to go in. Splitting them lets Argon assign jobs by capability and keep hot work and gas work apart. Both units share the same base, arm and safety rules.

**How far is this from a real robot?**
The perception and decision logic carry over as they are, since they run on CPU from camera frames. The gaps are a real force-torque sensor in place of the spring model, real bolt torque and thread friction, legs that actually walk under arm load (the demo replays the gait kinematically), the trained lidar navigation policy in place of the pure-pursuit fallback, a site-standard gas threshold, and real painted-steel and flange data.

**What did you build in the hackathon versus reuse?**
We built the rust data pipeline and sim-style data, the detector fine-tune, the localizer, the scrub environment and wear model, both PPO policies, the lift-mast robot, the orchestrator and leak job, the stripe and crack checks, the flange scene, the Isaac bridge, the HUD, and the GB10 stream. For the fleet we built Argon's fusion and edge rules, the Unit 2 fixer robot and its re-torque skill, lidar navigation for both units, the fleet HUD, and the single-take camera. We reused the Go2 and UR3 models, the Poly Haven CC0 textures, the CC BY 4.0 rig model, and Isaac Sim / Isaac Lab.

**What's the cost argument?**
Don't invent numbers. Say: "Today every one of these tasks means people flown offshore and working at height or next to a leak. The robot does the routine ones and sends the exceptions to a person."
