# Pitch: Rig Robot, one quadruped with two jobs (5 minutes)

Agentverse 2026, Robotics track. There are 10 slides; the deck is `docs/pitch.html` (← → or space to move, F for fullscreen; `docs/pitch.pdf` is the same deck). The one-page summary is `docs/technical_summary.pdf`.

The judging weights tell us where to spend the time. Perception-driven action (25 %) is covered on slides 4, 5 and 7, adaptation (25 %) on slides 4, 7 and 8, edge autonomy (15 %) on slide 9, relevance and the critical failure mode (20 %) on slides 2 and 8, and ambition and quality (15 %) on slides 6, 9 and throughout.

Items marked **TBD** are still being produced:
- the final one-take video file
- the live two-job loop in Isaac 6.2 with the trained models
- Gemini latency
- the GB10 TensorRT numbers (`edge-bench`)
- unit 2, the gas robot (`leak-bot`)
- the gas plume window (`gas-plume`)
- the fleet planner (`fleet-planner`)
- storm adaptation (`storm-adapt`)
- Go2 walking (`go2-walk`)

Fill them in before the talk, or drop the line.

| # | Slide | Time | Cumulative |
|---|---|---|---|
| 1 | Hook | 0:15 | 0:15 |
| 2 | The problem: two chronic jobs | 0:25 | 0:40 |
| 3 | One platform, two jobs | 0:25 | 1:05 |
| 4 | **Demo video** (≤ 60 s) | 1:05 | 2:10 |
| 5 | Perception drives every action | 0:30 | 2:40 |
| 6 | The scrub policy and the force problem | 0:30 | 3:10 |
| 7 | The gas-leak job: loose bolt or crack? | 0:30 | 3:40 |
| 8 | When things go wrong | 0:25 | 4:05 |
| 9 | Edge autonomy, and what's coming | 0:30 | 4:35 |
| 10 | Limits and close | 0:25 | 5:00 |

---

## 1. Hook (0:00–0:15)

**On screen:** the night rig hero render. Title: *Rig Robot: one quadruped, two jobs.*

**Say:**
> "Two jobs keep sending people into the most dangerous place on an offshore platform: fixing rust, and chasing small gas leaks. We built one robot that does both, and that knows when to call a human."

## 2. The problem: two chronic jobs (0:15–0:40)

**On screen:** two columns.
- **Corrosion**: pipework rusts constantly in salt spray, and people fix it by hand from rope access.
- **Small gas leaks**: bolted flanges work loose, and a person with a gas detector walks up to find them.

Underneath: **dangerous · costly · never finished**. The user is the asset-integrity and maintenance crews.

**Say:**
> "Neither job is hard. Both are dangerous, slow and endless. The robot has to be trusted, and there are two things it must never do: paint over rust, and put a wrench on a cracked flange."

*(This sets up the critical failure modes. Come back to them on slide 8.)*

## 3. One platform, two jobs (0:40–1:05)

**On screen:** the robot (Go2 + 0–0.45 m lift mast + UR3) and two loops:
- **Rust**: detect → spray → soak 30 s → scrub (PPO) → verify (detector + Gemini) → paint
- **Leak**: gas check → walk to flange → inspect bolts and face → re-torque → verify (stripe aligned, gas falling)

Underneath: *one state machine, one camera, one arm, one decision log; gas pre-empts everything.*

**Say:**
> "It's a Unitree Go2 with a lift mast and a UR3 arm, on a full rig in Isaac Sim 6.2. At every patrol stop it checks the gas first. If the air is clean, it hunts rust: spray, soak, scrub with a reinforcement-learning policy, verify, paint. If there's gas, it finds the flange, reads the bolts, and tightens the loose one. Both jobs share one brain, and gas always wins."

## 4. Demo video (1:05–2:10)

**DEMO CUE:** say the intro line, then **play the video** (≤ 60 s, one take). The final file is **TBD**. The current renders are in `~/Downloads/rig-videos/`; `demo_combined.mp4` is the v2 scrub render. Don't talk over it except for the call-outs. Read them off the HUD as they happen.

**Intro (5 s):**
> "This is one take. We change the world while it works, and nothing tells the robot. Watch the HUD."

**Call-outs while it plays:**
1. On the detector mask or pipe map: *"That's the detector. The red mask becomes the target grid on the pipe."*
2. At the mid-run change (scripted in the sim, not announced to the robot):
   - New rust: *"New rust. It sees it and re-plans."*
   - A bolt loosening mid-scrub: *"Gas. It drops the scrub, fixes the bolt, and comes back to the scrub exactly where it left off."*
3. At BACK_OFF, HOLD or PAUSE_PERSON: *"Too much force, so it backs off"* / *"Lens blocked, so it holds"* / *"A worker walked in, so it pauses."*

**Fallback if the video fails:** open the live HUD at https://rig.zeecloud.org. If the sim isn't streaming, use the mock replay at `/?mock=1`: press Space to start, U to make an unscripted change, and ? to show the controls card.

## 5. Perception drives every action (2:10–2:40)

**On screen:** two pipelines side by side.
- **Rust**: frame → 96 px patches → MobileNetV4 + LoRA heatmap → (θ, z) grid on the pipe → spray path, scrub targets, done check. Numbers: **F1 0.977** on 881 real photos, **13/13** held-out Isaac renders.
- **Leak**: frame → project the known bolt positions → read each red torque stripe (how far each bolt turned) → classical crack check → diagnosis. Numbers: **40/40 cracks found, 0/40 false alarms**; stripe angle error ≤ 0.019 rad (about 1°).

**Say:**
> "Nothing is a fixed script. For rust, a 1-millisecond MobileNet heatmap is projected onto the pipe and becomes the work plan. For leaks we didn't need ML. Every bolt has a torque stripe. When the bolt turns, the stripe breaks, and we measure that to about a degree. A classical crack check found 40 of 40 cracks with no false alarms. Both checks have an 'unsure' answer, and unsure means take another look."

## 6. The scrub policy and the force problem (2:40–3:10)

**On screen:** a v1 vs v2 table (rig geometry, a new rust patch every episode):

| | v1 | v2 |
|---|---|---|
| Success | 100 % | 97.6 % |
| Episodes over 40 N | **76.8 %** | **1.0 %** |
| Mean / p95 max force | 49.9 / 70.1 N | 29.9 / 35.7 N |

**Say:**
> "Scrubbing is PPO in Isaac Lab across 2,048 parallel environments, with a wear model where rust comes off four times faster once it has soaked. Version 1 cleaned everything by pressing too hard: three episodes in four broke our 40-newton limit. We retrained it with a quadratic over-force penalty and a 38-newton cap. Episodes over the limit fell from 77 % to 1 %, and success only dropped to 97.6 %."

## 7. The gas-leak job: loose bolt or crack? (3:10–3:40)

**On screen:** the crack-check figure (`docs/leak/crack_check.png`: clean / cracked / cracked through a dirty lens) and three outcomes from the mock:
- loose bolts 1 and 5 → re-torque both → gas clears → **DONE**
- cracked flange → back off 3 m → **ESCALATE** (critical)
- dirty lens → 2 extra views → **ESCALATE** (unsure)

**Say:**
> "The gas sensor tells the robot that something is leaking. The camera tells it what. A loose bolt is something it may fix: it tightens the worst bolt, then checks that the stripe lines up and the gas falls. A crack it must never touch: it backs off and raises a critical alert. A dirty lens hides cracks before it hides the stripes, so a blurry 'no crack' isn't trusted. It becomes 'unsure', and a human gets the call."

## 8. When things go wrong (3:40–4:05)

**On screen:** the failure-mode table (the critical rows are highlighted) next to a HUD screenshot (`docs/img/hud_mock_backoff.png`, the mock replay at BACK_OFF):

| Failure | Response |
|---|---|
| **Paints over rust** | VERIFY: < 5 % rust, ≥ 80 % visible; Gemini veto; 3 failed cycles → ESCALATE |
| **Tightens a cracked flange** | crack possible → never touch bolts; back off 3 m, critical alert |
| Gas ≥ 0.20 LEL or sensor fault | GAS_RETREAT (fail-safe) |
| Gas during rust work | pause, fix the leak, resume |
| Force > 40 N | BACK_OFF: retract 5 cm, press × 0.7 |
| Person < 3 m · camera blocked | PAUSE_PERSON · HOLD |

**Say:**
> "Before every step, the checks run in a fixed order: is the gas sensor working, is gas below the limit, is anyone near. The most severe hazard wins. Every response is on the HUD and in a decision log, with the evidence behind it."

## 9. Edge autonomy, and what's coming (4:05–4:35)

**On screen:**
- **On the robot (CPU):** rust detector 1.1 ms; scrub policy 0.012 ms; leak check about 90 ms per frame (measured on a laptop CPU); state machine. TensorRT on the GB10: **TBD**.
- **Cloud, optional:** the Gemini second opinion for rust, asynchronous.
- **Training only:** Modal A10G / L40S and our GB10.
- **Coming (TBD):** unit 2, a gas robot with methane %LEL and H₂S sensors, an OGI-style camera and an acoustic sensor, using cast-and-surge source localisation. Also a 3D plume with leak rate (orifice equation), a fleet planner with gas pre-emption, storm adaptation and Go2 walking.

**Say:**
> "The whole fast loop runs on the robot's CPU. Gemini is advice; if the satellite link drops, behaviour is unchanged. Next is a second robot built for gas: it smells its way to the source, and a fleet planner sends the nearest unit when gas appears."

*(Cut the "coming" sentence if you're over time.)*

## 10. Limits and close (4:35–5:00)

**On screen:** limits on the left, the tagline on the right.
- Simulation only. The robot is posed kinematically, the brush force is a spring model, and bolt torque isn't simulated.
- The gas model and thresholds are illustrative. The leak checks still need re-testing on Isaac stream frames. The rust render test is small (13 crops).

Tagline: *Find it. Fix it. Prove it. Ask when unsure.*

**Say:**
> "To be clear: this is simulation, and our gas numbers are illustrative. But the logic is the point. One robot finds rust and leaks, fixes the routine ones, proves they're fixed, and asks a human about everything else. Thank you."

---

## Q&A cheat sheet (keep each answer to about 30 s)

**Is this just a scripted animation?**
No. Every transition comes from perception. The rust mask becomes the target grid, the stripe reading picks which bolt to tighten, and the state machine moves on only when the camera or the gas sensor confirms the result. The mid-run changes, new rust or a bolt working loose, are scripted in the sim but aren't announced to the robot. Each decision is logged with its evidence in `decisions.jsonl`.

**Why no ML for the leak checks?**
We didn't need it, and a classical check is easier to trust. A painted torque stripe is an industry convention: if it's broken, the bolt has turned. The crack check looks for long, thin, dark shapes on the flange face and measures their length in millimetres. It found 40/40 cracks with 0/40 false alarms on held-out renders; 6/40 clean flanges came back "unsure", which costs a second look, not a miss. An ONNX crack-model path exists (validation F1 0.98 after one epoch), but it is off by default.

**What if it tightens a bolt on a cracked flange?**
That's the failure we designed against. A crack probability ≥ 0.5 is never a robot repair. While a crack is merely possible (unsure), no bolt is touched. And a dirty lens can't produce a "no crack", because view quality below 0.85 turns it into "unsure". The safety sweep in the tests checks that it never tightens a cracked flange, never tightens in gas at the limit, and never finishes "done" with a loose bolt left.

**What do the gas numbers mean?**
They are fractions of the lower explosive limit (LEL): above 0.05 means investigate, and 0.20 or more means stop and retreat. They are illustrative, not from a site standard; the gas model is a simple distance-decay field. A real plume model and a dedicated gas robot are in progress (TBD).

**How does it handle a leak in the middle of a scrub?**
The gas check runs before every scrub step. Above 0.05 LEL it pauses the scrub, walks to the flange, re-torques the bolt (bolt 2 in the mock), checks that the stripe is aligned and the gas is falling, then resumes the scrub in the same state and finishes the rust job. In the mock that run ends DONE.

**How good is the rust detector, really?**
On 881 real photos it scores F1 0.977, recall 0.983 and AUROC 0.998. On held-out Isaac renders it gets 13/13, against 10/13 before we added sim-style data. The caveats: the render set is small, and one real source of weathered surfaces gives up to 13 % false positives. We set the threshold for recall, because a miss is the costly error.

**Why did scrub success drop from 100 % to 97.6 %?**
That was deliberate. v1 got 100 % by pressing too hard: 76.8 % of its episodes went over 40 N, with a peak of 87.5 N. v2 keeps 99 % of episodes under the limit. The episodes it doesn't finish go back to VERIFY, which scrubs again or escalates. It never paints over them.

**What stops it damaging the pipe?**
There are three layers: the policy is trained against over-force; the setpoint is clamped to 38 N outside the network; and the orchestrator's hard 40 N gate backs off (retract 5 cm, press × 0.7) and escalates after 3 back-offs.

**What if Gemini is wrong or offline?**
Gemini is only consulted for rust, at two moments: when detection is still ambiguous, and just before painting. It can veto a paint, at most twice, but it can't make the robot skip a check. With no key, no network or a bad reply, behaviour is unchanged. The leak job doesn't use Gemini at all.

**What runs on the edge?**
Everything in the fast loop: the rust detector (ONNX on CPU, 1.1 ms), the leak check (numpy, about 90 ms per frame on a laptop CPU, 3 frames per view), the scrub policy (0.012 ms) and the state machine. GB10 TensorRT numbers are TBD. Training used Modal GPUs and the GB10.

**Why one robot for two jobs?**
Both jobs need the same things: walk the rig, look closely at pipework, use an arm, and know when to stop. Sharing one state machine means one set of safety rules, and gas pre-emption works across jobs for free. A dedicated gas unit and a fleet planner are the next step (TBD).

**How far is this from a real robot?**
The perception and decision logic carry over as they are, since they run on CPU from camera frames. The gaps are a real force-torque sensor in place of the spring model, real bolt torque and thread friction, legs that actually walk under arm load (Go2 walking is TBD), a site-standard gas threshold, and real painted-steel and flange data.

**What did you build in the hackathon versus reuse?**
We built the rust data pipeline and sim-style data, the detector fine-tune, the localizer, the scrub environment and wear model, both PPO policies, the lift-mast robot, the orchestrator and leak job, the stripe and crack checks, the flange scene, the Isaac bridge, the HUD, and the GB10 stream. We reused the Go2 and UR3 models, the Poly Haven CC0 textures, the CC BY 4.0 rig model, and Isaac Sim / Isaac Lab.

**What's the cost argument?**
Don't invent numbers. Say: "Today every one of these tasks means people flown offshore and working at height or next to a leak. The robot does the routine ones and sends the exceptions to a person."
