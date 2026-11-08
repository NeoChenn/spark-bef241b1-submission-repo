# Pitch: Rig Rust Repair (5 minutes)

Agentverse 2026, Robotics track. There are 10 slides; the deck is `docs/pitch.html` (← → or space to move, F for fullscreen; `docs/pitch.pdf` is the same deck). The one-page summary is `docs/technical_summary.pdf`.

The judging weights tell us where to spend the time. Perception-driven action (25 %) is covered on slides 4–5, adaptation (25 %) on slides 4 and 7, edge autonomy (15 %) on slide 8, relevance and the critical failure mode (20 %) on slides 2, 6 and 7, and ambition and quality (15 %) throughout.

Items marked **TBD** are still being produced by other agents: the live repair-loop numbers, HUD screenshots, the final video file and Gemini latency. Fill them in before the talk, or drop the line.

| # | Slide | Time | Cumulative |
|---|---|---|---|
| 1 | Hook | 0:20 | 0:20 |
| 2 | The problem | 0:30 | 0:50 |
| 3 | What we built | 0:25 | 1:15 |
| 4 | **Demo video** (≤ 60 s) | 1:05 | 2:20 |
| 5 | Perception drives every action | 0:35 | 2:55 |
| 6 | The scrub policy and the force problem | 0:35 | 3:30 |
| 7 | When things go wrong | 0:30 | 4:00 |
| 8 | Edge autonomy: where everything runs | 0:25 | 4:25 |
| 9 | Limits, honestly | 0:20 | 4:45 |
| 10 | Close | 0:15 | 5:00 |

---

## 1. Hook (0:00–0:20)

**On screen:** a hero still of the night rig scene with the Go2 at the pipe (from the stream at rig.zeecloud.org or the cinematic render). Title: *Rig Rust Repair: a robot dog that finds and fixes pipe corrosion.*

**Say:**
> "Every offshore platform is slowly rusting. Today the fix is a person on a rope, 30 metres up, in salt spray, next to live hydrocarbons, with a wire brush. We built a robot that finds the rust, repairs it, checks its own work, and calls a human only when it isn't sure."

## 2. The problem (0:20–0:50)

**On screen:** three words, each with one line under it.
- **Dangerous**: height, weather, the chance of a gas release.
- **Costly**: every repair needs people flown offshore.
- **Never finished**: kilometres of pipe per platform, corroding again.

User: the asset-integrity and maintenance crews on the platform.

**Say:**
> "The users are the asset-integrity crews. Corrosion is their chronic job. It isn't hard work, but it is dangerous, slow and endless. That is the kind of work a robot should take over. The robot has to be trusted, though. The worst thing it can do is paint over rust and hide it."

*(That last line sets up the critical failure mode. Come back to it on slide 7.)*

## 3. What we built (0:50–1:15)

**On screen:** the robot (Unitree Go2 + 0–0.45 m lift mast + UR3 arm) and the repair loop as a strip:
`detect → spray → soak 30 s → scrub (PPO) → verify (detector + Gemini) → paint`.

**Say:**
> "It's a Unitree Go2 quadruped with a lift mast and a UR3 arm, on a full offshore rig in Isaac Sim 6.2. It patrols, spots rust, sprays remover, waits 30 seconds for it to soak, scrubs with a reinforcement-learning policy, checks the result with its own detector and a Gemini second opinion, and only then paints."

## 4. Demo video (1:15–2:20)

**DEMO CUE:** say the intro line, then **play the video** (≤ 60 s, one take). File: **TBD** (it will be in `~/Downloads/rig-videos/`; check the name with the video agent). Don't talk over it except for the three call-outs below. Read them off the HUD as they happen.

**Intro (5 s):**
> "This is one take. Nothing is scripted except that we change the world while it works. Watch the HUD in the corner."

**Call-outs while it plays:**
1. When the detector inset shows the red mask: *"That's the detector. The red mask becomes the robot's target grid on the pipe."*
2. At the unscripted change (a new rust patch appears mid-soak or mid-scrub): *"New rust. It wasn't told. It sees it, re-plans and sprays the new patch."* The HUD shows `RE-PLAN`.
3. At BACK_OFF, HOLD or PAUSE_PERSON (whichever the take shows): *"Too much force, so it backs off"* / *"Lens blocked, so it holds"* / *"A worker walked in, so it pauses."*

**Fallback if the video fails:** open https://rig.zeecloud.org (the live stream), or show HUD screenshots (**TBD**) and talk through the `runs/decisions.jsonl` trace.

## 5. Perception drives every action (2:20–2:55)

**On screen:** the pixels-to-target pipeline: camera frame → 96 px patches → heatmap → mask → ray–cylinder projection → (θ, z) grid on the pipe → spray path / scrub targets / done check. Detector card: MobileNetV4-Small + LoRA, 0.26 M trainable parameters; **F1 0.977** on 881 real photos; **13/13** on held-out Isaac renders (the previous model got 10/13).

**Say:**
> "Nothing here is a fixed script. The classifier is a MobileNetV4 with LoRA adapters, 1.1 milliseconds per crop on a CPU. We slide it over the frame to get a heatmap and project that onto the pipe's surface. That grid decides where to spray, where to scrub, and when the job is done. Our first model worked on real photos but missed rendered rust, so we generated sim-style training data from the scene's own textures. On held-out renders it went from 10 out of 13 to 13 out of 13."

## 6. The scrub policy and the force problem (2:55–3:30)

**On screen:** a two-column v1 vs v2 table (rig geometry, a new rust patch injected every episode):

| | v1 | v2 |
|---|---|---|
| Success | 100 % | 97.6 % |
| Episodes over 40 N | **76.8 %** | **1.0 %** |
| Mean / p95 max force | 49.9 / 70.1 N | 29.9 / 35.7 N |

**Say:**
> "Scrubbing is a PPO policy trained in Isaac Lab across 2,048 parallel environments, with a physical wear model: rust comes off in proportion to force times speed, four times faster once it has soaked. Version 1 cleaned every pipe, but it did that by pressing too hard. Three episodes in four went over our 40-newton limit. We retrained it with a quadratic over-force penalty and a 38-newton cap on the setpoint. Episodes over the limit fell from 77 % to 1 %, and success only dropped to 97.6 %. We would rather have a robot that is slightly slower and doesn't damage the pipe."

## 7. When things go wrong (3:30–4:00)

**On screen:** the failure-mode table (the critical row is highlighted):

| Failure | Response |
|---|---|
| **Paints over rust** (critical) | VERIFY needs < 5 % rust, 80 % of the target visible, few uncertain cells; Gemini can veto; 3 failed cycles → **ESCALATE_TO_HUMAN** with evidence |
| Over-force > 40 N | **BACK_OFF**: retract 5 cm, press × 0.7; 3× → escalate |
| Ambiguous (p 0.35–0.65) | up to 2 **SECOND_VIEW**s → Gemini → human |
| Camera blocked | **HOLD**; escalate after 5 s |
| Person < 3 m | **PAUSE_PERSON**; resume when clear |
| New rust mid-run | **RE-PLAN**: spray it, extend the soak |

**Say:**
> "Here's the failure that matters: painting over rust. Before it paints, the robot has to see less than 5 % rust over most of the patch. Gemini Robotics-ER gets a veto, and if it still isn't clean after three tries, the robot stops and sends a human a work order with the evidence. Every one of these responses is on the HUD and in a decision log, with the confidence and force that caused it."

## 8. Edge autonomy: where everything runs (4:00–4:25)

**On screen:** a three-zone diagram.
- **On the robot (CPU)**: detector 1.1 ms, scrub policy 0.012 ms (measured on a laptop CPU), orchestrator at 10 Hz.
- **Cloud, optional**: Gemini second opinion, asynchronous, latency **TBD**.
- **Training only**: Modal A10G (detector), Modal L40S and our GB10 (PPO, 2,048 envs).
- The sim renders on an NVIDIA GB10 and streams to the browser.

**Say:**
> "Everything in the fast loop runs on the robot's CPU: perception, the policy and the state machine. Gemini is advice. If the network drops, the robot behaves exactly as it does without it, which matters on a platform with a satellite link."

## 9. Limits, honestly (4:25–4:45)

**On screen:** four bullets.
- Simulation only. The robot is posed kinematically in the demo bridge, and the brush force is a spring model.
- The scrub policy is trained on grid state that the localizer fills in deployment.
- The sim-render check is small (13 crops).
- Next steps: walking legs under arm load, a real force-torque sensor, and real painted-steel data.

**Say:**
> "To be clear about what this is: it's simulation. The brush force is a model, not a sensor, and our sim-render test is small. The next steps are real force sensing and real painted-steel data."

## 10. Close (4:45–5:00)

**On screen:** one line, *Find it. Fix it. Prove it. Ask when unsure.*, with the live stream URL and the repo.

**Say:**
> "A robot that finds rust, fixes it, proves it's fixed, and asks a human when it isn't sure. Thank you."

---

## Q&A cheat sheet (keep each answer to about 30 s)

**Is this just a scripted animation?**
No. Every transition comes from perception. The detector's mask becomes the target grid, and the state machine moves on only when the camera confirms it. The new rust in the video isn't announced to the robot; it finds it by re-detecting every 5 s during the soak. Each decision is logged with its evidence in `decisions.jsonl`.

**How good is the detector, really?**
On 881 real photos it scores F1 0.977, recall 0.983 and AUROC 0.998. On held-out Isaac renders it gets 13/13 crops right, against 10/13 for the model trained without sim-style data. The caveats: the render set is small, and one real source of weathered surfaces gives up to 13 % false positives. We set the threshold for recall, because a miss is the costly error.

**Why MobileNetV4 and not a bigger model?**
DINOv2-S is about 1.5 F1 points better on real photos, but it is 30× the compute and about 50× the CPU latency (68 ms against 1.1 ms on 4 threads). Before the sim-style data was added, MobileNetV4 also handled the sim look better than the ViT (synthetic F1 0.795 against 0.419). LoRA trains 0.26 M parameters and matched full fine-tuning (F1 0.977 against 0.972).

**Why did success drop from 100 % to 97.6 %?**
That was deliberate. v1 got 100 % by pressing too hard: 76.8 % of its episodes went over 40 N, with a peak of 87.5 N. v2 keeps 99 % of episodes under the limit, and its mean time to clean is 6.8 s against 5.2 s. The episodes it doesn't finish go back to VERIFY, which scrubs again or escalates. It never paints over them.

**What stops the robot from damaging the pipe?**
There are three layers: the policy is trained against over-force; its setpoint is clamped to 38 N outside the network; and the orchestrator has a hard 40 N gate that backs off (retract 5 cm, press × 0.7) and escalates after 3 back-offs.

**What if Gemini is wrong or offline?**
Gemini is only consulted at two moments: when detection is still ambiguous, and just before painting. It can veto a paint, at most twice, but it can't make the robot skip a check. With no key, no network or a bad reply, the orchestrator does exactly what it would do without Gemini.

**What runs on the edge?**
The detector (ONNX on CPU, 1.1 ms), the localizer, the scrub policy (an ONNX MLP, 0.012 ms measured on a laptop CPU) and the state machine. Training used Modal GPUs and the GB10; at run time nothing is remote except the optional Gemini call.

**How does it handle the mid-run change?**
During the soak the robot re-detects every 5 s, and during the scrub every 30 control steps. A new connected rust region becomes a new tracked patch. The robot sprays it and restarts the 30 s soak clock, then scrubs all the patches. The PPO policy was also trained with a new patch injected mid-episode (30 % of training episodes, and every evaluation episode).

**Why spray and soak? Why not just scrub?**
In our wear model, rust comes off 4× faster once it has been sprayed and soaked for 30 s (1.5× while it is still soaking). The policy is penalised for scrubbing unsprayed areas, so the chemistry is part of what it learned, not decoration.

**How far is this from a real robot?**
The perception and the decision logic carry over as they are, since they run on CPU from camera frames. The arm control is trained in sim with domain randomisation (stiffness 1.5–4 kN/m, abrasion ±50 %, pipe pose). The gaps are a real force-torque sensor in place of the spring model, legs that actually walk under arm load, and painted-steel training data.

**What did you build in the hackathon versus reuse?**
We built the dataset pipeline, the sim-style synthetic data, the detector fine-tune, the localizer, the orchestrator, the scrub environment and rust physics, both PPO policies, the lift-mast robot, the Isaac bridge, and the GB10 stream. We reused the Go2 and UR3 models, the Poly Haven CC0 textures, the CC BY 4.0 rig model, and Isaac Sim / Isaac Lab.

**What's the cost argument?**
Don't invent numbers. Say: "Every repair today needs people flown out and working at height. The robot does the routine patches and sends the hard ones to a human, so people go offshore for exceptions, not for routine work."
