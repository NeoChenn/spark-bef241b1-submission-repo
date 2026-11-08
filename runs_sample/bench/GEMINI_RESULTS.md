# Gemini 4 Argon as the scrub controller: results

Model `gemini-4-argon`, thinking `low`, CPU kinematic mock (`bench/scrub_sim.py`), 30 identical seeded scenarios per arm (first 30 of the 200 CPU-mock training scenarios), 95% CI (Wilson for rates, t for means). Total API spend for this run $2.61. Source: `scrub_gemini.json`, chart `scrub_vs_gemini.png`, code `bench/scrub_vs_gemini.py` + `bench/scrub_gemini.py`.

Time models for Gemini:
- **Unlimited time** (best case for Gemini): the sim pauses while Gemini thinks; each reply is applied at once and held for 0.5 s of sim time, so it gets 2 decisions per sim second.
- **Real time** (honest deployment): the sim keeps running during the measured API latency; the brush holds the last command (starts lifted) and each reply lands when it arrives, computed from the state at request time.
- Time budget = PPO's mean CPU-mock clean time, 7.45 s (n=200).

## Main table

| controller | n | success | success within PPO time | time to clean s | rust removed | removed %/s | decision latency (median) | decisions/s | calls/ep | wall s/ep | API cost/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Random | 30 | 0.0% [0.0%, 11.3%] | 0.0% [0.0%, 11.3%] | - | 30.5% | 2.04 [1.65, 2.43] | 0.045 ms | 30 (every step) | - | 2.60 | $0 (on robot) |
| Greedy scripted (before RL) | 30 | 30.0% [16.7%, 47.9%] | 0.0% [0.0%, 11.3%] | 12.53 [11.41, 13.65] | 68.8% | 4.99 [4.19, 5.80] | 0.081 ms | 30 (every step) | - | 2.51 | $0 (on robot) |
| PPO v2 (after RL) | 30 | 93.3% [78.7%, 98.2%] | 53.3% [36.1%, 69.8%] | 7.57 [6.67, 8.47] | 94.5% | 13.32 [11.46, 15.18] | 0.074 ms | 30 (every step) | - | 1.17 | $0 (on robot) |
| Gemini, unlimited time | 30 | 36.7% [21.9%, 54.5%] | 0.0% [0.0%, 11.3%] | 11.49 [10.11, 12.87] | 83.0% | 6.35 [5.54, 7.15] | 8.5 s | 2.02 | 27.7 | 297.83 | $0.081 |
| Gemini, real time | 30 | 0.0% [0.0%, 11.3%] | 0.0% [0.0%, 11.3%] | - | 3.2% | 0.21 [0.08, 0.35] | 8.0 s | 0.14 | 2.2 | 14.97 | $0.006 |

## Headline

- Before RL: random 0.0% [0.0%, 11.3%], greedy scripted 30.0% [16.7%, 47.9%] success.
- After RL: PPO v2 93.3% [78.7%, 98.2%] success, 53.3% [36.1%, 69.8%] within 7.45 s; decides in 0.074 ms on CPU, $0 per episode.
- Gemini, unlimited time: 36.7% [21.9%, 54.5%] success, 83% of rust removed, 28 calls and 298 s wall per 15 s episode (~20x slower than real time), $0.081/episode.
- Gemini, real time: 0.0% [0.0%, 11.3%] success, 3% of rust removed; at 0.14 decisions/s it gets ~2.2 calls per 15 s episode.
- Decision latency: PPO 0.074 ms on CPU vs Gemini 8.5 s/call (~114,312x).

## Reference at larger n (same arms)

| controller | CPU mock n=200 success | Isaac n=256 success | Isaac time to clean s | Isaac peak N | Isaac >40 N |
|---|---|---|---|---|---|
| PPO v2 (after RL) | 97.0% [93.6%, 98.6%] | 95.3% [92.0%, 97.3%] | 7.05 [6.72, 7.38] | 30.7 [29.7, 31.6] | 1.9% [0.8%, 4.5%] |
| Greedy scripted (before RL) | 32.0% [25.9%, 38.8%] | 32.0% [26.6%, 38.0%] | 10.96 [10.33, 11.59] | 38.8 [37.4, 40.3] | 46.5% [40.5%, 52.6%] |
| Random | 0.0% [0.0%, 1.9%] | 0.0% [0.0%, 1.5%] | - | 49.2 [47.4, 51.0] | 68.4% [62.4%, 73.8%] |

Caveats: kinematic CPU mock, not PhysX (PPO/greedy/random success matches Isaac within CI); mock contact forces are unreliable, so force claims use the Isaac numbers; Gemini was not run in Isaac (API latency would need a paused PhysX sim). Gemini sees a JSON state summary (brush pose, force, nearest rust cells, rust centroid), not the 327-D observation; it commands the same [d_theta, d_z, press] target interface as greedy and random.

## Gemini vision arm (rust detection, `detect.json`, `detect.png`)

Same GeminiVerifier "rust" prompt; score = confidence if rust, 1-confidence if clean, 0.5 if cannot_tell; threshold 0.5. All 13 held-out sim crops + a stratified 60-image sample (seed 0) of the 485-image synthetic test split (regenerated locally with `detection/synth_sim.generate(seed=0)`, identical 485/202 split). Bootstrap 95% CIs in `detect.json`.

| model | held-out sim F1 | held-out sim AUROC | synth F1 | synth AUROC | cannot_tell | latency/image median | errors |
|---|---|---|---|---|---|---|---|
| MNv4 full-FT (sim-only, on robot) | 0.91 | 0.98 | 0.995 (n=485) | 1.00 | - | 2.6 ms CPU | 0 |
| Colour stand-in (on robot) | 0.71 | 0.71 | 0.72 (n=485) | 0.78 | - | 4.3 ms CPU | 0 |
| gemini-4-argon | 0.80 | 0.96 | 0.72 (n=60) | 0.97 | 31% / 22% | 9.5 s / 8.7 s API | 0 |
| gemini-robotics-er-2-preview | 0.62 (n=12) | 0.86 | 0.59 (n=46) | 0.93 | 25% / 15% | 3.5 s API | 15 of 73 calls failed |

Gemini ranks well (AUROC 0.93-0.97) but abstains often ("cannot_tell"), which pulls F1 down; the on-robot MNv4 is better on both sets and ~3,000x faster. The deployed LoRA checkpoint was not available here (published: real-photo F1 0.977, 13/13 held-out sim).

## Gemini planner arm (orchestration, `orch.json`, `orch.png`)

fleet/ coordinator + CPU sim, seeded 15-min shifts (leak, worker walk-up, SUPV approvals); planner = OrchestrationAgent on gemini-4-argon (thinking low) re-planning at t=0 and on every finding; agent calls block the sim clock. Rule-based n=100, Gemini arms n=10 shifts each (first 10 rule seeds), total API spend $1.08. Mean [95% CI].

| arm | n | leak onset to isolated s | all work done at s | jobs/h | approvals requested | unsafe executed | LLM requests/shift | LLM latency/request | API cost/shift |
|---|---|---|---|---|---|---|---|---|---|
| Rule-based coordinator + operator | 100 | 398.9 [387.1, 410.6] | 788.6 [776.0, 801.2] | 88.1 | 2.0 | 0 | - | - | $0 |
| Gemini planner + coordinator | 10 | 418.0 [352.5, 483.5] | 812.1 [773.0, 851.2] | 85.2 | 3.6 | 0 | 6.7 | 17.1 s | $0.073 |
| Gemini planner, link cut mid-shift | 10 | 413.8 [349.7, 477.9] | 797.5 [755.5, 839.5] | 85.2 | 2.2 | 0 | 3.4 | 22.3 s | $0.035 |

On the same 10 seeds the rule-based arm isolates the leak in 428.4 s mean and finishes all work at 805.4 s, so the Gemini planner is within noise of the coordinator alone (10 s faster isolation, 7 s later finish), never executes an unsafe action, and asks for more approvals. With the link cut the coordinator carries on. Without a key the planner arm reproduces the rule-based arm exactly (20/20 shifts).
