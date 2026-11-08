# Scrub control benchmark

## CPU kinematic mock, train scenarios

| method | n | success | time to clean s | peak force N | p95 N | >40 N | removed %/s | decision latency | cost/ep |
|---|---|---|---|---|---|---|---|---|---|
| PPO v2 (ONNX) | 200 | 97.0% [93.6%, 98.6%] | 7.45 [7.05, 7.85] | 34.8 [33.7, 36.0] | 43.08 | 19.5% [14.6%, 25.5%] | 14.82 [13.83, 15.80] | 0.013 ms (p90 0.015) | $0 (on robot) |
| Greedy scripted | 200 | 32.0% [25.9%, 38.8%] | 10.87 [10.25, 11.48] | 35.8 [34.3, 37.3] | 49.11 | 35.0% [28.7%, 41.8%] | 5.49 [5.05, 5.93] | 0.008 ms (p90 0.147) | $0 (on robot) |
| Random | 200 | 0.0% [0.0%, 1.9%] | - | 42.6 [40.7, 44.4] | 59.08 | 58.0% [51.1%, 64.6%] | 1.00 [0.87, 1.13] | 0.021 ms (p90 0.038) | $0 (on robot) |
| Gemini controller | 30 | see "Gemini 4 Argon as the controller" below | | | | | | | |

## Isaac Lab, train scenarios

| method | n | success | time to clean s | peak force N | p95 N | >40 N | removed %/s | decision latency | cost/ep |
|---|---|---|---|---|---|---|---|---|---|
| PPO v2 | 256 | 95.3% [92.0%, 97.3%] | 7.05 [6.72, 7.38] | 30.7 [29.7, 31.6] | 35.85 | 1.9% [0.8%, 4.5%] | 15.22 [14.35, 16.09] | 8.34 ms / 256 envs batched | $0 (on robot) |
| Greedy scripted | 256 | 32.0% [26.6%, 38.0%] | 10.96 [10.33, 11.59] | 38.8 [37.4, 40.3] | 53.7 | 46.5% [40.5%, 52.6%] | 5.65 [5.17, 6.12] | 37.97 ms / 256 envs batched | $0 (on robot) |
| Random | 256 | 0.0% [0.0%, 1.5%] | - | 49.2 [47.4, 51.0] | 68.95 | 68.4% [62.4%, 73.8%] | 0.95 [0.84, 1.06] | 8.87 ms / 256 envs batched | $0 (on robot) |
| Gemini controller | - | not run in Isaac (CPU mock only) | | | | | | | |

## CPU kinematic mock, rig scenarios

| method | n | success | time to clean s | peak force N | p95 N | >40 N | removed %/s | decision latency | cost/ep |
|---|---|---|---|---|---|---|---|---|---|
| PPO v2 (ONNX) | 200 | 96.0% [92.3%, 98.0%] | 7.70 [7.32, 8.08] | 34.2 [33.5, 35.0] | 43.06 | 19.5% [14.6%, 25.5%] | 13.66 [12.92, 14.41] | 0.012 ms (p90 0.014) | $0 (on robot) |
| Greedy scripted | 200 | 28.0% [22.2%, 34.6%] | 11.14 [10.48, 11.81] | 35.2 [34.0, 36.4] | 49.31 | 32.5% [26.4%, 39.3%] | 5.13 [4.72, 5.55] | 0.008 ms (p90 0.161) | $0 (on robot) |
| Random | 200 | 0.0% [0.0%, 1.9%] | - | 41.4 [39.9, 42.8] | 57.27 | 55.0% [48.1%, 61.7%] | 0.95 [0.82, 1.07] | 0.019 ms (p90 0.032) | $0 (on robot) |
| Gemini controller | - | train preset only (section below) | | | | | | | |

## Isaac Lab, rig scenarios

| method | n | success | time to clean s | peak force N | p95 N | >40 N | removed %/s | decision latency | cost/ep |
|---|---|---|---|---|---|---|---|---|---|
| PPO v2 | 256 | 97.3% [94.5%, 98.7%] | 7.06 [6.74, 7.39] | 29.9 [29.4, 30.3] | 34.91 | 1.2% [0.4%, 3.4%] | 15.39 [14.56, 16.22] | 8.33 ms / 256 envs batched | $0 (on robot) |
| Greedy scripted | 256 | 33.2% [27.7%, 39.2%] | 11.17 [10.57, 11.77] | 38.3 [37.1, 39.5] | 53.03 | 46.5% [40.5%, 52.6%] | 5.66 [5.24, 6.09] | 37.32 ms / 256 envs batched | $0 (on robot) |
| Random | 256 | 0.0% [0.0%, 1.5%] | - | 48.6 [46.9, 50.2] | 70.03 | 68.0% [62.0%, 73.4%] | 0.93 [0.82, 1.04] | 7.42 ms / 256 envs batched | $0 (on robot) |
| Gemini controller | - | not run in Isaac (CPU mock only) | | | | | | | |

## Isaac Lab prior evals (existing files)

| run | n | success | time to clean s | mean peak N | p95 N | >40 N |
|---|---|---|---|---|---|---|
| v2 best (model_149, v2c) | 1026 | 97.6% [96.4%, 98.3%] | 6.75 | 29.9 | 35.7 | 1.0% [0.5%, 1.8%] |
| v2a ablation | 512 | 99.8% [98.9%, 100.0%] | 5.98 | 39.5 | 53.2 | 43.2% [38.9%, 47.5%] |
| v2b ablation | 512 | 99.0% [97.7%, 99.6%] | 6.26 | 32.9 | 42.1 | 7.6% [5.6%, 10.2%] |
| v1 (model_149) | 513 | 100.0% [99.3%, 100.0%] | 5.19 | 49.9 | 70.1 | 76.8% [73.0%, 80.2%] |
| v1 domain-rand | 513 | 100.0% [99.3%, 100.0%] | 4.09 | 49.8 | 71.5 | 78.8% [75.0%, 82.1%] |

## CPU mock joint-lag sensitivity

| arm | success | peak N | >40 N |
|---|---|---|---|
| PPO v2 (ONNX) tau=0.05 | 94.0% [87.5%, 97.2%] | 34.7 [33.5, 35.8] | 21.0% [14.2%, 30.0%] |
| Greedy scripted tau=0.05 | 31.0% [22.8%, 40.6%] | 36.0 [34.2, 37.7] | 38.0% [29.1%, 47.8%] |
| PPO v2 (ONNX) tau=0.1 | 94.0% [87.5%, 97.2%] | 32.3 [31.6, 33.1] | 5.0% [2.1%, 11.2%] |
| Greedy scripted tau=0.1 | 10.0% [5.5%, 17.4%] | 34.1 [32.4, 35.7] | 27.0% [19.3%, 36.4%] |

## Gemini 4 Argon as the controller

CPU kinematic mock, gemini-4-argon (thinking low), 30 identical seeded scenarios per arm (`scrub_gemini.json`, chart `scrub_vs_gemini.png`). Total API spend $2.61.

| controller | n | success | success within PPO time | time to clean s | rust removed | removed %/s | decision latency (median) | decisions/s | calls/ep | wall s/ep | API cost/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Random | 30 | 0.0% [0.0%, 11.3%] | 0.0% [0.0%, 11.3%] | - | 30.5% | 2.04 [1.65, 2.43] | 0.045 ms | 30 (every step) | - | 2.60 | $0 (on robot) |
| Greedy scripted (before RL) | 30 | 30.0% [16.7%, 47.9%] | 0.0% [0.0%, 11.3%] | 12.53 [11.41, 13.65] | 68.8% | 4.99 [4.19, 5.80] | 0.081 ms | 30 (every step) | - | 2.51 | $0 (on robot) |
| PPO v2 (after RL) | 30 | 93.3% [78.7%, 98.2%] | 53.3% [36.1%, 69.8%] | 7.57 [6.67, 8.47] | 94.5% | 13.32 [11.46, 15.18] | 0.074 ms | 30 (every step) | - | 1.17 | $0 (on robot) |
| Gemini, unlimited time | 30 | 36.7% [21.9%, 54.5%] | 0.0% [0.0%, 11.3%] | 11.49 [10.11, 12.87] | 83.0% | 6.35 [5.54, 7.15] | 8.5 s | 2.02 | 27.7 | 297.83 | $0.081 |
| Gemini, real time | 30 | 0.0% [0.0%, 11.3%] | 0.0% [0.0%, 11.3%] | - | 3.2% | 0.21 [0.08, 0.35] | 8.0 s | 0.14 | 2.2 | 14.97 | $0.006 |

- scenarios: first 30 of the 200 seeded CPU-mock training scenarios (bench.stats.episode_seeds(200, base=0)); every arm sees the same ones
- unlimited time: sim paused while Gemini thinks; each reply applied immediately and held for 0.5 s of sim time (2 decisions per sim second), best case for Gemini
- real time: sim keeps running during the measured API latency; the brush holds the last command (starts lifted) and each reply is applied when it arrives, computed from the state at request time
- success within PPO time: cleaned (<= 5% rust left) within 7.45 s, PPO's mean CPU-mock clean time (n=200)
- forces: CPU-mock contact forces are not reliable; use the Isaac numbers for force claims
- wall time: local arms: CPU compute per episode; Gemini unlimited: sum of API round trips (~20x slower than the 15 s sim episode it controls). Gemini real time: a deployment is bound to the 15 s episode; the measured 110 s/ep here is CPU contention from 8 concurrent Python sims in one process, not API time (~2.2 calls x 8 s)
- latency: all arms in this run share a contended CPU; uncontended PPO ONNX latency is 0.013 ms median (scrub.json, n=200 run)
- Decision latency: PPO 0.074 ms on CPU vs Gemini 8.5 s/call (~114,312x).

Reference, same arms at larger n:

| controller | CPU mock n=200 success | Isaac n=256 success | Isaac time to clean s | Isaac peak N | Isaac >40 N |
|---|---|---|---|---|---|
| PPO v2 (after RL) | 97.0% [93.6%, 98.6%] | 95.3% [92.0%, 97.3%] | 7.05 [6.72, 7.38] | 30.7 [29.7, 31.6] | 1.9% [0.8%, 4.5%] |
| Greedy scripted (before RL) | 32.0% [25.9%, 38.8%] | 32.0% [26.6%, 38.0%] | 10.96 [10.33, 11.59] | 38.8 [37.4, 40.3] | 46.5% [40.5%, 52.6%] |
| Random | 0.0% [0.0%, 1.9%] | 0.0% [0.0%, 1.5%] | - | 49.2 [47.4, 51.0] | 68.4% [62.4%, 73.8%] |
