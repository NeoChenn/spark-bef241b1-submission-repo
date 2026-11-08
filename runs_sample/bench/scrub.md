# Scrub control benchmark

## CPU kinematic mock, train scenarios

| method | n | success | time to clean s | peak force N | p95 N | >40 N | removed %/s | decision latency | cost/ep |
|---|---|---|---|---|---|---|---|---|---|
| PPO v2 (ONNX) | 200 | 97.0% [93.6%, 98.6%] | 7.45 [7.05, 7.85] | 34.8 [33.7, 36.0] | 43.08 | 19.5% [14.6%, 25.5%] | 14.82 [13.83, 15.80] | 0.017 ms (p90 0.022) | $0 (on robot) |
| Greedy scripted | 200 | 32.0% [25.9%, 38.8%] | 10.87 [10.25, 11.48] | 35.8 [34.3, 37.3] | 49.11 | 35.0% [28.7%, 41.8%] | 5.49 [5.05, 5.93] | 0.013 ms (p90 0.198) | $0 (on robot) |
| Random | 200 | 0.0% [0.0%, 1.9%] | - | 42.6 [40.7, 44.4] | 59.08 | 58.0% [51.1%, 64.6%] | 1.00 [0.87, 1.13] | 0.044 ms (p90 0.064) | $0 (on robot) |
| Gemini controller | - | not run - needs GEMINI_API_KEY | | | | | | | |

## Isaac Lab, train scenarios

| method | n | success | time to clean s | peak force N | p95 N | >40 N | removed %/s | decision latency | cost/ep |
|---|---|---|---|---|---|---|---|---|---|
| Gemini controller | - | not run in Isaac Lab (API latency would need a paused sim; CPU mock only) | | | | | | | |

## CPU kinematic mock, rig scenarios

| method | n | success | time to clean s | peak force N | p95 N | >40 N | removed %/s | decision latency | cost/ep |
|---|---|---|---|---|---|---|---|---|---|
| PPO v2 (ONNX) | 200 | 96.0% [92.3%, 98.0%] | 7.70 [7.32, 8.08] | 34.2 [33.5, 35.0] | 43.06 | 19.5% [14.6%, 25.5%] | 13.66 [12.92, 14.41] | 0.013 ms (p90 0.015) | $0 (on robot) |
| Greedy scripted | 200 | 28.0% [22.2%, 34.6%] | 11.14 [10.48, 11.81] | 35.2 [34.0, 36.4] | 49.31 | 32.5% [26.4%, 39.3%] | 5.13 [4.72, 5.55] | 0.008 ms (p90 0.157) | $0 (on robot) |
| Random | 200 | 0.0% [0.0%, 1.9%] | - | 41.4 [39.9, 42.8] | 57.27 | 55.0% [48.1%, 61.7%] | 0.95 [0.82, 1.07] | 0.026 ms (p90 0.038) | $0 (on robot) |
| Gemini controller | - | not run - needs GEMINI_API_KEY | | | | | | | |

## Isaac Lab, rig scenarios

| method | n | success | time to clean s | peak force N | p95 N | >40 N | removed %/s | decision latency | cost/ep |
|---|---|---|---|---|---|---|---|---|---|
| Gemini controller | - | not run in Isaac Lab (API latency would need a paused sim; CPU mock only) | | | | | | | |

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
