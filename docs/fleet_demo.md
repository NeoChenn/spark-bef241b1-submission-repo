# Fleet demo: two robots and one command centre

This is the source material for the 1-page PDF and the 5-minute pitch. It covers the swarm version of the demo, in which two robots are coordinated by Argon. `docs/pitch.md` and `docs/technical_summary.html` still describe the earlier single-robot story. Where a number below also appears there, the source is the same.

Every number cites the file it comes from. Paths are on `origin/fleet-demo` unless a branch or commit is named. Lines marked **TODO** are waiting for a recorded run. Fill them in from the named file, or drop the line. Never fill them in from memory.

## The swarm in one line

> When a fixed gas detector alarms, Argon pauses the spark-producing rust work, sends the gas robot to confirm and fix the leak, and resumes the rust job when the zone is clear. If the link to Argon is cut, the robots' edge rules make the same calls.

## The fleet

| Unit | Body | Tools | Job | Decides on the edge | Calls the cloud |
|---|---|---|---|---|---|
| Unit 1 "Scrubber" (`unit1`) | Go2 + 0–0.45 m lift mast + UR3 | squishing brush pad, spray/paint nozzle, personal gas sniffer | Rust: detect → spray → soak → scrub (PPO) → verify → paint | rust detector, scrub policy, state machine, force/person/gas gates | Gemini rust second opinion (optional, asynchronous) |
| Unit 2 "Fixer" (`unit2`) | Go2 + UR3 | 2-finger gripper / torque tool; IR and catalytic %LEL, H₂S cell, ultrasound, OGI-style camera | Gas: cast-and-surge localisation → inspect flange → re-torque bolt → verify | sensor suite, localisation, stripe and crack check | none |
| Fixed detectors (`G-F1` … `G-F6`) | yellow F&G point boxes beside each flange | gas sensor + status LED: green, amber, red, grey (fault) | continuous monitoring, raise the alarm | LED thresholds | none |
| Argon | command centre, `gemini-4-argon` | fusion, capability routing, spark rule, zone clearance, maintenance tickets | coordinator | deterministic fusion and edge rules run without it | one structured-JSON call per new verdict |

Sources: unit ids, routing, LED and the Argon prompt are in `integration/dual_argon.py` (`SCRUBBER`, `FIXER`, `ROUTE`, `led`, `SYSTEM`). The Unit 2 sensors are in `policy/gasbot/sensors.py` (`PointGasSensor`, `H2SCell`, `AcousticSensor`, `ogi_view`, `SensorSuite`). The robot body is in `sim/assets/go2_ur3/go2_ur3_lift.urdf`. The model id is `MODEL` in `policy/control/backend.py`.

Routing is by capability (`ROUTE` in `integration/dual_argon.py`). `go_check` and `fix_leak` go to `unit2`. `hold`, `resume_rust` and `retreat` go to `unit1`, except that an edge-rule `retreat` goes to whichever robot is at the gas. `log_sensor_fault` goes to the supervisor inbox (`supv`).

### Where each model runs

| Component | Runs on | Latency | Source |
|---|---|---|---|
| Rust classifier, MobileNetV4-S + LoRA (2.49 M params, 0.19 GMACs) | robot CPU, ONNX | 1.1 ms (4 threads), 2.6 ms (1 thread), batch 1 | `detection/RESULTS.md` |
| Scrub policy (MLP, 327 → 4) | robot CPU, ONNX | 0.012 ms, 10 Hz loop | `docs/technical_summary.html` (Apple M5 Max CPU, batch 1) |
| Leak check (8 bolt stripes + crack check, numpy) | Unit 2 CPU | ~90 ms per frame, 3 frames per view | `docs/technical_summary.html` (Apple M5 Max CPU) |
| Gas sensor models (IR t90 6 s, catalytic t90 9 s, H₂S t90 25 s) | Unit 2 | sensor response, not compute | `policy/gasbot/sensors.py` |
| Fusion + edge rules + spark rule | edge, pure Python | not measured (a few comparisons per detector) | `integration/dual_argon.py` (`fuse`, `edge_rule`, `spark_hot`) |
| Argon, `gemini-4-argon`, thinking `low` | cloud API, asynchronous, never in the fast loop | **6.0 s** for one call (879 tokens in, 114 out). **TODO:** cite the timeline file once `argon-prompt` pushes it | coordinator live check on `fleet-demo` @ 44b54de; per-call `latency_ms` is logged as `argon_call` rows by `Timeline` in `integration/dual_argon.py` |
| Argon timeout before the edge rule takes over | edge | 25 s | `edge_timeout_s` in `integration/dual_argon.py` |
| Earlier live Argon shift (the coordinator, not this demo) | cloud | mean 10.3 s, 15 of 19 calls ok, $0.1487 over a 720 s shift | `3b58313:runs_sample/argon_live/summary.json` (branch `argon-brain`) |
| Isaac Sim 6.2 render + stream | NVIDIA GB10 | RTX, JPEG over WebSocket | `docs/technical_summary.html` |
| Edge TensorRT on the GB10 | GB10 GPU | **TBD** (not measured) | `docs/pitch.md` |

Cost per Argon call: at the intro price of $2 per million input tokens and $10 per million output tokens (`policy/control/limiter.py`, valid until 2026-12-01), 879 in and 114 out works out to about $0.0029. This is derived, not billed.

## The scenario, beat by beat (the video)

One take, ≤ 60 s, HUD burned in. The timestamps come from the recorded run.

**TODO:** take every `t` from lane `cpu-runner`'s `runs_sample/fleet_demo/` timeline (`finding`, `argon_request`, `argon_call`, `command`, `ticket` and `vetoed` rows) and from `argon-prompt`'s recorded calls. Video time = timeline `t` + the render offset, so note the offset here when it is known.

The demo leak is at F5 / G-F5, about 5.5 m from Unit 1, and the stuck head is G-F6. Both come from the coordinator's scenario. `integration/fleet_scenario.py` on `fleet/leak-scenario` (7c95d4d and later) defaults to F5 / F6, matching the recorded timeline.

| # | Video time | What happens | Timeline row to cite | HUD shows |
|---|---|---|---|---|
| 1 | TODO 0:00 | Unit 1 scrubbing rust on the main pipe. Unit 2 docked. | first `command` / state rows | rust map, force, G-F5 green |
| 2 | TODO | A bolt works loose at F5. G-F5 goes amber, then red. | `finding` G-F5 (`fixed_lel` ≥ 5 → amber, ≥ 10 → red) | LED colour, %LEL |
| 3a | TODO | Spark rule: G-F5 is amber/red and ~5.5 m from Unit 1, inside the 6 m spark radius. Unit 1 HOLDs and steps back. The rust job is PAUSED, not cancelled. | `command` hold, `source: edge`, reason `spark rule: …` | HOLD, "rust paused" |
| 3b | TODO | Unit 1's sniffer reads low at ~5.5 m (outside the 2 m confirm range), so the verdict is UNVERIFIED. Argon is asked. | `finding` UNVERIFIED, then `argon_request` key `G-F5:UNVERIFIED` | verdict, "Argon pending" |
| 3c | TODO (request + TODO s) | Argon replies: `go_check` F5 → unit2. | `argon_call` (`latency_ms`, tokens, cost), `command` go_check | Argon line, latency |
| 4a | TODO | Unit 2 walks in and sniffs (cast-and-surge). Both sources agree, so the verdict is LEAK_CONFIRMED. | `finding` LEAK_CONFIRMED | Unit 2 %LEL, distance |
| 4b | TODO | `fix_leak` F5: the gripper re-torques the loose bolt. Gas falls and G-F5 turns green. | `command` fix_leak; `finding` OK | stripe aligned, gas falling |
| 5 | TODO | Zone clear: Argon is asked once and `resume_rust` goes to unit1. Unit 1 finishes the scrub and paints. Unit 2 returns to dock. | `argon_request` key `zone:clear`, `command` resume_rust | RESUMED, DONE |
| 6 (opt.) | TODO | G-F6 is stuck high while Unit 2 passes reading clean. The verdict is PIPE_SENSOR_FAULT and ticket MT-001 is raised. No repair is dispatched. | `finding` PIPE_SENSOR_FAULT, `ticket` MT-001 | grey LED, ticket |
| 7 (opt.) | TODO | Cut link: the same call is made by the edge rule, with `source: edge`. | `link` cut row, `command` with `source: edge` | LINK CUT |

## Fusion logic: fixed detector vs robot sniffer

From `fuse()` in `integration/dual_argon.py`. The readings are in %LEL. "Near" means a robot within `NEAR_M` = 2 m of the detector, and the reading used is the nearest robot's (`fuse_fleet`). Thresholds: CLEAR 5, ALARM 10, STOP 20 %LEL.

| Fixed detector | Robot sniffer | Verdict | Edge-rule commands (`edge_rule`) |
|---|---|---|---|
| ≥ 10 | none within 2 m | **UNVERIFIED** | hold unit1 + go_check unit2 |
| ≥ 10 | ≥ 5, within 2 m | **LEAK_CONFIRMED** | fix_leak unit2 |
| 10 to < 20 | < 5, within 2 m | **PIPE_SENSOR_FAULT** | log_sensor_fault → maintenance ticket (no repair) |
| ≥ 20 | any | never only a sensor fault: `hazard = true` | hazard actions at once on the edge, even with the link up (fix pending, see below) |
| < 10 (clean or dead) | ≥ 10, within 2 m | **LEAK_UNSEEN_BY_FIXED** | fix_leak unit2 + log_sensor_fault on the head |
| < 10 | below alarm, or far | **OK** | resume_rust (only if unit1 is held and the spark rule is clear) |
| any | robot ≥ 20, or a crack | `hazard = true` | hold unit1 + retreat for the robot at the gas |

The LED is green below 5, amber from 5, red from 10, and grey for a faulted head. A faulted head still shows red at ≥ 20 %LEL (`led(lel, fault)`).

## Safety envelope

- **Hazard first.** In `fuse()`, a reading of ≥ 20 %LEL from either source sets `hazard`. `led()` won't grey out a faulted head at that level. The Argon prompt states that "anything ≥ 20 %LEL is real gas regardless of a fault verdict" (`SYSTEM`). A robot reading ≥ 20 %LEL, or a crack, makes the edge rule hold and retreat (`edge_rule`, `integration/dual_argon.py`). The hazard-first fix in `dual_argon` goes further: a fixed reading ≥ 20 %LEL is never treated as just a sensor fault, and hazard actions are issued at once by the edge rule even while the link to Argon is up. **TODO:** this fix isn't on `fleet-demo` at b851bc7, where `fuse()` can still return PIPE_SENSOR_FAULT for a fixed reading ≥ 20 with a clean robot beside it. Cite the commit once it lands. The older single-robot gates still apply: GAS_RETREAT at ≥ 0.20 LEL (`docs/technical_summary.html`), and the evacuation latch at 50 %LEL that holds until gas falls below 10 % (`LEL_EVACUATE`, `LEL_CLEAR` in `policy/control/safety.py`).
- **Spark rule.** Wire brushing is spark-producing hot work. Any detector within `SPARK_RADIUS_M` = 6 m of unit1 that is amber or red holds unit1 (a faulted head counts only at ≥ 20 %LEL) at once on the edge, without waiting for Argon. A `resume_rust` that arrives while a nearby detector is still amber or red is logged as `vetoed` and dropped (`spark_hot`, `ArgonLink.tick`, `ArgonLink._issue`).
- **No self-approval.** Gemini can't approve its own supervised job. `approve_request` only files to the human operator inbox ("only a human approves"). Paint, rust repair, re-torque and valve isolation are raised from AUTO to SUPV. Safety jobs can't be cancelled or pushed down, an ESD can only be reset on site, and the 40 N force limit and 3 m person pause can't be changed by the agent (`policy/control/envelope.py`).
- **Cut-link edge rule.** With the link cut (`ArgonLink.cut(True)` or mode `edge`), each verdict is turned into commands by the same deterministic `edge_rule` and issued at once. An Argon error, a timeout over 25 s or an invalid reply (bad JSON, a command outside the enum, an unknown target) falls back to the same rule, and the reason is tagged on the command (`ArgonLink._finish`). Argon's reply is parsed against a fixed command enum and target set, so it can't name a robot outside `ROUTE` except for a retreat (`parse_reply`).
- **Edge gates outside any model:** over 40 N → back off; a person within 3 m → pause; the scrub setpoint is clamped to 38 N (`policy/control/safety.py`, `policy/SCRUB.md`).

## Rubric map

The weights come from `HANDOFF.md`. The video moments refer to the beats above.

| Criterion (weight) | Video moment | Metric to quote | Source |
|---|---|---|---|
| Perception-driven action (25 %) | Beats 2–4: the detector LED and both %LEL readings decide the verdict, and the verdict decides the command | Verdict → command per row of the fusion table; rust detector F1 0.977 on 881 real photos, 13/13 held-out Isaac renders; crack check 40/40 found, 0/40 false alarms | `integration/dual_argon.py`, `detection/RESULTS.md`, `docs/technical_summary.html`; beat times **TODO** from `runs_sample/fleet_demo/` |
| Adaptation to a mid-run change (25 %) | Beat 2→3a: a bolt loosens mid-run (scripted in the sim at a fixed time; the robots and Argon are not told in advance). Unit 1 holds and pauses, Unit 2 is dispatched, and Unit 1 resumes the same job | Time from G-F5 red to hold (expected at the same tick, since the spark rule is on the edge); time to go_check; time to green; time to resume. **TODO** from the timeline | `runs_sample/fleet_demo/` (TODO) |
| Edge autonomy (15 %) | Beat 7, cut link; also the spark-rule hold in 3a, which never waits for the cloud | Same command with `source: edge`; 25 s timeout; perception on CPU at 1.1 ms (rust) and ~90 ms (leak check) | `integration/dual_argon.py`, `detection/RESULTS.md`, `docs/technical_summary.html`; cut-link run **TODO** |
| Relevance + the critical failure mode on screen (20 %) | Beat 3a, spark near gas: hot work is held. Beat 6: a faulty head raises a ticket instead of a needless repair. A crack is never re-torqued | Spark radius 6 m; STOP 20 %LEL; ticket MT-001; crack p ≥ 0.5 → never a robot repair | `integration/dual_argon.py`, `docs/technical_summary.html` |
| Ambition and quality (15 %) | The whole take: two robots, a fixed detector network, Gemini command centre, one Isaac scene | Live Argon latency 6.0 s and about $0.003 a call; scrub v2 over 40 N in 1.0 % of episodes (v1: 76.8 %) at 97.6 % success | `fleet-demo` @ 44b54de live check (TODO timeline file), `policy/control/limiter.py`, `policy/SCRUB.md` |

## Best use of Gemini

- **Argon reasons over two disagreeing sensors.** It gets a structured snapshot with the fixed readings, the nearest sniffer and its distance, both robots' states and the deterministic verdict. It returns schema-constrained JSON: a verdict, a 90-character operator line, and a list of commands with a robot, target, confidence and a one-sentence rationale that cites both readings (`SCHEMA`, `build_prompt`, `integration/dual_argon.py`). The prompt tells it to prefer the conservative command when the evidence is mixed.
- **It's advice, and it's bounded.** Every reply is checked against the command enum, the known targets and the confidence range. Anything else falls back to the edge rule (`parse_reply`). The safety envelope sits outside the model: no self-approval, fixed force and person limits (`policy/control/envelope.py`).
- **It isn't in the fast loop.** Calls run in a background thread. A reply lands at the request time plus the real latency, and the robots keep running in the meantime (`ArgonLink._collect`). One call per new verdict per sensor, plus one `zone:clear` call, keeps the cost to a few calls a run.
- **It's recorded and replayable.** Every call is logged with its latency, tokens and cost, and `ReplayCaller` replays the exact recorded replies at their recorded latency. The video can run offline and stay faithful to a live run (`ReplayCaller`, `Timeline`).
- **Gemini elsewhere in the stack:** `gemini-robotics-er-2-preview` gives the rust second opinion and pre-paint veto (`policy/gemini_verifier.py`). It grades aerial rust at 16/18, with 2.3–6.9 s per image (`fleet/jobs/AERIAL.md`). It reads gauges at 1.71 % MAE on a 20-crop sample, with a mean of 8.7 s (`fleet/jobs/ROUNDS.md`). `gemini-3.1-pro-preview` wrote the 120-scenario test analysis that found the faint-rust-under-glare failure (`docs/gemini_test_report/report.md`).

## What's real vs stand-in

| Real (runs as code and is tested) | Stand-in |
|---|---|
| Fusion, edge rules, spark rule, zone clearance, tickets, link-cut fallback (`integration/dual_argon.py`, `tests/test_dual_argon.py`) | The robots are posed kinematically in Isaac, not walked under PhysX (`docs/technical_summary.html`); Go2 walking is a waypoint follower in flight (lane `go2-walk`) |
| Live `gemini-4-argon` calls with measured latency and tokens; the replay reproduces them | The gas field and thresholds are illustrative, not a site standard (`docs/pitch.md`); the fixed detector reads the sim gas model, and a stuck head is injected with `stuck_at` |
| Rust detector, scrub PPO v2, stripe and crack check, with numbers above | Bolt torque and thread friction aren't simulated; the brush force is a spring model (`docs/technical_summary.html`) |
| Unit 2 sensor models with lag, noise, drift, cross-sensitivity and catalytic poisoning (`policy/gasbot/sensors.py`) | Unit 2 localisation numbers on the demo deck: **TODO**, none are recorded on `fleet-demo` yet (`python -m policy.gasbot.evaluate` produces them) |
| Envelope: no self-approval, fixed limits (`policy/control/envelope.py`) | The operator inbox and maintenance ticket are log rows, not a real CMMS |
| | The video timestamps and the cut-link run are **TODO** until `runs_sample/fleet_demo/` lands |
