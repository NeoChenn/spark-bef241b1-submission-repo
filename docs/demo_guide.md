# Demo guide: two-robot fleet with Argon

A replay of the recorded live run (`runs_sample/fleet_demo`, gemini-4-argon, 88 s). Unit 1 Scrubber does rust, Unit 2 Fixer does gas, and Argon is the command centre.

## Open it

```
uv run uvicorn rig.web.server:app --port 8080
```

If port 9000 is taken by a running sim, add `SIM_PORT=9079` in front. Then open:

**http://localhost:8080/?replay=runs_sample/fleet_demo**

Press **T** (or click **Guide**, top left). Each step has a **Jump** button that seeks the replay and opens the right panels. **K** pauses and plays, and **H** lists every key.

## Steps

| # | Jump | Say | Keys |
|---|---|---|---|
| 1 | 0 s (paused) | Two Go2 robots on the top deck. Unit 1 Scrubber does rust, Unit 2 Fixer does gas. Six fixed detectors G-F1…G-F6. Argon is the command centre. | B fleet, A control, ⇧M map |
| 2 | 2 s | Unit 1 sprays remover, soaks, then scrubs. Brush force stays under 40 N. | V vitals, N mission |
| 3 | 4.5 s | G-F6 reads 14 % LEL, but Unit 2 beside it reads 2 %: PIPE_SENSOR_FAULT. Argon files ticket MT-001 for a human (9.7 s). No repair is sent. | G gas, A, L log |
| 4 | 17.5 s | Bolt 3 at F5 loosens. G-F5 goes amber (18.8 s), then red (20.6 s). The spark rule holds Unit 1 at once, on the edge. | G, A |
| 5 | 23.5 s | Argon sends Unit 2 to check (24.8 s). The "resume rust" in the same reply is refused, because G-F5 is still red. | A, L |
| 6 | 40.5 s | Unit 2 sniffs: LEAK_CONFIRMED (41.1 s). Argon sends fix_leak (49.7 s), and Unit 2 re-torques bolt 3 (53–57 s). | B, A |
| 7 | 57.5 s | G-F5 is green at 60.6 s. Argon clears the zone and Unit 1 resumes (68.7 s). | G, A |
| 8 | 80.5 s | Unit 1 verifies and paints, and Unit 2 docks. 4 Argon calls, 6.4 s mean latency, $0.0123 in total. | V, M pipe |
| 9 | live only | ⇧8 cuts the Gemini link and ⇧A opens the order box. In a replay they only add a banner and a log row; in a live run the edge rules take over and Argon reads the order. | ⇧8, ⇧A |

## Keys

| Key | Does |
|---|---|
| T | Guide |
| K | Pause / play |
| Space | Start |
| H or ? | All controls |
| ⌘K or / | Command palette |
| P | Presenter mode (clean video) |
| X | Close all panels |
| Esc | Close a view |
| A B G L M N V | Control centre, Fleet, Gas, Decisions, Pipe map, Mission, Vitals |
| ⇧M ⇧F ⇧S ⇧O ⇧B | Mission map, First person, Sensors, Orchestrator, Benchmarks |

You can also start the replay at any time with a URL parameter, for example `&t=40`.
