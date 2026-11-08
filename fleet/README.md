# Fleet

Several robots doing many rig jobs at once, coordinated by one dispatcher. Pure Python on the CPU; no
Isaac Sim needed.

```
python -m fleet.demo            # 15-minute shift; Gemini explains priority changes if a key is set
python -m fleet.demo --offline  # never calls Gemini (template lines)
python -m fleet.rigshift --offline  # policy.fleet's 3-deck scenario on this coordinator
pytest tests/test_fleet_coordinator.py tests/test_fleet_unified.py tests/test_fleet_api.py
```

| file | what |
|---|---|
| `contract.py` | shared types: Robot, Job, Level, Finding, JobResult, JobExecutor, World, REGISTRY |
| `coordinator.py` | queue, matching, Hungarian assignment, deconfliction, approvals, follow-ups, safety, timeline |
| `api.py` | agent API for function calling: queue, cancel, approve_request, reprioritise, fleet_state; refusals |
| `rigplanner.py` | 3-deck walkway graph with stairs and narrow corridors (`policy.fleet.rigmap`) for Q and C |
| `state.py` | `fleet_state.v1` snapshots for the HUD (same schema as `policy.fleet`) |
| `risk.py` | risk priority gas > crack > rust (severity x exposure) > flange check > inspection |
| `jobs/flange.py` | flange leak check (AUTO) and re-torque (SUPV) from the leak orchestrator's trace |
| `rigshift.py` | 3 robots, 3 decks: new rust, worker, 2 gas alarms, robot failure, crack |
| `sim.py` | walkway graph, per-kind motion, scripted events, gas sensing, `runs/fleet/timeline.jsonl` |
| `jobs/rust_repair.py` | the Go2 + UR3 repair (`policy.scenario_runner`) as a `rust_repair` job |
| `demo.py` | Q1, Q2, A1, C1 on the full rig: rounds, a rust repair, a mid-shift gas leak |

## Coordinator

`Coordinator(robots, world, executors, planner, auto_approve=False)`; call `tick(1.0)` once per second.

- **Queue**: jobs sorted by priority (1 urgent .. 5 routine), then age. Duplicates (same kind and target
  still open) are skipped.
- **Matching**: robot needs every capability in `job.needs` plus the executor's `needs`, and its kind must
  be in `job.robot_kinds` (or the executor's) if that set is non-empty.
- **Assignment**: Hungarian over idle robots x queued jobs. Cost: travel time on the planner's route +
  120 s per unit of missing battery + 3600 s per priority level (strict priority). Deadlines in
  `params["deadline"]` raise priority by up to 2 levels in the last 600 s. Battery in Wh per robot kind
  (`POWER`); a robot whose battery after travel, work and the walk back to its dock would be under the 15 %
  reserve does not bid.
- **Preemption**: a job with priority <= 2 that no idle robot can take may take a robot doing priority >= 3
  work (lower importance). The bumped job goes back to the queue. An approved SUPV job that is already
  working (an arm mid-paint) is never preempted. Each preemption adds a `priority_change` event with one
  operator-facing line, written by Gemini when an explainer is set (`demo.GeminiExplainer`, max 2 calls,
  template fallback on any error). `used_gemini` is recorded on the event and on the coordinator.
- **Deconfliction**: one active job per asset (`job.target`) and per zone (`job.params["zone"]`); a
  quadruped may not enter a walkway segment another quadruped is on (`segment_wait` / `segment_clear`).
- **Levels**: AUTO runs; SUPV waits in `approvals` until `approve(job_id)` or `reject(job_id)`
  (`auto_approve=True` for unattended runs; never for jobs an agent queued). `approve` sets
  `params["approved"]`, and an agent (`by` gemini/agent/planner) cannot approve a job it requested; ASSIST runs, then is `HANDED_TO_HUMAN`; HUMAN goes straight
  to `crew`. A job kind with no executor goes to the crew; an executor that raises gives `FAILED`.
- **Follow-ups**: every `Finding.follow_up` job is submitted with `parent` set (gas leak -> valve isolate).
- **Safety**: a person within 3 m (`World.people`) pauses the robot (no motion, no work); planners with
  `set_obstacles` route around people and failed robots (`replanned`). A robot that needs its reserve to
  reach the dock goes back and requeues its job; an idle robot under 30 % charges early. Two pads per dock
  (`DOCK_QUEUE`). Routine work within 8 m of a reported, unisolated leak stops (`gas_zone_hold`).
  `fail_robot(id)` requeues its job; a job no robot can reach for 120 s goes to the crew.
- **Work time**: the robot works for `finished_t - started_t` of the executor's JobResult (else
  `est_duration_s`); the result is applied when that time is over.

## Sim and timeline

Quadrupeds and crawlers move on a graph of the rig walkway (`patrol_route_m` in
`sim/output/full_rig_meta.json`, else `PATROL_ROUTE` in `sim/rig_details.py`), split at T-junctions, with
a short straight spur to the asset. Only edges added as `stairs` change deck. Quadrupeds stay at deck
height; crawlers climb the spur onto the asset. Drones climb to 60 m, fly straight and stop 4 m above
the target. ROVs stay at z <= -3 m.

`runs/fleet/timeline.jsonl` has one JSON object per line:

```
{"t": 12.0, "robot": "Q1", "kind": "Q", "position": [x, y, z], "yaw": 1.57, "state": "MOVING", "job": "gauge_read-001"}
{"t": 205.0, "event": "gas_leak_injected", "job": null, "robot": null, "detail": {"position": [21.0, -10.2, 43.4], ...}}
```

Robot lines are written once per robot per second and give the state at time `t`. Robot states: IDLE,
MOVING, WAITING_SEGMENT, WORKING, PAUSED_PERSON, DOCKING, DOCK_QUEUE, CHARGING, FAILED. Events: job_queued,
approval_requested, approved, rejected, assigned (bids, cost), job_started, job_done, finding (position
when known), handed_to_crew, preempted, requeued, priority_change, segment_wait, segment_clear,
person_pause, person_clear, low_battery, docked, charged, gas_leak_injected, gas_detected, gas_alarm,
duplicate_skipped, replanned, gas_zone_hold, robot_failure, dock_queue, cancelled, approval_note,
approval_refused, agent_call, agent_refused.
`FleetSim(..., states_path=...)` also writes one `fleet_state.v1` snapshot per second (`fleet/state.py`).

Gas leaks live in `world.state["gas_leaks"]` as `{"id", "position", "rate_g_s", "t0", "source",
"isolated"}`. A robot with `gas_sensor` within 6 m reports the leak (`gas_detected`); if none does, the
fixed detector alarms after `alarm_after_s` (`gas_alarm`). Either queues an urgent `gas_survey`.

## Demo shift

Default `--planner rig`: Q1 and Q2 dock at the rig dock (2 pads), the shift adds six flange checks (F3 is
loose -> SUPV re-torque) and valve checks on the middle and lower decks. `--planner walkway` keeps the
single-deck graph. Outputs: `runs/fleet/timeline.jsonl` and `runs/fleet/shift/` (v1 states, map, PNG,
summary).

Q1 (arm, spray, camera), Q2 (camera, thermal, gas sensor), A1 drone (camera, thermal, gas sensor, starts
at 45 % battery), C1 crawler (UT probe). Jobs at t=0: 7 gauge reads, 6 valve checks, 3 aerial surveys,
2 crawler wall-thickness checks, 1 rust repair (SUPV; the scripted operator approves each request after
40 s). At t=150 a crew member walks to the rust repair site (pauses Q1, then Q2). At t=205 a leak starts
at a flange near valve_05; Q2 smells it while passing, the urgent survey preempts Q2's gauge read, its
finding spawns a SUPV `valve_isolate` that waits for approval and then preempts Q1, and the isolation's
finding spawns an ASSIST leak-repair prep handed to the crew.

Kinds from any `fleet.jobs.*` module that imports are used; kinds still missing use the stub executors in
`demo.py` (`gauge_read`, `valve_check`, `aerial_survey`, `crawler_ut`, `gas_survey`, `valve_isolate`,
`leak_repair_prep`). Stubs read the rig meta and `world.state` but do no perception.

## Rust repair job

`rust_repair`: quadruped with `arm` and `spray`, level SUPV because the repair ends with paint. The CPU
mock runs detect -> spray -> scrub -> verify -> paint in one call, so the approval is given before the job
starts, not between verify and paint. `run_scenario` measures the real outcome against the simulated
rust: success -> DONE; safe escalation -> HANDED_TO_HUMAN with a HUMAN follow-up; unsafe outcomes
(painted over rust, missed rust, force not contained, timeout) -> FAILED with a critical
`unsafe_repair_outcome` finding and a re-inspection job for the crew, never success. Scenario settings come
from `job.params["scenario"]`.
