# Multi-robot gas-leak incident

```
python -m fleet.incident                     # --leak-at 15 --approve-after 30 --out runs/fleet
pytest tests/test_fleet_incident.py          # 7 CPU tests, about 3 s
C:\isaacsim\python.bat sim\fleet_replay.py --timeline sim\output\fleet\incident_timeline_render.jsonl ^
    --out sim\output\fleet\incident_replay.mp4 --bitrate 3.5
```

One incident, four robots, one coordinator (`fleet/coordinator.py` in `fleet/sim.py`'s `FleetSim`):

1. **Leak.** At t = 15 s a 3.5 g/s leak starts at a flange near valve_05 (21.0, -10.2) on the top deck.
2. **Detect.** The fixed gas detector alarms 6 s later. The coordinator queues an urgent `gas_survey` that
   only the drone may take.
3. **Localise.** The Drone (A1) flies from the helideck and runs the real `gas_survey` executor
   (`fleet/jobs/gas_valves.py`): box search, plume climb, least-squares fit with a 95 % region. It proposes
   valve_06, because valve_05 is predicted to put the robot in 0.33 LEL.
4. **Approve.** The `valve_isolate` follow-up is SUPV. It sits in the coordinator's approval queue until the
   scripted operator approves it 30 s later (`approved`, `by: operator`). Nothing is assigned or actuated before
   that. A test runs with no approval and checks the valve stays open and the leak stays live.
5. **Isolate.** Dog-1 (Q1, arm) closes valve_06: gas and person check at the stance, IK reach, torque ramp,
   quarter turn.
6. **Confirm.** The executor's follow-up confirm survey is restricted to a quadruped (`--confirm-by
   quadruped`, the default) so the check does not come from the drone that made the estimate. Dog-2 (Q2)
   re-samples the eight strongest survey points and reports `leak_isolated`. Its follow-up `leak_repair_prep`
   (ASSIST) goes to the crew.
7. **UT.** The Crawler (C1) runs the real `ut_thickness` executor (`fleet/jobs/crawler.py`) on a 0.6 m patch
   of the leaking spool (25 points).

Dog-2, its prep job and the crawler stand on the walkway line y = -9 m beside the flange skid (`STANCE_Y`):
the source estimate itself is inside the skid model. The executors still sample at the survey points and
the estimate held in the job params.

## Measured (seed 0, `runs/fleet/incident/summary.json`, copy in `sim/output/fleet/incident_summary.json`)

| milestone | sim time after the leak |
|---|---|
| detected (fixed detector) | 6 s |
| localised by the Drone (finding) | 77 s |
| approval requested | 77 s |
| operator approval | 107 s (30 s wait) |
| valve turned (executor applies the turn when the job starts) | 110 s |
| isolation job done | 260 s |
| confirmed by Dog-2 (`leak_isolated`) | 312 s |
| crawler UT done | 501 s |

- Estimate (20.96, -10.09) against truth (21.0, -10.2): error 0.12 m, 95 % radius 0.24 m, truth inside.
  Peak gas at the drone 0.126 LEL (stop limit 0.20). 22 samples.
- Gas at the source 0.42 LEL before, 0.04 after (background 0.02). The confirm survey's plume excess fell from
  0.070 to 0.004 LEL at the same points (6 % of before; the threshold for "isolated" is 25 %).
- Valve broke away at 6.1 N.m (limit 9.5 N.m). UT minimum 11.97 mm, allowed 9.70 mm: no finding.
- Executors: `gas_survey`, `valve_isolate`, `ut_thickness`, `gauge_read` are the registered ones.
  `leak_repair_prep` has no executor module and uses the `fleet/demo.py` stub. Gemini is off (`FLEET_GEMINI=0`
  by default here); the leak report and approval text are the template.

## Video (`sim/output/fleet/incident_*`)

`runs/fleet/timeline.jsonl` is the real-time FleetSim log (1 line per robot per second). The video is rendered
from `runs/fleet/incident/timeline_render.jsonl`, the same schema with these changes, all made in
`fleet/incident.py` (`render_timeline`):

- **Time-compressed.** 1.5x within 3 s of a key event, 6x while any robot moves or samples, 24x otherwise
  (the 150 s valve job and the 150 s UT scan pass in seconds). About 500 s of incident become 76 s. The HUD
  clock is render time; each event line carries the real time as `T+…s` and in `detail.true_t`.
- **Filtered events.** Only the incident's events (leak, alarm, assignments, starts, approval, findings, done,
  crew hand-off). Job ids are shown as `kind target`.
- **Drone survey path.** In the coordinator log the drone hovers while the survey runs, because the executor
  computes the whole survey when the job starts and reports only its result. For the video the drone is flown
  through the survey's eight strongest logged sample points and ends at the executor's final position. The
  points are real samples; the order and the straight lines between them are drawn.
- **`gas_isolated` event** at the end of the isolation job, so the plume fades when Dog-1 finishes. In the
  simulation the feed is cut when the job starts.

On screen Q1, Q2, A1, C1 are labelled Dog-1, Dog-2, Drone, Crawler in #2EE6D6, #FFB020, #F3F5F8, #A78BFA
(HUD rows, tags, ID plates; `ROBOT_NAMES` / `ROBOT_COLORS` in `sim/fleet_replay.py`). The data keeps the ids.

## What this is and is not

- **Kinematic replay.** Robots move on the walkway graph at fixed speeds; the render poses them from the log.
  No PhysX, no locomotion policy, no collision. The arm animation on the valve is the replay's generic
  "working" pose, not the IK solution the executor computed.
- **CPU plume model.** The gas is the 2-D analytic plume in `fleet/gas/plume.py` (exponential decay, log-normal
  noise, here no wind) at sensor height. The drone samples it as if it were at deck height. No CFD, no
  obstacles. The rendered plume is a visual cue that fades on `gas_isolated`; it is not the model's field.
- **Scripted parts.** The leak, the fixed-detector alarm delay, the operator's approval after 30 s, the
  robots' start positions, the quadruped-only confirm, and the UT target and patch are set by the scenario.
  Everything between (assignment, routing, localisation, valve choice, gating, torque, confirmation, UT
  readings) comes from the coordinator and the executors.
