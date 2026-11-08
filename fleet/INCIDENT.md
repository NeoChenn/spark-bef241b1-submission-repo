# Gas-leak incident: fixed detectors, two robot dogs, one human approval

```
python -m fleet.incident                     # --leak-at 15 --approve-after 30 --out runs/fleet
pytest tests/test_fleet_incident.py          # 10 CPU tests, a few seconds
C:\isaacsim\python.bat sim\fleet_replay.py --timeline sim\output\fleet\incident_timeline_render.jsonl ^
    --out sim\output\fleet\incident_replay.mp4 --bitrate 3.5
```

The fleet is two robot dogs, both Go2 + lift + UR3: **Dog-1** (Q1, valve tool on the arm) and **Dog-2** (Q2, gas
sensor). They respond to the **fixed gas detectors** installed around the rig (`fleet/sensors.py`), and a
**human operator** approves the risky action. One coordinator (`fleet/coordinator.py`) runs it in
`fleet/sim.py`'s `FleetSim`. No drone and no crawler: their executors and tests stay in `fleet/jobs/aerial.py`
and `crawler.py`, but the incident, the demo shift and the video do not use them (`--ut` re-adds the crawler
for the old UT ending).

1. **Leak.** At t = 15 s a 3.5 g/s leak starts at a flange near valve_05 (21.0, -10.2) on the top deck.
2. **Detect.** The point gas detectors read the modelled plume once a second. GD-06, 1.8 m from the source,
   crosses its 10 % LEL threshold 9 s after the leak (0.107 LEL) and raises the alarm (`detector_alarm` with its
   id, position and reading, then `gas_alarm`). No timer: the old fixed `alarm_after_s` is switched off for this
   leak.
3. **Localise.** The coordinator queues an urgent `gas_survey` for a dog, starting on the walkway beside GD-06
   (not at the leak, which nobody knows yet). Dog-2 walks there and runs the real `gas_survey` executor
   (`fleet/jobs/gas_valves.py`): 6 m box search, plume climb, least-squares fit with a 95 % region. It proposes
   valve_06, because at valve_05 the robot would stand in a predicted 0.32 LEL.
4. **Approve.** The `valve_isolate` follow-up is SUPV. It sits in the coordinator's approval queue until the
   scripted operator approves it 30 s later (`approved`, `by: operator`). Nothing is assigned or actuated before
   that; a test runs with no approval and checks the valve stays open and the leak stays live.
5. **Isolate.** Dog-1 closes valve_06: gas and person check at the stance, IK reach, torque ramp, quarter turn.
6. **Confirm.** The executor's confirm survey is restricted to a dog. Dog-2 re-samples the eight strongest
   survey points and reports `leak_isolated`; GD-06 clears.
7. **Hand-over.** The confirm's follow-up `leak_repair_prep` (ASSIST) is prepared by Dog-2 (barrier, gas
   monitor) and handed to the crew.

## Fixed detectors (`fleet/sensors.py`)

Point gas detectors on 1.5 m posts (head at z = 44.0 m) on the walkway side of the two top-deck process lines
in `sim/output/full_rig_meta.json`: the north line (valves 01-04, gauges 01-04, y = 23.4..24.3) and the south
line (valves 05-06, gauges 05-07, y = -10.7). Alarm at 0.10 LEL, clear below 0.05 LEL.

| id | x, y (m) | where |
|---|---|---|
| GD-01 | -21.0, 22.6 | north line, west end (gauge_01, valve_01) |
| GD-02 | -9.0, 22.6 | north line (gauge_04, valve_02) |
| GD-03 | -2.5, 22.6 | north line, east end (valve_04) |
| GD-04 | 11.0, -9.9 | south line, west |
| GD-05 | 15.5, -9.9 | south line (gauge_05) |
| GD-06 | 19.2, -9.9 | south line by valve_05 and its flanges |
| GD-07 | 26.0, -9.9 | south line (gauge_06) |
| GD-08 | 33.5, -9.9 | south line, east end (valve_06, gauge_07) |

The three F&G optical detectors that `fg_optics_clean` cleans keep their ids and positions from
`fleet/jobs/rounds.py` (a test checks they match): `fg_ir_01` (-14.0, 22.6) and `fg_ir_03` (30.0, -9.9) are
IR flame detectors and read no gas; `fg_ogd_02` (21.0, -9.9) is open-path gas over a modelled 6 m beam
(x 18..24), alarm at 1 LEL.m. In this run it does not reach 1 LEL.m before the valve is closed.

## Measured (seed 0, `runs/fleet/incident/summary.json`, copy in `sim/output/fleet/incident_summary.json`)

| milestone | sim time after the leak |
|---|---|
| GD-06 alarm (0.107 LEL) | 9 s |
| Dog-2 starts the survey (walk from mid-deck) | 37 s |
| localised by Dog-2 (finding) | 103 s |
| approval requested | 103 s |
| operator approval | 133 s (30 s wait) |
| valve turned (the executor applies the turn when the job starts) | 136 s |
| isolation job done (Dog-1) | 286 s |
| confirmed by Dog-2 (`leak_isolated`) | 321 s |
| repair prep handed to the crew | 382 s |

- Estimate (21.0, -10.2) against truth (21.0, -10.2): error below 0.01 m, 95 % radius 0.46 m, truth inside.
  Peak gas at Dog-2 0.123 LEL (stop limit 0.20). 18 samples.
- Gas at the source 0.42 LEL before, 0.04 after (background 0.02). The confirm survey's plume excess fell from
  0.062 to 0.004 LEL at the same points (6 % of before; "isolated" needs 25 % or less).
- Valve broke away at 6.1 N.m (limit 9.5 N.m).
- Executors: `gas_survey`, `valve_isolate`, `gauge_read` are the registered ones. `leak_repair_prep` has no
  executor module and uses the `fleet/demo.py` stub. Gemini is off (`FLEET_GEMINI=0` by default here); the leak
  report and approval text are the template.

## Video (`sim/output/fleet/incident_*`)

`incident_replay.mp4` (1280x720, 24 fps) and `incident_still_*.png`. It is rendered from
`runs/fleet/incident/timeline_render.jsonl` (copy: `incident_timeline_render.jsonl`), the FleetSim log
(`incident_timeline.jsonl`) with these changes, all made in `fleet/incident.py` (`render_timeline`):

- **Time-compressed.** 1.5x within 3 s of a key event, 6x while a robot moves or samples, 24x otherwise. About
  400 s of incident become 65 s. The HUD clock is render time; each event line carries the real time as `T+…s`
  and in `detail.true_t`.
- **Filtered events.** Only the incident's events (leak, detector alarm/clear, gas alarm, assignments, starts,
  approval, findings, done, crew hand-off). Job ids are shown as `kind target`.
- **Survey path.** In the coordinator log Dog-2 stands still while the survey runs, because the executor
  computes the whole survey when the job starts and reports only its result. For the video Dog-2 is walked
  through the survey's eight strongest logged sample points, held on the walkway side of the line
  (y >= -9.0 m). This hides a real gap: the executor's seeker moves freely in 2-D, and its eight strongest
  samples lie at y = -11.4..-12.7 m, beyond the pipe line (y = -10.7), where a dog cannot stand. The model
  treats the dog's sensor as reaching them. After the survey the sim puts Dog-2 back on the walkway line
  (`WalkwaySide` in `fleet/incident.py`); the confirm survey re-reads the same points the same way.
- **`gas_isolated` event** at the end of the isolation job, so the plume fades when Dog-1 finishes. In the
  simulation the feed is cut when the job starts; GD-06's `detector_clear` (logged at T+137 s) is shown at the
  same moment as the plume fade for the same reason.

On screen: fixed detectors are grey posts with yellow heads (`FixedDetector` in `sim/fleet_models.py`, built from
`fleet/sensors.py`); a detector in alarm flashes a red-amber beacon and its line shows in the HUD
(`GD-06 alarm 0.11 LEL`), and the camera cuts to it. Q1 and Q2 are labelled Dog-1 and Dog-2 in #2EE6D6 and
#FFB020. Robots not in the timeline (the drone, the crawler) are not built.

## What this is and is not

- **Kinematic replay.** Robots move on the walkway graph at fixed speeds; the render poses them from the log.
  No PhysX, no locomotion policy, no collision. The arm animation on the valve is the replay's generic
  "working" pose, not the IK solution the executor computed.
- **Modelled plume and detector readings.** The gas is the 2-D analytic plume in `fleet/gas/plume.py`
  (exponential decay, no wind here). Detector readings are that field with a transport delay (0.5 m/s from the
  source) and a 4 s first-order rise; no noise, no CFD, no obstacles, no detector drift or fault. The robots'
  gas sensor samples the same field with modelled noise. The rendered plume is a visual cue that fades on
  `gas_isolated`; it is not the model's field.
- **Scripted parts.** The leak, the detector positions and thresholds, the operator's approval after 30 s,
  the robots' start positions and the dog-only confirm are set by the scenario. Everything between (alarm
  time and detector, assignment, routing, localisation, valve choice, gating, torque, confirmation) comes from
  the detector model, the coordinator and the executors.
