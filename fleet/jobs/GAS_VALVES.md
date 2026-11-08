# Gas survey, valve isolation, filter changeover and panel buttons

`fleet/jobs/gas_valves.py` registers four job executors against the shared contract (`fleet/contract.py`).
Helpers are in `fleet/gas/`. Tests: `tests/test_jobs_gas_valves.py` (40 tests, about 17 s on CPU, no Isaac).

| Job | Robot | Level | What it does |
|---|---|---|---|
| `gas_survey` | Q or A + `gas_sensor` | AUTO | Patrol round, detect, climb the plume, fit the source with a 95% radius. Produces a critical `gas_leak` Finding and a follow-up `valve_isolate` for the nearest upstream valve |
| `gas_survey` (`mode="confirm"`) | Q + `gas_sensor` | AUTO | Re-samples the strongest plume points after isolation. Gas down to 25% or less of before gives `leak_isolated`. Otherwise `isolation_ineffective` plus a `valve_isolate` for the next upstream valve |
| `valve_isolate` | Q + `arm` | SUPV | Approval, then gas and person check, IK reach check, torque ramp against the limit, quarter turn. Then a follow-up confirming survey |
| `filter_changeover` | Q + `arm` | SUPV | High dP, then approval, reach check, duplex lever moved to the standby element, verify dP fell. Follow-up HUMAN job to replace the fouled element |
| `panel_button` | Q + `arm` | SUPV | `pump_start` or `alarm_ack` only. It refuses an ESD reset and never presses it, along with any reset, bypass, override, inhibit, trip or shutdown button |

## Contract for the coordinator (PR #7, `fleet-coordinator`)

- **Approval.** The coordinator holds a SUPV job in `approvals` and calls `run` only after `approve(job_id)`. The
  executor is a second gate: if `job.params["approved"]` (or `params["approval"]["approved"]`) is explicitly
  `False`, the job actuates nothing. It returns `WAITING_APPROVAL` with an `approval_required` Finding, and that
  Finding's evidence holds the request (`what / where / why / risk_if_delayed / text`).
  `executor.approval_request(job, world)` gives the same request before a run. Follow-up `valve_isolate` jobs carry
  `params["approval_request"]`, which Gemini writes. The executor does not require `approved=True`, because the
  coordinator's `approve()` does not set it. If it starts setting it, the gate can become strict.
- **Leak-triggered surveys.** The coordinator queues `gas_survey(target=<leak id>, position=<alarm position>,
  params={"zone": id})`. A survey whose target is not a deck name runs a 6 m box search around that position
  instead of the full round. A deck-wide round uses target `top_deck` (or `params["round"]=True`). The leak id
  passes along as `params["leak"]` and `params["zone"]` through `valve_isolate`, then the confirm survey, then
  `leak_repair_prep` (ASSIST, camera), the same chain as the coordinator's stubs. The id is the job's label from the
  coordinator, and the survey does not read the leak's position from it.
- **Refusals** return `HANDED_TO_HUMAN`, a Finding explaining why, and a `Level.HUMAN` follow-up job (for example
  `valve_isolate_manual` or `esd_reset`). A person within 3 m returns `FAILED` ("paused, retry").
- **Valve choice skips valves in gas.** If the fitted plume, with 1.5 times the amplitude as margin, predicts
  0.20 of LEL or more at a valve's stance, that valve is skipped and the next one is proposed. In the coordinator's
  demo leak, valve_05 is 1.1 m from the flange, so the robot proposes valve_06.
- **Integration check.** The coordinator demo (`fleet/demo.py --offline`, branch `fleet-coordinator`) was run with
  these executors in place of its stubs. The leak at t=205 is detected by Q2 at t=217. The survey (84 s) gives a
  finding and `valve_isolate valve_06`, which waits for approval, is approved at t=347 and is done by Q1 at t=512.
  The confirm survey (A1 drone) ends with `leak_isolated`, and `leak_repair_prep` is handed to the crew at t=619.
- **Simulated truth** lives in `World.state`:
  - `gas_leaks = [{"id", "position", "rate_g_s", "t0", "source", "isolated"}]`: the coordinator's format. Optional
    fields used only here are `amplitude_lel`, `decay_m`, `flow_dir`, `isolating_valves` and `residual`. The plume
    amplitude is `rate_g_s × 0.114`, so 3.5 g/s gives the gas branch's full strength of 0.4 of LEL. Isolation sets
    `isolated=True`, `isolated_by`, and `rate_g_s` times `residual` (0.05).
  - `process_flow_dir = [fx, fy]` (or per-line `flow_dir`) defines upstream.
  - `wind = [vx, vy]`
  - `valves[id] = {"breakaway_nm", "lever_m", "handle_offset", "closed"}`
  - `filters[id] = {"position_m", "dp_bar", "dp_limit_bar", "active", "standby_dp_bar", "standby_fouled"}`
  - `panel = {"esd_tripped", "alarms", "pressed"}`
  - `seed` fixes the sensor noise.
- `filter_jobs(world)` creates `filter_changeover` jobs for every filter with dP at or above its limit.

## Gas survey: how the leak is found

1. **Round.** Sample every 2 m along `patrol_route_m` from `full_rig_meta.json`. Each sample averages 4 readings
   (2 s). Detection is an excess of more than 0.015 of LEL over background, about 7 sigma of the averaged noise.
   The robot leaves the route at the approach limit, 0.10 of LEL. If it ever reads the stop limit, 0.20 of LEL,
   it backs off at once.
2. **Climb.** Start from the strongest sample that is below the approach limit. Four probes 0.5 m apart give the
   gradient, and the robot steps along it with an upwind bias. The step is predictive: the plume grows roughly
   exponentially toward the source, so the step is shortened until the predicted reading stays under the approach
   limit. The robot never walks into the source.
3. **Fit.** Grid search over source xy (0.1 m cells, ±8 m), with the amplitude in closed form (capped at 1.0 of
   LEL as a prior). Weighted least squares uses the sensor's noise model. The 95% region is delta-chi2 ≤ 5.99
   (2 dof), inflated when the reduced chi2 is above 1. The **uncertainty radius** is the farthest point of that
   region from the estimate.
4. **Infotaxis-lite.** While the radius is above 0.5 m (at most 12 samples), sample next where the plausible
   sources disagree most about the reading (predicted variance over noise, small travel penalty). Only points that
   keep the reading under the approach limit are allowed, assuming 1.5 times the fitted amplitude.
5. **Valve choice.** The nearest valve upstream along the process flow (`flow_dir`), preferring the same deck.
   Without a flow direction it is the nearest valve. If the confirming survey shows the gas did not fall, the next
   upstream valve is proposed, again under SUPV approval. This is the adaptation path.

### Measured localization (seeded scenarios from the tests, 20 seeds each, top deck)

Run: `python -m fleet.gas.measure 20` (the scenarios the tests use). Errors are horizontal; the estimate
is xy only.

| Scenario | Source | Wind (m/s) | Mean error | Max error | First / final 95% radius | Truth inside radius | Samples | Sim time |
|---|---|---|---|---|---|---|---|---|
| north_rack_east_wind | (-12.0, 23.0) | (2.0, 0.5) | 0.22 m | 1.09 m | 1.38 / 0.51 m | 20/20 | 77 | 302 s |
| south_rack_ne_wind | (22.0, -8.0) | (-1.0, 1.0) | 0.08 m | 0.13 m | 0.34 / 0.27 m | 20/20 | 47 | 177 s |
| west_walkway_north_wind | (-19.5, 15.0) | (0.0, 1.5) | 0.07 m | 0.22 m | 0.35 / 0.35 m | 20/20 | 87 | 328 s |
| south_junction_calm | (9.0, -6.5) | calm | 0.43 m | 1.43 m | 1.93 / 0.85 m | 20/20 | 44 | 158 s |
| **all 80 runs** | | | **0.20 m** (median 0.10, p90 0.35) | **1.43 m** | | **80/80** | | |

The bound stated in the tests: on every seed the error is under 1.5 m and inside the reported 95% radius, and the
mean error per scenario is under 0.5 m. The calm case is the hardest. Without wind the plume is isotropic, so
distance and leak strength trade off along the line of approach. The amplitude prior and the information samples
resolve this, and the radius stays honestly wide until they do. Peak gas at the robot was 0.22 of LEL, once, on
the fixed patrol route before backing off. The climb and the information samples stayed under 0.20 in every test
run.

### Arm checks (`fleet/gas/reach.py`, `integration/kinematics.py`)

- **Reach.** Stances 0.35 or 0.5 m in front of the device (the facing comes from the nearest gauge on the same
  rack, or toward the patrol route), centred or 0.3 m to either side, at 8 headings. Base pose is
  `pose(x, y, deck + 0.32, yaw)`. The approach is horizontal into the handle. Reached means IK error under 1 cm
  and under 0.1 rad. Rig valves (1.1 m above the deck) are reached, and the test gets an IK error under 1 cm. A
  handle 2.2 m above the deck gets no stance in 48 tried, so the job refuses and hands over. A probe of 288
  stances found reachable stances at 0.6, 1.1 and 1.5 m, and none at 2.0 m. One IK solve takes about 80 ms.
- **Torque.** The limit is 38 N (the scrub-v2 force cap) times the lever length (default 0.25 m), which gives
  9.5 N·m. The arm ramps in 0.25 N·m steps. If the valve has not broken away at the limit, the job stops, does
  not force it, and hands over (test: a seized valve at 20 N·m is refused).
- **Quarter turn** is `ur3_wrist_3_joint` ±π/2 from the IK solution, kept inside the joint limits.
- **Gas at the stance** goes through the ported safety supervisor. At or above 0.20 of LEL the job refuses and
  hands to a person with breathing apparatus.

## Gemini

`fleet/gas/report.py` sends one `generate_json` call per leak (`policy/gemini_verifier.py`; model chosen by
`pick_text_model`). It writes the operator leak report and the approval request (what, where, why,
risk_if_delayed) from the localization numbers only. The reply is rejected, and the template is used, if a field
is empty or the valve id is missing. There is no key, `FLEET_GEMINI=0`, an error, or more than 6 calls per
minute, the template is used too. `used_gemini` is set on the JobResult and in the Finding evidence. No decision
depends on Gemini.

A live check ran once, on `gemini-3.1-pro-preview` in 17 s, for north_rack_east_wind seed 0. The report kept
the estimate (-12.0, 22.9), the 0.32 m radius, the 0.102 LEL peak, and valve_03 3.5 m upstream. After that run
the prompt was tightened to say that gas values are fractions of LEL and that the limits are the robot's own.

## Reused from `gas-leak-detection` (teammate branch, not modified)

- Plume constants from `flange_robot/env.py`: background 0.02 of LEL, noise 0.004, full-strength 0.4,
  decay exp(-d / 1.5 m). This branch adds wind stretch and log-normal turbulence (`fleet/gas/plume.py`).
- The safety supervisor and its limits, ported nearly verbatim from `flange_robot/safety.py` and
  `flange_robot/config.py`: stop at 0.20 of LEL, pause for a person within 3 m, confidence gate, permit gate.
  `fleet/gas/safety.py` adds the 0.10 approach limit.
- The alert pattern from `flange_robot/alerts.py`: structured data first, a deterministic template that is always
  available, and Gemini rewriting only on the slow path.
- That branch depends on MuJoCo and is not merged here, so the minimum was ported rather than imported. Its flange
  diagnosis, re-torque policy and PPO environment cover a different job and are not used.

## Mocked or not done

- The plume is a 2-D analytic model at sensor height. There is no CFD, no obstacles, no wind gusts or meander,
  and no source height. The fit assumes the same model family as the truth, with known decay length and measured
  wind, so the accuracy above is an upper bound for real use. Detection range is a few metres: a leak 11 m off
  the route is missed, and a test shows it.
- The robot walks in straight lines over the deck footprint. There is no collision or walkability check off the
  patrol route, and no Isaac Sim run.
- Valves have no facing, line topology or handle type in the rig meta. Upstream comes from a `flow_dir` vector.
  Torque, breakaway, lever, filter dP and panel state are simulated values in `World.state`.
- Filter and panel positions are not in the rig meta. The tests use positions on the north rack.
- The approval text for filter and panel jobs is template only. Gemini writes only the leak report and the
  valve approval.
