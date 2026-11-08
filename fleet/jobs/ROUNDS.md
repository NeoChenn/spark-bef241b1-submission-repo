# Operator rounds jobs (`fleet/jobs/rounds.py`)

These jobs cover the walk an offshore operator makes every shift, done by the quadruped (Go2 + lift + UR3). Every executor is registered with `fleet.contract.register` when the module is imported. Each one returns a `JobResult` with log lines, `Finding`s (follow-up jobs attached) and `used_gemini`. None of them launch Isaac or need a network connection.

| kind | robot | level | needs | est. | what it does |
|---|---|---|---|---|---|
| `gauge_read` | Q | AUTO | camera | 300 s | Reads the 7 rig gauges two ways: an OpenCV needle-angle reader and Gemini vision. |
| `thermal_acoustic_survey` | Q | AUTO | thermal | 360 s | Tracks temperature and vibration for each valve over rounds. Finds drift and step changes, not one-off spikes. |
| `escape_route_check` | Q | AUTO | camera | 150 s | Walks `PATROL_ROUTE` and checks it with the sim's `route_blockages`, plus dropped objects. |
| `spill_response` | Q | SUPV | arm | 240 s | Rings a small spill with absorbent socks once the operator approves. |
| `fg_optics_clean` | Q | AUTO | arm, camera | 180 s | Wipes a salt-fouled fire & gas (F&G) detector window, then queues an ASSIST bump test. |

`JobResult.finished_t - started_t` is computed per run from the walking distance at `robot.speed_mps` plus the time spent at each stop. The `est.` column is the class default for planning.

## Follow-ups spawned

| finding | severity | follow-up job | level |
|---|---|---|---|
| `gauge_out_of_range` (red band >= 12.8 bar / below 2.4 bar) | critical / warning | `operator_process_check` | HUMAN |
| `gauge_reading_disagreement` (readers differ by > 5 % FS) | warning | `gauge_read` with `second_look=True` (square-on, no glare) | AUTO |
| ... the readers still disagree on the second look | warning | `manual_gauge_read` | HUMAN |
| `gauge_second_look` / `gauge_unreadable` | info / warning | re-look, then `manual_gauge_read` | AUTO, then HUMAN |
| `temperature_drift/step`, `vibration_drift/step` | warning | `condition_closeup` | ASSIST |
| ... the same, at +10 C over baseline or >= 7.1 mm/s (ISO 10816 zone D) | critical | `maintenance_work_order` | HUMAN |
| `loose_item_on_route` (<= 0.6 m and <= 10 kg) | warning | `remove_loose_item` (needs arm) | SUPV |
| `escape_route_blocked` (large object) / `escape_route_obstructed` (structure) | critical | `clear_obstruction` | HUMAN |
| `spill_contained` | warning | `spill_waste_disposal` | HUMAN |
| `spill_beyond_robot` (radius > 1.5 m, or not enough socks) | critical | `spill_cleanup_crew` (status HANDED_TO_HUMAN) | HUMAN |
| `fg_optics_cleaned` | info | `fg_bump_test` (robot holds the test gas or flame simulator, a technician confirms at the panel) | ASSIST |
| `fg_window_still_obscured` after 3 wipes | warning | `fg_window_replace` | HUMAN |

The `World.state` keys each job reads and writes are listed in the module docstring. All of them are optional: when a key is missing, the job builds a synthetic default.

## Gauge reading

- **Dial truth.** Each gauge's true value comes from `sim/output/full_rig_meta.json` (`reading_bar`, 0 to 16 bar). `World.state["gauge_values"]` can override it to inject faults.
- **Dial geometry.** Taken from `sim/tools/make_generated_assets.py`. The scale runs clockwise from 135 deg (image frame, y down) through 270 deg of arc. The red band covers 0.8 to 1.0 of the scale.
- **Camera crop.** Real frames can be passed in `World.state["gauge_frames"]`. Without them, `fleet/perception/gauge.py:render_gauge_crop` renders a crop from the real dial texture. It draws a needle at the true angle, then adds perspective tilt (yaw up to ±35 deg, pitch up to ±25 deg), roll, Gaussian blur, a specular glare blob and sensor noise. The "hard" set uses 1.5 times the tilt, blur, glare and noise.
- **Classical reader** (`ClassicalGaugeReader`):
  1. Finds the dial edge as the inner rim of the dark bezel, using a black-hat filter. Additive glare does not affect this filter.
  2. Fits an ellipse to that rim and maps it back to a circle with an affine transform.
  3. Removes any remaining in-plane twist by locating the tick-free gap at the bottom of the scale.
  4. Takes the weighted circular mean direction of the red needle pixels, inside the tick ring and away from the red band.
  5. Sets confidence from the needle pixel count, how tightly the needle directions agree, and the dial aspect ratio.
- **Gemini reader** (`GeminiGaugeReader`): a REST `generateContent` call made with the helpers in `policy/gemini_verifier.py` (`load_api_key`, `encode_jpeg`, `post_json`). The prompt describes the dial in words and the reply is structured JSON: `{reading_bar, fraction, confidence, readable, reason}`. The model is `GEMINI_GAUGE_MODEL`, else `GEMINI_MODEL`, else `gemini-robotics-er-2-preview`. Calls are capped by `FLEET_GEMINI_BUDGET` (default 24 per reader instance). `FLEET_OFFLINE=1` turns Gemini off. If there is no key, no network, the budget is spent or the reply is bad, `read` returns `None` and the job carries on with the classical reading only.
- **Fusion.** When the two readers agree, the job uses their confidence-weighted mean. When they differ by more than 5 % of full scale, it raises a `gauge_reading_disagreement` finding and queues a square-on second look. If the second look still disagrees, a person reads the gauge.

### Measured accuracy (synthetic crops, errors in % of full scale; 1 % FS = 0.16 bar)

The run that produced these numbers is `C:\isaacsim\python.bat -m fleet.perception.gauge_eval --n 300 --gemini 20`. The raw output is in `fleet/testdata/rounds/gauge_accuracy.json`.

| reader / set | n | unreadable | MAE | median | p95 | max | within 2 % | within 5 % |
|---|---|---|---|---|---|---|---|---|
| classical, normal crops | 300 | 1 | **1.79 %** (0.29 bar) | 1.55 % | 4.32 % | 8.3 % | 64 % | 98 % |
| classical, hard crops | 300 | 11 | **4.40 %** (0.70 bar) | 2.48 % | 10.3 % | 79.6 % | 39 % | 79 % |
| Gemini (`gemini-robotics-er-2-preview`), 20-crop sample | 20 | 1 | **1.71 %** (0.27 bar) | 1.05 % | 4.76 % | 6.3 % | 79 % | 95 % |
| classical, on the same 20 crops | 20 | 1 | 2.19 % | 2.31 % | 4.23 % | 4.6 % | 47 % | 100 % |

- **What the Gemini sample is.** It has 20 crops: the 7 rig gauges at their true values, 9 normal synthetic crops and 4 hard ones. Gemini was called live once per crop; the mean latency was 8.7 s. Both readers declared the same hard crop unreadable (heavy glare), which is the correct outcome for that crop.
- **Readers disagreeing.** On the 20-crop sample, the two readers disagreed by more than 5 % on 5 crops. In 1 of the 20 crops at least one reader was off by more than 5 %, and the disagreement check flagged that crop. The other 4 flags were cases where both readers were within about 5 % but on opposite sides of the truth, so those second looks were not needed.
- **Red-band alarm.** The classical reader put the gauge on the right side of the 12.8 bar line 98.3 % of the time on normal crops and 94.8 % on hard crops.
- **Classical confidence.** On hard crops, readings with confidence >= 0.5 have a 2.87 % MAE and make up 77.5 % of readings. The lower-confidence rest is where the second look and Gemini add value.

### Not measured or not done

- **Gemini's accuracy on hard crops in general is not measured.** The live sample is only 20 crops, 4 of them hard, to stay within the agreed number of calls.
- **No real Isaac Sim gauge renders were read.** The `gauge_closeup` camera exists in the rig meta, but Isaac was not launched (the 6 GB GPU is shared). The reader accepts real frames through `World.state["gauge_frames"]`.
- **Crop placement is assumed.** The crops assume the gauge sits roughly in the centre of the frame. Finding and cropping the gauge in a wide frame is left to the detector or to Gemini pointing.
- **Recorded replies, not live calls, in tests.** The tests replay the recorded Gemini replies (`fleet/testdata/rounds/manifest.json`) through the real parser with a fake transport. They never call the API.

## Thermal and acoustic trends

- **History.** History is kept per asset in `World.state["condition_history"]`: temperature in deg C and vibration velocity in mm/s RMS, one value per round. A new measurement for the current round can be added through `condition_now`.
- **Synthetic default** (`fleet/perception/trend.py:synth_condition_history`), 30 rounds:
  - `valve_02` temperature drifts +0.6 C per round from round 18.
  - `valve_05` vibration steps up by 2.4 mm/s at round 22.
  - `valve_03` has a single 12 C spike at round 25. This must not raise an alarm.
- **Detector:**
  1. A 3-point median filter removes single spikes.
  2. The series is converted to z-scores against the first 10 rounds as baseline, and a two-sided CUSUM is run (k = 0.5, h = 6).
  3. A change is only reported if the last 3 rounds also sit at least max(3 sigma, 2 C or 0.5 mm/s) away from the baseline.
  4. The change point and its shape come from the better fit of two least-squares models: a step to a new level, or a hinge into a ramp.
- **What a finding carries.** The baseline mean and sd, the current mean, delta and delta in sigma, slope per round, change and alarm rounds, rounds to the critical limit and the ISO 10816 zone.
- **On the synthetic data** the drift and the step are found at exactly rounds 18 and 22, and the spike is ignored. This was checked on the synthetic history only; there is no real sensor data.

## Escape route

- **Route and blockage check from the sim.** `fleet/perception/route.py` takes `PATROL_ROUTE`, `TOP_DECK_Z` and `route_blockages` directly from the source of `sim/rig_details.py` using `ast`, so the sim's own code runs without importing pxr.
- **Fixed blockages** come from the rig meta's `patrol_route_blocked_samples`, from `World.state["route_blocked_samples"]`, or from running `route_blockages` on `World.state["deck_triangles"]`. Nearby samples are grouped into clusters, and each cluster is reported with its chainage (distance along the route).
- **Dropped objects** come from `World.state["dropped_objects"]`. An object on the 1.2 m corridor that is <= 0.6 m and <= 10 kg gets a SUPV `remove_loose_item` job; a larger one gets a HUMAN `clear_obstruction` job. Objects off the route are logged as info.
- **Mocked:** the robot does not actually detect objects; their positions come straight from `World.state`.

## Spill and F&G optics

- **Spill response.** The ring radius is the spill radius plus 0.25 m. Socks are 1.2 m long with 0.15 m overlap, and the robot carries 8 by default. The finding includes every sock's placement (x, y, yaw).
- **F&G optics clean.**
  - The F&G window is measured on a synthetic close-up of a salt-crusted lens (`fleet/perception/lens.py`): Hough circle on the window, then the share of bright deposit pixels.
  - Each wipe (spray plus microfibre) removes 75 % of the crust, up to 3 wipes, until the window is below 5 % obscured. The log notes the F&G inhibit being requested and released.
  - Three default detectors sit near the gauge racks because the rig meta has no F&G detectors.
- **Mocked:**
  - the arm motions (socks, wipe passes) are not simulated;
  - the wipe efficiency is a fixed assumption;
  - the lens images are synthetic.

## Running

```
C:\isaacsim\python.bat <runner>  # inserts the repo root and C:\Users\ys732\.isaac-deps(\test) into sys.path, then pytest.main(["tests/test_jobs_rounds.py"])
C:\isaacsim\python.bat -m fleet.perception.gauge_eval --n 300 --gemini 0   # classical accuracy only, no API calls
```
