# Lidar object detector: results

Rule-based detector for objects lying in the patrol path (`patrol/detect.py`, numpy only). Scored against
ground truth from the 2D patrol sim with `python -m patrol.eval_detect` (about 2 min on a laptop CPU).

## Method

1. Convert each lidar return to world coordinates using the robot pose.
2. Project each return onto the path centre-line. Keep it if all of these hold:
   - lateral offset under `tube`
   - 0.2 to 3.0 m ahead of the robot along the path
   - it projects onto the path itself, not past the end of the path or the search window. This uses the
     tangential residual check (`PathRef.along_residual`), which fixes points beyond the window being
     given a small, wrong lateral offset.
3. The pipes lie outside the tube, so whatever is left is an object on the walkway.
4. Cluster the remaining returns in beam order: a gap over `cluster_gap` starts a new cluster, and clusters
   wrap around ±180°.
5. Each cluster becomes one detection:
   - centre = mean of its points, pushed back along the ray by half the width
   - width = chord between its end points + `width_pad_beams` × range × beam spacing
6. **Pickable** when width ≤ `pick_max_width`, which is 0.11 m for the gripper and container.
7. `Tracker` confirms a detection after `confirm` consecutive frames, matching within 0.25 m in the world frame.

## Evaluation setup

| | |
|---|---|
| Worlds | 48 envs, seed 777 (held out). Random curved paths, pipe walls 0.42 to 1.1 m from the centre-line. Objects have r 3 to 25 cm, up to 3 per world, each present with probability 0.8. |
| Robot | Driven by the pure-pursuit baseline. A frame is sampled every 3rd step (0.3 s), giving 3,840 frames and 1,898 ground-truth object instances. |
| Lidar | 360° scan with per-env Gaussian noise σ of 0 to 3 cm and 2% dropouts. |
| Ground truth | The object intrudes into the lane (\|lateral of centre\| − r < 0.32 m), its centre is 0.2 to 3.0 m ahead, and at least one noiseless beam hits it. |
| Matching | A detection counts as a true positive when it is within 0.25 m of a ground-truth centre. A detection on an object outside the ground-truth set (just outside the lane, or occluded) is ignored. Anything else, such as walls or noise, is a false positive. |

## Sweep (360 beams, 72 configs)

Pick rule: the best F1 with precision ≥ 0.95.

| tube (m) | cluster_gap (m) | min_points | confirm | P | R | F1 |
|---|---|---|---|---|---|---|
| **0.36** | **0.12** | **1** | **1** | **0.998** | **0.988** | **0.993** |
| 0.36 | 0.18 | 1 | 1 | 0.999 | 0.986 | 0.993 |
| 0.36 | 0.08 | 1 | 1 | 0.991 | 0.992 | 0.992 |
| 0.32 | 0.12 | 1 | 1 | 0.999 | 0.978 | 0.989 |
| 0.36 | 0.18 | 2 | 1 | 1.000 | 0.971 | 0.985 |
| 0.25 | 0.12 | 1 | 1 | 0.999 | 0.901 | 0.947 |
| 0.25 | 0.12 | 1 | 2 | 0.999 | 0.867 | 0.929 |
| 0.25 | 0.12 | 1 | 3 | 0.999 | 0.822 | 0.902 |

- A narrow tube misses objects that only poke into the lane.
- `confirm` > 1 costs recall here because frames are 0.3 s apart and objects enter at the 3 m edge.
- `min_points` 2 drops the smallest objects.
- Full table: `runs/patrol/detect_eval.json`.

**Chosen config (now the `DetectCfg` defaults):** `tube=0.36, cluster_gap=0.12, min_points=1, confirm=1,
width_pad_beams=1.0, pick_max_width=0.11`.

| Metric (chosen config, 360 beams) | Value |
|---|---|
| Precision / recall / F1 | 0.998 / 0.988 / 0.993 |
| False positives per frame | 0.0008 (3 in 3,840 frames) |
| Recall by distance: 0–1 / 1–2 / 2–3 m | 0.992 / 0.991 / 0.978 |
| Recall by size (diameter): < 11 cm / 11–25 cm / > 25 cm | 0.988 / 0.981 / 0.993 |
| Width error, mean (bias) / mean absolute | −4.6 cm / 5.3 cm |
| Pickable decision accuracy, among true positives | 0.962 |

Width padding sweep: padding of 1 to 3 beam spacings moves the bias from −4.6 to −0.4 cm. However,
pickable accuracy drops from 0.962 to 0.925 because more objects near the 11 cm boundary flip, so the
padding stays at 1.0.

## Beam count (recommendation for the robot)

| Beams (360°) | Resolution | P | R | F1 | Recall at 2–3 m | Pickable accuracy |
|---|---|---|---|---|---|---|
| 360 | 1° | 0.998 | 0.988 | 0.993 | 0.978 | 0.962 |
| **180** | **2°** | **0.999** | **0.992** | **0.996** | **0.986** | 0.943 |
| 72 | 5° | 0.999 | 0.970 | 0.984 | 0.908 | 0.954 |

**Recommended sensor: a 360° scan with ≥ 180 beams (2°), at least 4 m range, at 10 Hz.**
- 72 beams loses about 8% of objects at 2 to 3 m, because small objects fall between beams.
- 360 beams adds nothing over 180.

The navigation policy pools any scan down to 36 sectors, so the same sensor feeds both.

**The scan plane has to be low.** A 2D scan only sees objects taller than its mounting height, so either:
- put the 2D lidar about 5 to 10 cm above the deck, or
- use a 3D lidar (the Go2's L1 class) and keep the returns in a 3 to 30 cm height band above the deck,
  flattened to 2D.

## Limitations

- **Clean sim walls.** Precision is near-perfect partly because the sim's walls are clean cylinders along
  the path. The real rig has supports, valves and flanges that can stick into the lane, and they will read
  as objects. Two mitigations:
  - lower `tube`, which costs recall (see the table)
  - map the static scene on the first pass and subtract it
- **Tube width.** `tube=0.36` leaves only 6 cm of margin to the narrowest sim corridor (0.42 m half-width).
  In narrower corridors, use `tube = min(0.36, half_width − 0.06)`.
- **Confirm frames.** `confirm=1` wins in this evaluation because there are no speckle returns and frames
  are 0.3 s apart. On a real sensor at 10 Hz, use `confirm=2`: it adds 0.1 s of latency and removes
  single-frame speckle.
- **Width bias.** Width is underestimated by about 4.6 cm, because the visible arc is shorter than the
  diameter and edge beams graze the object. So objects up to about 15 cm can be called pickable. The gripper
  or container check should confirm before lifting, or `pick_max_width` should be lowered.
- **Bins.** The start and end bins sit on the path, so they will be detected near the path ends. The mission
  layer should suppress detections within about 0.5 m of known bin positions.
- **Round objects only.** Objects are modelled as circles. Long thin items such as pipes or tools lying
  across the path would read as wide and get classed not-pickable.
- **No intensity.** The sim lidar has no intensity channel, so the "lidar strength" idea is implemented as
  range geometry. If the sensor reports intensity, an extra threshold, for example for reflective metal
  debris, can be added in `detect_frame`.
