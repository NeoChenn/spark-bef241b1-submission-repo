# Aerial (A) and crawler (C) jobs

Files: `fleet/jobs/aerial.py`, `fleet/jobs/crawler.py`, helpers in `fleet/aerial/` (`flight.py`,
`thermal.py`, `patches.py`, `gemini.py`, `build_testdata.py`), test images and recorded Gemini replies in
`fleet/testdata/aerial/`, tests in `tests/test_jobs_aerial.py` (26 tests).

| kind | robot | needs | level | result |
|---|---|---|---|---|
| `flare_tip_inspection` | A | thermal | AUTO | hot/cold burner nozzle -> Finding + HUMAN `flare_tip_maintenance` |
| `corrosion_survey` | A | camera | AUTO | ISO 4628-3 Ri per patch; Ri >= 3 -> `rust_repair` (Q) or `coating_touchup` (C); with `params["verify_of"]` it re-images after a repair |
| `coating_touchup` | C | spray | SUPV | surface prep + spray path, coverage; marks the patch repaired, hands a verify job to A |
| `ut_thickness` | C | ut_probe | AUTO | UT grid on a vessel shell; below nominal - CA -> critical + HUMAN `ffs_assessment` + dense re-grid |

All four call `fleet.contract.register(...)` at import. The coordinator only needs `import fleet.jobs.aerial,
fleet.jobs.crawler`.

## Coordination story (tested end to end through the executors, no coordinator)

`test_chain_drone_gemini_crawler_quadruped_reimage`:

1. Drone `corrosion_survey` flies the standoff perimeter to 5 patches (plan: 673 m, 223 s, 10 % battery,
   2 no-fly detours), grades each with the classical estimator and Gemini, fuses the two.
2. Ri >= 3 spawns SUPV follow-ups: `ri4_1` (pipe 1.1 m above the middle deck, inside the Go2 + lift + UR3 band
   0.2-1.8 m) -> `rust_repair` for Q; `ri5_0` (vessel shell) and `ri3_2` (structure) -> `coating_touchup` for C.
   Ri 2 and Ri 0 patches spawn nothing.
3. The operator approves (`params["approved"] = True`); C preps and sprays, Q's repair is a stub in the test
   (Q's `rust_repair` executor lives on another branch) that calls `fleet.jobs.crawler.mark_repaired`.
4. Each fix returns a `corrosion_survey` follow-up for A with `verify_of`; the drone re-images, Gemini and the
   classical grader both read Ri 0, findings are `repair_verified`. If the patch still grades Ri >= 2 the
   finding is `repair_not_verified` and the repair is re-queued (`test_reimage_flags_a_repair_that_did_not_hold`).

Status convention: a fix that worked returns `DONE` with an info Finding whose `follow_up` holds the verify job
(the coordinator should queue `follow_up` from every result, not only `FINDING`).

## Measured numbers

Measured = produced by code in this branch on the test data (reproduce with `pytest -s tests/test_jobs_aerial.py`).

**Ri graders vs known grade** (18 patches, 3 per grade at the ISO reference areas 0 / 0.05 / 0.5 / 1 / 8 / 40 %,
base coats `blue_metal_plate`, grey, white, green, safety yellow; rust from `rust_coarse_01` / `Rust004`, spot
shapes from `rusty_metal_02`):

| grader | exact Ri | within 1 grade |
|---|---|---|
| classical (HSV rust-area) | 18/18 = 1.00 | 1.00 |
| classical, 36 extra patches at random % inside each class | 34/36 = 0.94 | 1.00 |
| Gemini (`gemini-robotics-er-2-preview`, live calls, recorded) | 16/18 = 0.89 | 1.00 |
| fused (agree -> that grade; disagree -> higher) | 17/18 = 0.94 | 1.00 |

Gemini misses: `ri0_0` (clean blue plate read as Ri 1, "a few tiny pinpoint rust spots") and `ri3_1` (1 % read
as 0.4 %, Ri 2). The fusion takes the higher grade, so it recovers `ri3_1` and inherits the harmless `ri0_0`.
All 9 repaired patches: Gemini Ri 0 (9/9), classical Ri 0 (9/9). Gemini latency 2.3-6.9 s per image.
On the 15 rusted patches Gemini's rust % was within a factor of 2.5 of the truth (e.g. 38 vs 40, 8.5 vs 8,
0.03 vs 0.05); it read 0 % on 2 of the 3 clean ones.

Caveat: the classical grader's colour rule and the synthetic patches share the same idea of "rust colour", so
its 100 % is an upper bound; Gemini had no such tuning (zero-shot prompt with the ISO table).

**Thermal nozzle detector**: 20/20 degraded nozzles found (hot +260 C or cold -380 C, 10 seeds x 24 m / 32 m
range, random pointing error +-4 px), 0 false alarms on 20 healthy frames (nozzles 560 +- 8 C).
Gemini second opinion on the false-colour frames (recorded): 3/3 (healthy -> none, nozzle 5 hot, nozzle 3 cold).

**Flight planner** (helideck start (0, 0, 81.3), stacked rings every 12 m from z 24 at 8 m standoff):

| wind | no-fly cylinder (centre, r) | survey length / time / battery | min no-fly clearance |
|---|---|---|---|
| calm | (-58, 8), 17.3 m | 2310 m / 422 s / 19 % | 2.9 m |
| 6 m/s toward 200 deg | (-65, 5.4), 24.8 m | 2366 m / 431 s / 21 % | 2.9 m |
| 10 m/s toward +x (plume onto the rig) | (-45.5, 8), 29.8 m | 2365 m / 472 s / 27 % | 3.0 m |
| 13 m/s | - | rejected: wind above 12 m/s limit | - |

Rig-box clearance is 8 m (the standoff) on every plan. Tests check the sampled path (0.5 m) never enters the
no-fly volume or the rig box for 5 wind cases, that a straight path through the flame is caught, and that wind,
gust and battery limits reject plans.

**Coating touch-up** (`ri5_0`, 0.8 m patch, Ri 5): prep 0.544 m2 over 0.245 m2 rust, 17 passes, 16.1 m path,
coverage 100 %, two coats (50 % overlap) 100 %, 0.46 L paint, 90 s spray + 33 min prep (nominal rates).

## Not measured / modelled / mocked

* **No Isaac Sim.** Drone camera images are 2D composites of repo textures, not renders of `full_rig.usda`.
  Thermal frames are a parametric model (sky, boom, tip, nozzle spots, flame, NETD noise), not radiometry.
* **Flare geometry** comes from `sim/cinematic.py` (`build_flare` base/tip literals and flame blob table, read
  with `ast`; it raises if they disappear). The no-fly radius = flame reach 5.3 m + 12 m heat margin, + half the
  plume drift (20 m plume / 8 m/s rise x wind); these margins are engineering guesses, not API 521 radiation calcs.
* **Rig volume** is the `bounds_m` box from `full_rig_meta.json` (z clipped to sea level). The real outline is not
  a box, so perimeter flight is conservative; patches are imaged from outside that box.
* **Drone performance** (Matrice 30T-class: 262 Wh, 380 W hover, 12 m/s wind limit, 25 % reserve) is nominal.
* **Patch positions** in `demo_world()` are illustrative, placed on deck heights from the rig meta.
* **Q's `rust_repair`** is a stub in the tests; dispatch only decides Q vs C by `on_pipe` and the 0.2-1.8 m band.
* **Crawler**: prep/spray rates, DFT 250 um, VS 0.7, TE 0.6 are nominal; the crawler reuses the drone's patch
  image as its close-up. UT readings are truth + 0.05 mm noise with 3 % couplant drop-outs (re-taken);
  `t_min = nominal - corrosion allowance` stands in for a pressure-design minimum thickness.
* **Gemini** runs from recorded replies by default (`fleet/testdata/aerial/gemini_replies.json`, keyed by image
  hash) so tests and demos are offline and repeatable; images it has not seen fall back to the classical grade.
  `FLEET_GEMINI_LIVE=1` makes the executors call Gemini live. 32 live calls were made to record: 18 patches,
  9 repaired patches, 3 thermal frames, 2 early probes. Regenerate with
  `python -m fleet.aerial.build_testdata --gemini` (needs `GEMINI_API_KEY`).
* The thermal Gemini opinion inside the executor only fires live (the executor's frame depends on the planned
  viewpoint, so its hash is not in the recording).
* `JobResult.to_json()` (contract) leaves `needs`/`robot_kinds` of follow-up jobs as Python sets; dump with a
  `default=` handler (the contract test of this branch does).

## World.state used

```python
world.state["wind"] = {"speed_mps": 6.0, "toward_deg": 200.0, "gust_mps": 9.0}   # toward = where air goes
world.state["flare"] = {"nozzle_temps_c": [560, 548, ...], "nozzle_pitch_m": 0.45}
world.state["topsides_patches"] = {"ri4_1": {"image": "ri4_1.jpg", "rust_pct": 8.0, "base": "grey",
    "fill": "Rust004", "seed": 1013, "position": [-12, 26, 32.35], "on_pipe": True, "size_m": [0.6, 0.6]}}
world.state["vessels"] = {"vessel_11_52": {"nominal_mm": 12.7, "ca_mm": 3.0, "radius_m": 1.2,
    "height_m": 4.0, "uniform_loss_mm": 0.6, "pits": [{"theta_deg": 130, "z_m": 1.4, "depth_mm": 3.5, "sigma_m": 0.12}]}}
```

Outputs written back: `topsides_patches[id]["repaired" | "verified" | "history"]`, `corrosion_grades`.
`fleet.jobs.aerial.demo_world()` builds a ready world (hot nozzle 5, five patches, 6 m/s wind).
