# Live repair run (`policy/repair_live.py`)

Isaac Lab repair loop on the rig deck: orchestrator + PPO scrub policy (rsl_rl inference) + rust detector on the
inspection camera. Writes `repair_live.mp4`, `repair_live_close.mp4`, `decisions.jsonl`, `telemetry.jsonl`,
`run_info.json` and `summary.json` into `--out_dir`.

## Detector (`--detector DIR`, default `runs/detector/best`)

`DIR` must hold `model.onnx` + `meta.json` (the mnv4 LoRA detector, real test F1 0.977, see `detection/RESULTS.md`).
It is loaded with `RustLocalizer.from_checkpoint(DIR, patch=96, stride=48)`; threshold and temperature come from
`meta.json`. onnxruntime is taken from Isaac's bundle (`exts/isaacsim.pip.onnx/pip_prebundle`) if it is not already
importable. If a file is missing or the model fails to load, the run falls back to the colour stand-in and says why.

## Scrub policy (`--ckpt`, `--storm L`, `--storm-ckpt`)

`--ckpt` is the calm v2 checkpoint. With `--storm L` (0-1) the env runs the storm disturbances at level L
(gusts on the arm, deck sway, wet pipe; `policy/storm.py`), and `--storm-ckpt` (storm v3, a drop-in with the same
obs/action layout) replaces `--ckpt`. If `--storm-ckpt` is missing, v2 runs and the summary says so.

## Labelling

The start-up log prints `SCRUB POLICY ...` and `DETECTOR ...`. `run_info.json` (written before the loop) and
`summary.json` carry `detector`, `detector_info` (kind `lora_onnx` or `colour_standin`, dir, threshold,
temperature, `fallback_reason`), `policy`, `policy_info` (kind `scrub_v2` or `storm_v3`, ckpt) and `storm`.
Captions and video titles should quote these fields rather than assume the trained models ran.

Limitation: `LabRepairBridge` has no `weather()` hook, so the orchestrator's storm reactions (lens wipe, wait for
lull, gust gate) do not fire here; the storm acts only through the env physics.

## Getting the checkpoints (Modal profile `jacobyuan99`)

```
MODAL_PROFILE=jacobyuan99 modal volume get rust-ckpt detector/best runs/detector/
MODAL_PROFILE=jacobyuan99 modal volume ls rust-ckpt scrub/          # find the v2 and storm v3 run folders
MODAL_PROFILE=jacobyuan99 modal volume get rust-ckpt scrub/<v3_run>/model_<it>.pt runs/scrub_v3/
```

## Run

```
C:\isaacsim\python.bat policy\repair_live.py --worker --ckpt runs\scrub_v2\model.pt --out_dir runs\repair_live ^
    --detector runs\detector\best --storm 1.0 --storm-ckpt runs\scrub_v3\model_<it>.pt --no-gemini
```

CPU test of the wiring (no Isaac, no real model): `tests/test_repair_live_detector.py`.
