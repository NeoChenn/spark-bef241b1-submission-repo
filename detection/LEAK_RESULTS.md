# Crack classifier (gas-leak job): results

**Deployed model: `mobilenetv4_conv_small` + LoRA, trained on `leak_v1`** (real crack photos plus synthetic flange crops).
It is the same model, training script, calibration, threshold rule, unsure band and ONNX export as the rust model
(`detection/RESULTS.md`). `detection/train_leak.py` runs `detection/train.py` unchanged and only renames the positive
class from rust to crack.

The checkpoint is in the repo at `detection/checkpoints/crack_v1/{model.onnx, meta.json, metrics.json}` (10 MB). Rust
keeps its checkpoint on Modal; this one is small and Modal was not reachable from where it was trained.
`LeakLocalizer.default()` loads it, and falls back to the classical line check if the file or onnxruntime is missing.

| | value |
|---|---|
| Input | 32x32 windows of the flange face, resized to 224x224, `/255`, ImageNet mean/std, NCHW float32 |
| Output | 1 logit. `p_crack = sigmoid(logit / T)` with **T = 0.992** |
| Decision | crack if `p_crack >= threshold`. If `0.35 <= p_crack <= 0.65` and nothing is above threshold, take another view |
| Windows | every 8 px, kept if at least 60% of the window is flange face (bolts, bore and rim masked out) |
| Test, all 720 windows | 0.5 cut: acc 0.999, 1 false alarm, 0 missed. Val-picked threshold: P 1.000, R 0.933 |
| Test by source (360 each) | val-picked threshold: real P 1.000 / R 0.933, synthetic P 1.000 / R 0.933 |
| Whole flange views at the Isaac camera (40 cracked, 40 clean) | **40/40 cracks found, 0 false alarms** |
| Harder sweep (glare, dust up to 0.4, 72 cracked views) | **53 confirmed** (classical: 46), rest "unsure", **0 silent misses, 0 false alarms** |
| Size and latency | 2.49 M params, 0.19 GMACs. ONNX on CPU, batch 1: **2.7 ms on 1 thread, 1.2 ms on 4 threads** |

## Data (`leak_v1`, built by `detection/prepare_leak_data.py`)

- **Real:** 1200 crack + 1200 clean photos from Hugging Face `mohammadnajeeb/concrete_crack_images` (CC BY 4.0).
- **Synthetic:** 1200 crack + 1200 clean crops of flange faces rendered by `policy/leak/render.py` through the 640x360 wrist
  camera at 1.0-1.25 m, with random view, dust, glare and loose bolts. Half the clean crops are hard negatives (bolt,
  rim, bore and stripe edges), so "edge" does not mean "crack".
- Split 70/15/15 per class: train 3360, val 720, test 720. File names start with `real_` or `synth_`, so the training
  report gives real and synthetic scores separately, as for `rust_v2`.

## Training

Same as rust: AdamW (LoRA lr 1e-3), warmup then cosine, 12 epochs max, best epoch by val PR-AUC, early stop
(patience 4), temperature scaling on val, threshold picked on val for recall at least 0.95.

Trained on CPU (4 threads) in 10.4 min. It stopped early at epoch 9 and kept epoch 5.

| epoch | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| val PR-AUC | 0.989 | 0.998 | 0.9997 | 0.9998 | 0.9999 | 1.000 |

```
python -m detection.prepare_leak_data --zip <concrete_crack_images>/data/test.zip --out data/leak_v1
python -m detection.train_leak --data data/leak_v1 --out runs/leak_detector/mnv4s_lora --epochs 12 --threads 4
python -m detection.eval_leak_sim                 # classical vs trained, whole flange views at the Isaac camera
```

## What we found on the way

- **Which windows get checked mattered more than the model.** At 1.1 m the flange face is a ring only about 30 px wide.
  With 16 px steps and windows that had to be fully face, each view gave 0-11 windows, and cracks near bolts or the
  rim were never looked at. The trained model then found 18/40 cracks, the same as a hand-made dark-line score.
  Steps of 8 px and windows at least 60% face fixed it (40/40). The model can handle partly-covered windows because
  it was trained with edge windows as negatives.
- **The threshold rule picked a very high threshold (p 0.99999).** The rule takes the highest threshold that still
  catches 95% of val cracks. Here val is perfectly separable, so that threshold sits at the top of the crack scores.
  On whole flanges it costs nothing (40/40 at both 0.5 and the picked threshold), because a real crack has several
  windows and at least one scores very high. We kept the rule unchanged, to match rust.

## Limits (say these in the summary)

- **The data is easy.** Val and test are near-perfect. The synthetic crops come from the same renderer as the
  evaluation views, so the sim numbers are optimistic.
- **Concrete is not steel.** The real photos teach "a thin dark crack line", not what a cracked steel flange looks like.
- **Not yet checked on Isaac renders.** Lighting, materials and noise there will differ. Re-run
  `detection/eval_leak_sim.py` on wrist-camera frames from the Isaac stream before trusting the numbers.
- The classical line check (`LeakLocalizer()` with no model) is kept as the fallback and as a cross-check.
