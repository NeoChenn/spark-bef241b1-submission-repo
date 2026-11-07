# Rust classifier: results

**Deployed model: `mobilenetv4_conv_small` + LoRA, trained on `rust_v2`** (real photos plus synthetic sim-domain crops).
It is saved at `rust-ckpt:/detector/best/{model.onnx, model.pt, meta.json, metrics.json}` (Modal profile `jacobyuan99`) and locally at `runs/detector/best/`.

| | value |
|---|---|
| Input | RGB, resized straight to 224x224 (no crop), `/255`, ImageNet mean/std, NCHW float32 |
| Output | 1 logit. `p_rust = sigmoid(logit / T)` with **T = 2.607** |
| Decision | rust if `p_rust >= 0.476` (logit >= -0.249). If `0.35 <= p_rust <= 0.65`, take a second view |
| Real test (881 photos) | acc 0.981, P 0.970, **R 0.983**, F1 0.977, AUROC 0.998, PR-AUC 0.997 |
| Synthetic sim-domain test (485) | F1 1.000. The rust_v1 model scores 0.795 here |
| Held-out sim renders (13 labelled crops) | **13/13** correct. The rust_v1 model gets 10/13 |
| Size and latency | 2.49 M params, 0.19 GMACs. ONNX on CPU, batch 1: **2.6 ms on 1 thread, 1.1 ms on 4 threads**. A10 GPU, torch, batch 1: 3.4 ms |

Inference helper: `detection.export.OnnxRustClassifier("runs/detector/best").predict(pil_image)`.
It returns `{p_rust, rust, unsure, logit}`.

## Setup
- **Backbones**: timm, ImageNet-pretrained (DINOv2 for the ViT), single-logit head.
- **LoRA**: peft, r=16, alpha=32, dropout 0.1. The head is fully trainable (`modules_to_save`).
  - ViT: LoRA on attention `qkv` and `proj`.
  - MobileNetV4: LoRA on all 28 pointwise 1x1 convs.
  - ResNet18: has only 3 1x1 convs, so LoRA goes on all 19 non-stem convs.
  - CNNs also train their BatchNorm affine params.
  - Before export, LoRA is merged into the weights (`merge_and_unload`), so the ONNX graph is a plain timm model.
- **Loss**: BCEWithLogits with `pos_weight = n_clean / n_rust` (1.74 on v1).
- **Augmentation**: RandomResizedCrop(224, scale 0.6-1), horizontal and vertical flips, rotation ±15°, ColorJitter (b/c 0.25, s 0.15, **hue 0.02**).
- **Training**: AdamW (LoRA lr 1e-3, full fine-tune 3e-4), 1 epoch warmup then cosine, bf16 AMP on GPU, 12 epochs. The best epoch is kept by val PR-AUC, with early stopping (patience 4).
- **Threshold** is picked on val for recall ≥ 0.95 at the highest threshold that reaches it. On rust_v2 it is the lowest of the real-only and synthetic-only val thresholds, so recall ≥ 0.95 holds in each domain. Without that, the easy synthetic data would raise the threshold.
- **Calibration**: temperature scaling fitted on val.
- **Data**: `rust_v1` has 5894 real images (train 1537 rust / 2667 clean, val 304/505, test 362/519). `rust_v2` = `rust_v1` + 3400 synthetic images (see below).

## rust_v1 bake-off (Modal A10G)
Metrics are at 0.5 and at the val-picked threshold ("thr").

| run | trainable / total | split | acc@0.5 | P@0.5 | R@0.5 | F1@0.5 | AUROC | PR-AUC | P@thr | R@thr | F1@thr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mnv4_conv_small LoRA | 0.26M / 2.73M | train | 0.9955 | 0.9909 | 0.9967 | 0.9938 | 0.9999 | 0.9998 | 0.9954 | 0.9935 | 0.9945 |
| | | val | 0.9666 | 0.9511 | 0.9605 | 0.9558 | 0.9970 | 0.9953 | 0.9731 | 0.9507 | 0.9617 |
| | | test | 0.9807 | 0.9726 | 0.9807 | 0.9766 | 0.9976 | 0.9966 | 0.9751 | 0.9724 | 0.9737 |
| mnv4_conv_small full FT | 2.49M / 2.49M | train | 0.9933 | 0.9954 | 0.9863 | 0.9908 | 0.9998 | 0.9997 | 0.9974 | 0.9837 | 0.9905 |
| | | val | 0.9703 | 0.9636 | 0.9572 | 0.9604 | 0.9949 | 0.9929 | 0.9698 | 0.9507 | 0.9601 |
| | | test | 0.9716 | 0.9668 | 0.9641 | 0.9654 | 0.9948 | 0.9947 | 0.9831 | 0.9613 | 0.9721 |
| resnet18 LoRA | 0.58M / 11.75M | train | 0.9964 | 0.9929 | 0.9974 | 0.9951 | 0.9999 | 0.9998 | 0.9954 | 0.9954 | 0.9954 |
| | | val | 0.9691 | 0.9544 | 0.9638 | 0.9591 | 0.9945 | 0.9934 | 0.9601 | 0.9507 | 0.9554 |
| | | test | 0.9728 | 0.9543 | 0.9807 | 0.9673 | 0.9945 | 0.9928 | 0.9619 | 0.9751 | 0.9684 |
| vit_small_patch14_dinov2 LoRA | 0.44M / 22.07M | train | 0.9962 | 0.9916 | 0.9980 | 0.9948 | 0.9999 | 0.9998 | 0.9980 | 0.9876 | 0.9928 |
| | | val | 0.9839 | 0.9739 | 0.9836 | 0.9787 | 0.9987 | 0.9979 | 0.9897 | 0.9507 | 0.9698 |
| | | test | 0.9830 | 0.9728 | 0.9862 | 0.9794 | 0.9986 | 0.9980 | 0.9805 | 0.9724 | 0.9764 |

Calibration and cost. "cm" is the test confusion matrix at thr. "Unsure" is the share of test samples in the unsure band.

| run | thr (cal. p) | T | test cm tn/fp/fn/tp | test ECE raw → cal | val ECE raw → cal | unsure % | params | GMACs | ORT CPU b1, 1 thr / 4 thr | GPU b1 |
|---|---|---|---|---|---|---|---|---|---|---|
| mnv4 LoRA | 0.623 | 2.12 | 510/9/10/352 | 0.017 → 0.019 | 0.026 → 0.018 | 1.1 | 2.49M | 0.19 | 2.8 / 0.9 ms | 3.5 ms |
| mnv4 full FT | 0.586 | 1.99 | 513/6/14/348 | 0.021 → 0.012 | 0.021 → 0.014 | 2.2 | 2.49M | 0.19 | 4.3 / 1.8 ms | 4.5 ms |
| resnet18 LoRA | 0.583 | 1.61 | 505/14/9/353 | 0.017 → 0.021 | 0.024 → 0.020 | 1.8 | 11.2M | 1.82 | 40.7 / 10.3 ms | 2.3 ms |
| DINOv2-S LoRA | 0.903 | 1.50 | 512/7/10/352 | 0.015 → 0.013 | 0.013 → 0.014 | 0.5 | 21.6M | 5.53 | 144.6 / 67.9 ms | 5.4 ms |

GMACs are counted with fvcore. CPU latency is the Modal container CPU. On the dev laptop (Core Ultra 7 255H) mnv4 runs at 3.2 ms (1 thread) and 1.1 ms (4 threads).

- **LoRA vs full fine-tune rule** (use full FT if LoRA val F1 < 0.90 or more than 2 points below full FT): **LoRA is kept.**
  - v1: val F1@0.5 is 0.9558 for LoRA vs 0.9604 for full FT (0.5 points apart).
  - v2: 0.9654 vs 0.9652.
  - A local CPU rerun of v1 gives the same picture: LoRA 0.9568 vs full FT 0.9600, test F1@thr 0.9725 for both.
- **DINOv2-S** is the accuracy ceiling at about +1.5 F1 points on test. It costs 30x the FLOPs and 50x the CPU latency, so MobileNetV4 is the edge pick.

## Sim-domain gap and rust_v2
The scene's rust comes from photo-scanned Poly Haven textures. The v1 models handle real photos well but do worse on the sim look:
- the rendered deck grate was missed;
- the gauge face was called rust (p 0.998);
- on the synthetic sim-domain test set, the best v1 model reaches only F1 0.795.

**rust_v2** (`detection/synth_sim.py`, `runs/data/rust_v2`, volume `rust-data:/rust_v2`) adds 1500 rust and 1900 clean synthetic images (224 px).
- **Rust**: crops of `rust_coarse_01`, `rusty_metal_02` and `metal_grate_rusty`, plus 400 rust blobs (soft irregular masks, like `damage_patch`) composited onto painted pipe.
- **Clean**: `metal_plate`, `blue_metal_plate`, the red `box_profile_metal_sheet`, concrete, 8 plain pipe paints, `green_metal_rust` crops with < 3% rust-hue pixels, and crops of the clean `full_rig_*` renders.
  - `green_metal_rust` crops with 3-15% rust-hue pixels were dropped. No crop exceeded 15%.
  - `rusty_painted_metal` was left out as ambiguous.
- **Rendering**: each crop gets cylindrical pipe shading (random light direction, specular streak), an optional clean background, then exposure, gamma, white balance, blur, noise and JPEG.
- **Leak-free split**: each 1024 px texture is cut into a 4x4 grid of regions, and each region (`group_id`) goes to one split (12 train / 2 val / 2 test). Paint groups and render images are split the same way.
- **Held out of training**: the `preview_*` renders and `full_rig_gauge_closeup.png`.

### rust_v1 vs rust_v2 (mnv4 LoRA, A10G)
| | v1 LoRA | v2 LoRA | v2 full FT |
|---|---|---|---|
| real val F1@thr / R@thr | 0.962 / 0.951 | 0.951 / 0.951 | 0.954 / 0.951 |
| **real test**: acc / P / R / F1 @thr | 0.981 / 0.975 / 0.972 / 0.974 | **0.981 / 0.970 / 0.983 / 0.977** | 0.977 / 0.972 / 0.972 / 0.972 |
| real test AUROC / PR-AUC | 0.998 / 0.997 | 0.998 / 0.997 | 0.996 / 0.995 |
| **synthetic test** F1 @thr (n=485) | 0.795 | **1.000** | 0.993 |
| **held-out sim renders**, labelled correct | 10/13 | **13/13** | 12/13 |
| test ECE raw → cal (all of test) | 0.017 → 0.019 | 0.011 → 0.012 | 0.015 → 0.012 |

Other v1 models on the synthetic test (F1@thr): mnv4 full FT 0.714, resnet18 LoRA 0.590, DINOv2-S LoRA 0.419. On the held-out renders: 11/13, 12/13 and 11/13.

### Held-out sim renders (hand-picked crops from `sim/output`, `python -m detection.eval <model_dir>`)
| render | crop | label | v1 LoRA p_rust | v2 LoRA p_rust |
|---|---|---|---|---|
| preview_repair_closeup | damage patch (rust_coarse_01) | rust | 0.980 RUST | 0.991 RUST |
| preview_repair_closeup | rusty steel post (rusty_metal_02) | rust | 0.987 RUST | 1.000 RUST |
| preview_repair_closeup | deck grate (metal_grate_rusty) | rust | 0.458 clean (unsure) | 0.731 RUST |
| preview_repair_closeup | green painted pipe | clean | 0.139 clean | 0.150 clean |
| preview_repair_closeup | green painted pipe right | clean | 0.013 clean | 0.007 clean |
| preview_repair_closeup | rust streak/crack on paint | ambiguous | 0.948 RUST | 0.026 clean |
| preview_hero | rusty steel support | rust | 0.995 RUST | 1.000 RUST |
| preview_hero | rusty steel support 2 | rust | 0.998 RUST | 1.000 RUST |
| preview_hero | deck grate | rust | 0.169 clean | 0.799 RUST |
| preview_hero | green painted pipe | clean | 0.213 clean | 0.294 clean |
| preview_hero | frame beam (rusty_painted_metal) | ambiguous | 0.553 clean (unsure) | 0.034 clean |
| preview_hero | brown pipe (rusty_painted_metal) | ambiguous | 0.091 clean | 0.142 clean |
| preview_hero | grey pipe on deck | ambiguous | 0.123 clean | 0.994 RUST |
| full_rig_gauge_closeup | yellow pipe | clean | 0.156 clean | 0.000 clean |
| full_rig_gauge_closeup | blue pipe | clean | 0.121 clean | 0.000 clean |
| full_rig_gauge_closeup | gauge face | clean | 0.998 RUST | 0.380 clean (unsure) |
| full_rig_gauge_closeup | blue pipe right | clean | 0.033 clean | 0.005 clean |
| preview_deck_end | whole frame | ambiguous | 0.026 clean | 0.066 clean |
| preview_overview | whole frame | ambiguous | 0.200 clean | 0.252 clean |

### False positives by source (real test, at the deployed threshold)
| source | class | n | v1 LoRA flagged | v2 LoRA flagged |
|---|---|---|---|---|
| rf100 | clean | 451 | 0.4% | 0.4% |
| binkhoa | clean | 68 | 10.3% | 13.2% |
| rf100 | rust | 103 | 96.1% | 97.1% |
| binkhoa | rust | 63 | 92.1% | 96.8% |
| iamcloud | rust (rust-only source) | 196 | 99.5% | 99.5% |

## Caveats
- **Source shortcut risk**: `iamcloud` contains only rust images, so its ~100% recall may come partly from recognising the source.
- **No real painted-steel negatives**: none of the real data is clean painted steel. Painted-metal false positives are tested only on synthetic and render crops. `binkhoa` clean images (weathered or industrial surfaces) give the highest real false-positive rate, around 10-13%.
- **The synthetic test is easy**: it shares the generator and texture families with training, held apart only by texture region. Its 100% says the gap on these textures is closed, not that the model generalises to other rust textures.
- **The held-out render set is tiny**: 13 labelled crops. Treat it as a sanity check, not a benchmark.
- **The deck reads as rust**: `metal_grate_rusty` crops are scored as rust, which is semantically right but means wide views that include deck will be flagged. Point the camera at the pipe surface, or crop to the pipe mask, before classifying.
- **`rusty_painted_metal` (brown pipe, frame beams) is unlabelled**: the model calls it clean. Decide with the sim team whether that material counts as "rust to repair".

## Reproduce
```bash
# CPU smoke tests (~40 s)
.venv/bin/python -m pytest tests/test_detection.py -q
# build rust_v2 from a local copy of rust_v1, then upload it
.venv/bin/modal volume get rust-data rust_v1 runs/data
.venv/bin/python -m detection.synth_sim --v1 runs/data/rust_v1 --out runs/data/rust_v2
MODAL_PROFILE=jacobyuan99 .venv/bin/modal volume put -f rust-data runs/data/rust_v2 /rust_v2
# GPU training. --detach so a dropped local connection does not kill the run. A10G because L4 capacity was unavailable.
DETECTOR_GPU=A10G MODAL_PROFILE=jacobyuan99 .venv/bin/modal run --detach detection/modal_app.py --all --no-promote          # v1 bake-off, 4 runs
DETECTOR_GPU=A10G DETECTOR_DATA=rust_v2 MODAL_PROFILE=jacobyuan99 .venv/bin/modal run --detach detection/modal_app.py --model mobilenetv4_conv_small
# local CPU fallback (keep threads at 8 or fewer on the laptop: 16 threads is ~6x slower)
.venv/bin/python -u -m detection.train --data-dir runs/data/rust_v2 --out-dir runs/detector/mnv4_lora_v2 --threads 8 --workers 5
# recompute the threshold from saved logits, run the sim check, promote
.venv/bin/python -m detection.train --rescore runs/gpu/mobilenetv4_conv_small_lora_v2 --data-dir runs/data/rust_v2
.venv/bin/python -m detection.eval runs/gpu/mobilenetv4_conv_small_lora_v2
MODAL_PROFILE=jacobyuan99 .venv/bin/modal run detection/modal_app.py --promote-only --force mobilenetv4_conv_small_lora_v2
```
The deployed v2 run was trained before the per-domain threshold existed in `train.py`. It was rescored from its saved logits with `--rescore`, which is the same code path that `train.py` now runs.
