# Debris and person detectors: results

Two more camera skills for the rig robot. Both are trained **the same way as the rust and crack models**: real
photos plus sim-domain crops, MobileNetV4 small + LoRA through the team's `detection/train.py` (unchanged, via
`detection/train_task.py`), temperature calibration, a threshold picked on val for recall at least 0.95, an unsure
band, and ONNX export. Code: `detection/rig_vision.py`. Robot side: `policy/deck/job.py`.

| | Debris (`checkpoints/debris_v2`) | Person (`checkpoints/person_v1`) |
|---|---|---|
| Job | find loose objects on the deck so the arm can pick them up | see people, so the 3 m pause comes from the robot's own camera |
| Real photos | `alexNova/MRS-Trash-Detection` (Apache-2.0): cans, bottles, packaging, batteries, cables; clean concrete photos (CC BY 4.0) | `VincentGOURBIN/ppe-detection` (CC BY 4.0): workers on industrial and building sites; background windows of the same photos |
| Sim-domain crops | litter and drawn cans dropped onto the rig's deck textures and renders; deck features (painted lines, seams, bolts, drains, shadows) and **4,588 hard-negative crops mined from clean rig renders** (each mined window saved twice with different camera effects) | people composited onto rig renders and deck crops; non-person photo crops composited the same way, so "a pasted rectangle" is not a shortcut |
| Images (train / val / test) | 11,420 / 1,740 / 614 | 3,508 / 746 / 746 |
| Test accuracy at threshold | **97.9%** (real 97.4%, sim 98.1%) | **97.7%** (real 98.7%, sim 96.3%) |
| Threshold / temperature | p 0.875 / 1.31 | p 0.256 / 1.62 |
| Training (CPU, 3 threads) | 40 min, best epoch 10 | 15 min, best epoch 10 |
| Speed | 1.3 ms per window on 4 CPU threads | 1.2 ms per window |

## Whole-frame results on a held-out deck

`python -m detection.eval_rig_vision` builds frames on `preview_deck_end.png`, a deck-level render that no training
set used, with litter and people from the held-out test splits. The deck has equipment that must not be picked up:
fire extinguishers, gas bottles, pipe supports and a valve wheel. Every debris frame gets a random exposure change, and
half get a soft shadow across the deck.

**Debris** (60 frames, 75 objects, 24 empty decks):

| Method | Objects found | False alarms | Empty decks flagged |
|---|---|---|---|
| v1 model alone | 72 / 75 | 770 | 24 / 24 |
| v2 model alone (with hard negatives) | 75 / 75 | 418 | 24 / 24 |
| v2 model + clean reference, plain brightness difference | 75 / 75 | 188 | 11 / 24 |
| **v2 model + clean reference, shadow-robust change (shipped)** | **73 / 75** | **0** | **0 / 24** |
| Change detection alone, no model (baseline) | 75 / 75 | 2 | 0 / 24 |

**Person** (40 frames, 30 people at 2-10 m, 10 empty):

| Method | People found | Empty frames flagged | 3 m pause decision right |
|---|---|---|---|
| Model + distance from box size only | 30 / 30 | 0 / 10 | 21 / 40 (19 needless pauses) |
| **Model + depth camera (shipped)** | **30 / 30** | **0 / 10** | **40 / 40** |

## What we learned

1. **A window classifier alone cannot tell litter from deck equipment it has never seen.** v1 flagged every empty
   deck: to it, an extinguisher is "an object on the deck". Mining hard negatives from the rig's own clean renders
   cut false alarms by half, but every empty deck was still flagged.
2. **The fix is a clean reference, as airport foreign-object-debris systems use.** The robot patrols fixed stops, so it
   keeps a picture of each stop from when the deck was known clean. Only windows where the deck changed are
   classified. The change check ignores shadows and light (they change brightness smoothly and keep colour; an object
   changes both). The arm aims at the middle of the changed pixels, not the window centre, and hits on the same
   object are merged.
3. **Honestly, in this mock the reference does most of the work.** Change detection without the model does about as
   well (75/75, 2 false alarms) as with it (73/75, 0 false alarms; the 2 misses are objects lying close together, and
   the robot finds the second one when it looks again after the first pickup). The model's job is to reject changes
   that are not litter: people's boots, a hose in use, tools being used, puddles. The mock's harmless changes are
   shadows and light only, which the change check already ignores. So the model's value is only partly shown until
   it is tested on Isaac frames with those changes.
4. **"Is it a person" is a model question; "how far" is a depth question.** The largest window is the frame height,
   so the box size alone cannot say "closer than about 5 m". Window edges are also too coarse to find the feet. With
   the depth camera (the Go2 has one; Isaac renders depth), the robot takes the pixels inside the detection that stand
   up off the deck and uses their distance: 40/40 pause decisions right. The mock's depth is perfect, so that is a
   best case.

## Limits (say these in the summary)

- The held-out frames are a render with pasted objects and people, not Isaac camera frames. Re-run
  `detection/eval_rig_vision.py` on frames from the Isaac stream.
- Person sprites in the mock are rectangular photo crops (no cut-out). Training used the same kind of composite for
  both classes, so the rectangle itself is not a cue, but it is not what a person on the rig looks like in Isaac.
- The real litter photos are product-style close-ups; the clean-floor photos are concrete. Synthetic deck crops
  balance this.
- The debris reference must be retaken if the deck changes on purpose (new equipment, moved hoses), or every change
  will be a candidate for the model to judge.
- The PPE dataset is a Roboflow export with 3 augmented copies per source photo; splits are made per source photo.

## Reproduce

```
python -m detection.prepare_rig_data --task debris --trash <MRS-Trash-Detection> --concrete <concrete_crack_images>/data/test.zip --out data/debris_v1
python -m detection.train_task --task debris --data data/debris_v1 --out runs/debris_detector/mnv4s_lora
python -m detection.mine_debris_negatives --v1-ckpt runs/debris_detector/mnv4s_lora --trash <MRS-Trash-Detection> --out data/debris_v2
python -m detection.train_task --task debris --data data/debris_v2 --out runs/debris_detector/mnv4s_lora_v2
python -m detection.prepare_rig_data --task person --ppe <ppe-detection> --out data/person_v1
python -m detection.train_task --task person --data data/person_v1 --out runs/person_detector/mnv4s_lora
python -m detection.eval_rig_vision
python -m policy.leak.mock --leak none --debris litter --person-at 2
```
